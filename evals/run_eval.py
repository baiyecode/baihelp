"""售后抽取评估跑分脚本（standalone harness，无 pytest、无第三方依赖）。

用法：
    uv run python evals/run_eval.py             # 真实模型跑分（需已配置 LLM_API_KEY）
    uv run python evals/run_eval.py --self-test # 离线自测：脚本化假模型验证判分/门禁/异常路径，
                                                # 不发任何网络请求，不需要 API key

字段级判分：
    - order_no:           与 expected 完全相等（两侧先去空白；null==null 通过；
                          返回 null 但期望有值、或返回有值但期望 null 均判失败）
    - complaint_type:     与 expected 完全相等
    - expected_solution:  keywords 中每个词（不区分大小写包含）都出现在返回字符串中

阈值门禁（binding）：complaint_type == 100%，order_no >= 90%，expected_solution >= 80%。
全部达标 exit 0，未达标 exit 1（并打印未达标项）。配置/数据错误 exit 2。

单条抽取抛异常（ExtractionError、网络错误、解析失败等）时该条三个字段全部判负，
继续跑完剩余样例。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langchain_core.runnables import Runnable

from app.core.config import get_settings
from app.llm.factory import get_chat_model
from app.schemas.extraction import AfterSalesExtraction
from app.services.extraction import extract_after_sales

CASES_PATH = Path(__file__).resolve().parent / "extraction_cases.jsonl"

FIELDS: tuple[str, ...] = ("order_no", "complaint_type", "expected_solution")
THRESHOLDS: dict[str, float] = {
    "complaint_type": 1.00,
    "order_no": 0.90,
    "expected_solution": 0.80,
}
THRESHOLD_LABELS: dict[str, str] = {
    "complaint_type": "== 100%",
    "order_no": ">= 90%",
    "expected_solution": ">= 80%",
}

# 样例集结构与分布约束（id 前缀即 bucket）
BUCKET_MINIMUMS: dict[str, int] = {"reg": 8, "noord": 3, "mixed": 3, "typo": 3, "vague": 3}
ALL_CATEGORIES = {"退货退款", "换货", "维修", "物流问题", "发票售后", "其他"}


# ---------------------------------------------------------------------------
# 数据加载与校验
# ---------------------------------------------------------------------------

def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    """逐行读取 jsonl 样例集，跳过空行与 # 注释行。"""
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            cases.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} JSON 解析失败: {exc}") from exc
    if not cases:
        raise ValueError(f"{path} 中没有找到任何样例")
    return cases


def validate_cases(cases: list[dict[str, Any]]) -> None:
    """校验样例集形状、分布下限、六类别覆盖与 keywords 自洽性，失败抛 ValueError。"""
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    categories: set[str] = set()
    bucket_counts: dict[str, int] = {}

    for case in cases:
        cid = str(case.get("id", ""))
        text = case.get("text")
        expected = case.get("expected")
        keywords = case.get("keywords")
        if not cid or not isinstance(text, str) or not isinstance(expected, dict) or not isinstance(keywords, list):
            problems.append(f"{cid or '?'}: 缺少 id/text/expected/keywords 或类型不对")
            continue
        if cid in seen_ids:
            problems.append(f"{cid}: id 重复")
        if text in seen_texts:
            problems.append(f"{cid}: text 重复")
        seen_ids.add(cid)
        seen_texts.add(text)

        try:
            expected_obj = AfterSalesExtraction.model_validate(expected)
        except Exception as exc:
            problems.append(f"{cid}: expected 不符合 AfterSalesExtraction: {exc}")
            continue

        if not (1 <= len(keywords) <= 3) or not all(isinstance(k, str) and k for k in keywords):
            problems.append(f"{cid}: keywords 需为 1-3 个非空字符串")
        for kw in keywords:
            if isinstance(kw, str) and kw not in expected_obj.expected_solution:
                problems.append(f"{cid}: keyword {kw!r} 不是 expected_solution 的子串（自洽性）")

        bucket_counts[cid.split("-")[0]] = bucket_counts.get(cid.split("-")[0], 0) + 1
        categories.add(expected_obj.complaint_type)

    if len(cases) < 20:
        problems.append(f"总条数 {len(cases)} < 20")
    for bucket, minimum in BUCKET_MINIMUMS.items():
        actual = bucket_counts.get(bucket, 0)
        if actual < minimum:
            problems.append(f"bucket {bucket!r}: {actual} 条 < 最低 {minimum} 条")
    missing = ALL_CATEGORIES - categories
    if missing:
        problems.append(f"类别覆盖缺失: {sorted(missing)}")

    if problems:
        raise ValueError("样例集校验失败：\n" + "\n".join(f"  - {p}" for p in problems))


# ---------------------------------------------------------------------------
# 判分
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """单条样例的三字段判分结果。"""

    case_id: str
    passed: dict[str, bool]
    got: AfterSalesExtraction | None
    error: str | None = None


@dataclass
class Summary:
    """三指标汇总。"""

    total: int
    counts: dict[str, int]
    accuracies: dict[str, float]


def score_case(
    expected: AfterSalesExtraction, keywords: list[str], got: AfterSalesExtraction | None
) -> dict[str, bool]:
    """字段级判分。got 为 None（该条抽取失败）时三字段全部判负。"""
    if got is None:
        return {field: False for field in FIELDS}

    got_order = got.order_no.strip() if isinstance(got.order_no, str) else None
    expected_order = expected.order_no.strip() if isinstance(expected.order_no, str) else None
    if expected_order is None:
        order_ok = got_order is None
    else:
        order_ok = got_order == expected_order

    solution = (got.expected_solution or "").lower()
    solution_ok = all(kw.lower() in solution for kw in keywords)

    return {
        "order_no": order_ok,
        "complaint_type": got.complaint_type == expected.complaint_type,
        "expected_solution": solution_ok,
    }


async def evaluate(cases: list[dict[str, Any]], model: Any) -> list[CaseResult]:
    """顺序跑全部样例：extract_after_sales 异常时三字段全负并继续。"""
    results: list[CaseResult] = []
    for case in cases:
        expected = AfterSalesExtraction.model_validate(case["expected"])
        got: AfterSalesExtraction | None = None
        error: str | None = None
        try:
            got = await extract_after_sales(model, case["text"])
        except Exception as exc:  # noqa: BLE001 - 单条失败不中断整场评估
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            CaseResult(
                case_id=str(case["id"]),
                passed=score_case(expected, case["keywords"], got),
                got=got,
                error=error,
            )
        )
    return results


def summarize(results: list[CaseResult]) -> Summary:
    total = len(results)
    counts = {field: sum(1 for r in results if r.passed[field]) for field in FIELDS}
    accuracies = {field: (counts[field] / total if total else 0.0) for field in FIELDS}
    return Summary(total=total, counts=counts, accuracies=accuracies)


def gate_check(summary: Summary) -> tuple[bool, list[str]]:
    """阈值门禁：返回 (是否全部达标, 未达标字段列表)。"""
    failed = [field for field in FIELDS if summary.accuracies[field] < THRESHOLDS[field]]
    return (not failed), failed


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------

def _mark(ok: bool) -> str:
    return "ok" if ok else "X"


def print_report(results: list[CaseResult], summary: Summary) -> tuple[bool, list[str]]:
    """打印逐条表格、汇总准确率与门禁结论；返回门禁结果。"""
    width = max((len(r.case_id) for r in results), default=7)
    header = f"{'id':<{width}}  {'order_no':<8}  {'complaint_type':<14}  {'expected_solution':<17}  error"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        marks = "  ".join(f"{_mark(r.passed[field]):<{len(field)}}" for field in FIELDS)
        error_note = "" if r.error is None else f"  ERROR: {r.error}"
        print(f"{r.case_id:<{width}}  {marks}{error_note}")

    percents = {field: f"{summary.accuracies[field] * 100:.1f}%" for field in FIELDS}
    print()
    print(
        f"准确率（{summary.total} 条）: "
        + " | ".join(f"{field} {summary.counts[field]}/{summary.total} ({percents[field]})" for field in FIELDS)
    )

    gate_ok, failed = gate_check(summary)
    details = " | ".join(
        f"{field} {THRESHOLD_LABELS[field]} -> {'达标' if field not in failed else '未达标'}"
        for field in FIELDS
    )
    print(f"门禁: {details}")
    if gate_ok:
        print("门禁结论: PASS")
    else:
        print(f"门禁结论: FAIL（未达标: {', '.join(failed)}）")
    return gate_ok, failed


def print_failure_details(results: list[CaseResult], cases: list[dict[str, Any]]) -> None:
    """对有字段判负的样例打印期望值与实际值，便于人工定位。"""
    failures = [r for r in results if not all(r.passed.values())]
    if not failures:
        return
    by_id = {case["id"]: case for case in cases}
    print("\n失败详情:")
    for r in failures:
        expected = AfterSalesExtraction.model_validate(by_id[r.case_id]["expected"])
        print(f"  - {r.case_id}:")
        print(f"      expected: order_no={expected.order_no!r}, complaint_type={expected.complaint_type!r}, "
              f"expected_solution={expected.expected_solution!r}")
        if r.got is None:
            print(f"      got:      <抽取失败> {r.error}")
        else:
            print(f"      got:      order_no={r.got.order_no!r}, complaint_type={r.got.complaint_type!r}, "
                  f"expected_solution={r.got.expected_solution!r}")


# ---------------------------------------------------------------------------
# 自测用脚本化假模型（不发网络请求）
# ---------------------------------------------------------------------------

def _extract_text(model_input: Any) -> str:
    """从链式调用传入的 ChatPromptValue / 消息列表 / dict 中还原顾客消息原文。"""
    messages = getattr(model_input, "messages", None)
    if messages:
        return str(messages[-1].content)
    if isinstance(model_input, dict):
        return str(model_input.get("text", ""))
    return str(model_input)


def _result_dict(parsed: AfterSalesExtraction | None, parsing_error: str | None) -> dict[str, Any]:
    return {"raw": None, "parsed": parsed, "parsing_error": parsing_error}


def _correct_behavior(case: dict[str, Any]) -> Callable[[], dict[str, Any]]:
    """返回一个行为工厂：给出该样例完全正确的结构化结果。"""
    expected = AfterSalesExtraction.model_validate(case["expected"])

    def behavior() -> dict[str, Any]:
        return _result_dict(expected.model_copy(), None)

    return behavior


class _ScriptedChain(Runnable):
    """假的结构化输出链：invoke 直接返回脚本化结果（或抛出脚本化异常）。

    langchain-core 1.x 的 Runnable 是普通 ABC（非 pydantic），用普通 __init__ 挂行为。
    """

    def __init__(self, behavior: Callable[[str], Any]):
        super().__init__()
        self._behavior = behavior

    def invoke(self, model_input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self._behavior(_extract_text(model_input))


class _ScriptedModel:
    """假模型：with_structured_output 返回按顾客原文查表的脚本化链。"""

    def __init__(self, behaviors_by_text: dict[str, Callable[[], Any]]):
        self._behaviors = behaviors_by_text

    def with_structured_output(self, schema: Any, include_raw: bool = False, **kwargs: Any) -> _ScriptedChain:
        def behavior(text: str) -> Any:
            factory = self._behaviors.get(text)
            if factory is None:
                raise AssertionError("self-test 内部错误：脚本假模型收到了不匹配任何样例的文本")
            return factory()

        return _ScriptedChain(behavior=behavior)


def build_scripted_behaviors(cases: list[dict[str, Any]]) -> dict[str, Callable[[], Any]]:
    """为每条样例生成预设行为：默认返回完全正确的答案，再注入若干错误场景。"""
    by_id = {case["id"]: case for case in cases}
    behaviors: dict[str, Callable[[], Any]] = {case["id"]: _correct_behavior(case) for case in cases}

    def wrong_complaint_type() -> dict[str, Any]:
        base = AfterSalesExtraction.model_validate(by_id["reg-02"]["expected"])
        return _result_dict(base.model_copy(update={"complaint_type": "其他"}), None)

    behaviors["reg-02"] = wrong_complaint_type

    def wrong_order_no() -> dict[str, Any]:
        base = AfterSalesExtraction.model_validate(by_id["reg-03"]["expected"])
        return _result_dict(base.model_copy(update={"order_no": "SO99999999999"}), None)

    behaviors["reg-03"] = wrong_order_no

    def missing_keyword() -> dict[str, Any]:
        base = AfterSalesExtraction.model_validate(by_id["typo-01"]["expected"])
        return _result_dict(base.model_copy(update={"expected_solution": "饭盒不保温，帮您登记处理"}), None)

    behaviors["typo-01"] = missing_keyword

    def model_call_boom() -> dict[str, Any]:
        raise RuntimeError("模拟模型调用异常")

    behaviors["reg-04"] = model_call_boom

    def parsing_failure() -> dict[str, Any]:
        return _result_dict(None, "模拟结构化解析失败")

    behaviors["reg-05"] = parsing_failure

    return behaviors


def _expected_marks(cases: list[dict[str, Any]]) -> dict[str, dict[str, bool]]:
    """脚本化场景下每条样例应有的三字段判分。"""
    all_pass = {field: True for field in FIELDS}
    marks = {case["id"]: dict(all_pass) for case in cases}
    marks["reg-02"] = {"order_no": True, "complaint_type": False, "expected_solution": True}
    marks["reg-03"] = {"order_no": False, "complaint_type": True, "expected_solution": True}
    marks["typo-01"] = {"order_no": True, "complaint_type": True, "expected_solution": False}
    for broken in ("reg-04", "reg-05"):
        marks[broken] = {field: False for field in FIELDS}
    return marks


def _summary_from_counts(total: int, counts: dict[str, int]) -> Summary:
    accuracies = {field: (counts[field] / total if total else 0.0) for field in FIELDS}
    return Summary(total=total, counts=counts, accuracies=accuracies)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def run_self_test() -> int:
    """离线自测：校验样例集 + 用脚本化假模型验证判分/汇总/门禁/异常路径/exit code。"""
    cases = load_cases()
    try:
        validate_cases(cases)
    except ValueError as exc:
        print(f"[self-test] {exc}", file=sys.stderr)
        return 1
    print(f"[self-test] 样例集校验通过：{len(cases)} 条，bucket 分布与六类别覆盖符合要求")

    try:
        # 场景一：注入错误的脚本模型 -> 逐条判分、汇总、门禁都应符合手工推演
        behaviors = build_scripted_behaviors(cases)
        fake = _ScriptedModel({case["text"]: behaviors[case["id"]] for case in cases})
        results = asyncio.run(evaluate(cases, fake))
        summary = summarize(results)
        gate_ok, failed_fields = print_report(results, summary)
        print_failure_details(results, cases)

        expected_marks = _expected_marks(cases)
        for r in results:
            assert r.passed == expected_marks[r.case_id], (
                f"{r.case_id} 判分不符：got {r.passed}, expected {expected_marks[r.case_id]}"
            )
        expected_counts = {
            field: sum(1 for case in cases if expected_marks[case["id"]][field]) for field in FIELDS
        }
        assert summary.total == len(cases), f"total 应为 {len(cases)}"
        for field in FIELDS:
            assert summary.counts[field] == expected_counts[field], (
                f"{field} 计数不符: got {summary.counts[field]}, expected {expected_counts[field]}"
            )
            # 22 条中恰好 3 条各错一个不同维度：三指标均为 19/22 = 86.4%
            assert abs(summary.accuracies[field] - 19 / 22) < 1e-9, f"{field} 准确率不符"
        # 门禁应恰好挂在 order_no 与 complaint_type 上（expected_solution 86.4% >= 80% 仍达标）
        assert gate_ok is False, "错误注入场景门禁应为 FAIL"
        assert sorted(failed_fields) == ["complaint_type", "order_no"], (
            f"未达标字段应为 order_no+complaint_type，got {failed_fields}"
        )
        print("[self-test] 场景一通过：判分、汇总与门禁 FAIL 路径符合预期（对应 live exit 1）")

        # 场景二：全对模型 -> 三指标 100%，门禁 PASS（对应 live exit 0）
        perfect = _ScriptedModel({case["text"]: _correct_behavior(case) for case in cases})
        results2 = asyncio.run(evaluate(cases, perfect))
        summary2 = summarize(results2)
        gate_ok2, failed2 = gate_check(summary2)
        assert all(all(r.passed.values()) for r in results2), "全对模型应三字段全过"
        assert all(summary2.accuracies[field] == 1.0 for field in FIELDS), "全对模型三指标应 100%"
        assert gate_ok2 and failed2 == [], "全对场景门禁应为 PASS"
        print("[self-test] 场景二通过：全对场景门禁 PASS 路径符合预期（对应 live exit 0）")

        # 场景三：门禁函数边界（含空结果不崩溃、阈值恰好压线）
        gate_empty, _ = gate_check(_summary_from_counts(0, {field: 0 for field in FIELDS}))
        assert gate_empty is False, "空结果应判 FAIL 而非崩溃"
        ok_boundary, _ = gate_check(_summary_from_counts(10, {
            "order_no": 9, "complaint_type": 10, "expected_solution": 8
        }))
        assert ok_boundary is True, "恰好压线（90%/100%/80%）应算达标"
        _, failed_c = gate_check(_summary_from_counts(10, {
            "order_no": 10, "complaint_type": 9, "expected_solution": 10
        }))
        assert failed_c == ["complaint_type"], "complaint_type <100% 必须挂"
        _, failed_o = gate_check(_summary_from_counts(10, {
            "order_no": 8, "complaint_type": 10, "expected_solution": 10
        }))
        assert failed_o == ["order_no"], "order_no <90% 必须挂"
        _, failed_s = gate_check(_summary_from_counts(10, {
            "order_no": 10, "complaint_type": 10, "expected_solution": 7
        }))
        assert failed_s == ["expected_solution"], "expected_solution <80% 必须挂"
        print("[self-test] 场景三通过：门禁边界与空结果处理符合预期")

    except AssertionError as exc:
        print(f"[self-test] 失败：{exc}", file=sys.stderr)
        return 1

    print("\n[self-test] 全部通过：样例集校验、字段判分、汇总、门禁（含 FAIL/PASS 与 exit 1/0 映射）、"
          "异常单条不中断均已验证。")
    return 0


def run_live() -> int:
    """真实模型跑分：构建模型 -> 顺序抽取 -> 判分 -> 门禁 exit code。"""
    try:
        settings = get_settings()
        model = get_chat_model(settings)
    except Exception as exc:  # noqa: BLE001 - 配置错误给出可操作的提示
        print(f"[配置错误] 无法构建聊天模型：{exc}", file=sys.stderr)
        print("请确认已在项目根目录 .env（参考 .env.example）或环境变量中配置 LLM_API_KEY。", file=sys.stderr)
        return 2

    try:
        cases = load_cases()
        validate_cases(cases)
    except ValueError as exc:
        print(f"[数据错误] {exc}", file=sys.stderr)
        return 2

    results = asyncio.run(evaluate(cases, model))
    summary = summarize(results)
    gate_ok, _ = print_report(results, summary)
    print_failure_details(results, cases)
    return 0 if gate_ok else 1


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="售后抽取评估跑分（字段级三指标 + 阈值门禁）")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="离线自测：脚本化假模型验证判分与门禁逻辑，不发起网络请求，不需要 API key",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    return run_live()


if __name__ == "__main__":
    sys.exit(main())
