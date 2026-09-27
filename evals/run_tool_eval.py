"""工具选型评估跑分脚本(standalone harness,无 pytest;以标注样例替代 TDD)。

用法：
    uv run python evals/run_tool_eval.py             # 真实模型跑分（需已配置 LLM_API_KEY）
    uv run python evals/run_tool_eval.py --self-test # 离线自测：脚本化假模型验证判分/门禁/异常路径，
                                                     # 不发任何网络请求，不需要 API key

选型判分（模型行为不可单测，用标注样例评估集替代单测）：
    - 每条样例单轮消费：模型绑五工具后 invoke（非流式），预测 = ai.tool_calls[0]["name"]；
      模型没有发起任何工具调用（tool_calls 为空）时预测记 None；
    - expected_tool == "none"（闲聊）时预测为 None 判对、调了任何工具都判错；
      其余样例预测与 expected_tool 完全相等判对。

阈值门禁（binding）三条：
    1. 整体选型准确率 >= 90%；
    2. none 类（闲聊）误调工具次数 = 0；
    3. 「邮费是多少」样例（faq-postage-miss）必须选中 query_faq——该样例同时是
       漏召回记录：模型应选中 query_faq，但 FAQ 种子 question 刻意不含「邮费/运费」，
       下游 LIKE 检索落空，属预期漏召回；本门禁只看工具选择，不看检索命中。
全部达标 exit 0，未达标 exit 1（并打印未达标项）。配置/数据错误 exit 2。

单条 invoke 抛异常（网络错误、解析失败等）时该条判负，继续跑完剩余样例。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from langchain_core.messages import AIMessage, HumanMessage

from app.core.config import get_settings
from app.llm.factory import get_chat_model
from app.tools import get_all_tools

CASES_PATH = Path(__file__).resolve().parent / "tool_cases.jsonl"

VALID_TOOLS: frozenset[str] = frozenset(
    {"query_order", "query_product", "query_logistics", "query_faq", "create_ticket"}
)
# 闲聊样例的期望标签:不调任何工具
NONE_LABEL = "none"

# 门禁一:整体选型准确率下限
OVERALL_THRESHOLD = 0.90
# 门禁三:邮费漏召回记录样例的 id 与必须选中的工具
POSTAGE_CASE_ID = "faq-postage-miss"
POSTAGE_EXPECTED_TOOL = "query_faq"

# 样例集分布约束(id 前缀即 bucket;邮费记录样例不计入 bucket,单独校验)
BUCKET_EXPECTED: dict[str, int] = {
    "order": 4,
    "product": 3,
    "logistics": 4,
    "faq": 3,
    "ticket": 2,
    "none": 2,
}
TOTAL_EXPECTED = sum(BUCKET_EXPECTED.values()) + 1  # 18 条标注 + 1 条邮费记录样例


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
    """校验样例集形状、分布、id/query 唯一性与邮费记录样例，失败抛 ValueError。"""
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    bucket_counts: dict[str, int] = {}

    for case in cases:
        cid = str(case.get("id", ""))
        query = case.get("query")
        expected = case.get("expected_tool")
        if not cid or not isinstance(query, str) or not isinstance(expected, str):
            problems.append(f"{cid or '?'}: 缺少 id/query/expected_tool 或类型不对")
            continue
        if cid in seen_ids:
            problems.append(f"{cid}: id 重复")
        if query in seen_queries:
            problems.append(f"{cid}: query 重复")
        seen_ids.add(cid)
        seen_queries.add(query)
        if expected not in VALID_TOOLS and expected != NONE_LABEL:
            problems.append(f"{cid}: expected_tool {expected!r} 不在五工具 + none 之内")
        note = case.get("note")
        if note is not None and not isinstance(note, str):
            problems.append(f"{cid}: note 需为字符串")
        # 邮费记录样例单独校验,不进 bucket 统计
        if cid != POSTAGE_CASE_ID:
            bucket = cid.split("-")[0]
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    if len(cases) != TOTAL_EXPECTED:
        problems.append(f"总条数 {len(cases)} != {TOTAL_EXPECTED}(18 条标注 + 1 条邮费记录样例)")
    for bucket, expected_count in BUCKET_EXPECTED.items():
        actual = bucket_counts.get(bucket, 0)
        if actual != expected_count:
            problems.append(f"bucket {bucket!r}: {actual} 条 != 期望 {expected_count} 条")

    postage = [c for c in cases if str(c.get("id", "")) == POSTAGE_CASE_ID]
    if len(postage) != 1:
        problems.append(f"漏召回记录样例 {POSTAGE_CASE_ID} 应恰好 1 条,实际 {len(postage)} 条")
    elif postage[0].get("expected_tool") != POSTAGE_EXPECTED_TOOL:
        problems.append(f"{POSTAGE_CASE_ID}: expected_tool 应为 {POSTAGE_EXPECTED_TOOL}")
    elif not postage[0].get("note"):
        problems.append(f"{POSTAGE_CASE_ID}: 缺少 note(漏召回说明是必填的记录字段)")

    if problems:
        raise ValueError("样例集校验失败：\n" + "\n".join(f"  - {p}" for p in problems))


# ---------------------------------------------------------------------------
# 判分
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """单条样例的选型判分结果。"""

    case_id: str
    expected_tool: str
    predicted_tool: str | None
    passed: bool
    error: str | None = None


@dataclass
class Summary:
    """选型评估汇总(对应三条门禁)。"""

    total: int
    correct: int
    accuracy: float
    none_total: int
    none_false_calls: int
    postage_ok: bool


def judge(expected_tool: str, predicted: str | None) -> bool:
    """按标签判对错:预测 None 归一为 none 标签后与期望完全比对。"""
    predicted_label = NONE_LABEL if predicted is None else predicted
    return predicted_label == expected_tool


def predict_tool(bound_model: Any, query: str) -> tuple[str | None, str | None]:
    """单轮消费:绑定模型 invoke 一条顾客消息,取首个工具调用名。

    返回 (预测工具名或 None, 异常说明或 None);tool_calls 为空即预测 None。
    """
    try:
        ai = bound_model.invoke([HumanMessage(content=query)])
    except Exception as exc:  # noqa: BLE001 - 单条失败不中断整场评估
        return None, f"{type(exc).__name__}: {exc}"
    tool_calls = getattr(ai, "tool_calls", None) or []
    if not tool_calls:
        return None, None
    return str(tool_calls[0]["name"]), None


def evaluate(cases: list[dict[str, Any]], model: Any) -> list[CaseResult]:
    """绑一次五工具,顺序跑完全部样例;单条异常只判负该条。"""
    bound = model.bind_tools(get_all_tools())
    results: list[CaseResult] = []
    for case in cases:
        expected = str(case["expected_tool"])
        predicted, error = predict_tool(bound, str(case["query"]))
        passed = error is None and judge(expected, predicted)
        results.append(
            CaseResult(
                case_id=str(case["id"]),
                expected_tool=expected,
                predicted_tool=predicted,
                passed=passed,
                error=error,
            )
        )
    return results


def summarize(results: list[CaseResult]) -> Summary:
    """汇总整体准确率、none 类误调次数与邮费样例选型结果。"""
    total = len(results)
    correct = sum(1 for r in results if r.passed)
    none_results = [r for r in results if r.expected_tool == NONE_LABEL]
    none_false_calls = sum(1 for r in none_results if r.predicted_tool is not None)
    postage = next((r for r in results if r.case_id == POSTAGE_CASE_ID), None)
    postage_ok = (
        postage is not None
        and postage.error is None
        and postage.predicted_tool == POSTAGE_EXPECTED_TOOL
    )
    return Summary(
        total=total,
        correct=correct,
        accuracy=(correct / total if total else 0.0),
        none_total=len(none_results),
        none_false_calls=none_false_calls,
        postage_ok=postage_ok,
    )


def gate_check(summary: Summary) -> tuple[bool, list[str]]:
    """三条门禁:整体准确率 / none 误调 / 邮费样例;返回 (是否全部达标, 未达标项)。"""
    failed: list[str] = []
    if summary.accuracy < OVERALL_THRESHOLD:
        failed.append("overall")
    if summary.none_false_calls != 0:
        failed.append("none_false_calls")
    if not summary.postage_ok:
        failed.append("postage_faq")
    return (not failed), failed


GATE_LABELS: dict[str, str] = {
    "overall": f"整体准确率 >= {OVERALL_THRESHOLD:.0%}",
    "none_false_calls": "none 类误调 = 0",
    "postage_faq": f"「邮费」样例选中 {POSTAGE_EXPECTED_TOOL}",
}


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------

def _mark(ok: bool) -> str:
    return "ok" if ok else "X"


def _display_tool(name: str | None) -> str:
    return name if name is not None else "(未调工具)"


def print_report(results: list[CaseResult], summary: Summary) -> tuple[bool, list[str]]:
    """打印逐条表格、汇总指标与门禁结论;返回门禁结果。"""
    width = max((len(r.case_id) for r in results), default=7)
    header = f"{'id':<{width}}  {'expected':<15}  {'predicted':<15}  判分  error"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        error_note = "" if r.error is None else f"  ERROR: {r.error}"
        print(
            f"{r.case_id:<{width}}  {r.expected_tool:<15}  "
            f"{_display_tool(r.predicted_tool):<15}  {_mark(r.passed):<4}{error_note}"
        )

    print()
    print(f"整体准确率({summary.total} 条): {summary.correct}/{summary.total} ({summary.accuracy:.1%})")
    print(f"none 类误调: {summary.none_false_calls}/{summary.none_total}")
    print(
        f"邮费样例({POSTAGE_CASE_ID}): "
        + (
            f"选中 {POSTAGE_EXPECTED_TOOL}"
            if summary.postage_ok
            else f"未选中 {POSTAGE_EXPECTED_TOOL}"
        )
    )

    gate_ok, failed = gate_check(summary)
    details = " | ".join(
        f"{GATE_LABELS[gate]} -> {'达标' if gate not in failed else '未达标'}" for gate in GATE_LABELS
    )
    print(f"门禁: {details}")
    if gate_ok:
        print("门禁结论: PASS")
    else:
        print(f"门禁结论: FAIL（未达标: {', '.join(failed)}）")
    return gate_ok, failed


def print_failure_details(results: list[CaseResult]) -> None:
    """对判负样例打印期望与实际选型,便于人工定位。"""
    failures = [r for r in results if not r.passed]
    if not failures:
        return
    print("\n失败详情:")
    for r in failures:
        print(f"  - {r.case_id}: expected={r.expected_tool!r}, "
              f"got={_display_tool(r.predicted_tool)!r}"
              + (f" (ERROR: {r.error})" if r.error is not None else ""))


# ---------------------------------------------------------------------------
# 自测用脚本化假模型(不发网络请求)
# ---------------------------------------------------------------------------

def _extract_text(model_input: Any) -> str:
    """从 invoke 传入的消息列表 / 单条消息 / 字符串中还原顾客消息原文。"""
    if isinstance(model_input, (list, tuple)):
        model_input = model_input[-1] if model_input else ""
    content = getattr(model_input, "content", None)
    return str(model_input) if content is None else str(content)


def _scripted_ai(tool_name: str | None) -> AIMessage:
    """构造伪模型应答:tool_name=None 表示不调工具(纯文本闲聊应答)。"""
    if tool_name is None:
        return AIMessage(content="您好,很高兴为您服务~")
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": {}, "id": f"call_scripted_{tool_name}"}],
    )


class _ScriptedBound:
    """假的绑定模型:invoke 按顾客原文查表返回脚本化 AIMessage(或抛脚本化异常)。"""

    def __init__(self, behaviors_by_text: dict[str, Callable[[], Any]]):
        self._behaviors = behaviors_by_text

    def invoke(self, model_input: Any, config: Any = None, **kwargs: Any) -> Any:
        text = _extract_text(model_input)
        factory = self._behaviors.get(text)
        if factory is None:
            raise AssertionError("self-test 内部错误:脚本假模型收到了不匹配任何样例的文本")
        return factory()


class _ScriptedModel:
    """假模型:bind_tools 忽略工具列表,返回按顾客原文查表的脚本化绑定。"""

    def __init__(self, behaviors_by_text: dict[str, Callable[[], Any]]):
        self._behaviors = behaviors_by_text

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedBound:
        return _ScriptedBound(behaviors_by_text=self._behaviors)


def _correct_behavior(case: dict[str, Any]) -> Callable[[], Any]:
    """返回一个行为工厂:给出该样例完全正确的伪 tool_calls。"""
    expected = str(case["expected_tool"])
    scripted_name = None if expected == NONE_LABEL else expected
    return lambda: _scripted_ai(scripted_name)


def _behavior_boom() -> Any:
    raise RuntimeError("模拟模型调用异常")


def build_scripted_behaviors(cases: list[dict[str, Any]]) -> dict[str, Callable[[], Any]]:
    """为每条样例生成预设行为:默认返回完全正确的答案,再注入若干错误场景。"""
    by_id = {case["id"]: case for case in cases}
    behaviors: dict[str, Callable[[], Any]] = {
        str(case["query"]): _correct_behavior(case) for case in cases
    }

    # 错误注入一:选错工具(订单状态查询被答成物流查询)
    behaviors[str(by_id["order-02"]["query"])] = lambda: _scripted_ai("query_logistics")
    # 错误注入二:模型调用抛异常(该条判负,不中断整场)
    behaviors[str(by_id["logistics-04"]["query"])] = _behavior_boom
    # 错误注入三:闲聊误调工具(none 类门禁应挂)
    behaviors[str(by_id["none-01"]["query"])] = lambda: _scripted_ai("query_faq")
    # 错误注入四:邮费样例没选 query_faq(门禁三应挂)
    behaviors[str(by_id[POSTAGE_CASE_ID]["query"])] = lambda: _scripted_ai("query_product")
    return behaviors


def _expected_passes(cases: list[dict[str, Any]]) -> dict[str, bool]:
    """错误注入场景下每条样例应有的判分。"""
    broken = {"order-02", "logistics-04", "none-01", POSTAGE_CASE_ID}
    return {str(case["id"]): str(case["id"]) not in broken for case in cases}


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def _expect_value_error(action: Callable[[], Any], what: str) -> None:
    """断言 action 抛 ValueError,否则自测失败。"""
    try:
        action()
    except ValueError:
        return
    raise AssertionError(f"{what} 应抛 ValueError")


def run_self_test() -> int:
    """离线自测:校验样例集 + 用脚本化假模型验证判分/汇总/三门禁/异常路径/exit code。"""
    cases = load_cases()
    try:
        validate_cases(cases)
    except ValueError as exc:
        print(f"[self-test] {exc}", file=sys.stderr)
        return 1
    print(f"[self-test] 样例集校验通过:{len(cases)} 条(18 条标注 + 1 条邮费记录样例),"
          f"分布 {BUCKET_EXPECTED} 符合要求")

    try:
        # 场景一:注入错误的脚本模型 -> 逐条判分、汇总、三门禁都应符合手工推演
        fake = _ScriptedModel(build_scripted_behaviors(cases))
        results = evaluate(cases, fake)
        summary = summarize(results)
        gate_ok, failed_gates = print_report(results, summary)
        print_failure_details(results)

        expected_passes = _expected_passes(cases)
        for r in results:
            assert r.passed == expected_passes[r.case_id], (
                f"{r.case_id} 判分不符:got {r.passed}, expected {expected_passes[r.case_id]}"
            )
        broken_count = sum(1 for ok in expected_passes.values() if not ok)
        assert summary.total == len(cases), f"total 应为 {len(cases)}"
        assert summary.correct == len(cases) - broken_count, (
            f"correct 应为 {len(cases) - broken_count}, got {summary.correct}"
        )
        assert abs(summary.accuracy - (len(cases) - broken_count) / len(cases)) < 1e-9, (
            "整体准确率不符"
        )
        assert summary.none_total == BUCKET_EXPECTED["none"], "none 类条数不符"
        assert summary.none_false_calls == 1, "none 类误调应为 1(注入了 none-01 误调)"
        assert summary.postage_ok is False, "邮费样例被注入选错,postage_ok 应为 False"
        assert gate_ok is False, "错误注入场景门禁应为 FAIL"
        assert failed_gates == ["overall", "none_false_calls", "postage_faq"], (
            f"三条门禁应全部挂,顺序为 overall/none_false_calls/postage_faq,got {failed_gates}"
        )
        print("[self-test] 场景一通过:判分、汇总与三门禁 FAIL 路径符合预期(对应 live exit 1)")

        # 场景二:全对模型 -> 100%,三门禁 PASS(对应 live exit 0)
        perfect = _ScriptedModel(
            {str(case["query"]): _correct_behavior(case) for case in cases}
        )
        results2 = evaluate(cases, perfect)
        summary2 = summarize(results2)
        gate_ok2, failed2 = print_report(results2, summary2)
        assert all(r.passed for r in results2), "全对模型应全过"
        assert summary2.accuracy == 1.0, "全对场景准确率应 100%"
        assert summary2.none_false_calls == 0 and summary2.postage_ok, "全对场景 none/邮费应达标"
        assert gate_ok2 and failed2 == [], "全对场景门禁应为 PASS"
        print("[self-test] 场景二通过:全对场景门禁 PASS 路径符合预期(对应 live exit 0)")

        # 场景三:门禁边界(阈值恰好压线)与空结果不崩溃
        ok_boundary, _ = gate_check(
            Summary(total=10, correct=9, accuracy=0.9, none_total=2,
                    none_false_calls=0, postage_ok=True)
        )
        assert ok_boundary is True, "恰好压线(90%)应算达标"
        _, failed_low = gate_check(
            Summary(total=10, correct=8, accuracy=0.8, none_total=2,
                    none_false_calls=0, postage_ok=True)
        )
        assert failed_low == ["overall"], "准确率 <90% 必须挂 overall"
        _, failed_none = gate_check(
            Summary(total=10, correct=10, accuracy=1.0, none_total=2,
                    none_false_calls=1, postage_ok=True)
        )
        assert failed_none == ["none_false_calls"], "none 类误调 1 次必须挂"
        _, failed_postage = gate_check(
            Summary(total=10, correct=10, accuracy=1.0, none_total=2,
                    none_false_calls=0, postage_ok=False)
        )
        assert failed_postage == ["postage_faq"], "邮费样例未选 query_faq 必须挂"
        empty_summary = summarize([])
        assert empty_summary.accuracy == 0.0, "空结果准确率应为 0.0 而非崩溃"
        gate_empty, _ = gate_check(empty_summary)
        assert gate_empty is False, "空结果应判 FAIL 而非崩溃"
        print("[self-test] 场景三通过:门禁边界与空结果处理符合预期")

        # 场景四:样例集校验的失败路径(脏数据必须被拦下)
        duplicated = [dict(case) for case in cases]
        duplicated.append(dict(duplicated[0]))
        _expect_value_error(lambda: validate_cases(duplicated), "id 重复的样例集")
        wrong_tool = [dict(case) for case in cases]
        wrong_tool[0]["expected_tool"] = "query_weather"
        _expect_value_error(lambda: validate_cases(wrong_tool), "expected_tool 越界的样例集")
        wrong_dist = [case for case in cases if case["id"] != "order-01"]
        _expect_value_error(lambda: validate_cases(wrong_dist), "分布不符的样例集")
        print("[self-test] 场景四通过:脏数据(重复 id/越界工具/分布不符)均被校验拦下")

    except AssertionError as exc:
        print(f"[self-test] 失败:{exc}", file=sys.stderr)
        return 1

    print("\n[self-test] 全部通过:样例集校验、选型判分、汇总、三门禁(含 FAIL/PASS 与 exit 1/0 映射)、"
          "异常单条不中断均已验证。")
    return 0


def run_live() -> int:
    """真实模型跑分:构建模型 -> 绑五工具逐条选型 -> 判分 -> 三门禁 exit code。"""
    try:
        settings = get_settings()
        model = get_chat_model(settings)
    except Exception as exc:  # noqa: BLE001 - 配置错误给出可操作的提示
        print(f"[配置错误] 无法构建聊天模型:{exc}", file=sys.stderr)
        print("请确认已在项目根目录 .env（参考 .env.example）或环境变量中配置 LLM_API_KEY。",
              file=sys.stderr)
        return 2

    try:
        cases = load_cases()
        validate_cases(cases)
    except ValueError as exc:
        print(f"[数据错误] {exc}", file=sys.stderr)
        return 2

    results = evaluate(cases, model)
    summary = summarize(results)
    gate_ok, _ = print_report(results, summary)
    print_failure_details(results)
    return 0 if gate_ok else 1


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="工具选型评估跑分(整体准确率 + none 误调 + 邮费样例三门禁)")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="离线自测:脚本化假模型验证判分与门禁逻辑,不发起网络请求,不需要 API key",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    return run_live()


if __name__ == "__main__":
    sys.exit(main())
