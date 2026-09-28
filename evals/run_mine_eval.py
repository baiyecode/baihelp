r"""QA 挖矿评估跑分脚本(standalone harness,无 pytest;以标注样例替代 TDD)。

用法：
    uv run python evals/run_mine_eval.py             # 真 LLM 抽取跑分(需已配置 LLM_API_KEY)
    uv run python evals/run_mine_eval.py --self-test # 离线自测:罐头 LLM 验证判分/门禁/脏输出路径,
                                                     # 不发任何网络请求,不需要 API key / MySQL

判分(模型行为不可单测,用标注样例评估集替代单测;数据集契约见 qa_mine_cases.jsonl):
    - 每条样例是一通客服对话;live 用 mine_qa 同款 Prompt(qa_extraction_prompt.md)逐通
      调真 LLM 抽取,输出的 JSON 数组经 extract_json_array 解析并按 _extract_batch 同款
      口径清洗(缺字段/空串剔除);评估一次只送一通会话(挖矿管线是按批,这里逐通送
      以隔离单例判分),source 用样例 id;
    - 挖矿样例(expect_qa 非空):每条 expect 要求存在至少一个抽取对,其 question 含
      q_contains 全部关键词、answer 含 a_contains 全部关键词;
    - 噪音样例(expect_qa == []):纯寒暄 / 无实质答案转人工 / 敷衍推诿话术 /
      具体订单号个案 + mock 数据对话,应抽出 0 对。

门禁(binding)两条:
    1. 全部 expect_qa 被满足(每个挖矿样例的每条 expect 都被至少一个抽取对命中);
    2. 噪音行 0 抽取(寒暄/转人工/敷衍话术/个案 mock 数据不得被挖成知识)。
全部达标 exit 0,未达标 exit 1(并打印未达标项)。配置/数据错误 exit 2。

单条抽取抛异常(网络错误、输出解析失败等)该条判负,继续跑完剩余样例。
数据集为空时门禁真空为真——由 validate_cases(总条数/分布/至少 1 条噪音)拦住。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.core.config import get_settings
from app.knowledge.mine_qa import extract_json_array
from app.prompts.loader import load_qa_extraction_prompt

CASES_PATH = Path(__file__).resolve().parent / "qa_mine_cases.jsonl"

VALID_ROLES: frozenset[str] = frozenset({"user", "assistant"})

# 样例集分布约束(id 前缀即 bucket;钉死 Task 7 六话题 + 变说法 + 包邮数字 + 一会话双话题
# + 噪音 + 敷衍话术/个案 mock 两类反例)
BUCKET_EXPECTED: dict[str, int] = {
    "refund": 2,  # 退款到账时效(直接问 + 「钱什么时候回来」变说法)
    "address": 1,  # 修改收货地址
    "coupon": 1,  # 优惠券过期
    "invoice": 1,  # 发票抬头修改
    "exchange": 1,  # 换货运费承担
    "presale": 1,  # 预售发货时间
    "postage": 1,  # 包邮门槛满 99(关键数字保留)
    "multi": 1,  # 一通会话双话题(发货时效 + 支付方式)
    "noise": 2,  # 噪音对照:纯寒暄 + 无实质答案转人工
    "warranty": 1,  # 保修时效
    "hedge": 1,  # 反例:敷衍推诿回答(无任何具体事实,纯话术)
    "mock": 1,  # 反例:具体订单号个案查询 + 系统 mock 数据
}
TOTAL_EXPECTED = sum(BUCKET_EXPECTED.values())  # 14 条


# ---------------------------------------------------------------------------
# 数据加载与校验
# ---------------------------------------------------------------------------

def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    """逐行读取 jsonl 样例集,跳过空行与 # 注释行。"""
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
    """校验样例集形状、分布、噪音行数、id 唯一性与关键词可满足性,失败抛 ValueError。

    关键词可满足性:q_contains 每项须出现在对话的 user 轮原文里、a_contains 每项
    须出现在 assistant 轮原文里(抽取的 q/a 源自对话原话,Prompt 规则要求保留原始
    问法与关键口径);关键词在对话里都凑不齐的 expect 是作者拍脑袋,先拦下。
    """
    problems: list[str] = []
    seen_ids: set[str] = set()
    bucket_counts: dict[str, int] = {}
    noise_total = 0

    for case in cases:
        cid = str(case.get("id", ""))
        dialogue = case.get("dialogue")
        expect_qa = case.get("expect_qa")
        if not cid:
            problems.append("样例缺少 id")
            continue
        if cid in seen_ids:
            problems.append(f"{cid}: id 重复")
        seen_ids.add(cid)
        if not isinstance(dialogue, list) or not dialogue:
            problems.append(f"{cid}: dialogue 必须是非空轮次列表")
            continue
        user_texts: list[str] = []
        assistant_texts: list[str] = []
        for index, turn in enumerate(dialogue):
            if (
                not isinstance(turn, dict)
                or turn.get("role") not in VALID_ROLES
                or not isinstance(turn.get("content"), str)
                or not turn["content"].strip()
            ):
                problems.append(
                    f"{cid}: 第 {index} 轮须为含 role(user/assistant)与非空 content 的对象"
                )
                continue
            (user_texts if turn["role"] == "user" else assistant_texts).append(turn["content"])
        if not isinstance(expect_qa, list):
            problems.append(f"{cid}: expect_qa 必须是列表")
            continue
        for index, expect in enumerate(expect_qa):
            if not isinstance(expect, dict):
                problems.append(f"{cid}: expect {index} 必须是对象")
                continue
            for key, texts in (("q_contains", user_texts), ("a_contains", assistant_texts)):
                value = expect.get(key)
                if (
                    not isinstance(value, list)
                    or not value
                    or not all(isinstance(item, str) and item.strip() for item in value)
                ):
                    problems.append(f"{cid}: expect {index} 的 {key} 须为非空字符串的非空列表")
                    continue
                if texts:
                    missing = [item for item in value if not any(item in text for text in texts)]
                    if missing:
                        problems.append(
                            f"{cid}: expect {index} 的 {key} 关键词 {missing} 不在对话原文中,"
                            "语料满足不了的 expect 须回改"
                        )
        if not expect_qa:
            noise_total += 1
        bucket = cid.split("-")[0]
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    if len(cases) != TOTAL_EXPECTED:
        problems.append(f"总条数 {len(cases)} != {TOTAL_EXPECTED}")
    if noise_total < 1:
        problems.append("至少需要 1 条纯噪音对话(expect_qa=[])作为 0 抽取门禁的对照")
    for bucket, expected_count in BUCKET_EXPECTED.items():
        actual = bucket_counts.get(bucket, 0)
        if actual != expected_count:
            problems.append(f"bucket {bucket!r}: {actual} 条 != 期望 {expected_count} 条")

    if problems:
        raise ValueError("样例集校验失败:\n" + "\n".join(f"  - {p}" for p in problems))


# ---------------------------------------------------------------------------
# 会话渲染与抽取(mine_qa 同款口径)
# ---------------------------------------------------------------------------

def render_dialogue(case_id: str, dialogue: list[dict[str, Any]]) -> str:
    """把评估对话渲染成 mine_qa._render_conversation 同款会话块(供 Prompt 消费)。

    评估集对话为人工预筛的 user/assistant 纯文本轮,管线里的确定性预过滤
    (tool 回执行/过短残句)在此是不必要的;块格式必须与挖矿一致,否则评估
    失真。source= 用样例 id,既照抄进抽取结果的 source 字段,也供 --self-test
    的罐头 LLM 按例查表。
    """
    lines = [f"【会话开始 source={case_id}】"]
    lines.extend(f"{turn['role']}:{turn['content']}" for turn in dialogue)
    lines.append("【会话结束】")
    return "\n".join(lines)


def sanitize_pairs(items: list[dict]) -> list[tuple[str, str]]:
    """按 mine_qa._extract_batch 同款口径清洗抽取元素,返回 (question, answer) 对。

    缺字段/空串/非字符串的脏元素静默剔除,不让单条脏元素炸整例;source 不参与
    判分,直接丢弃。
    """
    pairs: list[tuple[str, str]] = []
    for item in items:
        question, answer = item.get("question"), item.get("answer")
        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(answer, str) or not answer.strip():
            continue
        pairs.append((question.strip(), answer.strip()))
    return pairs


async def extract_pairs(llm: Any, prompt: Any, block: str) -> list[tuple[str, str]]:
    """渲染 Prompt 调 LLM 抽一通对话,返回清洗后的 (question, answer) 对列表。

    LLM 返回非文本内容视为输出契约违约抛 ValueError(与 mine_qa 同口径)。
    """
    messages = prompt.invoke({"conversations": block}).to_messages()
    response = await llm.ainvoke(messages)
    content = response.content
    if not isinstance(content, str):
        raise ValueError(f"LLM 返回非文本内容:{type(content).__name__}")
    return sanitize_pairs(extract_json_array(content))


# ---------------------------------------------------------------------------
# 判分
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """单条样例的挖矿判分结果。"""

    case_id: str
    is_noise: bool
    expect_total: int
    expect_met: int
    extracted: int
    passed: bool
    error: str | None = None


@dataclass
class Summary:
    """挖矿评估汇总(对应两条门禁)。"""

    normal_total: int
    expect_total: int
    expect_met: int
    noise_total: int
    noise_zero_extract: int


def _expect_satisfied(expect: dict[str, Any], pairs: list[tuple[str, str]]) -> bool:
    """一条 expect 是否被至少一个抽取对满足(q/a 关键词逐项子串)。"""
    return any(
        all(key in question for key in expect["q_contains"])
        and all(key in answer for key in expect["a_contains"])
        for question, answer in pairs
    )


async def evaluate(cases: list[dict[str, Any]], llm: Any, prompt: Any) -> list[CaseResult]:
    """逐通送 LLM 抽取并判分;单例异常只判负该例,不中断整场评估。"""
    results: list[CaseResult] = []
    for case in cases:
        case_id = str(case["id"])
        expects = list(case["expect_qa"])
        is_noise = not expects
        try:
            block = render_dialogue(case_id, case["dialogue"])
            pairs = await extract_pairs(llm, prompt, block)
            error = None
        except Exception as exc:  # noqa: BLE001 - 单例失败不中断整场评估
            pairs, error = [], f"{type(exc).__name__}: {exc}"
        if is_noise:
            passed = error is None and len(pairs) == 0
            results.append(
                CaseResult(case_id, True, 0, 0, len(pairs), passed, error)
            )
        else:
            met = sum(1 for expect in expects if _expect_satisfied(expect, pairs))
            passed = error is None and met == len(expects)
            results.append(
                CaseResult(case_id, False, len(expects), met, len(pairs), passed, error)
            )
    return results


def summarize(results: list[CaseResult]) -> Summary:
    """汇总 expect 满足数与噪音 0 抽取数(异常例不计入达标)。"""
    normal = [r for r in results if not r.is_noise]
    noise = [r for r in results if r.is_noise]
    return Summary(
        normal_total=len(normal),
        expect_total=sum(r.expect_total for r in normal),
        expect_met=sum(r.expect_met for r in normal),
        noise_total=len(noise),
        noise_zero_extract=sum(
            1 for r in noise if r.error is None and r.extracted == 0
        ),
    )


def gate_check(summary: Summary) -> tuple[bool, list[str]]:
    """两条门禁:expect_qa 全满足 / 噪音行 0 抽取;返回 (是否全部达标, 未达标项)。"""
    failed: list[str] = []
    if summary.expect_met != summary.expect_total:
        failed.append("expect_qa")
    if summary.noise_zero_extract != summary.noise_total:
        failed.append("noise_zero")
    return (not failed), failed


GATE_LABELS: dict[str, str] = {
    "expect_qa": "全部 expect_qa 被满足",
    "noise_zero": "噪音行抽取 = 0 对",
}


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------

def _mark(ok: bool) -> str:
    return "ok" if ok else "X"


def print_report(results: list[CaseResult], summary: Summary) -> tuple[bool, list[str]]:
    """打印逐条表格、汇总指标与门禁结论;返回门禁结果。"""
    width = max((len(r.case_id) for r in results), default=7)
    header = f"{'id':<{width}}  {'类型':<4}  {'expect':>7}  {'抽取':>4}  判分  error"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        expect_cell = "噪音" if r.is_noise else f"{r.expect_met}/{r.expect_total}"
        error_note = "" if r.error is None else f"  ERROR: {r.error}"
        print(
            f"{r.case_id:<{width}}  {'噪音' if r.is_noise else '挖矿':<4}  "
            f"{expect_cell:>7}  {r.extracted:>4}  {_mark(r.passed):<4}{error_note}"
        )

    print()
    print(
        f"expect_qa 满足({summary.normal_total} 个挖矿样例): "
        f"{summary.expect_met}/{summary.expect_total}"
    )
    print(
        f"噪音行 0 抽取({summary.noise_total} 条噪音): "
        f"{summary.noise_zero_extract}/{summary.noise_total}"
    )

    gate_ok, failed = gate_check(summary)
    details = " | ".join(
        f"{GATE_LABELS[gate]} -> {'达标' if gate not in failed else '未达标'}" for gate in GATE_LABELS
    )
    print(f"门禁: {details}")
    if gate_ok:
        print("门禁结论: PASS")
    else:
        print(f"门禁结论: FAIL(未达标: {', '.join(failed)})")
    return gate_ok, failed


def print_failure_details(results: list[CaseResult]) -> None:
    """对判负样例打印期望与实际抽取概况,便于人工定位。"""
    failures = [r for r in results if not r.passed]
    if not failures:
        return
    print("\n失败详情:")
    for r in failures:
        if r.is_noise:
            note = f"噪音行应抽 0 对,实际抽 {r.extracted} 对"
        else:
            note = f"{r.expect_total - r.expect_met}/{r.expect_total} 条 expect 未满足,实际抽 {r.extracted} 对"
        if r.error is not None:
            note += f"(ERROR: {r.error})"
        print(f"  - {r.case_id}: {note}")


# ---------------------------------------------------------------------------
# 自测用罐头 LLM(不发网络请求)
# ---------------------------------------------------------------------------

_SOURCE_RE = re.compile(r"【会话开始 source=([^】]*)】")


def _source_from_block(model_input: Any) -> str:
    """从 Prompt 渲染出的消息(列表)里还原会话块的 source= 样例 id。"""
    if isinstance(model_input, (list, tuple)):
        model_input = model_input[-1] if model_input else ""
    content = getattr(model_input, "content", None)
    text = str(model_input) if content is None else str(content)
    match = _SOURCE_RE.search(text)
    if match is None:
        raise AssertionError("self-test 内部错误:罐头 LLM 收到了不含 source 标记的会话块")
    return match.group(1)


def _canned_response(payload: Any) -> Any:
    """构造伪模型应答:payload 即 ainvoke 应返回的 .content(字符串或脏值)。"""
    return SimpleNamespace(content=payload)


class _ScriptedLLM:
    """罐头 LLM:ainvoke 按会话块 source= 样例 id 查表返回脚本化应答(或抛脚本化异常)。"""

    def __init__(self, behaviors_by_source: dict[str, Callable[[], Any]]):
        self._behaviors = behaviors_by_source

    async def ainvoke(self, model_input: Any, config: Any = None, **kwargs: Any) -> Any:
        source = _source_from_block(model_input)
        factory = self._behaviors.get(source)
        if factory is None:
            raise AssertionError(
                f"self-test 内部错误:罐头 LLM 收到了未脚本化的会话 source={source}"
            )
        return factory()


def _correct_payload(case: dict[str, Any]) -> str:
    """按样例的 expect 生成完全可满足判分的罐头 JSON 数组文本。

    每条 expect 各造一对(question/answer 直接由关键词拼接,子串判分必然命中);
    噪音样例造空数组。内容不追求拟真——自测的是判分与门禁逻辑,不是模型。
    """
    case_id = str(case["id"])
    pairs = [
        {
            "source": case_id,
            "question": "".join(expect["q_contains"]),
            "answer": "".join(expect["a_contains"]),
        }
        for expect in case["expect_qa"]
    ]
    return json.dumps(pairs, ensure_ascii=False)


def _behavior_boom() -> Any:
    raise RuntimeError("模拟 LLM 调用异常")


def build_scripted_behaviors(cases: list[dict[str, Any]]) -> dict[str, Callable[[], Any]]:
    """为每条样例生成预设行为:默认返回完全正确的抽取,再注入若干错误场景。"""
    behaviors: dict[str, Callable[[], Any]] = {
        str(case["id"]): (lambda case=case: _canned_response(_correct_payload(case)))
        for case in cases
    }

    # 错误注入一:挖矿样例被抽成空数组(expect 漏满足)
    behaviors["refund-01"] = lambda: _canned_response("[]")
    # 错误注入二:抽到了问题但答案口径丢了关键词(a_contains 漏满足)
    behaviors["coupon-01"] = lambda: _canned_response(
        json.dumps(
            [{"source": "coupon-01", "question": "优惠券过期了还能用吗?", "answer": "不可以了哦。"}],
            ensure_ascii=False,
        )
    )
    # 错误注入三:噪音行被挖出 1 对(噪音门禁应挂)
    behaviors["noise-01"] = lambda: _canned_response(
        json.dumps(
            [{"source": "noise-01", "question": "在吗?", "answer": "在的亲~"}],
            ensure_ascii=False,
        )
    )
    # 错误注入四:输出不是 JSON(解析失败,该例判负不中断整场)
    behaviors["invoice-01"] = lambda: _canned_response("抱歉,我无法输出 JSON 数组。")
    return behaviors


def _expected_passes(cases: list[dict[str, Any]]) -> dict[str, bool]:
    """错误注入场景下每条样例应有的判分。"""
    broken = {"refund-01", "coupon-01", "noise-01", "invoice-01"}
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


async def run_self_test() -> int:
    """离线自测:校验样例集 + 罐头 LLM 验证判分/汇总/两门禁(含 FAIL/PASS)/脏输出路径。"""
    try:
        cases = load_cases()
        validate_cases(cases)
    except ValueError as exc:
        print(f"[self-test] {exc}", file=sys.stderr)
        return 1
    noise_rows = sum(1 for case in cases if not case["expect_qa"])
    print(
        f"[self-test] 样例集校验通过:{len(cases)} 条(挖矿 {len(cases) - noise_rows} + "
        f"噪音 {noise_rows}),分布 {BUCKET_EXPECTED} 符合要求"
    )
    prompt = load_qa_extraction_prompt()

    try:
        # 场景一:注入错误的罐头 LLM -> 逐条判分、汇总、两门禁都应符合手工推演
        fake = _ScriptedLLM(build_scripted_behaviors(cases))
        results = await evaluate(cases, fake, prompt)
        summary = summarize(results)
        gate_ok, failed_gates = print_report(results, summary)
        print_failure_details(results)

        expected_passes = _expected_passes(cases)
        for r in results:
            assert r.passed == expected_passes[r.case_id], (
                f"{r.case_id} 判分不符:got {r.passed}, expected {expected_passes[r.case_id]}"
            )
        broken_count = sum(1 for ok in expected_passes.values() if not ok)
        broken_normal = broken_count - 1  # 噪音注入不计 expect 汇总
        assert summary.normal_total == len(cases) - noise_rows, "挖矿样例条数不符"
        assert summary.expect_total == sum(
            len(case["expect_qa"]) for case in cases if case["expect_qa"]
        ), "expect 总数不符"
        assert summary.expect_met == summary.expect_total - broken_normal, (
            f"expect_met 应为 {summary.expect_total - broken_normal}, got {summary.expect_met}"
        )
        assert summary.noise_total == noise_rows, "噪音条数不符"
        assert summary.noise_zero_extract == noise_rows - 1, (
            "噪音 0 抽取应差 1(注入了 noise-01 误抽)"
        )
        assert gate_ok is False, "错误注入场景门禁应为 FAIL"
        assert failed_gates == ["expect_qa", "noise_zero"], (
            f"两条门禁应全部挂,顺序为 expect_qa/noise_zero,got {failed_gates}"
        )
        print("[self-test] 场景一通过:判分、汇总与两门禁 FAIL 路径符合预期(对应 live exit 1)")

        # 场景二:全对罐头 LLM -> expect 全满足、噪音 0 抽取,两门禁 PASS(对应 live exit 0)
        perfect = _ScriptedLLM(
            {
                str(case["id"]): (lambda case=case: _canned_response(_correct_payload(case)))
                for case in cases
            }
        )
        results2 = await evaluate(cases, perfect, prompt)
        summary2 = summarize(results2)
        gate_ok2, failed2 = print_report(results2, summary2)
        assert all(r.passed for r in results2), "全对模型应全过"
        assert summary2.expect_met == summary2.expect_total, "全对场景 expect 应全满足"
        assert summary2.noise_zero_extract == summary2.noise_total, "全对场景噪音应 0 抽取"
        assert gate_ok2 and failed2 == [], "全对场景门禁应为 PASS"
        print("[self-test] 场景二通过:全对场景门禁 PASS 路径符合预期(对应 live exit 0)")

        # 场景三:抽取口径边界 —— 围栏 JSON / 脏元素剔除 / 非文本应答 / 次序无关
        by_id = {str(case["id"]): case for case in cases}
        good_pair = {"source": "x", "question": "q", "answer": "a"}

        def _correct_pair(case_id: str) -> dict[str, str]:
            """取该样例全对罐头输出里的第一个抽取对(字段合法,判分必满足)。"""
            return json.loads(_correct_payload(by_id[case_id]))[0]

        # 3a. Markdown 代码围栏包裹的 JSON 照样解析(extract_json_array 容错)
        fenced = _ScriptedLLM(
            {
                "refund-02": lambda: _canned_response(
                    "```json\n" + _correct_payload(by_id["refund-02"]) + "\n```"
                )
            }
        )
        fenced_pairs = await extract_pairs(
            fenced, prompt, render_dialogue("refund-02", by_id["refund-02"]["dialogue"])
        )
        assert fenced_pairs and all(
            _expect_satisfied(expect, fenced_pairs) for expect in by_id["refund-02"]["expect_qa"]
        ), "围栏 JSON 里的正确抽取应可判分"
        # 3b. 脏元素(缺 answer / 空问题 / 空 answer)被静默剔除,好元素幸存
        dirty = _ScriptedLLM(
            {
                "address-01": lambda: _canned_response(
                    json.dumps(
                        [
                            {"source": "x", "question": "没有答案的问题"},
                            {"source": "x", "question": "", "answer": "空问题"},
                            {"source": "x", "question": "问题不是空串", "answer": "   "},
                            _correct_pair("address-01"),
                        ],
                        ensure_ascii=False,
                    )
                )
            }
        )
        dirty_pairs = await extract_pairs(
            dirty, prompt, render_dialogue("address-01", by_id["address-01"]["dialogue"])
        )
        assert len(dirty_pairs) == 1 and _expect_satisfied(
            by_id["address-01"]["expect_qa"][0], dirty_pairs
        ), "脏元素应被静默剔除,好元素幸存"
        # 3b'. 非对象元素属输出契约违约,extract_json_array 立即抛 ValueError
        # (与 mine_qa 口径一致:契约违约炸整批,与单元素脏字段静默剔除不同层)
        contract = _ScriptedLLM(
            {"address-01": lambda: _canned_response('[{"question": "q", "answer": "a"}, "oops"]')}
        )
        try:
            await extract_pairs(
                contract, prompt, render_dialogue("address-01", by_id["address-01"]["dialogue"])
            )
            raise AssertionError("非对象元素应抛 ValueError(契约违约)")
        except ValueError:
            pass
        # 3c. 多对抽取中 expect 被第二对满足也算命中(次序无关)
        second_pair = _ScriptedLLM(
            {
                "exchange-01": lambda: _canned_response(
                    json.dumps(
                        [good_pair, _correct_pair("exchange-01")],
                        ensure_ascii=False,
                    )
                )
            }
        )
        second_pairs = await extract_pairs(
            second_pair, prompt, render_dialogue("exchange-01", by_id["exchange-01"]["dialogue"])
        )
        assert len(second_pairs) == 2, "两对干净抽取都应保留"
        assert _expect_satisfied(by_id["exchange-01"]["expect_qa"][0], second_pairs), (
            "expect 应可被任一抽取对满足(次序无关)"
        )
        # 3d. LLM 返回非文本内容:该例判 ValueError(判负)而非崩溃
        non_text = _ScriptedLLM({"presale-01": lambda: _canned_response(42)})
        try:
            await extract_pairs(
                non_text, prompt, render_dialogue("presale-01", by_id["presale-01"]["dialogue"])
            )
            raise AssertionError("非文本内容应抛 ValueError")
        except ValueError:
            pass
        # 3e. LLM 调用抛异常:evaluate 单例判负、不中断
        boom_cases = [dict(case) for case in cases if case["id"] in {"postage-01", "noise-02"}]
        boom = _ScriptedLLM({str(case["id"]): _behavior_boom for case in boom_cases})
        boom_results = await evaluate(boom_cases, boom, prompt)
        assert len(boom_results) == 2 and all(not r.passed for r in boom_results), (
            "异常例应判负"
        )
        assert all(r.error is not None for r in boom_results), "异常例应带 ERROR 说明"
        print("[self-test] 场景三通过:围栏 JSON/脏元素剔除/次序无关/非文本与异常判负符合预期")

        # 场景四:样例集校验的失败路径(脏数据必须被拦下)
        duplicated = [dict(case) for case in cases]
        duplicated.append(dict(duplicated[0]))
        _expect_value_error(lambda: validate_cases(duplicated), "id 重复的样例集")
        bad_role = [dict(case) for case in cases]
        bad_role[0] = dict(bad_role[0])
        bad_role[0]["dialogue"] = [{"role": "system", "content": "你是谁"}]
        _expect_value_error(lambda: validate_cases(bad_role), "role 越界的样例集")
        bad_expect = [dict(case) for case in cases]
        bad_expect[0] = dict(bad_expect[0])
        bad_expect[0]["expect_qa"] = [{"q_contains": ["关键词"]}]
        _expect_value_error(lambda: validate_cases(bad_expect), "expect 缺 a_contains 的样例集")
        ghost_keyword = [dict(case) for case in cases]
        ghost_victim = next(case for case in ghost_keyword if case["expect_qa"])
        ghost_victim["expect_qa"] = [
            dict(ghost_victim["expect_qa"][0], a_contains=["对话里根本没有的词"])
        ]
        _expect_value_error(
            lambda: validate_cases(ghost_keyword), "关键词不在对话原文中的样例集"
        )
        empty_dialogue = [dict(case) for case in cases]
        empty_dialogue[0] = dict(empty_dialogue[0])
        empty_dialogue[0]["dialogue"] = []
        _expect_value_error(lambda: validate_cases(empty_dialogue), "空对话的样例集")
        # 无噪音对照:剔除全部 expect_qa==[] 行(不论 id 前缀是 noise/hedge/mock),
        # 让「至少 1 条噪音」守卫真正触发
        no_noise = [dict(case) for case in cases if case["expect_qa"]]
        _expect_value_error(lambda: validate_cases(no_noise), "无噪音对照的样例集")
        wrong_dist = [case for case in cases if case["id"] != "refund-02"]
        _expect_value_error(lambda: validate_cases(wrong_dist), "分布不符的样例集")
        print("[self-test] 场景四通过:脏数据(重复 id/role 越界/缺字段/幽灵关键词/空对话/无噪音/分布不符)均被拦下")

    except AssertionError as exc:
        print(f"[self-test] 失败:{exc}", file=sys.stderr)
        return 1

    print(
        "\n[self-test] 全部通过:样例集校验、抽取清洗、判分、两门禁(含 FAIL/PASS 与 "
        "exit 1/0 映射)、单例异常不中断均已验证。"
    )
    return 0


def run_live() -> int:
    """真 LLM 跑分:构建模型 -> 逐通抽取 -> 判分 -> 两门禁 exit code。"""
    try:
        settings = get_settings()
        cases = load_cases()
        validate_cases(cases)
    except Exception as exc:  # noqa: BLE001 - 配置/数据错误给出可操作的提示
        print(f"[配置/数据错误] {exc}", file=sys.stderr)
        print("请确认已在项目根目录 .env(参考 .env.example)配置 LLM_API_KEY。", file=sys.stderr)
        return 2

    # LLM factory 依赖较重(拉起 openai/langchain 链路),留到确需真跑时导入
    from app.llm.factory import get_chat_model

    try:
        model = get_chat_model(settings)
    except Exception as exc:  # noqa: BLE001 - 模型装配失败归为配置错误
        print(f"[配置错误] 无法构建聊天模型:{exc}", file=sys.stderr)
        return 2

    prompt = load_qa_extraction_prompt()
    results = asyncio.run(evaluate(cases, model, prompt))
    summary = summarize(results)
    gate_ok, _ = print_report(results, summary)
    print_failure_details(results)
    return 0 if gate_ok else 1


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(
        description="QA 挖矿评估跑分(expect_qa 全满足 + 噪音 0 抽取两门禁)"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="离线自测:罐头 LLM 验证判分与门禁逻辑,不发起网络请求,不需要 API key",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return asyncio.run(run_self_test())
    return run_live()


if __name__ == "__main__":
    sys.exit(main())
