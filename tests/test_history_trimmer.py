"""Tests for HistoryTrimmer in app.services.history."""

import logging

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.services.history import HistoryTrimmer


def _constant_counter(_message: BaseMessage) -> int:
    """Fake counter: every message costs exactly 100 tokens."""
    return 100


def _text_length_counter(message: BaseMessage) -> int:
    """按字符数计 token:消息字数即成本,预算可精确卡在成对/成组的缝隙上。"""
    return len(message.text)


def _conversation() -> list[BaseMessage]:
    """[system, u1, a1, u2, a2] — two complete user/assistant turns."""
    return [
        SystemMessage(content="系统提示"),
        HumanMessage(content="u1"),
        AIMessage(content="a1"),
        HumanMessage(content="u2"),
        AIMessage(content="a2"),
    ]


def test_first_turn_keeps_system_and_user() -> None:
    """A first turn with no history yet is returned unchanged."""
    messages = [SystemMessage(content="系统提示"), HumanMessage(content="u1")]
    trimmer = HistoryTrimmer(_constant_counter, budget=500)

    result = trimmer.trim(messages)

    assert result == messages
    assert len(messages) == 2  # input list never mutated


def test_exact_fit_keeps_all() -> None:
    """budget=500 equals 5 x 100: system and both complete turns are kept."""
    messages = _conversation()
    trimmer = HistoryTrimmer(_constant_counter, budget=500)

    result = trimmer.trim(messages)

    assert result == messages
    assert len(messages) == 5  # input list never mutated


def test_drops_oldest_turn_first() -> None:
    """budget=450: the oldest complete turn is dropped, the newest kept."""
    system, u1, a1, u2, a2 = _conversation()
    trimmer = HistoryTrimmer(_constant_counter, budget=450)

    result = trimmer.trim([system, u1, a1, u2, a2])

    assert result == [system, u2, a2]


def test_single_message_over_budget_fallback(caplog: pytest.LogCaptureFixture) -> None:
    """budget=50: the newest user message alone exceeds it -> [system, u2] + warning."""
    system, u1, a1, u2, a2 = _conversation()
    trimmer = HistoryTrimmer(_constant_counter, budget=50)

    with caplog.at_level(logging.WARNING):
        result = trimmer.trim([system, u1, a1, u2, a2])

    assert result == [system, u2]
    assert "预算" in caplog.text


# ---------------------------------------------------------------------------
# 工具轮原子组感知(修复轮 1):杜绝孤儿 ToolMessage 致上游 400
# ---------------------------------------------------------------------------


def _tool_round_messages() -> tuple[SystemMessage, list[BaseMessage], HumanMessage]:
    """构造一轮完整工具往返 + 待答新问句,字数即 token:

    [system(2)] + [Human(100), AIMessage(tool_calls, 空文本 0),
    ToolMessage(100), AIMessage(100)] + [Human(10)] —— 工具轮整组 300,
    system + 新问句垫底 12。预算 270/294 时,旧按对裁剪装得下
    [ToolMessage, AIMessage] 对、装不下前置 [Human, AIMessage(tool_calls)] 对。
    """
    system = SystemMessage(content="系统")
    tool_round = [
        HumanMessage(content="问" * 100),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "query_logistics",
                    "args": {"order_id": "1001"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="果" * 100, tool_call_id="call_1"),
        AIMessage(content="答" * 100),
    ]
    newest = HumanMessage(content="新" * 10)
    return system, tool_round, newest


@pytest.mark.parametrize("budget", [270, 294])
def test_tool_round_never_split_into_orphan_tool_message(budget: int) -> None:
    """孤儿复现(评审参数 budget=270/294):预算装得下 [ToolMessage, AIMessage] 对、
    装不下前置 [Human, AIMessage(tool_calls)] 对时,旧按对裁剪会产出
    [system, ToolMessage, AIMessage, Human] —— 孤儿 ToolMessage 发往上游是硬 400,
    且坏历史永久滞留(每次请求都失败 → 不再追加历史 → 会话死锁)。
    工具轮是原子组:装不下整组就整组剔除,输出只留 [system, 新问句]。
    """
    system, tool_round, newest = _tool_round_messages()
    trimmer = HistoryTrimmer(_text_length_counter, budget=budget)

    result = trimmer.trim([system, *tool_round, newest])

    assert result == [system, newest]
    assert not any(isinstance(m, ToolMessage) for m in result)


def test_tool_round_kept_atomically_on_exact_fit() -> None:
    """原子组完整保留(回归护栏):预算恰好装下整组(fit 为 <=)时,工具轮四条
    一条不少——修复不得把装得下的完整轮误删,更不得在组中间拆对。"""
    system, tool_round, newest = _tool_round_messages()
    messages = [system, *tool_round, newest]
    trimmer = HistoryTrimmer(_text_length_counter, budget=312)  # 12 垫底 + 整组 300

    result = trimmer.trim(messages)

    assert result == messages
    assert len(messages) == 6  # input list never mutated


def test_orphan_tool_message_dropped_even_within_budget() -> None:
    """协议残缺组普适清扫:ToolMessage 的前置不是带 tool_calls 的 AIMessage 时,
    即使预算充裕,该 ToolMessage 连同同轮残片(其后的人工回答)也必须整组剔除,
    不得出现在裁剪输出中;更旧的合法轮次不受牵连。"""
    system = SystemMessage(content="系统")
    plain_turn = [HumanMessage(content="一" * 100), AIMessage(content="二" * 100)]
    malformed_round = [
        HumanMessage(content="问" * 50),
        ToolMessage(content="孤" * 50, tool_call_id="call_x"),
        AIMessage(content="终" * 50),
    ]
    newest = HumanMessage(content="新" * 10)
    trimmer = HistoryTrimmer(_text_length_counter, budget=10_000)

    result = trimmer.trim([system, *plain_turn, *malformed_round, newest])

    assert result == [system, *plain_turn, newest]
    assert not any(isinstance(m, ToolMessage) for m in result)


def test_budget_walk_continues_past_dropped_malformed_round() -> None:
    """残缺组剔除先于预算步进:预算 222 装得下旧普通轮(200+12 垫底)、装不下
    残缺组(300)。若残缺组触发「遇装不下即停」,旧轮会被误杀;过滤后的
    预算步进应继续考察更旧合法轮次。"""
    system = SystemMessage(content="系统")
    plain_turn = [HumanMessage(content="一" * 100), AIMessage(content="二" * 100)]
    malformed_round = [
        HumanMessage(content="问" * 100),
        ToolMessage(content="孤" * 100, tool_call_id="call_x"),
        AIMessage(content="终" * 100),
    ]
    newest = HumanMessage(content="新" * 10)
    trimmer = HistoryTrimmer(_text_length_counter, budget=222)

    result = trimmer.trim([system, *plain_turn, *malformed_round, newest])

    assert result == [system, *plain_turn, newest]


def test_tool_calls_without_result_round_dropped() -> None:
    """镜像残缺:申请单(AIMessage.tool_calls)没等到任何 tool 结果——上游同样硬 400,
    与孤儿 ToolMessage 同罪,整组剔除。"""
    system = SystemMessage(content="系统")
    broken_round = [
        HumanMessage(content="问" * 50),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "query_logistics",
                    "args": {"order_id": "1001"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
    ]
    newest = HumanMessage(content="新" * 10)
    trimmer = HistoryTrimmer(_text_length_counter, budget=10_000)

    result = trimmer.trim([system, *broken_round, newest])

    assert result == [system, newest]
    assert not any(m.tool_calls for m in result if isinstance(m, AIMessage))
