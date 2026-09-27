"""In-memory conversation history storage."""

import logging
from typing import Callable, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

logger = logging.getLogger(__name__)


class SessionStore:
    """Keep each session's message history in a process-local dict.

    No locking is needed: all access happens on a single asyncio event loop.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = {}

    def get(self, session_id: str) -> list[BaseMessage]:
        """Return the session's history, creating an empty one for unknown ids."""
        return self._sessions.setdefault(session_id, [])

    def append(self, session_id: str, messages: Sequence[BaseMessage]) -> None:
        """Append messages to the session's history."""
        self._sessions.setdefault(session_id, []).extend(messages)


class HistoryTrimmer:
    """Trim a conversation window to fit a token budget, keeping whole rounds.

    ``SessionStore.get`` returns the live internal list, so ``trim`` never
    mutates its input; it always builds and returns a fresh list.

    轮次(round)是裁剪的原子单位:普通轮即一对 ``[Human, AIMessage]``;工具轮
    ``[Human, AIMessage(tool_calls), *ToolMessages, AIMessage(最终)]`` 整组纳入/
    剔除,绝不拆散——拆散会产生孤儿 ToolMessage,上游 OpenAI 兼容 API 直接硬 400,
    且坏历史永久滞留(每次请求都失败 → 不再追加新历史 → 会话死锁)。协议残缺的
    组(孤儿 ToolMessage / 零结果申请单)在预算步进前整组丢弃。
    """

    def __init__(self, token_counter: Callable[[BaseMessage], int], budget: int) -> None:
        self._token_counter = token_counter
        self._budget = budget

    def trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Return a budget-fitting window of ``messages`` without mutating them.

        Convention: ``messages[0]`` is the system message; the rest is the
        history, optionally ending with the newest user message that has not
        been answered yet.

        Semantics:
        - The system message is always kept first.
        - The newest user message (the last ``HumanMessage``) is the message
          about to be answered and is always kept. If it alone exceeds the
          whole budget, fall back to ``[system, that user message]`` and log a
          warning: the reply would be based on an over-budget window either
          way, so the question must survive.
        - Completed history is partitioned into user-led rounds (each
          ``HumanMessage`` starts a new round). A round is an atomic group:
          kept or dropped whole, never split, so an orphan ``ToolMessage``
          can never appear in the output.
        - Rounds violating the tool protocol are dropped before the budget
          walk (see ``_is_well_formed_round``); dropping them does not stop
          the walk — older valid rounds are still considered.
        - Valid rounds are then walked from the end (newest first). Each
          round is kept only while the running total stays within the
          budget; the walk stops at the first round that does not fit —
          newer rounds outrank older ones, no skipping. A round may land
          exactly on the budget (fit is ``<=``).
        - The system message and the newest user message are unconditional:
          only the completed rounds are budget-enforced. When the input does
          not end with an unanswered user message, the budget applies to
          complete rounds only.
        """
        system = messages[0]
        rest = messages[1:]

        newest_user_index = next(
            (i for i in range(len(rest) - 1, -1, -1) if isinstance(rest[i], HumanMessage)),
            None,
        )
        if newest_user_index is not None:
            newest_user = rest[newest_user_index]
            newest_user_cost = self._token_counter(newest_user)
            if newest_user_cost > self._budget:
                logger.warning(
                    "最新用户消息自身超过预算（%d > %d），仅保留 system 与该消息",
                    newest_user_cost,
                    self._budget,
                )
                return [system, newest_user]

        # A trailing user message is the unanswered new message: keep it
        # unconditionally and trim only the completed rounds before it.
        pending: BaseMessage | None = None
        if rest and isinstance(rest[-1], HumanMessage):
            pending = rest[-1]
            rest = rest[:-1]

        total = self._token_counter(system)
        if pending is not None:
            total += self._token_counter(pending)

        # 组感知裁剪:分组 → 剔除协议残缺组 → 整组自新向旧纳入。
        # 工具轮永不拆散;预算语义(无跳选、恰好装下保留)与按对时代一致。
        kept: list[BaseMessage] = []
        for group in reversed(self._valid_rounds(rest)):
            round_cost = sum(self._token_counter(m) for m in group)
            if total + round_cost > self._budget:
                break
            kept = [*group, *kept]
            total += round_cost

        result = [system, *kept]
        if pending is not None:
            result.append(pending)
        return result

    @classmethod
    def _valid_rounds(cls, rest: list[BaseMessage]) -> list[list[BaseMessage]]:
        """按用户轮分组并过滤协议残缺组。

        残缺组整组丢弃(含组首问句)——半个轮次同样污染上下文,且「输出只含
        完整合法轮次」是本组件的最小不变量;过滤先于预算步进,更旧的合法
        轮次不受牵连。
        """
        return [
            group
            for group in cls._partition_rounds(rest)
            if cls._is_well_formed_round(group)
        ]

    @staticmethod
    def _partition_rounds(rest: list[BaseMessage]) -> list[list[BaseMessage]]:
        """把除 system 外的历史按用户轮分组:每个 HumanMessage 开启新组,
        组内是该问句及其全部回复(工具轮的申请单/结果/最终回答同组)。
        ToolMessage 不开新组——它天然归属其前置申请单所在轮。"""
        rounds: list[list[BaseMessage]] = []
        current: list[BaseMessage] = []
        for message in rest:
            if isinstance(message, HumanMessage) and current:
                rounds.append(current)
                current = [message]
            else:
                current.append(message)
        if current:
            rounds.append(current)
        return rounds

    @staticmethod
    def _is_well_formed_round(group: list[BaseMessage]) -> bool:
        """工具协议校验(二者任一违反即整组残缺,上游 OpenAI 兼容 API 都硬 400):

        - 任一 ToolMessage 的前置必须是本组的 AIMessage(tool_calls) 或其连续结果;
        - 任一申请单(AIMessage.tool_calls)必须收到至少一条 tool 结果(组收尾同样校验)。
        """
        open_calls = False    # 处于「申请单已见、结果段未闭合」状态
        results_seen = False  # 当前申请单是否已收到至少一条结果
        for message in group:
            if isinstance(message, ToolMessage):
                if not open_calls:
                    return False  # 孤儿 ToolMessage:前置不是申请单
                results_seen = True  # 连续结果共享同一申请单,open_calls 保持
            elif isinstance(message, AIMessage) and message.tool_calls:
                if open_calls and not results_seen:
                    return False  # 上一张申请单零结果又被新申请单覆盖
                open_calls = True
                results_seen = False
            else:
                if open_calls and not results_seen:
                    return False  # 结果段被普通消息闭合但零结果
                open_calls = False
        return not (open_calls and not results_seen)  # 组尾申请单零结果同样残缺
