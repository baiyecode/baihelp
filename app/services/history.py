"""In-memory conversation history storage."""

import logging
from typing import Callable, Sequence

from langchain_core.messages import BaseMessage, HumanMessage

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
    """Trim a conversation window to fit a token budget, keeping whole turns.

    ``SessionStore.get`` returns the live internal list, so ``trim`` never
    mutates its input; it always builds and returns a fresh list.
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
        - Completed history before the newest user message is walked from the
          end in complete (user, assistant) pairs (index stepping by 2). Each
          pair is kept only while the running total stays within the budget;
          the walk stops at the first pair that does not fit — newer turns
          outrank older ones, no skipping. A pair may land exactly on the
          budget (fit is ``<=``).
        - The system message and the newest user message are unconditional:
          only the historical pairs are budget-enforced. When the input does
          not end with an unanswered user message, the budget applies to
          complete pairs only.
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
        # unconditionally and trim only the completed turns before it.
        pending: BaseMessage | None = None
        if rest and isinstance(rest[-1], HumanMessage):
            pending = rest[-1]
            rest = rest[:-1]

        total = self._token_counter(system)
        if pending is not None:
            total += self._token_counter(pending)

        kept: list[BaseMessage] = []
        index = len(rest)
        while index >= 2:
            pair = rest[index - 2 : index]  # one complete (user, assistant) turn
            pair_cost = sum(self._token_counter(m) for m in pair)
            if total + pair_cost > self._budget:
                break
            kept = pair + kept
            total += pair_cost
            index -= 2

        result = [system, *kept]
        if pending is not None:
            result.append(pending)
        return result
