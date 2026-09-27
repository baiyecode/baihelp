"""Tests for HistoryTrimmer in app.services.history."""

import logging

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.services.history import HistoryTrimmer


def _constant_counter(_message: BaseMessage) -> int:
    """Fake counter: every message costs exactly 100 tokens."""
    return 100


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
