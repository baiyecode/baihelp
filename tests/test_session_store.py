"""Tests for app.services.history."""

from langchain_core.messages import AIMessage, HumanMessage

from app.services.history import SessionStore


def test_append_then_get_roundtrip() -> None:
    """Messages appended to a session come back in order."""
    store = SessionStore()
    messages = [HumanMessage(content="你好"), AIMessage(content="您好！")]

    store.append("session-1", messages)

    assert store.get("session-1") == messages


def test_unknown_session_returns_empty() -> None:
    """Getting an unknown session id yields an empty history."""
    store = SessionStore()

    assert store.get("no-such-session") == []


def test_sessions_isolated() -> None:
    """Two session ids never bleed into each other."""
    store = SessionStore()
    first = [HumanMessage(content="第一条")]
    second = [HumanMessage(content="第二条")]

    store.append("session-a", first)
    store.append("session-b", second)

    assert store.get("session-a") == first
    assert store.get("session-b") == second
