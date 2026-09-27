"""In-memory conversation history storage."""

from typing import Sequence

from langchain_core.messages import BaseMessage


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
