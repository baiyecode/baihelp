"""Tests for app.core.config."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings

SETTINGS_ENV_VARS = (
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_API_KEY",
    "HISTORY_TOKEN_BUDGET",
    "REQUEST_TIMEOUT",
)


def _clear_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every Settings-related env var so each test starts clean."""
    for var in SETTINGS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_missing_api_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_settings_env(monkeypatch)
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "llm_api_key" in str(excinfo.value)


def test_defaults_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_settings_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "test-api-key")
    settings = Settings(_env_file=None)
    assert settings.history_token_budget == 3000
    assert settings.request_timeout == 60
