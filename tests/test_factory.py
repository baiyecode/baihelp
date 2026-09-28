"""Tests for app.llm.factory."""

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import Settings
from app.llm import factory


def _make_settings() -> Settings:
    """Hermetic settings with non-default values to catch wiring mistakes."""
    return Settings(
        _env_file=None,
        llm_model="unit-test-model",
        llm_base_url="https://llm.example.invalid/v1",
        llm_api_key="sk-unit-test",
        embedding_api_key="e",
        request_timeout=77,
    )


def test_factory_passes_env_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory must forward settings to init_chat_model verbatim, pinned to openai."""
    settings = _make_settings()
    captured: dict[str, Any] = {}
    sentinel = object()

    def recorder(model: str, **kwargs: Any) -> Any:
        captured["model"] = model
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(factory, "init_chat_model", recorder)

    result = factory.get_chat_model(settings)

    assert result is sentinel
    assert captured["model"] == "unit-test-model"
    assert captured["model_provider"] == "openai"
    assert captured["base_url"] == "https://llm.example.invalid/v1"
    assert captured["api_key"] == "sk-unit-test"
    assert captured["timeout"] == 77


def test_factory_builds_openai_compatible_model() -> None:
    """Real init_chat_model accepts the pinned kwargs and yields a BaseChatModel."""
    model = factory.get_chat_model(_make_settings())

    assert isinstance(model, BaseChatModel)
