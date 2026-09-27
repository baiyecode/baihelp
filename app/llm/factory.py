"""Build the chat model client from settings for OpenAI-compatible providers."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    """Create a chat model pointed at the configured OpenAI-compatible endpoint."""
    return init_chat_model(
        model=settings.llm_model,
        model_provider="openai",
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.request_timeout,
    )
