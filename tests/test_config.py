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
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "MILVUS_DB_PATH",
    "RETRIEVAL_TOP_K",
    "RETRIEVAL_SCORE_THRESHOLD",
    "CHUNK_MAX_CHARS",
    "CHUNK_OVERLAP_CHARS",
    "QA_MINE_BATCH_SIZE",
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
    settings = Settings(_env_file=None, embedding_api_key="e")
    assert settings.llm_provider == "deepseek"
    assert settings.llm_base_url == "https://api.deepseek.com/v1"
    assert settings.llm_model == "deepseek-chat"
    assert settings.history_token_budget == 3000
    assert settings.request_timeout == 60


def test_database_settings_defaults() -> None:
    """默认连接串指向 Docker MySQL 的 baihelp 库,超时/重试为章节数值。"""
    settings = Settings(_env_file=None, llm_api_key="sk-test", embedding_api_key="e")
    assert settings.database_url == (
        "mysql+aiomysql://baihelp:baihelp@127.0.0.1:3306/baihelp?charset=utf8mb4"
    )
    assert settings.tool_timeout_seconds == 10.0
    assert settings.tool_max_retries == 1

def test_database_url_env_override() -> None:
    """环境变量可覆盖连接串(测试换 SQLite 用同一开关)。"""
    settings = Settings(_env_file=None, llm_api_key="sk-test", embedding_api_key="e", database_url="sqlite+aiosqlite://")
    assert settings.database_url == "sqlite+aiosqlite://"


def test_settings_knowledge_keys():
    s = Settings(_env_file=None, llm_api_key="k", embedding_api_key="e")
    assert s.embedding_base_url == "https://api.siliconflow.cn/v1"
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.embedding_dim == 1024
    assert s.milvus_db_path == "data/milvus/baihelp.db"
    assert (s.retrieval_top_k, s.retrieval_score_threshold) == (3, 0.5)
    assert (s.chunk_max_chars, s.chunk_overlap_chars) == (500, 80)
    assert s.qa_mine_batch_size == 4

def test_settings_requires_embedding_key():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_api_key="k")
