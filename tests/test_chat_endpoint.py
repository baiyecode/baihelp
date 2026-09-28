"""Tests for the SSE chat streaming endpoint (app.main + app.api.chat)."""

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, Request
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import app.api.chat as chat_api
from app.api.sse import format_delta_chunk, format_done
from app.core.config import Settings
from app.db import models  # noqa: F401  import 即注册 ORM,create_all 才有表可建
from app.db.base import Base
from app.db.engine import build_session_factory
from app.main import app
from app.services.history import SessionStore
from app.tools import build_default_registry, get_all_tools
from tests.test_tools_db import FakeRetriever


class FakeModel:
    """``astream`` 可控假模型：按配置产出 chunk，可在第 N 个 chunk 处抛异常。"""

    def __init__(self, chunks: list[str], fail_on: int | None = None) -> None:
        self._chunks = chunks
        self._fail_on = fail_on
        self.streamed_messages: list[list[BaseMessage]] = []
        # bind_tools 每次调用收到的工具列表（端点接线断言用：registry 五工具绑给模型）
        self.bound_tools: list[list] = []

    def get_num_tokens(self, text: str) -> int:
        """假计数器：与真实签名一致（面向 str），按字符数计 token。"""
        return len(text)

    def bind_tools(self, tools: list) -> "FakeModel":
        """记录收到的工具列表并返回 self（真模型 bind_tools 的最小仿真）。"""
        self.bound_tools.append(list(tools))
        return self

    async def astream(self, messages: list[BaseMessage]) -> AsyncIterator[AIMessageChunk]:
        self.streamed_messages.append(list(messages))
        for index, text in enumerate(self._chunks, start=1):
            if self._fail_on is not None and index == self._fail_on:
                raise RuntimeError("模拟上游故障")
            yield AIMessageChunk(content=text)


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch, chunks: list[str], fail_on: int | None = None
) -> FakeModel:
    """Replace the endpoint's model/settings seams with hermetic fakes.

    DI 缝隙：端点模块级的 ``get_model`` / ``get_settings``，monkeypatch 后
    真实模型工厂与真实配置都不会被触达。
    """
    fake = FakeModel(chunks, fail_on=fail_on)
    settings = Settings(_env_file=None, llm_api_key="sk-test", embedding_api_key="e")
    monkeypatch.setattr(chat_api, "get_model", lambda: fake)
    monkeypatch.setattr(chat_api, "get_settings", lambda: settings)
    return fake


def _sse_payloads(body: str) -> list[str]:
    """Split an SSE wire body into its ``data:`` payloads (terminator dropped)."""
    return [
        block.removeprefix("data: ")
        for block in body.split("\n\n")
        if block.startswith("data: ")
    ]


@pytest_asyncio.fixture
async def client(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[httpx.AsyncClient]:
    """每个测试独立会话存储与会话工厂，避免跨测试串扰。

    ASGITransport 不触发 lifespan,故端点依赖的三样在此手工补齐:
    - monkeypatch ``chat_api.get_session_factory`` 注入 SQLite 内存工厂
      (StaticPool 共享同一连接 + create_all,写穿落库有表可写);
    - app.state.tool_registry 挂真五工具注册表(与 lifespan 组装同构);
    - app.state.retriever 挂空结果 FakeRetriever:get_retriever 的缺省取用来源
      (与 lifespan 组装同构,假模型不发工具申请单,此处仅备而不用)。
    """
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory: async_sessionmaker[AsyncSession] = build_session_factory(engine)
        monkeypatch.setattr(chat_api, "get_session_factory", lambda request: factory)
        app.state.store = SessionStore()
        app.state.tool_registry = build_default_registry(
            Settings(_env_file=None, llm_api_key="sk-test", embedding_api_key="e")
        )
        app.state.retriever = FakeRetriever([])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as async_client:
            yield async_client
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_stream_emits_delta_events_then_done(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """每个 chunk 依次下发 delta 事件，最后以 [DONE] 终止；响应为 SSE 类型。"""
    chunks = ["您好", "！请问", "有什么可以帮您？"]
    _install_fakes(monkeypatch, chunks)

    response = await client.post(
        "/api/chat/stream", json={"session_id": "s-1", "message": "在吗"}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    expected = "".join(format_delta_chunk(text) for text in chunks) + format_done()
    assert response.text == expected


@pytest.mark.asyncio
async def test_reply_persisted_to_store(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """流结束后，store 里是 [用户消息, 完整聚合回复]，而非逐段碎片。"""
    chunks = ["您好", "，亲", "～"]
    _install_fakes(monkeypatch, chunks)

    await client.post(
        "/api/chat/stream", json={"session_id": "s-2", "message": "在吗"}
    )

    history = app.state.store.get("s-2")
    assert history == [HumanMessage(content="在吗"), AIMessage(content="您好，亲～")]


@pytest.mark.asyncio
async def test_unknown_session_id_creates_conversation(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """陌生 session_id 不报错：200 且 store 里注册出了该会话。"""
    _install_fakes(monkeypatch, ["好的"])

    response = await client.post(
        "/api/chat/stream", json={"session_id": "brand-new", "message": "你好"}
    )

    assert response.status_code == 200
    # 直接观察内部 dict：get() 自身也会注册，membership 才能证明是端点注册的。
    assert "brand-new" in app.state.store._sessions


@pytest.mark.asyncio
async def test_stream_error_midway_emits_error_event(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """第 2 个 chunk 处故障：首段 delta 已下发，随后 error 事件 + [DONE]，状态仍 200。"""
    _install_fakes(monkeypatch, ["第一段", "第二段"], fail_on=2)

    response = await client.post(
        "/api/chat/stream", json={"session_id": "s-4", "message": "在吗"}
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    assert len(payloads) == 3
    first_event = json.loads(payloads[0])
    assert first_event["choices"][0]["delta"]["content"] == "第一段"
    error_event = json.loads(payloads[1])
    assert error_event["error"]["message"]
    assert payloads[2] == "[DONE]"
    # 出错不写入任何历史：该会话没有消息记录。
    assert app.state.store.get("s-4") == []


@pytest.mark.asyncio
async def test_endpoint_binds_registry_tools(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """端点把 app.state.tool_registry 的五工具绑给模型：fake 收到长度 5 的列表。"""
    fake = _install_fakes(monkeypatch, ["好的"])

    await client.post(
        "/api/chat/stream", json={"session_id": "s-tools", "message": "在吗"}
    )

    # 只绑一次（第一段），且工具列表与 get_all_tools 的注册顺序完全一致
    assert len(fake.bound_tools) == 1
    assert len(fake.bound_tools[0]) == 5
    assert [tool.name for tool in fake.bound_tools[0]] == [
        tool.name for tool in get_all_tools()
    ]


def test_endpoint_persistence_uses_app_state_factory() -> None:
    """端点写穿用的是 lifespan 预建那台工厂：get_session_factory(request) 原样返回 app.state.session_factory。"""
    # 用一次性 FastAPI 实例当 scope["app"],不触碰共享模块级 app 的 state
    fresh_app = FastAPI()
    fresh_app.state.session_factory = object()  # 哨兵:只验身份,不需要真工厂
    request = Request({"type": "http", "app": fresh_app})

    assert chat_api.get_session_factory(request) is fresh_app.state.session_factory


def test_endpoint_retriever_uses_app_state_retriever() -> None:
    """检索器缝:query_faq 用的是 lifespan 预建那只检索器,get_retriever(request) 原样返回 app.state.retriever。"""
    fresh_app = FastAPI()
    fresh_app.state.retriever = object()  # 哨兵:只验身份,不需要真检索器
    request = Request({"type": "http", "app": fresh_app})

    assert chat_api.get_retriever(request) is fresh_app.state.retriever
