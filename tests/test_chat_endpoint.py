"""Tests for the SSE chat streaming endpoint (app.main + app.api.chat)."""

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage

import app.api.chat as chat_api
from app.api.sse import format_delta_chunk, format_done
from app.core.config import Settings
from app.main import app
from app.services.history import SessionStore


class FakeModel:
    """``astream`` 可控假模型：按配置产出 chunk，可在第 N 个 chunk 处抛异常。"""

    def __init__(self, chunks: list[str], fail_on: int | None = None) -> None:
        self._chunks = chunks
        self._fail_on = fail_on
        self.streamed_messages: list[list[BaseMessage]] = []

    def get_num_tokens(self, text: str) -> int:
        """假计数器：与真实签名一致（面向 str），按字符数计 token。"""
        return len(text)

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
    settings = Settings(_env_file=None, llm_api_key="sk-test")
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
async def client():
    """每个测试独立会话存储，避免会话跨测试串扰。"""
    app.state.store = SessionStore()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as async_client:
        yield async_client


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
