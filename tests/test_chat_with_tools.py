r"""stream_chat_reply 两段式工具轮编排的行为测试(FakeToolModel + 真 ToolRegistry)。

假模型语义(task-9-brief Step 1):第一次 astream 若 messages 无 ToolMessage 且
本单配置了 tool_calls,则 yield 单个含 tool_calls 的 AIMessageChunk;否则
(次轮回灌 / 未配置 tool_calls)yield 文本 chunks。每次 astream 记录收到的
messages,bind_tools 计数,供绑定次数与第二段输入形状断言。

工具走真 ToolRegistry.execute(受控假工具 @tool 注册),覆盖:

1. 工具轮帧序:running(带 args)→ done(带截断 summary)→ 最终 deltas → [DONE];
   store 一次性追加 [Human, AIMessage(tool_calls), ToolMessage, AIMessage(最终)];
2. 写穿落库四行:user / assistant(content=None, tool_calls 非空)/
   tool(tool_call_id 对上)/ assistant(最终文本);
3. 首段无文本(Review Focus #1):assistant 行 content IS NULL,SSE 无首段 delta;
4. 绑定纪律:bind_tools 仅 1 次,第二段 astream 收到 ToolMessage,不再绑工具;
5. 第二轮带工具历史:trimmer 裁剪含 ToolMessage 不炸,SSE 与 ch01 同形,单段收敛;
6. 无 tool_calls 路径:帧序列与 ch01 完全一致(delta→[DONE]),落 user+assistant 两行;
7. 工具失败仍收敛:execute 返回 ok=False → done 帧照发(summary 含「失败」)、
   错误文本回灌 ToolMessage、最终回答照常流式、[DONE]。
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api.sse import format_delta_chunk, format_done, format_tool_event
from app.db.base import Base
from app.db.engine import build_session_factory
from app.db.models import Message
from app.services.chat import stream_chat_reply
from app.services.history import HistoryTrimmer, SessionStore
from app.services.persistence import ChatPersistence
from app.tools.registry import ToolRegistry

# 裁剪预算放宽到最大值:除测试 5 外不考察裁剪本身,只保证不炸
_BUDGET = 10_000

# FakeToolModel 首轮吐出的申请单(固定 id/args,断言逐字段可比)
_ECHO_CALL = {
    "name": "_echo_tool",
    "args": {"keyword": "物流"},
    "id": "call_1",
    "type": "tool_call",
}
_BOOM_CALL = {
    "name": "_boom_tool",
    "args": {"keyword": "物流"},
    "id": "call_9",
    "type": "tool_call",
}


# ---------------------------------------------------------------------------
# 测试基建:FakeToolModel / 受控假工具 / 工厂与模板 fixtures / 跑一轮的助手
# ---------------------------------------------------------------------------


class FakeToolModel:
    """两段式假模型:首轮产出工具申请单 chunk,回灌轮或无申请单时产出文本 chunks。"""

    def __init__(self, tool_calls: list[dict], final_chunks: list[str]) -> None:
        self._tool_calls = tool_calls
        self._final_chunks = final_chunks
        self.bind_count = 0
        self.bound_tools: list | None = None
        self.streamed_messages: list[list[BaseMessage]] = []

    def get_num_tokens(self, text: str) -> int:
        """假计数器:按字符数计 token(与 trimmer 的 lambda 计数口径一致)。"""
        return len(text)

    def bind_tools(self, tools: list) -> "FakeToolModel":
        """记录绑定次数与工具列表后原样返回自身(bound.astream 即本模型 astream)。"""
        self.bind_count += 1
        self.bound_tools = list(tools)
        return self

    async def astream(self, messages: list[BaseMessage]) -> AsyncIterator[AIMessageChunk]:
        """首轮(messages 无 ToolMessage 且配置了申请单)吐申请单 chunk;否则吐文本。"""
        self.streamed_messages.append(list(messages))
        if self._tool_calls and not any(isinstance(m, ToolMessage) for m in messages):
            yield AIMessageChunk(content="", tool_calls=self._tool_calls)
            return
        for text in self._final_chunks:
            yield AIMessageChunk(content=text)


class _KeywordInput(BaseModel):
    """受控假工具统一入参:单个 keyword 必填。"""

    keyword: str = Field(description="关键词")


@tool(args_schema=_KeywordInput)
async def _echo_tool(keyword: str) -> str:
    """原样回显关键词的假工具(成功路径)。"""
    return f"echo:{keyword}"


@tool(args_schema=_KeywordInput)
async def _boom_tool(keyword: str) -> str:
    """总是抛 RuntimeError 的假工具(registry 捕获后回灌错误文本)。"""
    raise RuntimeError("数据库连接失败")


def _make_registry(*tools: Any) -> ToolRegistry:
    """构造真实 ToolRegistry 并注册受控假工具(execute 走真实现)。"""
    registry = ToolRegistry()
    for item in tools:
        registry.register(item)
    return registry


@pytest_asyncio.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """SQLite 内存库(StaticPool 共享同一连接)+ create_all,交出会话工厂。"""
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield build_session_factory(engine)
    finally:
        await engine.dispose()


@pytest.fixture
def system_template() -> ChatPromptTemplate:
    """最小 system 模板:stream_chat_reply 以 shop_name=本店 填充。"""
    return ChatPromptTemplate.from_messages([("system", "你是{shop_name}的客服助手")])


async def _run_round(
    model: FakeToolModel,
    *,
    registry: ToolRegistry | None,
    persistence: ChatPersistence | None,
    store: SessionStore,
    system_template: ChatPromptTemplate,
    session_id: str,
    message: str,
) -> list[str]:
    """跑一轮 stream_chat_reply 并收集全部 SSE 帧(registry/persistence 走关键字传参)。"""
    trimmer = HistoryTrimmer(lambda m: len(m.text), _BUDGET)
    return [
        frame
        async for frame in stream_chat_reply(
            model,
            store,
            trimmer,
            system_template,
            session_id,
            message,
            registry=registry,
            persistence=persistence,
        )
    ]


# ---------------------------------------------------------------------------
# 1. 工具轮帧序 + 内存 store(persistence=None:仅内存,R1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_round_frames_and_stream(system_template: ChatPromptTemplate) -> None:
    """帧序 = running(带 args)→ done(带 summary)→ 最终 deltas → [DONE];
    store 一次性追加 [Human, AIMessage(tool_calls), ToolMessage, AIMessage(最终)]。
    """
    model = FakeToolModel(tool_calls=[dict(_ECHO_CALL)], final_chunks=["查询到", ":已发货"])
    registry = _make_registry(_echo_tool)
    store = SessionStore()

    frames = await _run_round(
        model,
        registry=registry,
        persistence=None,
        store=store,
        system_template=system_template,
        session_id="s-round",
        message="订单 1001 的物流到哪了?",
    )

    assert frames == [
        format_tool_event("_echo_tool", "running", args={"keyword": "物流"}),
        format_tool_event("_echo_tool", "done", summary="echo:物流"),
        format_delta_chunk("查询到"),
        format_delta_chunk(":已发货"),
        format_done(),
    ]
    assert store.get("s-round") == [
        HumanMessage(content="订单 1001 的物流到哪了?"),
        AIMessage(content="", tool_calls=[dict(_ECHO_CALL)]),
        ToolMessage(content="echo:物流", tool_call_id="call_1"),
        AIMessage(content="查询到:已发货"),
    ]


# ---------------------------------------------------------------------------
# 2. 工具轮写穿落库四行
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_round_persists_rows(
    session_factory: async_sessionmaker[AsyncSession],
    system_template: ChatPromptTemplate,
) -> None:
    """DB 四行:user / assistant(content=None, tool_calls 非空)/ tool / assistant(最终)。"""
    persistence = ChatPersistence(session_factory)
    model = FakeToolModel(
        tool_calls=[dict(_ECHO_CALL)], final_chunks=["已发货", ",预计明天送达"]
    )
    registry = _make_registry(_echo_tool)
    store = SessionStore()

    frames = await _run_round(
        model,
        registry=registry,
        persistence=persistence,
        store=store,
        system_template=system_template,
        session_id="u-db",
        message="订单 1001 的物流到哪了?",
    )

    assert frames[-1] == format_done()  # 流照常收敛,帧细节由测试 1/3 锁定

    async with session_factory() as session:
        rows = (await session.scalars(select(Message).order_by(Message.id))).all()

    assert [row.role for row in rows] == ["user", "assistant", "tool", "assistant"]
    assert len({row.conversation_id for row in rows}) == 1  # 同一会话
    assert rows[0].content == "订单 1001 的物流到哪了?"
    # assistant 申请单行:首段无文本 → content 为 NULL,tool_calls JSON 非空且保真
    assert rows[1].content is None
    assert json.loads(json.dumps(rows[1].tool_calls)) == [_ECHO_CALL]
    # tool 结果行:tool_call_id 对上,正文为工具真实返回
    assert rows[2].tool_call_id == "call_1"
    assert rows[2].content == "echo:物流"
    # 最终 assistant 正文行
    assert rows[3].content == "已发货,预计明天送达"


# ---------------------------------------------------------------------------
# 3. 首段无文本(Review Focus #1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_round_without_first_text(
    session_factory: async_sessionmaker[AsyncSession],
    system_template: ChatPromptTemplate,
) -> None:
    """首段只吐申请单不吐字:assistant 行 content IS NULL,SSE 无首段 delta(不出现空正文帧)。"""
    persistence = ChatPersistence(session_factory)
    model = FakeToolModel(tool_calls=[dict(_ECHO_CALL)], final_chunks=["亲,已", "发货了"])
    registry = _make_registry(_echo_tool)
    store = SessionStore()

    frames = await _run_round(
        model,
        registry=registry,
        persistence=persistence,
        store=store,
        system_template=system_template,
        session_id="u-silent",
        message="订单 1001 的物流到哪了?",
    )

    # 帧以 running 开头:首段没有产生任何 delta(更没有空 content 的 delta)
    assert frames == [
        format_tool_event("_echo_tool", "running", args={"keyword": "物流"}),
        format_tool_event("_echo_tool", "done", summary="echo:物流"),
        format_delta_chunk("亲,已"),
        format_delta_chunk("发货了"),
        format_done(),
    ]

    async with session_factory() as session:
        rows = (await session.scalars(select(Message).order_by(Message.id))).all()

    assistant_rows = [row for row in rows if row.role == "assistant"]
    assert rows[1].content is None  # 申请单行正文 IS NULL,而非空串
    assert len(assistant_rows) == 2
    assert assistant_rows[-1].content == "亲,已发货了"


# ---------------------------------------------------------------------------
# 4. 绑定纪律:bind 一次,第二段不再绑
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_round_binding(system_template: ChatPromptTemplate) -> None:
    """bind_tools 仅被调 1 次;第二段 astream 收到的 messages 含 ToolMessage,模型未再绑工具。"""
    model = FakeToolModel(tool_calls=[dict(_ECHO_CALL)], final_chunks=["好的", "回复"])
    registry = _make_registry(_echo_tool)
    store = SessionStore()

    await _run_round(
        model,
        registry=registry,
        persistence=None,
        store=store,
        system_template=system_template,
        session_id="s-bind",
        message="订单 1001 的物流到哪了?",
    )

    assert model.bind_count == 1
    assert model.bound_tools == registry.tools  # 注册顺序原样交给 bind_tools

    first_pass, second_pass = model.streamed_messages
    assert not any(isinstance(m, ToolMessage) for m in first_pass)
    # 第二段输入 = 第一段 messages + [AIMessage(tool_calls), ToolMessage],无再绑
    assert len(second_pass) == len(first_pass) + 2
    assert second_pass[: len(first_pass)] == first_pass
    assert isinstance(second_pass[-2], AIMessage) and second_pass[-2].tool_calls
    assert isinstance(second_pass[-1], ToolMessage)
    assert second_pass[-1].tool_call_id == "call_1"


# ---------------------------------------------------------------------------
# 5. 第二轮带工具历史:trimmer 不炸,单段收敛
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_turn_with_tool_history(
    session_factory: async_sessionmaker[AsyncSession],
    system_template: ChatPromptTemplate,
) -> None:
    """store 预置上一轮 4 条历史(含 ToolMessage):新问题不走工具,直接文本回答,
    SSE 与 ch01 同形(delta→[DONE]),trimmer 裁剪含 ToolMessage 的历史不炸。"""
    persistence = ChatPersistence(session_factory)
    store = SessionStore()
    store.append(
        "s-2nd",
        [
            HumanMessage(content="订单 1001 的物流到哪了?"),
            AIMessage(content="", tool_calls=[dict(_ECHO_CALL)]),
            ToolMessage(content="echo:物流", tool_call_id="call_1"),
            AIMessage(content="查询到:已发货"),
        ],
    )
    model = FakeToolModel(tool_calls=[dict(_ECHO_CALL)], final_chunks=["这是", "新回答"])
    registry = _make_registry(_echo_tool)

    frames = await _run_round(
        model,
        registry=registry,
        persistence=persistence,
        store=store,
        system_template=system_template,
        session_id="s-2nd",
        message="退货政策是什么?",
    )

    assert frames == [format_delta_chunk("这是"), format_delta_chunk("新回答"), format_done()]
    assert model.bind_count == 1
    assert len(model.streamed_messages) == 1  # 无工具轮 → 无第二段

    # 裁剪后的输入:system 打头、新问题收尾、上一轮 ToolMessage 完整保留
    trimmed = model.streamed_messages[0]
    assert trimmed[0].type == "system"
    assert trimmed[-1] == HumanMessage(content="退货政策是什么?")
    assert any(isinstance(m, ToolMessage) for m in trimmed)

    # store 只追加本轮一问一答,旧历史(含 ToolMessage)原样保留
    history = store.get("s-2nd")
    assert len(history) == 6
    assert history[-2] == HumanMessage(content="退货政策是什么?")
    assert history[-1] == AIMessage(content="这是新回答")
    assert history[2] == ToolMessage(content="echo:物流", tool_call_id="call_1")

    # 写穿照常:本轮 user + assistant 两行
    async with session_factory() as session:
        rows = (await session.scalars(select(Message).order_by(Message.id))).all()
    assert [row.role for row in rows] == ["user", "assistant"]
    assert rows[0].content == "退货政策是什么?"
    assert rows[1].content == "这是新回答"


# ---------------------------------------------------------------------------
# 6. 无 tool_calls 路径回归(与 ch01 完全一致)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_tool_round_unchanged(
    session_factory: async_sessionmaker[AsyncSession],
    system_template: ChatPromptTemplate,
) -> None:
    """模型不调工具(已绑工具但申请单为空):帧序列与 ch01 完全一致,落 user+assistant 两行。"""
    persistence = ChatPersistence(session_factory)
    model = FakeToolModel(tool_calls=[], final_chunks=["您好", "有什么可以帮您?"])
    registry = _make_registry(_echo_tool)
    store = SessionStore()

    frames = await _run_round(
        model,
        registry=registry,
        persistence=persistence,
        store=store,
        system_template=system_template,
        session_id="u-plain",
        message="你好",
    )

    assert frames == [
        format_delta_chunk("您好"),
        format_delta_chunk("有什么可以帮您?"),
        format_done(),
    ]
    assert model.bind_count == 1
    assert len(model.streamed_messages) == 1  # 无第二段
    assert store.get("u-plain") == [
        HumanMessage(content="你好"),
        AIMessage(content="您好有什么可以帮您?"),
    ]

    async with session_factory() as session:
        rows = (await session.scalars(select(Message).order_by(Message.id))).all()

    assert [row.role for row in rows] == ["user", "assistant"]
    assert rows[0].content == "你好"
    assert rows[1].content == "您好有什么可以帮您?"


# ---------------------------------------------------------------------------
# 7. 工具失败仍收敛
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_error_still_converges(system_template: ChatPromptTemplate) -> None:
    """execute 返回 ok=False:done 帧照发(summary 含「失败」),错误文本回灌 ToolMessage,
    最终回答照常流式并以 [DONE] 收敛——工具失败不炸流。"""
    model = FakeToolModel(
        tool_calls=[dict(_BOOM_CALL)], final_chunks=["抱歉", "工具开小差了"]
    )
    registry = _make_registry(_boom_tool)
    store = SessionStore()

    frames = await _run_round(
        model,
        registry=registry,
        persistence=None,
        store=store,
        system_template=system_template,
        session_id="s-boom",
        message="订单 1001 的物流到哪了?",
    )

    assert frames[0] == format_tool_event("_boom_tool", "running", args={"keyword": "物流"})
    done_event = json.loads(frames[1].removeprefix("data: ").strip())
    assert done_event["tool"]["status"] == "done"
    assert "失败" in done_event["tool"]["summary"]  # registry 错误文本进 summary
    assert frames[2:-1] == [
        format_delta_chunk("抱歉"),
        format_delta_chunk("工具开小差了"),
    ]
    assert frames[-1] == format_done()

    # 错误文本原样回灌 ToolMessage,模型据此作答;store 仍完整落四条
    history = store.get("s-boom")
    assert len(history) == 4
    assert history[2].tool_call_id == "call_9"
    assert history[2].content.startswith("工具执行失败")
    assert history[3] == AIMessage(content="抱歉工具开小差了")
