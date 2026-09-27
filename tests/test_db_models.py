"""app.db.models / app.db.engine 的行为测试(SQLite 内存库跑 create_all)。

MySQL 端绝不 create_all(建表唯一依据是用户 DDL);测试库由 ORM create_all
建等价 schema,验证默认值、外键可写与枚举/JSON 的往返保真。
"""

from collections.abc import AsyncIterator
from json import dumps, loads

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db import models
from app.db.base import Base
from app.db.engine import build_engine, build_session_factory, ping


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


async def _seed_conversation(session: AsyncSession, user_id: str) -> models.Conversation:
    """先落一通会话,供 messages / tickets 外键引用。"""
    conversation = models.Conversation(user_id=user_id)
    session.add(conversation)
    await session.flush()
    return conversation


@pytest.mark.asyncio
async def test_conversation_status_defaults_to_in_progress(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """插入 Conversation 不传 status,默认「进行中」。"""
    async with session_factory() as session:
        conversation = models.Conversation(user_id="u-001")
        session.add(conversation)
        await session.commit()

        assert conversation.id is not None
        assert conversation.status == "进行中"


@pytest.mark.asyncio
async def test_ticket_status_defaults_to_pending(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """插入 Ticket 不传 status,默认「待处理」。"""
    async with session_factory() as session:
        conversation = await _seed_conversation(session, "u-002")
        ticket = models.Ticket(
            ticket_no="T20260701008",
            conversation_id=conversation.id,
            description="商品破损,要求退款",
            ticket_type="售后",
        )
        session.add(ticket)
        await session.commit()

        assert ticket.status == "待处理"


@pytest.mark.asyncio
async def test_message_foreign_key_writable_and_tool_roundtrip(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Message 挂 conversation_id 外键可写;role='tool' + tool_call_id + JSON tool_calls 往返保真。"""
    tool_calls = [
        {
            "id": "call_001",
            "type": "function",
            "function": {"name": "query_order", "arguments": dumps({"order_id": "1001"})},
        }
    ]
    async with session_factory() as session:
        conversation = await _seed_conversation(session, "u-003")
        session.add_all(
            [
                models.Message(
                    conversation_id=conversation.id,  # 外键引用真实会话
                    role="assistant",
                    content=None,  # assistant 纯工具调用时正文可为空
                    tool_calls=tool_calls,
                ),
                models.Message(
                    conversation_id=conversation.id,
                    role="tool",
                    content="订单 1001 已发货",
                    tool_call_id="call_001",
                ),
            ]
        )
        await session.commit()

    # 换新会话重查,绕开身份映射缓存,验证真实落库往返
    async with session_factory() as reader:
        rows = (
            (await reader.execute(select(models.Message).order_by(models.Message.id)))
            .scalars()
            .all()
        )

    assert [message.role for message in rows] == ["assistant", "tool"]
    assert rows[0].content is None
    assert loads(dumps(rows[0].tool_calls)) == tool_calls  # JSON 往返保真
    assert rows[1].content == "订单 1001 已发货"
    assert rows[1].tool_call_id == "call_001"


@pytest.mark.asyncio
async def test_ticket_type_roundtrip(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """中文枚举值原样存取:Ticket(ticket_type='售后') 往返保真。"""
    async with session_factory() as session:
        conversation = await _seed_conversation(session, "u-004")
        session.add(
            models.Ticket(
                ticket_no="T20260701009",
                conversation_id=conversation.id,
                description="七天无理由退货",
                ticket_type="售后",
            )
        )
        await session.commit()

    async with session_factory() as reader:
        ticket = await reader.get(models.Ticket, "T20260701009")

    assert ticket is not None
    assert ticket.ticket_type == "售后"
    assert ticket.conversation_id == conversation.id


@pytest.mark.asyncio
async def test_ping_ok_on_sqlite_memory() -> None:
    """连通时 ping 静默通过(以 SQLite 内存库验证 SELECT 1 路径)。"""
    engine = build_engine("sqlite+aiosqlite://")
    try:
        await ping(engine)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_ping_raises_runtime_error_with_compose_hint_when_unreachable() -> None:
    """连不上时抛 RuntimeError,消息含「docker compose up -d」操作提示。"""
    # 127.0.0.1:1 无服务监听,连接必被拒绝(离线确定性)
    engine = build_engine("mysql+aiomysql://baihelp:baihelp@127.0.0.1:1/baihelp")
    try:
        with pytest.raises(RuntimeError, match="docker compose up -d"):
            await ping(engine)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_engine_ping_failure_message() -> None:
    """(task-10 启动契约)对坏 DSN 的 engine 调 ping → RuntimeError 消息含「docker compose」。"""
    # 127.0.0.1:1 无服务监听,连接必被拒绝(离线确定性)
    engine = build_engine("mysql+aiomysql://baihelp:baihelp@127.0.0.1:1/baihelp")
    try:
        with pytest.raises(RuntimeError, match="docker compose"):
            await ping(engine)
    finally:
        await engine.dispose()


def test_session_factory_disables_expire_on_commit() -> None:
    """会话工厂必须 expire_on_commit=False(提交后属性仍可读,配合写穿门面)。"""
    engine = build_engine("sqlite+aiosqlite://")
    factory = build_session_factory(engine)
    assert factory.kw["expire_on_commit"] is False
