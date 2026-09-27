r"""app.services.persistence 写穿门面的行为测试(SQLite 内存库跑 create_all)。

ChatPersistence 每次调用独立 session、写完即提交——本文件在调用之外另开会话
查库复核,验证(task-9-brief Step 1):

- ensure_conversation 幂等:同 user_id 两次调用返回同一会话 id,库里只有一行;
- 四个 log 方法各落一行且字段对:user 行正文、assistant 纯工具调用行
  content 为空 + tool_calls JSON 往返保真、tool 行带 tool_call_id、
  assistant 正文行;
- 会话工厂以公开属性 session_factory 暴露(R2 裁决:编排层构造 ToolContext 用)。
"""

from collections.abc import AsyncIterator
from json import dumps, loads

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.engine import build_session_factory
from app.db.models import Conversation, Message
from app.services.persistence import ChatPersistence


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


@pytest.mark.asyncio
async def test_ensure_conversation_is_idempotent(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """同 user_id 两次建档返回同一会话 id;库里只有一行,公开属性暴露工厂。"""
    persistence = ChatPersistence(session_factory)
    assert persistence.session_factory is session_factory  # R2:编排层构造 ToolContext 用

    first = await persistence.ensure_conversation("u-idem")
    second = await persistence.ensure_conversation("u-idem")

    assert first == second

    async with session_factory() as session:
        conversations = (
            (await session.scalars(select(Conversation))).all()
        )

    assert len(conversations) == 1
    assert conversations[0].id == first
    assert conversations[0].user_id == "u-idem"


@pytest.mark.asyncio
async def test_log_methods_persist_one_row_each(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """四个 log 方法各落一行:字段逐一对上,全部挂在同一会话下。"""
    persistence = ChatPersistence(session_factory)
    conversation_id = await persistence.ensure_conversation("u-log")
    tool_calls = [
        {"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1", "type": "tool_call"}
    ]

    await persistence.log_user_message(conversation_id, "退货政策是什么?")
    await persistence.log_assistant_tool_call(conversation_id, None, tool_calls)
    await persistence.log_tool_result(conversation_id, "call_1", "七天无理由退货")
    await persistence.log_assistant_message(conversation_id, "亲,支持七天无理由退货哦")

    # 独立会话查库复核:每次调用都是已提交的短事务,跨会话可见
    async with session_factory() as session:
        rows = (await session.scalars(select(Message).order_by(Message.id))).all()

    assert [row.role for row in rows] == ["user", "assistant", "tool", "assistant"]
    assert all(row.conversation_id == conversation_id for row in rows)
    assert rows[0].content == "退货政策是什么?"
    assert rows[0].tool_calls is None
    # assistant 纯工具调用行:正文为空,申请单 JSON 往返保真
    assert rows[1].content is None
    assert loads(dumps(rows[1].tool_calls)) == tool_calls
    # tool 行:对号入座的 tool_call_id + 结果正文
    assert rows[2].tool_call_id == "call_1"
    assert rows[2].content == "七天无理由退货"
    assert rows[2].tool_calls is None
    # assistant 最终正文行
    assert rows[3].content == "亲,支持七天无理由退货哦"
    assert rows[3].tool_call_id is None
