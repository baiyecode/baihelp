"""app.tools.knowledge / app.tools.tickets 两个 DB 工具的行为测试(SQLite 内存库)。

验证(query_faq / create_ticket):

- SQLite 工厂预置一条 FAQ 后,query_faq.ainvoke 命中含答案文本(问:/答: 拼接);
- keyword 无命中时返回文本含「未找到」;
- create_ticket.ainvoke 注入 conversation_id/session_factory,返回含工单号,
  库里该行 status='待处理';
- Schema 排除断言:create_ticket.args_schema.model_fields 键集
  == {description, ticket_type}——conversation_id/session_factory 为
  InjectedToolArg 注入参数,不入模型可见的入参 schema。
"""

import re
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db import models
from app.db.base import Base
from app.db.engine import build_session_factory
from app.tools.knowledge import query_faq
from app.tools.tickets import create_ticket


@pytest_asyncio.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """SQLite 内存库(StaticPool 共享同一连接)+ create_all,预置一条 FAQ,交出会话工厂。"""
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = build_session_factory(engine)
        async with factory() as s, s.begin():
            s.add(
                models.Faq(
                    question="退货政策是什么?",
                    answer="签收后7天内支持无理由退货,需保持商品完好。",
                    category="售后",
                )
            )
        yield factory
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# query_faq
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_faq_hits_seeded_faq(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """keyword「退货」命中预置 FAQ:返回文本含问句与答案文本。"""
    result = await query_faq.ainvoke(
        {"keyword": "退货", "session_factory": session_factory}
    )

    assert "退货政策是什么?" in result  # 问:…
    assert "签收后7天内支持无理由退货,需保持商品完好。" in result  # 答:…


@pytest.mark.asyncio
async def test_query_faq_no_match_reports_not_found(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """keyword「邮费」无命中:返回文本含「未找到」。"""
    result = await query_faq.ainvoke(
        {"keyword": "邮费", "session_factory": session_factory}
    )

    assert "未找到" in result


# ---------------------------------------------------------------------------
# create_ticket
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_ticket_persists_pending_row(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """建工单:返回文本含工单号与「待处理」,库里真实落一行待处理工单。"""
    # 先建一通会话拿 conversation_id(tickets.conversation_id 外键指向真实行)
    async with session_factory() as s, s.begin():
        conversation = models.Conversation(user_id="u-tool")
        s.add(conversation)
        await s.flush()
        conversation_id = conversation.id

    result = await create_ticket.ainvoke(
        {
            "description": "商品破损,要求退款",
            "ticket_type": "售后",
            "conversation_id": conversation_id,
            "session_factory": session_factory,
        }
    )

    assert "状态:待处理" in result
    # 工单号格式 T + 8 位日期 + 3 位序号
    match = re.search(r"T\d{11}", result)
    assert match is not None, result
    ticket_no = match.group(0)

    async with session_factory() as s:
        row = await s.get(models.Ticket, ticket_no)

    assert row is not None
    assert row.status == "待处理"
    assert row.conversation_id == conversation_id
    assert row.ticket_type == "售后"
    assert row.description == "商品破损,要求退款"


# ---------------------------------------------------------------------------
# args_schema 排除断言(InjectedToolArg 不入模型可见 schema)
# ---------------------------------------------------------------------------


def test_create_ticket_args_schema_excludes_injected_args() -> None:
    """conversation_id / session_factory 为注入参数,不得出现在 args_schema。"""
    assert set(create_ticket.args_schema.model_fields.keys()) == {
        "description",
        "ticket_type",
    }
