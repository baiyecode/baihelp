"""app.db.seed 的幂等灌数据测试(SQLite 内存库跑 create_all)。

seed 是数据类产出:同一 SQLite 会话工厂上连跑两次,验证——

- 各表行数恒定:faq 8、演示会话 1(user_id="seed-demo")、其消息 3、tickets 1,
  返回值 dict[str, int] 与库里实况一致;
- 演示会话消息 role user/assistant/tool 各一,tool 行带 tool_call_id;
- 8 条 faq 的 question 一律不含「邮费」「运费」子串——用户问「邮费是多少」时
  query_faq 必须 LIKE 落空,这是漏召回验收(acceptance ⑥)的前提;
- question 含「退货政策」的行存在且 answer 含「七天」(acceptance ⑤ 判据词)。
"""

from collections.abc import AsyncIterator
from datetime import datetime

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.engine import build_session_factory
from app.db.models import Conversation, Faq, Message, Ticket
from app.db.seed import seed


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


# ---------------------------------------------------------------------------
# 幂等:两连跑各表行数恒定
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_twice_is_idempotent(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """同一工厂连跑两次:返回 dict[str, int] 且各表行数恒定,库里实况一致。"""
    first = await seed(session_factory)
    second = await seed(session_factory)

    expected = {"conversations": 1, "messages": 3, "faq": 8, "tickets": 1}
    assert first == expected
    assert second == expected

    # 不信返回值,直接查库复核
    async with session_factory() as session:
        faq_count = await session.scalar(select(func.count()).select_from(Faq))
        conversation_count = await session.scalar(
            select(func.count()).select_from(Conversation)
        )
        message_count = await session.scalar(select(func.count()).select_from(Message))
        ticket_count = await session.scalar(select(func.count()).select_from(Ticket))

        # 演示工单固定号 T{当日YYYYMMDD}901,给真实工单(create_ticket 当日 001 起)留低位号段
        tickets = (await session.scalars(select(Ticket))).all()

    assert (faq_count, conversation_count, message_count, ticket_count) == (8, 1, 3, 1)
    assert [ticket.ticket_no for ticket in tickets] == [
        "T" + datetime.now().strftime("%Y%m%d") + "901"
    ]


@pytest.mark.asyncio
async def test_seed_demo_conversation_messages_shape(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """演示会话:user_id="seed-demo" 恰 1 通,其消息恒 3 条且 role 各一,
    tool 行带 tool_call_id、assistant 行带工具调用申请单。"""
    await seed(session_factory)
    await seed(session_factory)  # 第二遍必须原样跳过,不产生重复会话/消息

    async with session_factory() as session:
        conversations = (
            (
                await session.scalars(
                    select(Conversation).where(Conversation.user_id == "seed-demo")
                )
            )
            .unique()
            .all()
        )
        assert len(conversations) == 1
        conversation = conversations[0]

        messages = (
            (
                await session.scalars(
                    select(Message)
                    .where(Message.conversation_id == conversation.id)
                    .order_by(Message.id)
                )
            )
            .unique()
            .all()
        )

        assert len(messages) == 3
        assert [message.role for message in messages] == ["user", "assistant", "tool"]
        assistant_message, tool_message = messages[1], messages[2]
        assert assistant_message.tool_calls  # assistant 行带工具调用申请单
        assert tool_message.tool_call_id  # tool 行带 tool_call_id,与申请单对号
        assert tool_message.tool_call_id in str(assistant_message.tool_calls)


# ---------------------------------------------------------------------------
# FAQ 内容验收前提:邮费/运费词条缺席 + 七天判据词在场
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_faq_questions_exclude_postage_words(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """8 条 faq 的 question 一律不含「邮费」「运费」子串(漏召回验收的前提);
    运费承担方只允许出现在 answer 列。"""
    await seed(session_factory)

    async with session_factory() as session:
        faqs = (await session.scalars(select(Faq))).all()

    assert len(faqs) == 8
    for faq in faqs:
        assert "邮费" not in faq.question, f"question 不得含「邮费」:{faq.question}"
        assert "运费" not in faq.question, f"question 不得含「运费」:{faq.question}"


@pytest.mark.asyncio
async def test_seed_faq_return_policy_answer_contains_seven_days(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """question 含「退货政策」的行存在,answer 含「七天」(acceptance ⑤ 判据词)。"""
    await seed(session_factory)

    async with session_factory() as session:
        faqs = (
            (
                await session.scalars(
                    select(Faq).where(Faq.question.like("%退货政策%"))
                )
            )
            .unique()
            .all()
        )

    assert faqs, "必须存在 question 含「退货政策」的行"
    assert any("七天" in faq.answer for faq in faqs)
