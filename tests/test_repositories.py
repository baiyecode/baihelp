"""app.db.repositories 的行为测试(SQLite 内存库跑 create_all)。

仓储是瘦函数:收 AsyncSession、函数内只 flush 不 commit——本文件以
``async with session.begin()`` 充当调用方提交事务,验证:

- get_or_create_conversation:同 user_id 复用同一行(多行时取最近一条),无则建;
- add_message:tool 行(role/tool_call_id/content)与 assistant 行 JSON tool_calls 往返保真;
- search_faq:LIKE 元字符(% / _ / \\)一律转义按字面匹配,空 keyword 短路空列表;
- create_ticket:当日序号 001/002 递增,主键冲突用 savepoint 回滚换号重试。
"""

from collections.abc import AsyncIterator
from datetime import datetime
from json import dumps, loads

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db import models
from app.db.base import Base
from app.db.engine import build_session_factory
from app.db.repositories import (
    add_message,
    create_ticket,
    get_or_create_conversation,
    search_faq,
)


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """SQLite 内存库(StaticPool 共享同一连接)+ create_all,交出单个会话。"""
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with build_session_factory(engine)() as s:
            yield s
    finally:
        await engine.dispose()


async def _seed_faq(session: AsyncSession, question: str) -> models.Faq:
    """预置一条 FAQ,question 由用例指定。"""
    faq = models.Faq(question=question, answer="略", category="测试")
    session.add(faq)
    await session.flush()
    return faq


# ---------------------------------------------------------------------------
# get_or_create_conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_or_create_conversation_reuses_row_for_same_user(session: AsyncSession) -> None:
    """同 user_id 两次调用返回同一行;首建走 status 默认「进行中」。"""
    async with session.begin():
        first = await get_or_create_conversation(session, "u-001")
        second = await get_or_create_conversation(session, "u-001")

        assert first.id is not None
        assert second.id == first.id
        assert first.status == "进行中"


@pytest.mark.asyncio
async def test_get_or_create_conversation_creates_one_row_per_user(session: AsyncSession) -> None:
    """不同 user_id 各建一行,互不串号。"""
    async with session.begin():
        conv_a = await get_or_create_conversation(session, "u-001")
        conv_b = await get_or_create_conversation(session, "u-002")

        assert conv_a.id != conv_b.id
        assert conv_a.user_id == "u-001"
        assert conv_b.user_id == "u-002"


@pytest.mark.asyncio
async def test_get_or_create_conversation_picks_latest_row_for_existing_user(
    session: AsyncSession,
) -> None:
    """同 user_id 已有多行(并发遗留)时取最近一条(id 最大),不另建。"""
    async with session.begin():
        older = models.Conversation(user_id="u-dup")
        session.add(older)
        await session.flush()
        newer = models.Conversation(user_id="u-dup")
        session.add(newer)
        await session.flush()

        picked = await get_or_create_conversation(session, "u-dup")

        assert picked.id == newer.id


# ---------------------------------------------------------------------------
# add_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_message_persists_tool_row(session: AsyncSession) -> None:
    """tool 行:role='tool' + tool_call_id + 正文,落库往返保真。"""
    async with session.begin():
        conversation = await get_or_create_conversation(session, "u-msg")
        message = await add_message(
            session,
            conversation.id,
            role="tool",
            content="退货政策:七天无理由",
            tool_call_id="call_x",
        )

        assert message.id is not None
        await session.refresh(message)  # 发 SELECT 从库里重读,验证真实落库

    assert message.conversation_id == conversation.id
    assert message.role == "tool"
    assert message.content == "退货政策:七天无理由"
    assert message.tool_call_id == "call_x"
    assert message.tool_calls is None


@pytest.mark.asyncio
async def test_add_message_assistant_tool_calls_json_roundtrip(session: AsyncSession) -> None:
    """assistant 行:tool_calls JSON 往返相等;纯工具调用时 content 为空。"""
    tool_calls = [{"name": "query_faq", "args": {}, "id": "call_x"}]
    async with session.begin():
        conversation = await get_or_create_conversation(session, "u-msg")
        message = await add_message(
            session,
            conversation.id,
            role="assistant",
            tool_calls=tool_calls,
        )

        assert message.id is not None
        await session.refresh(message)  # 发 SELECT 从库里重读,验证真实落库

    assert message.content is None  # assistant 纯工具调用时正文可空
    assert message.tool_call_id is None
    assert loads(dumps(message.tool_calls)) == tool_calls  # JSON 往返保真


# ---------------------------------------------------------------------------
# search_faq
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_faq_hits_by_keyword(session: AsyncSession) -> None:
    """question 含 keyword 即命中,不含的不串进来。"""
    async with session.begin():
        hit = await _seed_faq(session, "退货政策是什么?")
        await _seed_faq(session, "如何开发票?")

        results = await search_faq(session, "退货")

    assert [faq.id for faq in results] == [hit.id]
    assert results[0].question == "退货政策是什么?"


@pytest.mark.asyncio
async def test_search_faq_no_match_returns_empty_list(session: AsyncSession) -> None:
    """keyword 无命中返回空列表。"""
    async with session.begin():
        await _seed_faq(session, "退货政策是什么?")

        results = await search_faq(session, "邮费")

    assert results == []


@pytest.mark.asyncio
async def test_search_faq_like_metacharacters_match_literally(session: AsyncSession) -> None:
    """keyword 含 % / _ / \\ 时按字面匹配,不当通配符(评审焦点 #2)。

    每个元字符配一条「差一个字符」的诱饵行:若转义失效,LIKE 会把它误命中。
    """
    async with session.begin():
        underscore = await _seed_faq(session, "满100减5_专享")
        underscore_decoy = await _seed_faq(session, "满100减50专享")  # _ 当通配符会误命中
        percent = await _seed_faq(session, "晒单返现5%")
        percent_decoy = await _seed_faq(session, "晒单返现50")  # % 当通配符会误命中
        backslash = await _seed_faq(session, "资料在 D:\\手册 目录")
        backslash_decoy = await _seed_faq(session, "资料在 D:X手册 目录")  # \ 当转义符会串位

        by_underscore = await search_faq(session, "减5_专")
        by_percent = await search_faq(session, "返现5%")
        by_backslash = await search_faq(session, "D:\\手册")

    assert [faq.id for faq in by_underscore] == [underscore.id]
    assert underscore_decoy.id not in [faq.id for faq in by_underscore]
    assert [faq.id for faq in by_percent] == [percent.id]
    assert percent_decoy.id not in [faq.id for faq in by_percent]
    assert [faq.id for faq in by_backslash] == [backslash.id]
    assert backslash_decoy.id not in [faq.id for faq in by_backslash]


@pytest.mark.asyncio
async def test_search_faq_empty_keyword_returns_empty_list(session: AsyncSession) -> None:
    """空 keyword 短路返回空列表,不发 LIKE 查询。"""
    async with session.begin():
        await _seed_faq(session, "退货政策是什么?")

        results = await search_faq(session, "")

    assert results == []


# ---------------------------------------------------------------------------
# create_ticket
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_ticket_assigns_sequential_daily_numbers(session: AsyncSession) -> None:
    """同日两张工单:序号 001、002,格式 T + 当日YYYYMMDD + 三位序号。"""
    date_prefix = "T" + datetime.now().strftime("%Y%m%d")
    async with session.begin():
        conversation = await get_or_create_conversation(session, "u-ticket")
        ticket_a = await create_ticket(session, conversation.id, "商品破损,要求退款", "售后")
        ticket_b = await create_ticket(session, conversation.id, "物流三天未更新", "投诉")

    assert ticket_a.ticket_no == f"{date_prefix}001"
    assert ticket_b.ticket_no == f"{date_prefix}002"
    assert ticket_a.status == "待处理"  # 默认状态
    assert ticket_a.ticket_type == "售后"
    assert ticket_b.ticket_type == "投诉"
    assert ticket_a.conversation_id == conversation.id


@pytest.mark.asyncio
async def test_create_ticket_collision_retry(session: AsyncSession) -> None:
    """主键冲突重试:预置当日 003/004 两个已占工单号,使计数 +1/+2 的前两个
    候选号连续撞主键,第三次重试换到空闲号落库;attempts 语义不外泄,
    照常返回 Ticket 且提交后真正可查。
    """
    date_prefix = "T" + datetime.now().strftime("%Y%m%d")
    async with session.begin():
        conversation = await get_or_create_conversation(session, "u-collision")
        # 当日已存在 003、004:计数为 2 → 候选号依次 003(撞)、004(撞)、005(空闲)
        session.add_all(
            [
                models.Ticket(
                    ticket_no=f"{date_prefix}003",
                    conversation_id=conversation.id,
                    description="历史工单A",
                    ticket_type="售后",
                ),
                models.Ticket(
                    ticket_no=f"{date_prefix}004",
                    conversation_id=conversation.id,
                    description="历史工单B",
                    ticket_type="投诉",
                ),
            ]
        )
        await session.flush()

        ticket = await create_ticket(session, conversation.id, "申请价保退款", "售后")

        # 重试细节(attempts)不外泄:返回值就是普通 Ticket,字段齐整
        assert isinstance(ticket, models.Ticket)
        assert ticket.ticket_no == f"{date_prefix}005"
        assert ticket.conversation_id == conversation.id
        assert ticket.status == "待处理"

        session.expire_all()  # 提交前先驱逐缓存,提交后重读验真实落库
        persisted = await session.get(models.Ticket, f"{date_prefix}005")

    assert persisted is not None
    assert persisted.description == "申请价保退款"
