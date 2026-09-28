r"""app.tools.knowledge / app.tools.tickets 两个 DB 工具的行为测试(SQLite 内存库)。

验证(query_faq / create_ticket):

- query_faq 数据源已从 faq 表 LIKE 换成注入的语义检索器(FakeRetriever 模拟
  KnowledgeRetriever):返回「问:…\n答:…」拼接,questions 多行问法压成单行
  (以「 / 」相连),多条命中以空行相连,空结果回兜底话术;
  args_schema 键集仍只有 keyword——retriever 与 session_factory 同为
  InjectedToolArg 注入参数,模型不可见;
- create_ticket.ainvoke 注入 conversation_id/session_factory,返回含工单号,
  库里该行 status='待处理';
- Schema 排除断言:create_ticket.args_schema.model_fields 键集
  == {description, ticket_type}——conversation_id/session_factory 为
  InjectedToolArg 注入参数,不入模型可见的入参 schema。
"""

import re
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db import models
from app.db.base import Base
from app.db.engine import build_session_factory
from app.knowledge.retriever import RetrievedKnowledge
from app.tools.knowledge import QueryFaqInput, query_faq
from app.tools.tickets import create_ticket


class FakeRetriever:
    """query_faq 注入用假检索器:retrieve 与 KnowledgeRetriever.retrieve 同形。

    契约:``async retrieve(query, session_factory) -> list[RetrievedKnowledge]``,
    原样返回预置结果并记录 (query, session_factory);本文件与
    test_chat_endpoint / test_tool_registry 的注入改造共用,T10 评估亦复用。
    """

    def __init__(self, results: list[RetrievedKnowledge]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, Any]] = []

    async def retrieve(
        self, query: str, session_factory: Any
    ) -> list[RetrievedKnowledge]:
        self.calls.append((query, session_factory))
        return list(self.results)


@pytest_asyncio.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """SQLite 内存库(StaticPool 共享同一连接)+ create_all,交出会话工厂。

    query_faq 已不读 faq 表(语义检索走 knowledge_chunks 回表,由 FakeRetriever
    模拟),本夹具仅供 create_ticket 落表;两种工具共用一个空库即可。
    """
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield build_session_factory(engine)
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# query_faq
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_faq_hits_seeded_faq(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """语义命中(FakeRetriever 注入):返回文本含问句与答案文本,keyword 原样进检索器。"""
    retriever = FakeRetriever(
        [
            RetrievedKnowledge(
                chunk_id=1,
                category="售后",
                questions="退货政策是什么?",
                answer="签收后7天内支持无理由退货,需保持商品完好。",
                score=0.9,
            )
        ]
    )

    result = await query_faq.ainvoke(
        {"keyword": "退货", "session_factory": session_factory, "retriever": retriever}
    )

    assert "退货政策是什么?" in result  # 问:…
    assert "签收后7天内支持无理由退货,需保持商品完好。" in result  # 答:…
    assert retriever.calls == [("退货", session_factory)]


@pytest.mark.asyncio
async def test_query_faq_no_match_reports_not_found(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """检索器空结果:返回文本含「未找到」。"""
    result = await query_faq.ainvoke(
        {"keyword": "邮费", "session_factory": session_factory, "retriever": FakeRetriever([])}
    )

    assert "未找到" in result


def test_query_faq_args_schema_unchanged() -> None:
    """契约锁定:入参 schema 键集仍只有 keyword(retriever 为注入参数,不入 schema)。"""
    assert set(QueryFaqInput.model_fields) == {"keyword"}


@pytest.mark.asyncio
async def test_query_faq_output_format_locked(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """出参格式锁定:questions 多行问法压单行(「 / 」相连)保「问:…\n答:…」结构;多条以空行相连。"""
    retriever = FakeRetriever(
        [
            RetrievedKnowledge(
                chunk_id=7,
                category="售后",
                questions="退货政策是什么?\n怎么申请退货?\n退货要运费吗?",
                answer="签收后7天内支持无理由退货,需保持商品完好。",
                score=0.9,
            )
        ]
    )

    result = await query_faq.ainvoke(
        {"keyword": "退货", "session_factory": session_factory, "retriever": retriever}
    )

    assert result == (
        "问:退货政策是什么? / 怎么申请退货? / 退货要运费吗?\n"
        "答:签收后7天内支持无理由退货,需保持商品完好。"
    )

    multi = FakeRetriever(
        [
            RetrievedKnowledge(
                chunk_id=1, category="售后", questions="问法甲", answer="答案一", score=0.9
            ),
            RetrievedKnowledge(
                chunk_id=2, category="物流", questions="问法乙", answer="答案二", score=0.8
            ),
        ]
    )
    multi_result = await query_faq.ainvoke(
        {"keyword": "退货", "session_factory": session_factory, "retriever": multi}
    )
    assert multi_result == "问:问法甲\n答:答案一\n\n问:问法乙\n答:答案二"


@pytest.mark.asyncio
async def test_query_faq_fallback_text_verbatim(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """空结果兜底话术逐字保留。"""
    result = await query_faq.ainvoke(
        {"keyword": "邮费", "session_factory": session_factory, "retriever": FakeRetriever([])}
    )

    assert result == "未找到与「邮费」相关的 FAQ,请换个关键词或转人工客服。"


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
