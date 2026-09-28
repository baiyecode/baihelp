r"""KnowledgeRetriever 检索器测试:SQLite 内存库 + 假 embedder/repo(不触网、不碰真 Milvus)。

检索流水线锁定(embed → search → 阈值过滤 → MySQL 回表 → score 降序):

- 回表与排序:repo 返回 [2,1] 两 id、MySQL 两行 done → 按 score 降序返回,
  RetrievedKnowledge 五字段(chunk_id/category/questions/answer/score)齐全;
- 阈值语义:score 即余弦相似度(高分更相似,证据见 milvus_repo.py 模块注),
  按「相似度阈值」消费——score < score_threshold 淘汰,恰好等于阈值保留;
- 状态过滤:回表 SQL 带 vectorize_status='done',pending 行不返回;
- 零命中:repo.search 空 → 空列表,不回表。
"""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.engine import build_session_factory
from app.db.models import KnowledgeChunk
from app.knowledge.retriever import KnowledgeRetriever

# 假向量内容无关紧要(repo 是假的),恒定即可对上 repo 收到的查询向量
QUERY_VECTOR = [0.1, 0.2, 0.3]


class FakeEmbedder:
    """固定向量的假嵌入:记录收到的 texts,恒返回预设向量。"""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vector for _ in texts]


class FakeRepo:
    """假 Milvus 仓储:search 原样返回预设 (id, score) 列表并记录调用参数。"""

    def __init__(self, results: list[tuple[int, float]]) -> None:
        self.results = results
        self.calls: list[tuple[list[float], int]] = []

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[int, float]]:
        self.calls.append((list(query_vector), top_k))
        return list(self.results)


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


def _chunk(chunk_id: int, *, status: str = "done") -> KnowledgeChunk:
    """构造指定 id 的 chunk 行(显式主键,配合 repo 预设的命中 id)。"""
    return KnowledgeChunk(
        id=chunk_id,
        category=f"分类{chunk_id}",
        questions=f"问法{chunk_id}甲\n问法{chunk_id}乙",
        answer=f"答案{chunk_id}",
        vectorize_status=status,
    )


async def _seed(factory: async_sessionmaker[AsyncSession], *chunks: KnowledgeChunk) -> None:
    async with factory() as session, session.begin():
        for chunk in chunks:
            session.add(chunk)


# ---------------------------------------------------------------------------
# 检索流水线
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_backfills_from_mysql_and_orders_by_score(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """repo 命中 [2,1]、MySQL 两行 done → score 降序返回,五字段齐全,embed/search 接线正确。"""
    await _seed(session_factory, _chunk(1), _chunk(2))
    embedder = FakeEmbedder(QUERY_VECTOR)
    repo = FakeRepo([(2, 0.9), (1, 0.8)])  # Milvus 自身的返回序,与 score 降序一致
    retriever = KnowledgeRetriever(embedder, repo, top_k=3, score_threshold=0.5)

    results = await retriever.retrieve("退货政策", session_factory)

    assert [item.chunk_id for item in results] == [2, 1]  # score 降序
    assert [item.score for item in results] == [0.9, 0.8]
    first, second = results
    assert (first.category, first.questions, first.answer) == (
        "分类2",
        "问法2甲\n问法2乙",
        "答案2",
    )
    assert (second.category, second.questions, second.answer) == (
        "分类1",
        "问法1甲\n问法1乙",
        "答案1",
    )
    # 接线锁定:query 原样进 embed;查询向量与 top_k 原样进 repo.search
    assert embedder.calls == [["退货政策"]]
    assert repo.calls == [(QUERY_VECTOR, 3)]


@pytest.mark.asyncio
async def test_retrieve_filters_below_threshold(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """score 0.4 < 0.5 被滤(库里有 done 行也不返回);恰好 0.5 达标保留(≥ 语义)。"""
    await _seed(session_factory, _chunk(1))
    retriever = KnowledgeRetriever(FakeEmbedder(QUERY_VECTOR), FakeRepo([(1, 0.4)]))

    assert await retriever.retrieve("退货政策", session_factory) == []

    at_threshold = KnowledgeRetriever(
        FakeEmbedder(QUERY_VECTOR), FakeRepo([(1, 0.5)]), score_threshold=0.5
    )
    hits = await at_threshold.retrieve("退货政策", session_factory)
    assert [item.chunk_id for item in hits] == [1]


@pytest.mark.asyncio
async def test_retrieve_skips_pending_rows(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """一行 done 一行 pending:回表只认 done,pending 行不返回。"""
    await _seed(session_factory, _chunk(1), _chunk(2, status="pending"))
    retriever = KnowledgeRetriever(
        FakeEmbedder(QUERY_VECTOR), FakeRepo([(2, 0.9), (1, 0.8)])
    )

    results = await retriever.retrieve("退货政策", session_factory)

    assert [item.chunk_id for item in results] == [1]  # 只有 done 的 1 号


@pytest.mark.asyncio
async def test_retrieve_empty_when_no_hits(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """零命中:embed/search 照常走,回表短路,返回空列表。"""
    embedder = FakeEmbedder(QUERY_VECTOR)
    repo = FakeRepo([])
    retriever = KnowledgeRetriever(embedder, repo)

    results = await retriever.retrieve("退货政策", session_factory)

    assert results == []
    assert embedder.calls == [["退货政策"]]
    assert repo.calls == [(QUERY_VECTOR, 3)]  # top_k 走默认值 3
