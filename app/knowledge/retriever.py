r"""知识检索器:query 向量语义检索 + MySQL 回表,query_faq 的新数据源。

流水线(embed → search → 阈值过滤 → 回表 → score 降序):
- embed(query) 得查询向量,repo.search 取 top_k 个 (chunk_id, score);
- score 即余弦相似度本身(高分更相似,证据见 milvus_repo.py 模块注),按
  「相似度阈值」消费:score < score_threshold 淘汰,恰好等于阈值保留;
- 幸存 id 回 MySQL knowledge_chunks,``WHERE id IN(...) AND
  vectorize_status='done'`` 取原文——pending 行与 Milvus 有向量但库里查无的
  id 天然出局;
- 结果按 score 降序排列;零命中或阈值下全滤空 → 空列表,不回表。
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import KnowledgeChunk


@dataclass
class RetrievedKnowledge:
    """单条检索命中:Milvus 的 (chunk_id, score) 加上 MySQL 回表的原文三格。"""

    chunk_id: int
    category: str
    questions: str
    answer: str
    score: float


class KnowledgeRetriever:
    """向量语义检索门面:调用方只见 retrieve(),嵌入/近邻/回表细节全部封在此。"""

    def __init__(self, embedder, repo, *, top_k: int = 3, score_threshold: float = 0.5) -> None:
        self._embedder = embedder
        self._repo = repo
        self._top_k = top_k
        self._score_threshold = score_threshold

    async def retrieve(
        self, query: str, session_factory: async_sessionmaker[AsyncSession]
    ) -> list[RetrievedKnowledge]:
        """检索与 query 语义最近的知识块:阈值过滤 → done 回表 → score 降序。"""
        [query_vector] = await self._embedder.embed([query])
        # 阈值过滤在回表前:低于阈值的连 MySQL 都不必查
        scores = {
            chunk_id: score
            for chunk_id, score in self._repo.search(query_vector, self._top_k)
            if score >= self._score_threshold
        }
        if not scores:
            return []

        async with session_factory() as session, session.begin():
            rows = (
                await session.scalars(
                    select(KnowledgeChunk).where(
                        KnowledgeChunk.id.in_(list(scores)),
                        KnowledgeChunk.vectorize_status == "done",
                    )
                )
            ).all()
            # 会话内立即取纯值组装结果,不让 ORM 对象逃出事务边界(MissingGreenlet 陷阱)
            return sorted(
                (
                    RetrievedKnowledge(
                        chunk_id=row.id,
                        category=row.category,
                        questions=row.questions,
                        answer=row.answer,
                        score=scores[row.id],
                    )
                    for row in rows
                ),
                key=lambda item: item.score,
                reverse=True,
            )
