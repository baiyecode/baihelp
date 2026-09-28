r"""ingest 建库编排测试:SQLite 内存库 + fakes(不触网、不落盘、不碰真 Milvus)。

- Phase A:自然键 (category, section_path, questions, answer,NULL 归一空串)
  幂等入库,pending 落库 + 文档内 prev/next 指针(只链新行,重跑不重写);
- Phase B:捞全表 pending(与语料无关)分批 embed → upsert(行只含
  {"id","vector"})→ vector_id 回填置 done,逐批事务做检查点,失败中止留
  pending,重跑自动补齐。

故障注入说明:run_ingest 是 Phase A+B 一体的公开入口,Phase A 的中间态
(pending)只能靠 Phase B 故障冻结后观察——phase_a 指针测试用 fail_on_call=1
让 Phase B 首次 upsert 即炸;续跑测试(Review Focus 1)用 fail_on_call=2 配合
vector_batch_size=1 造出「部分 done + 残留 pending」的中断现场:若首调即炸则
零进度,「上次 done 的不重 embed」将无从断言,故注入点取第二次调用。
"""

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.engine import build_session_factory
from app.db.models import KnowledgeChunk
from app.knowledge.chunker import build_vector_text
from app.knowledge.ingest import IngestStats, run_ingest


# ---------------------------------------------------------------------------
# fakes(brief 约定,测试与本任务共用):FakeEmbedder 确定性向量,FakeRepo 记录调用
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """确定性假嵌入:按文本 sha256 生成 dim 维向量;记录每次调用收到的 texts。"""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vector(text) for text in texts]

    def vector(self, text: str) -> list[float]:
        """单文本确定性向量:sha256 摘要字节循环取值归一到 [0, 1]。"""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self.dim)]


class FakeRepo:
    """假 Milvus 仓储:记录 upsert 收到的行;可配置第 N 次调用抛异常(故障注入)。"""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.upsert_calls: list[list[dict]] = []
        self.search_calls: list[list[float]] = []
        self.fail_on_call = fail_on_call

    def upsert_vectors(self, rows: list[dict]) -> None:
        self.upsert_calls.append([dict(row) for row in rows])
        if self.fail_on_call is not None and len(self.upsert_calls) == self.fail_on_call:
            raise RuntimeError(f"模拟 Milvus upsert 失败(第 {self.fail_on_call} 次调用)")

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[int, float]]:
        self.search_calls.append(list(query_vector))
        return []


# ---------------------------------------------------------------------------
# 语料与夹具:两 section 文档,每节恰产出一块
# ---------------------------------------------------------------------------

_CORPUS_MD = (
    "---\n"
    "category: 测试分类\n"
    "content_type: faq\n"
    "key_clauses:\n"
    "---\n"
    "### 节一\n"
    "- 问法: 问法甲 / 问法甲变体\n"
    "答案一。\n"
    "### 节二\n"
    "- 问法: 问法乙\n"
    "{answer_two}"
)


def _write_corpus(directory: Path, *, answer_two: str = "答案二。") -> Path:
    """写入两 section 语料文档(每节恰一块)并返回目录;answer_two 可注入修订。"""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "测试文档.md").write_text(
        _CORPUS_MD.format(answer_two=answer_two), encoding="utf-8"
    )
    return directory


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
# Phase A:pending 落库 + 指针;自然键幂等
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase_a_inserts_pending_with_pointers(
    session_factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    """两 section 文档 → 两行 pending,文档内 prev/next 相互指链,向量字段尚空。"""
    corpus = _write_corpus(tmp_path / "corpus")
    # 故障注入冻结 Phase A 产出状态:repo 首次 upsert 即炸 → Phase B 中止,
    # Phase A 已提交的 pending 行(含指针)得以被观察
    with pytest.raises(RuntimeError, match="模拟"):
        await run_ingest(
            session_factory, FakeEmbedder(), FakeRepo(fail_on_call=1), corpus_dir=corpus
        )

    async with session_factory() as session:
        rows = (
            (await session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id)))
        ).all()

    assert len(rows) == 2
    assert all(row.vectorize_status == "pending" for row in rows)
    assert all(row.vector_id is None for row in rows)
    first, second = rows
    assert first.prev_chunk_id is None
    assert first.next_chunk_id == second.id
    assert second.prev_chunk_id == first.id
    assert second.next_chunk_id is None


@pytest.mark.asyncio
async def test_phase_a_natural_key_idempotent(
    session_factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    """同语料两连跑:第二遍 inserted==0、skipped==首遍行数;改一处 answer → 只新插该块。"""
    corpus = _write_corpus(tmp_path / "corpus")
    embedder, repo = FakeEmbedder(), FakeRepo()

    first = await run_ingest(session_factory, embedder, repo, corpus_dir=corpus)
    assert (first.inserted, first.skipped_existing, first.chunks_in_db) == (2, 0, 2)
    assert first.vectorized == 2

    second = await run_ingest(session_factory, embedder, repo, corpus_dir=corpus)
    assert (second.inserted, second.skipped_existing, second.chunks_in_db) == (0, 2, 2)
    assert second.vectorized == 0  # 已全部 done,无 pending 可捞

    # 改一处 answer(节二)再跑:自然键变化 → 只新插该块,旧行原样保留不重写
    _write_corpus(corpus, answer_two="答案二(已修订)。")
    third = await run_ingest(session_factory, embedder, repo, corpus_dir=corpus)
    assert (third.inserted, third.skipped_existing, third.chunks_in_db) == (1, 1, 3)
    assert third.vectorized == 1  # 新块随跑被 Phase B 补齐

    async with session_factory() as session:
        rows = (
            (await session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id)))
        ).all()

    assert [row.answer for row in rows] == ["答案一。", "答案二。", "答案二(已修订)。"]
    assert all(row.vectorize_status == "done" for row in rows)
    # 指针回填幂等:首遍链原样保留(未重写),孤插的修订块不成链
    assert rows[0].next_chunk_id == rows[1].id
    assert rows[2].prev_chunk_id is None
    assert rows[2].next_chunk_id is None


# ---------------------------------------------------------------------------
# Phase B:vector_id 回填 + done;断点续跑
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase_b_backfills_vector_id_and_done(
    session_factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    """正常跑完:vector_id 回填、状态 done;FakeRepo 只收 {"id","vector"} 两键且 id
    对齐 chunk 主键;嵌入文本精确等于 build_vector_text 三格拼接。"""
    corpus = _write_corpus(tmp_path / "corpus")
    embedder, repo = FakeEmbedder(), FakeRepo()

    stats = await run_ingest(session_factory, embedder, repo, corpus_dir=corpus)
    assert isinstance(stats, IngestStats)
    assert stats.vectorized == 2

    async with session_factory() as session:
        rows = (
            (await session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id)))
        ).all()

    flat_rows = [row for call in repo.upsert_calls for row in call]
    assert [set(row) for row in flat_rows] == [{"id", "vector"}, {"id", "vector"}]
    assert [row["id"] for row in flat_rows] == [row.id for row in rows]

    texts = [text for call in embedder.calls for text in call]
    assert texts == [build_vector_text(row.category, row.questions, row.answer) for row in rows]
    # 字面钉格式:分类/问/答各占一行(半角冒号),questions 多行原样嵌入
    assert texts[0] == "分类:测试分类\n问:节一\n问法甲\n问法甲变体\n答:答案一。"
    assert [row["vector"] for row in flat_rows] == [embedder.vector(text) for text in texts]

    assert [row.vector_id for row in rows] == [str(row.id) for row in rows]
    assert all(row.vectorize_status == "done" for row in rows)


@pytest.mark.asyncio
async def test_phase_b_resumes_pending_after_failure(
    session_factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    """Review Focus 1:中断(部分 done + 残留 pending)→ 重跑只补 pending、最终全 done。"""
    corpus = _write_corpus(tmp_path / "corpus")
    with pytest.raises(RuntimeError, match="模拟"):
        await run_ingest(
            session_factory,
            FakeEmbedder(),
            FakeRepo(fail_on_call=2),
            corpus_dir=corpus,
            vector_batch_size=1,
        )

    async with session_factory() as session:
        rows = (
            (await session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id)))
        ).all()
    # 检查点语义:第 1 块已随批提交为 done,第 2 块失败中止留 pending
    assert [row.vectorize_status for row in rows] == ["done", "pending"]

    resume_embedder, resume_repo = FakeEmbedder(), FakeRepo()
    stats = await run_ingest(
        session_factory, resume_embedder, resume_repo, corpus_dir=corpus, vector_batch_size=1
    )

    # 上次 done 的不重 embed:只嵌入残留 pending 那一块的向量文本
    assert resume_embedder.calls == [
        [build_vector_text(rows[1].category, rows[1].questions, rows[1].answer)]
    ]
    assert [row["id"] for call in resume_repo.upsert_calls for row in call] == [rows[1].id]
    assert (stats.inserted, stats.vectorized, stats.chunks_in_db) == (0, 1, 2)

    async with session_factory() as session:
        rows_after = (
            (await session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id)))
        ).all()
    assert all(row.vectorize_status == "done" for row in rows_after)  # 最终全 done
    assert all(row.vector_id is not None for row in rows_after)


@pytest.mark.asyncio
async def test_mined_pending_picked_up(
    session_factory: async_sessionmaker[AsyncSession], tmp_path: Path
) -> None:
    """挖矿手工插入的 pending 行:空语料目录 → Phase A 零插入,Phase B 仍捞起置 done。"""
    async with session_factory() as session, session.begin():
        session.add(
            KnowledgeChunk(
                category="挖矿分类",
                questions="挖出来的问题",
                answer="挖出来的答案",
                content_type="faq",  # section_path 留 NULL(挖矿行无章节路径)
            )
        )

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    embedder, repo = FakeEmbedder(), FakeRepo()
    stats = await run_ingest(session_factory, embedder, repo, corpus_dir=empty_dir)

    assert (stats.inserted, stats.skipped_existing, stats.chunks_in_db) == (0, 0, 1)
    assert stats.vectorized == 1  # Phase B 捞全表 pending,与语料无关
    assert embedder.calls == [["分类:挖矿分类\n问:挖出来的问题\n答:挖出来的答案"]]

    async with session_factory() as session:
        row = ((await session.scalars(select(KnowledgeChunk)))).one()
    assert row.vectorize_status == "done"
    assert row.vector_id == str(row.id)
