r"""建库编排:语料切块幂等入库(Phase A)→ pending 批量向量化双写(Phase B)。

双写状态机(MySQL knowledge_chunks 为原文权威源,Milvus 集合 knowledge 存向量):
- Phase A:load_corpus → chunk_document,按自然键 (category, section_path,
  questions, answer,NULL 归一空串) 去重后以 pending 落库,并在同事务第二遍
  回填文档内 prev/next 指针——只链本次新插行,绝不重写既有行(重跑幂等);
- Phase B:捞全表 pending(与语料无关,挖矿插入的块同样被捞起),按
  vector_batch_size 分批 embed → upsert(行只含 {"id","vector"},主键即 chunk
  id,按主键幂等)→ 回填 vector_id 置 done;逐批事务提交做检查点,任一批失败
  即中止抛出:已成功批次保持 done,失败及后续批次留 pending,重跑自动补齐。

CLI 见文件尾 __main__(组合根:建引擎/真组件/ensure_collection 都在那边)。
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import KnowledgeChunk
from app.knowledge.chunker import build_vector_text, chunk_document
from app.knowledge.corpus import load_corpus


class _Embedder(Protocol):
    """嵌入客户端结构约定(生产 BgeM3Embedder / 测试 FakeEmbedder 共同满足)。"""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class _VectorRepo(Protocol):
    """向量仓储结构约定(生产 MilvusKnowledgeRepo / 测试 FakeRepo 共同满足)。"""

    def upsert_vectors(self, rows: list[dict]) -> None: ...


@dataclass
class IngestStats:
    """建库统计:chunks_in_db 为表内总行数(Phase A 后,Phase B 不增行);
    inserted / skipped_existing 属 Phase A;vectorized 属 Phase B(本轮
    pending → done 的行数)。"""

    chunks_in_db: int
    inserted: int
    skipped_existing: int
    vectorized: int


def _natural_key(
    category: str, section_path: str | None, questions: str, answer: str
) -> tuple[str, str, str, str]:
    """Phase A 自然键:四元组,NULL(目前仅 section_path 可空)归一空串。"""
    return (category, section_path or "", questions, answer)


async def run_ingest(
    session_factory: async_sessionmaker[AsyncSession],
    embedder: _Embedder,
    repo: _VectorRepo,
    *,
    corpus_dir: Path,
    max_chars: int = 500,
    overlap_chars: int = 80,
    vector_batch_size: int = 64,
) -> IngestStats:
    """跑完整建库:Phase A 语料切块幂等入库 → Phase B pending 批量向量化。

    embedder 需实现 ``async embed(texts) -> list[list[float]]``(与 texts 顺序
    对齐);repo 需实现 ``upsert_vectors(rows)``(行只含 {"id","vector"} 两键);
    集合创建(ensure_collection)与 repo.close() 是调用方(组合根)的职责。
    任一批向量化失败即原样抛出(失败即中止留 pending),不吞异常不重试。
    """
    # ---- Phase A:语料切块,自然键幂等入库(空目录 → 零插入) ----
    docs = load_corpus(corpus_dir)
    doc_chunk_lists = [
        chunk_document(doc, max_chars=max_chars, overlap_chars=overlap_chars)
        for doc in docs
    ]

    inserted = 0
    skipped_existing = 0
    new_groups: list[list[KnowledgeChunk]] = []  # 各文档的新插行(保序),供指针回填
    async with session_factory() as session, session.begin():
        existing = await session.execute(
            select(
                KnowledgeChunk.category,
                KnowledgeChunk.section_path,
                KnowledgeChunk.questions,
                KnowledgeChunk.answer,
            )
        )
        known_keys = {_natural_key(*row) for row in existing}
        for chunks in doc_chunk_lists:
            group: list[KnowledgeChunk] = []
            for chunk in chunks:
                key = _natural_key(
                    chunk.category, chunk.section_path, chunk.questions, chunk.answer
                )
                if key in known_keys:
                    skipped_existing += 1
                    continue
                row = KnowledgeChunk(
                    category=chunk.category,
                    questions=chunk.questions,
                    answer=chunk.answer,
                    section_path=chunk.section_path,
                    content_type=chunk.content_type,
                    is_key_clause=chunk.is_key_clause,
                    vectorize_status="pending",
                )
                session.add(row)
                group.append(row)
                known_keys.add(key)  # 同批内重复自然键也只入一次
                inserted += 1
            if group:
                new_groups.append(group)
        await session.flush()  # 拿自增 id
        # 同事务第二遍:指针回填,只链各文档新插行的相邻关系(重跑不重写既有行)
        for group in new_groups:
            for prev_row, next_row in zip(group, group[1:], strict=False):
                prev_row.next_chunk_id = next_row.id
                next_row.prev_chunk_id = prev_row.id
        chunks_in_db = await session.scalar(
            select(func.count()).select_from(KnowledgeChunk)
        )

    # ---- Phase B:捞全表 pending,分批 embed → upsert → 置 done(逐批检查点) ----
    vectorized = 0
    while True:
        async with session_factory() as session, session.begin():
            rows = (
                await session.scalars(
                    select(KnowledgeChunk)
                    .where(KnowledgeChunk.vectorize_status == "pending")
                    .order_by(KnowledgeChunk.id)
                    .limit(vector_batch_size)
                )
            ).all()
            if not rows:
                break
            texts = [
                build_vector_text(row.category, row.questions, row.answer) for row in rows
            ]
            vectors = await embedder.embed(texts)
            if len(vectors) != len(rows):  # 状态机护栏:错位即炸,防带缺向量置 done
                raise RuntimeError(
                    f"嵌入返回数 {len(vectors)} 与输入文本数 {len(rows)} 不一致"
                )
            repo.upsert_vectors(
                [{"id": row.id, "vector": vector} for row, vector in zip(rows, vectors, strict=True)]
            )
            for row, _ in zip(rows, vectors, strict=True):
                row.vector_id = str(row.id)
                row.vectorize_status = "done"
            vectorized += len(rows)
            # 事务随 with 正常退出即提交 = 检查点;异常则本批回滚留 pending

    return IngestStats(
        chunks_in_db=chunks_in_db,
        inserted=inserted,
        skipped_existing=skipped_existing,
        vectorized=vectorized,
    )


if __name__ == "__main__":
    import argparse

    def _parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog="python -m app.knowledge.ingest",
            description="知识库建库:语料切块幂等入库(MySQL)+ pending 批量向量化双写(Milvus)",
        )
        parser.add_argument(
            "--corpus-dir",
            type=Path,
            default=Path("data/knowledge"),
            help="语料目录(含 front-matter 的 .md 文件)",
        )
        return parser.parse_args()

    args = _parse_args()  # --help 在此退出,不触达下方重依赖导入

    async def _main() -> None:
        """读配置建引擎与真组件 → run_ingest → 打印统计;finally dispose+close。

        pymilvus 导入会对仓库根 .env 执行 load_dotenv(实测见 task-5-report),
        故生产组件延迟到确需运行时才导入;ensure_collection 幂等建集合。
        """
        from app.core.config import get_settings
        from app.db.engine import build_engine, build_session_factory
        from app.knowledge.embedding import BgeM3Embedder
        from app.knowledge.milvus_repo import MilvusKnowledgeRepo

        settings = get_settings()
        engine = build_engine(settings.database_url)
        embedder = BgeM3Embedder(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
        repo = MilvusKnowledgeRepo(settings.milvus_db_path, dim=settings.embedding_dim)
        try:
            repo.ensure_collection()
            stats = await run_ingest(
                build_session_factory(engine),
                embedder,
                repo,
                corpus_dir=args.corpus_dir,
                max_chars=settings.chunk_max_chars,
                overlap_chars=settings.chunk_overlap_chars,
            )
        finally:
            await engine.dispose()
            repo.close()
        print("ingest 完成:")
        print(f"  chunks_in_db     = {stats.chunks_in_db}")
        print(f"  inserted         = {stats.inserted}")
        print(f"  skipped_existing = {stats.skipped_existing}")
        print(f"  vectorized       = {stats.vectorized}")

    asyncio.run(_main())
