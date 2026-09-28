"""MilvusKnowledgeRepo 集成测试:进程内 milvus-lite 真库(tmp_path),无网络无外部服务。

含 Review Focus 4 的 COSINE 换算校准:query 与 A 全同、与 B 正交,
断言 A.score ≈ 1.0 且 A.score > B.score。
"""

import os
from pathlib import Path

import pytest

_ENV_BEFORE_PYMILVUS = dict(os.environ)

from app.knowledge.milvus_repo import COLLECTION_NAME, MilvusKnowledgeRepo  # noqa: E402

# pymilvus 导入时会对仓库根 .env 执行 load_dotenv(实测证据见 task-5-report.md),
# 把 DATABASE_URL/LLM_API_KEY 等注入 os.environ,污染 test_config 的
# Settings(_env_file=None) 默认值断言;导入完成立即还原导入前环境。
os.environ.clear()
os.environ.update(_ENV_BEFORE_PYMILVUS)

DIM = 3
VEC_A = [1.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0]


def _make_repo(db_path: Path) -> MilvusKnowledgeRepo:
    repo = MilvusKnowledgeRepo(db_path=db_path, dim=DIM)
    repo.ensure_collection()
    return repo


def test_ensure_collection_idempotent(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "milvus.db")
    try:
        repo.ensure_collection()  # 连调两次不炸
    finally:
        repo.close()


def test_upsert_same_id_no_duplicate(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "milvus.db")
    try:
        repo.upsert_vectors([{"id": 1, "vector": VEC_A}])
        repo.upsert_vectors([{"id": 1, "vector": [0.5, 0.5, 0.7071]}])  # 同 id 再 upsert
        hits = repo.search(VEC_A, top_k=10)
        assert [hit_id for hit_id, _ in hits] == [1]  # 只有 1 行,同 id 覆盖而非重复
    finally:
        repo.close()


def test_search_scores_rank_similar_first(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "milvus.db")
    try:
        repo.upsert_vectors([{"id": 1, "vector": VEC_A}, {"id": 2, "vector": VEC_B}])
        results = repo.search(VEC_A, top_k=2)  # query 与 A 全同、与 B 正交
        assert len(results) == 2
        (id_a, score_a), (id_b, score_b) = results
        assert id_a == 1
        assert score_a == pytest.approx(1.0, abs=1e-6)  # 全同 → score ≈ 1.0
        assert id_b == 2
        assert score_b == pytest.approx(0.0, abs=1e-6)  # 正交 → score ≈ 0.0
        assert score_a > score_b
    finally:
        repo.close()


def test_close_then_reopen_persists(tmp_path: Path) -> None:
    db_path = tmp_path / "milvus.db"
    repo = _make_repo(db_path)
    repo.upsert_vectors([{"id": 7, "vector": VEC_A}, {"id": 8, "vector": VEC_B}])
    repo.close()

    reopened = MilvusKnowledgeRepo(db_path=db_path, dim=DIM)
    try:
        reopened.ensure_collection()  # 已存在,应幂等跳过
        hits = reopened.search(VEC_A, top_k=2)
        assert [hit_id for hit_id, _ in hits] == [7, 8]  # close 后新开 client 数据仍在
    finally:
        reopened.close()


def test_collection_name_is_knowledge() -> None:
    assert COLLECTION_NAME == "knowledge"
