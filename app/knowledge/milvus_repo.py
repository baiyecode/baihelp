"""Milvus Lite 知识集合仓储:把 pymilvus 细节封装在类内,不向外泄漏 MilvusClient。

score 语义(实测校准,证据见 task-5-report.md):milvus-lite 3.2.1 / pymilvus 3.0.2
在 metric_type="COSINE" 下,search 返回的 hit["distance"] 就是余弦相似度本身
(全同 → 1.0,60° → 0.5,正交 → 0.0,相反 → −1.0),并非标准"距离 = 1 − 相似度"。
因此 score 直接取该值,高分即更相似;若按 1 − distance 换算,全同向量反而得 0 分,
与本仓储"相似优先"的语义相反。
"""

from pathlib import Path

from pymilvus import DataType, MilvusClient
from pymilvus.milvus_client.index import IndexParams

COLLECTION_NAME = "knowledge"


class MilvusKnowledgeRepo:
    def __init__(self, db_path: str | Path, dim: int) -> None:
        self._client = MilvusClient(str(db_path))
        self._dim = dim

    def ensure_collection(self) -> None:
        """建集合(schema:INT64 主键 auto_id=False + FLOAT_VECTOR dim;索引 COSINE),幂等。

        集合已存在时(如持锁进程被硬杀后落盘为 released 状态)补 load_collection:
        实测已加载状态下 load_collection 为幂等 no-op,故新建与已存在两分支统一收口调用。
        """
        if not self._client.has_collection(COLLECTION_NAME):
            schema = MilvusClient.create_schema(auto_id=False)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)
            index_params = IndexParams()
            index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
            self._client.create_collection(COLLECTION_NAME, schema=schema, index_params=index_params)
        self._client.load_collection(COLLECTION_NAME)

    def upsert_vectors(self, rows: list[dict]) -> None:
        """按主键 id 覆盖写入;同 id 重复 upsert 不产生重复行。

        行 dict 只能含 {"id", "vector"} 两键(集合未开 dynamic field):多传任何
        其他键(如原文 text)即抛 DataNotMatchException,调用方负责裁剪字段。
        """
        self._client.upsert(COLLECTION_NAME, data=rows)

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[int, float]]:
        """向量近邻检索,返回 (id, score) 列表;score 为余弦相似度,越大越相似。"""
        hits = self._client.search(COLLECTION_NAME, data=[query_vector], limit=top_k)[0]
        return [(hit["id"], float(hit["distance"])) for hit in hits]

    def close(self) -> None:
        self._client.close()
