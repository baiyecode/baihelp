"""BgeM3Embedder 密封测试:monkeypatch 模块内 AsyncOpenAI,不触网。

覆盖:请求参数(model/完整 texts/base_url/api_key 透传)、乱序 index 还原顺序、
每条向量长度 == dim、空输入零请求。
"""

import pytest

import app.knowledge.embedding as emb_mod
from app.knowledge.embedding import BgeM3Embedder

DIM = 4


class FakeEmbeddings:
    """记录 create 调用参数,按预设乱序 index 返回 embedding 数据。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.shuffled_indexes: list[list[int]] = []

    async def create(self, *, model: str, input: list[str]):  # noqa: A002 - 对齐 openai SDK 形参
        self.calls.append({"model": model, "input": input})
        order = self.shuffled_indexes.pop(0) if self.shuffled_indexes else list(range(len(input)))
        data = []
        for idx in order:
            item = type("EmbeddingData", (), {})()
            item.index = idx
            item.embedding = [float(idx)] * DIM
            data.append(item)
        return type("CreateResponse", (), {"data": data})()


class FakeAsyncOpenAI:
    """替身构造器:记录实例化参数,挂上共享的 FakeEmbeddings。"""

    instances: list["FakeAsyncOpenAI"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.embeddings = FakeEmbeddings()
        FakeAsyncOpenAI.instances.append(self)


@pytest.fixture()
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> list[FakeAsyncOpenAI]:
    FakeAsyncOpenAI.instances = []
    monkeypatch.setattr(emb_mod, "AsyncOpenAI", FakeAsyncOpenAI)
    return FakeAsyncOpenAI.instances


def _make_embedder() -> BgeM3Embedder:
    return BgeM3Embedder(
        base_url="https://fake.example/v1",
        api_key="test-key",
        model="BAAI/bge-m3",
        dim=DIM,
    )


@pytest.mark.asyncio
async def test_embed_passes_model_and_full_texts(fake_openai: list[FakeAsyncOpenAI]) -> None:
    embedder = _make_embedder()
    texts = ["运费怎么算", "退货政策", "发货时效"]
    await embedder.embed(texts)
    assert len(fake_openai) == 1
    assert fake_openai[0].kwargs == {"base_url": "https://fake.example/v1", "api_key": "test-key"}
    calls = fake_openai[0].embeddings.calls
    assert len(calls) == 1
    assert calls[0]["model"] == "BAAI/bge-m3"
    assert calls[0]["input"] == texts  # 完整 texts 一次性传入,未截断未改序


@pytest.mark.asyncio
async def test_embed_restores_order_by_index(fake_openai: list[FakeAsyncOpenAI]) -> None:
    embedder = _make_embedder()
    fake_openai[0].embeddings.shuffled_indexes.append([2, 0, 1])  # 服务端返回乱序 index
    result = await embedder.embed(["甲", "乙", "丙"])
    # 按 index 还原:index 2 → [2.0]*4,index 0 → [0.0]*4,index 1 → [1.0]*4
    assert result == [[0.0] * DIM, [1.0] * DIM, [2.0] * DIM]


@pytest.mark.asyncio
async def test_embed_each_vector_length_equals_dim(fake_openai: list[FakeAsyncOpenAI]) -> None:
    embedder = _make_embedder()
    result = await embedder.embed(["a", "b"])
    assert len(result) == 2
    assert all(len(vec) == DIM for vec in result)


@pytest.mark.asyncio
async def test_embed_empty_input_makes_zero_requests(fake_openai: list[FakeAsyncOpenAI]) -> None:
    embedder = _make_embedder()
    assert await embedder.embed([]) == []
    assert fake_openai[0].embeddings.calls == []  # 未发任何请求
