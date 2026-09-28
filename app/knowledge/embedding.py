"""BGE-M3 嵌入客户端:OpenAI 兼容 embeddings 端点封装(SiliconFlow 等)。

顺序契约:返回向量与传入 texts 按 index 一一对齐;空输入直接返回空列表,不发请求。
"""

from openai import AsyncOpenAI


class BgeM3Embedder:
    def __init__(self, base_url: str, api_key: str, model: str, dim: int) -> None:
        self.dim = dim
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入,返回与 texts 顺序对齐的向量列表;空输入零请求。"""
        if not texts:
            return []
        response = await self._client.embeddings.create(model=self._model, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
