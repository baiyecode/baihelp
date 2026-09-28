"""Tests for the after-sales extraction endpoint (app.main + app.api.extract)."""

import httpx
import pytest
import pytest_asyncio
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda

import app.api.extract as extract_api
from app.core.config import Settings
from app.main import app
from app.schemas.extraction import AfterSalesExtraction

_AFTER_SALES_TEXT = "订单号SO20260927001，屏幕摔裂了想退货退款，麻烦尽快处理"


class FakeModel:
    """``with_structured_output`` 可控假模型：返回固定 include_raw 结果或抛异常。"""

    def __init__(
        self,
        parsed: AfterSalesExtraction | None = None,
        error: Exception | None = None,
    ) -> None:
        self._parsed = parsed
        self._error = error
        # (schema, include_raw) 调用记录：锁定 include_raw 契约。
        self.calls: list[tuple[type, bool]] = []
        # 结构化 runnable 收到的渲染后 prompt（ChatPromptValue）。
        self.chain_inputs: list[ChatPromptValue] = []

    def with_structured_output(self, schema: type, *, include_raw: bool = False):
        self.calls.append((schema, include_raw))

        async def _invoke(prompt_value: ChatPromptValue) -> dict:
            self.chain_inputs.append(prompt_value)
            if self._error is not None:
                raise self._error
            return {
                "raw": AIMessage(content=""),
                "parsed": self._parsed,
                "parsing_error": None,
            }

        return RunnableLambda(_invoke)


def _install_fake(
    monkeypatch: pytest.MonkeyPatch,
    parsed: AfterSalesExtraction | None = None,
    error: Exception | None = None,
) -> FakeModel:
    """Replace the endpoint's model/settings seams with hermetic fakes.

    DI 缝隙：端点模块级的 ``get_model`` / ``get_settings``，monkeypatch 后
    真实模型工厂与真实配置都不会被触达。
    """
    fake = FakeModel(parsed=parsed, error=error)
    settings = Settings(_env_file=None, llm_api_key="sk-test", embedding_api_key="e")
    monkeypatch.setattr(extract_api, "get_model", lambda: fake)
    monkeypatch.setattr(extract_api, "get_settings", lambda: settings)
    return fake


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_extract_returns_fixed_fields_json(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """成功路径：200 且响应体恰为三字段 JSON，text 确实流入抽取 prompt。"""
    parsed = AfterSalesExtraction(
        order_no="SO20260927001",
        complaint_type="退货退款",
        expected_solution="退货退款，尽快处理",
    )
    fake = _install_fake(monkeypatch, parsed=parsed)

    response = await client.post(
        "/api/extract", json={"text": _AFTER_SALES_TEXT}
    )

    assert response.status_code == 200
    assert response.json() == {
        "order_no": "SO20260927001",
        "complaint_type": "退货退款",
        "expected_solution": "退货退款，尽快处理",
    }
    # include_raw 契约固定：后续 eval/验收依赖 raw/parsed/parsing_error 形状。
    assert fake.calls == [(AfterSalesExtraction, True)]
    # 顾客原话作为 human 消息进入渲染后的 prompt。
    messages = fake.chain_inputs[0].to_messages()
    assert messages[-1].content == _AFTER_SALES_TEXT


@pytest.mark.asyncio
async def test_upstream_failure_returns_502(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """上游任何异常：502 且 detail 携带错误信息。"""
    _install_fake(monkeypatch, error=RuntimeError("模拟上游故障"))

    response = await client.post(
        "/api/extract", json={"text": _AFTER_SALES_TEXT}
    )

    assert response.status_code == 502
    assert response.json()["detail"]
