"""app.tools.ecommerce 三个 mock 工具的行为测试(纯内存,不接接口不建表)。

验证(query_order / query_product / query_logistics):

- random.seed(42) 下 .invoke 返回可 json.loads 的中文文本(ensure_ascii=False);
- query_order 含 order_id 回显与字段:商品名 / 金额 / 订单状态;
- query_product 含字段:价格 / 库存 / 评分;
- query_logistics 含:承运商(四家之一)、轨迹列表 2~4 条、预计送达;
- 三者 args_schema.model_fields 键集正确、description 非空;
- app.tools.get_all_tools() 注册顺序:订单 → 商品 → 物流 → FAQ → 工单。
"""

import json
import random

from app.tools import get_all_tools
from app.tools.ecommerce import query_logistics, query_order, query_product

# brief 锁定的四家承运商
_CARRIERS = {"顺丰速运", "中通快递", "圆通速递", "京东物流"}


def _has_chinese(text: str) -> bool:
    """返回文本是否含中文字符(粗校验「中文文本」这一约束)。"""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


# ---------------------------------------------------------------------------
# query_order
# ---------------------------------------------------------------------------


def test_query_order_returns_chinese_json_with_required_fields() -> None:
    """固定种子下返回可解析的中文 JSON:order_id 回显 + 商品名/金额/订单状态。"""
    random.seed(42)

    raw = query_order.invoke({"order_id": "ORD-1001"})

    assert _has_chinese(raw)
    data = json.loads(raw)
    assert data["订单号"] == "ORD-1001"  # order_id 回显
    assert data["商品名"]
    assert isinstance(data["金额"], (int, float))
    assert data["订单状态"]


# ---------------------------------------------------------------------------
# query_product
# ---------------------------------------------------------------------------


def test_query_product_returns_chinese_json_with_required_fields() -> None:
    """固定种子下返回可解析的中文 JSON:价格 / 库存 / 评分。"""
    random.seed(42)

    raw = query_product.invoke({"product_id": "P-2002"})

    assert _has_chinese(raw)
    data = json.loads(raw)
    assert data["商品ID"] == "P-2002"  # product_id 回显
    assert isinstance(data["价格"], (int, float))
    assert isinstance(data["库存"], int)
    assert 0.0 <= data["评分"] <= 5.0


# ---------------------------------------------------------------------------
# query_logistics
# ---------------------------------------------------------------------------


def test_query_logistics_returns_chinese_json_with_required_fields() -> None:
    """固定种子下返回可解析的中文 JSON:承运商、轨迹 2~4 条、预计送达。"""
    random.seed(42)

    raw = query_logistics.invoke({"order_id": "ORD-3003"})

    assert _has_chinese(raw)
    data = json.loads(raw)
    assert data["订单号"] == "ORD-3003"
    assert data["承运商"] in _CARRIERS
    assert 2 <= len(data["轨迹"]) <= 4
    assert data["预计送达"]


# ---------------------------------------------------------------------------
# args_schema / description 元信息
# ---------------------------------------------------------------------------


def test_mock_tools_args_schema_keys_and_description() -> None:
    """三个 mock 工具的 args_schema 键集正确,description 非空。"""
    expectations = [
        (query_order, {"order_id"}),
        (query_product, {"product_id"}),
        (query_logistics, {"order_id"}),
    ]
    for tool, field_keys in expectations:
        assert set(tool.args_schema.model_fields.keys()) == field_keys, tool.name
        assert tool.description.strip(), tool.name


# ---------------------------------------------------------------------------
# app.tools.get_all_tools 注册顺序
# ---------------------------------------------------------------------------


def test_get_all_tools_registration_order() -> None:
    """注册顺序:query_order / query_product / query_logistics / query_faq / create_ticket。"""
    names = [tool.name for tool in get_all_tools()]

    assert names == [
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    ]
