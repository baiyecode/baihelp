r"""ch02 工具箱:五个业务工具与统一注册入口。

- 电商 mock 三件套(query_order / query_product / query_logistics)在 ecommerce;
- 知识库检索(query_faq)在 knowledge;建工单(create_ticket)在 tickets;
- get_all_tools() 按固定注册顺序返回,顺序即绑定给模型的 tools 列表顺序。
"""

from app.tools.ecommerce import query_logistics, query_order, query_product
from app.tools.knowledge import query_faq
from app.tools.tickets import create_ticket

__all__ = [
    "query_order",
    "query_product",
    "query_logistics",
    "query_faq",
    "create_ticket",
    "get_all_tools",
]


def get_all_tools() -> list:
    """按注册顺序返回全部业务工具:订单 → 商品 → 物流 → FAQ → 工单。"""
    return [query_order, query_product, query_logistics, query_faq, create_ticket]
