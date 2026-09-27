r"""ch02 工具箱:五个业务工具与统一注册入口。

- 电商 mock 三件套(query_order / query_product / query_logistics)在 ecommerce;
- 知识库检索(query_faq)在 knowledge;建工单(create_ticket)在 tickets;
- get_all_tools() 按固定注册顺序返回,顺序即绑定给模型的 tools 列表顺序。
"""

from app.core.config import Settings
from app.tools.ecommerce import query_logistics, query_order, query_product
from app.tools.knowledge import query_faq
from app.tools.registry import ToolRegistry
from app.tools.tickets import create_ticket

__all__ = [
    "query_order",
    "query_product",
    "query_logistics",
    "query_faq",
    "create_ticket",
    "get_all_tools",
    "build_default_registry",
]


def get_all_tools() -> list:
    """按注册顺序返回全部业务工具:订单 → 商品 → 物流 → FAQ → 工单。"""
    return [query_order, query_product, query_logistics, query_faq, create_ticket]


def build_default_registry(settings: Settings) -> ToolRegistry:
    """按配置构建默认工具注册表:五工具按注册顺序入表,超时/重试上限取自 settings。

    供 lifespan 组装 app.state.tool_registry 用;TOOL_TIMEOUT_SECONDS /
    TOOL_MAX_RETRIES 未配置时走 Settings 默认值(10s / 1 次)。
    """
    registry = ToolRegistry(
        timeout_seconds=settings.tool_timeout_seconds,
        max_retries=settings.tool_max_retries,
    )
    for tool in get_all_tools():
        registry.register(tool)
    return registry
