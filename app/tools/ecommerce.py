r"""ch02 电商侧三个 mock 工具:订单 / 商品 / 物流。

约定(ch02 教学定位):
- 不接任何外部接口、不建表,数据全部由 random 现场生成,结果以
  ``json.dumps(..., ensure_ascii=False)`` 输出中文 JSON 文本,供上层直接回灌模型;
- 三者均为同步函数(@tool 包装成 StructuredTool),测试可直接 .invoke;
- args_schema 显式给出 Pydantic 模型,Field 描述用中文,帮助模型理解参数含义。
"""

import json
import random
from datetime import date, timedelta

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 入参 schema
# ---------------------------------------------------------------------------


class QueryOrderInput(BaseModel):
    """query_order 工具入参。"""

    order_id: str = Field(
        description="订单号,原样传入用户提供的编号即可,无需校验格式,如 1001、ORD-20260927-001"
    )


class QueryProductInput(BaseModel):
    """query_product 工具入参。"""

    product_id: str = Field(
        description="商品 ID,原样传入用户提供的编号即可,无需校验格式,如 1001、P-1001"
    )


class QueryLogisticsInput(BaseModel):
    """query_logistics 工具入参。"""

    order_id: str = Field(
        description="订单号,原样传入用户提供的编号即可,无需校验格式,如 1001、ORD-20260927-001"
    )


# ---------------------------------------------------------------------------
# mock 数据源(仅 ch02 演示用,与数据库无任何关联)
# ---------------------------------------------------------------------------

_PRODUCT_NAMES = ("无线蓝牙耳机", "智能保温杯", "便携折叠伞", "硅胶餐垫", "桌面加湿器")
_ORDER_STATUSES = ("待付款", "已付款", "已发货", "已签收", "已取消")
_CARRIERS = ("顺丰速运", "中通快递", "圆通速递", "京东物流")
# 轨迹事件按真实时序排列,sample 后按此顺序还原
_TRACK_EVENTS = ("订单已创建", "商家已发货", "包裹已到达转运中心", "派送中", "已签收")


def _dump(data: dict) -> str:
    """统一出口:中文原样输出(ensure_ascii=False),返回可 json.loads 的文本。"""
    return json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 三个工具
# ---------------------------------------------------------------------------


@tool(args_schema=QueryOrderInput)
def query_order(order_id: str) -> str:
    """按订单号查询订单信息,返回商品名、金额与订单状态。"""
    order = {
        "订单号": order_id,
        "商品名": random.choice(_PRODUCT_NAMES),
        "金额": round(random.uniform(50.0, 5000.0), 2),
        "订单状态": random.choice(_ORDER_STATUSES),
    }
    return _dump(order)


@tool(args_schema=QueryProductInput)
def query_product(product_id: str) -> str:
    """按商品 ID 查询商品信息,返回价格、库存与评分。"""
    product = {
        "商品ID": product_id,
        "商品名": random.choice(_PRODUCT_NAMES),
        "价格": round(random.uniform(9.9, 999.9), 2),
        "库存": random.randint(0, 999),
        "评分": round(random.uniform(3.0, 5.0), 1),
    }
    return _dump(product)


@tool(args_schema=QueryLogisticsInput)
def query_logistics(order_id: str) -> str:
    """按订单号查询物流信息,返回承运商、轨迹列表与预计送达时间。"""
    trace_count = random.randint(2, 4)
    events = random.sample(_TRACK_EVENTS, trace_count)
    events.sort(key=_TRACK_EVENTS.index)  # 按真实时序还原
    today = date.today()
    trace = [
        {
            "时间": (today - timedelta(days=trace_count - 1 - i)).isoformat(),
            "事件": event,
        }
        for i, event in enumerate(events)
    ]
    logistics = {
        "订单号": order_id,
        "承运商": random.choice(_CARRIERS),
        "轨迹": trace,
        "预计送达": (today + timedelta(days=random.randint(1, 3))).isoformat(),
    }
    return _dump(logistics)
