"""售后抽取结构化 Schema（spec §7.5）。"""

from typing import Literal

from pydantic import BaseModel, Field


class AfterSalesExtraction(BaseModel):
    """从单条顾客消息中抽取的售后登记信息。"""

    order_no: str | None = Field(description="订单号，如 SO20260927001；提取不到为 null")
    complaint_type: Literal["退货退款", "换货", "维修", "物流问题", "发票售后", "其他"]
    expected_solution: str = Field(description="用户期望的解决方案，原话概括")
