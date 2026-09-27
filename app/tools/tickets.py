r"""ch02 工单侧工具:创建人工工单(直连数据库)。

约定:
- 异步工具(@tool 包装协程函数),上层与测试一律走 .ainvoke;
- conversation_id / session_factory 均为 InjectedToolArg 注入参数:
  不进 args_schema(模型只见 description 与 ticket_type),由调用方携带;
- 工具内部自开事务:``async with session_factory() as s, s.begin()``,
  失败自然上抛,由上层 registry 统一捕获;
- 工单号生成与撞号重试细节都在仓储层(app.db.repositories.create_ticket)。
"""

from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.repositories import create_ticket as create_ticket_repo


class CreateTicketInput(BaseModel):
    """create_ticket 工具入参(注入参数 conversation_id/session_factory 不在此模型内)。"""

    description: str = Field(description="问题描述,尽量包含订单号、商品名与具体诉求")
    ticket_type: Literal["售后", "投诉", "咨询"] = Field(
        description="工单类型,三选一:售后(退款/退换货/维修)、"
        "投诉(对服务/物流/商品不满)、咨询(售前或使用疑问)"
    )


@tool(args_schema=CreateTicketInput)
async def create_ticket(
    description: str,
    ticket_type: Literal["售后", "投诉", "咨询"],
    *,
    conversation_id: Annotated[int, InjectedToolArg],
    session_factory: Annotated[async_sessionmaker[AsyncSession], InjectedToolArg],
) -> str:
    """为当前会话创建一张人工工单,交由人工客服跟进处理。"""
    async with session_factory() as session, session.begin():
        ticket = await create_ticket_repo(session, conversation_id, description, ticket_type)
        # 会话内先取标量值,避免会话关闭后读 ORM 属性(MissingGreenlet 陷阱)
        ticket_no = ticket.ticket_no
        status = ticket.status

    return f"已创建工单 {ticket_no},状态:{status}"
