r"""ch02 知识库侧工具:FAQ 检索(直连数据库)。

约定:
- 异步工具(@tool 包装协程函数),上层与测试一律走 .ainvoke;
- session_factory 为 InjectedToolArg 注入参数:不进 args_schema(模型不可见),
  由调用方在工具输入里显式携带;
- 工具内部自开事务:``async with session_factory() as s, s.begin()``,
  失败自然上抛,由上层 registry 统一捕获;
- 会话关闭前先取出纯文本,避免关闭后再读 ORM 属性(MissingGreenlet 陷阱)。
"""

from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.repositories import search_faq


class QueryFaqInput(BaseModel):
    """query_faq 工具入参。"""

    keyword: str = Field(description="检索关键词,如「退货」「开发票」")


@tool(args_schema=QueryFaqInput)
async def query_faq(
    keyword: str,
    *,
    session_factory: Annotated[async_sessionmaker[AsyncSession], InjectedToolArg],
) -> str:
    """按关键词检索 FAQ 知识库,返回命中的问答列表。"""
    async with session_factory() as session, session.begin():
        faqs = await search_faq(session, keyword)
        # 会话内先拼纯文本,不让 ORM 对象逃出事务边界
        lines = [f"问:{faq.question}\n答:{faq.answer}" for faq in faqs]

    if not lines:
        return f"未找到与「{keyword}」相关的 FAQ,请换个关键词或转人工客服。"
    return "\n\n".join(lines)
