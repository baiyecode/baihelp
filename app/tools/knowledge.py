r"""ch02/ch03 知识库侧工具:FAQ 语义检索(向量检索 + MySQL 回表)。

约定:
- 异步工具(@tool 包装协程函数),上层与测试一律走 .ainvoke;
- session_factory / retriever 为 InjectedToolArg 注入参数:不进 args_schema
  (模型不可见),由调用方在工具输入或 registry 的 ToolContext 里显式携带;
- 检索链路封在 KnowledgeRetriever:embed → Milvus 近邻 → 相似度阈值过滤 →
  knowledge_chunks 回表(vectorize_status='done')→ score 降序;
- 出参契约与 ch02 逐字不变:多条以空行相连,每条「问:…\n答:…」——questions
  内含换行(一条 chunk 的多种问法),拼接前压成单行,否则破坏行结构;
- 空结果回兜底话术,引导用户换关键词或转人工。
"""

from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.knowledge.retriever import KnowledgeRetriever


class QueryFaqInput(BaseModel):
    """query_faq 工具入参。"""

    keyword: str = Field(description="检索关键词,如「退货」「开发票」")


def _flatten_questions(questions: str) -> str:
    """多行问法压成单行:逐行去首尾空白、丢空行,以「 / 」相连。

    questions 内含换行(一条 chunk 存多种问法),不压平会把第 2 行起的问法
    混进「答:」行,破坏「问:…\n答:…」结构。
    """
    lines = (line.strip() for line in questions.splitlines())
    return " / ".join(line for line in lines if line)


@tool(args_schema=QueryFaqInput)
async def query_faq(
    keyword: str,
    *,
    session_factory: Annotated[async_sessionmaker[AsyncSession], InjectedToolArg],
    retriever: Annotated[KnowledgeRetriever, InjectedToolArg],
) -> str:
    """按语义检索知识库,建议传入用户问题原话或关键词"""
    hits = await retriever.retrieve(keyword, session_factory)
    # 检索器已在会话关闭前取纯文本,此处拿到的可直接拼接
    blocks = [
        f"问:{_flatten_questions(hit.questions)}\n答:{hit.answer}" for hit in hits
    ]

    if not blocks:
        return f"未找到与「{keyword}」相关的 FAQ,请换个关键词或转人工客服。"
    return "\n\n".join(blocks)
