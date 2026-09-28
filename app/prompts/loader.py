"""Load markdown prompt templates as LangChain chat prompt templates."""

from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

_TEMPLATES_DIR = Path(__file__).parent


def _read_template(filename: str) -> str:
    return (_TEMPLATES_DIR / filename).read_text(encoding="utf-8")


def load_system_prompt() -> ChatPromptTemplate:
    """客服 System Prompt 模板，渲染时需提供 ``shop_name``。"""
    return ChatPromptTemplate.from_messages(
        [("system", _read_template("system_prompt.md"))]
    )


def load_extraction_prompt() -> ChatPromptTemplate:
    """售后信息抽取模板，渲染时需提供 ``text``（顾客消息）。"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", _read_template("extraction_prompt.md")),
            ("human", "{text}"),
        ]
    )


def load_qa_extraction_prompt() -> ChatPromptTemplate:
    """历史对话挖 QA 模板，渲染时需提供 ``conversations``（拼接后的会话块文本）。

    模板的输出契约含 JSON 对象示例（字面花括号），故用 mustache 模板格式：
    system 模板里的单花括号原样保留；human 占位用三花括号 ``{{{conversations}}}``
    关闭 HTML 转义——会话文本含 <、&、" 时必须原样送达，不得被转义破坏。
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", _read_template("qa_extraction_prompt.md")),
            ("human", "{{{conversations}}}"),
        ],
        template_format="mustache",
    )
