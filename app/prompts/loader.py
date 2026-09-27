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
