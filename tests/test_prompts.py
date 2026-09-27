"""Tests for app.prompts template loader."""

from langchain_core.prompts import ChatPromptTemplate

from app.prompts.loader import load_extraction_prompt, load_system_prompt

SHOP_NAME = "喵帮帮商城"
SIX_CATEGORIES = ("退货退款", "换货", "维修", "物流问题", "发票售后", "其他")


def test_system_prompt_renders_shop_name() -> None:
    template = load_system_prompt()
    assert isinstance(template, ChatPromptTemplate)

    messages = template.invoke({"shop_name": SHOP_NAME}).to_messages()
    system_content = messages[0].content
    assert SHOP_NAME in system_content
    assert "客服" in system_content


def test_extraction_prompt_lists_six_categories() -> None:
    template = load_extraction_prompt()
    assert isinstance(template, ChatPromptTemplate)

    user_text = "我要退货，订单号 SO20260927001"
    messages = template.invoke({"text": user_text}).to_messages()
    system_content = messages[0].content

    for category in SIX_CATEGORIES:
        assert category in system_content
    assert "null" in system_content
    assert messages[-1].content == user_text
