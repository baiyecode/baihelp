"""售后结构化抽取端点。"""

import logging

from fastapi import APIRouter, HTTPException
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from app.core.config import get_settings
from app.llm.factory import get_chat_model
from app.schemas.extraction import AfterSalesExtraction
from app.services.extraction import extract_after_sales

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/extract", tags=["extract"])


class ExtractRequest(BaseModel):
    """POST /api/extract 请求体。"""

    text: str


def get_model() -> BaseChatModel:
    """构造真实聊天模型；测试经 monkeypatch 本函数注入假模型（DI 缝隙）。"""
    return get_chat_model(get_settings())


@router.post("")
async def extract(body: ExtractRequest) -> AfterSalesExtraction:
    """结构化抽取：成功返回三字段 JSON，上游失败映射为 502。"""
    model = get_model()
    try:
        return await extract_after_sales(model, body.text)
    except Exception as exc:
        logger.exception("抽取链执行失败")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
