"""SSE 流式客服对话端点。"""

from fastapi import APIRouter, Request
from fastapi.sse import EventSourceResponse
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.db.engine import build_engine, build_session_factory
from app.llm.factory import get_chat_model
from app.services.chat import stream_chat_reply
from app.services.history import HistoryTrimmer
from app.services.persistence import ChatPersistence

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatStreamRequest(BaseModel):
    """POST /api/chat/stream 请求体。"""

    session_id: str
    message: str


def get_model() -> BaseChatModel:
    """构造真实聊天模型；测试经 monkeypatch 本函数注入假模型（DI 缝隙）。"""
    return get_chat_model(get_settings())


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """构造真实会话工厂；测试经 monkeypatch 本函数注入 SQLite 内存工厂（DI 缝隙）。"""
    return build_session_factory(build_engine(get_settings().database_url))


@router.post("/stream")
async def chat_stream(body: ChatStreamRequest, request: Request) -> EventSourceResponse:
    """流式回复：历史预算裁剪 → 模型流式生成（可含工具轮）→ SSE 事件逐字节下发。

    registry 取自 lifespan 组装的 app.state.tool_registry（绑定五工具 + 超时重试）；
    persistence 走 get_session_factory 接缝组装写穿门面（会话建档 + 消息逐行落库）。
    """
    model = get_model()
    # 模型计数器面向 str（BaseLanguageModel.get_num_tokens），裁剪器面向
    # BaseMessage：经 .text 适配（spec §7.4 默认计数器）。
    trimmer = HistoryTrimmer(
        lambda message: model.get_num_tokens(message.text),
        get_settings().history_token_budget,
    )
    generator = stream_chat_reply(
        model,
        request.app.state.store,
        trimmer,
        request.app.state.system_template,
        body.session_id,
        body.message,
        registry=request.app.state.tool_registry,
        persistence=ChatPersistence(get_session_factory()),
    )
    # 直接返回 EventSourceResponse，把生成器的预格式化 SSE 串原样下发。
    # 不能挂 response_class=EventSourceResponse 走 producer 模式——那会把
    # 产出的字符串再 JSON 编码一层（data: "data: ..."），破坏线上格式。
    return EventSourceResponse(generator)
