"""流式客服回复编排：历史裁剪 → 模型流式生成 → SSE 事件下发。"""

import logging
from collections.abc import AsyncIterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from app.api.sse import format_delta_chunk, format_done, format_error_event
from app.services.history import HistoryTrimmer, SessionStore

logger = logging.getLogger(__name__)

# 商城名默认字面量：模板变量必须填充；商城名接入配置前先用该兜底值。
DEFAULT_SHOP_NAME = "本店"

# 流内错误对客户端展示的统一文案（不泄露上游细节，服务端日志里留全量堆栈）。
_ERROR_MESSAGE = "服务暂时不可用，请稍后再试"


async def stream_chat_reply(
    model: BaseChatModel,
    store: SessionStore,
    trimmer: HistoryTrimmer,
    system_template: ChatPromptTemplate,
    session_id: str,
    message: str,
) -> AsyncIterator[str]:
    """为一次用户消息产出可直接下发的 SSE 事件流。

    - 取会话历史，连同渲染后的 system 消息与新消息一起经 trimmer 预算裁剪
      （system 恒为 messages[0]，裁剪器约定如此）；
    - 逐 chunk 下发 delta 事件，空文本 chunk 跳过（部分提供方会发空增量）；
    - 流正常结束后把完整聚合回复与用户消息写回 store；
    - 中途任何异常：不写任何历史，只下发 error 事件 + [DONE]，
      客户端总能看到一条有终止的流。
    """
    full_text = ""
    try:
        history = store.get(session_id)
        system_message = system_template.format_messages(shop_name=DEFAULT_SHOP_NAME)[0]
        messages = trimmer.trim(
            [system_message, *history, HumanMessage(content=message)]
        )
        async for chunk in model.astream(messages):
            text = chunk.text or ""
            if not text:
                continue
            full_text += text
            yield format_delta_chunk(text)
        store.append(
            session_id, [HumanMessage(content=message), AIMessage(content=full_text)]
        )
    except Exception:
        logger.exception("流式回复失败 session_id=%s", session_id)
        yield format_error_event(_ERROR_MESSAGE)
        yield format_done()
        return
    yield format_done()
