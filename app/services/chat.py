r"""流式客服回复编排:写穿落库 → 历史裁剪 → 两段式生成(可含工具轮)→ SSE 事件下发。

两段式(spec §6):第一段绑工具流式,chunk 边吐 delta 边聚合;聚合含 tool_calls
则逐个执行工具(running/done 状态帧),结果以 ToolMessage 回灌,第二段裸模型
(不绑工具)基于同一段上下文收敛出最终回答。内存 store 仍「成功才追加」,工具轮
一次性追加 [Human, AIMessage(tool_calls), *ToolMessages, AIMessage(最终)]。
"""

import logging
from collections.abc import AsyncIterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from app.api.sse import (
    format_delta_chunk,
    format_done,
    format_error_event,
    format_tool_event,
)
from app.services.history import HistoryTrimmer, SessionStore
from app.services.persistence import ChatPersistence
from app.tools.registry import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

# 商城名默认字面量：模板变量必须填充；商城名接入配置前先用该兜底值。
DEFAULT_SHOP_NAME = "本店"

# 流内错误对客户端展示的统一文案（不泄露上游细节，服务端日志里留全量堆栈）。
_ERROR_MESSAGE = "服务暂时不可用，请稍后再试"

# 工具 done 帧 summary 截断长度(spec §5.2:截断 120 字供徽章悬浮展示)
_SUMMARY_MAX_CHARS = 120


async def stream_chat_reply(
    model: BaseChatModel,
    store: SessionStore,
    trimmer: HistoryTrimmer,
    system_template: ChatPromptTemplate,
    session_id: str,
    message: str,
    *,
    registry: ToolRegistry | None = None,
    persistence: ChatPersistence | None = None,
) -> AsyncIterator[str]:
    """为一次用户消息产出可直接下发的 SSE 事件流。

    - registry/persistence 为带 None 默认的关键字参数(R1 裁决):registry=None
      不绑工具(单段,同 ch01);persistence=None 跳过写穿(仅内存,同 ch01);
    - 取会话历史,连同渲染后的 system 消息与新消息一起经 trimmer 预算裁剪
      (system 恒为 messages[0],裁剪器约定如此;历史为内部活列表,绝不变异);
    - 第一段逐 chunk 下发 delta(空文本 chunk 跳过)同时聚合;聚合含 tool_calls
      则进入工具轮:running 帧 → registry.execute(错误也返回结果不抛)→ done 帧
      (summary 截断 120 字),结果组 ToolMessage 回灌,第二段用裸模型流式收敛;
    - 内存 store 在流正常结束后一次性追加完整往返;DB 写穿逐行落库(申请单行的
      tool_calls 为全量列表,首段无文本时正文落 NULL);
    - 中途任何异常:store 不写,只下发 error 事件 + [DONE],客户端总能看到一条
      有终止的流;DB 已落行为追加式日志,不回滚(spec §9)。
    """
    full_text = ""
    try:
        # 1. 写穿 user 行(persistence=None 跳过,行为同 ch01)
        conversation_id: int | None = None
        if persistence is not None:
            conversation_id = await persistence.ensure_conversation(session_id)
            await persistence.log_user_message(conversation_id, message)

        # 2. 组上下文:system + 内存历史 + 新消息,经裁剪器产出全新列表
        history = store.get(session_id)
        system_message = system_template.format_messages(shop_name=DEFAULT_SHOP_NAME)[0]
        messages = trimmer.trim(
            [system_message, *history, HumanMessage(content=message)]
        )

        # 3. 第一段:绑工具后流式,chunk 边吐 delta 边聚合(工具申请单藏在聚合里)
        aggregated: AIMessageChunk | None = None
        bound = model.bind_tools(registry.tools) if registry is not None else model
        async for chunk in bound.astream(messages):
            text = chunk.text or ""
            if text:
                full_text += text
                yield format_delta_chunk(text)
            aggregated = chunk if aggregated is None else aggregated + chunk

        if aggregated is not None and aggregated.tool_calls:
            # 4a. 工具轮:执行回灌 + 写穿,第二段裸模型收敛出最终回答
            first_text = full_text
            assistant_tool_call = AIMessage(
                content=first_text, tool_calls=aggregated.tool_calls
            )
            tool_calls = list(aggregated.tool_calls)
            tool_messages: list[ToolMessage] = []
            for tool_call in tool_calls:
                name, args = tool_call["name"], tool_call["args"]
                yield format_tool_event(name, "running", args=args)
                # execute 恒返回结果(失败文本可直接回灌),不抛异常
                result = await registry.execute(
                    name,
                    args,
                    # persistence=None 时注入字段为 None,registry 对 None 跳过注入
                    ToolContext(
                        conversation_id=conversation_id,
                        session_factory=(
                            persistence.session_factory
                            if persistence is not None
                            else None
                        ),
                    ),
                )
                yield format_tool_event(
                    name, "done", summary=result.content[:_SUMMARY_MAX_CHARS]
                )
                tool_messages.append(
                    ToolMessage(content=result.content, tool_call_id=tool_call["id"])
                )

            if persistence is not None:
                # 申请单一行带全量 tool_calls(spec §6:一条 assistant + N 条 tool);
                # 首段无文本时正文落 NULL,而非空串
                await persistence.log_assistant_tool_call(
                    conversation_id, first_text or None, tool_calls
                )
                for tool_call, tool_message in zip(tool_calls, tool_messages):
                    await persistence.log_tool_result(
                        conversation_id, tool_call["id"], tool_message.content
                    )

            # 第二段:裸模型(不绑工具)基于第一段上下文 + 工具往返流式收敛
            final_text = ""
            async for chunk in model.astream(
                [*messages, assistant_tool_call, *tool_messages]
            ):
                text = chunk.text or ""
                if text:
                    final_text += text
                    yield format_delta_chunk(text)

            if persistence is not None:
                await persistence.log_assistant_message(conversation_id, final_text)

            store.append(
                session_id,
                [
                    HumanMessage(content=message),
                    assistant_tool_call,
                    *tool_messages,
                    AIMessage(content=final_text),
                ],
            )
        else:
            # 4b. 无工具轮:ch01 原路径(store 语义不变;DB 落最终 assistant 行)
            if persistence is not None:
                await persistence.log_assistant_message(conversation_id, full_text)
            store.append(
                session_id, [HumanMessage(content=message), AIMessage(content=full_text)]
            )
    except Exception:
        logger.exception("流式回复失败 session_id=%s", session_id)
        yield format_error_event(_ERROR_MESSAGE)
        yield format_done()
        return
    yield format_done()
