r"""ch02 写穿门面:把编排层的落库意图翻译成一条条独立短事务。

约定(spec §6 / task-9-brief):
- 每次调用独立 session、写完即提交——聊天流不等下一个写点,失败按追加式日志
  语义留给上层异常路径(error 帧),本层不重试不回滚;
- ensure_conversation 复用仓储 get_or_create_conversation(同 user_id 取最近
  一通,无则新建),幂等语义由仓储保证;
- 会话工厂以公开属性 ``session_factory`` 暴露(R2 裁决):编排层构造
  ToolContext(conversation_id=…, session_factory=persistence.session_factory)
  时直接取用,工具内部自开事务。
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.repositories import add_message, get_or_create_conversation


class ChatPersistence:
    """聊天写穿门面:会话幂等建档 + 消息流水逐行落库(每次调用独立短事务)。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        # 公开属性(R2):编排层构造 ToolContext 时原样传入,工具注入用
        self.session_factory = session_factory

    async def ensure_conversation(self, user_id: str) -> int:
        """按 user_id 幂等取/建会话,返回会话 id。"""
        async with self.session_factory() as session, session.begin():
            conversation = await get_or_create_conversation(session, user_id)
            # 提交前读 id(flush 已赋值),不依赖提交后的属性加载
            return conversation.id

    async def log_user_message(self, conversation_id: int, content: str) -> None:
        """落一行 user 消息流水。"""
        async with self.session_factory() as session, session.begin():
            await add_message(session, conversation_id, role="user", content=content)

    async def log_assistant_tool_call(
        self, conversation_id: int, content: str | None, tool_calls: list[dict]
    ) -> None:
        """落一行 assistant 申请单:纯工具调用时 content 传 None(正文可空)。"""
        async with self.session_factory() as session, session.begin():
            await add_message(
                session,
                conversation_id,
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            )

    async def log_tool_result(
        self, conversation_id: int, tool_call_id: str, content: str
    ) -> None:
        """落一行 tool 结果:tool_call_id 对号入座申请单。"""
        async with self.session_factory() as session, session.begin():
            await add_message(
                session,
                conversation_id,
                role="tool",
                content=content,
                tool_call_id=tool_call_id,
            )

    async def log_assistant_message(self, conversation_id: int, content: str) -> None:
        """落一行 assistant 最终正文。"""
        async with self.session_factory() as session, session.begin():
            await add_message(session, conversation_id, role="assistant", content=content)
