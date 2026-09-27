r"""ch02 四件套仓储:会话 / 消息 / FAQ / 工单的瘦数据访问函数。

约定:
- 一律收 AsyncSession,函数内只 flush() 不 commit——事务边界归调用方;
- search_faq 对 LIKE 元字符(\ % _)做转义,用户输入一律按字面匹配,防通配注入;
- create_ticket 工单号 = "T" + 当日 YYYYMMDD + 三位当日序号(当日计数 + 1),
  主键冲突时回滚当前 savepoint 换号重试,至多尝试 3 次。
"""

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Faq, Message, Ticket

# 工单号冲突重试上限(含首次生成;全撞则抛 RuntimeError)
_TICKET_MAX_ATTEMPTS = 3


async def get_or_create_conversation(session: AsyncSession, user_id: str) -> Conversation:
    """按 user_id 取最近一通会话(order_by id desc limit 1);没有则新建。

    新建不显式传 status,走模型默认「进行中」;flush 只为立刻拿到自增 id,
    提交归调用方。
    """
    conversation = await session.scalar(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.id.desc())
        .limit(1)
    )
    if conversation is None:
        conversation = Conversation(user_id=user_id)
        session.add(conversation)
        await session.flush()
    return conversation


async def add_message(
    session: AsyncSession,
    conversation_id: int,
    *,
    role: str,
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    tool_call_id: str | None = None,
) -> Message:
    """追加一条消息流水:tool 行带 tool_call_id,assistant 纯工具调用时 content 可空。"""
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )
    session.add(message)
    await session.flush()
    return message


async def search_faq(session: AsyncSession, keyword: str, *, limit: int = 5) -> list[Faq]:
    r"""按 question LIKE 检索 FAQ,命中条数受 limit 约束。

    LIKE 元字符转义(评审焦点 #2):\ → \\、% → \%、_ → \_,并声明 escape="\\",
    保证用户输入按字面匹配;转义必须先处理 \ 本身,否则会二次转义。
    空 keyword 直接短路返回空列表,不发 LIKE。
    """
    if not keyword:
        return []
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    stmt = select(Faq).where(Faq.question.like(f"%{escaped}%", escape="\\")).limit(limit)
    return list((await session.scalars(stmt)).all())


async def create_ticket(
    session: AsyncSession,
    conversation_id: int,
    description: str,
    ticket_type: str,
) -> Ticket:
    """落一张人工工单,工单号 = "T" + 当日 YYYYMMDD + 三位当日序号(当日计数 + 1)。

    主键冲突(并发撞号)时回滚当前 savepoint 换下一个候选号重试,至多尝试
    ``_TICKET_MAX_ATTEMPTS`` 次:候选号在当日计数基础上叠加尝试次数,保证
    单调前进不原地打转;重试细节不外泄,成功即照常返回 Ticket。全部撞号
    则抛 RuntimeError。commit 归调用方。
    """
    prefix = "T" + date.today().strftime("%Y%m%d")
    for attempt in range(1, _TICKET_MAX_ATTEMPTS + 1):
        # 前缀只有字母数字,不含 LIKE 元字符,尾部的 % 是真实通配
        count = await session.scalar(
            select(func.count())
            .select_from(Ticket)
            .where(Ticket.ticket_no.like(prefix + "%"))
        )
        ticket_no = f"{prefix}{(count or 0) + attempt:03d}"
        ticket = Ticket(
            ticket_no=ticket_no,
            conversation_id=conversation_id,
            description=description,
            ticket_type=ticket_type,
        )
        try:
            # savepoint 内完成 INSERT:撞号只回滚本次候选,不污染外层事务
            async with session.begin_nested():
                session.add(ticket)
                await session.flush()
        except IntegrityError:
            continue  # savepoint 已随上下文异常自动回滚,作废候选号换下一个
        return ticket
    raise RuntimeError(
        f"工单号生成冲突:当日 {prefix} 序号重试 {_TICKET_MAX_ATTEMPTS} 次仍撞号"
    )
