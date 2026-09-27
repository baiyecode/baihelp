"""ch02 四张表的 ORM 映射,逐列对齐用户 DDL(scripts/sql/ch02-ddl.sql)。

约定:
- 仅做应用映射,MySQL 端建表以 DDL 为准(索引/外键命名以 DDL 文件为准,ORM 不必同名);
- 枚举一律用值本身(中文值原样),不做 label 转换;SQLite 上自动落 VARCHAR+CHECK;
- 不 import 任何 MySQL 方言类型,保证测试库可跨 SQLite 跑;
- SQLite 索引名全库全局(MySQL 按表隔离),故 DDL 里两张表重名的
  idx_conversation_id 在 ORM 侧加表名前缀消歧,列目标不变。
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

# DDL 主键是 BIGINT 自增;SQLite 只认「INTEGER PRIMARY KEY」为 rowid 别名(才自增),
# 故仅主键列对 sqlite 方言放宽为 Integer,其余库仍按 BigInteger 渲染
_PK_BIGINT = BigInteger().with_variant(Integer(), "sqlite")


class Conversation(Base):
    """客服会话壳:一通对话的统一身份,DDL 表 conversations。"""

    __tablename__ = "conversations"
    __table_args__ = (Index("idx_user_id", "user_id"),)

    id: Mapped[int] = mapped_column(_PK_BIGINT, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        Enum("进行中", "已转人工", "已结束", name="conversation_status"),
        default="进行中",  # Python 侧默认:flush 即赋值,提交后立即可读
        server_default="进行中",  # 服务端默认:与 DDL 的 DEFAULT 子句对齐
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), server_onupdate=func.now()
    )


class Message(Base):
    """会话消息流水:role 对齐 Chat Completions 协议,DDL 表 messages。"""

    __tablename__ = "messages"
    __table_args__ = (Index("idx_messages_conversation_id", "conversation_id"),)

    id: Mapped[int] = mapped_column(_PK_BIGINT, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id", name="fk_messages_conversation")
    )
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool", name="message_role"))
    # assistant 纯工具调用时正文可为空
    content: Mapped[str | None] = mapped_column(Text)
    # assistant 消息带的工具调用申请单(JSON 往返)
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)
    # tool 消息对应的申请单 id,回灌时对号入座
    tool_call_id: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Faq(Base):
    """FAQ 问答对:query_faq 的数据源,DDL 表 faq。"""

    __tablename__ = "faq"
    __table_args__ = (Index("idx_category", "category"),)

    id: Mapped[int] = mapped_column(_PK_BIGINT, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), server_onupdate=func.now()
    )


class Ticket(Base):
    """人工工单:工单号当业务主键,DDL 表 tickets。"""

    __tablename__ = "tickets"
    __table_args__ = (Index("idx_tickets_conversation_id", "conversation_id"),)

    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id", name="fk_tickets_conversation")
    )
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[str] = mapped_column(Enum("售后", "投诉", "咨询", name="ticket_type"))
    status: Mapped[str] = mapped_column(
        Enum("待处理", "已处理", name="ticket_status"),
        default="待处理",  # Python 侧默认:flush 即赋值,提交后立即可读
        server_default="待处理",  # 服务端默认:与 DDL 的 DEFAULT 子句对齐
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
