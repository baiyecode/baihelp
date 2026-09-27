"""异步 engine / 会话工厂 / 连通性探活。

启动缝(consumes):Settings.database_url 来自 app.core.config。
lifespan 用法:engine = build_engine(...) → await ping(engine) → 关停 dispose;
绝不在此跑 create_all(MySQL 建表唯一依据是用户 DDL)。
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def build_engine(database_url: str) -> AsyncEngine:
    """按 URL 构建异步 engine(MySQL 走 aiomysql,测试走 aiosqlite)。"""
    return create_async_engine(database_url)


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """构建会话工厂;expire_on_commit=False,提交后属性仍可直接读。"""
    return async_sessionmaker(engine, expire_on_commit=False)


async def ping(engine: AsyncEngine) -> None:
    """``SELECT 1`` 探活;失败抛 RuntimeError 并附「docker compose up -d」操作提示。"""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise RuntimeError(
            f"数据库连接失败:{exc}。请先 docker compose up -d 并执行建表 DDL"
            "(scripts/sql/ch02-ddl.sql)"
        ) from exc
