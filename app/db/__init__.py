"""ch02 数据层(应用映射 + 异步 engine)。

建表唯一依据是用户 DDL(scripts/sql/ch02-ddl.sql,MySQL 端由容器 initdb 执行);
ORM 只做应用映射,禁止对 MySQL 跑 create_all(仅测试用 SQLite 内存库 create_all)。
"""

from app.db.base import Base
from app.db import base, engine, models  # noqa: F401  子模块随包导入,models 导入即注册元数据

__all__ = ["Base", "base", "engine", "models"]
