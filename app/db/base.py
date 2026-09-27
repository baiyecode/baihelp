"""ORM 声明基类。"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """全部 ORM 模型的声明基类。

    仅测试用 SQLite 内存库跑 ``Base.metadata.create_all``;
    MySQL 端建表唯一依据是用户 DDL,应用启动绝不 create_all。
    """
