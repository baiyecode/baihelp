"""DDL↔ORM 一致性冒烟测试:防「用户 DDL」与「ORM 映射」两处定义漂移。

一致性面 = 表名 / 列名 / 主键 / 枚举值集合;本文件只比列名集合
(索引与外键命名以 DDL 文件为准,ORM 不必同名)。
MySQL 端建表唯一依据是 scripts/sql/ch02-ddl.sql,ORM 只做应用映射。
"""

import re
from pathlib import Path

import pytest

from app.db import models  # noqa: F401  导入即把四张表注册进 Base.metadata
from app.db.base import Base

DDL_PATH = Path(__file__).resolve().parent.parent / "scripts" / "sql" / "ch02-ddl.sql"

# 本章四张表(建表顺序:先 conversations,再依赖它的 messages / tickets)
CHAPTER_TABLES = ("conversations", "messages", "faq", "tickets")

# 匹配「CREATE TABLE <名> ( … ) ENGINE=…」整块;表体用非贪婪截到 ) ENGINE
_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(\w+)\s*\((.*?)\)\s*ENGINE", re.IGNORECASE | re.DOTALL
)
# 表体里非列定义行的首关键字(主键 / 索引 / 外键约束行)
_NON_COLUMN_KEYWORDS = frozenset({"PRIMARY", "KEY", "CONSTRAINT", "UNIQUE", "FOREIGN"})


def _ddl_column_names() -> dict[str, set[str]]:
    """正则解析 DDL 文件,返回「表名 -> 列名集合」。"""
    sql = DDL_PATH.read_text(encoding="utf-8")
    tables: dict[str, set[str]] = {}
    for match in _CREATE_TABLE_RE.finditer(sql):
        table_name, body = match.group(1), match.group(2)
        columns: set[str] = set()
        for raw_line in body.splitlines():
            line = raw_line.strip().rstrip(",")
            if not line:
                continue
            first_token = line.split()[0]
            if first_token.upper() in _NON_COLUMN_KEYWORDS:
                continue
            columns.add(first_token)
        tables[table_name] = columns
    return tables


def test_ddl_defines_exactly_the_four_chapter_tables() -> None:
    """DDL 文件应且仅应包含本章四张表。"""
    assert set(_ddl_column_names()) == set(CHAPTER_TABLES)


def test_orm_metadata_defines_exactly_the_four_chapter_tables() -> None:
    """ORM 元数据应且仅应包含本章四张表(无杂表)。"""
    assert set(Base.metadata.tables) == set(CHAPTER_TABLES)


@pytest.mark.parametrize("table_name", CHAPTER_TABLES)
def test_column_name_sets_match_ddl(table_name: str) -> None:
    """每张表的 ORM 列名集合与 DDL 列名集合逐一相等。"""
    ddl_columns = _ddl_column_names()[table_name]
    orm_columns = {column.name for column in Base.metadata.tables[table_name].columns}
    assert orm_columns == ddl_columns
