"""语料加载:解析 data/knowledge/*.md 首部 front-matter 并构建 KnowledgeDoc。

只做解析与加载,不做切分(切分是 chunker 的事);手写解析,不引 pyyaml。
front-matter 约定(spec §5.1):首行 ``---`` 开、单独一行 ``---`` 闭,支持
``key: value`` 单值与 ``key:`` 缩进列表两式;必需键 category / content_type /
key_clauses 缺失即 ValueError,坏内容不进库。
"""

import re
from pathlib import Path

from app.knowledge.chunker import KnowledgeDoc

_CLOSING_FENCE = "---"
_REQUIRED_SCALAR_KEYS = ("category", "content_type")
_REQUIRED_LIST_KEY = "key_clauses"
_CONTENT_TYPES = ("faq", "policy", "manual")
# 单值行:key 不含冒号与空白,值可空(空值视为列表键的开始)
_KEY_VALUE_RE = re.compile(r"^([^:\s]+)\s*[:：]\s*(.*)$")
# 列表项行:- 值(与键值行互斥)
_LIST_ITEM_RE = re.compile(r"^-\s+(.+)$")


def parse_front_matter(text: str) -> tuple[dict, str]:
    r"""解析首部 ``---`` 围栏块 → (元数据 dict, 去 front-matter 正文)。

    - 不以 ``---`` 开头:视为无 front-matter,返回 ({}, 原文);
    - 单值 ``key: value``(半角/全角冒号)存 str;``key:`` 空值开启缩进列表,
      后续 ``- item`` 行(缩进可有可无)依次追加;
    - 围栏有开无闭、或出现无法归类的行:raise ValueError;
    - 行分隔统一为 \n(读入时的 \r\n 在此归一,保证 chunker 句边界正常)。
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _CLOSING_FENCE:
        return {}, text

    meta: dict = {}
    list_key: str | None = None  # 当前正在收集缩进列表的键
    closing: int | None = None
    for idx in range(1, len(lines)):
        line = lines[idx]
        if line.strip() == _CLOSING_FENCE:
            closing = idx
            break
        stripped = line.strip()
        if not stripped:
            continue
        item = _LIST_ITEM_RE.match(stripped)
        if item and list_key is not None:
            meta[list_key].append(item.group(1).strip())
            continue
        kv = _KEY_VALUE_RE.match(stripped)
        if kv:
            key, value = kv.group(1), kv.group(2).strip()
            if value:
                meta[key] = value
                list_key = None
            else:
                meta[key] = []
                list_key = key
            continue
        raise ValueError(f"front-matter 第 {idx + 1} 行无法解析:{line!r}")
    if closing is None:
        raise ValueError("front-matter 未闭合(缺少收尾 ---)")
    return meta, "\n".join(lines[closing + 1 :])


def load_corpus(directory: Path) -> list[KnowledgeDoc]:
    """按文件名排序读取目录下全部 .md 语料 → KnowledgeDoc 列表。

    title 取文件名主干;category / content_type / key_clauses 三个必需键缺失
    (或 content_type 不属 faq/policy/manual)即 ValueError 并指明文件名。
    """
    docs: list[KnowledgeDoc] = []
    for path in sorted(directory.glob("*.md"), key=lambda p: p.name):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        for key in _REQUIRED_SCALAR_KEYS:
            value = meta.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{path.name}: front-matter 缺少必需键 {key}")
        if not isinstance(meta.get(_REQUIRED_LIST_KEY), list):
            raise ValueError(
                f"{path.name}: front-matter 缺少必需键 {_REQUIRED_LIST_KEY}(缩进列表)"
            )
        content_type = meta["content_type"]
        if content_type not in _CONTENT_TYPES:
            raise ValueError(
                f"{path.name}: content_type {content_type!r} 不属于 {'/'.join(_CONTENT_TYPES)}"
            )
        docs.append(
            KnowledgeDoc(
                title=path.stem,
                category=meta["category"],
                content_type=content_type,
                key_clauses=meta[_REQUIRED_LIST_KEY],
                body=body,
            )
        )
    return docs
