r"""Markdown 结构感知切分器:标题建 section 树 → 段落/句子两级切块 → 表格/围栏原子处理。

约定(spec §5.2 / §5.1):
- 标题行 #{1,3} 切 section,section_path = "根分类 > 章 > 节"("> "连接,根分类在首位);
- 代码围栏(```)内不参与边界判定:围栏内句号不作句边界、# 不作标题,围栏整体一块;
- 超长正文两级切分:先按空行分段,段仍超长按句切(句边界集 = 。!?!?! 与换行),
  拼块不超 max_chars;相邻块留 ~overlap_chars 重叠,重叠起点回退到最近句边界
  (保证完整成句),找不到句边界(超长无句读)则硬切;重叠塞不下时放弃重叠,
  保 ≤ max_chars 硬约束;
- 表格(连续 | 开头行,含分隔行)为原子单元:整表不超长则整体入块,超长按数据行
  分块,每块首两行复制表头行 + 分隔行;
- FAQ 型(content_type='faq')节首行「- 问法: A / B / C」提取进 questions 并从
  answer 剔除;policy/manual 该行按普通正文保留;
- category 语义:faq = 文档根分类;policy/manual = 上级标题路径(根节退回根分类);
  questions:faq = 节标题+变体(换行分隔),policy/manual = 所在节标题;
- is_key_clause:section_path 精确命中 doc.key_clauses;
- 纯逻辑零依赖:不 import pymilvus/sqlalchemy,不碰网络与文件系统。
"""

import re
from dataclasses import dataclass


@dataclass
class KnowledgeDoc:
    """语料文档(corpus 侧产出):body 为去 front-matter 后的整篇正文。"""

    title: str
    category: str  # 文档根分类:section_path 首位,faq 块的 category 基底
    content_type: str  # faq / policy / manual
    key_clauses: list[str]  # 关键条款节路径列表,命中则 is_key_clause=True
    body: str


@dataclass
class KnowledgeChunk:
    """向量块:category/questions/answer 三格拼向量文本,其余为入库检索元数据。"""

    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: bool


@dataclass
class _Section:
    """切分中间态:一个 section 的完整标题路径与自有正文(不含子标题行)。"""

    path: list[str]  # [根分类, 章, 节, ...],根分类恒在首位
    text: str  # 自有正文,首尾空白已剥


_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
# 问法行约定:节首行「- 问法: A / B / C」,半角/全角冒号均可,变体按 / 切分
_VARIANT_LINE_RE = re.compile(r"^-\s*问法[:：]\s*(.+?)\s*$")
# 句边界字符集:句号、全半角叹号问号与换行(spec §5.2)
_SENTENCE_BOUNDARIES = "。!?!?!\n"


def chunk_document(
    doc: KnowledgeDoc, *, max_chars: int = 500, overlap_chars: int = 80
) -> list[KnowledgeChunk]:
    """把 KnowledgeDoc 切成向量块列表(纯函数,无副作用)。

    各 section 独立切块、保序;标题下无正文的 section 不产出块,空 body 返回 []。
    """
    chunks: list[KnowledgeChunk] = []
    for section in _parse_sections(doc.body, doc.category):
        chunks.extend(_chunk_section(section, doc, max_chars, overlap_chars))
    return chunks


def build_vector_text(category: str, questions: str, answer: str) -> str:
    """向量文本固定三格拼接:分类/问/答各占一行(半角冒号,与 query_faq 展示格式一致)。"""
    return f"分类:{category}\n问:{questions}\n答:{answer}"


# ---- section 树构建 ----


def _parse_sections(body: str, category: str) -> list[_Section]:
    r"""按标题行(#{1,3})把正文切成平铺的 section 序列,每个记完整路径与自有正文。

    扫描线语义:遇标题先闭合当前 section,再按层级弹栈重建祖先链——父节正文止于
    首个子标题;首个标题前的内容归根节(path == [category]);四级及更深标题按普通
    正文处理;围栏内的 # 不是标题。纯空白 section 不产出。
    """
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []  # (层级, 标题) 尚未被更浅标题闭合的祖先链
    buf: list[str] = []
    in_fence = False

    def _flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append(_Section(path=[category] + [t for _, t in stack], text=text))
        buf.clear()

    for line in body.split("\n"):
        if in_fence:
            buf.append(line)
            if line.lstrip().startswith("```"):
                in_fence = False
            continue
        if line.lstrip().startswith("```"):
            in_fence = True
            buf.append(line)
            continue
        m = _HEADING_RE.match(line)
        if m:
            _flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
        else:
            buf.append(line)
    _flush()
    return sections


def _chunk_section(
    section: _Section, doc: KnowledgeDoc, max_chars: int, overlap_chars: int
) -> list[KnowledgeChunk]:
    """单个 section → 块列表:先按 content_type 定 category/questions,再切 answer。"""
    section_path = " > ".join(section.path)
    title = section.path[-1] if len(section.path) > 1 else ""  # 根节(标题前正文)无节标题
    text = section.text
    if doc.content_type == "faq":
        text, variants = _extract_question_variants(text)
        questions = "\n".join(([title] if title else []) + variants)
        category = doc.category
    else:
        questions = title
        # policy/manual:category = 上级标题路径;根节无上级,退回文档根分类
        category = " > ".join(section.path[:-1]) or doc.category

    answers: list[str] = []
    for kind, segment in _extract_segments(text):
        if kind == "prose":
            answers.extend(_split_long(segment, max_chars, overlap_chars))
        elif kind == "table":
            answers.extend(_split_table(segment, max_chars))
        else:  # fence:整体原子,即使超长也不切
            answers.append(segment)

    return [
        KnowledgeChunk(
            category=category,
            questions=questions,
            answer=answer,
            section_path=section_path,
            content_type=doc.content_type,
            is_key_clause=section_path in doc.key_clauses,
        )
        for answer in answers
    ]


def _extract_segments(text: str) -> list[tuple[str, str]]:
    r"""把节正文切成 (kind, 原文) 段序列,kind ∈ prose / table / fence,保序。

    table = 连续 | 开头行(空行即断);fence = ``` 开闭对(未闭合到文末);
    其余行为 prose。三段彼此独立切块,不做跨段拼块。
    """
    segments: list[tuple[str, str]] = []
    prose: list[str] = []
    table: list[str] = []
    fence: list[str] | None = None

    def _flush_prose() -> None:
        if prose:
            joined = "\n".join(prose).strip()
            if joined:
                segments.append(("prose", joined))
            prose.clear()

    def _flush_table() -> None:
        if table:
            segments.append(("table", "\n".join(table)))
            table.clear()

    for line in text.split("\n"):
        if fence is not None:
            fence.append(line)
            if line.lstrip().startswith("```"):
                segments.append(("fence", "\n".join(fence)))
                fence = None
            continue
        stripped = line.lstrip()
        if stripped.startswith("```"):
            _flush_prose()
            _flush_table()
            fence = [line]
        elif stripped.startswith("|"):
            _flush_prose()
            table.append(line)
        else:
            _flush_table()
            prose.append(line)
    _flush_prose()
    _flush_table()
    if fence is not None:  # 未闭合围栏:剩余内容整体一段
        segments.append(("fence", "\n".join(fence)))
    return segments


# ---- 块切分(prose 两级 + 重叠) ----


def _split_paragraphs(text: str) -> list[str]:
    r"""按空行(含纯空白行)分段;段首尾空白剥除,段内换行保留。"""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _sentence_units(text: str, max_chars: int) -> list[str]:
    """按句边界切句(边界字符留在句尾);超长句(无句读)硬切成 ≤ max_chars 片段。

    全部片段按序拼接还原原文。
    """
    units: list[str] = []
    start = 0
    for i, ch in enumerate(text):
        if ch in _SENTENCE_BOUNDARIES:
            units.append(text[start : i + 1])
            start = i + 1
    if start < len(text):
        units.append(text[start:])

    out: list[str] = []
    for unit in units:
        while len(unit) > max_chars:
            out.append(unit[:max_chars])
            unit = unit[max_chars:]
        if unit:
            out.append(unit)
    return out


def _overlap_tail(chunk: str, overlap_chars: int) -> str:
    """取相邻块重叠尾段:目标长 overlap_chars,起点回退到最近句边界之后(完整成句)。

    不足 overlap_chars 的短块整块作重叠;窗口内找不到句边界则硬切(不留则已,留则整句)。
    """
    if overlap_chars <= 0:
        return ""
    if len(chunk) <= overlap_chars:
        return chunk
    cut = len(chunk) - overlap_chars
    nearest = max(chunk.rfind(ch, 0, cut) for ch in _SENTENCE_BOUNDARIES)
    if nearest < 0:
        return chunk[cut:]  # 超长无句读:硬切
    return chunk[nearest + 1 :]


def _split_long(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """prose 两级切分装箱:空行分段,段超长再按句切;相邻块带句边界回退重叠。

    段与段粘合补回空行分隔;重叠 + 下一段塞不下 max_chars 时放弃重叠。
    """
    chunks: list[str] = []
    current = ""
    for para in _split_paragraphs(text):
        units = [para] if len(para) <= max_chars else _sentence_units(para, max_chars)
        for offset, unit in enumerate(units):
            # offset==0 且块内已有内容:与前面内容隔空行(段界);同段句子直接续
            joiner = "\n\n" if current and offset == 0 else ""
            if current and len(current) + len(joiner) + len(unit) > max_chars:
                chunks.append(current)
                seed = _overlap_tail(current, overlap_chars)
                if not seed or len(seed) + len(joiner) + len(unit) > max_chars:
                    seed, joiner = "", ""  # 重叠塞不下:放弃重叠,直启新块
                current = seed + joiner + unit
            else:
                current += joiner + unit
    if current:
        chunks.append(current)
    return chunks


# ---- 表格处理 ----


def _split_table(table: str, max_chars: int) -> list[str]:
    """表格原子切分:整表 ≤ max_chars 原样一块;超长按数据行装箱,每块首两行复制表头。

    单个数据行 + 表头仍放不下时整行入块(行不可再切),该块可超 max_chars。
    """
    lines = table.split("\n")
    if len(table) <= max_chars:
        return [table]
    head = "\n".join(lines[:2])  # 表头行 + |---| 分隔行
    blocks: list[str] = []
    current = head
    for row in lines[2:]:
        candidate = current + "\n" + row
        if len(candidate) > max_chars and current != head:
            blocks.append(current)
            current = head + "\n" + row
        else:
            current = candidate
    if current:
        blocks.append(current)
    return blocks


# ---- 问法行提取(FAQ 约定) ----


def _extract_question_variants(text: str) -> tuple[str, list[str]]:
    """抽 FAQ 节首行「- 问法: A / B / C」→ (剔除该行后的正文, 变体列表)。

    仅当节的首个非空行是问法行才生效(约定问法行恒居节首);变体按 / 与 ／
    切分,空白变体丢弃;非 faq 或无问法行时原样返回。
    """
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        m = _VARIANT_LINE_RE.match(line.strip())
        if m:
            variants = [v.strip() for v in re.split(r"[/／]", m.group(1)) if v.strip()]
            del lines[idx]
            return "\n".join(lines).strip(), variants
        break  # 首个非空行不是问法行:整节不抽
    return text, []
