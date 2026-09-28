"""chunk_document 结构感知切分测试:逐条钉 spec §5.2 规则(路径/递归/重叠/表格/问法/围栏)。"""

from app.knowledge.chunker import (
    KnowledgeDoc,
    build_vector_text,
    chunk_document,
)


def _doc(body: str, content_type: str = "policy", **kw) -> KnowledgeDoc:
    return KnowledgeDoc(title="测试文档", category="测试分类", content_type=content_type,
                        key_clauses=kw.get("key_clauses", []), body=body)


def test_hierarchical_sections_carry_section_path() -> None:
    """# 售后 下 ## 退货 的块:section_path 以 > 连接、根分类在首位。"""
    body = "# 售后\n本章介绍售后规则。\n\n## 退货\n签收后七天无理由退货。"
    chunks = chunk_document(_doc(body))

    by_path = {c.section_path: c for c in chunks}
    assert "测试分类 > 售后 > 退货" in by_path
    assert "签收后七天无理由退货。" in by_path["测试分类 > 售后 > 退货"].answer
    assert "测试分类 > 售后" in by_path  # 父节自有正文(简介)路径止于章


def test_long_section_recursed_by_paragraph_then_sentence() -> None:
    """>500 字符单段被按句拆成多块,每块 ≤ max_chars。"""
    body = "".join(f"这是第{i}句话,用来撑起超过五百字符的长段落。" for i in range(30))
    assert len(body) > 500

    chunks = chunk_document(_doc(body))

    assert len(chunks) > 1
    assert all(len(c.answer) <= 500 for c in chunks)


def test_overlap_starts_at_sentence_boundary() -> None:
    """相邻块重叠段起点在句号之后:后块开头必是完整原句,且该句也出现在前块中。"""
    sentences = [f"重叠校验第{i}句,内容各不相同且完整成句。" for i in range(30)]
    chunks = chunk_document(_doc("".join(sentences)))

    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        head = nxt.answer.split("。", 1)[0] + "。"
        assert head in sentences  # 完整成句,不以半个句子开头
        assert head in prev.answer  # 与前块形成重叠


def test_short_section_single_chunk_no_overlap() -> None:
    """不超长的节恰好一块,answer 即原文,无重叠副本。"""
    body = "订单签收后七天内可无理由退货,超期需联系客服。"
    chunks = chunk_document(_doc(body))

    assert len(chunks) == 1
    assert chunks[0].answer == body


def test_long_table_blocks_repeat_header() -> None:
    """超长表格按行分块,每块首两行 == 表头行 + |---| 分隔行,数据行不丢不重。"""
    rows = ["| 商品 | 价格 |", "|---|---|"]
    rows += [f"| 商品{i} | {i}0 元 |" for i in range(6)]
    table = "\n".join(rows)
    head = "\n".join(rows[:2])
    # 恰好容纳 表头两行 + 3 个数据行 的 max_chars(整表 8 行被压成 3 数据行/块)
    max_chars = len(head) + 3 * (len(rows[2]) + 1)

    chunks = chunk_document(_doc(table), max_chars=max_chars)

    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.answer.split("\n")[:2] == rows[:2]
        assert len(chunk.answer) <= max_chars
    assert [c.answer.split("\n")[2:] for c in chunks] == [rows[2:5], rows[5:8]]


def test_short_table_atomic() -> None:
    """不超长的表整体一块,原文逐字保留。"""
    table = "| 商品 | 价格 |\n|---|---|\n| 袜子 | 5 元 |"
    chunks = chunk_document(_doc(table))

    assert len(chunks) == 1
    assert chunks[0].answer == table


def test_faq_question_variants_extracted() -> None:
    """faq:首行「- 问法: A / B / C」进 questions 且该行从 answer 剔除;
    policy:同样内容整行留在 answer,questions == 节标题。"""
    body = (
        "### 优惠券使用\n"
        "- 问法: 怎么用券 / 优惠券在哪里 / 券过期了怎么办\n"
        "满99减10,下单自动抵扣。"
    )

    faq = chunk_document(_doc(body, content_type="faq"))[0]
    assert faq.questions == "优惠券使用\n怎么用券\n优惠券在哪里\n券过期了怎么办"
    assert "问法" not in faq.answer
    assert "满99减10,下单自动抵扣。" in faq.answer

    policy = chunk_document(_doc(body))[0]
    assert policy.questions == "优惠券使用"
    assert "- 问法: 怎么用券 / 优惠券在哪里 / 券过期了怎么办" in policy.answer


def test_key_clause_marked() -> None:
    """key_clauses 命中节路径 → is_key_clause True;未命中 False。"""
    body = "# 售后\n总则说明。\n\n## 退货\n七天无理由。"
    chunks = chunk_document(_doc(body, key_clauses=["测试分类 > 售后 > 退货"]))

    flagged = {c.section_path: c.is_key_clause for c in chunks}
    assert flagged["测试分类 > 售后 > 退货"] is True
    assert flagged["测试分类 > 售后"] is False


def test_code_fence_not_split() -> None:
    """围栏内句号不作为切分边界:围栏整体一块,前后正文各自成块。"""
    body = (
        "配置如下。\n"
        "```\n"
        "第一行含句号。第二行也含句号。\n"
        "```\n"
        "配置完毕。"
    )
    chunks = chunk_document(_doc(body))
    answers = [c.answer for c in chunks]

    assert len(chunks) == 3
    assert "```\n第一行含句号。第二行也含句号。\n```" in answers
    assert answers[0] == "配置如下。"
    assert answers[2] == "配置完毕。"


def test_vector_text_format() -> None:
    """build_vector_text 固定三格格式:分类/问/答各一行,冒号为半角。"""
    assert (
        build_vector_text("售后服务", "退货时效\n七天无理由", "签收后七天内可退。")
        == "分类:售后服务\n问:退货时效\n七天无理由\n答:签收后七天内可退。"
    )


def test_empty_body_and_bare_headings_yield_no_chunks() -> None:
    """边界:空 body 与纯标题(无正文)都不产出块。"""
    assert chunk_document(_doc("")) == []
    assert chunk_document(_doc("# 只有标题\n## 还有一级")) == []


def test_punctuation_free_long_text_hard_cut() -> None:
    """边界:无句读长文找不到句边界,硬切成 ≤ max_chars 的块,重叠随之硬切。"""
    body = "无标点长文本" * 100  # 600 字符无任何句读
    chunks = chunk_document(_doc(body))

    assert len(chunks) >= 2
    assert all(len(c.answer) <= 500 for c in chunks)
    assert chunks[0].answer == body[:500]  # 首块为硬切
