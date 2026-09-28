"""corpus 加载器测试:front-matter 手写解析钉行为,真实语料跑 chunker 钉判据词。"""

import re
from pathlib import Path

import pytest

from app.knowledge.chunker import KnowledgeDoc, chunk_document
from app.knowledge.corpus import load_corpus, parse_front_matter

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"


# ---- parse_front_matter(内联样例钉解析行为) ----


def test_parses_scalar_and_indented_list_keys() -> None:
    """三键样例:key: value 单值两枚 + key: 缩进列表;正文不含 front-matter 围栏。"""
    text = (
        "---\n"
        "category: 售后服务\n"
        "content_type: policy\n"
        "key_clauses:\n"
        "  - 售后服务 > 退货政策 > 退货时效\n"
        "  - 售后服务 > 退货政策 > 运费说明\n"
        "---\n"
        "# 退货政策\n"
        "签收后七天无理由退货。"
    )

    meta, body = parse_front_matter(text)

    assert meta == {
        "category": "售后服务",
        "content_type": "policy",
        "key_clauses": [
            "售后服务 > 退货政策 > 退货时效",
            "售后服务 > 退货政策 > 运费说明",
        ],
    }
    assert body == "# 退货政策\n签收后七天无理由退货。"
    assert "---" not in body  # 围栏行不落入正文


def test_text_without_front_matter_returns_empty_meta() -> None:
    """不以 --- 开头的文本视为无 front-matter:meta 为空 dict,原文原样返回。"""
    text = "# 标题\n正文。"

    meta, body = parse_front_matter(text)

    assert meta == {}
    assert body == text


def test_unclosed_front_matter_raises_value_error() -> None:
    """有开无闭的 --- 围栏是格式错误:raise ValueError,不静默吞正文。"""
    text = "---\ncategory: 售后服务\n\n# 标题\n正文。"

    with pytest.raises(ValueError, match="front-matter"):
        parse_front_matter(text)


def test_fullwidth_colon_value_and_unindented_list_item_accepted() -> None:
    """全角冒号单值与未缩进列表项均可解析(手写解析保持宽容)。"""
    text = "---\ncategory:商品咨询\nkey_clauses:\n- 商品咨询 > 发票\n---\n正文。"

    meta, body = parse_front_matter(text)

    assert meta == {"category": "商品咨询", "key_clauses": ["商品咨询 > 发票"]}
    assert body == "正文。"


# ---- load_corpus(必需要键、文件名序、title 取主干) ----


def test_load_corpus_reads_md_files_sorted_by_filename(tmp_path: Path) -> None:
    """按文件名排序加载 .md;title 取文件名主干;非 .md 文件忽略。"""
    (tmp_path / "退货政策.md").write_text(
        "---\ncategory: 售后服务\ncontent_type: policy\nkey_clauses:\n"
        "  - 售后服务 > 退货政策 > 退货时效\n---\n# 退货政策\n七天无理由。",
        encoding="utf-8",
    )
    (tmp_path / "商品FAQ.md").write_text(
        "---\ncategory: 商品咨询\ncontent_type: faq\nkey_clauses:\n---\n"
        "### 发货时效\n- 问法: 什么时候发货 / 几天发货\n48 小时内发货。",
        encoding="utf-8",
    )
    (tmp_path / "备注.txt").write_text("不是语料。", encoding="utf-8")

    docs = load_corpus(tmp_path)

    assert [d.title for d in docs] == ["商品FAQ", "退货政策"]  # Unicode 码点排序
    policy = docs[1]
    assert policy.category == "售后服务"
    assert policy.content_type == "policy"
    assert policy.key_clauses == ["售后服务 > 退货政策 > 退货时效"]
    assert policy.body == "# 退货政策\n七天无理由。"
    faq = docs[0]
    assert faq.content_type == "faq"


@pytest.mark.parametrize("missing", ["category", "content_type", "key_clauses"])
def test_load_corpus_missing_required_key_raises(tmp_path: Path, missing: str) -> None:
    """必需键(category/content_type/key_clauses)缺失 → ValueError 且指明文件名。"""
    lines = ["---"]
    if missing != "category":
        lines.append("category: 售后服务")
    if missing != "content_type":
        lines.append("content_type: policy")
    if missing != "key_clauses":
        lines += ["key_clauses:", "  - 售后服务 > 退货政策 > 退货时效"]
    lines += ["---", "# 文档", "正文。"]
    path = tmp_path / "坏文档.md"
    path.write_text("\n".join(lines), encoding="utf-8")

    with pytest.raises(ValueError, match="坏文档"):
        load_corpus(tmp_path)


def test_load_corpus_rejects_unknown_content_type(tmp_path: Path) -> None:
    """content_type 不属于 faq/policy/manual → ValueError(坏类型不进库)。"""
    (tmp_path / "怪文档.md").write_text(
        "---\ncategory: 商品咨询\ncontent_type: novel\nkey_clauses:\n---\n正文。",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="content_type"):
        load_corpus(tmp_path)


def test_load_corpus_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    """空目录返回空列表(挖矿场景 corpus_dir 无新增语料时 Phase A 零插入)。"""

    assert load_corpus(tmp_path) == []


# ---- 三份真实语料的完整性与判据词(data/knowledge/ 真文件) ----


def _load_docs() -> dict[str, KnowledgeDoc]:
    """加载真实语料目录,按 title 建索引(空索引视为语料缺失,直接失败)。"""
    docs = load_corpus(CORPUS_DIR)
    assert docs, f"语料目录缺失或为空:{CORPUS_DIR}"
    return {d.title: d for d in docs}


def test_corpus_files_chunk_cleanly() -> None:
    """三份语料一次性过 chunk_document(默认 500/80):块数 >10 且每块问答非空。"""
    docs = _load_docs()
    assert set(docs) == {"退货政策", "商品FAQ", "售后手册"}
    all_chunks = [chunk for doc in docs.values() for chunk in chunk_document(doc)]

    assert len(all_chunks) > 10
    assert all(c.answer.strip() for c in all_chunks), "存在空 answer 块"
    assert all(c.questions.strip() for c in all_chunks), "存在空 questions 块"


def test_refund_policy_content_and_judge_words() -> None:
    """退货政策:policy/售后服务;退货时效答含「七天」且命中 key_clauses;运费说明答含运费判据。"""
    policy = _load_docs()["退货政策"]
    assert policy.category == "售后服务"
    assert policy.content_type == "policy"
    assert "售后服务 > 退货政策 > 退货时效" in policy.key_clauses

    by_path = {c.section_path: c for c in chunk_document(policy)}
    assert "售后服务 > 退货政策 > 退货时效" in by_path
    assert "售后服务 > 退货政策 > 运费说明" in by_path
    assert "售后服务 > 退货政策 > 责任划分" in by_path
    assert "售后服务 > 退货政策 > 退货流程" in by_path

    aging = by_path["售后服务 > 退货政策 > 退货时效"]
    assert "七天" in aging.answer  # acceptance ⑤ 判据词
    assert aging.is_key_clause is True

    shipping = by_path["售后服务 > 退货政策 > 运费说明"]
    assert "满 99 元包邮" in shipping.answer  # acceptance ① 判据「包邮」「99」
    assert "运费 8 元" in shipping.answer
    assert "12 元" in shipping.answer  # 偏远地区

    responsibility = by_path["售后服务 > 退货政策 > 责任划分"]
    assert "质量" in responsibility.answer


def test_product_faq_structure_and_variants() -> None:
    """商品FAQ:faq/商品咨询;≥6 个 ### 节且每节首行为问法行;覆盖六主题;不含邮费/运费词条。"""
    faq = _load_docs()["商品FAQ"]
    assert faq.category == "商品咨询"
    assert faq.content_type == "faq"

    titles = re.findall(r"(?m)^###\s+(.+?)\s*$", faq.body)
    assert len(titles) >= 6
    for keyword in ("发货", "支付", "发票", "优惠券", "积分", "换货"):
        assert any(keyword in t for t in titles), f"缺 {keyword} 主题节"

    # 每节首行(节标题后第一个非空行)必须是「- 问法: A / B / C」
    for title in titles:
        section = re.search(
            rf"(?ms)^### {re.escape(title)}\n(.*?)(?=^### |\Z)", faq.body
        )
        assert section, f"节 {title} 未匹配到正文"
        first_line = next(
            line for line in section.group(1).split("\n") if line.strip()
        )
        assert re.match(r"-\s*问法[:：]", first_line.strip()), f"{title} 首行非问法行"

    # 保留 ch02 种子语义对照:节标题与问法变体不得出现「邮费」「运费」
    forbidden = re.findall(r"(?m)^### .*$|^- 问法[:：].*$", faq.body)
    assert forbidden
    assert all("邮费" not in line and "运费" not in line for line in forbidden)

    # questions = 节标题 + ≥2 个变体(共 ≥3 行),answer 非空
    chunks = chunk_document(faq)
    assert len(chunks) == len(titles)  # 每节一块(答案均不超长)
    assert all(len(c.questions.split("\n")) >= 3 for c in chunks)
    assert all(c.answer.strip() for c in chunks)


def test_after_sales_manual_structure() -> None:
    """售后手册:manual/售后服务;≥3 级标题深度;≥6 数据行物流表;>500 字符长节切多块。"""
    manual = _load_docs()["售后手册"]
    assert manual.category == "售后服务"
    assert manual.content_type == "manual"

    chunks = chunk_document(manual)

    # ≥3 级标题深度:# > ## > ### → section_path 至少 4 段(根分类起算)
    assert any(c.section_path.count(" > ") >= 3 for c in chunks)

    # 物流时效表:原始正文含 ≥6 个数据行(表头 + 分隔行之外),且确实进了块
    table_lines = [ln for ln in manual.body.split("\n") if ln.lstrip().startswith("|")]
    assert len(table_lines) >= 8  # 表头 1 + 分隔 1 + 数据 ≥6
    assert any("|---|" in c.answer for c in chunks)

    # >500 字符长节:被递归切成多块,每块 ≤ max_chars,相邻块首句重叠
    match = re.search(r"(?ms)^## 保修条款\n(.*?)(?=^#{1,3} |\Z)", manual.body)
    assert match, "缺「保修条款」节"
    assert len(match.group(1)) > 500

    warranty = [c for c in chunks if c.section_path.endswith("保修条款")]
    assert len(warranty) >= 2
    assert all(len(c.answer) <= 500 for c in warranty)
    head = warranty[1].answer.split("。", 1)[0] + "。"
    assert head in warranty[0].answer  # 重叠起点为完整原句
