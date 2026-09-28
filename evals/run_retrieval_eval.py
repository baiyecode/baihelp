r"""知识检索评估跑分脚本(standalone harness,无 pytest;以标注样例替代 TDD)。

用法：
    uv run python evals/run_retrieval_eval.py             # 真实检索跑分(需先建库 + EMBEDDING_API_KEY)
    uv run python evals/run_retrieval_eval.py --self-test # 离线自测:罐头语料 + 假嵌入/假仓储走通判分/门禁,
                                                          # 不发任何网络请求,不需要 API key / MySQL / Milvus

判分(检索质量不可单测,用标注样例评估集替代单测;数据集契约见 retrieval_cases.jsonl):
    - 每条样例是一个顾客问句;live 用 KnowledgeRetriever 真跑 embed → Milvus 近邻 →
      相似度阈值过滤 → MySQL 回表(done),按 score 降序得到命中列表;
    - hit 样例:命中位置(rank,1 起)<= min_rank 且该条 answer 含 answer_contains
      全部关键词(任一满足判位的命中即可)判对;
    - no_hit 样例:检索无任何 score >= 阈值的命中(命中列表为空)判对。

门禁(binding)两条:
    1. hit 样例全部在 min_rank 内命中且含关键词(召回质量);
    2. no_hit 样例全部零命中(阈值闸门,拒答不胡答)。
全部达标 exit 0,未达标 exit 1(并打印未达标项)。配置/数据错误 exit 2。

前置:live 跑分前先 `uv run python -m app.knowledge.ingest` 建库(语料切块入库 +
向量双写);评估器启动即预检 knowledge_chunks 的 done 行数,为 0 直接给出提示退出 2。
单条检索抛异常(连不上 MySQL/Milvus、嵌入端点报错等)该条判负,继续跑完剩余样例。
数据集为空时门禁真空为真——由 validate_cases(总条数/四条必含样例)拦住。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sqlalchemy import func, select

from app.core.config import get_settings
from app.db.engine import build_engine, build_session_factory
from app.db.models import KnowledgeChunk
from app.knowledge.retriever import KnowledgeRetriever

CASES_PATH = Path(__file__).resolve().parent / "retrieval_cases.jsonl"

MIN_CASES = 10
VALID_EXPECTS: frozenset[str] = frozenset({"hit", "no_hit"})
# brief 钉死的四条必含样例(问句 → 期望标签):含漏召回翻案/换说法/拒答/核心政策
REQUIRED_QUERIES: dict[str, str] = {
    "邮费是多少": "hit",
    "快递费怎么算": "hit",
    "你们老板是谁": "no_hit",
    "怎么退货": "hit",
}


# ---------------------------------------------------------------------------
# 数据加载与校验
# ---------------------------------------------------------------------------

def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    """逐行读取 jsonl 样例集,跳过空行与 # 注释行。"""
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            cases.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} JSON 解析失败: {exc}") from exc
    if not cases:
        raise ValueError(f"{path} 中没有找到任何样例")
    return cases


def validate_cases(cases: list[dict[str, Any]], *, top_k: int | None = None) -> None:
    """校验样例集形状、必含样例与 id/query 唯一性,失败抛 ValueError。

    top_k 传入时额外校验 hit 样例的 min_rank <= top_k(超过则判分永远无法满足)。
    """
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    required_missing = dict(REQUIRED_QUERIES)

    for case in cases:
        cid = str(case.get("id", ""))
        query = case.get("query")
        expect = case.get("expect")
        if not cid or not isinstance(query, str) or not query.strip():
            problems.append(f"{cid or '?'}: 缺少 id/query 或类型不对")
            continue
        if cid in seen_ids:
            problems.append(f"{cid}: id 重复")
        if query in seen_queries:
            problems.append(f"{cid}: query 重复")
        seen_ids.add(cid)
        seen_queries.add(query)
        if expect not in VALID_EXPECTS:
            problems.append(f"{cid}: expect {expect!r} 不在 hit/no_hit 之内")
            continue
        required_missing.pop(query, None)
        if expect == "hit":
            answer_contains = case.get("answer_contains")
            if (
                not isinstance(answer_contains, list)
                or not answer_contains
                or not all(isinstance(key, str) and key.strip() for key in answer_contains)
            ):
                problems.append(f"{cid}: hit 样例 answer_contains 须为非空字符串的非空列表")
            min_rank = case.get("min_rank")
            if isinstance(min_rank, bool) or not isinstance(min_rank, int) or min_rank < 1:
                problems.append(f"{cid}: hit 样例 min_rank 须为 >= 1 的整数")
            elif top_k is not None and min_rank > top_k:
                problems.append(f"{cid}: min_rank {min_rank} > 检索 top_k {top_k},判分永无法满足")
        else:
            if "answer_contains" in case or "min_rank" in case:
                problems.append(f"{cid}: no_hit 样例不得携带 answer_contains/min_rank")
        note = case.get("note")
        if note is not None and not isinstance(note, str):
            problems.append(f"{cid}: note 需为字符串")

    if required_missing:
        problems.append(f"缺少必含样例问句: {sorted(required_missing)}")
    if len(cases) < MIN_CASES:
        problems.append(f"总条数 {len(cases)} < {MIN_CASES}")

    if problems:
        raise ValueError("样例集校验失败:\n" + "\n".join(f"  - {p}" for p in problems))


# ---------------------------------------------------------------------------
# 判分
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """单条样例的检索判分结果。"""

    case_id: str
    query: str
    expect: str
    passed: bool
    detail: str
    error: str | None = None


@dataclass
class Summary:
    """检索评估汇总(对应两条门禁)。"""

    hit_total: int
    hit_passed: int
    nohit_total: int
    nohit_zero: int


def judge_hit(
    hits: list[Any], answer_contains: list[str], min_rank: int
) -> tuple[bool, str]:
    """hit 判定:rank(1 起)<= min_rank 内存在 answer 含全部关键词的命中。"""
    for rank, item in enumerate(hits, start=1):
        if rank > min_rank:
            break
        if all(key in item.answer for key in answer_contains):
            return True, f"rank={rank} score={item.score:.3f} 含全部关键词"
    return (
        False,
        f"前 {min(len(hits), min_rank)} 条命中内无 answer 含全部关键词的条目"
        if hits
        else "零命中",
    )


async def evaluate(
    cases: list[dict[str, Any]],
    retriever: KnowledgeRetriever,
    session_factory: Any,
) -> list[CaseResult]:
    """逐条真跑检索并判分;单条异常只判负该条,不中断整场评估。"""
    results: list[CaseResult] = []
    for case in cases:
        case_id, query = str(case["id"]), str(case["query"])
        expect = str(case["expect"])
        try:
            hits = await retriever.retrieve(query, session_factory)
            error = None
        except Exception as exc:  # noqa: BLE001 - 单条失败不中断整场评估
            hits, error = [], f"{type(exc).__name__}: {exc}"
        if expect == "no_hit":
            detail = "零命中" if not hits else f"{len(hits)} 条命中(期望拒答)"
            passed = error is None and not hits
        elif error is not None:
            passed, detail = False, "检索抛异常"
        else:
            passed, detail = judge_hit(hits, case["answer_contains"], int(case["min_rank"]))
        results.append(CaseResult(case_id, query, expect, passed, detail, error))
    return results


def summarize(results: list[CaseResult]) -> Summary:
    """汇总 hit 达标数与 no_hit 零命中数(异常例不计入达标)。"""
    hit_results = [r for r in results if r.expect == "hit"]
    nohit_results = [r for r in results if r.expect == "no_hit"]
    return Summary(
        hit_total=len(hit_results),
        hit_passed=sum(1 for r in hit_results if r.passed),
        nohit_total=len(nohit_results),
        nohit_zero=sum(1 for r in nohit_results if r.error is None and r.detail == "零命中"),
    )


def gate_check(summary: Summary) -> tuple[bool, list[str]]:
    """两条门禁:hit 全达标 / no_hit 全零命中;返回 (是否全部达标, 未达标项)。"""
    failed: list[str] = []
    if summary.hit_passed != summary.hit_total:
        failed.append("hit_recall")
    if summary.nohit_zero != summary.nohit_total:
        failed.append("no_hit_zero")
    return (not failed), failed


GATE_LABELS: dict[str, str] = {
    "hit_recall": "hit 行全部在 min_rank 内命中且含关键词",
    "no_hit_zero": "no_hit 行零命中",
}


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------

def _mark(ok: bool) -> str:
    return "ok" if ok else "X"


def print_report(results: list[CaseResult], summary: Summary) -> tuple[bool, list[str]]:
    """打印逐条表格、汇总指标与门禁结论;返回门禁结果。"""
    width = max((len(r.case_id) for r in results), default=7)
    query_width = max((len(r.query) for r in results), default=6)
    header = f"{'id':<{width}}  {'expect':<6}  {'query':<{query_width}}  判分  命中详情"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        error_note = "" if r.error is None else f"  ERROR: {r.error}"
        print(
            f"{r.case_id:<{width}}  {r.expect:<6}  {r.query:<{query_width}}  "
            f"{_mark(r.passed):<4}  {r.detail}{error_note}"
        )

    print()
    print(f"hit 命中达标({summary.hit_total} 条): {summary.hit_passed}/{summary.hit_total}")
    print(f"no_hit 零命中({summary.nohit_total} 条): {summary.nohit_zero}/{summary.nohit_total}")

    gate_ok, failed = gate_check(summary)
    details = " | ".join(
        f"{GATE_LABELS[gate]} -> {'达标' if gate not in failed else '未达标'}" for gate in GATE_LABELS
    )
    print(f"门禁: {details}")
    if gate_ok:
        print("门禁结论: PASS")
    else:
        print(f"门禁结论: FAIL(未达标: {', '.join(failed)})")
    return gate_ok, failed


def print_failure_details(results: list[CaseResult]) -> None:
    """对判负样例打印期望与实际命中概况,便于人工定位。"""
    failures = [r for r in results if not r.passed]
    if not failures:
        return
    print("\n失败详情:")
    for r in failures:
        note = f"expect={r.expect}, detail={r.detail!r}"
        if r.error is not None:
            note += f" (ERROR: {r.error})"
        print(f"  - {r.case_id}: {note}")


# ---------------------------------------------------------------------------
# live 预检与主流程
# ---------------------------------------------------------------------------

async def _count_chunks(session_factory: Any) -> tuple[int, int]:
    """knowledge_chunks 总行数与 done 行数(建库预检)。"""
    async with session_factory() as session:
        total = await session.scalar(select(func.count()).select_from(KnowledgeChunk))
        done = await session.scalar(
            select(func.count())
            .select_from(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "done")
        )
    return int(total or 0), int(done or 0)


async def _evaluate_live(settings: Any, cases: list[dict[str, Any]], embedder: Any, repo: Any) -> int:
    """live 主流程:建集合 → 建库预检 → 逐条检索判分 → 两门禁 exit code。"""
    engine = build_engine(settings.database_url)
    try:
        session_factory = build_session_factory(engine)
        repo.ensure_collection()
        total, done = await _count_chunks(session_factory)
        print(f"建库预检:knowledge_chunks 共 {total} 行,其中已向量化(done) {done} 行")
        if done == 0:
            print(
                "[前置未满足] 请先运行 uv run python -m app.knowledge.ingest 建库,再跑检索评估。",
                file=sys.stderr,
            )
            return 2
        retriever = KnowledgeRetriever(
            embedder,
            repo,
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
        )
        results = await evaluate(cases, retriever, session_factory)
        summary = summarize(results)
        gate_ok, _ = print_report(results, summary)
        print_failure_details(results)
        return 0 if gate_ok else 1
    finally:
        await engine.dispose()


def run_live() -> int:
    """真实检索跑分:配置校验 -> 建库预检 -> 逐条检索判分 -> 两门禁 exit code。"""
    try:
        settings = get_settings()
        cases = load_cases()
        validate_cases(cases, top_k=settings.retrieval_top_k)
    except Exception as exc:  # noqa: BLE001 - 配置/数据错误给出可操作的提示
        print(f"[配置/数据错误] {exc}", file=sys.stderr)
        print(
            "请确认 .env(参考 .env.example)已配置 LLM/EMBEDDING_API_KEY,"
            "检索评估还需先 python -m app.knowledge.ingest 建库。",
            file=sys.stderr,
        )
        return 2

    # pymilvus 导入即对仓库根 .env 执行 load_dotenv(实测见 app/main.py 注),
    # 生产组件(嵌入客户端/Milvus 仓储)延迟到确需真跑时导入,保 --self-test 零副作用
    from app.knowledge.embedding import BgeM3Embedder
    from app.knowledge.milvus_repo import MilvusKnowledgeRepo

    embedder = BgeM3Embedder(
        settings.embedding_base_url,
        settings.embedding_api_key,
        settings.embedding_model,
        settings.embedding_dim,
    )
    repo = MilvusKnowledgeRepo(settings.milvus_db_path, dim=settings.embedding_dim)
    try:
        return asyncio.run(_evaluate_live(settings, cases, embedder, repo))
    except Exception as exc:  # noqa: BLE001 - MySQL/Milvus/嵌入端点等基建失败
        print(f"[运行错误] {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "请确认 MySQL 已启动且执行过建表 DDL、python -m app.knowledge.ingest 已建库。",
            file=sys.stderr,
        )
        return 2
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# 自测:罐头语料 + 假嵌入/假仓储(零外部依赖)
# ---------------------------------------------------------------------------

# 罐头知识库(仿 data/knowledge 语料口径;id 即 MySQL chunk 主键):9 条 done + 1 条 pending
_CANNED_KNOWLEDGE: tuple[tuple[int, str, str, str], ...] = (
    (1, "满多少包邮\n运费多少钱", "订单实付金额满 99 元包邮,不满 99 元的订单收取运费 8 元,偏远地区统一收取运费 12 元。", "done"),
    (2, "怎么退货\n七天无理由退货", "商品签收后七天内可申请无理由退货,超过七天未申请的订单系统自动关闭退货入口。", "done"),
    (3, "退款多久能到账\n退货流程", "在订单详情页提交退货申请,仓库验收合格后发起退款,退款在一至三个工作日内按原支付路径退回。", "done"),
    (4, "优惠券过期了还能用吗", "优惠券自发放之日起十五日内有效,过期不可恢复也不能补发,可去领券中心领新券。", "done"),
    (5, "怎么开发票\n发票抬头错了怎么办", "本店默认开具电子发票,确认收货后二十四小时内自动开票,抬头错误可直接在订单页修改。", "done"),
    (6, "支持什么付款方式", "支持微信支付、支付宝、银行卡及花呗分期。", "done"),
    (7, "下单后多久发货", "现货商品在付款后四十八小时内发出,法定节假日顺延。", "done"),
    (8, "尺码不合适怎么换货", "签收后七天内支持同款换码或换色,质量问题换货来回运费由商家承担。", "done"),
    (9, "保修期是多久", "电器类商品保修期为签收后十二个月,人为损坏不在保修范围内。", "done"),
    (10, "已下架商品还能买吗", "已下架商品暂不支持购买。", "pending"),
)

_CANNED_ANSWERS: dict[int, str] = {row[0]: row[2] for row in _CANNED_KNOWLEDGE}

# hit 样例 → 罐头库里应召回的知识块(缺映射或关键词对不上罐头答案都视为自测内部错误)
_DEFAULT_CHUNK_BY_QUERY: dict[str, int] = {
    "邮费是多少": 1,
    "快递费怎么算": 1,
    "怎么退货": 2,
    "退款多久能到账": 3,
    "退货流程是什么": 3,
    "优惠券过期了还能用吗": 4,
    "怎么开发票": 5,
    "支持什么付款方式": 6,
    "下单后多久发货": 7,
    "尺码不合适怎么换货": 8,
    "保修期是多久": 9,
}
SELF_TEST_TOP_K = 3
SELF_TEST_THRESHOLD = 0.5


class _ScriptedEmbedder:
    """假嵌入:embed 按问句查表返回哨兵向量(向量内容无意义,仅作检索路由键)。"""

    def __init__(self, vector_by_text: dict[str, list[float]]):
        self._vectors = vector_by_text
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        try:
            return [self._vectors[text] for text in texts]
        except KeyError as exc:  # pragma: no cover - 自测内部错误
            raise AssertionError(f"罐头嵌入收到了未脚本化的问句:{exc}") from exc


class _ScriptedRepo:
    """假 Milvus 仓储:search 按哨兵向量查表返回预设 (id, score) 列表并裁剪到 top_k。

    表值可为 Exception 实例:search 时原样抛出,模拟 Milvus 故障注入。
    """

    def __init__(self, results_by_vector: dict[tuple[float, ...], Any]):
        self._results = results_by_vector
        self.calls: list[tuple[list[float], int]] = []

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[int, float]]:
        self.calls.append((list(query_vector), top_k))
        stored = self._results[tuple(query_vector)]
        if isinstance(stored, Exception):
            raise stored
        return list(stored)[:top_k]


def build_scripted_world(
    cases: list[dict[str, Any]], overrides: dict[str, Any] | None = None
) -> tuple[_ScriptedEmbedder, _ScriptedRepo]:
    """构造脚本化检索世界:hit 例默认召回其罐头知识块(rank 1, score 0.9),
    no_hit 例默认零召回;overrides 按 case id 覆盖预设 (id, score) 列表或注入异常。

    构造期即校验:每个 hit 例有罐头映射、且关键词全部出现在罐头答案里——
    数据集关键词与语料(罐头库即其缩样)错位会让 --self-test 当场炸出。
    """
    overrides = overrides or {}
    vector_by_text: dict[str, list[float]] = {}
    results_by_vector: dict[tuple[float, ...], Any] = {}
    for index, case in enumerate(cases):
        case_id, query, expect = str(case["id"]), str(case["query"]), str(case["expect"])
        sentinel = [float(index + 1), 0.0, 0.0]
        if expect == "no_hit":
            results: Any = []
        else:
            chunk_id = _DEFAULT_CHUNK_BY_QUERY.get(query)
            if chunk_id is None:
                raise AssertionError(
                    f"self-test 内部错误:hit 样例 {case_id}({query})缺罐头语料映射"
                )
            missing = [
                key for key in case["answer_contains"] if key not in _CANNED_ANSWERS[chunk_id]
            ]
            if missing:
                raise AssertionError(
                    f"self-test 内部错误:样例 {case_id} 关键词 {missing} 不在罐头答案内,"
                    "请对照 data/knowledge 原文核实"
                )
            results = [(chunk_id, 0.9)]
        results = overrides.get(case_id, results)
        vector_by_text[query] = sentinel
        results_by_vector[tuple(sentinel)] = results
    return _ScriptedEmbedder(vector_by_text), _ScriptedRepo(results_by_vector)


async def _seed_canned_knowledge(engine: Any, session_factory: Any) -> None:
    """建表并把罐头知识库落 SQLite 内存库(显式主键对齐假仓储预设的命中 id)。"""
    from app.db.base import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with session_factory() as session, session.begin():
        session.add_all(
            KnowledgeChunk(
                id=chunk_id,
                category="罐头分类",
                questions=questions,
                answer=answer,
                vectorize_status=status,
            )
            for chunk_id, questions, answer, status in _CANNED_KNOWLEDGE
        )


def _expect_value_error(action: Callable[[], Any], what: str) -> None:
    """断言 action 抛 ValueError,否则自测失败。"""
    try:
        action()
    except ValueError:
        return
    raise AssertionError(f"{what} 应抛 ValueError")


async def run_self_test() -> int:
    """离线自测:校验样例集 + 罐头世界验证判分/汇总/两门禁(含 FAIL/PASS)/异常路径。"""
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool

    try:
        cases = load_cases()
        validate_cases(cases, top_k=SELF_TEST_TOP_K)
    except ValueError as exc:
        print(f"[self-test] {exc}", file=sys.stderr)
        return 1
    hit_rows = sum(1 for case in cases if case["expect"] == "hit")
    nohit_rows = len(cases) - hit_rows
    print(
        f"[self-test] 样例集校验通过:{len(cases)} 条(hit {hit_rows} + no_hit {nohit_rows}),"
        f"含四条必含样例(邮费/快递费换说法/老板拒答/七天退货)"
    )

    # 罐头环境:SQLite 内存库 + 假嵌入/假仓储 + 真检索器(阈值/回表/排序逻辑全真)
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    try:
        session_factory = build_session_factory(engine)
        await _seed_canned_knowledge(engine, session_factory)

        # 场景一:错误注入 -> 逐条判分、汇总、两门禁都应符合手工推演
        overrides: dict[str, Any] = {
            # 注入一:邮费样例召回含 pending 块(done 回表过滤)+ 不含关键词的块 → 关键词漏满足
            "faq-postage": [(10, 0.95), (2, 0.6)],
            # 注入二:退货样例零召回 → hit 例零命中应判负
            "policy-return-how": [],
            # 注入三:老板拒答样例被召回 1 条 → no_hit 门禁应挂
            "nohit-boss": [(6, 0.9)],
            # 注入四:换货样例召回的 3 条都不含「换货」 → min_rank 内漏召回应判负
            "faq-exchange": [(1, 0.9), (6, 0.85), (5, 0.8)],
        }
        embedder, repo = build_scripted_world(cases, overrides)
        retriever = KnowledgeRetriever(
            embedder, repo, top_k=SELF_TEST_TOP_K, score_threshold=SELF_TEST_THRESHOLD
        )
        results = await evaluate(cases, retriever, session_factory)
        summary = summarize(results)
        gate_ok, failed_gates = print_report(results, summary)
        print_failure_details(results)

        broken = {"faq-postage", "policy-return-how", "nohit-boss", "faq-exchange"}
        for r in results:
            assert r.passed == (r.case_id not in broken), (
                f"{r.case_id} 判分不符:got {r.passed}"
            )
        assert summary.hit_total == hit_rows, "hit 条数不符"
        assert summary.hit_passed == hit_rows - 3, "hit 达标数应差 3(注入三例)"
        assert summary.nohit_total == nohit_rows, "no_hit 条数不符"
        assert summary.nohit_zero == nohit_rows - 1, "no_hit 零命中应差 1(注入老板误召回)"
        assert gate_ok is False, "错误注入场景门禁应为 FAIL"
        assert failed_gates == ["hit_recall", "no_hit_zero"], (
            f"两条门禁应全部挂,顺序为 hit_recall/no_hit_zero,got {failed_gates}"
        )
        print("[self-test] 场景一通过:判分、汇总与两门禁 FAIL 路径符合预期(对应 live exit 1)")

        # 场景二:无注入的罐头世界 -> 全部达标,两门禁 PASS(对应 live exit 0)
        embedder2, repo2 = build_scripted_world(cases)
        retriever2 = KnowledgeRetriever(
            embedder2, repo2, top_k=SELF_TEST_TOP_K, score_threshold=SELF_TEST_THRESHOLD
        )
        results2 = await evaluate(cases, retriever2, session_factory)
        summary2 = summarize(results2)
        gate_ok2, failed2 = print_report(results2, summary2)
        assert all(r.passed for r in results2), "全对世界应全过"
        assert summary2.hit_passed == summary2.hit_total, "全对场景 hit 应全达标"
        assert summary2.nohit_zero == summary2.nohit_total, "全对场景 no_hit 应全零命中"
        assert gate_ok2 and failed2 == [], "全对场景门禁应为 PASS"
        print("[self-test] 场景二通过:全对场景门禁 PASS 路径符合预期(对应 live exit 0)")

        # 场景三:判分边界 —— 阈值恰等保留/差一点淘汰、rank 恰压线/出界、异常判负
        async def _eval_ids(
            case_ids: list[str], world_overrides: dict[str, Any]
        ) -> list[CaseResult]:
            """单例评估辅助:按指定覆盖构造脚本化世界并评估指定样例。"""
            world_embedder, world_repo = build_scripted_world(cases, world_overrides)
            world_retriever = KnowledgeRetriever(
                world_embedder,
                world_repo,
                top_k=SELF_TEST_TOP_K,
                score_threshold=SELF_TEST_THRESHOLD,
            )
            picked = [dict(case) for case in cases if case["id"] in case_ids]
            return await evaluate(picked, world_retriever, session_factory)

        # 3a. score 恰等于阈值 0.5 → 保留(retriever 的 >= 语义)且关键词命中 → 判对
        results_at = await _eval_ids("policy-refund-eta", {"policy-refund-eta": [(3, 0.5)]})
        assert results_at[0].passed and results_at[0].detail.startswith("rank=1"), (
            f"恰好等于阈值应保留并判对,got {results_at[0].detail}"
        )
        # 3b. score 0.49 < 阈值 → 过滤后零命中,hit 例判负
        results_below = await _eval_ids("policy-refund-eta", {"policy-refund-eta": [(3, 0.49)]})
        assert not results_below[0].passed, "低于阈值被过滤,hit 例应判负"
        # 3c. 含关键词的块恰在 rank=min_rank(3)→ 判对;在 rank 4(被 top_k=3 截断)→ 判负
        results_on_edge = await _eval_ids(
            "policy-return-process",
            {"policy-return-process": [(1, 0.9), (5, 0.85), (3, 0.8)]},
        )
        assert results_on_edge[0].passed and results_on_edge[0].detail.startswith("rank=3"), (
            f"rank=min_rank 压线应判对,got {results_on_edge[0].detail}"
        )
        results_off_edge = await _eval_ids(
            "policy-return-process",
            {"policy-return-process": [(1, 0.9), (5, 0.85), (6, 0.8), (3, 0.75)]},
        )
        assert not results_off_edge[0].passed, "含关键词块被 top_k 截断(rank 4)应判负"
        # 3d. 检索抛异常(Milvus 故障注入):该例判负带 ERROR,不中断
        results_boom = await _eval_ids(
            "manual-warranty", {"manual-warranty": RuntimeError("模拟 Milvus 故障")}
        )
        assert not results_boom[0].passed and results_boom[0].error is not None, (
            "检索异常例应判负且带 ERROR"
        )
        print("[self-test] 场景三通过:阈值恰等/差一点、rank 压线/出界、异常判负均符合预期")

        # 场景四:样例集校验的失败路径(脏数据必须被拦下)
        duplicated = [dict(case) for case in cases]
        duplicated.append(dict(duplicated[0]))
        _expect_value_error(lambda: validate_cases(duplicated, top_k=3), "id 重复的样例集")
        bad_expect = [dict(case) for case in cases]
        bad_expect[0] = dict(bad_expect[0])
        bad_expect[0]["expect"] = "maybe"
        _expect_value_error(lambda: validate_cases(bad_expect, top_k=3), "expect 越界的样例集")
        no_keywords = [dict(case) for case in cases]
        no_keywords[0] = dict(no_keywords[0])
        no_keywords[0].pop("answer_contains")
        _expect_value_error(lambda: validate_cases(no_keywords, top_k=3), "hit 缺 answer_contains")
        nohit_with_rank = [dict(case) for case in cases]
        nohit_victim = next(case for case in nohit_with_rank if case["expect"] == "no_hit")
        nohit_victim["min_rank"] = 2
        _expect_value_error(
            lambda: validate_cases(nohit_with_rank, top_k=3), "no_hit 携带 min_rank 的样例集"
        )
        big_rank = [dict(case) for case in cases]
        big_rank[0] = dict(big_rank[0])
        big_rank[0]["min_rank"] = 5
        _expect_value_error(
            lambda: validate_cases(big_rank, top_k=3), "min_rank 超过 top_k 的样例集"
        )
        missing_required = [case for case in cases if case["query"] != "邮费是多少"]
        _expect_value_error(
            lambda: validate_cases(missing_required, top_k=3), "缺必含问句的样例集"
        )
        truncated = cases[: MIN_CASES - 1]
        _expect_value_error(lambda: validate_cases(truncated, top_k=3), "总条数不足的样例集")
        print("[self-test] 场景四通过:脏数据(重复 id/expect 越界/缺字段/no_hit 带 rank/min_rank 越界/缺必含/条数不足)均被拦下")

    except AssertionError as exc:
        print(f"[self-test] 失败:{exc}", file=sys.stderr)
        return 1
    finally:
        await engine.dispose()

    print(
        "\n[self-test] 全部通过:样例集校验、罐头语料关键词核对、hit/no_hit 判分、两门禁"
        "(含 FAIL/PASS 与 exit 1/0 映射)、阈值与 rank 边界、单条异常不中断均已验证。"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(
        description="知识检索评估跑分(hit min_rank 内含关键词 + no_hit 零命中两门禁)"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="离线自测:罐头语料 + 假嵌入/假仓储验证判分与门禁逻辑,零外部依赖",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return asyncio.run(run_self_test())
    return run_live()


if __name__ == "__main__":
    sys.exit(main())
