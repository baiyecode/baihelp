r"""mine_qa 历史对话挖知识管线测试:SQLite 内存库 + FakeLLM(密封,不触网不碰真模型)。

覆盖 brief 六测:
- extract_json_array 容错解析:裸数组 / ```json 围栏 / 前后废话三种输入均解析,
  坏 JSON(含顶层非数组、元素非对象)raise ValueError;
- 主流程:FakeLLM 返回 2 对 → staging 2 行 extracted → 去重后 knowledge_chunks
  2 行 pending(category=对话挖掘/content_type=faq/问法存原始问法)、staging 置 kept;
- 去重两轴:批内两对话同问题仅标点差异 → 1 kept 1 discarded;
  knowledge_chunks 预置同问法行(questions 多行问法逐行比对)→ 全部 discarded;
- 批失败隔离:某批 LLM 抛异常 → 函数不抛、该批 staging 零行、后续批照常;
- --self-test 离线自测:子进程跑 ``python -m app.knowledge.mine_qa --self-test``
  退出码 0(内部 sqlite 内存 + 两条内联假会话 + 罐头 LLM,零外部依赖)。

子进程说明:用 sys.executable 直跑模块,与 ``uv run python -m`` 同解释器同环境
(uv run pytest 本身就在 uv 管理的 .venv 里),避免测试内嵌套 uv run 的锁与开销;
Task 11 acceptance ⑦ 会用 uv 命令原样跑。
"""

import json
import subprocess
import sys
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.engine import build_session_factory
from app.db.models import Conversation, KnowledgeChunk, Message, QaExtractionStaging
from app.knowledge.mine_qa import extract_json_array, mine_qa
from app.prompts.loader import load_qa_extraction_prompt


# ---------------------------------------------------------------------------
# fakes:罐头 LLM 按序弹出预设响应,异常项原样抛出,记录每次收到的消息组
# ---------------------------------------------------------------------------


class FakeLLM:
    """假 LLM:ainvoke(messages) 按序返回预设内容(鸭子型响应对象)。"""

    def __init__(self, *responses: str | Exception) -> None:
        self._responses = list(responses)
        self.calls: list[list] = []

    async def ainvoke(self, messages: list) -> SimpleNamespace:
        self.calls.append(messages)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=response)


@pytest_asyncio.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """SQLite 内存库(StaticPool 共享同一连接)+ create_all,交出会话工厂。"""
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield build_session_factory(engine)
    finally:
        await engine.dispose()


def _qa_payload(pairs: list[tuple[str, str, str]]) -> str:
    """(source, question, answer) 元组列表 → 罐头 JSON 数组文本。"""
    return json.dumps(
        [{"source": source, "question": question, "answer": answer} for source, question, answer in pairs],
        ensure_ascii=False,
    )


async def _add_conversation(
    session_factory: async_sessionmaker[AsyncSession],
    turns: tuple[tuple[str, str], ...],
) -> int:
    """插入一通假会话(user/assistant 文本轮),返回自增 id(挖矿的 source)。"""
    async with session_factory() as session, session.begin():
        conversation = Conversation(user_id="mine-test")
        session.add(conversation)
        await session.flush()  # 立刻拿自增 id
        session.add_all(
            Message(conversation_id=conversation.id, role=role, content=content)
            for role, content in turns
        )
        return conversation.id


# ---------------------------------------------------------------------------
# extract_json_array:容错解析(brief 测试一)
# ---------------------------------------------------------------------------


def test_extract_json_array_tolerant() -> None:
    """裸数组 / ```json 围栏 / ``` 围栏 / 前后废话四种输入均解析出同一数组;
    坏 JSON(无数组、坏语法、顶层非数组、元素非对象)一律 ValueError。"""
    expected = [{"source": "1", "question": "问", "answer": "答"}]
    payload = _qa_payload([("1", "问", "答")])

    assert extract_json_array(payload) == expected  # 裸数组
    assert extract_json_array(f"```json\n{payload}\n```") == expected  # json 围栏
    assert extract_json_array(f"```\n{payload}\n```") == expected  # 无语言标注围栏
    assert extract_json_array(f"好的,以下是抽取结果:\n{payload}\n希望有帮助!") == expected  # 前后废话

    for bad in ("这不是 JSON", "", "[{'单引号': 真值}]", '{"source": "1"}', '["元素不是对象"]'):
        with pytest.raises(ValueError):
            extract_json_array(bad)


# ---------------------------------------------------------------------------
# 主流程:抽取 → 暂存 → 整体去重 → 入库(brief 测试二)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mine_qa_writes_staging_and_knowledge(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """两对话抽出 2 对 → staging 2 行 extracted→kept、knowledge_chunks 2 行 pending;
    入库字段口径:category=对话挖掘、content_type=faq、questions=原始问法、
    section_path/is_key_clause/prev/next 默认、vectorize_status=pending。"""
    conv_1 = await _add_conversation(
        session_factory,
        (("user", "退款多久能到账?"), ("assistant", "一至三个工作日内按原支付路径退回。")),
    )
    conv_2 = await _add_conversation(
        session_factory,
        (("user", "发票抬头写错了能改吗?"), ("assistant", "开票前可以在订单页直接修改。")),
    )
    llm = FakeLLM(
        _qa_payload(
            [
                (str(conv_1), "退款多久能到账?", "一至三个工作日内按原支付路径退回。"),
                (str(conv_2), "发票抬头写错了能改吗?", "开票前可以在订单页直接修改。"),
            ]
        )
    )

    stats = await mine_qa(session_factory, llm, batch_size=4)

    assert stats == {
        "conversations": 2,
        "batches": 1,
        "extracted": 2,
        "kept": 2,
        "discarded": 0,
    }
    # LLM 只被调一批:消息为 system(模板)+ human(两块会话文本),会话块带 source 标记
    assert len(llm.calls) == 1
    assert len(llm.calls[0]) == 2
    human_text = llm.calls[0][-1].content
    assert f"【会话开始 source={conv_1}】" in human_text
    assert f"【会话开始 source={conv_2}】" in human_text
    assert "退款多久能到账?" in human_text

    async with session_factory() as session:
        staging = (
            (await session.scalars(select(QaExtractionStaging).order_by(QaExtractionStaging.id)))
        ).all()
        chunks = (
            (await session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id)))
        ).all()

    expected_batch_no = f"QA-{datetime.now().strftime('%Y%m%d')}-01"
    assert [row.batch_no for row in staging] == [expected_batch_no, expected_batch_no]
    assert [row.source_ref for row in staging] == [str(conv_1), str(conv_2)]
    assert [row.question for row in staging] == ["退款多久能到账?", "发票抬头写错了能改吗?"]
    assert [row.answer for row in staging] == [
        "一至三个工作日内按原支付路径退回。",
        "开票前可以在订单页直接修改。",
    ]
    assert all(row.status == "kept" for row in staging)

    assert [chunk.questions for chunk in chunks] == [
        "退款多久能到账?",
        "发票抬头写错了能改吗?",
    ]
    assert [chunk.answer for chunk in chunks] == [
        "一至三个工作日内按原支付路径退回。",
        "开票前可以在订单页直接修改。",
    ]
    assert all(chunk.category == "对话挖掘" for chunk in chunks)
    assert all(chunk.content_type == "faq" for chunk in chunks)
    assert all(chunk.vectorize_status == "pending" for chunk in chunks)
    assert all(chunk.section_path is None for chunk in chunks)
    assert all(not chunk.is_key_clause for chunk in chunks)
    assert all(chunk.prev_chunk_id is None and chunk.next_chunk_id is None for chunk in chunks)


# ---------------------------------------------------------------------------
# 去重两轴:批内规范化等值 / 对比既有 knowledge(brief 测试三、四)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mine_qa_dedup_within_batch(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """两对话抽出同问题(仅全/半角问号差异)→ 规范化等值:1 kept 1 discarded,
    knowledge_chunks 只入首现的原始问法。"""
    conv_1 = await _add_conversation(
        session_factory,
        (("user", "怎么修改收货地址?"), ("assistant", "发货前可在订单页修改。")),
    )
    conv_2 = await _add_conversation(
        session_factory,
        (("user", "收货地址填错了怎么改?"), ("assistant", "发货前都来得及改。")),
    )
    llm = FakeLLM(
        _qa_payload(
            [
                (str(conv_1), "怎么修改收货地址?", "发货前可在订单页修改。"),
                (str(conv_2), "怎么修改收货地址？", "发货前可在订单页修改。"),
            ]
        )
    )

    stats = await mine_qa(session_factory, llm)

    assert (stats["extracted"], stats["kept"], stats["discarded"]) == (2, 1, 1)

    async with session_factory() as session:
        staging = (
            (await session.scalars(select(QaExtractionStaging).order_by(QaExtractionStaging.id)))
        ).all()
        chunks = ((await session.scalars(select(KnowledgeChunk)))).all()

    assert [row.status for row in staging] == ["kept", "discarded"]
    assert len(chunks) == 1
    assert chunks[0].questions == "怎么修改收货地址?"  # 首现原始问法入库
    assert chunks[0].answer == "发货前可在订单页修改。"


@pytest.mark.asyncio
async def test_mine_qa_dedup_vs_existing_knowledge(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """knowledge_chunks 预置同问法行(questions 多行问法逐行归一比对)→
    本轮抽取全部 discarded,不重复入 knowledge。"""
    conv_1 = await _add_conversation(
        session_factory,
        (("user", "退款多久能到账?"), ("assistant", "三个工作日内。")),
    )
    conv_2 = await _add_conversation(
        session_factory,
        (("user", "退款到账时间呢?"), ("assistant", "原路退回。")),
    )
    async with session_factory() as session, session.begin():
        session.add(
            KnowledgeChunk(
                category="售后",
                questions="退款多久到账?\n退款到账时间",  # 多行问法,逐行比对
                answer="既有答案",
                content_type="faq",
            )
        )
    llm = FakeLLM(
        _qa_payload(
            [
                (str(conv_1), "退款多久到账!", "新答案"),  # 与第一行仅标点差异
                (str(conv_2), "退款到账时间!!", "新答案2"),  # 与第二行仅标点差异
            ]
        )
    )

    stats = await mine_qa(session_factory, llm)

    assert (stats["extracted"], stats["kept"], stats["discarded"]) == (2, 0, 2)

    async with session_factory() as session:
        staging = ((await session.scalars(select(QaExtractionStaging)))).all()
        chunk_count = await session.scalar(select(func.count()).select_from(KnowledgeChunk))

    assert all(row.status == "discarded" for row in staging)
    assert chunk_count == 1  # 既有行原样,挖出的重复对不入库


# ---------------------------------------------------------------------------
# 批失败隔离(brief 测试五)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mine_qa_batch_failure_isolated(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """batch_size=1 两批,首批 LLM 抛异常:函数不抛、该批 staging 零行、
    第二批照常抽取入库(批号沿用其序号 -02)。"""
    conv_1 = await _add_conversation(
        session_factory,
        (("user", "积分怎么用?"), ("assistant", "下单直接抵现。")),
    )
    conv_2 = await _add_conversation(
        session_factory,
        (("user", "预售什么时候发货?"), ("assistant", "按商品页预售期发货。")),
    )
    llm = FakeLLM(
        RuntimeError("模拟模型超时"),
        _qa_payload([(str(conv_2), "预售什么时候发货?", "按商品页预售期发货。")]),
    )

    stats = await mine_qa(session_factory, llm, batch_size=1)  # 不应抛异常

    assert (stats["conversations"], stats["batches"]) == (2, 2)
    assert (stats["extracted"], stats["kept"], stats["discarded"]) == (1, 1, 0)

    async with session_factory() as session:
        staging = ((await session.scalars(select(QaExtractionStaging)))).all()

    assert len(staging) == 1  # 失败批零行
    assert staging[0].batch_no.endswith("-02")  # 后续批照常编号
    assert staging[0].source_ref == str(conv_2)
    assert staging[0].status == "kept"


# ---------------------------------------------------------------------------
# --self-test 离线自测(brief 测试六)
# ---------------------------------------------------------------------------


def test_self_test_offline() -> None:
    """``python -m app.knowledge.mine_qa --self-test`` 退出码 0 且打印 PASS:
    内部 sqlite 内存 + 两条内联假会话 + 罐头 LLM,零外部依赖。"""
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "app.knowledge.mine_qa", "--self-test"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "PASS" in result.stdout


# ---------------------------------------------------------------------------
# Prompt 模板输出契约(三要素:仅一个 JSON 数组 / 元素三字段 / 过滤规则)
# ---------------------------------------------------------------------------


def test_qa_extraction_prompt_contract() -> None:
    """load_qa_extraction_prompt 渲染:system 含输出契约三要素,human 原样承载
    会话块文本(不得 HTML 转义/吞花括号,避免破坏 transcript)。"""
    template = load_qa_extraction_prompt()
    conversations = "【会话开始 source=7】\nuser: 问? <含>特殊&字符\nassistant: 答。\n【会话结束】"
    messages = template.invoke({"conversations": conversations}).to_messages()

    system = messages[0].content
    assert messages[-1].content == conversations  # human 原样,无转义
    assert "JSON 数组" in system  # ① 仅输出一个 JSON 数组
    for field in ("source", "question", "answer"):
        assert f'"{field}"' in system  # ② 元素三字段
    for noise in ("寒暄", "转人工", "工单"):
        assert noise in system  # ③ 过滤规则:寒暄/转人工/工单受理不产出
    assert "source=" in system  # source 取会话块标记里的标识
