r"""历史客服对话挖知识管线:分批 LLM 抽取 → qa_extraction_staging 暂存 → 整体去重入库。

流程(mine_qa):
1. 读全部会话与消息(不按 user_id 过滤),按确定性规则预过滤消息后,每通会话
   渲染成一个带 source 标记的会话块文本;预过滤后无有效消息的会话整通跳过;
2. 按 batch_size(每批会话数)分批,逐批经注入的 LLM 抽问答对;批失败(LLM 调用
   异常、返回非文本或输出解析失败)只隔离该批——零暂存行、函数不抛,后续批照常;
3. 每批独立事务写入 qa_extraction_staging(status=extracted,batch_no=QA-YYYYMMDD-NN,
   NN 为本轮批次序号);LLM 输出里缺字段/空串的脏元素静默剔除,不炸整批;
4. 全部批次跑完后整体去重(单事务):规范化问题(normalize_question)先在本轮
   staging 行内去重(首现保留),再对比 knowledge_chunks 全表既有问法(questions
   列按行拆分、逐行归一入集合);重复置 discarded,保留置 kept,并以「原始问法」
   INSERT knowledge_chunks(category=对话挖掘、content_type=faq、
   vectorize_status 默认 pending、section_path/prev/next 不设)。

LLM 经参数注入(鸭子类型:ainvoke(messages) → .content 为 str 即可),管线不
import LLM factory,测试与 --self-test 均可注入假 LLM。

CLI:``uv run python -m app.knowledge.mine_qa [--batch-size N] [--self-test]``。
--self-test 零外部依赖(sqlite 内存 + 内联两条假会话 + 罐头 LLM),全管线跑通
打印统计与 PASS,退出码 0。
"""

import argparse
import asyncio
import json
import logging
import re
from datetime import datetime
from types import SimpleNamespace

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.engine import build_engine, build_session_factory
from app.db.models import Conversation, KnowledgeChunk, Message, QaExtractionStaging
from app.knowledge.normalize import normalize_question
from app.prompts.loader import load_qa_extraction_prompt

logger = logging.getLogger(__name__)

# 挖矿行入库 knowledge_chunks 的固定口径(brief 钉死)
MINED_CATEGORY = "对话挖掘"
MINED_CONTENT_TYPE = "faq"

# 消息预过滤:正文 strip 后不足该长度视为过短残句(含空正文/单字),不渲染给 LLM
_MIN_MESSAGE_CHARS = 2

# Markdown 代码围栏:```json ... ``` 或 ``` ... ```(允许语言标注行)
_JSON_FENCE_RE = re.compile(r"```[^\n]*\n?(.*?)```", re.DOTALL)


def extract_json_array(text: str) -> list[dict]:
    """从 LLM 输出容错解析出一个 JSON 数组(元素必须都是对象)。

    依次尝试各候选文本:先取 Markdown 围栏内的代码块,再退化为整段文本截取
    首 ``[`` 到末 ``]`` 的切片(容前后废话)。候选解析成功但顶层不是数组、
    或数组元素不是对象,视为输出契约违约立即抛 ``ValueError``;全部候选都
    解析失败同样抛 ``ValueError``。
    """
    candidates = [block.strip() for block in _JSON_FENCE_RE.findall(text)]
    candidates.append(text.strip())
    for candidate in candidates:
        start, end = candidate.find("["), candidate.rfind("]")
        if start == -1 or end <= start:
            continue
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list):
            raise ValueError(f"LLM 输出 JSON 顶层必须是数组,得到 {type(parsed).__name__}")
        if not all(isinstance(item, dict) for item in parsed):
            raise ValueError("LLM 输出 JSON 数组的元素必须是对象")
        return parsed
    raise ValueError("LLM 输出中未找到可解析的 JSON 数组")


def _render_conversation(conversation_id: int, messages: list[Message]) -> str | None:
    """把一通会话渲染成会话块文本;预过滤后无有效消息则返回 None(整通跳过)。

    确定性预过滤(brief 口径的「过短/纯工具轮」):tool 回执行、assistant 纯
    工具轮(tool_calls 在场且无正文)、空正文与过短残句一律剔除;语义过滤
    (寒暄/无答案/转人工/工单受理)交给 Prompt 规则由 LLM 裁决。
    """
    lines = [f"【会话开始 source={conversation_id}】"]
    for message in messages:
        content = (message.content or "").strip()
        if message.role == "tool":
            continue
        if message.role == "assistant" and message.tool_calls and not content:
            continue
        if len(content) < _MIN_MESSAGE_CHARS:
            continue
        lines.append(f"{message.role}:{content}")
    lines.append("【会话结束】")
    return "\n".join(lines) if len(lines) > 2 else None


def _batch_no(date_text: str, ordinal: int) -> str:
    """批次号:QA-YYYYMMDD-NN(NN 为本轮批次序号,零填充到两位)。"""
    return f"QA-{date_text}-{ordinal:02d}"


async def _extract_batch(
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
    batch_blocks: list[str],
) -> list[tuple[str | None, str, str]]:
    """渲染 Prompt 调 LLM 抽一批会话,返回 (source, question, answer) 三元组列表。

    缺字段/空串/非字符串的脏元素静默剔除,不让单条脏元素炸整批;source 缺失
    或为空时置 None(staging.source_ref 可空,仅作溯源线索)。
    """
    messages = prompt.invoke({"conversations": "\n\n".join(batch_blocks)}).to_messages()
    response = await llm.ainvoke(messages)
    content = response.content
    if not isinstance(content, str):
        raise ValueError(f"LLM 返回非文本内容:{type(content).__name__}")
    pairs: list[tuple[str | None, str, str]] = []
    for item in extract_json_array(content):
        question, answer, source = item.get("question"), item.get("answer"), item.get("source")
        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(answer, str) or not answer.strip():
            continue
        pairs.append(
            (
                source.strip() if isinstance(source, str) and source.strip() else None,
                question.strip(),
                answer.strip(),
            )
        )
    return pairs


async def _existing_question_keys(session: AsyncSession) -> set[str]:
    """knowledge_chunks 全表既有问法的规范化键集(本规模 <千行,读全表)。

    questions 列可含多行问法(语料行是「主问法+变体」逐行存放),按行拆分、
    逐行归一,任一行撞键即视为已有同问法知识。
    """
    keys: set[str] = set()
    for blob in await session.scalars(select(KnowledgeChunk.questions)):
        keys.update(key for line in (blob or "").splitlines() if (key := normalize_question(line)))
    return keys


async def mine_qa(
    session_factory: async_sessionmaker[AsyncSession],
    llm: BaseChatModel,
    *,
    batch_size: int = 4,
) -> dict[str, int]:
    """挖矿主流程:分批抽取 → 暂存 → 整体去重入库,返回统计。

    返回 {conversations, batches, extracted, kept, discarded}:conversations 为
    预过滤后实际参与挖矿(渲染出会话块)的会话数;batches 为发起的批次数
    (失败批也计入);extracted = kept + discarded 为写入 staging 的行数;
    batch 失败只隔离该批,函数不抛。
    """
    if batch_size < 1:
        raise ValueError(f"batch_size 必须 >= 1,得到 {batch_size}")

    # 1. 读全部会话与消息,按会话归组渲染(本规模全表读,<千行)
    async with session_factory() as session:
        conversations = (
            (await session.scalars(select(Conversation).order_by(Conversation.id)))
        ).all()
        messages = ((await session.scalars(select(Message).order_by(Message.id)))).all()
    messages_by_conversation: dict[int, list[Message]] = {}
    for message in messages:
        messages_by_conversation.setdefault(message.conversation_id, []).append(message)
    blocks = [
        block
        for conversation in conversations
        if (block := _render_conversation(conversation.id, messages_by_conversation.get(conversation.id, [])))
        is not None
    ]

    stats = {"conversations": len(blocks), "batches": 0, "extracted": 0, "kept": 0, "discarded": 0}
    if not blocks:
        return stats

    prompt = load_qa_extraction_prompt()
    date_text = datetime.now().strftime("%Y%m%d")
    staging_ids: list[int] = []

    # 2-3. 分批抽取:批失败隔离 + 每批独立事务落 staging(extracted,检查点语义)
    for ordinal, offset in enumerate(range(0, len(blocks), batch_size), start=1):
        stats["batches"] += 1
        batch_blocks = blocks[offset : offset + batch_size]
        try:
            pairs = await _extract_batch(llm, prompt, batch_blocks)
        except Exception:
            logger.exception("挖矿批次失败,该批隔离跳过(batch_no=%s)", _batch_no(date_text, ordinal))
            continue
        async with session_factory() as session, session.begin():
            rows = [
                QaExtractionStaging(
                    batch_no=_batch_no(date_text, ordinal),
                    source_ref=source,
                    question=question,
                    answer=answer,
                )
                for source, question, answer in pairs
            ]
            session.add_all(rows)
            await session.flush()  # 立刻拿自增 id,供去重阶段按 id 重取
            staging_ids.extend(row.id for row in rows)
        stats["extracted"] += len(rows)

    # 4. 整体去重(单事务):本轮 staging 首现保留 + 对比既有 knowledge 问法
    if staging_ids:
        async with session_factory() as session, session.begin():
            rows = (
                (
                    await session.scalars(
                        select(QaExtractionStaging)
                        .where(QaExtractionStaging.id.in_(staging_ids))
                        .order_by(QaExtractionStaging.id)
                    )
                )
            ).all()
            seen = await _existing_question_keys(session)
            for row in rows:
                key = normalize_question(row.question)
                if not key or key in seen:  # 纯标点串归一为空,视为无实义问题,一律剔除
                    row.status = "discarded"
                else:
                    row.status = "kept"
                    seen.add(key)
                    stats["kept"] += 1
                    session.add(
                        KnowledgeChunk(
                            category=MINED_CATEGORY,
                            questions=row.question,  # 入库用原始问法(标点原样)
                            answer=row.answer,
                            content_type=MINED_CONTENT_TYPE,
                        )
                    )
            stats["discarded"] = stats["extracted"] - stats["kept"]
    return stats


# ---------------------------------------------------------------------------
# --self-test:零外部依赖的离线自测
# ---------------------------------------------------------------------------


class _CannedLLM:
    """自测罐头 LLM:ainvoke 恒返回预置 JSON 文本(鸭子型响应)。"""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    async def ainvoke(self, messages: list) -> SimpleNamespace:
        return SimpleNamespace(content=self._payload)


# 自测内联假会话(user/assistant 文本轮)
_SELFTEST_DIALOGUES: tuple[tuple[tuple[str, str], ...], ...] = (
    (
        ("user", "优惠券过期了还能用吗?"),
        ("assistant", "过期券不可恢复也不能补发,可去领券中心领新券哦。"),
    ),
    (
        ("user", "发票抬头写错了怎么改?"),
        ("assistant", "开票前可在订单页直接修改抬头。"),
    ),
)


async def _self_test() -> dict[str, int]:
    """离线自测:sqlite 内存库 + 内联两条假会话 + 罐头 LLM,全管线跑通。

    端到端断言抽取 → 暂存 → 去重 → 入库的结果;失败即抛(进程退出码非 0),
    通过则返回统计并打印 PASS;不依赖 API key / MySQL / 真模型 / 磁盘文件。
    """
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        session_factory = build_session_factory(engine)

        conversation_ids: list[int] = []
        async with session_factory() as session, session.begin():
            for turns in _SELFTEST_DIALOGUES:
                conversation = Conversation(user_id="selftest")
                session.add(conversation)
                await session.flush()  # 立刻拿自增 id,供罐头 JSON 对号 source
                session.add_all(
                    Message(conversation_id=conversation.id, role=role, content=content)
                    for role, content in turns
                )
                conversation_ids.append(conversation.id)

        payload = json.dumps(
            [
                {
                    "source": str(conversation_ids[0]),
                    "question": "优惠券过期了还能用吗?",
                    "answer": "过期券不可恢复也不能补发,可去领券中心领新券哦。",
                },
                {
                    "source": str(conversation_ids[1]),
                    "question": "发票抬头写错了怎么改?",
                    "answer": "开票前可在订单页直接修改抬头。",
                },
            ],
            ensure_ascii=False,
        )
        stats = await mine_qa(session_factory, _CannedLLM(payload))
        expected = {"conversations": 2, "batches": 1, "extracted": 2, "kept": 2, "discarded": 0}
        if stats != expected:
            raise AssertionError(f"self-test 统计不符:期望 {expected},实际 {stats}")

        async with session_factory() as session:
            staging = ((await session.scalars(select(QaExtractionStaging)))).all()
            chunks = ((await session.scalars(select(KnowledgeChunk)))).all()
        if len(staging) != 2 or any(row.status != "kept" for row in staging):
            raise AssertionError(f"self-test staging 状态异常:{[row.status for row in staging]}")
        if len(chunks) != 2 or any(chunk.vectorize_status != "pending" for chunk in chunks):
            raise AssertionError(
                f"self-test knowledge_chunks 状态异常:{len(chunks)} 行,"
                f"{[chunk.vectorize_status for chunk in chunks]}"
            )
        return stats
    finally:
        await engine.dispose()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="历史客服对话挖知识(分批抽取/暂存/去重入库)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="每批会话数(缺省读 settings.qa_mine_batch_size)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="离线自测:sqlite 内存 + 内联假会话 + 罐头 LLM,打印统计与 PASS,退出码 0",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    async def _main() -> None:
        """--self-test 走离线自测;否则读配置建引擎/LLM 真跑挖矿并打印统计。"""
        args = _parse_args()
        if args.self_test:
            stats = await _self_test()
            print("self-test 完成,各统计项:")
            for name, value in stats.items():
                print(f"  {name}: {value}")
            print("PASS")
            return

        # 真跑的装配(配置/引擎/LLM)全部留在 __main__ 局部,模块顶层与管线不 import factory
        from app.core.config import get_settings
        from app.llm.factory import get_chat_model

        settings = get_settings()
        batch_size = args.batch_size if args.batch_size is not None else settings.qa_mine_batch_size
        engine = build_engine(settings.database_url)
        try:
            stats = await mine_qa(
                build_session_factory(engine), get_chat_model(settings), batch_size=batch_size
            )
        finally:
            await engine.dispose()
        print("挖矿完成,各统计项:")
        for name, value in stats.items():
            print(f"  {name}: {value}")

    asyncio.run(_main())
