r"""app.tools.registry 工具注册表的行为测试(假工具为主 + 一例真实 SQLite 集成)。

验证(ToolRegistry / ToolContext / ToolExecutionResult,签名按 task-7-brief 锁定):

- register / get / tools:注册顺序即 tools 顺序;get 命中返回同一对象;get 未知返回 None;
- 重复注册同名工具 raise ValueError;
- execute 未知工具 → ok=False、error 含「未知工具」、content 为可回灌错误文本(不抛);
- 校验失败(缺必填参数)→ ok=False、error 含「参数校验失败」;
- 工具抛异常 → ok=False、error 含异常信息、attempts == 1 + max_retries(异常也重试,spec §7.4);
- 超时(asyncio.sleep 慢工具 + timeout_seconds=0.05)→ ok=False、error 含「超时」、
  attempts == 1 + max_retries;
- 隐藏参数注入:工具函数签名带 conversation_id/session_factory(不在 args_schema)时,
  execute 从 context 把同名字段补进调用;普通工具(签名 == schema)不收多余键;
- 成功路径:ok=True、content 为工具返回文本、attempts == 1。

约定:与实现统一走 ainvoke(registry 内部只调 tool.ainvoke)。
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

import pytest
import pytest_asyncio
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db import models
from app.db.base import Base
from app.db.engine import build_session_factory
from app.knowledge.retriever import RetrievedKnowledge
from app.tools import get_all_tools
from app.tools.ecommerce import query_order
from app.tools.knowledge import query_faq
from app.tools.registry import ToolContext, ToolExecutionResult, ToolRegistry
from tests.test_tools_db import FakeRetriever

# ---------------------------------------------------------------------------
# 测试用假工具(@tool 包装,与真实工具同为 StructuredTool)
# ---------------------------------------------------------------------------


class _FakeInput(BaseModel):
    """假工具统一入参:单个 keyword 必填。"""

    keyword: str = Field(description="关键词")


@tool(args_schema=_FakeInput)
async def _echo_tool(keyword: str) -> str:
    """原样回显关键词的假工具(成功路径)。"""
    return f"echo:{keyword}"


@tool(args_schema=_FakeInput)
async def _boom_tool(keyword: str) -> str:
    """总是抛 RuntimeError 的假工具(异常重试路径)。"""
    raise RuntimeError(f"数据库连接失败:{keyword}")


@tool(args_schema=_FakeInput)
async def _slow_tool(keyword: str) -> str:
    """睡 5 秒的假工具(超时路径;wait_for 会在超时阈值处取消,不会真等 5 秒)。"""
    await asyncio.sleep(5)
    return "不应到达"


def _make_spy_tool() -> tuple[Any, dict[str, Any]]:
    """构造记录实参的假工具:签名带 conversation_id/session_factory 隐藏参数(不在 args_schema)。"""
    received: dict[str, Any] = {}

    @tool(args_schema=_FakeInput)
    async def spy_tool(
        keyword: str,
        *,
        conversation_id: Annotated[int, InjectedToolArg],
        session_factory: Annotated[Any, InjectedToolArg],
    ) -> str:
        """记录收到的隐藏参数后回显 conversation_id 的假工具。"""
        received["keyword"] = keyword
        received["conversation_id"] = conversation_id
        received["session_factory"] = session_factory
        return f"conversation={conversation_id}"

    return spy_tool, received


class _AinvokeRecorder:
    """透传包装:记录 registry 实际传给 ainvoke 的 payload(验证普通工具不收多余键)。

    除 ainvoke 外的一切属性(name / args_schema / coroutine / func)都委托给内层工具,
    registry 的注册与隐藏参数探测对包装无感。langchain 对普通工具 schema 外的
    多余键会静默丢弃,函数级断言不可见,必须在 ainvoke 边界记录。
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.seen: list[dict] = []

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    async def ainvoke(self, payload: dict, **kwargs: Any) -> str:
        self.seen.append(payload)
        return await self._inner.ainvoke(payload, **kwargs)


# ---------------------------------------------------------------------------
# 注册管理:register / get / tools
# ---------------------------------------------------------------------------


def test_register_get_tools_follow_registration_order() -> None:
    """按注册顺序返回 tools;get 命中返回同一对象;get 未知返回 None。"""
    registry = ToolRegistry()
    for t in get_all_tools():
        registry.register(t)

    assert [t.name for t in registry.tools] == [
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    ]
    assert registry.get("query_faq") is query_faq
    assert registry.get("no_such_tool") is None


def test_register_duplicate_name_raises_value_error() -> None:
    """重复注册同名工具 raise ValueError。"""
    registry = ToolRegistry()
    registry.register(query_order)

    with pytest.raises(ValueError, match="重复注册"):
        registry.register(query_order)


# ---------------------------------------------------------------------------
# execute:未知工具 / 校验失败
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_unknown_tool_returns_feedable_error() -> None:
    """未知工具 → ok=False、error 含「未知工具」、content 为可回灌错误文本(不抛)。"""
    registry = ToolRegistry()

    result = await registry.execute("no_such_tool", {"keyword": "x"})

    assert result.ok is False
    assert result.error is not None and "未知工具" in result.error
    # content 恒为可回灌给模型的文本:失败时固定「工具执行失败: {error}」
    assert result.content == f"工具执行失败: {result.error}"


@pytest.mark.asyncio
async def test_execute_validation_failure_reports_error() -> None:
    """缺必填参数(query_order 不给 order_id)→ ok=False、error 含「参数校验失败」。"""
    registry = ToolRegistry()
    registry.register(query_order)

    result = await registry.execute("query_order", {})

    assert result.ok is False
    assert result.error is not None and "参数校验失败" in result.error
    assert result.content == f"工具执行失败: {result.error}"


# ---------------------------------------------------------------------------
# execute:异常重试 / 超时重试
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tool_exception_retries_then_reports() -> None:
    """工具抛异常也重试:max_retries=2 → attempts == 3、error 含异常信息、ok=False。"""
    registry = ToolRegistry(max_retries=2)
    registry.register(_boom_tool)

    result = await registry.execute("_boom_tool", {"keyword": "爆"})

    assert result.ok is False
    assert result.error is not None and "数据库连接失败" in result.error  # 异常信息
    assert result.attempts == 1 + 2
    assert result.content == f"工具执行失败: {result.error}"


@pytest.mark.asyncio
async def test_execute_timeout_retries_then_reports() -> None:
    """慢工具超时:timeout_seconds=0.05、max_retries=1 → attempts == 2、error 含「超时」。"""
    registry = ToolRegistry(timeout_seconds=0.05, max_retries=1)
    registry.register(_slow_tool)

    result = await registry.execute("_slow_tool", {"keyword": "慢"})

    assert result.ok is False
    assert result.error is not None and "超时" in result.error
    assert result.attempts == 1 + 1
    assert result.content == f"工具执行失败: {result.error}"


# ---------------------------------------------------------------------------
# execute:隐藏参数注入 / 普通工具不收多余键
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_injects_hidden_args_from_context() -> None:
    """隐藏参数(不在 args_schema)由 context 同名字段补进调用:conversation_id / session_factory。"""
    spy_tool, received = _make_spy_tool()
    registry = ToolRegistry()
    registry.register(spy_tool)
    sentinel_factory = object()
    context = ToolContext(conversation_id=42, session_factory=sentinel_factory)

    result = await registry.execute("spy_tool", {"keyword": "退货"}, context)

    assert result.ok is True
    assert result.content == "conversation=42"
    assert received["keyword"] == "退货"
    assert received["conversation_id"] == 42
    assert received["session_factory"] is sentinel_factory


@pytest.mark.asyncio
async def test_execute_plain_tool_receives_no_extra_keys() -> None:
    """普通工具(签名 == schema)不收多余键:payload 原样透传,context 字段不混入。"""
    recorder = _AinvokeRecorder(_echo_tool)
    registry = ToolRegistry()
    registry.register(recorder)
    context = ToolContext(conversation_id=7, session_factory=object())

    result = await registry.execute("_echo_tool", {"keyword": "你好"}, context)

    assert result.ok is True
    assert recorder.seen == [{"keyword": "你好"}]


# ---------------------------------------------------------------------------
# execute:成功路径 / 真实 DB 工具端到端
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_success_path_returns_tool_text() -> None:
    """成功路径:ok=True、content 为工具返回文本(可 json.loads)、attempts == 1、error 为 None。"""
    registry = ToolRegistry()
    registry.register(query_order)

    result = await registry.execute("query_order", {"order_id": "ORD-1001"})

    assert isinstance(result, ToolExecutionResult)
    assert result.ok is True
    assert result.attempts == 1
    assert result.error is None
    data = json.loads(result.content)
    assert data["订单号"] == "ORD-1001"


@pytest_asyncio.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """SQLite 内存库(StaticPool 共享同一连接)+ create_all,预置一条 FAQ,交出会话工厂。"""
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = build_session_factory(engine)
        async with factory() as s, s.begin():
            s.add(
                models.Faq(
                    question="退货政策是什么?",
                    answer="签收后7天内支持无理由退货,需保持商品完好。",
                    category="售后",
                )
            )
        yield factory
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_execute_real_query_faq_with_context_injection(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """真实 DB 工具端到端:registry 把 ToolContext 的 session_factory/retriever 注入 query_faq 并返回命中文本。"""
    registry = ToolRegistry()
    registry.register(query_faq)
    context = ToolContext(
        conversation_id=1,
        session_factory=session_factory,
        retriever=FakeRetriever(
            [
                RetrievedKnowledge(
                    chunk_id=1,
                    category="售后",
                    questions="退货政策是什么?",
                    answer="签收后7天内支持无理由退货,需保持商品完好。",
                    score=0.9,
                )
            ]
        ),
    )

    result = await registry.execute("query_faq", {"keyword": "退货"}, context)

    assert result.ok is True
    assert result.attempts == 1
    assert "退货政策是什么?" in result.content
    assert "签收后7天内支持无理由退货,需保持商品完好。" in result.content
