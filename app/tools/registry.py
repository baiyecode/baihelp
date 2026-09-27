r"""ch02 工具注册表:注册管理 / 参数校验 / 超时重试 / 错误捕获 / 隐藏参数注入。

约定(spec §7.4):
- register(tool) 重名 raise ValueError;tools 按注册顺序返回(顺序即 bind_tools 顺序);
  get(name) 未知返回 None;
- execute(name, arguments, context) 恒返回 ToolExecutionResult(不抛),顺序:
  1. 查名:未知工具 → 错误结果(error 含「未知工具」);
  2. ``args_schema.model_validate`` 校验:ValidationError → 错误结果(error 含「参数校验失败」);
  3. 隐藏参数注入:隐藏参数集 = 函数签名参数名 − args_schema 属性名(即 InjectedToolArg
     注入参数,模型不可见故不入 schema),从 context 取同名非 None 字段塞进 payload;
  4. 循环 1 + max_retries 次 ``asyncio.wait_for(tool.ainvoke(payload), timeout)``:
     TimeoutError 与普通 Exception 均记错误并重试,穷尽后返回错误结果。
- content 恒为「可回灌给模型的文本」:成功 = 工具返回文本;失败 = ``工具执行失败: {error}``,
  由上层直接包进 ToolMessage,让模型向用户解释,而不是把异常抛给端点。
"""

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError


@dataclass
class ToolContext:
    """工具执行上下文:registry 执行时补给声明了同名隐藏参数(InjectedToolArg)的工具。"""

    conversation_id: int
    session_factory: Any


@dataclass
class ToolExecutionResult:
    """单次工具执行结果:content 恒为可回灌给模型的文本(成功为结果,失败为错误说明)。"""

    ok: bool
    content: str
    error: str | None = None
    attempts: int = 1


class ToolRegistry:
    """工具注册表:管理五业务工具的注册与统一执行(校验 → 注入 → 超时重试 → 错误回灌)。"""

    def __init__(self, timeout_seconds: float = 10.0, max_retries: int = 1) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        # dict 保持插入序:tools 属性的注册顺序语义靠它
        self._tools: dict[str, object] = {}
        # 注册时探测一次各工具的隐藏参数集(函数签名参数名 − args_schema 属性名)
        self._hidden_args: dict[str, frozenset[str]] = {}

    def register(self, tool) -> None:
        """注册工具;重名 raise ValueError(同一工具名只允许一份实现)。"""
        if tool.name in self._tools:
            raise ValueError(f"工具「{tool.name}」重复注册")
        self._tools[tool.name] = tool
        self._hidden_args[tool.name] = self._detect_hidden_args(tool)

    @property
    def tools(self) -> list:
        """按注册顺序返回全部工具(供 bind_tools 使用)。"""
        return list(self._tools.values())

    def get(self, name: str) -> object | None:
        """按名取工具;未知返回 None。"""
        return self._tools.get(name)

    async def execute(
        self,
        name: str,
        arguments: dict,
        context: ToolContext | None = None,
    ) -> ToolExecutionResult:
        """统一执行入口:查名 → 校验 → 注入隐藏参数 → 超时重试执行;任何失败都返回错误结果。"""
        # 1. 查名:未知工具不抛,给模型一条可解释的错误文本
        tool = self._tools.get(name)
        if tool is None:
            error = f"未知工具:{name}"
            return ToolExecutionResult(ok=False, content=f"工具执行失败: {error}", error=error)

        # 2. 参数校验:缺必填/类型不符在此拦截,错误信息回灌给模型自行纠正
        schema = getattr(tool, "args_schema", None)
        if schema is not None:
            try:
                schema.model_validate(arguments)
            except ValidationError as exc:
                error = f"参数校验失败:{exc}"
                return ToolExecutionResult(ok=False, content=f"工具执行失败: {error}", error=error)

        # 3. 隐藏参数注入:context 同名的非 None 字段补进 payload(不污染调用方的 arguments)
        payload = dict(arguments)
        hidden = self._hidden_args.get(name, frozenset())
        if hidden and context is not None:
            for arg_name in hidden:
                value = getattr(context, arg_name, None)
                if value is not None:
                    payload[arg_name] = value

        # 4. 超时重试执行:1 + max_retries 次,超时/异常都记错误后重试,穷尽才失败
        total_attempts = 1 + self.max_retries
        error: str | None = None
        for attempt in range(1, total_attempts + 1):
            try:
                raw = await asyncio.wait_for(tool.ainvoke(payload), self.timeout_seconds)
            except TimeoutError:
                error = f"工具「{name}」执行超时(>{self.timeout_seconds:g}s)"
            except Exception as exc:  # 工具层异常统一在此捕获,不再向端点扩散
                error = f"工具「{name}」第 {attempt} 次尝试异常:{exc}"
            else:
                content = raw if isinstance(raw, str) else str(raw)
                return ToolExecutionResult(ok=True, content=content, attempts=attempt)

        return ToolExecutionResult(
            ok=False,
            content=f"工具执行失败: {error}",  # 失败文本同样回灌,由模型向用户解释
            error=error,
            attempts=total_attempts,
        )

    @staticmethod
    def _detect_hidden_args(tool) -> frozenset[str]:
        """隐藏参数集 = 函数签名参数名 − args_schema 属性名(即 InjectedToolArg 注入参数)。

        异步工具的函数存于 ``coroutine``、同步工具存于 ``func``;两者皆无(自定义 BaseTool
        子类)则视为无隐藏参数。
        """
        func = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
        if func is None:
            return frozenset()
        schema = getattr(tool, "args_schema", None)
        schema_keys = set(schema.model_fields.keys()) if schema is not None else set()
        return frozenset(inspect.signature(func).parameters) - schema_keys
