# Ch01 设计 Spec：电商智能客服系统 · 纯对话跑通

- 日期：2026-09-27
- 状态：待用户评审
- 流程：Superpowers Architectural 路径（brainstorm 三节设计已逐节确认）
- 上游文档：`dev-notes/ch01.md`（过程留痕）

## 1. 背景与目标

电商智能客服系统的第一章：把**纯对话链路**端到端跑通，为后续章节（工具调用、Agent 循环、向量检索、可观测）预留清晰接缝。本章的产出是一个可 curl 验收的 FastAPI 服务。

## 2. 范围

**做：**

1. 多轮对话接口 `/api/chat/stream`：SSE 流式输出、逐 token 推送
2. Prompt 模板化管理（`ChatPromptTemplate` + 模板文件），System Prompt 定义客服角色与行为约束
3. 结构化抽取接口 `/api/extract`：`with_structured_output` 把售后描述提取为固定字段
4. 最简多轮上下文：内存会话存储 + token 预算裁剪

**不做（显式排除）：** 工具调用、Agent 循环、数据库持久化、聊天页面（后续章节，页面届时走 Vibe Coding 不走本流程）。

## 3. 已确认决策记录（用户逐项拍板）

| 决策 | 结论 | 确认方式 |
|---|---|---|
| 本章是否做聊天页面 | 不做，curl 验收 | 澄清问答 |
| 开发期默认模型 | DeepSeek（`deepseek-chat`），GPT/Claude/Ollama 进 `.env.example` 可切换 | 澄清问答 |
| 售后诉求枚举 | 退货退款 / 换货 / 维修 / 物流问题 / 发票售后 / 其他（六类） | 澄清问答 |
| 代码组织 | 方案 A 分层单体（否决 B 单文件、C LangGraph 管会话） | 方案对比确认 |
| 设计三节（架构契约 / 链路细节 / 测试风险） | 逐节确认通过 | 分节评审 |
| 流程定制 | 可单测代码走 TDD；纯 Prompt/数据类产出用标注样例/评估集验证替代 | 用户工作要求 |
| 文档查证 | 写码前用 Context7 核对 API（已完成查证，结论见 dev-notes） | 用户工作要求 |

## 4. 架构

分层单体，目录结构：

```
Baihelp/
├── app/
│   ├── api/            # 路由：chat.py（SSE）、extract.py、healthz；sse.py（delta 格式化器）
│   ├── core/           # config.py（pydantic-settings 读 .env）
│   ├── llm/            # factory.py（init_chat_model 工厂）
│   ├── prompts/        # system_prompt.md、extraction_prompt.md + loader
│   ├── services/       # chat.py、extraction.py、history.py（store+trimmer）
│   ├── schemas/        # 请求/响应模型、AfterSalesExtraction
│   └── main.py         # FastAPI 装配
├── evals/
│   ├── extraction_cases.jsonl   # ≥20 条标注样例
│   └── run_eval.py              # 评估脚本，出字段级报告
├── tests/              # pytest 单元测试
├── dev-notes/ch01.md   # 过程留痕
├── .env.example / .env（git 忽略）
└── pyproject.toml      # uv 管理
```

## 5. API 契约

### 5.1 POST /api/chat/stream

请求体：`{"session_id": "s1", "message": "我上周买的鞋子开胶了"}`

响应：`text/event-stream`，**OpenAI 兼容 delta 格式**（与"应用侧统一 OpenAI 协议"哲学一致）：

```
data: {"choices":[{"index":0,"delta":{"content":"亲"}}]}

data: {"choices":[{"index":0,"delta":{"content":"您好"}}]}

data: [DONE]
```

流内错误（SSE 开流后无法改 HTTP 状态码）：先发 `data: {"error": {"message": "..."}}` 再发 `data: [DONE]`。开流前错误（如请求体非法）直接返回 422。

### 5.2 POST /api/extract

请求体：`{"text": "<一段售后描述>"}`

响应 200：`{"order_no": "SO20260927001" | null, "complaint_type": "<六类之一>", "expected_solution": "<原话概括>"}`

上游失败返回 502 `{"detail": "..."}`。

### 5.3 GET /api/healthz

`{"status": "ok", "provider": "deepseek", "model": "deepseek-chat"}`

## 6. 数据流

**对话**：请求 → `ChatService`：`SessionStore.get(session_id)` 取历史 → `HistoryTrimmer.trim([system] + history + [新 user 消息])` 压入预算 → `llm.astream()` 逐 `AIMessageChunk` → SSE 格式化器推给客户端 → 流结束把聚合回复 `append` 回 `SessionStore`。

**抽取**：请求 → `ExtractionService`：`ChatPromptTemplate(抽取模板) | llm.with_structured_output(AfterSalesExtraction, include_raw=True)` → Pydantic 对象序列化为 JSON。无会话状态。

## 7. 组件设计

### 7.1 配置（`app/core/config.py`）

pydantic-settings 读 `.env`：

```ini
LLM_PROVIDER=deepseek
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_API_KEY=sk-xxx
HISTORY_TOKEN_BUDGET=3000
REQUEST_TIMEOUT=60
```

`.env.example` 附 GPT / Claude（Anthropic OpenAI 兼容端点）/ Ollama 注释示例。缺 `LLM_API_KEY` 时启动 fail-fast 并给出可读错误。

### 7.2 模型工厂（`app/llm/factory.py`）

`get_chat_model()` → `init_chat_model(model=..., model_provider="openai", base_url=..., api_key=...)`（Context7 已核对的官方兼容端点接法）。

### 7.3 Prompt 管理（`app/prompts/`）

- `system_prompt.md`：客服角色设定 + 行为约束（中文回复、简洁、**不编造订单状态、不承诺超出权限的赔付**、引导提供订单号、不确定就说明），预留 `{shop_name}` 变量；`loader.py` 加载文件构造 `ChatPromptTemplate`
- `extraction_prompt.md`：六类枚举定义 + "提取不到返回 null" 规则

### 7.4 会话存储与裁剪（`app/services/history.py`）

- `SessionStore`：`dict[session_id, list[BaseMessage]]`，`get/append`；async 单事件循环内不加锁；陌生 `session_id` 自动建会话
- `HistoryTrimmer.trim(messages) -> list[BaseMessage]`：System 消息永远保留；从最新往回按**整轮**（user+assistant 成对）纳入，预算耗尽即停；单条超预算时保底保留最新 user 消息并记 warning。token 计数器可插拔：默认 `model.get_num_tokens`，测试注入假计数器

### 7.5 抽取 Schema（`app/schemas/extraction.py`）

```python
class AfterSalesExtraction(BaseModel):
    order_no: str | None = Field(description="订单号，如 SO20260927001；提取不到为 null")
    complaint_type: Literal["退货退款", "换货", "维修", "物流问题", "发票售后", "其他"]
    expected_solution: str = Field(description="用户期望的解决方案，原话概括")
```

## 8. 测试与评估策略

### 8.1 可单测代码 → TDD（红-绿-重构）

| 单元 | 测试要点 |
|---|---|
| `HistoryTrimmer` | system 永远保留；整轮成对保留；超预算丢最老轮次；单条超预算保底规则 |
| `SessionStore` | append/get；多会话互不串扰 |
| 模型工厂 | env 注入 → `init_chat_model` 参数透传正确 |
| SSE 格式化器 | chunk → delta 格式；`[DONE]` 终止；流内 error 事件 |
| 配置加载 | 字段解析；缺 API key 报清晰错误 |

### 8.2 纯 Prompt / 数据产出 → 评估集验证（替代 TDD 的环节）

- `evals/extraction_cases.jsonl`：≥20 条真实风格中文售后描述，覆盖边界：无订单号、诉求混合、口语/错别字、期望方案含糊；每条带标注 JSON
- `evals/run_eval.py`：逐条跑抽取链，字段级判分——`complaint_type` 枚举必须全对、`order_no` 精确匹配、`expected_solution` 关键词包含；输出报告。阈值：`complaint_type` 100%、`order_no` ≥90%、`expected_solution` ≥80%；不达标改 Prompt 重跑（Prompt 任务的"红绿循环"）
- System Prompt 行为约束：3-5 条对抗样例（诱导编造订单状态 / 诱导承诺仅退款），跑通后人工审阅回复

### 8.3 端到端

真 DeepSeek 冒烟 + curl 验收（见 §10）。

## 9. 错误处理

| 场景 | 行为 |
|---|---|
| LLM 超时/断连（流中） | 流内 error 事件 + `[DONE]` |
| LLM 失败（/extract） | 502 + detail |
| 缺 API key 等配置错误 | 启动 fail-fast |
| 陌生 session_id | 自动建会话 |
| 单条消息超预算 | 保底保留最新 user 消息 + warning 日志 |

## 10. 交付与验收映射

| 验收标准 | 验证方式 |
|---|---|
| 1. curl 可见流式回复 | `curl -N -X POST .../api/chat/stream -d '{...}'` 逐行看到 delta |
| 2. 第二轮接住第一轮上下文 | 同 `session_id` 两轮 curl，第二轮问"我刚才说了什么" |
| 3. 售后描述 → 结构化 JSON | curl `/api/extract` + 评估集跑分报告 |

交付物：功能演示命令（README 或 dev-notes 内）、测试结果（pytest + eval 报告）、`dev-notes/ch01.md` 全程留痕。

## 11. 依赖与风险

- 依赖：`fastapi`、`uvicorn`、`langchain[openai]`（含 langchain-core / langchain-openai）、`pydantic-settings`；测试：`pytest`、`pytest-asyncio`、`httpx`
- **Python 3.14 兼容性风险**：本机 3.14 很新，langchain/pydantic 轮子若不兼容，用 uv 钉 `.python-version=3.12`（语言运行时版本不属于用户点名锁定的选型，允许就地处理）
- **DeepSeek function calling 风险**：`with_structured_output` 默认走 tools；实现第一步实测，异常则降级 `method="json_schema"`（同一 API 的参数档位，不算换方案）；两者皆不可用才停下来问用户
- 事实核查记录：FastAPI 原生 SSE（`fastapi.sse.EventSourceResponse`）、`init_chat_model`、`with_structured_output(include_raw=True)`、`astream()` chunk 聚合，均已 Context7 核对（2026-09-27，见 dev-notes）

## 12. 开放问题

无 —— 全部决策已在 §3 闭环。
