# Baihelp — 电商智能客服系统

基于 [Superpowers](https://github.com/obra/superpowers) 开发方法论逐章构建的电商智能客服系统。每一章独立可运行、可验收,开发过程逐阶段留痕。

| 章节 | 主题 | 状态 |
|---|---|---|
| Ch01 | 纯对话:SSE 流式输出、多轮上下文、Prompt 模板化、售后结构化抽取 | ✅ 完成 |
| Ch02 | Function Calling:LangChain `@tool` 单轮工具调用 + MySQL 持久化 | ✅ 完成 |
| Ch03+ | Agent 循环(LangGraph)、向量检索、可观测 | 🚧 规划中 |

技术栈:FastAPI · SQLAlchemy 2.0(异步) · MySQL 8.4(Docker) · LangChain 1.x · SSE · pytest

## 功能

**对话核心(Ch01)**

- `POST /api/chat/stream` — SSE 逐 token 流式输出,OpenAI 兼容 delta 格式(`data: {"choices":[...delta...]}` + `[DONE]`)
- 多轮上下文 — 内存会话存储 + token 预算整轮裁剪(System Prompt 恒保留,工具轮原子组整体纳入/剔除)
- `POST /api/extract` — 售后描述结构化抽取:订单号 / 诉求类型(六类枚举) / 期望方案
- Prompt 模板化 — System Prompt 与抽取指令均为 `app/prompts/*.md` 模板文件,改 Prompt 不动代码

**工具调用(Ch02)**

- 五个业务工具以 `@tool` 注册:`query_order` / `query_product` / `query_logistics`(内部 mock,不接真实接口)、`query_faq`(SQL LIKE 查 faq 表)、`create_ticket`(写 tickets 表);模型自主决定调不调、调哪个
- 单轮收敛 — 第一段绑工具流式,聚合出工具调用则执行并把结果回灌,第二段用裸模型流式最终回答(结构上杜绝连环调用)
- 工具基础设施 — ToolRegistry 统一注册管理、Pydantic 参数校验、超时重试、错误捕获后回灌模型
- MySQL 写穿 — user / assistant / tool 消息实时落 conversations / messages 表;建表 DDL 见 `scripts/sql/ch02-ddl.sql`
- 工具轨迹徽章 — 聊天页气泡内实时显示本次调用了哪些工具(虚线徽章,查询中闪烁、完成落定),普通问答零变化

## 快速开始

前提:Python 3.14+、[uv](https://docs.astral.sh/uv/)、Docker Desktop。

```bash
git clone https://github.com/baiyecode/baihelp.git
cd baihelp
uv sync

cp .env.example .env        # 填入 LLM_API_KEY;本地 .env 的 DATABASE_URL 端口改成 3307(见下)

docker compose up -d        # 起 MySQL(宿主 3307,首启自动按 scripts/sql/ch02-ddl.sql 建表)
uv run python -m app.db.seed  # 幂等灌测试数据(FAQ 8 条 / 演示会话 / 演示工单)

uv run uvicorn app.main:app   # 启动即 ping 数据库,连不上会 fail-fast 并给出操作提示
# 打开 http://127.0.0.1:8000/
```

> 端口说明:容器内 MySQL 是 3306,docker-compose 映射到宿主 **3307**(避开常见的本机 MySQL 服务);`.env.example` 的默认值是 3306,本地跑请把 `DATABASE_URL` 的端口改成 3307。

### 体验演示

打开聊天页,依次问:

| 你问 | 会发生什么 |
|---|---|
| 「订单 1001 的物流到哪了」 | 气泡上方出现工具徽章(query_order / query_logistics),回答基于工具返回的 mock 数据 |
| 「退货政策是什么」 | 触发 query_faq,LIKE 命中种子 FAQ 并作答 |
| 「邮费是多少」 | query_faq 被选中但关键词查不到——**已知漏召回**,模型诚实兜底;这是刻意留的演示,留给后续向量检索升级 |

### 模型供应商

应用侧统一 OpenAI 协议(`init_chat_model`),改 `.env` 即可换接;工具调用要求所接模型支持 function calling。

| 供应商 | `LLM_BASE_URL` | `LLM_MODEL` 示例 |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Ollama 本地 | `http://localhost:11434/v1` | `qwen3:8b` |

## 配置(.env)

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_API_KEY` | 必填 | 模型密钥,缺失时启动即失败 |
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_PROVIDER` | DeepSeek | OpenAI 兼容端点三元组 |
| `DATABASE_URL` | `mysql+aiomysql://…3306/baihelp` | 异步连接串;compose 环境改 3307 |
| `HISTORY_TOKEN_BUDGET` | 3000 | 会话历史 token 预算 |
| `REQUEST_TIMEOUT` | 60 | 模型请求超时(秒) |
| `TOOL_TIMEOUT_SECONDS` | 10 | 单次工具执行超时 |
| `TOOL_MAX_RETRIES` | 1 | 工具超时/异常重试次数 |

## 验证

```bash
uv run pytest                                # 91 个单元测试
bash scripts/acceptance.sh                   # 端到端 6 判据:流式 / 上下文记忆 / 结构化抽取 / 工具调用 / FAQ 命中 / 漏召回演示
uv run python evals/run_eval.py              # Ch01 抽取评估(22 例,字段级门禁)
uv run python evals/run_tool_eval.py         # Ch02 工具选型评估(19 例,门禁:整体 ≥90% / 闲聊误调 0)
uv run python evals/run_tool_eval.py --self-test   # 离线自检,无需 API key
```

最近一次 live 验收与评估结果归档在 [evals/report.md](evals/report.md)。

## 项目结构

```
app/
├── api/        # 路由:chat(SSE,含工具事件帧)、extract、sse 格式化器
├── core/       # .env 配置(pydantic-settings,启动 fail-fast)
├── db/         # 引擎/ORM 模型(对齐 DDL)/仓储/幂等种子数据
├── llm/        # init_chat_model 模型工厂
├── prompts/    # Prompt 模板文件 + 加载器
├── services/   # 会话存储、裁剪器、对话编排(两段式单轮工具链)、写穿持久化
├── tools/      # @tool 五工具 + ToolRegistry(校验/超时重试/错误回灌)
└── schemas/    # Pydantic 模型
docker-compose.yml   # MySQL 8.4(3307→3306,initdb 自动建表)
scripts/             # 端到端验收脚本 + 建表 DDL
evals/               # 标注评估集 + 跑分器 + 报告归档
tests/               # pytest 单元测试
docs/superpowers/    # 各章设计 spec 与实施计划
dev-notes/           # 开发过程逐阶段留痕(ch01 / ch02)
```

## 开发方法

每章走完整 Superpowers 流程:brainstorm → 设计 spec(用户评审)→ 实施计划 → subagent-driven 逐任务实现(每任务独立实现者 + 评审者,TDD;模型行为类产出以标注评估集替代单测)→ live 验收 → 全分支 code review。过程按「每完成一个阶段追记一段」的规则落在 [dev-notes/ch01.md](dev-notes/ch01.md) 与 [dev-notes/ch02.md](dev-notes/ch02.md)。

## 路线图

- [x] Ch01 纯对话跑通
- [x] Ch02 Function Calling 单轮工具链 + 消息写穿持久化
- [ ] Agent 循环(LangGraph)
- [ ] 会话历史 DB 化(上下文读路径,当前仍为内存)
- [ ] 向量检索(Milvus)——根治 FAQ 关键词漏召回
- [ ] 可观测(Langfuse)
