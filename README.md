# Baihelp — 电商智能客服系统

基于 Superpowers 开发方法论逐章构建的电商智能客服系统。已完成 **Ch01 · 纯对话**（多轮对话、SSE 流式输出、Prompt 模板化、售后信息结构化抽取）与 **Ch02 · Function Calling**（五工具单轮调用、MySQL 写穿落库）。

> 开发流程：brainstorm → spec 评审 → 实施计划 → subagent-driven 逐任务实现（每任务独立评审）→ 全分支 code review → live 验收。全程留痕见 [dev-notes/ch01.md](dev-notes/ch01.md)。

## 功能（Ch01）

- **多轮对话接口** `POST /api/chat/stream` — SSE 流式输出、逐 token 推送，OpenAI 兼容 delta 格式（`data: {"choices":[...],"delta":{...}}` + `data: [DONE]`）
- **Prompt 模板化管理** — 客服角色 System Prompt 与抽取指令均为模板文件（`app/prompts/*.md`），改 Prompt 不动代码
- **结构化抽取** `POST /api/extract` — `with_structured_output` 把售后描述提取为固定字段：订单号 / 诉求类型（退货退款·换货·维修·物流问题·发票售后·其他）/ 期望方案
- **上下文管理** — 内存会话存储 + token 预算整轮裁剪（System Prompt 恒定保留，超预算丢最老轮次，单条超预算保底）
- **聊天页面** `GET /` — 浅蓝像素风单文件 Web 界面，气泡排布、逐字渲染、多轮连续对话
- **评估集** — 22 条标注样例（含无订单号 / 混合诉求 / 口语错别字等边界）+ 字段级判分与阈值门禁

## 功能（Ch02 · Function Calling）

- **单轮 Function Calling** — 五个工具以 `@tool` 注册（`query_order` / `query_product` / `query_logistics` / `query_faq` / `create_ticket`），模型按需决定是否调用及调用哪个；SSE 流内嵌 `data: {"tool"...}` 工具事件帧
- **MySQL 写穿** — 每轮对话的用户 / 助手消息写穿落库（conversations / messages 表），FAQ 检索与建工单读写真实数据库（本地无 MySQL 可退 SQLite）
- **工具轨迹徽章** — 聊天页面在回复上方实时渲染工具调用状态小徽章（running / done），普通问答时整行隐藏、视觉零变化

## 快速开始

```bash
# 1. 克隆并安装（需要 uv）
git clone https://github.com/baiyecode/baihelp.git
cd baihelp
uv sync

# 2. 配置模型供应商
cp .env.example .env   # 编辑 LLM_API_KEY，填入你的真实 key

# 3. 启动基础设施并灌测试数据（Ch02 起需要 MySQL）
docker compose up -d           # 起 MySQL（端口默认映射宿主 3307，.env 的 DATABASE_URL 需对应改成 3307）
uv run python -m app.db.seed   # 幂等灌入种子数据（FAQ / 演示会话 / 演示工单）

# 4. 启动
uv run uvicorn app.main:app

# 5. 打开聊天页面
#    http://127.0.0.1:8000/
```

### 模型供应商

应用侧统一 OpenAI 协议（`init_chat_model`），改 `.env` 三个变量即可换接：

| 供应商 | `LLM_BASE_URL` | `LLM_MODEL` 示例 |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 阿里云百炼 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` / `deepseek-v4-flash-0731` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Claude（兼容端点） | `https://api.anthropic.com/v1/` | `claude-sonnet-4-5` |
| Ollama 本地 | `http://localhost:11434/v1` | `qwen3:8b` |

## 验证

```bash
uv run pytest                        # 91 个单元测试
bash scripts/acceptance.sh           # 端到端验收（需服务已启动且已建库灌数据）：流式 / 上下文记忆 / 结构化抽取 / 工具调用 / FAQ 命中 / 漏召回演示
uv run python evals/run_eval.py      # 评估集跑分（门禁：六类 100% / 订单号 ≥90% / 期望方案 ≥80%）
uv run python evals/run_eval.py --self-test   # 判分器自检（无需 API key）
```

最近一次 live 验收与评估报告归档在 [evals/report.md](evals/report.md)。

## 项目结构

```
app/
├── api/        # 路由：chat（SSE，含工具事件帧）、extract、sse 格式化器
├── core/       # .env 配置（pydantic-settings，启动 fail-fast）
├── db/         # MySQL/SQLite 引擎、ORM 模型、仓储、幂等种子数据
├── llm/        # init_chat_model 模型工厂
├── prompts/    # PromptTemplate 模板文件 + 加载器
├── services/   # 会话存储、token 预算裁剪、对话与抽取服务、MySQL 写穿持久化
├── tools/      # @tool 工具：电商 mock 三件套 / query_faq / create_ticket + 注册表
└── schemas/    # Pydantic 模型（AfterSalesExtraction 等）
evals/          # 标注评估集 + 跑分脚本 + 对抗样例
scripts/        # 端到端验收脚本 + SQL DDL
tests/          # pytest 单元测试
docs/superpowers/  # Ch01 设计 spec 与实施计划
dev-notes/      # 开发过程逐阶段留痕
```

## 路线图

- [x] **Ch01** 纯对话跑通（流式 / Prompt 模板 / 结构化抽取 / 上下文管理）
- [x] **Ch02** Function Calling（实际交付：`@tool` 工具调用单轮 + MySQL 写穿；Agent 循环顺延至后续章节）
- [ ] Agent 循环（LangGraph）
- [ ] 会话持久化（SQLAlchemy）
- [ ] 向量检索（Milvus）
- [ ] 可观测（Langfuse）
