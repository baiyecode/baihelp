# Ch02 设计文档 — Function Calling 单轮工具链

- 日期:2026-09-27
- 状态:已批准(设计经计划审批确认;用户选定写穿式 + Subagent-driven,并提供建表 DDL)
- 上章:`2026-09-27-ch01-pure-chat-design.md`
- 用户 DDL(建表唯一依据):`J:\code\SQL\ch02-ddl.sql`,原样复制入仓库 `scripts/sql/ch02-ddl.sql`

## 1. 背景与目标

ch01 已跑通纯对话(SSE 流式 + 多轮记忆 + 结构化抽取)。本章给客服系统装上**查数据的能力**:用 LangChain `@tool` 定义五个业务工具,Function Calling 让模型自己决定调什么;工具链长在现有客服聊天入口上,用户在聊天页问一句即可触发。**只做单轮调用**:模型调一次工具就收敛,不做多轮 Agent Loop。

## 2. 范围

**做:**
1. FastAPI + SQLAlchemy 分层骨架;Docker 起 MySQL;按用户 DDL 建 faq / conversations / messages / tickets 四张表并灌测试数据
2. 五个业务工具:query_order / query_product / query_logistics(内部随机 mock,不接真实接口、不建表)、query_faq(SQL LIKE 查 faq 表)、create_ticket(写 tickets 表)
3. 工具基础设施:注册管理、参数 Schema 校验、执行错误处理、超时重试、工具结果回灌
4. 工具链接进 `/api/chat/stream`:模型定工具 → 执行 → 回灌收敛;最终回答仍逐 token 流式;工具执行段先推状态帧;聊天记录(含工具调用与结果)写穿落库;聊天页气泡显示工具轨迹徽章

**不做:** 多轮自动循环 Agent Loop;向量检索 / RAG;真实外部 API;Alembic 迁移;会话历史读路径 DB 化(读仍走内存,见 §3)。

## 3. 已确认决策记录

| # | 决策 | 用户原话/依据 |
|---|------|--------------|
| 1 | 历史读取=**写穿式** | "选第一个" —— 消息实时落 MySQL 作持久留痕,模型上下文仍读内存 SessionStore;已知限制:重启后模型失忆但记录可查 |
| 2 | 建表以用户 DDL 为准 | "这是表语句在这里:J:\code\SQL\ch02-ddl.sql 记得建表" —— 原样进仓库,docker-entrypoint-initdb.d 自动执行 |
| 3 | 执行方式 Subagent-driven | "Subagent-driven(推荐)" —— 与 ch01 相同 |
| 4 | conversations.user_id 承接浏览器 session_id | DDL 无 session_id 列;一个聊天窗口=一通会话,「新会话」按钮即新一通(user_id=UUID,64 位够装) |
| 5 | 单轮收敛的结构保证 | 第二段用**不带工具**的裸模型流式,模型无工具可调,结构性杜绝二次调用 |
| 6 | 聊天页改造走 Vibe Coding | 用户工作要求 #1 的明示例外 |
| 7 | 邮费漏召回是预期结果 | 验收标准 #3;种子 faq 故意不含「邮费/运费」词条,留痕给下一章向量检索升级 |

## 4. 架构

分层单体,延续 ch01 的 routes / core / llm / prompts / services / schemas,新增 `app/db/`(数据层)与 `app/tools/`(工具链):

```
app/
├── db/
│   ├── base.py          # DeclarativeBase
│   ├── models.py        # Conversation/Message/Faq/Ticket,逐列对齐用户 DDL
│   ├── engine.py        # create_async_engine + async_sessionmaker + ping()
│   ├── repositories.py  # 会话 get_or_create / 消息写入 / faq LIKE / 建工单
│   └── seed.py          # 幂等灌数据(python -m app.db.seed)
├── tools/
│   ├── ecommerce.py     # query_order / query_product / query_logistics(mock)
│   ├── knowledge.py     # query_faq
│   ├── tickets.py       # create_ticket(conversation_id 经 InjectedToolArg 注入)
│   └── registry.py      # ToolRegistry:注册/校验/超时重试/错误捕获/上下文注入
├── api/sse.py           # 增 format_tool_event;既有 delta/error/[DONE] 帧不动
├── services/chat.py     # 两段式单轮工具链编排 + 写穿落库
├── core/config.py       # 增 DATABASE_URL / TOOL_TIMEOUT_SECONDS / TOOL_MAX_RETRIES
└── static/chat.html     # 气泡内工具轨迹徽章(Vibe Coding)
docker-compose.yml       # MySQL 8.4,挂载 scripts/sql → /docker-entrypoint-initdb.d
```

## 5. API 契约

### 5.1 既有端点(不动)
`POST /api/extract`、`GET /api/healthz`、`GET /` 原样保留。`POST /api/chat/stream` 请求体不变(`{session_id, message}`)。

### 5.2 SSE 线格式(增量,兼容优先)
既有帧逐字节不动(ch01 测试锁):delta `data: {"choices":[{"index":0,"delta":{"content":"…"}}]}`、error、`data: [DONE]`。

新增**工具帧**(前端未知帧本就静默跳过,向后兼容):
```
data: {"tool":{"name":"query_logistics","args":{"order_id":"1001"},"status":"running"}}
data: {"tool":{"name":"query_logistics","status":"done","summary":"顺丰速运,已到上海转运中心(截断120字)"}}
```
- 工具执行前推 `running` 帧,执行完推 `done` 帧(带截断 summary 供徽章展示)
- 多工具调用仅理论存在;本章按单工具实现,帧序列为 running→done 各一

### 5.3 环境变量(增)
| 键 | 默认 | 说明 |
|----|------|------|
| DATABASE_URL | mysql+aiomysql://baihelp:baihelp@127.0.0.1:3306/baihelp?charset=utf8mb4 | 异步连接串 |
| TOOL_TIMEOUT_SECONDS | 10 | 单次工具执行超时 |
| TOOL_MAX_RETRIES | 1 | 超时/异常后重试次数(总尝试 = 1+重试) |

## 6. 数据流(一次带工具调用的请求)

```
用户消息
  → 写穿:ensure_conversation(user_id=session_id) → INSERT messages(role=user)
  → 组上下文(system + 内存历史 + HumanMessage),HistoryTrimmer 裁剪
  → 第一段:model.bind_tools(五工具).astream() —— chunk 照旧吐 delta,同时相加聚合
  → 聚合含 tool_calls:
      推 running 帧 → registry.execute(校验→超时重试→错误捕获) → 推 done 帧
      → INSERT messages(assistant, tool_calls=申请单JSON, content=首段文本可空)
      → INSERT messages(tool, tool_call_id, content=结果)
      → 内存 store 追加 AIMessage(tool_calls=…)+ToolMessage
      → 第二段:裸 model.astream([system, 历史, Human, AIMessage(tool_calls), ToolMessage…])
        逐 token 吐最终回答 → INSERT messages(assistant, content=最终回答) → [DONE]
  → 聚合无 tool_calls:走 ch01 原路径(INSERT assistant 行 → [DONE])
  → 任一异常:error 帧 + [DONE](ch01 语义;DB 已落行为追加式日志,不回滚)
```

## 7. 组件设计

### 7.1 ORM 模型(逐列对齐用户 DDL)
- **conversations**:id BIGINT PK 自增;user_id VARCHAR(64) + idx_user_id;status ENUM('进行中','已转人工','已结束') 默认 '进行中';created_at/updated_at 默认 CURRENT_TIMESTAMP(updated_at 另含 ON UPDATE,经 server_onupdate 映射)
- **messages**:id PK;conversation_id FK(fk_messages_conversation)+ idx_conversation_id;role ENUM('user','assistant','tool');content TEXT NULL;tool_calls JSON NULL;tool_call_id VARCHAR(64) NULL;created_at
- **faq**:id PK;question VARCHAR(512);answer TEXT;category VARCHAR(64)+idx_category;created_at/updated_at
- **tickets**:ticket_no VARCHAR(32) **主键**;conversation_id FK(fk_tickets_conversation)+idx_conversation_id;description TEXT;ticket_type ENUM('售后','投诉','咨询');status ENUM('待处理','已处理') 默认 '待处理';created_at
- 测试用 SQLite 内存库由 ORM `create_all` 建等价 schema;另设 **DDL↔ORM 列名一致性冒烟测试**(解析 scripts/sql/ch02-ddl.sql 的列名集合 vs ORM 元数据)防漂移

### 7.2 repositories(瘦函数,收 AsyncSession)
- `get_or_create_conversation(session, user_id) -> Conversation`:按 user_id 取最近一条,无则 INSERT(status='进行中')
- `add_message(session, conversation_id, role, content=None, tool_calls=None, tool_call_id=None) -> Message`
- `search_faq(session, keyword, limit=5) -> list[Faq]`:`question LIKE %keyword%`
- `create_ticket(session, conversation_id, description, ticket_type) -> Ticket`:工单号 `T+YYYYMMDD+3位当日序号`(当日计数+1,主键冲突重试)

### 7.3 工具(五个,`@tool` + Pydantic args_schema)
| 工具 | 参数 | 行为 |
|------|------|------|
| query_order | order_id: str | mock:商品名/金额/下单时间/支付方式/订单状态,random 生成 |
| query_product | product_id: str | mock:名称/价格/库存/评分 |
| query_logistics | order_id: str | mock:承运商 + 2~4 条轨迹 + 预计送达 |
| query_faq | keyword: str | LIKE 查 faq.question,拼问答对列表;空结果返回「未找到相关常见问题」 |
| create_ticket | description: str;ticket_type: Literal['售后','投诉','咨询'];conversation_id: Annotated[int, InjectedToolArg](不在 Schema) | 落 tickets,返回工单号+状态 |

mock 工具用 `random` 但接口形态固定(演示);种子 faq 8 条含「退货政策」命中样例、**不含「邮费/运费」**。

### 7.4 ToolRegistry
- `register(tool)` / `tools` 列表(供 bind_tools)/ `get(name)`
- `async execute(name, arguments, context) -> ToolExecutionResult(ok, content, error, attempts)`:
  1. 未知工具 → 错误结果(不抛)
  2. `args_schema.model_validate` 校验,ValidationError → 错误结果
  3. 注入 context(如 conversation_id)中工具声明的 InjectedToolArg 参数
  4. `asyncio.wait_for(tool.ainvoke(payload), timeout)`,超时/异常按 TOOL_MAX_RETRIES 重试,最终失败返回错误结果(**错误也回灌给模型**,由模型向用户解释)
- ToolExecutionResult 序列化为 ToolMessage 内容:`ok → 结果文本;失败 → 「工具执行失败: …」`

### 7.5 编排改造(app/services/chat.py)
`stream_chat_reply` 增参:registry、persistence(写穿门面,包 session_factory)。流程见 §6。内存 store 语义不变:成功才追加;工具轮追加 [Human, AIMessage(tool_calls), ToolMessage…, AIMessage(最终)]。

### 7.6 端点接缝(app/api/chat.py + main.py)
- 模块级 `get_session_factory(request)` 新接缝(monkeypatch 注入 SQLite 工厂):收 `request` 参、返回 `request.app.state.session_factory`——与 get_model 的模块级 monkeypatch 惯例同型,但工厂本体由 lifespan 预建并挂 app.state,保证 ping 探活的 engine 与端点实际使用的 engine 是同一台,且不随请求重建连接池(评审裁定,替代初稿「无参同型」写法)
- `app.state.tool_registry` 与 session_factory 一并于 **lifespan** 组装(注册五工具)——create_app 在模块导入期执行,放那里会要求导入期即具备真实配置,破坏测试导入
- lifespan:engine 预建 + `SELECT 1` ping,fail-fast 提示「先 docker compose up -d 并执行建表 DDL」;关停 dispose。**不跑 create_all**(MySQL 端建表唯一依据是用户 DDL)

### 7.7 聊天页(Vibe Coding,预期效果)
助手气泡顶部出现工具轨迹徽章:收到 running 帧 → 像素风小徽章「⚒ query_logistics…」闪烁;done 帧 → 变「✓ query_logistics」(title 悬浮 summary);最终回答继续在同一气泡逐字追加。气泡结构拆为徽章行 + 正文 span,appendText 只写正文。

## 8. 测试与评估策略

- **TDD 任务**:config 增键;ORM+DDL 一致性;repositories;五工具(mock 以固定 seed 断言形态);registry(未知工具/校验失败/异常捕获/超时→重试→错误回灌,fake 慢工具);SSE 新帧格式锁;编排(FakeToolModel:帧序列 running→done→delta→[DONE]、写穿落库断言、第二段无工具断言、无工具路径回归、旧 4 测全绿——FakeModel 仅补 bind_tools)
- **标注样例替代 TDD**(工作要求 #1,模型行为不可单测):`evals/tool_cases.jsonl` ~18 条(query→期望工具或 none,含邮费漏召回样例)+ `evals/run_tool_eval.py`:bind 五工具非流式 invoke,比对 tool_calls[0].name,门禁整体 ≥90% 且无工具类(闲聊)不误调;`--self-test` 离线自检;live 结果追记 evals/report.md
- **验收脚本** `scripts/acceptance.sh` 扩展:④物流问题出工具帧+按结果作答 ⑤「退货政策」命中 ⑥「邮费是多少」漏召回演示(query_faq 帧存在 + 回复留痕,措辞不判分);①②③ 保持,live 若工具干扰②再微调话术并留痕
- 测试基础设施:现有 client fixture 补 SQLite 内存 session factory 接缝注入

## 9. 错误处理

- 工具失败(校验/超时重试穷尽/异常):**错误文本回灌模型**,模型向用户解释并建议转人工;SSE 仍正常收敛
- DB 不可达:启动 ping fail-fast 带操作提示;运行中写穿失败按既有异常路径走 error 帧
- 上游模型故障:沿用 ch01 error 帧 + [DONE],不写内存历史
- 落库失败不阻断流内已产出内容(追加式日志语义,失败留服务端日志)

## 10. 交付与验收映射

| 验收标准 | 落点 |
|---------|------|
| 1. 「订单 1001 的物流到哪了」出工具徽章并按结果作答 | 编排两段式 + 工具帧 + 聊天页徽章;acceptance ④ |
| 2. 「退货政策」query_faq 查到并作答 | query_faq + 种子数据;acceptance ⑤ |
| 3. 「邮费是多少」关键词查不出(预期漏召回) | 种子刻意缺词条;acceptance ⑥ + dev-notes 留痕给升级 |

交付物:演示命令、全套 pytest 结果、acceptance 6 判据、eval 报告、dev-notes/ch02.md、spec/plan 路径。

## 11. 依赖与风险

- 新依赖:sqlalchemy>=2.0、aiomysql;dev:aiosqlite
- 风险1:aiomysql 对 Python 3.14 兼容性 → 备选 asyncmy → PyMySQL;仍不通停下问(工作要求 #4)
- 风险2:百炼端点(现 .env 指向)tool-calling 支持度未知 → live 验收异常即停问,可切 DeepSeek 官方端点
- 风险3:中文 ENUM 在 SQLAlchemy Enum 的处理 → 统一用值本身,不做 label 转换;SQLite 测试库由 CHECK 约束保真
- 风险4:acceptance ②(复述订单号)可能诱发模型调 query_order → mock 回显 order_id,两条路径都应含该单号;live 若 flaky 调话术并留痕

## 12. 开放问题

无。全量决策见 §3;执行中走不通按工作要求 #4 停下问。
