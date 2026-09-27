# Ch02 Function Calling 单轮工具链 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给现有客服聊天装上单轮 Function Calling 工具链:五个 LangChain @tool 工具 + MySQL 写穿持久化 + SSE 工具状态帧 + 聊天页工具徽章。

**Architecture:** 分层单体延续 ch01:新增 `app/db/`(SQLAlchemy 异步数据层,建表以用户 DDL 为准)与 `app/tools/`(@tool 五工具 + ToolRegistry 基础设施);`stream_chat_reply` 改两段式——第一段绑工具流式并聚合,聚合出 tool_calls 则推工具帧、执行、回灌,第二段裸模型流式最终回答(结构性单轮收敛)。聊天记录写穿落 conversations/messages 表,模型上下文仍读内存 SessionStore。

**Tech Stack:** FastAPI 0.141+ / SQLAlchemy 2.0 异步 + aiomysql / MySQL 8.4(Docker)/ langchain[openai] 1.x(@tool、bind_tools)/ pytest + pytest-asyncio + aiosqlite(密封测试)。

**Spec:** `docs/superpowers/specs/2026-09-27-ch02-function-calling-design.md`

## Global Constraints

- 技术选型定死:FastAPI + SQLAlchemy + MySQL(Docker)+ LangChain @tool;走不通停下问用户,不得自行换方案
- MySQL 建表唯一依据 = `scripts/sql/ch02-ddl.sql`(用户 DDL 原样复制自 `J:\code\SQL\ch02-ddl.sql`);应用代码不对 MySQL 跑 create_all;DDL↔ORM 一致性面 = 表名/列名/主键/枚举值集合(索引与外键命名以 DDL 文件为准,ORM 不生成 MySQL DDL)
- 既有 SSE delta / error / `[DONE]` 帧逐字节不变;只增新事件类型不改旧格式
- 既有 23 个测试断言全数保绿;`FakeModel` 只许新增 `bind_tools` 方法,不许改既有断言
- 中文/协议枚举值原样:conversations.status ∈ {'进行中','已转人工','已结束'};messages.role ∈ {'user','assistant','tool'};tickets.ticket_type ∈ {'售后','投诉','咨询'};tickets.status ∈ {'待处理','已处理'}
- 工单号格式:`T` + `YYYYMMDD` + 3 位当日序号(如 T20260701008)
- conversations.user_id 承接浏览器 session_id(一个聊天窗口=一通会话)
- 第二段用不带工具的裸模型,单轮收敛是结构保证
- 包管理 uv,Python 3.14;提交信息 conventional 前缀 + 中文描述
- 三个 mock 工具不接真实接口、不建表,内部 random 生成
- `dev-notes/ch02.md` 由 controller 每阶段追记,实现者不写

## Review Focus

1. **纯工具调用无正文**(首段无 delta,assistant 行 content 为 NULL):气泡应只显徽章不显空文本 — Task 9 `test_tool_round_without_first_text` 钉
2. **LIKE 关键词含通配符**(`%`/`_`/`\`)或空串:search_faq 必须转义,空串返回空结果而非全表 — Task 4 `test_search_faq_escapes_like_wildcards` 钉
3. **提供方 chunk 差异**:tool_call_chunks 有内容而 text 为空;聚合靠 chunk 相加,不靠 text 拼接 — Task 9 聚合测试钉
4. **同一会话第二轮(历史含 ToolMessage)再进裁剪器**:HistoryTrimmer 面向 BaseMessage,.text 取值不得炸 — Task 9 `test_second_turn_with_tool_history` 钉
5. **工单号当日并发/冲突**:主键冲突重试换号,序号进位(如当日第 1000 单)不崩 — Task 4 `test_create_ticket_collision_retry` 钉

---

### Task 1: 依赖与配置(Settings 增键)

**Files:**
- Modify: `pyproject.toml`(dependencies 增 `sqlalchemy>=2.0`, `aiomysql>=0.2.0`;dev 增 `aiosqlite`)
- Modify: `app/core/config.py`(Settings 增三字段)
- Modify: `.env.example`(增三键注释示例)
- Test: `tests/test_config.py`(追加)

**Interfaces:**
- Produces: `Settings.database_url: str`(默认 `"mysql+aiomysql://baihelp:baihelp@127.0.0.1:3306/baihelp?charset=utf8mb4"`)、`Settings.tool_timeout_seconds: float = 10.0`、`Settings.tool_max_retries: int = 1`;后续所有任务经 `get_settings()` 消费

- [ ] **Step 1: 写失败测试**(追加到 tests/test_config.py,沿用既有构造风格 `Settings(_env_file=None, llm_api_key="sk-test")`)

```python
def test_database_settings_defaults() -> None:
    """默认连接串指向 Docker MySQL 的 baihelp 库,超时/重试为章节数值。"""
    settings = Settings(_env_file=None, llm_api_key="sk-test")
    assert settings.database_url == (
        "mysql+aiomysql://baihelp:baihelp@127.0.0.1:3306/baihelp?charset=utf8mb4"
    )
    assert settings.tool_timeout_seconds == 10.0
    assert settings.tool_max_retries == 1

def test_database_url_env_override() -> None:
    """环境变量可覆盖连接串(测试换 SQLite 用同一开关)。"""
    settings = Settings(_env_file=None, llm_api_key="sk-test", database_url="sqlite+aiosqlite://")
    assert settings.database_url == "sqlite+aiosqlite://"
```

- [ ] **Step 2: 跑测试确认失败** — `uv run pytest tests/test_config.py -v`,预期 FAIL(ValidationError: 缺 database_url 字段或 AttributeError)
- [ ] **Step 3: 实现** — Settings 增三字段(带默认值);`uv add sqlalchemy aiomysql`、`uv add --dev aiosqlite`;`.env.example` 增 `DATABASE_URL`(默认值原样)与 `TOOL_TIMEOUT_SECONDS=10`、`TOOL_MAX_RETRIES=1` 注释示例
- [ ] **Step 4: 跑测试确认通过** — `uv run pytest tests/test_config.py tests/test_factory.py -v`(factory 测试确认 init_chat_model 不受影响);再跑全套 `uv run pytest` 23 passed
- [ ] **Step 5: 提交** — `git add -A && git commit -m "feat(core): 数据库与工具运行时配置项(DATABASE_URL/超时/重试)+ 依赖"`

### Task 2: Docker MySQL + DDL 落库

**Files:**
- Create: `scripts/sql/ch02-ddl.sql`(从 `J:\code\SQL\ch02-ddl.sql` 原样复制,一字不改)
- Create: `docker-compose.yml`

**Interfaces:**
- Produces: 可用的 MySQL 8.4 实例(库 baihelp / 用户 baihelp / 密码 baihelp / 端口 3306),四张表已建;后续任务与 live 验收依赖

- [ ] **Step 1: 复制 DDL** — `cp /j/code/SQL/ch02-ddl.sql scripts/sql/ch02-ddl.sql`,git diff 确认内容与源一致
- [ ] **Step 2: 写 docker-compose.yml** — 服务名 `mysql`;镜像 `mysql:8.4`;环境 `MYSQL_DATABASE=baihelp`、`MYSQL_USER=baihelp`、`MYSQL_PASSWORD=baihelp`、`MYSQL_ROOT_PASSWORD=baihelp-root`;端口 `3306:3306`;卷 `mysql-data:/var/lib/mysql` 与 `./scripts/sql:/docker-entrypoint-initdb.d:ro`;command `--character-set-server=utf8mb4 --collation-server=utf8mb4_unicode_ci`;healthcheck `mysqladmin ping -h 127.0.0.1 -pbaihelp-root`(healthcheck 间隔 5s)
- [ ] **Step 3: 起库验证**(非 pytest,命令验证) — `docker compose up -d`,等 healthcheck healthy(约 30s),然后:

```bash
docker compose exec mysql mysql -ubaihelp -pbaihelp baihelp -e "SHOW TABLES; DESCRIBE conversations;"
```

预期输出含 faq / conversations / messages / tickets 四行,conversations 列含 user_id/status/created_at/updated_at。UTF-8 冒烟:`docker compose exec mysql mysql -ubaihelp -pbaihelp baihelp -e "SELECT '进行中' = '进行中';"` 返回 1。
- [ ] **Step 4: 提交** — `git add scripts/sql/ch02-ddl.sql docker-compose.yml && git commit -m "chore(db): Docker MySQL 8.4 编排 + 用户 DDL 落库(docker-entrypoint-initdb.d 自动建表)"`

### Task 3: DB 骨架(ORM 模型 + engine + DDL 一致性测试)

**Files:**
- Create: `app/db/__init__.py`、`app/db/base.py`、`app/db/models.py`、`app/db/engine.py`
- Test: `tests/test_db_models.py`、`tests/test_ddl_orm_consistency.py`

**Interfaces:**
- Produces:
  - `app.db.base.Base`(DeclarativeBase)
  - `app.db.models.Conversation / Message / Faq / Ticket`(字段见 spec §7.1;`Conversation.status` 默认 `'进行中'`,`Ticket.status` 默认 `'待处理'`)
  - `app.db.engine.build_engine(database_url: str) -> AsyncEngine`
  - `app.db.engine.build_session_factory(engine) -> async_sessionmaker[AsyncSession]`(expire_on_commit=False)
  - `app.db.engine.ping(engine) -> None`(失败 raise RuntimeError,消息含「docker compose up -d」操作提示)
- Consumes: Task 1 的 `Settings.database_url`

- [ ] **Step 1: 写失败测试** — `tests/test_ddl_orm_consistency.py`:正则解析 `scripts/sql/ch02-ddl.sql` 每个 `CREATE TABLE <名> (…)`,抽列名集合;断言与 `Base.metadata.tables` 四张表的列名集合逐一相等。`tests/test_db_models.py`:SQLite 内存(`sqlite+aiosqlite://`)create_all 后——插入 Conversation 断言 status 默认「进行中」;插入 Ticket 断言 status 默认「待处理」;Message 挂 conversation_id 外键可写,role='tool' + tool_call_id + JSON tool_calls 往返保真;Ticket(ticket_type='售后') 往返保真
- [ ] **Step 2: 跑测试确认失败** — `uv run pytest tests/test_ddl_orm_consistency.py tests/test_db_models.py -v`,预期 FAIL(ModuleNotFoundError: app.db)
- [ ] **Step 3: 实现** — base.py 定义 Base;models.py 四个模型:id 用 `BigInteger` 主键自增、`str` 列带与 DDL 相同长度、`Enum("进行中","已转人工","已结束", name="conversation_status")` 形式原样用值、时间列 `server_default=func.now()`(updated_at 另加 `server_onupdate=func.now()`)、索引按 DDL 同名列建(`__table_args__` 显式 `Index("idx_user_id", "user_id")` 等);engine.py 三函数按 Interfaces 签名
- [ ] **Step 4: 跑测试确认通过** — 同 Step 2 命令,PASS;全套 `uv run pytest` 绿
- [ ] **Step 5: 提交** — `git commit -m "feat(db): SQLAlchemy 异步骨架——四模型对齐用户 DDL + engine/工厂/ping"`

### Task 4: repositories 四件套

**Files:**
- Create: `app/db/repositories.py`
- Test: `tests/test_repositories.py`

**Interfaces:**
- Consumes: Task 3 的 Base/models
- Produces(后续 Task 5/6/9 依赖,签名精确):
  - `async def get_or_create_conversation(session: AsyncSession, user_id: str) -> Conversation`
  - `async def add_message(session: AsyncSession, conversation_id: int, *, role: str, content: str | None = None, tool_calls: list[dict] | None = None, tool_call_id: str | None = None) -> Message`
  - `async def search_faq(session: AsyncSession, keyword: str, *, limit: int = 5) -> list[Faq]`
  - `async def create_ticket(session: AsyncSession, conversation_id: int, description: str, ticket_type: str) -> Ticket`
- 事务约定:函数内只 `flush()`,commit 归调用方(session 上下文管理)

- [ ] **Step 1: 写失败测试**(SQLite 内存 session fixture,`async with session.begin()` 包裹调用)——
  - get_or_create 两次同 user_id 返回同一行,不同 user_id 各建一行
  - add_message 写 tool 行(role='tool'、tool_call_id、content)与 assistant 行(tool_calls=[{"name":"query_faq","args":{},"id":"call_x"}] JSON 往返相等)
  - search_faq:预置 question「退货政策是什么?」→ keyword「退货」命中;keyword「邮费」空列表;keyword 含 `%`/`_` 时按字面匹配(预置 question「满100减5_专享」验证转义后仍按字面命中);空 keyword 返回空列表
  - create_ticket:同日两张 → 序号 001、002,格式 `T+当日YYYYMMDD+序号`;mock 主键冲突(首两次返回已存在工单号)后重试成功 → `test_create_ticket_collision_retry` 断言最终落库且 attempts 语义不外泄(返回 Ticket 正常)
- [ ] **Step 2: 跑测试确认失败** — `uv run pytest tests/test_repositories.py -v`,FAIL(模块不存在)
- [ ] **Step 3: 实现** — get_or_create:`select(Conversation).where(user_id==…).order_by(Conversation.id.desc()).limit(1)`,无则 INSERT;search_faq:`question.like(f"%{escaped}%", escape="\\")`,转义 `\`→`\\`、`%`→`\%`、`_`→`\_`(注意 Review Focus #2);create_ticket:当日前缀 `select(func.count()).where(Ticket.ticket_no.like(prefix+"%"))` + 1 生成候选,IntegrityError 时 flush 回滚当前 savepoint 重试(至多 3 次)
- [ ] **Step 4: 跑测试确认通过** — 同 Step 2,PASS;全套绿
- [ ] **Step 5: 提交** — `git commit -m "feat(db): 会话/消息/FAQ/工单仓储——LIKE 转义与工单号冲突重试"`

### Task 5: seed 幂等灌数据

**Files:**
- Create: `app/db/seed.py`(含 `async def seed(session_factory) -> dict[str, int]` 与 `if __name__ == "__main__"` 入口:读 get_settings().database_url 建引擎、跑 seed、打印各表行数、dispose)
- Test: `tests/test_seed.py`

**Interfaces:**
- Consumes: Task 3 模型/engine、Task 4 repositories
- Produces: `uv run python -m app.db.seed` 幂等灌数据(faq 8 条、演示会话 1 通 + 消息 3 条、演示工单 1 张);live 验收与 query_faq 工具依赖其中 faq 行

- [ ] **Step 1: 写失败测试** — 同一 SQLite 工厂上连跑 `seed(f)` 两次:faq 恒 8 条;演示会话(user_id="seed-demo")只 1 通、其消息恒 3 条(role user/assistant/tool 各一,tool 行带 tool_call_id);tickets 表恒 1 张。另断言:8 条 faq 的 question 均不含「邮费」「运费」子串(漏召回验收的前提);question 含「退货政策」的行存在,answer 含「七天」(acceptance ⑤ 判据词)
- [ ] **Step 2: 跑测试确认失败** — `uv run pytest tests/test_seed.py -v`,FAIL
- [ ] **Step 3: 实现** — faq 8 条(问题/答案/分类,答案口语文风):①退货政策是什么?(七天无理由,质量问题是商家承担运费——答案可提「运费」但 question 不得含)/②怎么申请换货?/③一般多久发货?/④保修政策是什么?/⑤怎么开发票?/⑥支持哪些支付方式?/⑦会员积分怎么用?/⑧优惠券怎么使用?;幂等:faq 以 count>0 跳过、演示会话以 user_id="seed-demo" 存在即跳过、演示工单以固定号 `T{当日}901` 存在即跳过
- [ ] **Step 4: 跑测试确认通过** — 同 Step 2,PASS;全套绿
- [ ] **Step 5: 提交** — `git commit -m "feat(db): 幂等 seed——faq 8 条 + 演示会话/消息 + 示例工单(邮费词条刻意缺席)"`

### Task 6: 五个业务工具(@tool)

**Files:**
- Create: `app/tools/__init__.py`(导出 `get_all_tools() -> list`,注册顺序 query_order/query_product/query_logistics/query_faq/create_ticket)
- Create: `app/tools/ecommerce.py`、`app/tools/knowledge.py`、`app/tools/tickets.py`
- Test: `tests/test_tools_ecommerce.py`、`tests/test_tools_db.py`

**Interfaces:**
- Consumes: Task 4 的 search_faq/create_ticket
- Produces(Task 7/9/10 依赖):
  - `query_order(order_id: str) -> str`、`query_product(product_id: str) -> str`、`query_logistics(order_id: str) -> str`——LangChain StructuredTool(args_schema 为 Pydantic 模型),返回中文 JSON 文本(`json.dumps(..., ensure_ascii=False)`)
  - `query_faq(keyword: str, *, session_factory: Annotated[async_sessionmaker, InjectedToolArg]) -> str`;`create_ticket(description: str, ticket_type: Literal["售后","投诉","咨询"], *, conversation_id: Annotated[int, InjectedToolArg], session_factory: Annotated[async_sessionmaker, InjectedToolArg]) -> str`
  - 两个 DB 工具内部自开 session(`async with session_factory() as s, s.begin()`),InjectedToolArg 不入 args_schema
  - `from langchain_core.tools import InjectedToolArg`(实现时经 Context7 复核导入路径)

- [ ] **Step 1: 写失败测试** —
  - test_tools_ecommerce.py:`random.seed(42)` 下三工具 invoke 返回可 `json.loads` 的中文文本;query_order 含 order_id 回显与字段 商品名/金额/订单状态;query_product 含 价格/库存/评分;query_logistics 含 承运商(∈{顺丰速运,中通快递,圆通速递,京东物流})、轨迹列表 2~4 条、预计送达;三者 `.args_schema.model_fields` 键正确、description 非空
  - test_tools_db.py:SQLite 工厂预置一条 faq → `query_faq.ainvoke({"keyword":"退货","session_factory":f})` 命中含答案文本;keyword「邮费」→ 返回文本含「未找到」;`create_ticket.ainvoke({...,"conversation_id":cid,"session_factory":f})` 返回含工单号,库里该行 status='待处理';**Schema 排除断言**:`create_ticket.args_schema.model_fields` 键集 == {description, ticket_type}(conversation_id/session_factory 被注入排除)
- [ ] **Step 2: 跑测试确认失败** — `uv run pytest tests/test_tools_ecommerce.py tests/test_tools_db.py -v`,FAIL
- [ ] **Step 3: 实现** — @tool + Pydantic args_schema(Field 中文描述);mock 用 `random.choice/uniform/randint`;query_faq 调 search_faq 拼「问:…\n答:…」列表;create_ticket 调 create_ticket 返回「已创建工单 {ticket_no},状态:待处理」;ticket_type 枚举描述写清三类
- [ ] **Step 4: 跑测试确认通过** — 同 Step 2,PASS;全套绿
- [ ] **Step 5: 提交** — `git commit -m "feat(tools): 五业务工具——订单/商品/物流 mock + FAQ LIKE 查询 + 建工单(InjectedToolArg 注入)"`

### Task 7: ToolRegistry(校验/超时重试/错误捕获/上下文注入)

**Files:**
- Modify: `app/tools/registry.py`(新建)
- Test: `tests/test_tool_registry.py`

**Interfaces:**
- Consumes: Task 6 的五工具
- Produces(Task 9/10 依赖,签名精确):
  - `@dataclass class ToolContext: conversation_id: int; session_factory: Any`
  - `@dataclass class ToolExecutionResult: ok: bool; content: str; error: str | None = None; attempts: int = 1`
  - `class ToolRegistry:`
    - `def __init__(self, timeout_seconds: float = 10.0, max_retries: int = 1) -> None`
    - `def register(self, tool) -> None`(重名 raise ValueError)
    - `@property def tools(self) -> list`(注册顺序)
    - `def get(self, name: str) -> object | None`
    - `async def execute(self, name: str, arguments: dict, context: ToolContext | None = None) -> ToolExecutionResult`

- [ ] **Step 1: 写失败测试** —
  - register/get/tools 顺序;重复注册同名 raise ValueError;get 未知返回 None
  - execute 未知工具 → ok=False、error 含「未知工具」、content 为可回灌错误文本
  - 校验失败(缺必填参数)→ ok=False、error 含「参数校验失败」
  - 工具抛异常 → ok=False、error 含异常信息、attempts == 1 + max_retries(异常也重试,spec §7.4)
  - 超时(`asyncio.sleep` 慢工具 + timeout_seconds=0.05)→ ok=False、error 含「超时」、attempts == 1 + max_retries
  - **隐藏参数注入**:工具函数签名带 `session_factory`/`conversation_id` 隐藏参数(不在 args_schema)时,execute 从 context 补进调用;普通工具不收多余键
  - 成功路径:ok=True、content 为工具返回文本、attempts==1
- [ ] **Step 2: 跑测试确认失败** — `uv run pytest tests/test_tool_registry.py -v`,FAIL
- [ ] **Step 3: 实现** — execute 顺序:查名 → `tool.args_schema.model_validate(arguments)` → 隐藏参数集 = 函数签名参数名 − schema 属性名,把 context 同名字段塞入 payload → 循环 1+max_retries 次 `asyncio.wait_for(tool.ainvoke(payload), timeout)`,TimeoutError/Exception 均记错误并重试,穷尽后 ok=False;content 恒为「可回灌给模型的文本」(失败时 `工具执行失败: {error}`)
- [ ] **Step 4: 跑测试确认通过** — 同 Step 2,PASS;全套绿
- [ ] **Step 5: 提交** — `git commit -m "feat(tools): ToolRegistry——注册管理/参数校验/超时重试/错误捕获/隐藏参数注入"`

### Task 8: SSE 工具帧

**Files:**
- Modify: `app/api/sse.py`
- Test: `tests/test_sse.py`(追加,既有 4 测不动)

**Interfaces:**
- Produces(Task 9/12 依赖):`format_tool_event(name: str, status: str, args: dict | None = None, summary: str | None = None) -> str` — 帧体 `{"tool":{"name":…,"status":…}}`,`args`/`summary` 非 None 才拼入;紧凑分隔符、ensure_ascii=False、`data: ` 前缀 + `\n\n` 结尾,风格与 format_delta_chunk 一致

- [ ] **Step 1: 写失败测试**(追加到 tests/test_sse.py)

```python
def test_format_tool_event_running() -> None:
    assert format_tool_event("query_logistics", "running", args={"order_id": "1001"}) == (
        'data: {"tool":{"name":"query_logistics","status":"running","args":{"order_id":"1001"}}}\n\n'
    )

def test_format_tool_event_done_with_summary() -> None:
    assert format_tool_event("query_faq", "done", summary="命中 1 条") == (
        'data: {"tool":{"name":"query_faq","status":"done","summary":"命中 1 条"}}\n\n'
    )
```

- [ ] **Step 2: 跑测试确认失败** — `uv run pytest tests/test_sse.py -v`,FAIL(ImportError)
- [ ] **Step 3: 实现** — format_tool_event 按接口拼帧
- [ ] **Step 4: 跑测试确认通过** — 同 Step 2,PASS;全套绿
- [ ] **Step 5: 提交** — `git commit -m "feat(api): SSE 工具状态帧 format_tool_event(running/done)"`

### Task 9: 编排两段式 + 写穿落库

**Files:**
- Create: `app/services/persistence.py`
- Modify: `app/services/chat.py`(stream_chat_reply 重构)
- Test: `tests/test_persistence.py`、`tests/test_chat_with_tools.py`

**Interfaces:**
- Consumes: Task 3/4(DB)、Task 7(registry/ToolContext)、Task 8(format_tool_event)、既有 store/trimmer/template
- Produces(Task 10 依赖):
  - `class ChatPersistence:` `def __init__(self, session_factory) -> None`;`async def ensure_conversation(self, user_id: str) -> int`;`async def log_user_message(self, conversation_id: int, content: str) -> None`;`async def log_assistant_tool_call(self, conversation_id: int, content: str | None, tool_calls: list[dict]) -> None`;`async def log_tool_result(self, conversation_id: int, tool_call_id: str, content: str) -> None`;`async def log_assistant_message(self, conversation_id: int, content: str) -> None`(每次调用独立 session,写完即提交)
  - `stream_chat_reply(model, registry, store, trimmer, system_template, persistence, session_id, message) -> AsyncIterator[str]`(位置/关键字同 ch01 风格,registry 与 persistence 为新增两参)

- [ ] **Step 1: 写失败测试** —
  - test_persistence.py:SQLite 工厂上 ensure_conversation 幂等(两次同 id)、五方法各落一行且字段对(tool_calls JSON、tool 行带 tool_call_id)
  - test_chat_with_tools.py(`FakeToolModel`:第一次 astream 若 messages 无 ToolMessage 则 yield 单个含 tool_calls 的 AIMessageChunk,否则 yield 文本 chunks;记录每次收到的 messages 与 bind 调用次数)——
    1. `test_tool_round_frames_and_stream`:SSE 帧序列 = 工具 running 帧 → done 帧 → 最终回答 deltas → [DONE];store 历史为 [Human, AIMessage(tool_calls), ToolMessage, AIMessage(最终)]
    2. `test_tool_round_persists_rows`:DB 四行 —— user / assistant(content=None, tool_calls 非空)/ tool(tool_call_id 对上)/ assistant(最终文本)
    3. `test_tool_round_without_first_text`:首段无文本(Review Focus #1)——assistant 行 content IS NULL,SSE 无首段 delta
    4. `test_single_round_binding`:bind_tools 仅被调 1 次,第二次 astream 收到的 messages 含 ToolMessage 且模型未再绑工具(fake bind_count==1)
    5. `test_second_turn_with_tool_history`:store 预置上一轮 4 条历史(含 ToolMessage),新问题不走工具(fake 直接文本)→ trimmer 正常裁剪不炸、SSE 与 ch01 同形
    6. `test_no_tool_round_unchanged`:无 tool_calls 路径帧序列与 ch01 完全一致(delta→[DONE]),落 user+assistant 两行
    7. `test_tool_error_still_converges`:registry.execute 返回 ok=False → done 帧照发(summary 含「失败」)、ToolMessage 回灌错误文本、最终回答照常流式、[DONE]
- [ ] **Step 2: 跑测试确认失败** — `uv run pytest tests/test_persistence.py tests/test_chat_with_tools.py -v`,FAIL
- [ ] **Step 3: 实现** — persistence.py 五方法(经 Task 4 repositories);chat.py 按 spec §6 流程:写穿 user 行 → trim → `bound = model.bind_tools(registry.tools)` → 第一段流式(逐 chunk 吐 delta 同时 `aggregated = aggregated + chunk`)→ 聚合含 tool_calls 则逐个:running 帧 → `registry.execute(name, args, ToolContext(conversation_id=…, session_factory=persistence 的工厂))` → done 帧(summary=result.content 截断 120 字)→ 落 assistant(tool_calls)/tool 行 → 组 ToolMessage 列表 → 第二段 `model.astream(第一段messages + [AIMessage(tool_calls), *ToolMessages])`(不绑工具)→ 落最终 assistant 行 → store 一次性 append [Human, AIMessage(tool_calls), *ToolMessages, AIMessage(最终)];无 tool_calls 走 ch01 原路径;异常处理块保持 ch01 语义
- [ ] **Step 4: 跑测试确认通过** — 同 Step 2,PASS;全套绿(此时旧 4 个端点测试仍绿——端点还没接新参数,接缝在 Task 10)
- [ ] **Step 5: 提交** — `git commit -m "feat(chat): 两段式单轮工具链——状态帧/回灌收敛/写穿落库"`

### Task 10: 端点与 lifespan 接缝

**Files:**
- Modify: `app/api/chat.py`(增 `get_session_factory()` 接缝;chat_stream 组装 persistence + 取 app.state.tool_registry 传入)
- Modify: `app/main.py`(lifespan:build_engine → ping fail-fast(RuntimeError 消息含操作提示)→ app.state.session_factory / app.state.tool_registry = build_default_registry(get_settings());关停 dispose)
- Modify: `app/tools/__init__.py`(增 `build_default_registry(settings) -> ToolRegistry`,注册五工具,timeout/retry 取自 settings)
- Test: `tests/test_chat_endpoint.py`(FakeModel 补 bind_tools;client fixture 增 monkeypatch get_session_factory→SQLite 工厂、app.state.tool_registry=真 ToolRegistry(空注册或含五工具均可);既有断言一字不改)、`tests/test_factory.py` 不动

**Interfaces:**
- Consumes: Task 1 settings、Task 3 engine、Task 7 registry、Task 9 persistence/stream_chat_reply 新签名
- Produces: 完整可运行应用(启动即 ping DB,失败给出「docker compose up -d + 建表 DDL」提示)

- [ ] **Step 1: 写失败测试** — test_chat_endpoint.py 追加两条(不改旧断言):①`test_endpoint_binds_registry_tools`:fake.bind_tools 收到的列表长度 == 5(真 build_default_registry)②`test_engine_ping_failure_message`:对坏 DSN 的 engine 调 ping → RuntimeError 消息含「docker compose」。先跑确认 FAIL
- [ ] **Step 2: 实现** — 按 Interfaces 接线;旧测试因缺 get_session_factory/registry 接缝会红,fixture 补丁后回绿;全套 `uv run pytest` ≥ 40 passed 全绿
- [ ] **Step 3: 提交** — `git commit -m "feat(api): 端点接 registry/persistence 接缝,lifespan ping fail-fast + build_default_registry"`

### Task 11: 工具选型评估集(标注样例替代 TDD)

**Files:**
- Create: `evals/tool_cases.jsonl`(18 条:`{"id","query","expected_tool"}`)
- Create: `evals/run_tool_eval.py`

**Interfaces:**
- Consumes: Task 6 五工具(get_all_tools)、Task 1 settings
- Produces: `uv run python evals/run_tool_eval.py` live 跑分(门禁:整体 ≥90%、闲聊类误调 0、「邮费」样例必须选 query_faq);`--self-test` 离线自检(假模型固定应答,不花 key)

- [ ] **Step 1: 写评估集** — 18 条:query_order ×4(「订单 3003 付款了吗」「我订单 2002 现在什么状态」「查一下订单 1001」「帮我看看订单 5005 用什么方式付的款」)、query_product ×3、query_logistics ×4(含「订单 1001 的物流到哪了」)、query_faq ×3(含「退货政策是什么」)、create_ticket ×2(「再不处理我就投诉了,给我转人工」「商品摔坏了必须退货」)、none ×2(「你好,在吗」「谢谢」)+ 1 条漏召回记录样例「邮费是多少」(expected_tool=query_faq,id 标 `faq-postage-miss`,备注该样例工具应选中但 LIKE 查空,属预期漏召回)
- [ ] **Step 2: 写 runner** — 仿 evals/run_eval.py 骨架:load .env → init_chat_model → bind get_all_tools() → 逐条 invoke(非流式)→ `ai.tool_calls[0]["name"]`(无则 None)→ 对分;门禁三条;`--self-test` 用固定 fake(工具名回显规则)自验 runner 逻辑;输出 PASS/FAIL 明细 + 汇总
- [ ] **Step 3: 验证** — `uv run python evals/run_tool_eval.py --self-test` PASS(live 跑分在 Task 14 执行并归档 evals/report.md)
- [ ] **Step 4: 提交** — `git commit -m "test(evals): 工具选型评估集 18 例 + runner(门禁 ≥90%,含邮费漏召回记录样例)"`

### Task 12: 聊天页工具徽章(Vibe Coding 例外)

**Files:**
- Modify: `app/static/chat.html`

**不做 TDD、不派评审(用户明示例外)。预期效果(用户验收后迭代):**

- [ ] **Step 1: 气泡结构拆分** — appendBubble 的 `.bubble` 内含 `.badges`(工具轨迹行)+ `.bubble-text`(正文 span);appendText/showTyping 只操作 .bubble-text(修 gotFirst 清空逻辑指向正文 span)
- [ ] **Step 2: 帧处理分支** — SSE 循环增 `payload.tool` 分支:status=running → badges 行加 `.tool-badge.running` 徽章(文案 `⚒ {name}…`,闪烁);status=done → 徽章转 `.done`(文案 `✓ {name}`,title 属性设 summary 供悬浮)
- [ ] **Step 3: 像素风样式** — 徽章沿用像素边框/硬阴影变量,running 用 --blue 闪烁、done 用 #2fbf71
- [ ] **Step 4: 手工验证** — `uv run uvicorn app.main:app` 起服务,浏览器问「订单 1001 的物流到哪了」:徽章出现→变绿→正文流式;普通聊天(你好)无徽章;报错路径不回归
- [ ] **Step 5: 提交** — `git commit -m "feat(ui): 气泡内工具轨迹徽章——running 闪烁/done 落定+summary 悬浮(Vibe Coding 例外)"`

### Task 13: acceptance.sh 扩展 + README

**Files:**
- Modify: `scripts/acceptance.sh`(增验收④⑤⑥,沿用 UTF-8 临时文件 + Python delta 合并判据模式)
- Modify: `README.md`(快速开始增 Docker/seed 两步;功能节增 Ch02;路线图勾 Ch02、LangGraph 顺延;项目结构树增 db/tools)

**Interfaces:**
- Consumes: Task 5 种子数据判据词(退货政策答案含「七天」)、Task 8 工具帧格式
- Produces: `bash scripts/acceptance.sh` 六判据

- [ ] **Step 1: 扩展 acceptance.sh** — ④ 物流:POST「订单 1001 的物流到哪了」→ 判 `data: {"tool"` 帧存在 + 末事件 [DONE] + 打印合并回答;⑤ FAQ 命中:POST「退货政策是什么」→ 工具帧存在 + 合并 delta 含「七天」;⑥ 漏召回演示:POST「邮费是多少」→ 判工具帧含 query_faq(命中即 PASS),完整回答照打留痕、措辞不判分;汇总行更新为 6 判据
- [ ] **Step 2: 更新 README** — 按上述四处
- [ ] **Step 3: 语法冒烟** — `bash -n scripts/acceptance.sh` 通过(实跑在 Task 14)
- [ ] **Step 4: 提交** — `git commit -m "docs+test(scripts): 验收脚本扩至 6 判据;README 快速开始/路线图同步 Ch02"`

### Task 14: live 验收 + 全分支评审 + 合并交付

**Files:**
- Modify: `evals/report.md`(追记 Ch02 live 评估结果)、`dev-notes/ch02.md`(controller 追记)

- [ ] **Step 1: live 全链路** — docker compose up -d(库健康)→ `uv run python -m app.db.seed` → `uv run uvicorn app.main:app` → `bash scripts/acceptance.sh` 6/6 PASS → `uv run python evals/run_tool_eval.py` 门禁 PASS(若百炼端点 tool-calling 异常 → 停下问用户,工作要求 #4)
- [ ] **Step 2: 全分支 code review** — review-package(MERGE_BASE=9271ea0)→ 最强模型评审 → 单轮修复 + 定向复审 → 残留裁决
- [ ] **Step 3: 归档与合并** — evals/report.md、dev-notes 交付节;`git checkout main && git merge ch02/function-calling`;交付说明(演示命令/测试结果/文档路径)

## Self-Review

1. **Spec 覆盖**:§2 四项做→T1/T2(基建+建表)、T4-T7(工具+registry)、T8-T10(接线)、T9+T12(写穿+徽章)、T11(评估)、T13(验收);§5.2 新帧→T8;§7.1-7.7→T3/T4/T6/T7/T9/T10/T12;§8→T9/T11/T13;§9→T7(错误回灌)/T10(ping);§10 三验收→T13④⑤⑥。无缺口。
2. **步骤扫描**:每步单义;T3 索引/枚举、T4 转义规则、T5 八条 FAQ 内容、T9 帧序列均已钉死;无 TBD。
3. **类型一致**:ToolContext/ToolExecutionResult/ChatPersistence/repositories 签名在 T4→T6→T7→T9→T10 传递处逐一比对一致;`build_default_registry` 定义于 T10 文件列表(app/tools/__init__.py)而 T9 测试用真 ToolRegistry 直构,不依赖该函数——顺序安全。
4. **Review Focus**:五条各归 T4(#2/#5)、T9(#1/#3/#4)并有测试名钉住。
5. **比例**:计划以签名+断言为主,无成段实现体,与 spec 比例约 1:1.5。
