# Ch03 设计文档 — RAG 基础:向量语义检索知识库

- 日期:2026-09-28
- 状态:设计经计划审批通过(ExitPlanMode);spec 评审中
- 上章:`2026-09-27-ch02-function-calling-design.md`
- 用户 DDL(建表唯一依据):`J:\code\SQL\ch03-ddl.sql`,原样复制入仓库 `scripts/sql/ch03-ddl.sql`(自带 `SET NAMES utf8mb4`,ch02 R5 教训已吸收)

## 1. 背景与目标

ch02 留了两个钩子:acceptance ⑥「邮费是多少」LIKE 查空(已知漏召回)、评估样例 `faq-postage-miss`。本章根治它:把 `query_faq` 内部实现从关键词查表升级为**向量语义检索**(dense 单路),**工具入参出参契约不变**。同时建起离线建库子系统:知识文档结构感知切分、历史客服对话挖知识、MySQL(原文权威源)× Milvus Lite(向量)双写。验收:「邮费是多少」换说法能召回运费说明并答对;中断建库重跑,漏向量化的块被捡起补齐。

## 2. 范围

**做:**
1. 离线建库·文档处理:Markdown 知识文档(退货政策/商品 FAQ/售后手册)结构感知切分——标题层级切 section、超长递归切(段落→句子)、块间重叠裁到最近句号、大表格按行切每块复制表头
2. 离线建库·对话挖知识:CLI 定时任务(调度交部署侧),读 MySQL 历史会话分批喂 LLM 抽问答对 → qa_extraction_staging 暂存 → 整体去重 → 入库
3. 落库结构:每条知识 category/questions/answer 三格拼文本向量化;四类元数据(章节路径/内容类型/关键条款/前后块指针)只存不进向量
4. 双写:MySQL `knowledge_chunks`(pending→done、vector_id 回填)× Milvus Lite `knowledge` 集合(主键=MySQL id,upsert 幂等)
5. 在线检索:问题向量 → Milvus Top-K → MySQL 回表,替换 query_faq 内部实现

**不做:** 关键词召回/混合检索/重排(dense 单路);语义级(嵌入相似度)去重;知识库管理界面;faq 旧表迁移或删除(原地保留,query_faq 不再读它);mine_qa 的调度器实现(只交付可独立执行的命令);Agent 循环(LangGraph,后续章);Langfuse(后续章)。

## 3. 已确认决策记录

| # | 决策 | 用户原话/依据 |
|---|------|--------------|
| D1 | 嵌入 = SiliconFlow API(BAAI/bge-m3,1024 维,OpenAI 兼容 /embeddings) | AskUserQuestion 选定;不装 torch,与现有 OpenAI 协议同构 |
| D2 | 挖知识 job = CLI 命令,调度交部署侧 | "就搞个命令能启动的 job 就行,我们可以自己选择部署或者直接执行" |
| D3 | 对话语料 = MySQL 写穿库,ch03 seed 补灌历史会话 | 用户选定;source_ref=conversation_id |
| D4 | **Milvus = Lite 嵌入式**(`pymilvus[milvus-lite]`,`MilvusClient('data/milvus/baihelp.db')` 本地文件库),零额外容器,**不用 docker standalone** | 用户纠偏:"Milvus 使用下面的轻量版来部署,不要用 standalone";docker-compose 零改动 |
| D5 | 装机可行性已查证:milvus-lite 3.2.1 纯 Python 架构,faiss-cpu 1.15.1 / pyarrow 25.0.1 均有 cp314 win_amd64 wheel | PyPI 实查(dev-notes 阶段 0.5) |
| D6 | 技术选型定死,走不通停下问 | 工作要求④ |
| D7 | query_faq 契约不变:args 仍 `keyword: str`,出参仍「问:…\n答:…」文本列表;唯一微调 = description 改「用户问题原话或关键词」 | "工具的入参出参契约保持不变";ch02 eval 19 例重跑验回归 |
| D8 | 验收⑥语义反转:「邮费是多少」从漏召回留痕改为命中判分 | 本章验收标准① |
| D9 | Phase A 幂等 = 自然键(category, section_path, questions, answer)去重重入;语料改动导致的旧行残留为已知限制,课程规模可接受 | DDL 无指纹列,不新增列;重跑不产生重复块 |

## 4. 架构

延续分层单体,新增 `app/knowledge/`;docker-compose **不加任何服务**:

```
app/knowledge/
├── __init__.py
├── chunker.py        # Markdown 结构感知切分(纯逻辑,TDD 主战场;含「问法:」行提取)
├── corpus.py         # data/knowledge/*.md 读取 + front-matter 解析
├── embedding.py      # SiliconFlow BGE-M3 客户端(OpenAI 兼容 /embeddings)
├── milvus_repo.py    # MilvusClient 本地库:建集合/主键 upsert/top-K 检索
├── ingest.py         # 建库编排:MySQL(pending)→ Milvus → 回填 vector_id(done)
├── mine_qa.py        # 挖知识:分批抽取 → 暂存表 → 整体去重 → 入库
├── retriever.py      # KnowledgeRetriever:Top-K → MySQL 回表(query_faq 内部)
└── normalize.py      # 问题规范化(去空白标点、全半角归一,去重与测试共用)
data/knowledge/       # 语料:退货政策.md / 商品FAQ.md / 售后手册.md
data/milvus/          # baihelp.db 本地向量库(进 .gitignore)
scripts/sql/ch03-ddl.sql
```

CLI:`uv run python -m app.knowledge.ingest`(建库,幂等可重跑)、`uv run python -m app.knowledge.mine_qa`(挖矿,均带 `--help` 与统计输出)。

## 5. 组件设计

### 5.1 语料文件(data/knowledge/*.md)

三份 Markdown,front-matter 声明元信息(手写解析,不引 pyyaml):

```
---
category: 售后服务            # 该文档的根分类(进向量文本的 category 基底)
content_type: policy          # faq / policy / manual
key_clauses:                  # 关键条款章节路径列表(命中 is_key_clause=1)
  - 售后服务 > 退货政策 > 退货时效
---
```

- **退货政策.md**(policy):含「退货时效」(七天无理由,判据词「七天」)、「**运费说明**」(订单满 99 元包邮,不满收运费 8 元,偏远 12 元——验收①判据词「包邮」/「99」)、质量 vs 非质量责任划分等章节
- **商品FAQ.md**(faq):每节一个 `###` 标题=标准问法,节内首行 `- 问法: A / B / C` 列真实问法变体(chunker 提取进 questions 并从正文剔除,「问法:」为保留字约定);发货时效/支付方式/发票/优惠券等
- **售后手册.md**(manual):换货流程、保修条款、物流查询等章节路径较深(展示 section_path 与指针)

政策/手册型 questions=所在章节标题、category=上级标题路径(> 分隔);FAQ 型 questions=标准问法+变体(换行分隔)、category=front-matter 根分类。

### 5.2 chunker.py(纯逻辑)

`chunk_document(doc: KnowledgeDoc, *, max_chars=500, overlap_chars=80) -> list[KnowledgeChunk]`

1. 按标题层级(#/##/###)切 section,记 section_path(`根分类 > 章 > 节`,> 连接)
2. 超长 section 递归切:先按空行分段;段仍超长按句子切(。!?!?! 及换行为界);拼块不超 max_chars
3. 相邻块留 ~overlap_chars 重叠,重叠起点**回退到最近句号**,不留半截话;找不到句边界(超长无句读)则硬切
4. 表格(连续 `|` 开头行,含分隔行)为原子单元:整表不超长则整体入块;超长按行分块,**每块首行复制表头行+分隔行**
5. 产出:category、questions、answer(向量文本三格)、section_path、content_type、is_key_clause(front-matter key_clauses 命中);prev/next 由 ingest 插库后回填
6. 代码围栏(```)内内容不参与切分边界判定

### 5.3 embedding.py

`BgeM3Embedder`:openai SDK(现有依赖)指向 SiliconFlow:`client.embeddings.create(model="BAAI/bge-m3", input=[...])`,批内多条文本一次请求;异步;dim=settings.embedding_dim(1024)。SiliconFlow /embeddings 兼容参数以 Context7+实测核对,异常即停问(D6)。

### 5.4 milvus_repo.py

`MilvusKnowledgeRepo(path)`,包 pymilvus `MilvusClient`:

- `ensure_collection()`:schema 路径建集合 knowledge——`create_schema(auto_id=False)`、`id INT64 is_primary`、`vector FLOAT_VECTOR dim=1024`、`IndexParams(metric_type="COSINE")`;has_collection 判重,幂等
- `upsert_vectors(rows: list[{id, vector}])`:按主键 upsert(auto_id=False 才支持,正合设计);重跑幂等
- `search(query_vector, top_k) -> list[tuple[int, float]]`:返回 (id, score),score = 1 − distance(COSINE 距离与相似度换算,实现时以实测校准)
- `close()`

### 5.5 ingest.py(建库编排,验收②主体)

```
Phase A(结构落库):  读语料 → chunk → 自然键去重(D9)→ INSERT pending
                     → 同事务第二遍回填 prev/next 指针(同文档内链, mined 行为 NULL)
Phase B(向量化):    捞 vectorize_status=pending → 批量 embed(三格拼文本:
                     "分类:{category}\n问:{questions}\n答:{answer}")
                     → milvus upsert(主键=MySQL id)→ 回填 vector_id + status=done
```

- **幂等**:Phase A 自然键跳过已存在块(比较时 NULL 归一为空串,挖矿行 section_path 为 NULL 不漏判);Phase B 只处理 pending,upsert 按主键幂等——中断(embed 失败/进程被杀)后重跑,漏块自动补齐,验收②由此结构保证
- 向量文本拼接格式固定为常量(与检索侧无关,但评估与调试依赖其确定性)
- 每批 embed+upsert+回填为一个检查点:失败即中止,已完成的批保留 done,未处理留 pending

### 5.6 mine_qa.py(挖知识,需求②)

```
① 读历史:conversations/messages 取成对 user/assistant 正文
   (过滤:过短、纯工具轮 assistant、tool 行;每轮取最终 assistant 文本)
   按批(batch_no = QA-YYYYMMDD-NN)分组会话,每组一次 LLM 调用
② 抽取:LLM(Prompt 模板 app/prompts/qa_extraction_prompt.md,现有 loader 加载)
   输出 JSON 数组 [{source, question, answer}] → 写 qa_extraction_staging(extracted)
③ 去重入库:normalize(问题) 在暂存表内去重 + 对比 knowledge_chunks 已有 questions
   → 重复置 discarded,保留置 kept 并 INSERT knowledge_chunks
   (category="对话挖掘"、content_type='faq'、questions=真实问法、指针 NULL)
   → 落 pending,由 ingest Phase B 统一向量化(挖矿与建库共用一条 pending→done 通道)
```

- LLM 复用 `get_chat_model(settings)`;单批解析失败只报废该批(batch_no 留痕),不污染其他批
- `--batch-size` 可配;`--self-test` 离线自检(注入 fake LLM,不依赖 API key/MySQL);输出统计(会话数/抽取数/去重丢弃/入库数)
- 语义级去重明确不做(§2)

### 5.7 retriever.py + query_faq 改造(在线检索)

`KnowledgeRetriever`:异步 `retrieve(query: str, session_factory) -> list[RetrievedKnowledge]`(持 embedder + Milvus repo,MySQL 回表经传入的 session_factory)

```
embed(query) → milvus.search(vec, top_k=RETRIEVAL_TOP_K)
→ score 阈值过滤(RETRIEVAL_SCORE_THRESHOLD,默认 0.5)
→ MySQL 按 id IN(...) 回表(WHERE vectorize_status='done',原文权威源取最新文本)
→ 按 score 降序返回(category/questions/answer/score)
```

`query_faq` 改造:`keyword: str` 入参与「问:…\n答:…」出参格式逐字节沿用;新增 `retriever: Annotated[KnowledgeRetriever, InjectedToolArg]`(与 session_factory 同型注入,lifespan 组装真实实例,测试注入 FakeRetriever);空结果兜底话术原样保留;description 改为「按语义检索知识库,建议传入用户问题原话或关键词」。

### 5.8 seed 扩展(app/db/seed.py)

幂等补灌**历史客服对话**(挖矿语料):user_id=`seed-hist-<NN>` 共 8 通,每通 3~5 轮,话题覆盖退款到账时效、修改收货地址、优惠券过期、发票抬头、换货运费等(挖矿应产出的知识),并混入闲聊寒暄、无答案转人工等噪音(抽取应过滤);判据=user_id 存在即跳过,与既有三判据并列。

### 5.9 配置增量(app/core/config.py + .env.example)

| 键 | 默认 | 说明 |
|----|------|------|
| EMBEDDING_API_KEY | 必填(fail-fast 同 LLM_API_KEY) | SiliconFlow 密钥 |
| EMBEDDING_BASE_URL | https://api.siliconflow.cn/v1 | OpenAI 兼容端点 |
| EMBEDDING_MODEL | BAAI/bge-m3 | 嵌入模型 |
| EMBEDDING_DIM | 1024 | 向量维度 |
| MILVUS_DB_PATH | data/milvus/baihelp.db | Milvus Lite 本地库文件 |
| RETRIEVAL_TOP_K | 3 | 检索条数 |
| RETRIEVAL_SCORE_THRESHOLD | 0.5 | 相似度阈值 |
| CHUNK_MAX_CHARS / OVERLAP_CHARS | 500 / 80 | 切分参数 |
| QA_MINE_BATCH_SIZE | 4 | 每批会话数 |

## 6. 数据流

**在线(query_faq 被调用):** 用户消息 → 模型选 query_faq(keyword=用户问题)→ registry 注入 session_factory+retriever → retriever:embed → Milvus top-K → 阈值过滤 → MySQL 回表 → 拼「问:…\n答:…」→ 回灌模型 → 最终回答流式。Milvus/嵌入不可达时抛错,由 ToolRegistry 既有错误回灌机制兜底,SSE 照常收敛。

**离线(建库):** `mine_qa`(可选)→ staging → knowledge_chunks(pending);`ingest` → Phase A 结构落库 → Phase B embed+upsert+回填 done。演示顺序:建库/挖矿 job 先行,后起服务(进程内单机库跨进程并发为边缘场景,锁冲突由错误回灌兜底,文档标注)。

## 7. 测试与评估策略

- **TDD**:chunker 全规则(标题层级/递归/重叠裁句号/表格表头复制/问法行提取/元数据/围栏);normalize;ingest 幂等(fake embedder + fake milvus:自然键重入不重复、pending 补齐、指针回填、失败检查点);mine_qa(fake LLM:解析/过滤/去重/入库);retriever(fake 全家桶:回表过滤 done、阈值、排序、空结果);query_faq(契约锁定:入参 schema 键集、出参格式、FakeRetriever 注入);DDL↔ORM 一致性冒烟扩 knowledge_chunks + qa_extraction_staging;config 新键。既有 91 测试保绿
- **评估集替代 TDD**(工作要求①,Prompt/检索行为不可单测):
  - `evals/qa_mine_cases.jsonl` + `evals/run_mine_eval.py`:对话样例 → 期望抽出的问答对(问题要点+答案要点断言),`--self-test` 离线自检(仿 run_tool_eval.py),live 结果追记 evals/report.md
  - `evals/retrieval_cases.jsonl` + `evals/run_retrieval_eval.py`:问题 → 期望命中的 chunk 判据(含「邮费是多少」→ 运费块、「快递费怎么算」换说法样例、无关问题→低分不命中的反例),门禁:核心样例全中 + 反例零误召
- **acceptance.sh**:验收⑥反转为「邮费是多少」出 query_faq 帧 + 合并回复含「包邮」或「99」;⑤照旧判「七天」(路径已变向量);⑦新增:`python -m app.knowledge.mine_qa --self-test` 退出码 0(挖矿 job 可执行性判据);①②③④逐字不动

## 8. 错误处理

- embed / Milvus 写失败 → ingest 即刻中止,已完成批保留,未完成留 pending(失败即检查点,重跑补齐)
- 在线 Milvus/嵌入不可达 → 抛错 → ToolRegistry 错误文本回灌 → 模型向用户解释(既有机制,零新代码路径)
- 挖矿单批 LLM/解析失败 → 该批 staging 留 extracted 并日志留痕,继续后续批
- Settings 缺 EMBEDDING_API_KEY → 启动 fail-fast(同 LLM_API_KEY 惯例)

## 9. 交付与验收映射

| 验收标准 | 落点 |
|---------|------|
| ①「邮费是多少」换说法召回运费说明并答对 | 运费说明语料 + retriever + query_faq 改造;acceptance ⑥(新语义)+ retrieval eval |
| ②中断建库重跑漏块补齐 | ingest Phase B pending 补齐 + upsert 幂等;中断→重跑演示留痕 dev-notes |
| 契约不变 | query_faq args/出参锁定测试 + ch02 eval 19 例回归 |

交付物:演示命令、全套 pytest 结果、acceptance 全判据、双评估 live 报告、dev-notes/ch03.md、spec/plan 路径。

## 10. 依赖与风险

- 新依赖:`pymilvus[milvus-lite]`(版本以装机解析为准);openai SDK 复用现有;不引 pyyaml/langchain splitters/torch
- 风险1:milvus-lite 3.2.1 发布仅一个月,CI 主测矩阵未含 Windows+py3.14(wheel 已确认存在)→ Task 1 首步装机实测,不通停问(D6)
- 风险2:pymilvus ↔ milvus-lite 版本配对 → uv 解析冲突即停问
- 风险3:SiliconFlow /embeddings 参数兼容性(encoding_format/dim 参数)→ Context7+实测核对
- 风险4:COSINE 距离↔相似度换算方向 → 实现时以已知正反例实测校准阈值语义
- 风险5:ch02 eval 19 例受 description 微调影响 → 重跑对照,回归即调回

## 11. 开放问题

无。执行中走不通按工作要求④停下问。
