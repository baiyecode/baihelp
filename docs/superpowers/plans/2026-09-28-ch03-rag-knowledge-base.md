# Ch03 RAG 向量语义检索知识库 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `query_faq` 内部实现从 SQL LIKE 换成向量语义检索(契约不变),并建起离线建库子系统(文档结构感知切分 + 历史对话挖知识 + MySQL×Milvus Lite 双写)。

**Architecture:** 新增 `app/knowledge/` 包承载离线管线与在线检索;Milvus Lite(`pymilvus[milvus-lite]`)本地文件库零容器;MySQL `knowledge_chunks` 是原文权威源(pending→done 双写状态机,vector_id 回填),挖矿与建库共用同一条 pending→done 向量化通道;在线侧 Top-K 命中后按 id 回表取最新文本。

**Tech Stack:** FastAPI / SQLAlchemy 2.0 异步 + aiomysql / MySQL 8.4(Docker)/ langchain[openai](复用)/ openai SDK(SiliconFlow /embeddings,复用传递依赖)/ pymilvus[milvus-lite] / pytest + pytest-asyncio + aiosqlite。

**Spec:** `docs/superpowers/specs/2026-09-28-ch03-rag-knowledge-base-design.md`

## Global Constraints

- 技术选型定死:Milvus Lite 嵌入式(用户裁定,**不用 docker standalone**,docker-compose 零改动)+ MySQL 原文权威源 + BGE-M3(SiliconFlow API)+ dense 单路(不做关键词召回/混合检索/重排/语义去重);走不通停下问用户,不得自行换方案
- 涉及 pymilvus / milvus-lite / openai SDK / LangChain 的 API 用法,写码前用 Context7 核对(spec §5.4 已核MilvusClient 关键签名,实现时遇未知接口先查再写)
- MySQL 建表唯一依据 = `scripts/sql/ch03-ddl.sql`(用户 DDL `J:\code\SQL\ch03-ddl.sql` **逐字节原样**复制,首行已含 SET NAMES utf8mb4);应用代码不对 MySQL 跑 create_all
- `query_faq` 契约不变:args_schema 仍只有 `keyword: str`;出参仍是「问:…\n答:…」拼接、空结果兜底话术逐字保留(`未找到与「{keyword}」相关的 FAQ,请换个关键词或转人工客服。`);唯一可改 = 工具 description(改为建议传用户问题原话)
- 既有 91 个测试全数保绿,既有断言不许改(测试因新增必填配置项/新注入参数需要机械补参不算改断言,须在提交说明里点名)
- Milvus knowledge 集合:仅 `id`(INT64 主键,auto_id=False,=MySQL id)+ `vector`(FLOAT_VECTOR dim=1024)两字段,COSINE;score = 1 − distance
- 向量化文本拼接格式固定:`分类:{category}\n问:{questions}\n答:{answer}`
- Phase A 幂等自然键 = (category, section_path, questions, answer),比较时 NULL 归一空串
- 包管理 uv,Python 3.14;提交信息 conventional 前缀 + 中文描述;每个任务一个提交(或少量提交)
- `dev-notes/ch03.md` 由 controller 每阶段追记,实现者不写
- 单元测试一律离线密封:不碰真实 Milvus 文件之外的进程外服务、不调 SiliconFlow、不调 LLM(milvus-lite 是进程内库,允许在 tmp_path 用真库做集成测试)

## Review Focus

1. **建库中断留下 pending 残留**:embed 或 upsert 中途失败后重跑,漏块必须被补齐且不产生重复向量 —— Task 6 `test_phase_b_resumes_pending_after_failure` 钉
2. **超长表格按行切丢表头**:每个表格块首行必须是「表头行+分隔行」复制,丢了列语义就废 —— Task 3 `test_long_table_blocks_repeat_header` 钉
3. **重叠裁句号失败产出半截话**:重叠起点必须回退到最近句边界,无句边界才允许硬切 —— Task 3 `test_overlap_starts_at_sentence_boundary` 钉
4. **COSINE 距离方向搞反**(score=1−distance):相似文本得分必须高于不相似文本,阈值语义按此校准 —— Task 5 `test_search_scores_rank_similar_first` 钉
5. **query_faq 出参格式漂移破坏模型回灌**:多问法块的「问:」槽位换行会破坏行结构,必须以「 / 」连接为单行;兜底话术逐字不变 —— Task 9 `test_query_faq_output_format_locked` 钉

---

### Task 1: 依赖装机 + Milvus Lite 本机冒烟 + Settings 新键

**Files:**
- Modify: `pyproject.toml`(dependencies 增 `pymilvus[milvus-lite]`)
- Modify: `app/core/config.py`(新键)
- Modify: `.env.example`(新键注释块)、`.gitignore`(`data/milvus/`)
- Test: `tests/test_config.py`(追加)

**Interfaces:**
- Produces: `Settings` 新属性 `embedding_api_key: str`(必填,无默认)、`embedding_base_url: str = "https://api.siliconflow.cn/v1"`、`embedding_model: str = "BAAI/bge-m3"`、`embedding_dim: int = 1024`、`milvus_db_path: str = "data/milvus/baihelp.db"`、`retrieval_top_k: int = 3`、`retrieval_score_threshold: float = 0.5`、`chunk_max_chars: int = 500`、`chunk_overlap_chars: int = 80`、`qa_mine_batch_size: int = 4` —— 后续所有任务按这些名字消费

- [ ] **Step 1: 装机 + 本机冒烟(风险1 的实证,失败即停下问用户)**

```bash
uv add "pymilvus[milvus-lite]"
uv run python -c "
import os
os.makedirs('data/milvus', exist_ok=True)
from pymilvus import MilvusClient, DataType
c = MilvusClient('data/milvus/_smoke.db')
s = c.create_schema(auto_id=False)
s.add_field('id', DataType.INT64, is_primary=True)
s.add_field('vector', DataType.FLOAT_VECTOR, dim=8)
c.create_collection('smoke', schema=s)
c.insert('smoke', [{'id': 1, 'vector': [1.0]*8}])
r = c.search('smoke', data=[[1.0]*8], limit=1)
print('smoke-ok', r[0][0]['id'], round(r[0][0]['distance'], 4))
c.close()
os.remove('data/milvus/_smoke.db')
"
```

Expected: 输出 `smoke-ok 1 0.0`(距离 0 = 自身)。任何 wheel/导入/运行错误 → **停止并上报用户**(工作要求④,不换方案)。

- [ ] **Step 2: 写失败测试**

`tests/test_config.py` 追加:

```python
def test_settings_knowledge_keys():
    s = Settings(_env_file=None, llm_api_key="k", embedding_api_key="e")
    assert s.embedding_base_url == "https://api.siliconflow.cn/v1"
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.embedding_dim == 1024
    assert s.milvus_db_path == "data/milvus/baihelp.db"
    assert (s.retrieval_top_k, s.retrieval_score_threshold) == (3, 0.5)
    assert (s.chunk_max_chars, s.chunk_overlap_chars) == (500, 80)
    assert s.qa_mine_batch_size == 4

def test_settings_requires_embedding_key():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_api_key="k")
```

- [ ] **Step 3: 跑测试确认失败** —— `uv run pytest tests/test_config.py -v`,Expected: 新两测 FAIL(属性不存在)
- [ ] **Step 4: 实现** —— `app/core/config.py` 按 Interfaces 的名字与默认值逐键添加;`.env.example` 增注释块(含「SiliconFlow 注册 https://siliconflow.cn 拿 key」提示);`.gitignore` 增 `data/milvus/`
- [ ] **Step 5: 全套测试** —— 既有构造 `Settings(_env_file=None, llm_api_key=...)` 的测试文件按需机械补 `embedding_api_key="e"`(grep `_env_file=None` 全量排查);`uv run pytest`,Expected: 全 PASS(91 + 新 2)
- [ ] **Step 6: Commit** —— `feat(knowledge): pymilvus[milvus-lite] 装机冒烟 + Settings 知识库配置键`

---

### Task 2: ch03 DDL 入仓 + ORM 两表 + 一致性冒烟

**Files:**
- Create: `scripts/sql/ch03-ddl.sql`(用户 DDL 逐字节复制,内容=J:\code\SQL\ch03-ddl.sql 全文)
- Modify: `app/db/models.py`(KnowledgeChunk / QaExtractionStaging)
- Test: `tests/test_ddl_orm_consistency.py`、`tests/test_db_models.py`(追加)

**Interfaces:**
- Produces: ORM `KnowledgeChunk`(id, category, questions, answer, section_path, content_type, is_key_clause, prev_chunk_id, next_chunk_id, vector_id, vectorize_status, created_at, updated_at;self-FK prev/next → knowledge_chunks.id,mysql 端 BigInteger 主键沿用 ch02 的 sqlite Integer variant 惯例,`vectorize_status` 取值 {'pending','done'})、`QaExtractionStaging`(id, batch_no, source_ref, question, answer, status ∈ {'extracted','kept','discarded'})—— Task 6/8/9 按此消费

- [ ] **Step 1: DDL 原样入仓** —— `cp J:/code/SQL/ch03-ddl.sql scripts/sql/ch03-ddl.sql` 后 `git diff --no-index` 校验逐字节一致(唯一允许差异:无;ch03 源文件已自带 SET NAMES)
- [ ] **Step 2: 写失败测试** —— 一致性冒烟扩展为同时解析 ch02/ch03 两份 DDL(列名集合 vs ORM 元数据,复用既有测试的解析函数);`test_db_models.py` 追加:sqlite create_all 建出两新表 + KnowledgeChunk 自 FK + 默认值 vectorize_status='pending' 落库可读
- [ ] **Step 3: 跑测试确认失败** —— Expected: FAIL(表/模型不存在)
- [ ] **Step 4: 实现 ORM** —— 沿用 ch02 模型写法:中文注释对齐 DDL COMMENT、`server_onupdate` 处理 updated_at、Boolean 对应 TINYINT(1)、Enum 直接收值本身
- [ ] **Step 5: 跑测试确认通过** —— 全套 PASS
- [ ] **Step 6: live 灌库(存量卷,不 down -v)** —— `docker exec -i baihelp-mysql mysql -ubaihelp -pbaihelp baihelp < scripts/sql/ch03-ddl.sql`,随后 `docker exec baihelp-mysql mysql -ubaihelp -pbaihelp baihelp -e "SHOW CREATE TABLE knowledge_chunks\G"` 确认中文 COMMENT 无双 encode、枚举原样(容器未运行则先 `docker compose up -d` 等 healthy)
- [ ] **Step 7: Commit** —— `feat(db): ch03 两表 DDL 入仓 + ORM 对齐(knowledge_chunks / qa_extraction_staging)`

---

### Task 3: normalize + chunker(纯逻辑,TDD 主战场)

**Files:**
- Create: `app/knowledge/__init__.py`(空 docstring)、`app/knowledge/normalize.py`、`app/knowledge/chunker.py`
- Test: `tests/test_knowledge_normalize.py`、`tests/test_knowledge_chunker.py`

**Interfaces:**
- Produces:
  - `normalize_question(text: str) -> str`(全半角归一、去空白与标点、小写;结果用于去重比对)
  - `@dataclass KnowledgeDoc`:title/category/content_type/key_clauses/body
  - `@dataclass KnowledgeChunk`:category/questions/answer/section_path/content_type/is_key_clause
  - `chunk_document(doc: KnowledgeDoc, *, max_chars: int = 500, overlap_chars: int = 80) -> list[KnowledgeChunk]`
  - `build_vector_text(category: str, questions: str, answer: str) -> str`(固定格式 `分类:…\n问:…\n答:…`)

- [ ] **Step 1: 写失败测试(normalize)** —— `「优惠券怎么用!」` 与 `「优惠券怎么用？」` 与 `ｃａｓｅ ＡＢＣ`(全角)两两 normalize 相等;纯标点串归一为空串
- [ ] **Step 2: 跑失败 → Step 3: 实现 normalize(正则保留字母数字与 CJK)→ 跑过**
- [ ] **Step 4: 写失败测试(chunker,逐条钉 spec §5.2 规则)** —— 用内联 Markdown 字符串构造:

```python
def _doc(body: str, content_type: str = "policy", **kw) -> KnowledgeDoc:
    return KnowledgeDoc(title="测试文档", category="测试分类", content_type=content_type,
                        key_clauses=kw.get("key_clauses", []), body=body)
```

必测断言(测试名即规则名):
`test_hierarchical_sections_carry_section_path`(`# 售后`下`## 退货`的块 section_path == `"测试分类 > 售后 > 退货"`,> 连接、根分类在首位)、`test_long_section_recursed_by_paragraph_then_sentence`(>500 字符段落被拆成多块且每块 ≤ max_chars)、`test_overlap_starts_at_sentence_boundary`(Review Focus 3:相邻块重叠段起点为句号之后,不以半个句子开头;构造只有句号无 !? 的正文)、`test_short_section_single_chunk_no_overlap`(不超长不产生重叠)、`test_long_table_blocks_repeat_header`(Review Focus 2:8 行表格 + max_chars 压到 3 数据行 → 每块首两行 == 表头行+`|---|` 分隔行)、`test_short_table_atomic`(短表整体一块)、`test_faq_question_variants_extracted`(content_type='faq' 且节内首行 `- 问法: A / B / C` → questions == `"节标题\nA\nB\nC"` 且该行从 answer 剔除;content_type='policy' 时同样内容则整行留在 answer、questions==节标题)、`test_key_clause_marked`(key_clauses 命中节路径 → is_key_clause True,未命中 False)、`test_code_fence_not_split`(围栏内句号不作为切分边界)、`test_vector_text_format`(build_vector_text 精确串)
- [ ] **Step 5: 跑失败 → Step 6: 实现 chunker → 跑过** —— 实现要点:先按标题行(#{1,3})切 section 递归建树并记路径;块切分函数 `_split_long(text, max_chars, overlap_chars) -> list[str]`(段落→句子两级,重叠回退最近句边界,句边界集 = 。!?!?! 与换行,找不到边界硬切);表格识别 = 连续 `|` 开头行块,行切时块头复制 rows[0:2]
- [ ] **Step 7: 全套测试 PASS**
- [ ] **Step 8: Commit** —— `feat(knowledge): 结构感知切分器与问法规范化(重叠裁句/表头复制/元数据)`

---

### Task 4: corpus 加载器 + 三份语料文件

**Files:**
- Create: `app/knowledge/corpus.py`、`data/knowledge/退货政策.md`、`data/knowledge/商品FAQ.md`、`data/knowledge/售后手册.md`
- Test: `tests/test_knowledge_corpus.py`

**Interfaces:**
- Produces: `parse_front_matter(text: str) -> tuple[dict, str]`(识别首部 `---` 块,支持 `key: value` 与 `key:` 缩进列表两式,手写解析不引 pyyaml);`load_corpus(directory: Path) -> list[KnowledgeDoc]`(按文件名排序,front-matter 必需键缺失 raise ValueError)
- Consumes: KnowledgeDoc(Task 3)

- [ ] **Step 1: 写失败测试** —— 内联样例:front-matter 三键解析、缩进列表项解析、缺 category 时 ValueError、正文不含 front-matter 围栏
- [ ] **Step 2: 跑失败 → Step 3: 实现 corpus → 跑过**
- [ ] **Step 4: 撰写三份语料(内容要求钉死,措辞自定)**

  - `退货政策.md`(content_type=policy, category=售后服务):必须含章节 **「退货时效」**(answer 含判据词 `七天`)、**「运费说明」**(answer 含 `满 99 元包邮`、`运费 8 元`、偏远 `12 元` —— 验收①判据「包邮」「99」)、责任划分(质量/非质量)、退货流程;key_clauses 含 `售后服务 > 退货政策 > 退货时效`
  - `商品FAQ.md`(content_type=faq, category=商品咨询):≥6 个 `###` 节,每节首行 `- 问法: 变体1 / 变体2`(真实问法变体),覆盖发货时效/支付方式/发票/优惠券/会员积分/换货(运费/邮费词条只允许出现在运费说明,商品FAQ 节标题与问法不得含「邮费」「运费」——保留 ch02 种子语义对照)
  - `售后手册.md`(content_type=manual, category=售后服务):≥3 级标题深度,含一张 ≥6 行的宽表格(物流时效表,供表格切分实测)与一个 >500 字符的长节(供递归+重叠实测)
  
  全部语料一次性过 `chunk_document(settings 默认参数)`,断言无空 answer 块(写入 corpus 测试:`test_corpus_files_chunk_cleanly` 读 `data/knowledge/` 真文件跑 chunker,块数 >10 且每块 answer 非空、questions 非空)
- [ ] **Step 5: 全套测试 PASS → Step 6: Commit** —— `feat(knowledge): 语料加载器与三份知识文档(运费判据/问法变体/表格长节)`

---

### Task 5: 嵌入客户端 + Milvus 仓储

**Files:**
- Create: `app/knowledge/embedding.py`、`app/knowledge/milvus_repo.py`
- Test: `tests/test_knowledge_embedding.py`、`tests/test_knowledge_milvus_repo.py`

**Interfaces:**
- Produces:
  - `class BgeM3Embedder`:`__init__(self, base_url: str, api_key: str, model: str, dim: int)`;`async def embed(self, texts: list[str]) -> list[list[float]]`(顺序对齐;内部 openai `AsyncOpenAI(base_url=..., api_key=...).embeddings.create(model=..., input=texts)`,结果按 index 排序还原顺序;空列表直接返回空列表不发请求)
  - `COLLECTION_NAME = "knowledge"`;`class MilvusKnowledgeRepo`:`__init__(self, db_path: str | Path, dim: int)`(内部 `MilvusClient(str(db_path))`)、`ensure_collection(self) -> None`(schema 路径 auto_id=False + INT64 id 主键 + FLOAT_VECTOR dim + IndexParams COSINE,has_collection 幂等)、`upsert_vectors(self, rows: list[dict]) -> None`、`search(self, query_vector: list[float], top_k: int) -> list[tuple[int, float]]`((id, score),score = 1 − distance)、`close(self) -> None`

- [ ] **Step 1: 写失败测试(embedding,密封)** —— monkeypatch `app.knowledge.embedding.AsyncOpenAI` 为 fake(记录调用参数,返回乱序 index 的 embedding 数据)断言:请求含 model 与完整 texts、返回按 index 还原顺序、每条长度 == dim、空输入零请求
- [ ] **Step 2: 跑失败 → Step 3: 实现 embedding → 跑过**
- [ ] **Step 4: 写失败测试(milvus_repo,进程内真库 tmp_path)** —— `test_ensure_collection_idempotent`(连调两次不炸)、`test_upsert_same_id_no_duplicate`(同 id upsert 两次,search 或 query 只有 1 行)、`test_search_scores_rank_similar_first`(Review Focus 4:三维小向量,query 与 A 全同、与 B 正交;A.score > B.score 且 A.score ≈ 1.0)、`test_close_then_reopen_persists`(close 后新开 client 数据仍在 —— 本地文件持久性)
- [ ] **Step 5: 跑失败 → Step 6: 实现 milvus_repo(Task 1 冒烟已核签名,未知接口先 Context7)→ 跑过**
- [ ] **Step 7: 全套测试 PASS → Step 8: Commit** —— `feat(knowledge): BGE-M3 嵌入客户端与 Milvus Lite 知识集合仓储`

---

### Task 6: ingest 建库编排(双写状态机 + 幂等)

**Files:**
- Create: `app/knowledge/ingest.py`
- Test: `tests/test_knowledge_ingest.py`

**Interfaces:**
- Consumes: load_corpus/chunk_document/build_vector_text(Task 3/4)、BgeM3Embedder(Task 5)、MilvusKnowledgeRepo(Task 5)、ORM KnowledgeChunk(Task 2)
- Produces:
  - `@dataclass IngestStats`:chunks_in_db/inserted/skipped_existing/vectorized
  - `async def run_ingest(session_factory, embedder, repo, *, corpus_dir: Path, max_chars: int = 500, overlap_chars: int = 80, vector_batch_size: int = 64) -> IngestStats`
  - fakes 约定(测试与本任务共用):`FakeEmbedder.embed(texts) -> [[float]]`(确定性:按文本 hash 生成 dim 维)、`FakeRepo.upsert_vectors(rows)/search(...)`(记录调用)
  - CLI:`python -m app.knowledge.ingest`(get_settings → build_engine → 真组件 → run_ingest → 打印统计;`finally dispose+close`)

- [ ] **Step 1: 写失败测试**(全部用 sqlite 内存库 create_all + fakes):
  `test_phase_a_inserts_pending_with_pointers`(两 section 文档 → 两行 pending、section1.next_chunk_id==section2.id、section2.prev==section1.id)、`test_phase_a_natural_key_idempotent`(同语料跑两遍:第二遍 inserted==0、skipped==首遍行数;改一处 answer 再跑 → 只新插该块)、`test_phase_b_backfills_vector_id_and_done`(vector_id 非 NULL、status=='done'、FakeRepo 收到的 id 与向量文本 `build_vector_text` 格式精确断言)、`test_phase_b_resumes_pending_after_failure`(Review Focus 1:FakeRepo 第一次 upsert 抛异常 → run_ingest 抛出;重跑(新 FakeRepo)→ 上次 done 的不重 embed(FakeEmbedder 记录调用数)、残留 pending 被补齐、最终全 done)、`test_mined_pending_picked_up`(手工插一条 content_type='faq' 的 pending 行,corpus_dir 指向空目录跑 run_ingest → Phase A 零插入、Phase B 仍捞起该行向量化置 done —— Phase B 捞全表 pending,语料只进 Phase A)
- [ ] **Step 2: 跑失败 → Step 3: 实现 → Step 4: 跑过**
- [ ] **Step 5: CLI 冒烟(离线部分)** —— `uv run python -m app.knowledge.ingest --help` 退出 0(真跑放 Task 12)
- [ ] **Step 6: 全套测试 PASS → Step 7: Commit** —— `feat(knowledge): 建库编排双写状态机(自然键幂等 + pending 补齐 + vector_id 回填)`

---

### Task 7: seed 历史会话扩展(挖矿语料)

**Files:**
- Modify: `app/db/seed.py`
- Test: `tests/test_seed.py`(追加)

**Interfaces:**
- Produces: 历史会话 8 通,user_id 固定前缀 `seed-hist-01`…`seed-hist-08`(幂等判据:存在即跳过,与既有三判据并列;返回 dict 增键 `history_conversations`)
- 内容要求(措辞自定,话题钉死):每通 3~5 轮;话题覆盖 **退款到账时效**、**修改收货地址**、**优惠券过期**、**发票抬头修改**、**换货运费承担**、**预售发货时间** 等 6 个可挖知识主题;混入 2 通噪音(纯寒暄、无答案转人工);历史消息只含 role user/assistant 纯文本(不造 tool 行)

- [ ] **Step 1: 写失败测试** —— `test_seed_history_dialogues_idempotent`(灌两次,history_conversations 计数不变、messages 总数第二遍不变)、`test_seed_history_topics_present`(至少 6 个可挖主题关键词出现在 user 消息中)
- [ ] **Step 2: 跑失败 → Step 3: 实现(常量 `_HISTORY_DIALOGUES` + 既有 seed 函数追第四判据)→ Step 4: 全套 PASS**
- [ ] **Step 5: Commit** —— `feat(db): seed 补灌八通历史客服会话(挖矿语料 + 噪音对照)`

---

### Task 8: 挖矿 Prompt + mine_qa 管线

**Files:**
- Create: `app/prompts/qa_extraction_prompt.md`、`app/knowledge/mine_qa.py`
- Test: `tests/test_knowledge_mine_qa.py`

**Interfaces:**
- Consumes: normalize_question(Task 3)、KnowledgeChunk ORM(Task 2)、QaExtractionStaging(Task 2)、`get_chat_model`(现有 app/llm/factory)、`load_prompt` 惯例(app/prompts/loader,实现前读该文件沿用其 API)
- Produces:
  - Prompt 模板契约(模板内容实现者写,输出契约钉死):输入=若干会话文本;输出=**仅一个 JSON 数组**,元素 `{"source": "<会话标识>", "question": "...", "answer": "..."}`;规则:只收「用户问出了真实问题且客服给出了实质答案」的轮次;寒暄/无答案/转人工/工单受理类一律不产出
  - `extract_json_array(text: str) -> list[dict]`(容 ```json 围栏与前后废话,解析失败 raise ValueError)
  - `async def mine_qa(session_factory, llm, *, batch_size: int = 4) -> dict[str, int]`(返回 {conversations, batches, extracted, kept, discarded};batch_no = `QA-YYYYMMDD-NN`)
  - CLI:`python -m app.knowledge.mine_qa [--batch-size N] [--self-test]`

- [ ] **Step 1: 写失败测试**(sqlite 内存 + FakeLLM):
  `test_extract_json_array_tolerant`(裸数组/围栏包裹/前后废话三种输入均解析;坏 JSON raise)、`test_mine_qa_writes_staging_and_knowledge`(FakeLLM 返回 2 对 → staging 2 行 extracted → 去重后 knowledge_chunks 2 行 pending、staging 置 kept)、`test_mine_qa_dedup_within_batch`(两对话抽出同问题(仅标点差异)→ 1 kept 1 discarded)、`test_mine_qa_dedup_vs_existing_knowledge`(knowledge_chunks 预置同问法行 → 全部 discarded、不重复入 knowledge)、`test_mine_qa_batch_failure_isolated`(某批 FakeLLM 抛异常 → 该批 staging 无行、后续批正常、函数不抛)、`test_self_test_offline`(`uv run python -m app.knowledge.mine_qa --self-test` 退出码 0:内部 sqlite 内存 + 两条内联假会话 + 假 LLM,零外部依赖)
- [ ] **Step 2: 跑失败 → Step 3: 实现 → Step 4: 跑过**
- [ ] **Step 5: Commit** —— `feat(knowledge): 历史对话挖知识管线(分批抽取/暂存/整体去重入库)+ Prompt 模板`

---

### Task 9: retriever + query_faq 内部替换 + 接线

**Files:**
- Create: `app/knowledge/retriever.py`
- Modify: `app/tools/knowledge.py`、`app/tools/registry.py`(ToolContext 增字段)、`app/services/chat.py`(透传)、`app/api/chat.py`(DI 缝)、`app/main.py`(lifespan 组装)
- Test: `tests/test_knowledge_retriever.py`、`tests/test_tools_db.py`(query_faq 测试改造)、`tests/test_chat_endpoint.py`(client fixture 补 retriever)

**Interfaces:**
- Consumes: BgeM3Embedder / MilvusKnowledgeRepo(Task 5)、ORM KnowledgeChunk(Task 2)
- Produces:
  - `@dataclass RetrievedKnowledge`:chunk_id/category/questions/answer/score
  - `class KnowledgeRetriever`:`__init__(self, embedder, repo, *, top_k: int = 3, score_threshold: float = 0.5)`;`async def retrieve(self, query: str, session_factory) -> list[RetrievedKnowledge]`(embed → repo.search(top_k) → score ≥ score_threshold → MySQL `WHERE id IN(...) AND vectorize_status='done'` 回表 → score 降序;阈值下全滤空 → 返回空列表)
  - `ToolContext` 增字段 `retriever: Any | None = None`(默认 None 保既有构造)
  - `stream_chat_reply(..., retriever: Any | None = None)` → `ToolContext(conversation_id=…, session_factory=…, retriever=retriever)`
  - `app.api.chat` 模块级 `def get_retriever(request: Request) -> Any: return request.app.state.retriever`(与 get_session_factory 同型 DI 缝)→ 传入 stream_chat_reply
  - lifespan:`app.state.retriever = KnowledgeRetriever(embedder, repo, top_k=settings.retrieval_top_k, score_threshold=settings.retrieval_score_threshold)`,其中 `embedder = BgeM3Embedder(settings.embedding_base_url, settings.embedding_api_key, settings.embedding_model, settings.embedding_dim)`、`repo = MilvusKnowledgeRepo(settings.milvus_db_path, settings.embedding_dim)`;repo.ensure_collection() 后挂载,finally 里 repo.close()
  - query_faq 新签名:`async def query_faq(keyword: str, *, session_factory: Annotated[async_sessionmaker[AsyncSession], InjectedToolArg], retriever: Annotated[KnowledgeRetriever, InjectedToolArg]) -> str`

- [ ] **Step 1: 写失败测试(retriever,sqlite + fake embedder/repo)** —— `test_retrieve_backfills_from_mysql_and_orders_by_score`(repo 返回 [2,1] 两 id、MySQL 两行 done → 返回按 score 降序、字段齐全)、`test_retrieve_filters_below_threshold`(score 0.4 < 0.5 被滤)、`test_retrieve_skips_pending_rows`(一行 done 一行 pending → 只返回 done)、`test_retrieve_empty_when_no_hits`
- [ ] **Step 2: 跑失败 → Step 3: 实现 retriever → 跑过**
- [ ] **Step 4: 写失败/改造测试(query_faq 契约锁定)** —— `test_query_faq_args_schema_unchanged`(`set(QueryFaqInput.model_fields) == {"keyword"}`)、`test_query_faq_output_format_locked`(Review Focus 5:FakeRetriever 返回 questions 含三行变体 + category/answer → 输出恰为 `问:{三行以「 / 」相连}\n答:{answer}` 单条;多条以空行相连)、`test_query_faq_fallback_text_verbatim`(空结果 → `未找到与「邮费」相关的 FAQ,请换个关键词或转人工客服。`)、既有 query_faq 测试全部改用 FakeRetriever 注入(ToolContext(retriever=FakeRetriever(...)));`test_chat_endpoint.py` client fixture 补 `app.state.retriever = FakeRetriever([])` 与 get_retriever 缺省
- [ ] **Step 5: 跑失败 → Step 6: 实现(query_faq 内部换 retriever + description 改写;ToolContext/chat.py/api.py/main.py 按 Interfaces 接线)→ Step 7: 全套 PASS(91 基线 + 新增,零断言破坏)**
- [ ] **Step 8: Commit** —— `feat(knowledge): query_faq 切换向量语义检索(契约不变)+ 检索器与全链路接线`

---

### Task 10: 双评估集 + runners

**Files:**
- Create: `evals/qa_mine_cases.jsonl`、`evals/run_mine_eval.py`、`evals/retrieval_cases.jsonl`、`evals/run_retrieval_eval.py`
- Test: runners 自带 `--self-test`(仿 `evals/run_tool_eval.py` 模式,实现前先读该文件沿用其结构/门禁/报告风格)

**Interfaces:**
- Consumes: mine_qa 的 extract_json_array + Prompt(Task 8)、KnowledgeRetriever(Task 9)、BgeM3Embedder(Task 5)
- 数据集契约:
  - `qa_mine_cases.jsonl` ≥10 行:`{"id", "dialogue": [{"role","content"},…], "expect_qa": [{"q_contains": […], "a_contains": […]}]}`;其中 ≥1 行纯噪音对话 `expect_qa: []`(门禁:抽 0 对);主题覆盖 Task 7 六话题 + 变说法(如「钱什么时候回来」→ 退款时效)
  - `retrieval_cases.jsonl` ≥10 行:`{"id", "query", "expect": "hit", "answer_contains": […], "min_rank": 3}` 或 `{"id", "query", "expect": "no_hit"}`;必含:`邮费是多少`→hit(包邮/99)、`快递费怎么算`→hit(换说法)、`你们老板是谁`→no_hit、`怎么退货`→hit(七天)
- 门禁(mine):全部 expect_qa 被满足 + 噪音行 0 抽取;门禁(retrieval):hit 行在 min_rank 内命中且含关键词 + no_hit 行零命中

- [ ] **Step 1: 实现 run_mine_eval.py(--self-test:注入假 LLM 走通门禁逻辑)→ `--self-test` 退出 0**
- [ ] **Step 2: 撰写 qa_mine_cases.jsonl(逐行自校 JSON)**
- [ ] **Step 3: 实现 run_retrieval_eval.py(--self-test:FakeEmbedder+FakeRepo 罐头数据走通门禁逻辑)→ `--self-test` 退出 0**
- [ ] **Step 4: 撰写 retrieval_cases.jsonl**
- [ ] **Step 5: 全套 pytest PASS(runner 纯脚本不进 pytest;--self-test 即其测试)→ Step 6: Commit** —— `feat(evals): 挖矿与检索双评估集 + runners(--self-test 离线自检)`

---

### Task 11: acceptance.sh ⑥ 反转 + ⑦ 新增 + README

**Files:**
- Modify: `scripts/acceptance.sh`、`README.md`

**Interfaces:**
- Consumes: mine_qa --self-test(Task 8);验收⑥新判据词「包邮」或「99」(Task 4 语料)

- [ ] **Step 1: 改 acceptance.sh** —— ⑥ 反转:标题与注释改为「向量召回演示」,判据 = 工具帧含 query_faq **且** 合并回复含 `包邮` 或 `99`(沿用 merge_delta_text 辅助;漏召回历史语义在注释留痕一句话);⑦ 新增:`uv run python -m app.knowledge.mine_qa --self-test` 退出码 0 判 PASS(不依赖服务,放在健康检查之前执行);①②③④⑤ 逐字不动
- [ ] **Step 2: 改 README** —— 章节表 Ch03 行、功能节(RAG:双写/挖矿/向量检索三点)、快速开始(增 `python -m app.knowledge.mine_qa` / `python -m app.knowledge.ingest` 两行 + EMBEDDING_API_KEY 说明)、配置表新九键、演示问句表更新(「邮费是多少」从漏召回改为命中)、路线图勾选向量检索、结构树增 knowledge/corpus
- [ ] **Step 3: `bash -n scripts/acceptance.sh` 语法过 → Step 4: Commit** —— `feat(acceptance): ⑥反转向量召回判分 + ⑦挖矿自检判据;README ch03 化`

---

### Task 12: live 验收(建库 + 挖矿 + 断点重跑 + 端到端)

**Files:**
- Modify: `evals/report.md`(追加 Ch03 节)

**Interfaces:**
- Consumes: 前面全部任务的真组件

- [ ] **Step 1: 环境就位** —— 确认 `.env` 含 `EMBEDDING_API_KEY`(缺则**停下向用户要 key**,不得用测试值混过);MySQL 容器 healthy + ch03 DDL 已灌(Task 2)
- [ ] **Step 2: live 建库(含验收②中断演示)** ——
  ① `uv run python -m app.knowledge.ingest` → 记录统计;② **中断注入**:再次对空库/或手工把若干行置回 pending(UPDATE knowledge_chunks SET vectorize_status='pending', vector_id=NULL WHERE id IN (…)),然后运行一个 embed 中途 kill(或临时环境变量注入在第 N 批抛错的调试开关)→ 确认部分 done 部分 pending;③ 重跑 → 统计确认 pending 清零;④ `docker exec …mysql -e "SELECT vectorize_status, COUNT(*) FROM knowledge_chunks GROUP BY 1"` 全 done
- [ ] **Step 3: live 挖矿** —— `uv run python -m app.knowledge.mine_qa` → staging extracted/kept/discarded 计数合理(噪音被拒)、knowledge_chunks 增 pending 行 → 再跑一次 ingest 补向量化 → 全 done
- [ ] **Step 4: 起服务跑 acceptance** —— `uv run uvicorn app.main:app` 后 `bash scripts/acceptance.sh`:**全判据 PASS**(⑥ 必含「包邮/99」命中回答;⑤ 照旧「七天」);**Windows 重启纪律:netstat -ano 核对 PID 再 taskkill //F**(ch02 阶段 13 教训)
- [ ] **Step 5: live 双评估** —— `uv run python evals/run_mine_eval.py`、`uv run python evals/run_retrieval_eval.py` 门禁全过;ch02 回归:`uv run python evals/run_tool_eval.py` 19 例 ≥90% 且邮费样例必选 query_faq(风险5);结果追记 `evals/report.md`
- [ ] **Step 6: 停服务清理 → Commit** —— `docs(evals): ch03 live 验收与双评估结果归档`
