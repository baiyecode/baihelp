"""FastAPI 应用装配。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.core.config import get_settings
from app.db.engine import build_engine, build_session_factory, ping
from app.prompts.loader import load_system_prompt
from app.services.history import SessionStore
from app.tools import build_default_registry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # 启动即加载配置：缺 LLM_API_KEY 等配置错误 fail-fast（spec §7.1）。
    # 只放 lifespan 而非模块导入期，保证测试/工具 import 本模块无需真实 key。
    get_settings()
    # ch03 检索三件套同样延迟到 lifespan 导入:pymilvus 导入即对仓库根 .env 执行
    # load_dotenv(实测见 test_knowledge_milvus_repo.py 顶部注),放模块导入期会让
    # import app.main 污染 os.environ;嵌入客户端本无副作用,随检索器一并组装。
    from app.knowledge.embedding import BgeM3Embedder
    from app.knowledge.milvus_repo import MilvusKnowledgeRepo
    from app.knowledge.retriever import KnowledgeRetriever

    settings = get_settings()
    # 启动即 SELECT 1 探活：DB 连不上直接 raise 让启动失败（fail-fast），
    # ping 的错误消息自带「docker compose up -d + 建表 DDL」操作提示（spec §7.6）。
    # MySQL 端绝不 create_all——建表唯一依据是用户 DDL。
    engine = build_engine(settings.database_url)
    # Milvus 仓储在 try 外构造:文件打不开等失败同样 fail-fast,finally 统一 close
    repo = MilvusKnowledgeRepo(settings.milvus_db_path, settings.embedding_dim)
    try:
        await ping(engine)
        # 会话工厂与工具注册表挂 app.state 供端点取用；注册表同样放 lifespan
        # 而非 create_app：它读 settings，放导入期会让 import 本模块也要真实配置。
        app.state.session_factory = build_session_factory(engine)
        app.state.tool_registry = build_default_registry(settings)
        # 语义检索器挂 app.state(query_faq 注入用):嵌入客户端 + 建集合(幂等)
        # + 检索器,组装失败(如 Milvus 文件权限)一并 fail-fast 启动失败
        embedder = BgeM3Embedder(
            settings.embedding_base_url,
            settings.embedding_api_key,
            settings.embedding_model,
            settings.embedding_dim,
        )
        repo.ensure_collection()
        app.state.retriever = KnowledgeRetriever(
            embedder,
            repo,
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
        )
        yield
    finally:
        # 无论探活成败还是正常关停都释放资源,杜绝引擎/Milvus 客户端悬挂
        repo.close()
        await engine.dispose()


def create_app() -> FastAPI:
    """装配应用：共享会话存储与 system 模板挂在 app.state，注册路由。"""
    app = FastAPI(title="Baihelp 客服后端", lifespan=lifespan)
    app.state.store = SessionStore()
    app.state.system_template = load_system_prompt()
    app.include_router(chat_router)
    app.include_router(extract_router)

    @app.get("/api/healthz")
    async def healthz() -> dict[str, str]:
        settings = get_settings()
        return {
            "status": "ok",
            "provider": settings.llm_provider,
            "model": settings.llm_model,
        }

    # 聊天页面（Vibe Coding 例外产物）：每请求读文件，改 HTML 无需重启服务。
    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "chat.html")

    return app


app = create_app()
