"""FastAPI 应用装配。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.core.config import get_settings
from app.prompts.loader import load_system_prompt
from app.services.history import SessionStore


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # 启动即加载配置：缺 LLM_API_KEY 等配置错误 fail-fast（spec §7.1）。
    # 只放 lifespan 而非模块导入期，保证测试/工具 import 本模块无需真实 key。
    get_settings()
    yield


def create_app() -> FastAPI:
    """装配应用：共享会话存储与 system 模板挂在 app.state，注册路由。"""
    app = FastAPI(title="Mewhelp 客服后端", lifespan=lifespan)
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
