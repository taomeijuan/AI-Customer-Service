from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.agents.orchestrator import Orchestrator
from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.core.config import get_settings
from app.core.llm import get_chat_model
from app.db.engine import build_engine
from app.extraction.schemas import AfterSalesExtraction
from app.extraction.service import ExtractionService, build_structured_model
from app.tools.base import ToolRegistry
from app.tools.ecommerce import query_logistics, query_order, query_product
from app.tools.faq import build_query_faq_tool
from app.tools.ticket import build_create_ticket_tool

STATIC_DIR = Path(__file__).parent / "static"


def build_registry(session, conversation_id: int, settings) -> ToolRegistry:
    """请求级工具注册表：mock 三件套全局可用，faq/工单工具绑请求会话。"""
    reg = ToolRegistry(
        default_timeout=settings.tool_timeout, default_retries=settings.tool_retries
    )
    reg.register(query_order)
    reg.register(query_product)
    reg.register(query_logistics)
    reg.register(build_query_faq_tool(session))
    reg.register(build_create_ticket_tool(session, conversation_id))
    return reg


def create_app() -> FastAPI:
    app = FastAPI(title="ecom-cs", version="0.2.0")
    settings = get_settings()
    app.state.settings = settings
    _, app.state.session_factory = build_engine(settings)

    model = get_chat_model()
    structured = build_structured_model(model)
    app.state.extract_service = ExtractionService(structured)
    app.state.orchestrator = Orchestrator(
        model=model,
        registry_factory=lambda session, cid: build_registry(session, cid, settings),
        session_factory=app.state.session_factory,
        settings=settings,
    )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    app.include_router(chat_router)
    app.include_router(extract_router)
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


app = create_app()
