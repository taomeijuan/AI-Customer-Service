from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.chains.chat import ChatService
from app.core.config import get_settings
from app.core.llm import get_chat_model
from app.extraction.schemas import AfterSalesExtraction
from app.extraction.service import ExtractionService, build_structured_model
from app.memory.session import SessionStore

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="ecom-cs", version="0.1.0")
    app.state.settings = get_settings()
    app.state.store = SessionStore()
    model = get_chat_model()
    app.state.chat_service = ChatService(model)
    structured = build_structured_model(model)
    app.state.extract_service = ExtractionService(structured)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    app.include_router(chat_router)
    app.include_router(extract_router)
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


app = create_app()
