from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.chains.chat import ChatService
from app.core.config import get_settings
from app.core.llm import get_chat_model
from app.memory.session import SessionStore


def create_app() -> FastAPI:
    app = FastAPI(title="ecom-cs", version="0.1.0")
    app.state.settings = get_settings()
    app.state.store = SessionStore()
    app.state.chat_service = ChatService(get_chat_model())

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    app.include_router(chat_router)
    return app


app = create_app()
