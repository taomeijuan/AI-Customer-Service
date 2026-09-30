import logging
from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from app.memory.trimmer import trim_history

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str


async def get_or_create_session(
    req: ChatRequest, request: Request
) -> tuple[str, list]:
    """依赖阶段完成会话判定：此时响应未起流，抛 HTTPException 才是标准 404。"""
    store = request.app.state.store
    if req.conversation_id is not None:
        history = await store.get(req.conversation_id)
        if history is None:
            raise HTTPException(status_code=404, detail="会话不存在或已过期")
        return req.conversation_id, history
    return await store.create(), []


@router.post("/api/chat/stream", response_class=EventSourceResponse)
async def chat_stream(
    req: ChatRequest,
    request: Request,
    session: Annotated[tuple[str, list], Depends(get_or_create_session)],
) -> AsyncIterable[ServerSentEvent]:
    cid, history = session
    service = request.app.state.chat_service
    settings = request.app.state.settings

    yield ServerSentEvent(event="meta", data={"conversation_id": cid})
    pieces: list[str] = []
    try:
        trimmed = trim_history(history, budget_tokens=settings.token_budget)
        async for piece in service.astream(trimmed, req.message):
            pieces.append(piece)
            yield ServerSentEvent(event="delta", data={"text": piece})
    except Exception:
        logger.exception("chat stream failed, conversation_id=%s", cid)
        yield ServerSentEvent(
            event="error", data={"message": "服务暂时不可用，请稍后重试"}
        )
        return
    await request.app.state.store.append(
        cid, [HumanMessage(req.message), AIMessage("".join(pieces))]
    )
    yield ServerSentEvent(event="done", data={"conversation_id": cid})
