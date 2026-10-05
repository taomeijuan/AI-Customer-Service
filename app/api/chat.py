import logging
from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    conversation_id: int | None = None
    message: str = Field(min_length=1)
    user_id: str = Field(min_length=1, max_length=64)


async def get_or_create_session(req: ChatRequest, request: Request) -> int:
    """依赖阶段确定会话（响应未起流）：不存在抛 404；不带 id 则新建。"""
    from app.repositories.conversations import ConversationsRepo

    async with request.app.state.session_factory() as session:
        try:
            return await ConversationsRepo(session).ensure_conversation(
                req.user_id, req.conversation_id
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="会话不存在或已过期")


@router.post(
    "/api/chat/stream",
    response_class=EventSourceResponse,
    response_model=None,  # 流式生成器不生成 response model（pydantic 升级后必填）
)
async def chat_stream(
    req: ChatRequest,
    request: Request,
    conversation_id: Annotated[int, Depends(get_or_create_session)],
) -> AsyncIterable[ServerSentEvent]:
    """端点必须本身是异步生成器（response_class=EventSourceResponse 的约定）。"""
    orchestrator = request.app.state.orchestrator
    try:
        async for event, data in orchestrator.run(
            user_id=req.user_id, conversation_id=conversation_id, message=req.message
        ):
            yield ServerSentEvent(event=event, data=data)
    except Exception:
        logger.exception("chat stream failed, conversation_id=%s", conversation_id)
        yield ServerSentEvent(
            event="error", data={"message": "服务暂时不可用，请稍后重试"}
        )
