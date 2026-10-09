"""ch07 会话侧栏只读接口：列表 + 历史回载（越权按 404）。"""

from fastapi import APIRouter, HTTPException, Query, Request

from app.repositories.conversations import get_visible_messages, list_user_conversations

router = APIRouter()


@router.get("/api/conversations")
async def list_conversations(request: Request, user_id: str = Query(min_length=1, max_length=64)) -> dict:
    async with request.app.state.session_factory() as session:
        items = await list_user_conversations(session, user_id)
    return {"conversations": items}


@router.get("/api/conversations/{conversation_id}/messages")
async def conversation_messages(
    conversation_id: int, request: Request, user_id: str = Query(min_length=1, max_length=64)
) -> dict:
    async with request.app.state.session_factory() as session:
        msgs = await get_visible_messages(session, conversation_id, user_id)
    if msgs is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"messages": msgs}
