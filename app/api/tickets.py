"""人工工单直写端点（ch05）：前端「建工单」按钮调用，不经过 LLM。"""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.repositories.tickets import TicketsRepo

logger = logging.getLogger(__name__)
router = APIRouter()


class TicketRequest(BaseModel):
    conversation_id: int
    description: str = Field(min_length=1)
    ticket_type: Literal["售后", "投诉", "咨询"] = Field(
        default="售后", description="工单类型"
    )


@router.post("/api/tickets")
async def create_ticket(req: TicketRequest, request: Request) -> dict:
    async with request.app.state.session_factory() as session:
        try:
            ticket_no = await TicketsRepo(session).create(
                conversation_id=req.conversation_id,
                description=req.description,
                ticket_type=req.ticket_type,
            )
        except Exception:
            logger.exception("ticket create failed, conversation_id=%s", req.conversation_id)
            raise HTTPException(status_code=500, detail="工单创建失败，请稍后重试")
    return {"ok": True, "ticket_no": ticket_no, "message": "工单已创建，人工客服会尽快联系您"}
