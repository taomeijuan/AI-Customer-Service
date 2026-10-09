"""退款单直写端点（ch06）：前端退款表单提交调用，不经过 LLM。"""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.repositories.conversations import ConversationsRepo
from app.repositories.refunds import REFUND_REASONS, RefundsRepo

logger = logging.getLogger(__name__)
router = APIRouter()

ReasonCategory = Literal["七天无理由", "质量问题", "少件", "与描述不符", "其他"]


class RefundRequest(BaseModel):
    conversation_id: int
    order_no: str = Field(min_length=1, max_length=32)
    reason_category: ReasonCategory = Field(description="退款原因固定类目")
    amount: float = Field(gt=0, le=99999999.99, description="退款金额（decimal(10,2) 上限内）")


@router.post("/api/refunds")
async def create_refund(req: RefundRequest, request: Request) -> dict:
    if req.reason_category not in REFUND_REASONS:
        raise HTTPException(status_code=422, detail="退款原因类目不合法")
    async with request.app.state.session_factory() as session:
        conv = await ConversationsRepo(session).get(req.conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        try:
            refund_no = await RefundsRepo(session).create(
                conversation_id=req.conversation_id,
                order_no=req.order_no,
                reason_category=req.reason_category,
                amount=req.amount,
            )
        except Exception:
            logger.exception("refund create failed, conversation_id=%s", req.conversation_id)
            raise HTTPException(status_code=500, detail="退款单创建失败，请稍后重试")
    return {"ok": True, "refund_no": refund_no, "message": "退款单已提交，等待审核"}
