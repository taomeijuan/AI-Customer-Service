import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()


class ExtractRequest(BaseModel):
    text: str


@router.post("/api/extract")
async def extract(req: ExtractRequest, request: Request) -> dict:
    try:
        result = await request.app.state.extract_service.extract(req.text)
    except ValueError as e:
        logger.warning("extract failed: %s", e)
        raise HTTPException(
            status_code=422, detail="无法解析该售后描述，请补充订单号或诉求信息"
        )
    return result.model_dump()
