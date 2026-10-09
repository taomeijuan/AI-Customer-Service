"""ch06 /api/refunds 端点测试：落库、类目校验、404。"""

import httpx
import pytest
from sqlalchemy import select

from app.api.refunds import RefundRequest
from app.db.models import RefundOrder
from app.main import create_app


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def _new_conversation(app):
    async with _client(app) as c:
        r = await c.post(
            "/api/chat/stream",
            json={"user_id": "refund-user", "message": "你好", "conversation_id": None},
        )
    # 会话已建（上面走了一轮闲聊）；直接读回 conversation_id
    import json as _json

    return _json.loads([l for l in r.text.split("\n") if l.startswith("data:")][0][5:])["conversation_id"]


async def test_refund_request_model_rejects_bad_category():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        RefundRequest(conversation_id=1, order_no="1001", reason_category="不想要了", amount=10.0)
    ok = RefundRequest(conversation_id=1, order_no="1001", reason_category="七天无理由", amount=99.9)
    assert ok.reason_category == "七天无理由"


@pytest.mark.usefixtures("db_session")
async def test_create_refund_persists(session_factory, db_session):
    app = create_app()
    app.state.session_factory = session_factory
    cid = await _new_conversation(app)
    async with _client(app) as c:
        resp = await c.post(
            "/api/refunds",
            json={
                "conversation_id": cid,
                "order_no": "1001",
                "reason_category": "质量问题",
                "amount": 2749.83,
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["refund_no"].startswith("R")
    rows = (await db_session.execute(select(RefundOrder).where(RefundOrder.refund_no == body["refund_no"]))).scalars().all()
    assert len(rows) == 1
    assert rows[0].order_no == "1001" and rows[0].status == "待审核"


@pytest.mark.usefixtures("db_session")
async def test_create_refund_unknown_conversation_404(session_factory, db_session):
    app = create_app()
    app.state.session_factory = session_factory
    async with _client(app) as c:
        resp = await c.post(
            "/api/refunds",
            json={
                "conversation_id": 99999999,
                "order_no": "1001",
                "reason_category": "其他",
                "amount": 1.0,
            },
        )
    assert resp.status_code == 404
