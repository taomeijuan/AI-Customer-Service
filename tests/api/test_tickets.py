import httpx
import pytest
from sqlalchemy import select

from app.db.models import Ticket
from app.main import create_app
from app.repositories.conversations import ConversationsRepo


@pytest.fixture
async def client(session_factory, db_session):
    app = create_app()
    app.state.session_factory = session_factory
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, session_factory


@pytest.mark.usefixtures("db_session")
async def test_create_ticket_endpoint(client, db_session):
    """验收3：点「建工单」→ 直写 tickets 表（不调 LLM）。"""
    c, sf = client
    async with sf() as s:
        cid = await ConversationsRepo(s).ensure_conversation(user_id="u1")
    resp = await c.post(
        "/api/tickets",
        json={"conversation_id": cid, "description": "耳机右边没声音", "ticket_type": "售后"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["ticket_no"].startswith("T")
    row = (
        await db_session.execute(select(Ticket).where(Ticket.ticket_no == body["ticket_no"]))
    ).scalars().one()
    assert row.description == "耳机右边没声音" and row.status == "待处理"


@pytest.mark.usefixtures("db_session")
async def test_create_ticket_bad_type_422(client):
    c, sf = client
    async with sf() as s:
        cid = await ConversationsRepo(s).ensure_conversation(user_id="u2")
    resp = await c.post(
        "/api/tickets",
        json={"conversation_id": cid, "description": "x", "ticket_type": "不存在的类型"},
    )
    assert resp.status_code == 422
