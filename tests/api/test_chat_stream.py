import json
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import select

from app.agents.orchestrator import Orchestrator
from app.db.models import Conversation
from app.main import create_app
from tests.agents.test_orchestrator import FakeChunk, FakeModel, FakeRegistry


def sse_events(text):
    out = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        ev, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1])
        out.append((ev, data))
    return out


@pytest.fixture
async def client(session_factory, db_session):
    app = create_app()
    app.state.session_factory = session_factory  # 依赖与编排器统一指向测试库
    app.state.orchestrator = Orchestrator(
        model=FakeModel([[FakeChunk(text="您好"), FakeChunk(text="，在的")]]),
        registry_factory=lambda session, cid: FakeRegistry(),
        session_factory=session_factory,
        settings=SimpleNamespace(token_budget=4000),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, app


async def test_meta_delta_done_sequence(client):
    c, _ = client
    resp = await c.post(
        "/api/chat/stream",
        json={"user_id": "u1", "message": "你好", "conversation_id": None},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = sse_events(resp.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "meta" and kinds[-1] == "done"
    assert kinds.count("delta") == 2
    assert isinstance(events[0][1]["conversation_id"], int)  # 整型会话 id


async def test_second_turn_receives_first_turn_history(client):
    c, app = client
    r1 = await c.post(
        "/api/chat/stream",
        json={"user_id": "u1", "message": "第一句", "conversation_id": None},
    )
    cid = sse_events(r1.text)[0][1]["conversation_id"]
    await c.post(
        "/api/chat/stream",
        json={"user_id": "u1", "message": "第二句", "conversation_id": cid},
    )
    seen = app.state.orchestrator._model.calls[1]
    assert seen[1].content == "第一句"  # 上轮 user
    assert seen[2].type == "ai"  # 上轮 assistant
    assert seen[-1].content == "第二句"


async def test_unknown_conversation_id_404(client):
    c, _ = client
    resp = await c.post(
        "/api/chat/stream",
        json={"user_id": "u1", "message": "hi", "conversation_id": 999999},
    )
    assert resp.status_code == 404


async def test_user_id_persisted(client, db_session):
    c, _ = client
    resp = await c.post(
        "/api/chat/stream",
        json={"user_id": "u9", "message": "你好", "conversation_id": None},
    )
    cid = sse_events(resp.text)[0][1]["conversation_id"]
    conv = (
        await db_session.execute(
            select(Conversation).where(Conversation.id == cid)
        )
    ).scalars().one()
    assert conv.user_id == "u9" and conv.status == "进行中"


async def test_tool_frame_sequence(session_factory, db_session):
    app = create_app()
    app.state.session_factory = session_factory
    tc = {"name": "query_faq", "args": '{"keyword": "退货"}', "id": "call_1", "index": 0, "type": "tool_call_chunk"}
    model = FakeModel(
        [
            [FakeChunk(tool_call_chunks=[tc])],
            [FakeChunk(text="根据政策…")],
        ]
    )
    app.state.orchestrator = Orchestrator(
        model=model,
        registry_factory=lambda session, cid: FakeRegistry(),
        session_factory=session_factory,
        settings=SimpleNamespace(token_budget=4000),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/api/chat/stream",
            json={"user_id": "u1", "message": "退货政策是什么", "conversation_id": None},
        )
    events = sse_events(resp.text)
    kinds = [e for e, _ in events]
    assert kinds.index("tool") < kinds.index("delta")  # 工具状态帧在正文前
    tool_frames = [v for e, v in events if e == "tool"]
    assert tool_frames[0]["status"] == "running" and tool_frames[0]["tool"] == "query_faq"
    assert tool_frames[1]["status"] == "done" and tool_frames[1]["ok"] is True
    assert kinds[-1] == "done"


async def test_empty_message_422(client):
    c, _ = client
    resp = await c.post(
        "/api/chat/stream", json={"user_id": "u1", "message": "", "conversation_id": None}
    )
    assert resp.status_code == 422
