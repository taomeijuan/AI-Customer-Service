"""ch07 T9 侧栏只读接口：列表元数据、历史回载、越权 404。"""

import httpx
import pytest

from app.main import create_app
from app.repositories.conversations import ConversationsRepo


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
async def app_with_data(session_factory, db_session):
    """两个会话：一个聊过两轮、一个带摘要标记。"""
    from app.repositories.conversations import ConversationsRepo
    from app.repositories.messages import MessagesRepo
    from app.repositories.summaries import SummariesRepo
    from langchain.messages import AIMessage, HumanMessage
    from app.db.models import Conversation

    app = create_app()
    app.state.session_factory = session_factory
    async with session_factory() as session:
        cid1 = await ConversationsRepo(session).ensure_conversation("sidebar-u", None)
        await MessagesRepo(session).append(cid1, [HumanMessage("退货政策是什么"), AIMessage("七天无理由…")])
        await MessagesRepo(session).append(cid1, [HumanMessage("订单1001到哪了"), AIMessage("到杭州了")])
        cid2 = await ConversationsRepo(session).ensure_conversation("sidebar-u", None)
        await MessagesRepo(session).append(cid2, [HumanMessage("你好"), AIMessage("在的～")])
        await SummariesRepo(session).append_segment(cid2, 1, 2, "用户咨询过政策。", projection_budget=500)
        c1 = await session.get(Conversation, cid1)
    return app, cid1, cid2


async def test_list_newest_first_with_preview_and_summary(app_with_data):
    app, cid1, cid2 = app_with_data
    async with _client(app) as c:
        r = await c.get("/api/conversations", params={"user_id": "sidebar-u"})
    items = r.json()["conversations"]
    assert [it["id"] for it in items] == [cid2, cid1]  # 新在前（cid2 更大且刚更新）
    by_id = {it["id"]: it for it in items}
    assert by_id[cid1]["title"] == "退货政策是什么"
    assert by_id[cid1]["has_summary"] is False
    assert by_id[cid2]["has_summary"] is True


async def test_messages_reload_skips_tool_rows(app_with_data):
    app, cid1, _ = app_with_data
    async with _client(app) as c:
        r = await c.get(f"/api/conversations/{cid1}/messages", params={"user_id": "sidebar-u"})
    msgs = r.json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "退货政策是什么"


async def test_other_user_or_missing_404(app_with_data):
    app, cid1, _ = app_with_data
    async with _client(app) as c:
        r1 = await c.get(f"/api/conversations/{cid1}/messages", params={"user_id": "someone-else"})
        r2 = await c.get("/api/conversations/99999999/messages", params={"user_id": "sidebar-u"})
    assert r1.status_code == 404 and r2.status_code == 404


async def test_options_snapshot_roundtrip(session_factory, db_session):
    """ch07 回载即所见：按钮组快照随消息行往返（log 落库 → API 带出）。"""
    from langchain.messages import AIMessage, HumanMessage

    from app.repositories.messages import MessagesRepo

    async with session_factory() as session:
        cid = await ConversationsRepo(session).ensure_conversation("opt-u", None)
        await MessagesRepo(session).append(
            cid,
            [HumanMessage("我要退款"), AIMessage("这一单可以退 [1]")],
            citations=[{"n": 1, "chunk_id": 21, "section_path": "售后政策", "question": "q", "answer": "a"}],
            options={"options": ["申请退款"], "order": {"order_no": "1001", "product": "扫地机器人", "amount": 2749.83}},
        )
    app = create_app()
    app.state.session_factory = session_factory
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get(f"/api/conversations/{cid}/messages", params={"user_id": "opt-u"})
    msgs = r.json()["messages"]
    bot = msgs[-1]
    assert bot["citations"][0]["chunk_id"] == 21
    assert bot["options"]["options"] == ["申请退款"]
    assert bot["options"]["order"]["order_no"] == "1001"
