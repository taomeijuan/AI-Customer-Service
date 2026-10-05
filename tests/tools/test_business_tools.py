import pytest

from app.tools.ecommerce import query_logistics, query_order, query_product
from app.tools.faq import build_query_faq_tool
from app.tools.ticket import build_create_ticket_tool


async def test_query_order_shape():
    out = await query_order.ainvoke({"order_no": "1001"})
    assert out["order_no"] == "1001"
    assert {"status", "product", "amount", "created_at"} <= set(out)


async def test_query_product_shape():
    out = await query_product.ainvoke({"product_name": "空气炸锅"})
    assert {"product", "price", "stock", "promo"} <= set(out)


async def test_query_logistics_shape():
    out = await query_logistics.ainvoke({"order_no": "1001"})
    assert {"carrier", "tracking_no", "traces"} <= set(out)
    assert 3 <= len(out["traces"]) <= 5
    assert {"time", "desc"} <= set(out["traces"][0])


@pytest.mark.usefixtures("db_session")
async def test_query_faq_tool_hit_and_miss(db_session):
    from app.db.models import Faq

    db_session.add(Faq(question="退货政策是什么", answer="7天无理由", category="售后"))
    await db_session.commit()
    faq_tool = build_query_faq_tool(db_session)
    out = await faq_tool.ainvoke({"keyword": "退货"})
    assert out["count"] == 1 and "7天无理由" in out["items"][0]["answer"]
    out2 = await faq_tool.ainvoke({"keyword": "邮费"})
    assert out2["count"] == 0 and out2["items"] == []


@pytest.mark.usefixtures("db_session")
async def test_create_ticket_inserts(db_session):
    from app.repositories.conversations import ConversationsRepo

    cid = await ConversationsRepo(db_session).ensure_conversation(user_id="u1")
    make_ticket = build_create_ticket_tool(db_session, conversation_id=cid)
    out = await make_ticket.ainvoke({"description": "屏幕碎了", "ticket_type": "售后"})
    assert out["ok"] is True and out["ticket_no"].startswith("T")


@pytest.mark.usefixtures("db_session")
async def test_create_ticket_bad_type_rejected(db_session):
    from pydantic import ValidationError

    from app.repositories.conversations import ConversationsRepo

    cid = await ConversationsRepo(db_session).ensure_conversation(user_id="u2")
    make_ticket = build_create_ticket_tool(db_session, conversation_id=cid)
    with pytest.raises(ValidationError):
        await make_ticket.ainvoke({"description": "x", "ticket_type": "白嫖"})
