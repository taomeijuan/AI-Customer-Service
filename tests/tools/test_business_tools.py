from types import SimpleNamespace

import pytest

from app.tools.ecommerce import query_logistics, query_order, query_product
from app.tools.ticket import build_create_ticket_tool

SETTINGS_STUB = SimpleNamespace(retrieval_top_k=3, retrieval_score_threshold=0.45)


class FakeEmbedder:
    async def embed_one(self, text):
        return [1.0, 0.0]


class FakeMilvus:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def search(self, vector, top_k, score_threshold=None):
        self.queries.append({"vector": vector, "top_k": top_k, "threshold": score_threshold})
        return self.hits


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
