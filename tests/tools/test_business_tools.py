from types import SimpleNamespace

import pytest

from app.tools.ecommerce import query_logistics, query_order, query_product
from app.tools.faq import build_query_faq_tool
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


@pytest.mark.usefixtures("db_session")
async def test_query_faq_vector_search_contract(db_session):
    """ch03：内核换向量检索，契约不变（{count, items}）。"""
    milvus = FakeMilvus(
        hits=[
            {
                "id": 1,
                "distance": 0.82,
                "entity": {
                    "questions": "邮费与包邮规则",
                    "answer": "满99元包邮，否则8元邮费",
                    "category": "售后政策>运费说明",
                },
            }
        ]
    )
    faq_tool = build_query_faq_tool(FakeEmbedder(), milvus, SETTINGS_STUB)
    out = await faq_tool.ainvoke({"keyword": "邮费多少"})
    assert out["count"] == 1 and "满99" in out["items"][0]["answer"]
    assert milvus.queries[0]["top_k"] == 3
    assert milvus.queries[0]["threshold"] == 0.45
    # 低分全滤掉 → 空结果（对应 ch02 的「查不到」语义）
    empty = build_query_faq_tool(
        FakeEmbedder(), FakeMilvus(hits=[]), SETTINGS_STUB
    )
    out2 = await empty.ainvoke({"keyword": "邮费"})
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
