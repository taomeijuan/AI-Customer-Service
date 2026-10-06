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
async def test_query_faq_quality_control(db_session):
    """ch04：混合检索 + 生成质控三路径（正常/检索低置信/自评不足）。"""
    from types import SimpleNamespace

    from app.knowledge.reranker import Evidence
    from app.repositories.low_confidence import LowConfidenceRepo

    settings_stub = SimpleNamespace(retrieval_low_conf_threshold=0.45)
    ev = Evidence(chunk_id=7, section_path="售后政策/运费说明", question="邮费", answer="满99包邮", score=0.9)

    class FakeRetriever:
        def __init__(self, result):
            self.result = result
        async def retrieve(self, q):
            return self.result

    class FakeAnswerer:
        def __init__(self, outcome):
            self.outcome = outcome
        async def answer(self, q, evs):
            return self.outcome

    from app.generation.answerer import AnswerOutcome
    from app.knowledge.retriever import RetrievalResult
    from app.repositories.conversations import ConversationsRepo

    cid = await ConversationsRepo(db_session).ensure_conversation(user_id="qc-user")
    # ① 正常路径：final_answer + citations
    tool = build_query_faq_tool(
        db_session,
        FakeRetriever(RetrievalResult(evidences=[ev], low_confidence=False)),
        FakeAnswerer(AnswerOutcome(useful=True, answer="满99包邮[1]", citations=[{"n": 1, "chunk_id": 7, "section_path": "售后政策/运费说明", "question": "邮费", "answer": "满99包邮"}])),
        settings_stub, cid,
    )
    out = await tool.ainvoke({"keyword": "邮费多少"})
    assert out["refused"] is False and out["final_answer"] == "满99包邮[1]"
    assert out["citations"][0]["chunk_id"] == 7

    # ② 检索低置信：拒答 + 入池（retrieval_low_conf）
    tool2 = build_query_faq_tool(
        db_session,
        FakeRetriever(RetrievalResult(evidences=[], low_confidence=True)),
        FakeAnswerer(None), settings_stub, cid,
    )
    out2 = await tool2.ainvoke({"keyword": "量子速递多久到"})
    assert out2["refused"] is True
    rows = await LowConfidenceRepo(db_session).list_by_source("retrieval_low_conf")
    assert any(r.raw_question == "量子速递多久到" for r in rows)

    # ③ 自评不足：拒答 + 入池（self_check）
    tool3 = build_query_faq_tool(
        db_session,
        FakeRetriever(RetrievalResult(evidences=[ev], low_confidence=False)),
        FakeAnswerer(AnswerOutcome(useful=False, answer="这个问题我这边暂时答不了", reason="知识缺")),
        settings_stub, cid,
    )
    out3 = await tool3.ainvoke({"keyword": "量子速递多久到"})
    assert out3["refused"] is True
    rows3 = await LowConfidenceRepo(db_session).list_by_source("self_check")
    assert any("知识缺" in (r.reason or "") for r in rows3)


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
