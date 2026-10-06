import pytest
from sqlalchemy import select

from app.db.models import Conversation, Faq, Message, Ticket


@pytest.mark.usefixtures("db_session")
async def test_conversation_message_roundtrip(db_session):
    c = Conversation(user_id="u1")
    db_session.add(c)
    await db_session.flush()
    assert c.id and c.status == "进行中"
    db_session.add_all(
        [
            Message(conversation_id=c.id, role="user", content="你好"),
            Message(
                conversation_id=c.id,
                role="assistant",
                content=None,
                tool_calls=[
                    {"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1"}
                ],
            ),
            Message(conversation_id=c.id, role="tool", content="[]", tool_call_id="call_1"),
        ]
    )
    await db_session.commit()
    rows = (await db_session.execute(select(Message).order_by(Message.id))).scalars().all()
    assert [r.role for r in rows] == ["user", "assistant", "tool"]
    assert rows[1].tool_calls[0]["name"] == "query_faq"
    assert rows[1].created_at is not None


async def test_faq_and_ticket_defaults(db_session):
    db_session.add(Faq(question="退货政策是什么", answer="7天无理由", category="售后"))
    c = Conversation(user_id="u2")
    db_session.add(c)
    await db_session.flush()
    t = Ticket(ticket_no="T00000000001", conversation_id=c.id, description="x", ticket_type="售后")
    db_session.add(t)
    await db_session.commit()
    assert t.status == "待处理"


@pytest.mark.usefixtures("db_session")
async def test_knowledge_chunk_roundtrip(db_session):
    from app.db.models import KnowledgeChunk

    k = KnowledgeChunk(
        category="售后政策>退款",
        questions="退款多久到账",
        answer="1-3个工作日原路退回",
        section_path="售后政策/退款/退款时限",
        content_type="policy",
        is_key_clause=1,
    )
    db_session.add(k)
    await db_session.commit()
    assert k.id and k.vectorize_status == "pending" and k.is_key_clause == 1


@pytest.mark.usefixtures("db_session")
async def test_qa_staging_roundtrip(db_session):
    from app.db.models import QaExtractionStaging

    s = QaExtractionStaging(
        batch_no="mine-20261005-ab12cd34",
        source_ref="3",
        question="邮费多少",
        answer="满99包邮",
    )
    db_session.add(s)
    await db_session.commit()
    assert s.status == "extracted"


@pytest.mark.usefixtures("db_session")
async def test_low_confidence_question_roundtrip(db_session):
    from app.db.models import LowConfidenceQuestion

    row = LowConfidenceQuestion(
        conversation_id=None,
        raw_question="量子速递是什么时候发明的",
        source="self_check",
        reason="知识库无相关内容，生成自评不足",
    )
    db_session.add(row)
    await db_session.commit()
    await db_session.refresh(row)  # server_default 列需回读
    assert row.id and row.created_at is not None


@pytest.mark.usefixtures("db_session")
async def test_faith_case_roundtrip(db_session):
    from app.db.models import FaithCase

    row = FaithCase(
        eval_id="A01",
        bucket="A_policy",
        query="退款多久到账",
        strategy="hybrid_rerank",
        answer="保证3分钟到账",
        reason="证据里没有任何到账承诺",
        citations=[{"n": 1, "chunk_id": 5, "section_path": "售后政策/退款/退款时限", "question": "退款时限", "answer": "1-3个工作日"}],
        judge_model="deepseek-chat",
    )
    db_session.add(row)
    await db_session.commit()
    assert row.id and row.status == "未解决" and row.seen_count == 1
