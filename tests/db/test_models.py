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
