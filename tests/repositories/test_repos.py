import re

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import select

from app.db.models import Faq
from app.repositories.conversations import ConversationsRepo
from app.repositories.faq import FaqRepo
from app.repositories.messages import MessagesRepo
from app.repositories.tickets import TicketsRepo


@pytest.mark.usefixtures("db_session")
async def test_conversation_and_message_flow(db_session):
    conv_repo, msg_repo = ConversationsRepo(db_session), MessagesRepo(db_session)
    cid = await conv_repo.ensure_conversation(user_id="u1")
    assert await conv_repo.get(cid) is not None

    await msg_repo.append(cid, [HumanMessage("你好"), AIMessage("您好")])
    await msg_repo.append_tool_round(
        cid,
        tool_calls=[{"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1"}],
        tool_results=[{"id": "call_1", "name": "query_faq", "content": "[]"}],
    )
    history = await msg_repo.load_history(cid)
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)
    assert history[2].tool_calls[0]["name"] == "query_faq"  # assistant 申请
    assert isinstance(history[3], ToolMessage)
    assert history[3].tool_call_id == "call_1"
    assert history[3].name == "query_faq"  # 工具名从配对 assistant 的 tool_calls JSON 反查


async def test_load_history_empty(db_session):
    conv_repo, msg_repo = ConversationsRepo(db_session), MessagesRepo(db_session)
    cid = await conv_repo.ensure_conversation(user_id="u2")
    assert await msg_repo.load_history(cid) == []


async def test_ensure_existing_conversation(db_session):
    repo = ConversationsRepo(db_session)
    cid = await repo.ensure_conversation(user_id="u3")
    assert await repo.ensure_conversation(user_id="u3", conversation_id=cid) == cid
    import pytest as _pytest

    with _pytest.raises(KeyError):
        await repo.ensure_conversation(user_id="u3", conversation_id=99999)


@pytest.mark.usefixtures("db_session")
async def test_faq_search_hit_and_miss(db_session):
    db_session.add_all(
        [
            Faq(question="退货政策是什么", answer="7天无理由", category="售后"),
            Faq(question="怎么查物流", answer="订单详情页", category="物流"),
        ]
    )
    await db_session.commit()
    repo = FaqRepo(db_session)
    hits = await repo.search("退货政策")
    assert len(hits) == 1 and "退货" in hits[0]["question"]
    assert await repo.search("邮费") == []  # 验收3：预期漏召回


@pytest.mark.usefixtures("db_session")
async def test_ticket_no_generation(db_session):
    conv_repo = ConversationsRepo(db_session)
    cid = await conv_repo.ensure_conversation(user_id="u4")
    repo = TicketsRepo(db_session)
    n1 = await repo.create(conversation_id=cid, description="屏幕坏了", ticket_type="售后")
    n2 = await repo.create(conversation_id=cid, description="客服态度差", ticket_type="投诉")
    assert re.fullmatch(r"T\d{8}\d{3}", n1) and n1 != n2  # T+日期+当日序号
    assert n2[len(n2) - 3 :] == "002"  # 当日第二单
