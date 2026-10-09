"""ch07 T7 摘要任务单测：追加段/推边界/单飞skip/失败不推边界/不阻塞。"""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from langchain.messages import AIMessage, HumanMessage
from sqlalchemy import select

from app.db.models import Conversation, ConversationSummary
from app.memory.summarizer import Summarizer
from app.repositories.conversations import ConversationsRepo


class _FakeModel:
    def __init__(self, reply="用户咨询订单1001退款，未给手机号。", delay=0.0, boom=False):
        self.reply, self.delay, self.boom = reply, delay, boom
        self.seen_prompt = None

    async def ainvoke(self, msgs):
        self.seen_prompt = msgs
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.boom:
            raise RuntimeError("upstream down")
        return AIMessage(content=self.reply)


def _settings():
    return SimpleNamespace(summary_inject_reserve=500)


async def _mkconv(sf) -> int:
    async with sf() as session:
        return await ConversationsRepo(session).ensure_conversation("summ-job", None)


@pytest.mark.usefixtures("db_session")
async def test_schedule_appends_segment_and_advances_boundary(session_factory, db_session):
    cid = await _mkconv(session_factory)
    s = Summarizer(_FakeModel(), session_factory, _settings())
    batch = [(1, HumanMessage("订单1001能退吗")), (2, AIMessage("可以"))]
    task = s.schedule(cid, batch, projection=None)
    assert task is not None
    await task
    async with session_factory() as session:
        conv = await session.get(Conversation, cid)
        assert conv.summary_upto_msg_id == 2
        assert "订单1001" in (conv.summary or "")
        rows = (
            await session.execute(select(ConversationSummary).where(ConversationSummary.conversation_id == cid))
        ).scalars().all()
        assert len(rows) == 1 and rows[0].seq == 1


@pytest.mark.usefixtures("db_session")
async def test_single_flight_skips_second_trigger(session_factory, db_session, caplog):
    cid = await _mkconv(session_factory)
    slow = Summarizer(_FakeModel(delay=0.2), session_factory, _settings())
    batch = [(1, HumanMessage("x")), (2, AIMessage("y"))]
    with caplog.at_level(logging.INFO, logger="app.memory.summarizer"):
        t1 = slow.schedule(cid, batch, None)
        t2 = slow.schedule(cid, batch, None)  # 在途 → skip
        assert t2 is None and t1 is not None
        await t1
    assert "summary skip" in caplog.text


@pytest.mark.usefixtures("db_session")
async def test_failure_does_not_advance_boundary(session_factory, db_session):
    cid = await _mkconv(session_factory)
    s = Summarizer(_FakeModel(boom=True), session_factory, _settings())
    await s.schedule(cid, [(1, HumanMessage("x")), (2, AIMessage("y"))], None)
    async with session_factory() as session:
        conv = await session.get(Conversation, cid)
        assert conv.summary_upto_msg_id is None  # 边界未动，下轮重试
        rows = (
            await session.execute(select(ConversationSummary).where(ConversationSummary.conversation_id == cid))
        ).scalars().all()
        assert list(rows) == []


@pytest.mark.usefixtures("db_session")
async def test_old_projection_is_context_not_recompressed(session_factory, db_session):
    """压完不回头：旧梗概只作背景进 prompt，不被复述合并（prompt 里显式标注）。"""
    cid = await _mkconv(session_factory)
    model = _FakeModel()
    s = Summarizer(model, session_factory, _settings())
    await s.schedule(cid, [(3, HumanMessage("新问题"))], projection="〔第1段〕旧事实。")
    user_prompt = str(model.seen_prompt[1].content)
    assert "不用复述进新梗概" in user_prompt and "旧事实" in user_prompt


@pytest.mark.usefixtures("db_session")
async def test_long_summary_clipped(session_factory, db_session):
    cid = await _mkconv(session_factory)
    s = Summarizer(_FakeModel(reply="长" * 500), session_factory, _settings())
    await s.schedule(cid, [(1, HumanMessage("x"))], None)
    async with session_factory() as session:
        row = (
            await session.execute(select(ConversationSummary).where(ConversationSummary.conversation_id == cid))
        ).scalars().first()
        assert len(row.content) <= 220


@pytest.mark.usefixtures("db_session")
async def test_schedule_returns_immediately_nonblocking(session_factory, db_session):
    """不阻塞用户轮：schedule() 同步返回 Task（模型慢也不等）。"""
    cid = await _mkconv(session_factory)
    s = Summarizer(_FakeModel(delay=5), session_factory, _settings())
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    task = s.schedule(cid, [(1, HumanMessage("x"))], None)
    assert loop.time() - t0 < 0.05
    task.cancel()
