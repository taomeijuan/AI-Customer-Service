"""ch07 摘要仓储：追加段/边界推进/投影按预算从最新段重组。"""

import pytest
from sqlalchemy import select

from app.db.models import Conversation, ConversationSummary
from app.repositories.conversations import ConversationsRepo
from app.repositories.summaries import SummariesRepo


@pytest.mark.usefixtures("db_session")
async def test_append_segment_advances_boundary_and_projection(session_factory, db_session):
    async with session_factory() as session:
        cid = await ConversationsRepo(session).ensure_conversation("sum-user", None)
        repo = SummariesRepo(session)
        seq1 = await repo.append_segment(cid, from_msg_id=1, upto_msg_id=10, content="第一段梗概内容", projection_budget=500)
        seq2 = await repo.append_segment(cid, from_msg_id=11, upto_msg_id=20, content="第二段梗概内容", projection_budget=500)
        assert (seq1, seq2) == (1, 2)
        conv = await session.get(Conversation, cid)
        assert conv.summary_upto_msg_id == 20
        assert "〔第1段〕" in conv.summary and "〔第2段〕" in conv.summary
        segs = await repo.list_segments(cid)
        assert [s.seq for s in segs] == [1, 2]  # 只追加不改写


@pytest.mark.usefixtures("db_session")
async def test_projection_trimmed_to_budget_newest_first(session_factory, db_session):
    async with session_factory() as session:
        cid = await ConversationsRepo(session).ensure_conversation("sum-user2", None)
        repo = SummariesRepo(session)
        long = "很长的梗概" * 60  # >800 折算token
        await repo.append_segment(cid, 1, 5, long + "一", projection_budget=500)
        await repo.append_segment(cid, 6, 9, long + "二", projection_budget=500)
        conv = await session.get(Conversation, cid)
        assert "〔第2段〕" in (conv.summary or "")  # 预算只装得下最新一段
        assert "〔第1段〕" not in (conv.summary or "")  # 旧段视图级出局，分段表仍在
        segs = await repo.list_segments(cid)
        assert len(segs) == 2


@pytest.mark.usefixtures("db_session")
async def test_seq_unique_guard(session_factory, db_session):
    async with session_factory() as session:
        cid = await ConversationsRepo(session).ensure_conversation("sum-user3", None)
        await SummariesRepo(session).append_segment(cid, 1, 2, "内容", projection_budget=500)
        segs = await session.execute(
            select(ConversationSummary).where(ConversationSummary.conversation_id == cid)
        )
        assert len(segs.scalars().all()) == 1
