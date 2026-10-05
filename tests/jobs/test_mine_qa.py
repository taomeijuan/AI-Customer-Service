import pytest
from langchain.messages import AIMessage, HumanMessage
from sqlalchemy import select

from app.db.models import Conversation, KnowledgeChunk, Message, QaExtractionStaging
from app.jobs.mine_qa import mine_qa


class FakeLLM:
    """按批返回抽取结果。"""

    def __init__(self, batches):
        self.batches = list(batches)
        self.seen_pairs = []

    def with_structured_output(self, schema, **kwargs):
        return self

    async def ainvoke(self, messages):
        # messages 里含本批对话对文本，这里直接按序吐脚本
        self.seen_pairs.append(messages)
        return {"qa": self.batches.pop(0)}


class FakeEmbedder:
    """可控向量：同文本同向量，不同文本正交。"""

    async def embed(self, texts):
        return [[1.0, 0.0] if "邮费" in t else [0.0, 1.0] for t in texts]


class FakeMilvus:
    def upsert(self, rows):
        pass


def _settings_stub():
    return type(
        "S", (), {"mine_batch_size": 10, "retrieval_score_threshold": 0.45, "retrieval_top_k": 3}
    )()


@pytest.fixture
async def dialog_session(session_factory, db_session):
    """一段历史会话：两条 QA 对（邮费 + 发票）。"""
    async with session_factory() as s:
        c = Conversation(user_id="miner")
        s.add(c)
        await s.flush()
        s.add_all(
            [
                Message(conversation_id=c.id, role="user", content="邮费多少钱"),
                Message(conversation_id=c.id, role="assistant", content="普通订单满99元包邮，否则8元邮费"),
                Message(conversation_id=c.id, role="user", content="发票怎么开"),
                Message(conversation_id=c.id, role="assistant", content="订单详情页申请电子发票"),
            ]
        )
        await s.commit()
        return c.id


@pytest.mark.usefixtures("db_session")
async def test_mine_stages_then_dedup_and_persists(session_factory, db_session, dialog_session):
    llm = FakeLLM(
        batches=[
            [
                {"question": "邮费多少钱", "answer": "普通订单满99元包邮，否则8元邮费"},
                {"question": "发票怎么开", "answer": "订单详情页申请电子发票"},
            ]
        ]
    )
    stats = await mine_qa(
        session_factory=session_factory,
        llm=llm,
        embedder=FakeEmbedder(),
        milvus=FakeMilvus(),
        settings=_settings_stub(),
    )
    assert stats["extracted"] == 2 and stats["kept"] == 2
    assert stats["vectorized"] == 2  # 入库后立即补向量化
    rows = (
        await db_session.execute(select(QaExtractionStaging))
    ).scalars().all()
    assert [r.status for r in rows] == ["kept", "kept"]
    assert all(r.batch_no.startswith("mine-") for r in rows)
    # kept 进了知识库并已向量化
    chunks = (
        await db_session.execute(select(KnowledgeChunk))
    ).scalars().all()
    assert len(chunks) == 2 and all(c.vectorize_status == "done" for c in chunks)


@pytest.mark.usefixtures("db_session")
async def test_exact_duplicate_discarded(session_factory, db_session, dialog_session):
    # 库里已有完全相同的知识
    db_session.add(
        KnowledgeChunk(
            category="对话挖掘", questions="邮费多少钱", answer="普通订单满99元包邮，否则8元邮费"
        )
    )
    await db_session.commit()
    llm = FakeLLM(
        batches=[
            [
                {"question": "邮费多少钱", "answer": "普通订单满99元包邮，否则8元邮费"},
                {"question": "发票怎么开", "answer": "订单详情页申请电子发票"},
            ]
        ]
    )
    stats = await mine_qa(
        session_factory=session_factory,
        llm=llm,
        embedder=FakeEmbedder(),
        milvus=FakeMilvus(),
        settings=_settings_stub(),
    )
    assert stats["extracted"] == 2 and stats["discarded"] == 1 and stats["kept"] == 1


@pytest.mark.usefixtures("db_session")
async def test_vector_near_duplicate_discarded(session_factory, db_session, dialog_session):
    # 库里已有一条"邮费"知识（FakeEmbedder 同关键词同向量 → 相似度 1.0 > 0.95）
    db_session.add(
        KnowledgeChunk(category="对话挖掘", questions="邮费政策", answer="满99包邮")
    )
    await db_session.commit()
    llm = FakeLLM(
        batches=[
            [
                {"question": "邮费怎么算", "answer": "满99包邮"},  # 向量近重复
                {"question": "发票怎么开", "answer": "订单详情页申请电子发票"},  # 正交 → 保留
            ]
        ]
    )
    stats = await mine_qa(
        session_factory=session_factory,
        llm=llm,
        embedder=FakeEmbedder(),
        milvus=FakeMilvus(),
        settings=_settings_stub(),
    )
    assert stats["discarded"] == 1 and stats["kept"] == 1
