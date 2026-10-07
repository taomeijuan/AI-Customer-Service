"""图骨架测试：七类分流、置信度闸、固定话术、日志落库。"""
import pytest
from sqlalchemy import select

from app.db.models import LowConfidenceQuestion, Message
from app.knowledge.retriever import RetrievalResult
from app.knowledge.reranker import Evidence
from app.repositories.conversations import ConversationsRepo
from app.repositories.low_confidence import LowConfidenceRepo
from app.workflow.graph import build_workflow


def _ev(cid: int) -> Evidence:
    return Evidence(chunk_id=cid, section_path=f"售后政策/节{cid}", question=f"问题{cid}", answer=f"知识{cid}", score=0.9)


class FakeRetriever:
    def __init__(self, result):
        self.result = result
        self.called_with = []

    async def retrieve(self, query, strategy="hybrid_rerank", category_prefix=None):
        self.called_with.append(query)
        return self.result


class FakeClassifier:
    def __init__(self, intent="订单", error=False):
        self.intent = intent
        self.error = error

    async def classify(self, query):
        if self.error:
            raise RuntimeError("LLM 挂了")
        return self.intent


class FakeAgent:
    def __init__(self, text="Agent 回答"):
        self.text = text
        self.calls = []

    async def __call__(self, state):
        self.calls.append(dict(state))
        return {"final_text": self.text, "messages": [], "evidence": state.get("evidence", [])}


async def _ensure_conv(session_factory) -> int:
    async with session_factory() as s:
        return await ConversationsRepo(s).ensure_conversation(user_id="wf-user")


def _make(session_factory, retriever=None, classifier=None, agent=None):
    return build_workflow(
        retriever=retriever or FakeRetriever(RetrievalResult(evidences=[_ev(8)], low_confidence=False)),
        agent_node=agent or FakeAgent(),
        intent_classifier=classifier or FakeClassifier(),
        session_factory=session_factory,
        settings=type("S", (), {"retrieval_low_conf_threshold": 0.45, "agent_max_steps": 6})(),
    )


@pytest.mark.usefixtures("db_session")
async def test_knowledge_route_hits_retrieve_then_agent(session_factory, db_session):
    """知识类（商品咨询）：强制先检索，证据够才进 Agent。"""
    retriever = FakeRetriever(RetrievalResult(evidences=[_ev(8)], low_confidence=False))
    agent = FakeAgent()
    graph = _make(session_factory, retriever, FakeClassifier(intent="商品咨询"), agent)
    out = await graph.ainvoke({"query": "空气炸锅怎么用", "messages": []})
    assert retriever.called_with == ["空气炸锅怎么用"]  # 强制检索被走到
    assert len(agent.calls) == 1  # 证据够 → 进 Agent
    assert out["final_text"] == "Agent 回答"
    assert out["evidence"][0]["chunk_id"] == 8  # citations 全集进 state


@pytest.mark.usefixtures("db_session")
async def test_gate_weak_evidence_blocks_agent(session_factory, db_session):
    """证据弱：兜底话术 + 入池，Agent 不被调。"""
    retriever = FakeRetriever(RetrievalResult(evidences=[], low_confidence=True))
    agent = FakeAgent()
    graph = _make(session_factory, retriever, FakeClassifier(intent="退款退货"))
    cid = await _ensure_conv(session_factory)
    out = await graph.ainvoke({"query": "量子速递多久到", "messages": [], "conversation_id": cid})
    assert agent.calls == []  # 被闸拦下
    assert "暂时没有足够的资料" in out["final_text"]
    assert out["refusal"] is True
    rows = await LowConfidenceRepo(db_session).list_by_source("retrieval_low_conf")
    assert any(r.raw_question == "量子速递多久到" for r in rows)


@pytest.mark.usefixtures("db_session")
async def test_business_route_skips_retrieval(session_factory, db_session):
    """业务数据类（物流）：不预检索，直接进 Agent。"""
    retriever = FakeRetriever(RetrievalResult(evidences=[], low_confidence=False))
    agent = FakeAgent(text="物流信息如下")
    graph = _make(session_factory, retriever, FakeClassifier(intent="物流"), agent)
    await graph.ainvoke({"query": "订单1001的物流", "messages": []})
    assert retriever.called_with == []  # 不预检索
    assert len(agent.calls) == 1


@pytest.mark.usefixtures("db_session")
async def test_complaint_produces_options(session_factory, db_session):
    """投诉：安抚话术 + 两个可选项，Agent 不被调。"""
    agent = FakeAgent()
    graph = _make(session_factory, classifier=FakeClassifier(intent="投诉"), agent=agent)
    out = await graph.ainvoke({"query": "我要投诉", "messages": []})
    assert agent.calls == []
    assert out["options"] == ["转人工", "建工单"]
    assert out["final_text"]


@pytest.mark.usefixtures("db_session")
async def test_chitchat_fixed_text(session_factory, db_session):
    """闲聊：固定话术，零模型调用（Agent/检索都不动）。"""
    retriever = FakeRetriever(RetrievalResult(evidences=[], low_confidence=False))
    agent = FakeAgent()
    graph = _make(session_factory, retriever, FakeClassifier(intent="闲聊"), agent)
    out = await graph.ainvoke({"query": "你好呀", "messages": []})
    assert agent.calls == [] and retriever.called_with == []
    assert out["final_text"]  # 固定话术非空


@pytest.mark.usefixtures("db_session")
async def test_intent_classifier_error_falls_back_to_business(session_factory, db_session):
    """意图识别 LLM 挂 → 兜底「订单」→ 业务数据类直进 Agent。"""
    retriever = FakeRetriever(RetrievalResult(evidences=[], low_confidence=False))
    agent = FakeAgent()
    graph = _make(session_factory, retriever, FakeClassifier(error=True), agent)
    await graph.ainvoke({"query": "订单1001", "messages": []})
    assert retriever.called_with == []  # 兜底到业务路，不走检索
    assert len(agent.calls) == 1


@pytest.mark.usefixtures("db_session")
async def test_log_node_persists_messages(session_factory, db_session):
    cid = await _ensure_conv(session_factory)
    graph = _make(
        session_factory,
        FakeRetriever(RetrievalResult(evidences=[], low_confidence=False)),
        FakeClassifier(intent="物流"),
        FakeAgent(text="回答内容"),
    )
    await graph.ainvoke({"query": "订单1001的物流", "messages": [], "conversation_id": cid})
    rows = (
        await db_session.execute(
            select(Message).where(Message.conversation_id == cid).order_by(Message.id)
        )
    ).scalars().all()
    assert [r.role for r in rows] == ["user", "assistant"]
    assert rows[0].content == "订单1001的物流" and rows[1].content == "回答内容"
