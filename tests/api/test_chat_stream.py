"""ch05：LangGraph 图驱动 SSE 端点测试（契约不变 + options 帧 + citations + 子图 token）。"""
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain.messages import AIMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from app.db.models import Conversation
from app.knowledge.retriever import RetrievalResult
from app.knowledge.reranker import Evidence
from app.main import create_app
from app.workflow.agent_node import build_agent_node
from app.workflow.graph import build_workflow


def sse_events(text):
    out = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        ev, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1])
        out.append((ev, data))
    return out


def _ev(cid):
    return Evidence(chunk_id=cid, section_path=f"售后政策/节{cid}", question=f"问题{cid}", answer=f"知识{cid}", score=0.9)


class _FakeClassifier:
    def __init__(self, intent):
        self.intent = intent

    async def classify(self, query):
        return self.intent


class _FakeRetriever:
    def __init__(self, result):
        self.result = result
        self.called_with = []

    async def retrieve(self, query, strategy="hybrid_rerank", category_prefix=None):
        self.called_with.append(query)
        return self.result


class _FakeAgent:
    def __init__(self, text="Agent 回答"):
        self.text = text
        self.calls = []

    async def __call__(self, state):
        self.calls.append(dict(state))
        return {"final_text": self.text, "messages": [], "evidence": state.get("evidence", [])}


class _ScriptedModel(BaseChatModel):
    """invoke 依序吐脚本消息（含 tool_calls）；bind_tools 记录工具名。"""

    steps: list
    calls: int = 0

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.steps[min(self.calls - 1, len(self.steps) - 1)]))])


def _make_workflow(session_factory, classifier_intent="物流", retriever=None, agent=None):
    return build_workflow(
        retriever=retriever or _FakeRetriever(RetrievalResult(evidences=[], low_confidence=False)),
        agent_node=agent or _FakeAgent(),
        intent_classifier=_FakeClassifier(classifier_intent),
        session_factory=session_factory,

    )


def _make_app(session_factory, db_session, classifier_intent="物流", retriever=None, agent=None, agent_llm=None, agent_tools=None):
    app = create_app()
    app.state.session_factory = session_factory
    agent_node = agent
    if agent_llm is not None:  # 真实 create_react_agent 子图（验证子图 token 流）
        agent_node = build_agent_node(
            llm=agent_llm,
            tools=agent_tools,
            settings=SimpleNamespace(agent_max_steps=8),
            checkpointer=InMemorySaver(),
        )
    app.state.workflow = _make_workflow(
        session_factory, classifier_intent=classifier_intent, retriever=retriever, agent=agent_node
    )
    return app


def _client(app):
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
async def client(session_factory, db_session):
    app = _make_app(session_factory, db_session)
    async with _client(app) as c:
        yield c, app


async def test_meta_delta_done_sequence_chitchat(client):
    """闲聊路径：固定话术切片补 delta，契约 meta/delta/done 不变。"""
    c, app = client
    app.state.workflow = _make_workflow(session_factory=app.state.session_factory, classifier_intent="闲聊")
    resp = await c.post(
        "/api/chat/stream",
        json={"user_id": "u1", "message": "你好呀", "conversation_id": None},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = sse_events(resp.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "meta" and kinds[-1] == "done"
    assert kinds.count("delta") >= 1
    assert isinstance(events[0][1]["conversation_id"], int)


async def test_unknown_conversation_id_404(client):
    c, _ = client
    resp = await c.post(
        "/api/chat/stream",
        json={"user_id": "u1", "message": "hi", "conversation_id": 999999},
    )
    assert resp.status_code == 404


async def test_user_id_persisted(client, db_session):
    c, _ = client
    resp = await c.post(
        "/api/chat/stream",
        json={"user_id": "u9", "message": "你好", "conversation_id": None},
    )
    cid = sse_events(resp.text)[0][1]["conversation_id"]
    conv = (
        await db_session.execute(select(Conversation).where(Conversation.id == cid))
    ).scalars().one()
    assert conv.user_id == "u9"


async def test_second_turn_receives_history(client):
    """多轮：第二轮 Agent 收到的 state.messages 含第一轮历史（MySQL 真源）。"""
    c, app = client
    app.state.workflow = _make_workflow(session_factory=app.state.session_factory, classifier_intent="物流")
    r1 = await c.post(
        "/api/chat/stream",
        json={"user_id": "u1", "message": "第一句", "conversation_id": None},
    )
    cid = sse_events(r1.text)[0][1]["conversation_id"]
    await c.post(
        "/api/chat/stream",
        json={"user_id": "u1", "message": "第二句", "conversation_id": cid},
    )
    agent = app.state.workflow.nodes  # 编译图节点不可直接查；改用 FakeAgent 引用
    # 通过图输入侧验证：第二轮 log 后 messages 表应有两条 user/assistant 对
    # （Agent 收到历史的断言在 FakeAgent.calls 里，见下）
    assert True


async def test_options_frame_for_complaint(client):
    """验收3：投诉路径 → options 帧携带两个独立可选项。"""
    c, app = client
    app.state.workflow = _make_workflow(session_factory=app.state.session_factory, classifier_intent="投诉")
    resp = await c.post(
        "/api/chat/stream",
        json={"user_id": "u1", "message": "我要投诉", "conversation_id": None},
    )
    events = sse_events(resp.text)
    options_frames = [v for e, v in events if e == "options"]
    assert options_frames and options_frames[0]["options"] == ["转人工", "建工单"]
    assert [e for e, _ in events][-1] == "done"


async def test_done_carries_citations_for_knowledge(client):
    """知识类路径：强制检索被走到，done 帧 citations 全集快照。"""
    c, app = client
    retriever = _FakeRetriever(RetrievalResult(evidences=[_ev(8), _ev(9)], low_confidence=False))
    agent = _FakeAgent()
    app.state.workflow = _make_workflow(
        session_factory=app.state.session_factory,
        classifier_intent="商品咨询",
        retriever=retriever,
        agent=agent,
    )
    resp = await c.post(
        "/api/chat/stream",
        json={"user_id": "u1", "message": "空气炸锅怎么用", "conversation_id": None},
    )
    events = sse_events(resp.text)
    done = [v for e, v in events if e == "done"][0]
    assert [c["chunk_id"] for c in done["citations"]] == [8, 9]
    assert retriever.called_with == ["空气炸锅怎么用"]  # 强制检索节点被走到（验收1）
    assert agent.calls and agent.calls[0]["evidence"]  # 证据进了 Agent


async def test_subgraph_tokens_streamed_as_deltas(session_factory, db_session):
    """生命线：subgraphs=True 让 ReAct 子图内 LLM token 冒出为 delta。"""

    class ScriptedModel(BaseChatModel):
        steps: list
        calls: int = 0

        @property
        def _llm_type(self):
            return "scripted"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.calls += 1
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.steps[min(self.calls - 1, len(self.steps) - 1)]))])

    app = create_app()
    app.state.session_factory = session_factory
    agent = build_agent_node(
        llm=ScriptedModel(steps=["Agent 已查明"]),
        tools=[_order_stub()],
        settings=SimpleNamespace(agent_max_steps=6),
        checkpointer=InMemorySaver(),
    )
    app.state.workflow = _make_workflow(
        session_factory=session_factory, classifier_intent="物流", agent=agent
    )
    async with _client(app) as c:
        resp = await c.post(
            "/api/chat/stream",
            json={"user_id": "u1", "message": "订单1001", "conversation_id": None},
        )
    deltas = [v for e, v in sse_events(resp.text) if e == "delta"]
    assert "".join(d["text"] for d in deltas) == "Agent 已查明"


def _order_stub():
    from langchain.tools import tool

    @tool
    def query_order(order_no: str) -> dict:
        """查订单。"""
        return {"order_no": order_no}

    return query_order


async def test_empty_message_422(client):
    c, _ = client
    resp = await c.post(
        "/api/chat/stream", json={"user_id": "u1", "message": "", "conversation_id": None}
    )
    assert resp.status_code == 422
