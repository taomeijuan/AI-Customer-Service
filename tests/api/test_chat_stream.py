"""ch05：LangGraph 图驱动 SSE 端点测试（契约不变 + options 帧 + citations + 子图 token）。"""
import json
from types import SimpleNamespace

import httpx
import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import select

from app.db.models import Conversation
from app.repositories.messages import MessagesRepo
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

    async def classify_detail(self, query):
        from app.workflow.intent import ClassifyOutcome

        return ClassifyOutcome(intent=self.intent, confidence=0.9)

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
        checkpointer=InMemorySaver(),  # ch07 会话级线程：aget_state 需要 checkpointer
    )


def _make_app(session_factory, db_session, classifier_intent="物流", retriever=None, agent=None, agent_llm=None, agent_tools=None):
    app = create_app()
    app.state.session_factory = session_factory
    agent_node = agent
    if agent_llm is not None:  # 真实 create_react_agent 子图（验证子图 token 流）
        agent_node = build_agent_node(
            llm=agent_llm,
            tools=agent_tools,
            settings=SimpleNamespace(max_agent_steps=8),
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


async def test_second_turn_receives_history(session_factory, db_session):
    """多轮：第二轮 Agent 收到第一轮历史（ctx_history 精简史，MySQL 真源）。"""
    agent = _FakeAgent(text="回答")
    app = create_app()
    app.state.session_factory = session_factory
    app.state.workflow = _make_workflow(session_factory, classifier_intent="物流", agent=agent)
    async with _client(app) as c:
        r1 = await c.post(
            "/api/chat/stream",
            json={"user_id": "u1", "message": "第一句", "conversation_id": None},
        )
        cid = sse_events(r1.text)[0][1]["conversation_id"]
        await c.post(
            "/api/chat/stream",
            json={"user_id": "u1", "message": "第二句", "conversation_id": cid},
        )
    second = agent.calls[-1]
    texts = [m.text for m in second.get("ctx_history") or []]
    assert "第一句" in texts and "回答" in texts  # 第一轮问答对进入第二轮上下文


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
    """生命线：subgraphs=True 让 ReAct 子图内 LLM token 冒出为 delta。

    断言粒度=token 级：脚本模型一次吐 6 字 → 必须恰好 1 个 delta 帧原文直出。
    （若退化成尾部 3 字切片兜底，同一文本会被切成 4 帧，此测试即红。）
    """

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
        settings=SimpleNamespace(max_agent_steps=6),
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
    assert len(deltas) == 1  # token 级直出，不是 3 字切片
    assert deltas[0]["text"] == "Agent 已查明"
    done = [v for e, v in sse_events(resp.text) if e == "done"]
    assert done  # done 帧仍在


def _order_stub():
    from langchain.tools import tool

    @tool
    def query_order(order_no: str) -> dict:
        """查订单。"""
        return {"order_no": order_no}

    return query_order


async def test_tool_frames_realtime_before_deltas(session_factory, db_session):
    """工具帧由回调在工具执行当刻推（running→done），先于最终回答的 token 流；
    工具申请/结果消息不得泄漏进 delta。"""

    class ScriptedModel(BaseChatModel):
        calls: int = 0

        @property
        def _llm_type(self):
            return "scripted"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.calls += 1
            if self.calls == 1:
                msg = AIMessage(
                    content="我先查一下",  # 工具申请附带的前导话术：不得泄漏为 delta
                    tool_calls=[{"name": "query_order", "args": {"order_no": "1001"}, "id": "c1", "type": "tool_call"}],
                )
            else:
                msg = AIMessage(content="最终答案")
            return ChatResult(generations=[ChatGeneration(message=msg)])

    app = create_app()
    app.state.session_factory = session_factory
    agent = build_agent_node(
        llm=ScriptedModel(),
        tools=[_order_stub()],
        settings=SimpleNamespace(max_agent_steps=6),
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
    events = sse_events(resp.text)
    seq = [e for e, _ in events]
    deltas = [v["text"] for e, v in events if e == "delta"]
    tool_frames = [v for e, v in events if e == "tool"]
    assert [f["status"] for f in tool_frames] == ["running", "done"]  # 回调双帧
    assert tool_frames[0]["tool"] == "query_order"
    assert "".join(deltas) == "最终答案"  # 工具申请/ToolMessage/前导话术均未污染 delta
    assert "我先查一下" not in "".join(deltas)
    assert seq.index("tool") < seq.index("delta")  # chips 先于回答文字就位


async def test_empty_message_422(client):
    c, _ = client
    resp = await c.post(
        "/api/chat/stream", json={"user_id": "u1", "message": "", "conversation_id": None}
    )
    assert resp.status_code == 422


async def test_order_selector_interrupt_and_resume(session_factory, db_session):
    """ch06 验收4：无单号问退款 → SSE order_selector 帧；点选 resume → 子流程续跑完。"""
    from langgraph.checkpoint.memory import InMemorySaver as _Saver
    from langgraph.types import interrupt as _interrupt
    from app.knowledge.reranker import Evidence as _Ev
    from app.workflow.refund_flow import build_refund_prep as _build_refund

    class _RefundRetriever:
        async def retrieve(self, query, strategy="hybrid_rerank", category_prefix=None):
            return RetrievalResult(evidences=[], low_confidence=False)

        async def retrieve_multi(self, queries, strategy="hybrid_rerank", category_prefix=None):
            return RetrievalResult(
                evidences=[_Ev(21, "售后政策", "七天无理由", "签收7天内可退", 0.8)],
                low_confidence=False,
            )

    class _StubExpander:
        async def expand(self, query, context=None):
            return ["七天无理由退货条件", "拆封商品能否退货"]

    def _refund_prep(state):
        from app.tools.ecommerce import orders_summary

        if "1001" not in state["query"]:
            choice = _interrupt({"type": "order_selector", "orders": orders_summary(), "question": state["query"]})
            if not (choice or {}).get("order_no"):
                return {"final_text": "已取消", "refusal": False, "evidence": []}
        return {
            "order_no": "1001",
            "refusal": False,
            "evidence": [{"n": 1, "chunk_id": 21, "section_path": "售后政策", "question": "七天无理由", "answer": "签收7天内可退"}],
            "messages": [],
        }

    class _RouterClassifier(_FakeClassifier):
        """退款退货为主，但闲聊问句判闲聊——否则 u2 的「你好」也会进退款子流程被 interrupt。"""

        async def classify_detail(self, query):
            from app.workflow.intent import ClassifyOutcome

            intent = "闲聊" if "你好" in query else self.intent
            return ClassifyOutcome(intent=intent, confidence=0.9)

    app = create_app()
    app.state.session_factory = session_factory
    app.state.workflow = build_workflow(
        retriever=_RefundRetriever(),
        agent_node=_FakeAgent(text="订单 1001 可以退 [1]"),
        intent_classifier=_RouterClassifier("退款退货"),
        session_factory=session_factory,
        resolver=None,
        refund_prep=_refund_prep,  # 桩子流程：真 interrupt 语义 + 确定证据
        checkpointer=_Saver(),
    )
    async with _client(app) as c:
        # 第一段：无单号 → order_selector 帧 + done（本轮无正文）
        r1 = await c.post(
            "/api/chat/stream",
            json={"user_id": "u1", "message": "我要退款", "conversation_id": None},
        )
        ev1 = sse_events(r1.text)
        kinds1 = [e for e, _ in ev1]
        assert "order_selector" in kinds1
        selector = [v for e, v in ev1 if e == "order_selector"][0]
        assert [o["order_no"] for o in selector["orders"]][:2] == ["1001", "1002"]
        assert kinds1[-1] == "done"
        cid = ev1[0][1]["conversation_id"]

        # 校验：message 与 resume 二选一
        bad = await c.post(
            "/api/chat/stream",
            json={"user_id": "u1", "message": " hi", "resume": {"order_no": "1001"}, "conversation_id": cid},
        )
        assert bad.status_code == 422

        # 无 pending 的会话 resume → 409
        fresh = await c.post(
            "/api/chat/stream",
            json={"user_id": "u2", "message": "你好", "conversation_id": None},
        )
        fresh_cid = sse_events(fresh.text)[0][1]["conversation_id"]
        conflict = await c.post(
            "/api/chat/stream",
            json={"user_id": "u2", "resume": {"order_no": "1001"}, "conversation_id": fresh_cid},
        )
        assert conflict.status_code == 409

        # 第二段：点选订单 1001 → 恢复续跑 → Agent 回答 + citations
        r2 = await c.post(
            "/api/chat/stream",
            json={"user_id": "u1", "resume": {"order_no": "1001"}, "conversation_id": cid},
        )
        ev2 = sse_events(r2.text)
        kinds2 = [e for e, _ in ev2]
        done2 = [v for e, v in ev2 if e == "done"][0]
        assert "order_selector" not in kinds2  # 恢复跑不再弹选择器
        assert done2["citations"][0]["chunk_id"] == 21
        deltas2 = [v["text"] for e, v in ev2 if e == "delta"]
        assert "".join(deltas2) == "订单 1001 可以退 [1]"


async def test_resume_options_frame_carries_order(session_factory, db_session):
    """评审 M1 回归：refund_prep 的 options 更新必须发 options 帧（含订单摘要）。"""
    from langgraph.checkpoint.memory import InMemorySaver as _Saver

    async def _refund_with_options(state):
        return {
            "order_no": "1001",
            "refusal": False,
            "evidence": [],
            "options": ["申请退款"],
            "order_brief": {"order_no": "1001", "product": "扫地机器人", "amount": 2749.83},
            "messages": [],
        }

    app = create_app()
    app.state.session_factory = session_factory
    app.state.workflow = build_workflow(
        retriever=_FakeRetriever(RetrievalResult(evidences=[], low_confidence=False)),
        agent_node=_FakeAgent(text="可以退"),
        intent_classifier=_FakeClassifier("退款退货"),
        session_factory=session_factory,
        refund_prep=_refund_with_options,
        checkpointer=_Saver(),
    )
    async with _client(app) as c:
        resp = await c.post(
            "/api/chat/stream",
            json={"user_id": "u1", "message": "订单1001能退吗", "conversation_id": None},
        )
    frames = [v for e, v in sse_events(resp.text) if e == "options"]
    assert frames and frames[0]["options"] == ["申请退款"]
    assert frames[0]["order"]["order_no"] == "1001"
    assert frames[0]["order"]["amount"] == 2749.83


async def test_abandoned_selector_not_hijacked(session_factory, db_session):
    """评审 M2 回归：中断轮后发新消息，不得复用挂起线程（不弹旧选择器、正常走闲聊）。"""
    from langgraph.checkpoint.memory import InMemorySaver as _Saver
    from langgraph.types import interrupt as _interrupt

    def _selector_prep(state):
        return _interrupt({"type": "order_selector", "orders": [{"order_no": "1001"}], "question": state["query"]})

    class _IntentByQuery:
        def __init__(self):
            from app.workflow.intent import ClassifyOutcome
            self._ClassifyOutcome = ClassifyOutcome

        async def classify_detail(self, query):
            intent = "退款退货" if "退" in query else "闲聊"
            return self._ClassifyOutcome(intent=intent, confidence=0.9)

        async def classify(self, query):
            return (await self.classify_detail(query)).intent

    app = create_app()
    app.state.session_factory = session_factory
    app.state.workflow = build_workflow(
        retriever=_FakeRetriever(RetrievalResult(evidences=[], low_confidence=False)),
        agent_node=_FakeAgent(text="x"),
        intent_classifier=_IntentByQuery(),
        session_factory=session_factory,
        refund_prep=_selector_prep,
        checkpointer=_Saver(),
    )
    async with _client(app) as c:
        r1 = await c.post(
            "/api/chat/stream",
            json={"user_id": "u1", "message": "我要退款", "conversation_id": None},
        )
        kinds1 = [e for e, _ in sse_events(r1.text)]
        assert "order_selector" in kinds1
        cid = sse_events(r1.text)[0][1]["conversation_id"]

        # 放弃点选，直接问「你好」：必须正常闲聊，不得再弹选择器
        r2 = await c.post(
            "/api/chat/stream",
            json={"user_id": "u1", "message": "你好", "conversation_id": cid},
        )
        ev2 = sse_events(r2.text)
        kinds2 = [e for e, _ in ev2]
        assert "order_selector" not in kinds2, "新消息被挂起的旧选择器劫持"
        deltas = "".join(v["text"] for e, v in ev2 if e == "delta")
        assert deltas  # 闲聊话术正常流出


async def test_abandoned_selector_cancelled_not_hijacked(session_factory, db_session):
    """ch07 T8：点选挂起后发新消息 → 旧中断静默取消，新消息按自己的意图走。"""
    from langgraph.types import interrupt as _interrupt

    def _selector_prep(state):
        from app.tools.ecommerce import orders_summary

        _interrupt({"type": "order_selector", "orders": orders_summary(), "question": state["query"]})
        return {"final_text": "已取消", "refusal": False, "evidence": [], "order_no": ""}

    class _RouterClassifier(_FakeClassifier):
        async def classify_detail(self, query):
            from app.workflow.intent import ClassifyOutcome

            intent = "退款退货" if "退" in query else "闲聊"
            return ClassifyOutcome(intent=intent, confidence=0.9)

    app = create_app()
    app.state.session_factory = session_factory
    app.state.workflow = build_workflow(
        retriever=_FakeRetriever(RetrievalResult(evidences=[], low_confidence=False)),
        agent_node=_FakeAgent(text="不该走到这"),
        intent_classifier=_RouterClassifier("闲聊"),
        session_factory=session_factory,
        refund_prep=_selector_prep,
        checkpointer=InMemorySaver(),
    )
    async with _client(app) as c:
        r1 = await c.post("/api/chat/stream", json={"user_id": "u1", "message": "我要退款", "conversation_id": None})
        cid = sse_events(r1.text)[0][1]["conversation_id"]
        assert "order_selector" in [e for e, _ in sse_events(r1.text)]

        r2 = await c.post("/api/chat/stream", json={"user_id": "u1", "message": "你好", "conversation_id": cid})
        kinds2 = [e for e, _ in sse_events(r2.text)]
        assert "order_selector" not in kinds2  # 未被旧选择器劫持
        text2 = "".join(v["text"] for e, v in sse_events(r2.text) if e == "delta")
        assert text2  # 闲聊轮正常出话术
    async with session_factory() as session:
        msgs = await MessagesRepo(session).load_history(cid)
    texts = [m.text for m in msgs]
    assert "我要退款" in texts and "你好" in texts
    assert any("已取消" in t for t in texts)  # 取消轮也落了库


async def test_history_ctx_logged_including_chitchat(session_factory, db_session, caplog):
    import logging as _logging

    app = create_app()
    app.state.session_factory = session_factory
    app.state.workflow = _make_workflow(session_factory, classifier_intent="闲聊")
    with caplog.at_level(_logging.INFO, logger="app.memory.context_builder"):
        async with _client(app) as c:
            await c.post("/api/chat/stream", json={"user_id": "u1", "message": "在吗", "conversation_id": None})
    assert "history_ctx" in caplog.text and "当前句: 在吗" in caplog.text


async def test_resync_no_duplicate_history(session_factory, db_session):
    """两轮后 DB 恰 4 条、State 不重（wipe+resync 语义）。"""
    agent = _FakeAgent(text="回答")
    app = create_app()
    app.state.session_factory = session_factory
    app.state.workflow = _make_workflow(session_factory, classifier_intent="闲聊", agent=agent)
    async with _client(app) as c:
        r1 = await c.post("/api/chat/stream", json={"user_id": "u1", "message": "第一句", "conversation_id": None})
        cid = sse_events(r1.text)[0][1]["conversation_id"]
        await c.post("/api/chat/stream", json={"user_id": "u1", "message": "第二句", "conversation_id": cid})
    async with session_factory() as session:
        msgs = await MessagesRepo(session).load_history(cid)
    assert len(msgs) == 4  # 2 问 2 答
    snap = await app.state.workflow.aget_state(
        {"configurable": {"thread_id": f"ctx-{cid}"}}
    )
    hist_ids = [m.text for m in snap.values.get("messages", [])]
    assert hist_ids.count("第一句") == 1  # State 里也不翻倍


async def test_agent_receives_trimmed_ctx_history(session_factory, db_session):
    """agent 吃装配器的精简史（ctx_history），不是原始 messages。"""
    from types import SimpleNamespace as _NS

    agent = _FakeAgent(text="ok")
    app = create_app()
    app.state.session_factory = session_factory
    app.state.workflow = _make_workflow(session_factory, classifier_intent="物流", agent=agent)
    app.state.context_budget = _NS(layer1=5, layer2=5000, history=5005)  # 层1极小逼出截短
    async with _client(app) as c:
        r1 = await c.post("/api/chat/stream", json={"user_id": "u1", "message": "长问" + "句" * 40, "conversation_id": None})
        cid = sse_events(r1.text)[0][1]["conversation_id"]
        await c.post("/api/chat/stream", json={"user_id": "u1", "message": "第二问", "conversation_id": cid})
    st = agent.calls[-1]
    assert st.get("ctx_history") is not None
    assert all("…" not in (m.text or "") for m in st["messages"] if isinstance(m, HumanMessage))  # 层2截短只作用于assistant


async def test_no_cross_turn_channel_residue(session_factory, db_session):
    """评审 M2 回归：会话级线程下，知识轮的 citations 不得串进闲聊轮；
    闲聊轮的 final_text 残留不得骗过退款闸（带单号退款轮必须真进 Agent）。"""
    from langgraph.checkpoint.memory import InMemorySaver as _Saver

    retriever = _FakeRetriever(RetrievalResult(evidences=[_ev(8)], low_confidence=False))
    agent = _FakeAgent(text="知识回答 [1]")
    routed = {"intent": "商品咨询"}

    class _DynamicClassifier(_FakeClassifier):
        async def classify_detail(self, query):
            from app.workflow.intent import ClassifyOutcome

            return ClassifyOutcome(intent=routed["intent"], confidence=0.9)

    app = create_app()
    app.state.session_factory = session_factory
    app.state.workflow = build_workflow(
        retriever=retriever,
        agent_node=agent,
        intent_classifier=_DynamicClassifier("商品咨询"),
        session_factory=session_factory,
        checkpointer=_Saver(),
    )
    async with _client(app) as c:
        r1 = await c.post("/api/chat/stream", json={"user_id": "u1", "message": "空气炸锅怎么用", "conversation_id": None})
        cid = sse_events(r1.text)[0][1]["conversation_id"]
        done1 = [v for e, v in sse_events(r1.text) if e == "done"][0]
        assert done1.get("citations")  # 第一轮有引用

        routed["intent"] = "闲聊"
        r2 = await c.post("/api/chat/stream", json={"user_id": "u1", "message": "在吗", "conversation_id": cid})
        done2 = [v for e, v in sse_events(r2.text) if e == "done"][0]
        assert not done2.get("citations"), "上一轮 evidence 残留串进了闲聊轮"
