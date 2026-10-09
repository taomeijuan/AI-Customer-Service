"""ch06 退款子流程测试：提单号直通 / 槽位中断 / Command(resume) 恢复 / 证据注入。"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.knowledge.retriever import RetrievalResult
from app.knowledge.reranker import Evidence
from app.workflow.graph import build_workflow
from app.workflow.refund_flow import build_refund_prep


class _StubClassifier:
    def __init__(self, intent):
        self.intent = intent

    async def classify_detail(self, query):
        from app.workflow.intent import ClassifyOutcome

        return ClassifyOutcome(intent=self.intent, confidence=0.9)

    async def classify(self, query):
        return self.intent


class _StubResolver:
    async def resolve(self, query, history):
        return query  # 透传


class _StubRetriever:
    """实现 retrieve + retrieve_multi（子流程走多路检索）。"""

    def __init__(self, evidences=None):
        self.evidences = evidences or [
            Evidence(chunk_id=21, section_path="售后政策", question="七天无理由退货条件", answer="签收后7天内可退", score=0.8),
            Evidence(chunk_id=22, section_path="售后政策", question="退款时限", answer="1-3个工作日原路退回", score=0.7),
        ]

    async def retrieve(self, query, strategy="hybrid_rerank", category_prefix=None):
        return RetrievalResult(evidences=self.evidences, low_confidence=False)

    async def retrieve_multi(self, queries, strategy="hybrid_rerank", category_prefix=None):
        self.multi_called_with = list(queries)
        return RetrievalResult(evidences=self.evidences, low_confidence=False)


class _StubExpander:
    def __init__(self, queries=None):
        self.queries = queries or ["七天无理由退货条件", "拆封商品能否退货"]
        self.seen_context = None

    async def expand(self, query, context=None):
        self.seen_context = context
        return list(self.queries)


class _FakeAgent:
    def __init__(self, text="这一单可以退"):
        self.text = text
        self.calls = []

    async def __call__(self, state):
        self.calls.append(dict(state))
        return {"final_text": self.text, "messages": [], "evidence": state.get("evidence", [])}


def _make(classifier_intent="退款退货", retriever=None, expander=None, agent=None):
    retriever = retriever or _StubRetriever()
    refund_prep = build_refund_prep(
        retriever=retriever,
        expander=expander or _StubExpander(),
    )
    agent = agent or _FakeAgent()
    wf = build_workflow(
        retriever=retriever,
        agent_node=agent,
        intent_classifier=_StubClassifier(classifier_intent),
        session_factory=None,
        resolver=_StubResolver(),
        refund_prep=refund_prep,
        checkpointer=InMemorySaver(),
    )
    return wf, agent, retriever


CFG = {"configurable": {"thread_id": "t1"}, "recursion_limit": 12}


async def test_refund_with_order_no_goes_straight_through():
    """问句自带订单号 → 不弹选择器，扩写检索 → 证据+订单注入 Agent。"""
    expander = _StubExpander()
    wf, agent, retriever = _make(expander=expander)
    out = await wf.ainvoke({"query": "订单1001能退吗", "messages": []}, config=CFG)
    assert agent.calls and len(agent.calls) == 1
    assert out["final_text"] == "这一单可以退"
    assert [c["chunk_id"] for c in out["evidence"]] == [21, 22]
    assert out["order_no"] == "1001"
    assert retriever.multi_called_with == ["七天无理由退货条件", "拆封商品能否退货"]
    # 订单数据进了 Agent 注入消息
    injected = str(agent.calls[0].get("messages") or [])
    assert "扫地机器人" in injected or "订单 1001" in injected
    assert "七天无理由" in injected


async def test_refund_without_order_no_interrupts_then_resumes():
    """「这个能退吗」无单号 → interrupt 弹选择器；resume 回填后从断点续跑到 Agent。"""
    wf, agent, _ = _make()
    out1 = await wf.ainvoke({"query": "这个能退吗", "messages": []}, config=CFG)
    # ainvoke 结果里中断以 __interrupt__ 浮出（与 astream 探针一致的信号形态）
    assert "__interrupt__" in out1
    intr = out1["__interrupt__"][0]
    assert intr.value["type"] == "order_selector"
    assert [o["order_no"] for o in intr.value["orders"]] == ["1001", "1002", "1003", "1004", "1005"]
    assert agent.calls == []  # 中断时未进 Agent

    out2 = await wf.ainvoke(Command(resume={"order_no": "1002"}), config=CFG)
    assert out2["final_text"] == "这一单可以退"
    assert out2["order_no"] == "1002"
    assert len(agent.calls) == 1  # 恢复后恰好进一次 Agent


async def test_refund_cancel_selection_goes_to_log():
    """resume 不带单号（用户取消）→ 子流程结束语直达 log，不进 Agent。"""
    wf, agent, _ = _make()
    await wf.ainvoke({"query": "这个能退吗", "messages": []}, config=CFG)
    out = await wf.ainvoke(Command(resume={"order_no": ""}), config=CFG)
    assert "先不继续" in out["final_text"]
    assert agent.calls == []


async def test_refund_weak_evidence_falls_back():
    """扩写检索证据弱 → fallback 兜底话术（经退款闸），不进 Agent。"""
    weak = _StubRetriever(evidences=[Evidence(9, "s", "q", "a", 0.1)])
    # 让 retrieve_multi 返回弱置信
    weak.multi_low = True

    async def retrieve_multi_low(queries, strategy="hybrid_rerank", category_prefix=None):
        from app.knowledge.retriever import RetrievalResult as RR

        return RR(evidences=[], low_confidence=True)

    weak.retrieve_multi = retrieve_multi_low
    wf, agent, _ = _make(retriever=weak)
    out = await wf.ainvoke({"query": "订单1001能退吗", "messages": []}, config=CFG)
    assert out["refusal"] is True
    assert "暂时没有足够的资料" in out["final_text"]
    assert agent.calls == []


async def test_refund_regex_candidate_must_pass_whitelist():
    """评审 M3：数字在白名单外（年份/金额）不得当订单号——照走选择器。"""
    wf, agent, _ = _make()
    out = await wf.ainvoke({"query": "1999 元那单能退吗", "messages": []}, config=CFG)
    assert "__interrupt__" in out  # 1999 不在 1001-1005 → 提不到有效号 → 弹选择器


async def test_resume_illegal_order_no_treated_as_cancel():
    """评审 M3：resume 回传白名单外的单号按取消处理，绝不喂假数据生成器。"""
    wf, agent, _ = _make()
    await wf.ainvoke({"query": "这个能退吗", "messages": []}, config=CFG)
    out = await wf.ainvoke(Command(resume={"order_no": "888888"}), config=CFG)
    assert "先不继续" in out["final_text"]
    assert agent.calls == []


async def test_resume_order_brief_reaches_options_channel():
    """options 帧载荷：order_brief 通道随 refund_prep 更新浮出（前端表单数据源）。"""
    wf, _, _ = _make()
    await wf.ainvoke({"query": "这个能退吗", "messages": []}, config=CFG)
    async for chunk in wf.astream(
        Command(resume={"order_no": "1001"}), config=CFG, stream_mode="updates"
    ):
        upd = chunk.get("refund_prep") if isinstance(chunk, dict) else None
        if upd and "order_brief" in upd:
            assert upd["order_brief"]["order_no"] == "1001"
            assert upd["options"] == ["申请退款"]
            break
    else:
        raise AssertionError("refund_prep 更新里没看到 order_brief/options")


async def test_resolved_query_number_does_not_bypass_selector():
    """评审 m8 改判回归：单号出现在消解改写句（从历史推断）而不在用户原话里
    → 必须仍弹选择器。直通的唯一合法来源是原话自带单号。"""
    wf, agent, _ = _make()
    out = await wf.ainvoke(
        {"query": "订单1001怎么申请退款", "raw_query": "我要退款", "messages": []},
        config=CFG,
    )
    assert "__interrupt__" in out, "消解推断出的单号不应视为用户给了单号"
    assert agent.calls == []
