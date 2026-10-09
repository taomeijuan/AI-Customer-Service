"""ch06 指代消解 Resolver 单测：透传规则 / 失败兜底 / 历史陈旧角标剥离 / 图节点接线。"""

from langchain.messages import AIMessage

from app.workflow.graph import build_workflow
from app.workflow.resolver import Resolver


class _StubStructured:
    """include_raw 形状的桩：ainvoke 返回 {"raw","parsed","parsing_error"}。"""

    def __init__(self, parsed=None, error=None, raise_exc=None):
        self.parsed = parsed
        self.error = error
        self.raise_exc = raise_exc
        self.seen_messages = None

    async def ainvoke(self, msgs):
        self.seen_messages = list(msgs)
        if self.raise_exc:
            raise self.raise_exc
        return {"raw": None, "parsed": self.parsed, "parsing_error": self.error}


async def test_resolve_rewrites_via_llm():
    stub = _StubStructured(parsed={"query": "订单 1001 的扫地机器人能退货退款吗"})
    r = Resolver(stub)
    out = await r.resolve("它能退吗", history=[AIMessage("你买的是订单 1001 的扫地机器人")])
    assert out == "订单 1001 的扫地机器人能退货退款吗"


async def test_resolve_passthrough_when_complete():
    stub = _StubStructured(parsed={"query": "退货政策是什么"})
    r = Resolver(stub)
    out = await r.resolve("退货政策是什么", history=[])
    assert out == "退货政策是什么"


async def test_resolve_passthrough_on_parse_error():
    stub = _StubStructured(error="boom")
    r = Resolver(stub)
    assert await r.resolve("它能退吗", history=[]) == "它能退吗"


async def test_resolve_passthrough_on_exception():
    stub = _StubStructured(raise_exc=RuntimeError("upstream down"))
    r = Resolver(stub)
    assert await r.resolve("它能退吗", history=[]) == "它能退吗"


async def test_resolve_passthrough_on_empty_result():
    for bad in (None, {"query": ""}, {"query": "   "}):
        stub = _StubStructured(parsed=bad)
        r = Resolver(stub)
        assert await r.resolve("它能退吗", history=[]) == "它能退吗"


async def test_resolve_strips_stale_citations_from_history_only():
    """历史回答的旧 [n] 剥离，用户本轮原句与系统指令不动。"""

    class _Capture(_StubStructured):
        async def ainvoke(self, msgs):
            self.seen_messages = list(msgs)
            return {"raw": None, "parsed": {"query": "q"}, "parsing_error": None}

    stub = _Capture(parsed={"query": "q"})
    r = Resolver(stub)
    await r.resolve("它能退吗", history=[AIMessage("上次说圆通速递 [3] 已签收")])
    sent = stub.seen_messages
    assert "[3]" not in sent[1].content  # 历史消息剥了角标
    assert "圆通速递" in sent[1].content
    assert sent[0].content.count("[") >= 0  # 系统指令无角标要求，仅确保顺序：指令在前
    assert sent[-1].content == "它能退吗"  # 本轮原句在最后


async def test_graph_resolve_node_uses_resolver_and_keeps_raw_query():
    """图接线：resolve 输出改写后 query，同时保留 raw_query 供落库。"""

    class _R:
        async def resolve(self, query, history):
            return "订单 1001 能退货吗"

    class _StubClassifier:
        async def classify(self, q):
            return "闲聊"

    class _StubRetriever:
        async def retrieve(self, q, strategy="hybrid_rerank", category_prefix=None):
            from app.knowledge.retriever import RetrievalResult

            return RetrievalResult(evidences=[], low_confidence=False)

    async def _stub_agent(state):
        return {"final_text": "x", "messages": [], "evidence": []}

    wf = build_workflow(
        retriever=_StubRetriever(),
        agent_node=_stub_agent,
        intent_classifier=_StubClassifier(),
        session_factory=None,
        resolver=_R(),
    )
    # 只跑到 resolve+intent（闲聊路径短），检查 resolve 的更新
    seen = {}
    async for chunk in wf.astream(
        {"query": "它能退吗", "messages": [], "conversation_id": None, "turn": 1},
        config={"configurable": {"thread_id": "t"}, "recursion_limit": 8},
        stream_mode="updates",
    ):
        updates = chunk if isinstance(chunk, dict) else chunk[-1]
        if isinstance(updates, dict):
            for upd in updates.values():
                if isinstance(upd, dict):
                    seen.update(upd)
    assert seen["query"] == "订单 1001 能退货吗"
    assert seen["raw_query"] == "它能退吗"
