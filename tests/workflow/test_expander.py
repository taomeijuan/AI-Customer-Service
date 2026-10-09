"""ch06 扩写器与多路检索合并单测。"""

from app.knowledge.retriever import HybridRetriever, RetrievalResult
from app.knowledge.reranker import Evidence
from app.workflow.expander import Expander


# ---------- Expander ----------

class _StubStructured:
    def __init__(self, parsed=None, error=None, raise_exc=None):
        self.parsed = parsed
        self.error = error
        self.raise_exc = raise_exc

    async def ainvoke(self, msgs):
        if self.raise_exc:
            raise self.raise_exc
        return {"raw": None, "parsed": self.parsed, "parsing_error": self.error}


async def test_expand_returns_distinct_queries():
    stub = _StubStructured(parsed={"queries": ["七天无理由退货条件", "退货申请时间限制", "七天无理由退货条件"]})
    out = await Expander(stub).expand("这个能退吗")
    assert out == ["七天无理由退货条件", "退货申请时间限制"]  # 去重保序


async def test_expand_caps_at_four():
    stub = _StubStructured(parsed={"queries": [f"查询{i}" for i in range(10)]})
    assert len(await Expander(stub).expand("退货")) == 4


async def test_expand_fallback_on_parse_error():
    stub = _StubStructured(error="bad json")
    assert await Expander(stub).expand("怎么退款") == ["怎么退款"]


async def test_expand_fallback_on_too_few():
    stub = _StubStructured(parsed={"queries": ["只有一条"]})
    assert await Expander(stub).expand("怎么退款") == ["怎么退款"]


async def test_expand_fallback_on_exception():
    stub = _StubStructured(raise_exc=RuntimeError("down"))
    assert await Expander(stub).expand("怎么退款") == ["怎么退款"]


async def test_expand_passes_context_in_user_content():
    class _Capture(_StubStructured):
        async def ainvoke(self, msgs):
            self.seen = list(msgs)
            return {"raw": None, "parsed": {"queries": ["q1", "q2"]}, "parsing_error": None}

    stub = _Capture()
    await Expander(stub).expand("这个能退吗", context="订单 1001：扫地机器人")
    assert "订单 1001：扫地机器人" in stub.seen[0]


# ---------- retrieve_multi ----------

class _FakeRetriever:
    """替换 retrieve：记录调用并按预设返回。"""

    def __init__(self, scripted: dict[str, RetrievalResult]):
        self.scripted = scripted
        self.called = []

    async def retrieve(self, query, strategy="hybrid_rerank", category_prefix=None):
        self.called.append(query)
        return self.scripted.get(query, RetrievalResult(evidences=[], low_confidence=True))


async def test_multi_dedupes_by_chunk_id_keeping_highest_score():
    r1 = RetrievalResult(evidences=[
        Evidence(chunk_id=1, section_path="a", question="q", answer="x", score=0.8),
        Evidence(chunk_id=2, section_path="b", question="q", answer="y", score=0.5),
    ], low_confidence=False)
    r2 = RetrievalResult(evidences=[
        Evidence(chunk_id=1, section_path="a", question="q", answer="x", score=0.9),  # 同 chunk 更高分
        Evidence(chunk_id=3, section_path="c", question="q", answer="z", score=0.7),
    ], low_confidence=False)
    fake = _FakeRetriever({"q1": r1, "q2": r2})
    out = await HybridRetriever.retrieve_multi(fake, ["q1", "q2"])
    assert [e.chunk_id for e in out.evidences] == [1, 3, 2]  # 分数降序：1(0.9) > 3(0.7) > 2(0.5)
    assert out.evidences[0].score == 0.9  # 同 chunk 取最高
    assert out.low_confidence is False  # 有一路置信即放行


async def test_multi_low_conf_only_when_all_paths_weak():
    weak = RetrievalResult(evidences=[Evidence(9, "s", "q", "a", 0.1)], low_confidence=True)
    strong = RetrievalResult(evidences=[Evidence(8, "s", "q", "a", 0.8)], low_confidence=False)
    fake = _FakeRetriever({"q1": weak, "q2": strong})
    assert (await HybridRetriever.retrieve_multi(fake, ["q1", "q2"])).low_confidence is False

    fake2 = _FakeRetriever({"q1": weak, "q2": RetrievalResult(evidences=[], low_confidence=True)})
    out = await HybridRetriever.retrieve_multi(fake2, ["q1", "q2"])
    assert out.low_confidence is True and [e.chunk_id for e in out.evidences] == [9]


async def test_multi_empty_queries_guard():
    fake = _FakeRetriever({})
    out = await HybridRetriever.retrieve_multi(fake, ["", "  "])
    assert out.evidences == [] and out.low_confidence is True
