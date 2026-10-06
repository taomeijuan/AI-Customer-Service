import pytest

from app.evals.faithfulness import FaithfulnessJudge, aggregate_report
from app.repositories.faith_cases import FaithCasesRepo


class Out:
    def __init__(self, faithful, reason):
        self.faithful = faithful
        self.reason = reason


class FakeJudgeLLM:
    """with_structured_output(include_raw=True) 形态；按调用序返回脚本。"""

    def __init__(self, outs):
        self.outs = list(outs)

    def with_structured_output(self, schema, **kwargs):
        assert kwargs.get("method") == "function_calling"
        return self

    async def ainvoke(self, messages):
        out = self.outs.pop(0)
        return {"raw": None, "parsed": Out(*out), "parsing_error": None}


CITATIONS = [
    {"n": 1, "chunk_id": 6, "section_path": "售后政策/退款/退款时限", "question": "退款时限", "answer": "1-3个工作日"},
]


@pytest.mark.usefixtures("db_session")
async def test_faithful_answer_no_case(db_session):
    judge = FaithfulnessJudge(FakeJudgeLLM([(True, "完全被证据支持")]), FaithCasesRepo(db_session))
    verdict = await judge.verify(
        eval_id="A01", bucket="A_policy", query="退款多久到账",
        strategy="hybrid_rerank", answer="退款通常1-3个工作日[1]到账",
        citations=CITATIONS,
    )
    assert verdict.faithful is True
    assert await FaithCasesRepo(db_session).get("A01") is None  # 忠实回答不入台账


@pytest.mark.usefixtures("db_session")
async def test_hallucination_recorded(db_session):
    judge = FaithfulnessJudge(FakeJudgeLLM([(False, "「保证3分钟到账」证据里没有")]), FaithCasesRepo(db_session))
    verdict = await judge.verify(
        eval_id="A01", bucket="A_policy", query="退款多久到账",
        strategy="hybrid_rerank", answer="保证3分钟到账",
        citations=CITATIONS,
    )
    assert verdict.faithful is False
    row = await FaithCasesRepo(db_session).get("A01")
    assert row is not None and row.status == "未解决"
    assert row.citations == CITATIONS  # 快照 Top-K 全集
    assert row.judge_model == "deepseek-chat"


@pytest.mark.usefixtures("db_session")
async def test_recur_updates_snapshot(db_session):
    judge = FaithfulnessJudge(
        FakeJudgeLLM([(False, "编1"), (False, "编2")]), FaithCasesRepo(db_session)
    )
    for answer in ("第一版编造", "第二版编造"):
        await judge.verify(
            eval_id="A01", bucket="A_policy", query="q", strategy="dense",
            answer=answer, citations=CITATIONS,
        )
    row = await FaithCasesRepo(db_session).get("A01")
    assert row.seen_count == 2 and row.answer == "第二版编造" and row.strategy == "dense"


def test_aggregate_report_buckets():
    rows = [
        {"bucket": "A_policy", "recall@5": 0.8, "mrr": 0.6, "faithfulness": 0.9},
        {"bucket": "A_policy", "recall@5": 0.6, "mrr": 0.4, "faithfulness": 1.0},
        {"bucket": "B_model", "recall@5": 1.0, "mrr": 1.0, "faithfulness": 0.5},
    ]
    report = aggregate_report(rows)
    assert report["A_policy"]["recall@5"] == pytest.approx(0.7)
    assert report["A_policy"]["n"] == 2  # n=桶内题数（行数），均值按题算
    assert report["B_model"]["mrr"] == pytest.approx(1.0)
