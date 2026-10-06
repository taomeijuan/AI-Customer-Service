import pytest

from app.repositories.faith_cases import FaithCasesRepo
from app.repositories.low_confidence import LowConfidenceRepo

CITATIONS = [
    {"n": 1, "chunk_id": 5, "section_path": "售后政策/退款/退款时限", "question": "退款时限", "answer": "1-3个工作日"}
]


@pytest.mark.usefixtures("db_session")
async def test_low_confidence_insert(db_session):
    repo = LowConfidenceRepo(db_session)
    row_id = await repo.record(
        raw_question="量子速递什么时候发明的",
        source="self_check",
        reason="知识库无相关内容",
        conversation_id=None,
    )
    row = await repo.get(row_id)
    assert row.source == "self_check" and row.created_at is not None


@pytest.mark.usefixtures("db_session")
async def test_faith_case_new_then_recur(db_session):
    """台账核心语义：一题一行、跨轮 seen_count 累加、复发退回未解决。"""
    repo = FaithCasesRepo(db_session)
    await repo.record(
        eval_id="A01", bucket="A_policy", query="退款多久到账", strategy="hybrid_rerank",
        answer="保证3分钟到账", reason="证据里没有任何到账承诺",
        citations=CITATIONS, judge_model="deepseek-chat",
    )
    row = await repo.get("A01")
    assert row.seen_count == 1 and row.status == "未解决"
    assert row.citations == CITATIONS and row.judge_model == "deepseek-chat"

    await repo.mark_resolved("A01", resolution="知识库补了到账说明", status="已解决")
    row = await repo.get("A01")
    assert row.status == "已解决" and row.resolved_at is not None

    await repo.record(  # 复发：同一题再次被判编造
        eval_id="A01", bucket="A_policy", query="退款多久到账", strategy="hybrid_rerank",
        answer="肯定当天到账", reason="仍在承诺到账时间",
        citations=CITATIONS, judge_model="deepseek-chat",
    )
    row = await repo.get("A01")
    assert row.seen_count == 2
    assert row.status == "未解决"  # 复发退回待处理
    assert row.resolved_at is not None  # 保留上次解决时间，标记「复发」
    assert row.resolution is None  # 处置说明清空
    assert row.answer == "肯定当天到账"  # 答案快照更新为最近一次


@pytest.mark.usefixtures("db_session")
async def test_faith_case_no_need_fix(db_session):
    repo = FaithCasesRepo(db_session)
    await repo.record(
        eval_id="C02", bucket="C_colloquial", query="q", strategy="hybrid_rerank",
        answer="x", reason="y", citations=None, judge_model="deepseek-chat",
    )
    await repo.mark_resolved("C02", resolution="故意设置的编造样例，评估用", status="无需解决")
    row = await repo.get("C02")
    assert row.status == "无需解决" and "评估用" in row.resolution
