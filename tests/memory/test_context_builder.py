"""ch07 T6 装配与观测单测：固定顺序、材料合成、日志规格。"""

import logging
from types import SimpleNamespace

from langchain.messages import AIMessage, HumanMessage, SystemMessage

from app.memory.context_builder import (
    MATERIAL_HEADER,
    ContextAnchors,
    assemble_model_messages,
    build_context,
    log_context,
)

SETTINGS = SimpleNamespace(
    assistant_trunc_chars=80, rerank_top_k=5, evidence_per_item_reserve=400,
    summary_inject_reserve=500,
)
BUDGET = SimpleNamespace(layer1=10_000, layer2=5_000, history=15_000)


def _pairs():
    return [
        (1, HumanMessage("老问题")),
        (2, AIMessage("老回答")),
        (3, HumanMessage("新问题")),
        (4, AIMessage("新回答")),
    ]


def test_assemble_order_system_history_query_material():
    ctx = build_context(
        _pairs(), anchors=ContextAnchors(), budget=BUDGET, settings=SETTINGS,
        query="现在退货运费谁出", summary_projection="〔第1段〕用户咨询过退货。",
        evidence=[{"n": 1, "question": "退货运费", "answer": "非质量问题是买家承担"}],
    )
    msgs = assemble_model_messages(ctx, system="人设红线")
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[-1], HumanMessage) and msgs[-1].text.startswith(MATERIAL_HEADER)
    assert isinstance(msgs[-2], HumanMessage) and msgs[-2].text == "现在退货运费谁出"
    assert [type(m).__name__ for m in msgs[1:-2]] == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]
    assert "〔第1段〕" in msgs[-1].text and "[1] 退货运费" in msgs[-1].text


def test_no_material_when_no_projection_or_evidence():
    ctx = build_context(_pairs(), anchors=ContextAnchors(), budget=BUDGET, settings=SETTINGS, query="q")
    assert ctx.material is None
    msgs = assemble_model_messages(ctx)
    assert msgs[-1].text == "q" and len(msgs) == 5  # 4 历史 + 当前句，无 system 无材料


def test_evidence_double_cap_count_and_total():
    # 条数帽：单条预算充裕时最多进 RERANK_TOP_K 条
    ev = [{"n": i, "question": f"问{i}", "answer": "短答案"} for i in range(1, 12)]
    ctx = build_context(
        [], anchors=ContextAnchors(), budget=BUDGET, settings=SETTINGS, query="q", evidence=ev,
    )
    assert "[5]" in ctx.material and "[6]" not in ctx.material
    # 总帽：单条很大时提前截断（至少留 1 条）
    big = [{"n": i, "question": f"问{i}", "answer": "答" * 400} for i in range(1, 12)]
    ctx2 = build_context(
        [], anchors=ContextAnchors(), budget=BUDGET,
        settings=SimpleNamespace(assistant_trunc_chars=80, rerank_top_k=5,
                                 evidence_per_item_reserve=50, summary_inject_reserve=500),
        query="q", evidence=big,
    )
    assert "[1]" in ctx2.material and "[2]" not in ctx2.material


def test_log_model_ctx_and_history_ctx_full_spec(caplog):
    ctx = build_context(
        _pairs(), anchors=ContextAnchors(), budget=BUDGET, settings=SETTINGS,
        query="q", summary_projection="〔第1段〕梗概正文。",
    )
    with caplog.at_level(logging.INFO, logger="app.memory.context_builder"):
        log_context("model_ctx", 7, ctx, BUDGET)
    text = caplog.text
    assert "model_ctx conv=7" in text
    assert "条数 L1=4 L2=0" in text
    assert "tokens≈" in text
    assert "梗概正文。" in text          # 摘要**全文**不打折
    assert "user: 老问题" in text       # 滑窗逐条
    assert "当前句: q" in text


def test_split_budget_applied_via_build_context():
    long_ai = (2, AIMessage("字" * 500))
    pairs = [(1, HumanMessage("问")), long_ai, (3, HumanMessage("刚问")), (4, AIMessage("刚答"))]
    ctx = build_context(
        pairs, anchors=ContextAnchors(), budget=SimpleNamespace(layer1=20, layer2=5000, history=5020),
        settings=SETTINGS, query="q",
    )
    assert any("…" in (m.text or "") for m in ctx.plan.layer2)  # 超层1预算的轮被截短进层2
