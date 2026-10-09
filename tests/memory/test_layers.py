"""ch07 三层切分与层2渲染单测（轮原子性、截短规则、动态溢出、持久锚语义）。"""

from langchain.messages import AIMessage, HumanMessage, ToolMessage

from app.memory.layers import render_layer2_round, split_layers


def _round(start_id: int, user: str, ai: str, tool: bool = False):
    msgs = [(start_id, HumanMessage(user)), (start_id + 1, AIMessage(ai))]
    if tool:
        msgs = [
            (start_id, HumanMessage(user)),
            (start_id + 1, AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": f"c{start_id}", "type": "tool_call"}])),
            (start_id + 2, ToolMessage('{"order_no": "1001"}', tool_call_id=f"c{start_id}", name="query_order")),
            (start_id + 3, AIMessage(ai)),
        ]
    return msgs


def _long(text: str, n: int):
    return text * n


def _split(rounds_msgs, **kw):
    pairs = [p for r in rounds_msgs for p in r]
    base = dict(summary_upto=0, layer1_from=None, layer1_budget=10_000, layer2_budget=10_000, trunc_chars=80)
    base.update(kw)
    return split_layers(pairs, **base)


def test_small_history_all_layer1():
    plan = _split([_round(1, "问1", "答1"), _round(5, "问2", "答2")])
    assert len(plan.layer1) == 4 and not plan.layer2 and not plan.summarize_batch


def test_layer1_overflow_demotes_oldest_round_to_layer2():
    big = _round(1, "问1", _long("很长回答", 300))  # ~1200 tokens
    recent = _round(5, "问2", "答2")
    plan = _split([big, recent], layer1_budget=300, layer2_budget=5000)
    assert plan.layer1 and all(isinstance(m, HumanMessage) or (m.text or "") == "答2" for m in plan.layer1)
    assert plan.layer2, "降级轮应在层2"
    assert plan.new_layer1_from is not None and plan.new_layer1_from >= 4
    assert not plan.summarize_batch


def test_layer2_rules_truncation_and_tool_stub():
    tool_round = _round(1, "查订单", "已发货", tool=True)
    long_ai = _round(6, "问2", _long("字", 500))
    newest = _round(12, "刚问", "刚答")  # 保底最新一轮在层1，前两轮都被挤到层2
    plan = _split(
        [tool_round, long_ai, newest],
        layer1_budget=30,
        layer2_budget=5000,
        trunc_chars=80,
    )
    texts = [m.text for m in plan.layer2]
    assert "查订单" in texts and "问2" in texts  # 用户原话一个字不动
    assert any("【调用工具 query_order" in t for t in texts)  # 申请单一行标识
    assert any("【工具 query_order 结果略】" in t for t in texts)  # 结果一行标识
    assert any(t.startswith("字" * 80 + "…") for t in texts)  # assistant 留头 80 字
    # 轮原子性：标识成对出现，没有孤儿 ToolMessage
    assert all(not isinstance(m, ToolMessage) for m in plan.layer2)


def test_summarize_batch_is_oldest_overflow():
    r1 = _round(1, "老问题1", "老答案1")
    r2 = _round(3, "老问题2", "老答案2")
    r3 = _round(5, "新问", "新答")
    plan = _split([r1, r2, r3], layer1_budget=12, layer2_budget=8, trunc_chars=10)
    assert plan.summarize_batch, "层2超预算要有摘要候选"
    ids = {mid for mid, _ in plan.summarize_batch}
    assert 1 in ids or 2 in ids  # 最旧的先出去
    assert 5 not in ids


def test_persistent_anchors_respected():
    # 持久锚 layer1_from=4：id≤4 的轮恒为层2区（即便层1预算装得下）
    plan = _split(
        [_round(1, "问1", "答1"), _round(5, "问2", "答2")],
        layer1_from=4,
    )
    assert "问1" in [m.text for m in plan.layer2]  # 锚前区渲染进层2
    assert "问2" in [m.text for m in plan.layer1]


def test_summary_upto_excludes_layer0():
    plan = _split(
        [_round(1, "已被摘要", "答"), _round(5, "问2", "答2")],
        summary_upto=4,
    )
    texts1 = [m.text for m in plan.layer1]
    assert "已被摘要" not in texts1 + [m.text for m in plan.layer2]
    assert "问2" in texts1


def test_empty_input():
    plan = _split([])
    assert not plan.layer1 and not plan.layer2 and not plan.summarize_batch


def test_default_window_20_rounds_zero_demote_zero_summary():
    """验收 3：默认配置预算（6720/2880）下 20 轮正常对话不应触发任何降级/摘要。"""
    from app.core.config import get_settings
    from app.memory.budget import compute_budget

    b = compute_budget(get_settings())
    assert b.layer1 == 6720 and b.layer2 == 2880
    rounds = []
    mid = 1
    for k in range(20):
        rounds.append([(mid, HumanMessage(f"问题{k}：这单物流到哪了")), (mid + 1, AIMessage(f"回答{k}：已到杭州转运中心，预计明天派送"))])
        mid += 2
    pairs = [p for r in rounds for p in r]
    plan = split_layers(
        pairs, summary_upto=0, layer1_from=None,
        layer1_budget=b.layer1, layer2_budget=b.layer2,
        trunc_chars=80,
    )
    assert not plan.layer2, "装得下就不压：层2 应为空"
    assert not plan.summarize_batch, "不该触发摘要"
    assert len(plan.layer1) == 40
