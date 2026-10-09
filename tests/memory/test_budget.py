"""T3 预算对账：两组配置精确复现验收数字（ch07 的生命线）。"""

from types import SimpleNamespace

from app.memory.budget import compute_budget


def _s(**over):
    base = dict(
        model_context_window=64000, max_output_tokens=2000, max_user_input_tokens=2000,
        max_agent_steps=6, tool_result_max_tokens=1200, system_reserve=2000,
        evidence_per_item_reserve=400, rerank_top_k=10, summary_inject_reserve=500,
        safety_reserve=250, keep_turns=24, turn_steady_tokens=400,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_default_window_cascade_numbers():
    b = compute_budget(_s())
    assert b.peak == 9200 and b.fixed == 6750
    assert b.window_avail == 46050 and b.desired == 9600
    assert b.history == 9600  # 期望侧封顶（验收3：20 轮装得下）
    assert b.layer1 == 6720 and b.layer2 == 2880
    assert b.ok is True


def test_demo_config_matches_acceptance_numbers():
    """验收 2 演示配置：history=5650、层1=3954（int 浮点地板，与验收原值一致）、
    层2=history−层1=1696（自洽约束；验收给的 1695 与 3954 相加差 1，取残差规则）。"""
    b = compute_budget(_s(
        model_context_window=18000, max_agent_steps=3, rerank_top_k=5,
    ))
    assert b.peak == 5600 and b.fixed == 4750
    assert b.history == 5650  # 窗口侧成为瓶颈
    assert b.layer1 == 3954 and b.layer2 == 1696
    assert b.layer1 + b.layer2 == 5650  # 两层严格自洽
    assert b.ok is True


def test_tiny_window_reports_insufficient_and_zero_history():
    b = compute_budget(_s(model_context_window=9000))  # 9000-2000-9200-6750 < 0
    assert b.ok is False
    assert b.history == 0 and b.layer1 == 0 and b.layer2 == 0
