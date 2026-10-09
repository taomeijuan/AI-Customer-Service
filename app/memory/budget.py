"""ch07 T3：token 预算倒推——从模型窗口扣固定开销与当轮瞬时峰值，历史取期望/匀出较小者。

对账锚（用户验收 2 演示配置）：W18000/O2000/U2000/步3/工1200/K5 →
history=5650、层1=3955(floor 70%)、层2=1695。默认窗口 → history=9600(期望侧封顶)。
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class BudgetSpec:
    peak: int  # 当前轮瞬时峰值（用户输入 + 步数×工具结果）
    fixed: int  # 固定开销（system + 证据 + 梗概注入 + 安全余量）
    window_avail: int  # 窗口匀给历史的额度 A（可为负）
    desired: int  # 期望诉求 D = 想留轮数 × 每轮稳态
    history: int  # min(A, D)，负数归零
    layer1: int
    layer2: int
    ok: bool  # 启动自检：至少装得下一轮稳态


def compute_budget(settings: Any) -> BudgetSpec:
    peak = settings.max_user_input_tokens + settings.max_agent_steps * settings.tool_result_max_tokens
    fixed = (
        settings.system_reserve
        + settings.rerank_top_k * settings.evidence_per_item_reserve
        + settings.summary_inject_reserve
        + settings.safety_reserve
    )
    avail = settings.model_context_window - settings.max_output_tokens - peak - fixed
    desired = settings.keep_turns * settings.turn_steady_tokens
    history = max(0, min(avail, desired))
    layer1 = int(history * 0.7)
    return BudgetSpec(
        peak=peak,
        fixed=fixed,
        window_avail=avail,
        desired=desired,
        history=history,
        layer1=layer1,
        layer2=history - layer1,
        ok=history >= settings.turn_steady_tokens,
    )
