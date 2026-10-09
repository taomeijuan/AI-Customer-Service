"""ch07 三层切分与层 2 规则截短渲染（纯函数，不碰 DB）。

边界靠消息 id 表达（用户 DDL 语义）：
    id ≤ summary_upto            层0：已进摘要，不进窗口
    summary_upto < id ≤ layer1   层2：半压形态（持久锚划定的降级区）
    id > layer1                  层1：原文（本轮再超预算的部分动态滑入层2）

轮是原子单位（ch02 教训：工具申请单与结果不可裁开），按 HumanMessage 分轮；
层 2 截短后仍超预算 → 最旧整轮成为摘要批次候选（交给后台任务，压完推锚）。
"""

from dataclasses import dataclass, field

from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages import BaseMessage

from app.memory.tokens import count_message_tokens


@dataclass
class LayerPlan:
    layer1: list[BaseMessage] = field(default_factory=list)        # 原文
    layer2: list[BaseMessage] = field(default_factory=list)        # 截短渲染
    summarize_batch: list[tuple[int, BaseMessage]] = field(default_factory=list)  # (id,msg) 最旧批候选
    layer1_tokens: int = 0
    layer2_tokens: int = 0
    new_layer1_from: int | None = None  # 本轮动态降级后的层1锚（> 此 id 为层1）；层1空时 None


def group_rounds(pairs: list[tuple[int, BaseMessage]]) -> list[list[tuple[int, BaseMessage]]]:
    """按 HumanMessage 分轮（轮=原子裁剪/摘要单位）。开头非 user 的孤儿归进首轮。"""
    rounds: list[list[tuple[int, BaseMessage]]] = []
    for pair in pairs:
        if isinstance(pair[1], HumanMessage) or not rounds:
            rounds.append([pair])
        else:
            rounds[-1].append(pair)
    return rounds


def round_tokens(round_msgs: list[tuple[int, BaseMessage]]) -> int:
    return sum(count_message_tokens(m) for _, m in round_msgs)


def render_layer2_round(
    round_msgs: list[tuple[int, BaseMessage]], trunc_chars: int
) -> list[BaseMessage]:
    """层 2 规则截短：用户原话一个字不动；assistant 只留开头 N 字；
    大块工具结果换成一行标识（申请单同样收成一行，成对出现保持轮语义完整）。"""
    out: list[BaseMessage] = []
    for _, m in round_msgs:
        if isinstance(m, HumanMessage):
            out.append(m)
        elif isinstance(m, ToolMessage):
            out.append(AIMessage(content=f"【工具 {m.name or 'tool'} 结果略】"))
        elif isinstance(m, AIMessage) and m.tool_calls:
            names = "、".join(tc.get("name", "tool") for tc in m.tool_calls)
            out.append(AIMessage(content=f"【调用工具 {names}，详情略】"))
        elif isinstance(m, AIMessage):
            text = m.text or ""
            if len(text) > trunc_chars:
                text = text[:trunc_chars] + "…"
            out.append(AIMessage(content=text))
        else:
            out.append(m)
    return out


def split_layers(
    pairs: list[tuple[int, BaseMessage]],
    *,
    summary_upto: int,
    layer1_from: int | None,
    layer1_budget: int,
    layer2_budget: int,
    trunc_chars: int,
) -> LayerPlan:
    """切三层。summary_upto/layer1_from 为持久锚（None=未设）；动态溢出在本函数内产生。"""
    eligible = [(mid, m) for mid, m in pairs if mid > summary_upto]
    rounds = group_rounds(eligible)
    plan = LayerPlan()
    if not rounds:
        return plan

    # ① 按持久锚分区：轮末消息 id ≤ 锚 → 该轮属层2区（zone2）；其余为层1候选（zone1）
    anchor = layer1_from if layer1_from is not None else eligible[0][0] - 1
    zone2_idx = {i for i, r in enumerate(rounds) if r[-1][0] <= anchor}
    zone2 = [r for i, r in enumerate(rounds) if i in zone2_idx]
    zone1 = [r for i, r in enumerate(rounds) if i not in zone2_idx]

    # ② 层1：zone1 从最新往回装原文，超预算的最旧连续段滑入层2候选
    l1: list[list[tuple[int, BaseMessage]]] = []
    l1_cost = 0
    for rnd in reversed(zone1):
        cost = round_tokens(rnd)
        if l1 and l1_cost + cost > layer1_budget:
            break
        l1.insert(0, rnd)
        l1_cost += cost
    spilled = zone1[: len(zone1) - len(l1)]
    if l1:
        plan.new_layer1_from = l1[0][0][0] - 1

    # ③ 层2：zone2 + 本轮降级轮（旧→新），截短渲染从最新往回装；再溢出=摘要批次候选
    l2_source = zone2 + spilled
    rendered: list[list[BaseMessage]] = []
    l2_cost = 0
    keep_from = len(l2_source)  # 保留区起点（l2_source 索引）
    for i in range(len(l2_source) - 1, -1, -1):
        stubs = render_layer2_round(l2_source[i], trunc_chars)
        cost = sum(count_message_tokens(m) for m in stubs)
        if rendered and l2_cost + cost > layer2_budget:
            break
        rendered.insert(0, stubs)
        l2_cost += cost
        keep_from = i
    overflow_rounds = l2_source[:keep_from]

    plan.layer1 = [m for rnd in l1 for _, m in rnd]
    plan.layer2 = [m for stubs in rendered for m in stubs]
    plan.layer1_tokens, plan.layer2_tokens = l1_cost, l2_cost
    plan.summarize_batch = [p for rnd in overflow_rounds for p in rnd]
    return plan
