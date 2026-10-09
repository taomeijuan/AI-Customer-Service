"""ch07 T6：模型上下文装配——固定顺序拼装 + 可观测日志（纯函数）。

装配顺序（需求 3，防上游模板把 system 上提合并毁前缀缓存）：
    ① system 人设+红线+工具定义（每轮逐字相同；react 的 prompt/tools 天然占此位）
    ② 层2 截短消息  ③ 层1 原文消息
    ④ HumanMessage 用户当前句
    ⑤ 梗概投影 + 检索证据 → 合成**一条 user 消息**挂在④之后（自带非用户发言标注）

日志规格（需求 6）：model_ctx（Agent 每轮：摘要全文+滑窗逐条+条数+tokens 估算）；
history_ctx（resolve/intent 共用精简版，每轮必打）。
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.messages import BaseMessage

from app.memory.layers import LayerPlan, split_layers
from app.memory.tokens import estimate_tokens

logger = logging.getLogger(__name__)

MATERIAL_HEADER = "【背景材料与参考证据（系统注入，非用户发言）】"


@dataclass
class ContextAnchors:
    summary_upto: int = 0
    layer1_from: int | None = None


@dataclass
class ModelContext:
    history: list[BaseMessage] = field(default_factory=list)  # 层2+层1，按时间序
    query: str = ""
    material: str | None = None                                # ⑤ 合成文本
    plan: LayerPlan = field(default_factory=LayerPlan)
    tokens_est: int = 0


def build_context(
    pairs: list[tuple[int, BaseMessage]],
    *,
    anchors: ContextAnchors,
    budget: Any,  # BudgetSpec（layer1/layer2）
    settings: Any,
    query: str,
    summary_projection: str | None = None,
    evidence: list[dict] | None = None,
) -> ModelContext:
    plan = split_layers(
        pairs,
        summary_upto=anchors.summary_upto,
        layer1_from=anchors.layer1_from,
        layer1_budget=budget.layer1,
        layer2_budget=budget.layer2,
        trunc_chars=settings.assistant_trunc_chars,
    )
    history = [*plan.layer2, *plan.layer1]

    parts: list[str] = []
    if summary_projection:
        parts.append(MATERIAL_HEADER)
        parts.append(f"早期对话梗概：\n{summary_projection}")
    if evidence:
        if not parts:
            parts.append(MATERIAL_HEADER)
        # 证据条数与单条长度受 RERANK_TOP_K×单条预留 总预算封顶
        cap = settings.rerank_top_k * settings.evidence_per_item_reserve
        lines: list[str] = []
        used = estimate_tokens("\n".join(lines)) if lines else 0
        for i, ev in enumerate(evidence[: settings.rerank_top_k], start=1):
            line = f"[{i}] {ev.get('question', '')}：{ev.get('answer', '')}"
            cost = estimate_tokens(line)
            if lines and used + cost > cap:
                break
            lines.append(line)
            used += cost
        parts.append("检索证据（回答引用用 [n]）：\n" + "\n".join(lines))

    ctx = ModelContext(
        history=history,
        query=query,
        material="\n\n".join(parts) if parts else None,
        plan=plan,
    )
    ctx.tokens_est = (
        sum(_msg_tokens(m) for m in history)
        + estimate_tokens(query)
        + (estimate_tokens(ctx.material) if ctx.material else 0)
    )
    return ctx


def assemble_model_messages(
    ctx: ModelContext, *, system: str | None = None
) -> list[BaseMessage]:
    """固定顺序：①system → ②层2 → ③层1 → ④当前句 → ⑤材料（单条 user）。"""
    msgs: list[BaseMessage] = []
    if system:
        msgs.append(SystemMessage(system))
    msgs.extend(ctx.history)
    msgs.append(HumanMessage(ctx.query))
    if ctx.material:
        msgs.append(HumanMessage(ctx.material))
    return msgs


def _msg_tokens(m: BaseMessage) -> int:
    from app.memory.tokens import count_message_tokens

    return count_message_tokens(m)


def log_context(kind: str, conversation_id: Any, ctx: ModelContext, budget: Any) -> None:
    """可观测：kind ∈ {model_ctx, history_ctx}。摘要全文 + 滑窗逐条 + 条数 + tokens。"""
    p = ctx.plan
    logger.info(
        "%s conv=%s | 条数 L1=%d L2=%d | tokens≈%d（预算 L1=%d L2=%d，总history=%d）| 待摘要批=%d条",
        kind,
        conversation_id,
        len(p.layer1),
        len(p.layer2),
        ctx.tokens_est,
        budget.layer1,
        budget.layer2,
        budget.history,
        len(p.summarize_batch),
    )
    if ctx.material:
        logger.info("%s conv=%s | 摘要与材料全文 ↓\n%s", kind, conversation_id, ctx.material)
    for m in [*p.layer2, *p.layer1]:
        role = _mtype(m)
        text = (m.text or "").replace("\n", "⏎")
        logger.info("%s conv=%s | %s: %s", kind, conversation_id, role, text[:80])
    logger.info("%s conv=%s | 当前句: %s", kind, conversation_id, ctx.query[:80])


def _mtype(m: BaseMessage) -> str:
    t = m.type
    return {"human": "user", "ai": "assistant", "tool": "tool"}.get(t, t)
