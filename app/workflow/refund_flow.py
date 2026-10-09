"""ch06 退款确定性子流程节点（退款退货/售后 intents）。

流程：提订单号（白名单校验，不猜不编）→ 缺则 interrupt 弹订单选择器
（用户点选后 Command(resume) 恢复，本节点从头重跑、interrupt 直接返回点选值）→
拿订单数据 → Query 扩写 + retrieve_multi 强制检索政策条款 → 证据弱走置信度闸
→ 证据+订单数据注入主力 Agent 判「这一单能不能退」。
"""

import json
import logging
import re
from typing import Any

from langchain.messages import SystemMessage
from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from app.tools.ecommerce import order_data, orders_summary
from app.workflow.agent_node import _tool_answer_to_text

logger = logging.getLogger(__name__)

# 提单号候选：3-6 位数字是线索，但必须过白名单（评审 M3）——
# 「2026年买的」「1999元能退吗」里的数字提不到有效单号就交给选择器，绝不喂假数据生成器
ORDER_NO_RE = re.compile(r"(?<!\d)(\d{3,6})(?!\d)")

REFUND_INJECT_PROMPT = (
    "用户想办理退款/售后。请结合下面的订单数据和政策条款，判断这一单能不能退、"
    "怎么退，给出简短办理指引（不超过 200 字）。引用政策时用 [n] 角标。"
    "若政策明确支持，回复末尾提示用户可以点「申请退款」按钮提交退款单。"
)


def _known_order_nos() -> set[str]:
    return {o["order_no"] for o in orders_summary()}


def build_refund_prep(
    retriever: Any,
    expander: Any,  # async expand(query, context) -> list[str]
):
    async def refund_prep(state: dict) -> dict:
        try:
            writer = get_stream_writer()
        except Exception:
            writer = lambda data: None  # noqa: E731

        query = state["query"]

        # ① 订单号来源（用户拍板规则「问句直通、动作要确认」）：
        #   a) 用户原话里的数字（正则）——点名了就直接办
        #   b) 消解器槽位 ctx_order_no——仅当它是「追问刚才讨论的那个订单」时
        #      才会非空；发起动作未点名时消解器留空 → 走到这里没号 → 弹选择器
        # 两路都过白名单：历史没出现/编造的号（「1999 元那单」）一律无效。
        known = _known_order_nos()
        order_no = ""
        raw_text = state.get("raw_query") or query
        m = ORDER_NO_RE.search(raw_text)
        if m and m.group(1) in known:
            order_no = m.group(1)
        if not order_no:
            ctx = str(state.get("ctx_order_no") or "").strip()
            if ctx in known and ctx:
                order_no = ctx
        if not order_no:
            choice = interrupt(
                {"type": "order_selector", "orders": orders_summary(), "question": query}
            )
            picked = str((choice or {}).get("order_no") or "")
            if picked not in known:  # 评审 M3：resume 回传同样过白名单，非法值按取消处理
                return {
                    "final_text": "没有选订单的话，我先不继续了；想退的时候再叫我～",
                    "evidence": [],
                    "refusal": False,
                    "order_no": "",
                }
            order_no = picked

        # ② 拿订单数据（确定性假数据世界，直接调纯实现）
        order = order_data(order_no)
        order_text = _tool_answer_to_text("query_order", json.dumps(order, ensure_ascii=False))
        writer({"tool": {"tool": "query_order", "args": {"order_no": order_no}, "status": "done", "ok": True}})

        # ③ 扩写 + 多路检索，强制命中政策条款
        queries = await expander.expand(query, context=order_text)
        result = await retriever.retrieve_multi(queries, strategy="hybrid_rerank")
        if result.low_confidence:
            # 证据弱：交回退节点（记池 + 兜底话术都在那里，避免双重入池）
            return {"refusal": True, "evidence": [], "order_no": order_no}

        citations = [
            {
                "n": n,
                "chunk_id": ev.chunk_id,
                "section_path": ev.section_path,
                "question": ev.question,
                "answer": ev.answer,
            }
            for n, ev in enumerate(result.evidences, start=1)
        ]
        knowledge = "\n\n".join(f"[{c['n']}] {c['question']}：{c['answer']}" for c in citations)

        return {
            "order_no": order_no,
            "evidence": citations,
            "refusal": False,
            # m4：options 帧带订单摘要，前端表单不再依赖全局 lastOrder
            "options": ["申请退款"],
            "order_brief": {"order_no": order_no, "product": order["product"], "amount": order["amount"]},
            "messages": [
                SystemMessage(
                    f"已检索到以下政策条款，回答时必须用 [n] 角标引用：\n{knowledge}\n\n"
                    f"用户订单数据：{order_text}\n\n{REFUND_INJECT_PROMPT}"
                )
            ],
        }

    return refund_prep
