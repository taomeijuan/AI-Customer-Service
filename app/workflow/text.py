"""Workflow 共享文本处理：陈旧引用角标剥离（ch06 从 agent_node 抽出共用）。"""

import re

# 陈旧角标：历史 AI 回答里的 [n] 只属于当时的轮次，本轮没有对应证据。
# 不剥掉的话模型会把旧编号直接抄进新答案，前端拿到 citations 对不上 → 死文本。
CITE_RE = re.compile(r"\[\d+\]")


def strip_stale_citations(text: str) -> str:
    return CITE_RE.sub("", text)


def tool_answer_to_text(name: str, raw: str) -> str:
    """工具结果（JSON 串）→ 引用卡片的可读中文文本（ch06 实现，ch07 移到共享模块）。"""
    import json

    try:
        data = json.loads(raw)
    except Exception:
        return raw[:500].replace("{", "（").replace("}", "）") if raw[:1] == "{" else raw[:500]
    if not isinstance(data, dict):
        return str(data)[:500]

    if name == "query_order":
        return (
            f"订单 {data.get('order_no', '')}：状态「{data.get('status', '')}」，"
            f"商品 {data.get('product', '')}，金额 {data.get('amount', '')} 元，"
            f"下单时间 {data.get('created_at', '')}"
        )
    if name == "query_logistics":
        traces = data.get("traces") or []
        lines = "；".join(
            f"{t.get('time', '')} {t.get('desc', '')}"
            for t in traces
            if isinstance(t, dict)
        )
        return (
            f"订单 {data.get('order_no', '')}：承运 {data.get('carrier', '')}，"
            f"运单号 {data.get('tracking_no', '')}。轨迹：{lines}"
        )
    if name == "query_product":
        return (
            f"{data.get('product', '')}：价格 {data.get('price', '')} 元，"
            f"库存 {data.get('stock', '')}，促销：{data.get('promo', '')}"
        )
    parts = []
    for k, v in data.items():
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        parts.append(f"{k}：{v}")
    return "；".join(parts)[:500]
