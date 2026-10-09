"""主力 Agent 节点：create_react_agent + 结果提取（tool 帧 + citations + 最终回答）。

策略：react.ainvoke 一次拿到完整结果 → 提取工具帧和 citations → api 层切片补流式。
"""

import json
import logging
import re
from typing import Any

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.config import get_stream_writer
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

# 业务数据类问题（订单/物流/售后）本轮强制指令：模型在长历史会话里会「抄历史答案
# 跳过工具」，导致 0 工具帧、陈旧 [n] 角标变成死文本。本轮指令是最近一条 system
# 消息，优先级高于骨架 prompt。ch05 验收标准第 1 条：提订单号必须两步都查。
BUSINESS_TURN_PROMPT = (
    "【本轮强制要求】本轮属于订单/物流/售后实时数据查询，必须调用工具获取最新数据后再作答："
    "用户提到订单号时先调用 query_order 查订单状态、再调用 query_logistics 查物流轨迹，两步都做。"
    "即使对话历史里已有相似问题的答案，也必须重新调用工具核实，禁止直接抄历史答案或编造数据。"
)

# 陈旧角标：历史 AI 回答里的 [n] 只属于当时的轮次，本轮没有对应证据。
# 不剥掉的话模型会把旧编号直接抄进新答案，前端拿到 citations 对不上 → 死文本。
_CITE_RE = re.compile(r"\[\d+\]")


def _strip_stale_citations(text: str) -> str:
    return _CITE_RE.sub("", text)


def _tool_answer_to_text(name: str, raw: str) -> str:
    """工具结果（JSON 串）→ 引用卡片的可读中文文本。

    未格式化的工具引用在卡片上是一坨 {'status': ...} JSON——用户不可读。
    已知工具走字段模板，未知工具按 key：value 分行降级。
    """
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


class _ToolFrameHandler(AsyncCallbackHandler):
    """把 ReAct 子图内工具执行实时推成 running/done 帧（与回答 token 流交错）。

    ch05.8 前工具帧在 react.ainvoke 返回后才补推 → 前端先看到完整答案、
    后冒 chips。回调版在工具真正启动/结束的当刻推帧：running → （查询中）
    done，前端 chips 在回答流出前就位。
    """

    def __init__(self, writer) -> None:
        self._writer = writer
        self._stack: list[dict] = []  # react 顺序执行，栈深恒 ≤1

    async def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        name = serialized.get("name", "tool") if isinstance(serialized, dict) else "tool"
        args: dict = {}
        if isinstance(input_str, str) and input_str:
            try:
                parsed = json.loads(input_str)
                if isinstance(parsed, dict):
                    args = parsed
            except Exception:
                pass
        self._stack.append({"name": name, "args": args})
        self._writer({"tool": {"tool": name, "args": args, "status": "running", "ok": True}})

    async def on_tool_end(self, output, **kwargs) -> None:
        meta = self._stack.pop() if self._stack else {"name": "tool", "args": {}}
        self._writer({"tool": {"tool": meta["name"], "args": meta["args"], "status": "done", "ok": True}})

    async def on_tool_error(self, error, **kwargs) -> None:
        meta = self._stack.pop() if self._stack else {"name": "tool", "args": {}}
        self._writer({"tool": {"tool": meta["name"], "args": meta["args"], "status": "done", "ok": False}})

AGENT_SYSTEM_PROMPT = """你是"商城小助手"，一家电商平台的智能客服，可以调用工具查询实时数据。

职责范围：只回答与电商购物相关的问题（商品、订单、支付、物流、售后）。

行为约束：
1. 语气礼貌、简洁、口语化，单次回复不超过 200 字。
2. 需要实时数据时调用对应查询工具：
   - 用户提到具体订单号（如 1001）→ 必须先调用 query_order 查订单状态，再调用 query_logistics 查物流轨迹（两步都做）。
   - 只问商品信息 → 调用 query_product。
   - 工具返回的数据可以如实转述给用户。
   - 调用工具时只输出工具调用本身，不要附加任何说明文字（工具执行完再组织答案）。
3. 对话历史里用户已经说过的信息，直接引用作答，这不属于编造。
4. 回答中引用工具返回的数据时，在句末标注来源 [n]（n 从 1 开始按引用顺序编号）。
5. 涉及退款金额、投诉升级等超出工具能力的问题，建议用户转人工或建工单。
6. 绝不承诺知识条目和工具结果以外的到账时间、价格、库存、赔偿金额。
7. 与电商无关的问题，礼貌说明并引导回购物话题。"""


def build_agent_node(
    llm: Any,
    tools: list,
    settings: Any,
    checkpointer: Any | None = None,
):
    react = create_react_agent(
        llm,
        tools,
        prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
    )

    async def agent_node(state: dict) -> dict:
        from langchain_core.messages import BaseMessage

        try:
            writer = get_stream_writer()
        except Exception:
            writer = lambda data: None

        cid = state.get("conversation_id")
        turn = state.get("turn", 1)
        thread_id = f"{cid}:{turn}"

        injected: list = []
        from langchain.messages import AIMessage

        for m in state.get("messages") or []:
            if isinstance(m, BaseMessage):
                # 剥掉历史 AI 回答的陈旧 [n]（当前轮次的知识注入 SystemMessage 不动）
                if (
                    isinstance(m, AIMessage)
                    and not m.tool_calls
                    and isinstance(m.content, str)
                    and _CITE_RE.search(m.content)
                ):
                    m = m.model_copy(update={"content": _strip_stale_citations(m.content)})
                injected.append(m)
            elif isinstance(m, dict) and m.get("content"):
                injected.append(
                    SystemMessage(m["content"]) if m.get("role") == "system" else HumanMessage(m["content"])
                )

        evidence = list(state.get("evidence", []))  # 知识类路径的检索证据（如果有）

        # 业务数据类：本轮强制工具指令紧跟用户问题上游（长历史防抄写）
        if state.get("intent") in ("物流", "订单", "售后"):
            injected.append(SystemMessage(BUSINESS_TURN_PROMPT))

        try:
            result = await react.ainvoke(
                {"messages": injected + [HumanMessage(state["query"])]},
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": settings.agent_max_steps,
                    "callbacks": [_ToolFrameHandler(writer)],
                },
            )
            msgs = result["messages"]

            # 提取最终回答（最后一条有内容的 AIMessage）
            final_text = ""
            for m in reversed(msgs):
                if isinstance(m, ToolMessage):
                    continue
                if hasattr(m, "tool_calls") and m.tool_calls:
                    continue  # 跳过工具申请消息
                if hasattr(m, "content") and m.content:
                    final_text = m.content
                    break

            # 提取 ToolMessage → 作为 evidence 条目（可点击引用）+ tool 帧
            n = len(evidence)  # 知识类证据编号延续
            for m in msgs:
                if isinstance(m, ToolMessage):
                    n += 1
                    evidence.append({
                        "n": n,
                        "chunk_id": -n,  # 负数 = 工具结果（非知识 chunk）
                        "section_path": f"工具调用/{m.name or 'tool'}",
                        "question": state["query"],
                        "answer": _tool_answer_to_text(m.name or "tool", m.content),
                    })

            # 提取 AIMessage.tool_calls 的产出已由 _ToolFrameHandler 实时推帧，
            # 这里不再补推（避免与回调帧重复）。

        except Exception as e:
            logger.warning("agent node failed (%s), fallback text", e)
            final_text = "这个问题我这边处理时遇到了一点困难，帮您转人工确认会更稳妥"

        return {
            "final_text": final_text,
            "messages": [],
            "evidence": evidence,
        }

    return agent_node
