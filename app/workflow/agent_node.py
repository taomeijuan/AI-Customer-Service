"""主力 Agent 节点：create_react_agent + 结果提取（tool 帧 + citations + 最终回答）。

策略：react.ainvoke 一次拿到完整结果 → 提取工具帧和 citations → api 层切片补流式。
"""

import json
import logging
from typing import Any

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """你是"商城小助手"，一家电商平台的智能客服，可以调用工具查询实时数据。

职责范围：只回答与电商购物相关的问题（商品、订单、支付、物流、售后）。

行为约束：
1. 语气礼貌、简洁、口语化，单次回复不超过 200 字。
2. 需要订单、商品、物流的实时数据时，调用对应查询工具；工具返回的数据可以如实转述给用户。
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
        for m in state.get("messages") or []:
            if isinstance(m, BaseMessage):
                injected.append(m)
            elif isinstance(m, dict) and m.get("content"):
                injected.append(
                    SystemMessage(m["content"]) if m.get("role") == "system" else HumanMessage(m["content"])
                )

        evidence = list(state.get("evidence", []))  # 知识类路径的检索证据（如果有）

        try:
            result = await react.ainvoke(
                {"messages": injected + [HumanMessage(state["query"])]},
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": settings.agent_max_steps,
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
                        "answer": m.content[:500],
                    })

            # 提取 AIMessage.tool_calls → 推 tool 帧
            for m in msgs:
                for tc in getattr(m, "tool_calls", None) or []:
                    writer({"tool": {"tool": tc["name"], "args": tc.get("args", {}), "status": "done", "ok": True}})

        except Exception as e:
            logger.warning("agent node failed (%s), fallback text", e)
            final_text = "这个问题我这边处理时遇到了一点困难，帮您转人工确认会更稳妥"

        return {
            "final_text": final_text,
            "messages": [],
            "evidence": evidence,
        }

    return agent_node
