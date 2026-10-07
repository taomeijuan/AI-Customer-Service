"""主力 Agent 节点：create_react_agent 预构件 + get_stream_writer 实时推流。

与 ch04 的 yield 模式对齐：工具调用和 LLM token 通过 custom 流实时推送，
而不是等子图跑完后由父图被动检测。
"""

import logging
from typing import Any

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from app.generation.prompts import GENERATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """你是"商城小助手"，一家电商平台的智能客服，可以调用工具查询实时数据。

职责范围：只回答与电商购物相关的问题（商品、订单、支付、物流、售后）。

行为约束：
1. 语气礼貌、简洁、口语化，单次回复不超过 200 字。
2. 需要订单、商品、物流的实时数据时，调用对应查询工具；工具返回的数据可以如实转述给用户。
3. 对话历史里用户已经说过的信息，直接引用作答，这不属于编造。
4. 如果知识条目（[n] 编号）被注入到对话中，回答时用 [n] 角标引用来源。
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
            writer = lambda data: None  # 图外调用（测试直连）时 no-op
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

        final_text = ""
        try:
            # astream 子图：messages 模式拿 LLM token，updates 模式拿工具调用
            async for mode, chunk in react.astream(
                {"messages": injected + [HumanMessage(state["query"])]},
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": settings.agent_max_steps,
                },
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    msg_chunk, meta = chunk
                    # ToolMessage 是工具执行结果，不算 LLM token
                    if isinstance(msg_chunk, ToolMessage):
                        continue
                    # LLM token → delta 推给前端
                    if getattr(msg_chunk, "content", ""):
                        writer({"delta": {"text": msg_chunk.content}})
                        final_text += msg_chunk.content
                    # 工具申请 → running 帧
                    if getattr(msg_chunk, "tool_call_chunks", None):
                        for tcc in msg_chunk.tool_call_chunks:
                            name = tcc.get("name") if isinstance(tcc, dict) else getattr(tcc, "name", None)
                            if name:
                                writer({"tool": {"tool": name, "args": {}, "status": "running"}})
                elif mode == "updates":
                    for node_name, update in chunk.items():
                        if node_name == "tools" and isinstance(update, dict) and update.get("messages"):
                            for tm in update["messages"]:
                                if isinstance(tm, ToolMessage):
                                    writer({"tool": {"tool": tm.name or "", "args": {}, "status": "done", "ok": tm.status != "error"}})
        except Exception as e:
            logger.warning("agent node failed (%s), fallback text", e)
            final_text = "这个问题我这边处理时遇到了一点困难，帮您转人工确认会更稳妥"

        return {
            "final_text": final_text,
            "messages": [],
            "evidence": state.get("evidence", []),
        }

    return agent_node
