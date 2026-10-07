"""主力 Agent 节点：create_react_agent 预构件 + 知识注入 + 步数上限。"""

import logging
from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)


def build_agent_node(
    llm: Any,
    tools: list,  # 真实 @tool 对象：create_react_agent 负责绑定与执行（ToolNode）
    settings: Any,
    checkpointer: Any | None = None,
):
    """构建主图的 agent 节点（create_react_agent 编译子图 + 状态转换包装）。

    - prompt：客服 System Prompt + 负面知识（复用 ch04）
    - recursion_limit：settings.agent_max_steps（步数/消耗控制的最简实现）
    - checkpointer：InMemorySaver，thread_id={cid}:{turn}（分工制：只管轮内状态）
    """
    react = create_react_agent(
        llm,
        tools,
        prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
    )

    async def agent_node(state: dict) -> dict:
        from langchain_core.messages import BaseMessage

        cid = state.get("conversation_id")
        turn = state.get("turn", 1)
        thread_id = f"{cid}:{turn}"
        injected: list = []
        for m in state.get("messages") or []:
            if isinstance(m, BaseMessage):
                injected.append(m)  # 已是 LC 消息（历史/知识注入）
            elif isinstance(m, dict) and m.get("content"):
                injected.append(
                    SystemMessage(m["content"]) if m.get("role") == "system" else HumanMessage(m["content"])
                )
        try:
            result = await react.ainvoke(
                {"messages": injected + [HumanMessage(state["query"])]},
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": settings.agent_max_steps,
                },
            )
            final = result["messages"][-1].content
        except Exception as e:  # GraphRecursionError 等异常：兜底文本不炸会话
            logger.warning("agent node failed (%s), fallback text", e)
            final = "这个问题我这边处理时遇到了一点困难，帮您转人工确认会更稳妥"
        return {
            "final_text": final,
            "messages": [],
            "evidence": state.get("evidence", []),
        }

    return agent_node


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
