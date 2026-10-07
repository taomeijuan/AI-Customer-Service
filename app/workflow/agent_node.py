"""主力 Agent 节点：create_react_agent 预构件 + 知识注入 + 步数上限。"""

import logging
from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from app.generation.prompts import GENERATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def build_agent_node(
    llm: Any,
    tools: list,  # 真实 @tool 对象：create_react_agent 负责绑定与执行（ToolNode）
    settings: Any,
    session_factory: Any | None = None,  # 预留：Agent 节点内如需查库（如缺信息追问时补查）
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
        prompt=agent_prompt(),
        checkpointer=checkpointer or InMemorySaver(),
    )

    async def agent_node(state: dict) -> dict:
        cid = state.get("conversation_id")
        turn = state.get("turn", 1)
        thread_id = f"{cid}:{turn}"
        injected = [
            SystemMessage(m["content"]) if m.get("role") == "system" else HumanMessage(m["content"])
            for m in (state.get("messages") or [])
            if m.get("content")
        ]
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


def agent_prompt() -> str:
    return GENERATION_SYSTEM_PROMPT
