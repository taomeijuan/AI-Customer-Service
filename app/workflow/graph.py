"""LangGraph 工作流骨架：确定性编排，Agent 作为骨架里的核心节点。

流程：resolve(透传) → intent(七类) → route(写死规则)
  ├─ 知识类(商品咨询/退款退货) → retrieve → gate(条件边) → agent → log
  ├─ 业务数据类(物流/订单/售后) → agent → log
  ├─ 投诉 → comfort → log
  └─ 闲聊 → chitchat → log
"""

import logging
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from app.repositories.low_confidence import LowConfidenceRepo

logger = logging.getLogger(__name__)

FALLBACK_INTENT = "订单"  # 意图识别失败兜底=业务数据类（直进 Agent 最通用）

COMFORT_TEXT = (
    "非常抱歉给您带来了不好的体验，您的反馈我已经详细记录。"
    "您可以选择转接人工客服，或提交一个工单，我们都会尽快跟进处理。"
)
CHITCHAT_TEXT = "你好呀～我是商城小助手，商品、订单、物流、售后的问题都可以问我 😊"
FALLBACK_TEXT = "这个问题我这边暂时没有足够的资料，帮您转人工确认"


class WorkflowState(TypedDict, total=False):
    """贯穿全图的会话工作流状态。"""

    query: str
    conversation_id: int | None
    intent: str
    messages: Annotated[list, add_messages]  # 注入 Agent 的消息（知识条目等）
    evidence: list[dict]  # citations 全集快照（ch04 语义）
    options: list[str]  # 投诉路径的可选项
    refusal: bool  # 置信度闸拦截标记
    final_text: str  # 最终回复文本（各路径产出）
    turn: int


def _route_by_intent(state: WorkflowState) -> Literal["retrieve", "agent", "comfort", "chitchat"]:
    """分流规则写死在代码里：七类意图 → 四个出口。"""
    intent = state["intent"]
    if intent in ("商品咨询", "退款退货"):
        return "retrieve"  # 知识类：强制先检索
    if intent in ("物流", "订单", "售后"):
        return "agent"  # 业务数据类：直接进 Agent
    if intent == "投诉":
        return "comfort"
    return "chitchat"  # 闲聊


def _gate_after_retrieve(state: WorkflowState) -> Literal["fallback", "agent"]:
    """置信度闸：检索证据弱 → 兜底话术不进 Agent；够 → 放行。"""
    if state.get("refusal"):
        return "fallback"
    return "agent"


def build_workflow(
    retriever: Any,
    agent_node: Any,  # async callable(state) -> {"final_text", "messages", "evidence"}
    intent_classifier: Any,  # async classify(query) -> IntentName
    session_factory: Any | None,
    checkpointer: Any | None = None,
):
    """组装工作流图。agent_node 由 build_agent_node 产出（create_react_agent 子图包装）。"""

    async def resolve(state: WorkflowState) -> dict:
        return {}  # 指代消解本章透传

    async def intent(state: WorkflowState) -> dict:
        try:
            value = await intent_classifier.classify(state["query"])
        except Exception as e:  # 分类器挂掉不炸图：兜底业务数据类直进 Agent
            logger.warning("intent classifier failed, fallback to 订单: %s", e)
            value = FALLBACK_INTENT
        logger.info("intent=%s query=%s", value, state["query"])
        return {"intent": value}

    async def retrieve(state: WorkflowState) -> dict:
        result = await retriever.retrieve(state["query"], strategy="hybrid_rerank")
        if result.low_confidence:  # 置信度闸的信号源（ch04 语义）
            return {"refusal": True, "evidence": []}
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
        # 知识条目以 [n] 编号消息注入 Agent（要求回答带角标引用）
        knowledge = "\n\n".join(f"[{c['n']}] {c['question']}：{c['answer']}" for c in citations)
        from langchain.messages import SystemMessage

        return {
            "evidence": citations,
            "refusal": False,
            "messages": [SystemMessage(f"已检索到以下相关知识条目，回答时必须用 [n] 角标引用：\n{knowledge}")],
        }

    async def fallback(state: WorkflowState) -> dict:
        # 证据弱：兜底话术 + 记入低置信度池（数据飞轮入口）
        if session_factory is not None:
            async with session_factory() as session:
                await LowConfidenceRepo(session).record(
                    raw_question=state["query"],
                    source="retrieval_low_conf",
                    reason="置信度闸：检索证据不足",
                    conversation_id=state.get("conversation_id"),
                )
        return {"final_text": FALLBACK_TEXT, "refusal": True}

    async def comfort(state: WorkflowState) -> dict:
        return {"final_text": COMFORT_TEXT, "options": ["转人工", "建工单"]}

    async def chitchat(state: WorkflowState) -> dict:
        return {"final_text": CHITCHAT_TEXT}

    async def log(state: WorkflowState) -> dict:
        text = state.get("final_text", "")
        logger.info("workflow done: intent=%s path=%s len=%d", state.get("intent"), "gate-fallback" if state.get("refusal") else state.get("intent"), len(text))
        writer = get_stream_writer()  # custom 流：api 层从这里拿 final_text/citations/options
        writer(
            {
                "final_text": text,
                "citations": state.get("evidence", []),
                "options": state.get("options", []),
            }
        )
        if session_factory is not None and state.get("conversation_id"):
            async with session_factory() as session:
                from langchain.messages import AIMessage, HumanMessage

                from app.repositories.messages import MessagesRepo

                await MessagesRepo(session).append(
                    state["conversation_id"],
                    [HumanMessage(state["query"]), AIMessage(text)],
                )
        return {}

    builder = StateGraph(WorkflowState)
    builder.add_node("resolve", resolve)
    builder.add_node("intent", intent)
    builder.add_node("retrieve", retrieve)
    builder.add_node("fallback", fallback)
    builder.add_node("agent", agent_node)
    builder.add_node("comfort", comfort)
    builder.add_node("chitchat", chitchat)
    builder.add_node("log", log)

    builder.add_edge(START, "resolve")
    builder.add_edge("resolve", "intent")
    builder.add_conditional_edges("intent", _route_by_intent, ["retrieve", "agent", "comfort", "chitchat"])
    # 置信度闸 = 检索节点出边上的条件函数（非独立节点）
    builder.add_conditional_edges("retrieve", _gate_after_retrieve, ["fallback", "agent"])
    builder.add_edge("fallback", "log")
    builder.add_edge("agent", "log")
    builder.add_edge("comfort", "log")
    builder.add_edge("chitchat", "log")
    builder.add_edge("log", END)

    # checkpointer 由调用方（api 层）传入：InMemorySaver + thread_id={cid}:{turn}
    return builder.compile(checkpointer=checkpointer) if checkpointer is not None else builder.compile()
