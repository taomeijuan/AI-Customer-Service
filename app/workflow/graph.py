"""LangGraph 工作流骨架：确定性编排，Agent 作为骨架里的核心节点。

流程：resolve(透传) → intent(七类) → route(写死规则)
  ├─ 知识类(商品咨询/退款退货) → retrieve → gate(条件边) → agent → log
  ├─ 业务数据类(物流/订单/售后) → agent → log
  ├─ 投诉 → comfort → log
  └─ 闲聊 → chitchat → log
"""

import logging
import re
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated

from app.repositories.low_confidence import LowConfidenceRepo
from app.workflow.intent import INTENT_LOW_CONF_THRESHOLD

logger = logging.getLogger(__name__)

FALLBACK_INTENT = "订单"  # 意图识别失败兜底=业务数据类（直进 Agent 最通用）

# ch07.1（用户拍板：代码硬规则进意图节点，不让模型猜）：退款退货/售后类里，
# 只有「发起办理」句才进订单子流程；问通用规则（「退货政策是什么」）走知识检索。
# 办理词表命中或点名真实订单号 → 办理；否则视为规则咨询。
PROCESS_HINT_RE = re.compile("(我要|帮我|给我|我想|申请|办理|退掉|退了吧|赶紧|立刻|马上)")
ORDER_MENTION_RE = re.compile(r"(?<!\d)100[1-5](?!\d)")


def wants_refund_process(raw_query: str) -> bool:
    return bool(PROCESS_HINT_RE.search(raw_query or "")) or bool(ORDER_MENTION_RE.search(raw_query or ""))

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
    raw_query: str  # 用户原话（resolve 改写前的真源，落库用）
    action: bool  # ch07.1 意图附带的「发起办理」标志（退款退货/售后路由子流程与否）
    order_no: str  # ch06 退款子流程选定的订单号
    order_brief: dict  # ch06 退款子流程订单摘要（options 帧随发到前端）
    order_text: str  # ch07 退款子流程订单可读文本（进 material 注入）
    order_instructions: str  # ch07 退款判定指令（refund_prep 产出，agent 装配）
    ctx_history: list  # ch07 装配好的精简历史（层2截短+层1原文），节点喂模型用这个
    ctx_projection: str  # ch07 梗概投影文本（api 层装配进 state）
    ctx_order_no: str  # ch06 消解器判定的订单槽位（问句直通；动作留空弹选择器）


def _make_router(has_refund_prep: bool):
    """分流规则写死在代码里：八类意图 → 五出口（ch06：其他→Agent 兜底）。

    退款子流程未注入时，退款退货/售后回退到检索路径（兼容旧接线/旧测试）。
    """

    def _route_by_intent(state: WorkflowState) -> Literal["retrieve", "refund_prep", "agent", "comfort", "chitchat"]:
        intent = state["intent"]
        if intent in ("商品咨询",):
            return "retrieve"  # 知识类：强制先检索
        if intent in ("退款退货", "售后"):
            # ch07.1 修复「退货政策是什么」被选择器拦截：只有发起办理才进子流程，
            # 问通用规则仍走知识检索（用户心智：规则问题要答案，不是选订单）
            if state.get("action") and has_refund_prep:
                return "refund_prep"
            return "retrieve"
        if intent in ("物流", "订单", "其他"):
            return "agent"  # 业务数据类直接进 Agent；其他=拿不准也交给 Agent 澄清
        if intent == "投诉":
            return "comfort"
        return "chitchat"  # 闲聊

    return _route_by_intent


def _gate_after_retrieve(state: WorkflowState) -> Literal["fallback", "agent"]:
    """置信度闸：检索证据弱 → 兜底话术不进 Agent；够 → 放行。"""
    if state.get("refusal"):
        return "fallback"
    return "agent"


def _gate_after_refund(state: WorkflowState) -> Literal["fallback", "agent", "log"]:
    """退款子流程出口：证据弱→fallback（记池+兜底话术）；用户取消点选→log；
    正常拿到证据+订单→agent 判「这一单能不能退」。"""
    if state.get("refusal"):
        return "fallback"
    if state.get("final_text"):  # 子流程已给出结束语（如未选订单取消）
        return "log"
    return "agent"


def build_workflow(
    retriever: Any,
    agent_node: Any,  # async callable(state) -> {"final_text", "messages", "evidence"}
    intent_classifier: Any,  # async classify(query) -> IntentName
    session_factory: Any | None,
    resolver: Any | None = None,  # ch06：async resolve(query, history) -> str；None=透传
    refund_prep: Any | None = None,  # ch06：退款确定性子流程节点；None=退款走检索路径
    checkpointer: Any | None = None,
):
    """组装工作流图。agent_node 由 build_agent_node 产出（create_react_agent 子图包装）。"""

    async def resolve(state: WorkflowState) -> dict:
        """ch06 正式版：LLM 指代消解+改写+订单槽位；未注入/失败均透传。"""
        raw = state["query"]
        if resolver is None:
            return {"raw_query": raw, "ctx_order_no": ""}
        history_src = state.get("ctx_history")
        if history_src is None:
            history_src = state.get("messages") or []  # 兼容图直调/旧接线
        try:
            outcome = await resolver.resolve_detail(raw, list(history_src))
        except Exception as e:  # 双保险：Resolver 内部已兜底，图内再兜一层
            logger.warning("resolve node failed, pass-through: %s", e)
            return {"query": raw, "raw_query": raw, "ctx_order_no": ""}
        return {"query": outcome.query, "raw_query": raw, "ctx_order_no": outcome.order_no}

    async def intent(state: WorkflowState) -> dict:
        # ch06：classify_detail 拿置信度；分类器自带兜底，这里再兜一层防接口异常
        try:
            outcome = await intent_classifier.classify_detail(state["query"])
            value, confidence = outcome.intent, outcome.confidence
        except Exception as e:
            logger.warning("intent classifier failed, fallback to 订单: %s", e)
            value, confidence = FALLBACK_INTENT, 0.0
        # ch07.1（用户拍板：不让模型猜）：办理标志由代码规则对**用户原话**判定；
        # 消解器确认在追问具体订单（槽位非空）也算发起（ch06 问句直通语义）
        action = value in ("退款退货", "售后") and (
            wants_refund_process(state.get("raw_query") or state["query"])
            or bool((state.get("ctx_order_no") or "").strip())
        )
        logger.info("intent=%s confidence=%.2f action=%s query=%s", value, confidence, action, state["query"])
        if confidence < INTENT_LOW_CONF_THRESHOLD and session_factory is not None and state.get("conversation_id"):
            # 数据飞轮入口：拿不准的问题进池，ch09 用用户反馈校准；入池失败不炸会话
            try:
                async with session_factory() as session:
                    await LowConfidenceRepo(session).record(
                        raw_question=state["query"],
                        source="intent_low_conf",
                        reason=f"意图置信度低：{value}({confidence:.2f})",
                        conversation_id=state.get("conversation_id"),
                    )
            except Exception as e:
                logger.warning("intent low-conf pool record failed: %s", e)
        return {"intent": value, "action": action}

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
        # ch07：不再注入 SystemMessage——证据经 evidence 通道流动，
        # 装配层把「梗概投影+检索证据」合成一条 user 挂当前句之后（防 system 上提）
        return {
            "evidence": citations,
            "refusal": False,
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
                    [HumanMessage(state.get("raw_query") or state["query"]), AIMessage(text)],
                    citations=state.get("evidence") or None,
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
    router = _make_router(refund_prep is not None)
    router_paths = ["retrieve", "agent", "comfort", "chitchat"]
    if refund_prep is not None:
        router_paths.insert(1, "refund_prep")
        builder.add_node("refund_prep", refund_prep)
        # 退款闸：弱证据→fallback / 用户取消点选→log / 正常→agent
        builder.add_conditional_edges("refund_prep", _gate_after_refund, ["fallback", "agent", "log"])

    builder.add_edge(START, "resolve")
    builder.add_edge("resolve", "intent")
    builder.add_conditional_edges("intent", router, router_paths)
    # 置信度闸 = 检索节点出边上的条件函数（非独立节点）
    builder.add_conditional_edges("retrieve", _gate_after_retrieve, ["fallback", "agent"])
    builder.add_edge("fallback", "log")
    builder.add_edge("agent", "log")
    builder.add_edge("comfort", "log")
    builder.add_edge("chitchat", "log")
    builder.add_edge("log", END)

    # checkpointer 由调用方（api 层）传入：InMemorySaver + thread_id={cid}:{turn}
    return builder.compile(checkpointer=checkpointer) if checkpointer is not None else builder.compile()
