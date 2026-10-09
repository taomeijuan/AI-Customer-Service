"""聊天 SSE 端点（ch07：会话级线程 + 三层上下文装配 + 后台摘要调度）。

每轮编排（新消息）：
  挂起中断先静默取消（防旧选择器劫持新语义）→ 读锚点/投影/历史 →
  build_context 出精简史 → history_ctx 日志 → 层1锚只前进持久化 →
  层2溢出触发后台摘要（schedule，不阻塞）→ wipe+resync State 输入。
恢复（resume）：Command(resume) 直进会话线程（存在性校验在 Depends）。
"""

import json
import logging
import os
from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command
from pydantic import BaseModel, Field, model_validator

from app.db.models import Conversation
from app.memory.context_builder import ContextAnchors, build_context, log_context
from app.repositories.conversations import ConversationsRepo
from app.repositories.messages import MessagesRepo

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    conversation_id: int | None = None
    message: str | None = Field(default=None, min_length=1)
    user_id: str = Field(min_length=1, max_length=64)
    resume: dict | None = None  # ch06：订单选择器点选回填 {"order_no": "1001"}

    @model_validator(mode="after")
    def _check_message_xor_resume(self):
        """新消息与恢复流程二选一；恢复必须带会话 id。"""
        if bool(self.message) == bool(self.resume):
            raise ValueError("message 与 resume 必须二选一")
        if self.resume is not None and self.conversation_id is None:
            raise ValueError("resume 必须携带 conversation_id")
        return self


async def get_or_create_session(
    req: ChatRequest, request: Request
) -> int:
    """依赖阶段确定会话：不存在抛 404；不带 id 则新建。"""
    async with request.app.state.session_factory() as session:
        try:
            return await ConversationsRepo(session).ensure_conversation(
                req.user_id, req.conversation_id
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="会话不存在或已过期")


def _thread_config(request: Request, conversation_id: int) -> dict:
    """ch07 会话级线程：一个会话一个 checkpoint 线程，State 贯穿全话。"""
    settings = request.app.state.settings
    return {
        "configurable": {"thread_id": f"ctx-{conversation_id}"},
        "recursion_limit": settings.max_agent_steps + 4,
    }


async def _pending_interrupt(request: Request, conversation_id: int) -> bool:
    workflow = request.app.state.workflow
    try:
        snap = await workflow.aget_state(_thread_config(request, conversation_id))
    except Exception:
        return False
    return bool(snap and snap.next)


async def check_resume_pending(
    req: ChatRequest, request: Request, conversation_id: Annotated[int, Depends(get_or_create_session)]
) -> None:
    """resume 必须有挂起中断（状态即注册表，ch06 的 pending_resumes dict 退役）。"""
    if req.resume is not None and not await _pending_interrupt(request, conversation_id):
        raise HTTPException(status_code=409, detail="该会话没有待恢复的流程，请直接发消息")


@router.post(
    "/api/chat/stream",
    response_class=EventSourceResponse,
    response_model=None,  # 流式生成器不生成 response model（pydantic 升级后必填）
)
async def chat_stream(
    req: ChatRequest,
    request: Request,
    conversation_id: Annotated[int, Depends(get_or_create_session)],
    _resume_ok: Annotated[None, Depends(check_resume_pending)],
) -> AsyncIterable[ServerSentEvent]:
    """LangGraph 工作流驱动：subgraphs=True 冒出 ReAct 子图 token，custom 承接图内产出。"""
    workflow = request.app.state.workflow
    settings = request.app.state.settings
    session_factory = request.app.state.session_factory
    budget = request.app.state.context_budget
    config = _thread_config(request, conversation_id)

    if req.resume is not None:
        inputs: object = Command(resume=req.resume)
    else:
        # ① 有挂起中断=用户放弃了点选 → 静默走完取消分支再处理新消息
        if await _pending_interrupt(request, conversation_id):
            try:
                await workflow.ainvoke(Command(resume={"order_no": ""}), config)
            except Exception:
                logger.exception("abandoned-interrupt cancel failed, cid=%s", conversation_id)

        # ② 三层上下文装配（DB 是历史真源；锚点/投影随会话行取）
        async with session_factory() as session:
            conv = await session.get(Conversation, conversation_id)
            pairs = await MessagesRepo(session).load_history_with_ids(conversation_id)
        projection = (conv.summary or "") if conv else ""
        anchors = ContextAnchors(
            summary_upto=(conv.summary_upto_msg_id or 0) if conv else 0,
            layer1_from=(conv.layer1_from_msg_id if conv else None),
        )
        ctx = build_context(
            pairs,
            anchors=anchors,
            budget=budget,
            settings=settings,
            query=req.message or "",
            summary_projection=projection or None,
        )
        log_context("history_ctx", conversation_id, ctx, budget)

        # ③ 层1锚只前进持久化（降级=挪锚不搬数据）
        if ctx.plan.new_layer1_from and (conv.layer1_from_msg_id or 0) < ctx.plan.new_layer1_from:
            async with session_factory() as session:
                row = await session.get(Conversation, conversation_id)
                row.layer1_from_msg_id = ctx.plan.new_layer1_from
                await session.commit()

        # ④ 层2溢出 → 后台摘要任务（schedule 同步返回，不阻塞本轮）
        if ctx.plan.summarize_batch:
            logger.info(
                "summary trigger: conv=%s 层2约%d token>预算%d，溢出%d条待压",
                conversation_id, ctx.plan.layer2_tokens, budget.layer2, len(ctx.plan.summarize_batch),
            )
            request.app.state.summarizer.schedule(
                conversation_id, ctx.plan.summarize_batch, projection or None
            )

        # ⑤ State wipe+resync：DB 全史 + 本轮输入（add_messages 并入 agent 回吐的新消息）
        inputs = {
            "query": req.message,
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *(m for _, m in pairs),
                HumanMessage(req.message),
            ],
            "ctx_history": ctx.history,
            "ctx_projection": projection,
            "conversation_id": conversation_id,
        }

    deltas = 0
    tool_frames = 0
    final_payload: dict = {}
    # 看门缓冲：工具申请附带的正文是前导话术，信号一到即弃
    hold: list[str] = []
    HOLD = 12

    def emit(event: str, data: dict) -> ServerSentEvent:
        """统一出口。默认静默（终端不刷帧）；SSE_DEBUG=1 时逐帧落日志排查。"""
        if os.getenv("SSE_DEBUG") == "1":
            preview = json.dumps(data, ensure_ascii=False)[:160]
            logger.info("[sse] cid=%s event=%s data=%s", conversation_id, event, preview)
        return ServerSentEvent(event=event, data=data)

    yield emit("meta", {"conversation_id": conversation_id})
    try:
        async for chunk in workflow.astream(
            inputs,
            config=config,
            stream_mode=["messages", "updates", "custom"],
            subgraphs=True,
            version="v2",  # 统一 dict 格式 {type, ns, data}（langgraph>=1.1）
        ):
            typ = chunk["type"]
            ns = chunk["ns"]
            data = chunk["data"]
            if typ == "messages":
                # 真逐字流：只透传 agent 包装节点的回答 token。
                # langchain-core>=1.6 类层级翻转：AIMessageChunk 是 AIMessage
                # 子类，此处必须判 AIMessage。
                tok, meta = data
                if meta.get("langgraph_node") != "agent" or not isinstance(tok, AIMessage):
                    continue
                if getattr(tok, "tool_call_chunks", None) or getattr(tok, "tool_calls", None):
                    hold.clear()
                    continue
                if isinstance(tok.content, str) and tok.content:
                    hold.append(tok.content)
                    if len(hold) > HOLD:
                        deltas += 1
                        yield emit("delta", {"text": hold.pop(0)})
            elif typ == "custom":
                if isinstance(data, dict):
                    if "tool" in data:
                        tool_frames += 1
                        yield emit("tool", data["tool"])
                    for k in ("final_text", "citations", "options"):
                        if k in data:
                            final_payload[k] = data[k]
            elif typ == "updates":
                if ns == () and isinstance(data, dict):
                    intr = data.get("__interrupt__")  # 槽位中断（探针实证形态）
                    for item in intr or ():
                        value = getattr(item, "value", None)
                        if isinstance(value, dict) and value.get("type") == "order_selector":
                            yield emit(
                                "order_selector",
                                {"orders": value.get("orders", []), "question": value.get("question", "")},
                            )
                    for node_name, update in data.items():
                        if node_name == "__interrupt__":
                            continue
                        if node_name == "agent":
                            while hold:
                                deltas += 1
                                yield emit("delta", {"text": hold.pop(0)})
                        elif isinstance(update, dict) and update.get("options"):
                            frame: dict = {"options": update["options"]}
                            if update.get("order_brief"):
                                frame["order"] = update["order_brief"]
                            yield emit("options", frame)
    except Exception:
        logger.exception("workflow failed, conversation_id=%s", conversation_id)
        yield emit("error", {"message": "服务暂时不可用，请稍后重试"})
        return

    while hold:
        deltas += 1
        yield emit("delta", {"text": hold.pop(0)})

    # 固定话术路径（兜底/投诉/闲聊）没有 LLM token：补切片 delta 保逐字效果
    final_text = final_payload.get("final_text", "")
    if deltas == 0 and final_text:
        for i in range(0, len(final_text), 3):
            yield emit("delta", {"text": final_text[i : i + 3]})

    done_data: dict = {"conversation_id": conversation_id}
    if final_payload.get("citations"):
        done_data["citations"] = final_payload["citations"]
    yield emit("done", done_data)

    logger.info(
        "stream done: cid=%s deltas=%d tool_frames=%d citations=%d",
        conversation_id, deltas, tool_frames, len(final_payload.get("citations", [])),
    )
