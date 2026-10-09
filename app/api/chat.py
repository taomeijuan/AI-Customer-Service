import json
import logging
import os
from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain.messages import AIMessage
from pydantic import BaseModel, Field

from app.memory.trimmer import trim_history_groups
from app.repositories.conversations import ConversationsRepo
from app.repositories.messages import MessagesRepo

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    conversation_id: int | None = None
    message: str = Field(min_length=1)
    user_id: str = Field(min_length=1, max_length=64)


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


def _normalize_chunk(chunk):
    """(已退役) v1 多模式 + subgraphs 的流块归一为 (ns, mode, data)。

    ch05.8 起 astream 统一用 version="v2"（dict 格式：type/ns/data），
    此函数不再被调用，保留以防 v2 环境回退。
    """
    if len(chunk) == 3:
        return chunk
    return (), chunk[0], chunk[1]


@router.post(
    "/api/chat/stream",
    response_class=EventSourceResponse,
    response_model=None,  # 流式生成器不生成 response model（pydantic 升级后必填）
)
async def chat_stream(
    req: ChatRequest,
    request: Request,
    conversation_id: Annotated[int, Depends(get_or_create_session)],
) -> AsyncIterable[ServerSentEvent]:
    """LangGraph 工作流驱动：subgraphs=True 冒出 ReAct 子图 token，custom 承接图内产出。"""
    workflow = request.app.state.workflow
    settings = request.app.state.settings
    session_factory = request.app.state.session_factory

    # 历史从 MySQL 真源加载（分工制：checkpointer 只管轮内）
    async with session_factory() as session:
        history = await MessagesRepo(session).load_history(conversation_id)
    trimmed = trim_history_groups(history, budget_tokens=settings.token_budget)
    turn = len(history) + 1  # 单调递增，奇偶不碰撞
    inputs = {
        "query": req.message,
        "messages": trimmed,
        "conversation_id": conversation_id,
        "turn": turn,
    }
    config = {
        "configurable": {"thread_id": f"{conversation_id}:{turn}"},
        "recursion_limit": settings.agent_max_steps + 4,  # 父图多节点余量
    }

    deltas = 0
    tool_frames = 0
    final_payload: dict = {}
    # 看门缓冲：模型在多步工具循环里可能给工具申请附带前导话术
    # （如 "I'll check your order..."），这类文字不属最终答案。
    # 工具信号（tool_calls/tool_call_chunks）一到即清空缓冲；正常回答文字
    # 最多滞留 HOLD 个 token 后逐段放出（token 级实时性基本无损）。
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
                # 真逐字流：ReAct 子图 token 经 messages 模式实时冒出。
                # langgraph_node=="agent" = 我们的包装节点；工具结果（tools）
                # 与意图识别/改写器的 token（intent/retrieve）天然被滤。
                # 注意 langchain-core>=1.6 类层级已翻转：AIMessageChunk 是
                # AIMessage 子类，此处必须判 AIMessage。
                tok, meta = data
                if meta.get("langgraph_node") != "agent" or not isinstance(tok, AIMessage):
                    continue
                if getattr(tok, "tool_call_chunks", None) or getattr(tok, "tool_calls", None):
                    hold.clear()  # 工具申请出现：此前缓冲全是前导话术，弃
                    continue
                if isinstance(tok.content, str) and tok.content:
                    hold.append(tok.content)
                    if len(hold) > HOLD:
                        deltas += 1
                        yield emit("delta", {"text": hold.pop(0)})
            elif typ == "custom":
                # agent_node get_stream_writer 推流：tool 帧 + log 节点产出
                if isinstance(data, dict):
                    if "tool" in data:
                        tool_frames += 1
                        yield emit("tool", data["tool"])
                    for k in ("final_text", "citations", "options"):
                        if k in data:
                            final_payload[k] = data[k]
            elif typ == "updates":
                if ns == () and isinstance(data, dict):
                    for node_name, update in data.items():
                        if node_name == "agent":
                            # agent 节点完成：看门缓冲里是最终答案的尾巴，放流
                            while hold:
                                deltas += 1
                                yield emit("delta", {"text": hold.pop(0)})
                        elif (
                            isinstance(update, dict)
                            and node_name == "comfort"
                            and update.get("options")
                        ):
                            yield emit("options", {"options": update["options"]})
    except Exception:
        logger.exception("workflow failed, conversation_id=%s", conversation_id)
        yield emit("error", {"message": "服务暂时不可用，请稍后重试"})
        return

    while hold:  # 兜底冲刷：流结束仍有滞留 token
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
