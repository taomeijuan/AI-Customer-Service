import logging
from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain.messages import ToolMessage
from pydantic import BaseModel, Field

from app.memory.trimmer import trim_history
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
    """多模式 + subgraphs 的流块归一为 (ns, mode, data)。"""
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
    trimmed = trim_history(history, budget_tokens=settings.token_budget)
    turn = len(history) // 2 + 1
    inputs = {
        "query": req.message,
        "messages": trimmed,
        "conversation_id": conversation_id,
        "turn": turn,
    }
    config = {
        "configurable": {"thread_id": f"{conversation_id}:{turn}"},
        "recursion_limit": settings.agent_max_steps,
    }

    deltas = 0
    final_payload: dict = {}
    yield ServerSentEvent(event="meta", data={"conversation_id": conversation_id})
    try:
        async for chunk in workflow.astream(
            inputs,
            config=config,
            stream_mode=["messages", "updates", "custom"],
            subgraphs=True,
        ):
            ns, mode, data = _normalize_chunk(chunk)
            if mode == "messages":
                msg_chunk, meta = data
                # 只透传 ReAct 子图 LLM 节点的 token（意图识别的 LLM 不上屏）
                if meta.get("langgraph_node") == "model" and getattr(msg_chunk, "content", ""):
                    deltas += 1
                    yield ServerSentEvent(event="delta", data={"text": msg_chunk.content})
            elif mode == "updates":
                for node_name, update in (data or {}).items():
                    if not isinstance(update, dict):
                        continue
                    if node_name == "model" and update.get("messages"):
                        ai = update["messages"][-1]
                        for tc in getattr(ai, "tool_calls", None) or []:
                            yield ServerSentEvent(
                                event="tool",
                                data={"tool": tc["name"], "args": tc["args"], "status": "running"},
                            )
                    elif node_name == "tools" and update.get("messages"):
                        for tm in update["messages"]:
                            if isinstance(tm, ToolMessage):
                                yield ServerSentEvent(
                                    event="tool",
                                    data={"tool": tm.name, "args": {}, "status": "done", "ok": tm.status != "error"},
                                )
                    elif ns == () and node_name == "comfort" and update.get("options"):
                        yield ServerSentEvent(event="options", data={"options": update["options"]})
            elif mode == "custom":
                final_payload.update(data or {})
    except Exception:
        logger.exception("workflow failed, conversation_id=%s", conversation_id)
        yield ServerSentEvent(event="error", data={"message": "服务暂时不可用，请稍后重试"})
        return

    # 固定话术路径（兜底/投诉/闲聊）没有 LLM token：补切片 delta 保逐字效果
    final_text = final_payload.get("final_text", "")
    if deltas == 0 and final_text:
        for i in range(0, len(final_text), 3):
            yield ServerSentEvent(event="delta", data={"text": final_text[i : i + 3]})

    done_data: dict = {"conversation_id": conversation_id}
    if final_payload.get("citations"):
        done_data["citations"] = final_payload["citations"]
    yield ServerSentEvent(event="done", data=done_data)
