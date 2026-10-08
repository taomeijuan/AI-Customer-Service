import json
import logging
from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain.messages import ToolMessage
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
    final_payload: dict = {}

    def emit(event: str, data: dict) -> ServerSentEvent:
        """统一出口：每个 SSE 帧落一行日志（serve.sh 终端可见，排查前端帧诊断）"""
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
        ):
            ns, mode, data = _normalize_chunk(chunk)
            if mode == "custom":
                # agent_node get_stream_writer 推流：delta/tool 帧 + log 节点产出
                if "delta" in (data or {}):
                    deltas += 1
                    yield emit("delta", data["delta"])
                if "tool" in (data or {}):
                    yield emit("tool", data["tool"])
                for k in ("final_text", "citations", "options"):
                    if k in (data or {}):
                        final_payload[k] = data[k]
            elif mode == "updates":
                for node_name, update in (data or {}).items():
                    if not isinstance(update, dict):
                        continue
                    if ns == () and node_name == "comfort" and update.get("options"):
                        yield emit("options", {"options": update["options"]})
    except Exception:
        logger.exception("workflow failed, conversation_id=%s", conversation_id)
        yield emit("error", {"message": "服务暂时不可用，请稍后重试"})
        return

    # 固定话术路径（兜底/投诉/闲聊）没有 LLM token：补切片 delta 保逐字效果
    final_text = final_payload.get("final_text", "")
    if deltas == 0 and final_text:
        for i in range(0, len(final_text), 3):
            yield emit("delta", {"text": final_text[i : i + 3]})

    done_data: dict = {"conversation_id": conversation_id}
    if final_payload.get("citations"):
        done_data["citations"] = final_payload["citations"]
    yield emit("done", done_data)
