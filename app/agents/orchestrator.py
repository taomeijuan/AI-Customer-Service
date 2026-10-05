import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain.messages import AIMessage, HumanMessage, ToolMessage

from app.memory.trimmer import trim_history
from app.repositories.conversations import ConversationsRepo
from app.repositories.messages import MessagesRepo

logger = logging.getLogger(__name__)


def _tcc_get(item: Any, key: str) -> Any:
    """tool_call_chunk 元素兼容 dict / 对象两种形态。"""
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _aggregate_tool_calls(chunks: list) -> list[dict]:
    """把流式 tool_call_chunks 按序聚合成完整 [{name, args(dict), id}]。"""
    merged: dict[int, dict] = {}
    for chunk in chunks:
        for item in chunk:
            idx = _tcc_get(item, "index") or 0
            slot = merged.setdefault(idx, {"name": "", "args": "", "id": ""})
            if _tcc_get(item, "name"):
                slot["name"] = _tcc_get(item, "name")
            if _tcc_get(item, "id"):
                slot["id"] = _tcc_get(item, "id")
            args_piece = _tcc_get(item, "args") or ""
            slot["args"] += args_piece
    out = []
    for _, slot in sorted(merged.items()):
        try:
            args = json.loads(slot["args"]) if slot["args"] else {}
        except json.JSONDecodeError:
            args = {}
        out.append({"name": slot["name"], "args": args, "id": slot["id"]})
    return out


class Orchestrator:
    """单轮编排：DB 真源 + bind_tools + 注册表执行 + 回灌流式收敛。

    run() 产出 (event, data) 元组流：meta → [tool*] → delta* → done；
    异常以 ("error", {...}) 收尾，不抛出；未知会话 KeyError 由 api 层转 404。
    ch03 升级 Agent Loop：把「第二次调用」换回循环即可。
    """

    def __init__(
        self,
        model: Any,
        registry_factory: Any,
        session_factory: Any,
        settings: Any,
    ) -> None:
        self._model = model
        self._registry_factory = registry_factory
        self._session_factory = session_factory
        self._settings = settings

    async def run(
        self, user_id: str, conversation_id: int | None, message: str
    ) -> AsyncIterator[tuple[str, dict]]:
        async with self._session_factory() as session:
            conv_repo = ConversationsRepo(session)
            msg_repo = MessagesRepo(session)
            registry = self._registry_factory(session)

            cid = await conv_repo.ensure_conversation(user_id, conversation_id)
            yield "meta", {"conversation_id": cid}

            await msg_repo.append(cid, [HumanMessage(message)])
            history = await msg_repo.load_history(cid)
            messages = trim_history(history, budget_tokens=self._settings.token_budget)

            try:
                # ── 第一次调用：绑工具，边流边判是否走工具模式 ──
                bound = self._model.bind_tools(registry.all())
                text_pieces: list[str] = []
                tool_chunks: list = []
                tool_mode = False
                async for chunk in bound.astream(messages):
                    tcc = getattr(chunk, "tool_call_chunks", None)
                    if tcc:
                        tool_mode = True  # 判定为工具模式后不再放行 delta
                        tool_chunks.append(tcc)
                    elif not tool_mode:
                        piece = getattr(chunk, "text", "")
                        if piece:
                            text_pieces.append(piece)
                            yield "delta", {"text": piece}

                if tool_mode:
                    tool_calls = _aggregate_tool_calls(tool_chunks)
                    # ── 工具模式：执行 → 状态帧 → 落库 → 回灌 ──
                    results: list[dict] = []
                    for tc in tool_calls:
                        yield "tool", {
                            "tool": tc["name"],
                            "args": tc["args"],
                            "status": "running",
                        }
                        outcome = await registry.execute(tc["name"], tc["args"])
                        yield "tool", {
                            "tool": tc["name"],
                            "args": tc["args"],
                            "status": "done",
                            "ok": outcome["ok"],
                            "summary": (
                                "完成" if outcome["ok"] else f"失败: {outcome['error']}"
                            ),
                        }
                        results.append(
                            {
                                "id": tc["id"],
                                "name": tc["name"],
                                "content": json.dumps(outcome, ensure_ascii=False),
                            }
                        )
                    await msg_repo.append_tool_round(cid, tool_calls, results)
                    feed_messages = messages + [
                        AIMessage(content="", tool_calls=tool_calls)
                    ] + [
                        ToolMessage(content=r["content"], tool_call_id=r["id"], name=r["name"])
                        for r in results
                    ]
                    # ── 第二次调用：不绑工具，流式收敛 ──
                    final: list[str] = []
                    async for chunk in self._model.astream(feed_messages):
                        piece = getattr(chunk, "text", "")
                        if piece:
                            final.append(piece)
                            yield "delta", {"text": piece}
                    await msg_repo.append(cid, [AIMessage("".join(final))])
                else:
                    # ── 普通回答（ch01 老路）：完整回复落库 ──
                    await msg_repo.append(cid, [AIMessage("".join(text_pieces))])
            except Exception as e:  # 上游/网络等异常：error 帧收尾，不炸连接
                logger.exception("orchestrator failed, conversation_id=%s", cid)
                yield "error", {"message": "服务暂时不可用，请稍后重试"}
                return
            yield "done", {"conversation_id": cid}
