import asyncio
import uuid

from langchain_core.messages import BaseMessage


class SessionStore:
    """进程内会话存储。ch01 最简版：重启即失，单进程假设。"""

    def __init__(self) -> None:
        self._sessions: dict[str, list[BaseMessage]] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> str:
        cid = uuid.uuid4().hex
        async with self._lock:
            self._sessions[cid] = []
        return cid

    async def get(self, conversation_id: str) -> list[BaseMessage] | None:
        async with self._lock:
            history = self._sessions.get(conversation_id)
            return list(history) if history is not None else None

    async def append(self, conversation_id: str, messages: list[BaseMessage]) -> None:
        async with self._lock:
            history = self._sessions[conversation_id]  # 不存在时 KeyError 由调用方处理
            history.extend(messages)

    async def remove_if_empty(self, conversation_id: str) -> None:
        """清理空会话（流中断/出错且未回填时防泄漏）。"""
        async with self._lock:
            if not self._sessions.get(conversation_id):
                self._sessions.pop(conversation_id, None)
