from collections.abc import AsyncIterator
from typing import Any

from app.prompts.templates import build_chat_prompt


class ChatService:
    """组装 system+history+human 并流式产出文本。

    model 只需满足 ``astream(messages) -> AsyncIterator[带 .text 属性的 chunk]``，
    便于测试注入 Fake。
    """

    def __init__(self, model: Any) -> None:
        self._prompt = build_chat_prompt()
        self._model = model

    async def astream(self, history: list, user_input: str) -> AsyncIterator[str]:
        messages = self._prompt.format_messages(history=history, input=user_input)
        async for chunk in self._model.astream(messages):
            yield chunk.text

    @staticmethod
    async def collect(stream: AsyncIterator[str]) -> str:
        return "".join([p async for p in stream])
