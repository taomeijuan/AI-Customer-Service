from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages import BaseMessage
from sqlalchemy import select

from app.db.models import Message


class MessagesRepo:
    """消息流水：BaseMessage 列表 ↔ messages 表（含工具调用申请与结果）。

    LangChain 消息 type（human/ai）与 DB 枚举（user/assistant）显式映射。
    """

    _ROLE_MAP = {"human": "user", "ai": "assistant", "tool": "tool"}

    def __init__(self, session) -> None:
        self._session = session

    async def append(self, conversation_id: int, messages: list[BaseMessage]) -> None:
        """落普通消息（user / 纯文本 assistant）。带 tool_calls 的 AIMessage 必须走 append_tool_round。"""
        for m in messages:
            if isinstance(m, AIMessage) and m.tool_calls:
                raise ValueError("带 tool_calls 的消息请走 append_tool_round，避免申请单丢失")
            self._session.add(
                Message(
                    conversation_id=conversation_id,
                    role=self._ROLE_MAP[m.type],
                    content=m.text,
                )
            )
        await self._session.commit()

    async def append_tool_round(
        self,
        conversation_id: int,
        tool_calls: list[dict],
        tool_results: list[dict],
        ai_content: str = "",
    ) -> None:
        """落一轮工具往返：assistant(tool_calls 申请单) + N 条 tool 结果。

        ai_content：混发场景下工具调用前已播报的正文，落库保证下轮模型可见。
        tool_results: [{"id", "name", "content"}]，content 为回灌给模型的 JSON 字符串。
        """
        self._session.add(
            Message(
                conversation_id=conversation_id,
                role="assistant",
                content=ai_content or None,
                tool_calls=tool_calls,
            )
        )
        for r in tool_results:
            self._session.add(
                Message(
                    conversation_id=conversation_id,
                    role="tool",
                    content=r["content"],
                    tool_call_id=r["id"],
                )
            )
        await self._session.commit()

    async def load_history(self, conversation_id: int) -> list[BaseMessage]:
        """按序重建 BaseMessage 列表；assistant 行的 tool_calls JSON 还原为申请单。"""
        result = await self._session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        rows = result.scalars().all()
        name_by_call_id: dict[str, str] = {}
        for r in rows:
            if r.role == "assistant" and r.tool_calls:
                for tc in r.tool_calls:
                    if tc.get("id"):
                        name_by_call_id[tc["id"]] = tc.get("name")
        out: list[BaseMessage] = []
        for row in rows:
            if row.role == "user":
                out.append(HumanMessage(row.content or ""))
            elif row.role == "assistant":
                out.append(AIMessage(content=row.content or "", tool_calls=row.tool_calls or []))
            else:  # tool：工具名从配对 assistant 的 tool_calls 里按 tool_call_id 反查
                out.append(
                    ToolMessage(
                        content=row.content or "",
                        tool_call_id=row.tool_call_id,
                        name=name_by_call_id.get(row.tool_call_id or ""),
                    )
                )
        return out
