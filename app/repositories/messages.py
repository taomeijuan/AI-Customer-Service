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

    async def append(
        self,
        conversation_id: int,
        messages: list[BaseMessage],
        citations: list[dict] | None = None,
    ) -> None:
        """落普通消息（user / 纯文本 assistant）。带 tool_calls 的 AIMessage 必须走 append_tool_round。

        citations（ch07）：本轮引用快照，挂在最后一条 assistant 行——侧栏回载
        历史时还原 markdown 与可点击引用。
        """
        rows = []
        for m in messages:
            if isinstance(m, AIMessage) and m.tool_calls:
                raise ValueError("带 tool_calls 的消息请走 append_tool_round，避免申请单丢失")
            rows.append(
                Message(
                    conversation_id=conversation_id,
                    role=self._ROLE_MAP[m.type],
                    content=m.text,
                )
            )
        if citations:
            for r in reversed(rows):
                if r.role == "assistant":
                    r.citations = citations
                    break
        self._session.add_all(rows)
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

    async def load_dialog_pairs(self) -> dict[int, list[tuple[str, str]]]:
        """全部会话的 (用户问, 客服答) 配对，按会话分组，供离线挖知识。

        跳过 tool 消息与工具申请行；中间带 tool_calls 的 assistant 不截断配对，
        等最终带正文的 assistant 才成对。按会话分组避免「A 会话末尾未回复的
        user 消息」与「B 会话开头的 assistant 回复」错配成假问答对。
        """
        result = await self._session.execute(
            select(Message).order_by(Message.conversation_id, Message.id)
        )
        rows = result.scalars().all()
        grouped: dict[int, list[tuple[str, str]]] = {}
        pending_user: str | None = None
        current_cid: int | None = None
        for row in rows:
            if row.conversation_id != current_cid:  # 换会话：未回复的 pending 作废
                current_cid = row.conversation_id
                pending_user = None
                grouped.setdefault(current_cid, [])
            if row.role == "user":
                pending_user = row.content
            elif row.role == "assistant":
                if pending_user and row.content and not row.tool_calls:
                    grouped[current_cid].append((pending_user, row.content))
                    pending_user = None
            # tool 结果与工具申请行不打断配对
        return grouped

    async def load_history_with_ids(self, conversation_id: int) -> list[tuple[int, BaseMessage]]:
        """ch07 分层用：(消息 id, BaseMessage) 按序——层边界靠 id 表达，不搬数据。"""
        rows = await self._fetch_rows(conversation_id)
        name_by_call_id = self._tool_names(rows)
        out: list[tuple[int, BaseMessage]] = []
        for row in rows:
            out.append((row.id, self._row_to_message(row, name_by_call_id)))
        return out

    async def load_history(self, conversation_id: int) -> list[BaseMessage]:
        """按序重建 BaseMessage 列表；assistant 行的 tool_calls JSON 还原为申请单。"""
        rows = await self._fetch_rows(conversation_id)
        name_by_call_id = self._tool_names(rows)
        return [self._row_to_message(row, name_by_call_id) for row in rows]

    async def _fetch_rows(self, conversation_id: int):
        result = await self._session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return result.scalars().all()

    @staticmethod
    def _tool_names(rows) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for r in rows:
            if r.role == "assistant" and r.tool_calls:
                for tc in r.tool_calls:
                    if tc.get("id"):
                        mapping[tc["id"]] = tc.get("name")
        return mapping

    @staticmethod
    def _row_to_message(row, name_by_call_id: dict[str, str]) -> BaseMessage:
        if row.role == "user":
            return HumanMessage(row.content or "")
        if row.role == "assistant":
            return AIMessage(content=row.content or "", tool_calls=row.tool_calls or [])
        # tool：工具名从配对 assistant 的 tool_calls 里按 tool_call_id 反查
        return ToolMessage(
            content=row.content or "",
            tool_call_id=row.tool_call_id,
            name=name_by_call_id.get(row.tool_call_id or ""),
        )
