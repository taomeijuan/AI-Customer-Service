from sqlalchemy import select

from app.db.models import Conversation, Message


class ConversationsRepo:
    """会话壳数据访问。DB 为唯一真源：ch01 内存 store 由此替代。"""

    def __init__(self, session) -> None:
        self._session = session

    async def ensure_conversation(
        self, user_id: str, conversation_id: int | None = None
    ) -> int:
        """带 id 则校验存在且归属该用户（否则 KeyError→上层 404）；不带则新建。"""
        if conversation_id is not None:
            conv = await self.get(conversation_id)
            if conv is None or conv.user_id != user_id:  # 归属校验防跨用户写会话
                raise KeyError(f"会话不存在: {conversation_id}")
            return conversation_id
        conv = Conversation(user_id=user_id)
        self._session.add(conv)
        await self._session.commit()
        return conv.id

    async def get(self, conversation_id: int) -> Conversation | None:
        result = await self._session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        return result.scalars().one_or_none()


# ---- ch07 会话侧栏只读查询（独立函数，不撑大 ConversationsRepo 类职责）----
async def list_user_conversations(session, user_id: str) -> list[dict]:
    """该用户全部会话：新在前，带首问预览与已摘要标记。"""
    from sqlalchemy import text as sa_text

    rows = (
        await session.execute(
            sa_text(
                """
                SELECT c.id, c.updated_at, c.summary_upto_msg_id,
                       (SELECT m.content FROM messages m
                         WHERE m.conversation_id = c.id AND m.role='user'
                         ORDER BY m.id LIMIT 1) AS first_q
                FROM conversations c
                WHERE c.user_id = :u
                ORDER BY c.updated_at DESC, c.id DESC
                """
            ),
            {"u": user_id},
        )
    ).all()
    return [
        {
            "id": r[0],
            "title": (r[3] or "（新会话）")[:20],
            "updated_at": r[1].isoformat(timespec="seconds"),
            "has_summary": bool(r[2]),
        }
        for r in rows
    ]


async def get_visible_messages(session, conversation_id: int, user_id: str) -> list[dict] | None:
    """回载历史原文（跳过 tool 行与工具申请行）；非本人会话返回 None。"""
    conv = await session.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user_id:
        return None
    from sqlalchemy import select

    rows = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
    ).scalars().all()
    out: list[dict] = []
    for r in rows:
        if r.role == "tool" or (r.role == "assistant" and r.tool_calls and not r.content):
            continue
        out.append({"role": r.role, "content": r.content or "", "citations": r.citations or None})
    return out
