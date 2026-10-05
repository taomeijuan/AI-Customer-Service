from sqlalchemy import select

from app.db.models import Conversation


class ConversationsRepo:
    """会话壳数据访问。DB 为唯一真源：ch01 内存 store 由此替代。"""

    def __init__(self, session) -> None:
        self._session = session

    async def ensure_conversation(
        self, user_id: str, conversation_id: int | None = None
    ) -> int:
        """带 id 则校验存在（不存在抛 KeyError→上层 404）；不带则新建。"""
        if conversation_id is not None:
            conv = await self.get(conversation_id)
            if conv is None:
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
