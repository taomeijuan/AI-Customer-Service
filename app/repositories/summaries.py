"""ch07 分段摘要仓储：一段一行只追加；投影按需从分段表重组（视图级，非数据级）。"""

from sqlalchemy import func, select
from sqlalchemy import text as sa_text

from app.db.models import ConversationSummary
from app.memory.tokens import estimate_tokens


class SummariesRepo:
    def __init__(self, session) -> None:
        self._session = session

    async def list_segments(self, conversation_id: int) -> list[ConversationSummary]:
        rows = await self._session.execute(
            select(ConversationSummary)
            .where(ConversationSummary.conversation_id == conversation_id)
            .order_by(ConversationSummary.seq)
        )
        return list(rows.scalars().all())

    async def next_seq(self, conversation_id: int) -> int:
        cur = await self._session.execute(
            select(func.max(ConversationSummary.seq)).where(
                ConversationSummary.conversation_id == conversation_id
            )
        )
        return (cur.scalar() or 0) + 1

    async def append_segment(
        self,
        conversation_id: int,
        from_msg_id: int,
        upto_msg_id: int,
        content: str,
        projection_budget: int,
    ) -> int:
        """追加一段并同步推进 conversations 边界与投影（单事务）。返回 seq。

        投影按注入预算裁剪（从最新段往回装），装配期直接读列——一次查询，
        不现拼；投影列若因崩溃漂移，随时可从分段表整体重组。
        """
        seq = await self.next_seq(conversation_id)
        self._session.add(
            ConversationSummary(
                conversation_id=conversation_id,
                seq=seq,
                from_msg_id=from_msg_id,
                upto_msg_id=upto_msg_id,
                content=content,
            )
        )
        projection = await self.build_projection(conversation_id, keep_budget=projection_budget)
        # 评审 m4：Core text UPDATE 绕开 ORM onupdate，摘要完成不翻转侧栏 updated_at 排序
        await self._session.execute(
            sa_text(
                "UPDATE conversations SET summary_upto_msg_id=:u, summary=:p WHERE id=:cid"
            ),
            {"u": upto_msg_id, "p": projection, "cid": conversation_id},
        )
        await self._session.commit()
        return seq

    async def build_projection(self, conversation_id: int, keep_budget: int) -> str:
        """从分段表重组投影：从最新段往回取，超预算即停（旧段是视图级舍弃）。"""
        segments = await self.list_segments(conversation_id)
        parts: list[str] = []
        total = 0
        for seg in reversed(segments):
            cost = estimate_tokens(seg.content)
            if parts and total + cost > keep_budget:
                break
            parts.append(f"〔第{seg.seq}段〕{seg.content}")
            total += cost
        return "\n".join(reversed(parts))
