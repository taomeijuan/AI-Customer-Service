from sqlalchemy import select

from app.db.models import KnowledgeChunk


class KnowledgeRepo:
    """knowledge_chunks 数据访问：文档级幂等 + 双写状态机 MySQL 侧。"""

    def __init__(self, session) -> None:
        self._session = session

    async def upsert_chunks(self, chunks: list[dict]) -> list[int]:
        """按 (category, questions, answer) 内容精确匹配复用已有行（文档重跑不重复建）。"""
        ids: list[int] = []
        for c in chunks:
            existing = (
                await self._session.execute(
                    select(KnowledgeChunk).where(
                        KnowledgeChunk.category == c["category"],
                        KnowledgeChunk.questions == c["questions"],
                        KnowledgeChunk.answer == c["answer"],
                    )
                )
            ).scalars().one_or_none()
            if existing is None:
                existing = KnowledgeChunk(**c)
                self._session.add(existing)
                await self._session.flush()  # 拿自增 id
            ids.append(existing.id)
        await self._session.commit()
        return ids

    async def get(self, chunk_id: int) -> KnowledgeChunk | None:
        result = await self._session.execute(
            select(KnowledgeChunk).where(KnowledgeChunk.id == chunk_id)
        )
        return result.scalars().one_or_none()

    async def list_pending(self, limit: int = 100) -> list[KnowledgeChunk]:
        result = await self._session.execute(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_done(self, chunk_id: int, vector_id: str) -> None:
        chunk = await self.get(chunk_id)
        chunk.vector_id = vector_id
        chunk.vectorize_status = "done"
        await self._session.commit()

    async def link_neighbors(self, ordered_ids: list[int]) -> None:
        """同文档相邻 chunk 的 prev/next 指针回填。"""
        for prev_id, cur, next_id in zip(
            [None, *ordered_ids[:-1]], ordered_ids, [*ordered_ids[1:], None]
        ):
            chunk = await self.get(cur)
            chunk.prev_chunk_id = prev_id
            chunk.next_chunk_id = next_id
        await self._session.commit()
