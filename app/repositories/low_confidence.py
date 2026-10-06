from sqlalchemy import select

from app.db.models import LowConfidenceQuestion


class LowConfidenceRepo:
    """低置信度问题池：三条入池入口（检索低分 / 自评不足 / 用户反馈）。"""

    def __init__(self, session) -> None:
        self._session = session

    async def record(
        self,
        raw_question: str,
        source: str,
        reason: str | None = None,
        conversation_id: int | None = None,
    ) -> int:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id,
            raw_question=raw_question,
            source=source,
            reason=reason,
        )
        self._session.add(row)
        await self._session.commit()
        return row.id

    async def get(self, row_id: int) -> LowConfidenceQuestion | None:
        result = await self._session.execute(
            select(LowConfidenceQuestion).where(LowConfidenceQuestion.id == row_id)
        )
        return result.scalars().one_or_none()

    async def list_by_source(self, source: str, limit: int = 100) -> list:
        result = await self._session.execute(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.source == source)
            .order_by(LowConfidenceQuestion.id.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
