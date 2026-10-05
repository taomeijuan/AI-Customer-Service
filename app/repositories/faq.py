from sqlalchemy import or_, select

from app.db.models import Faq


class FaqRepo:
    """FAQ 关键词查询（SQL LIKE）。ch03 起检索改走向量库。"""

    def __init__(self, session) -> None:
        self._session = session

    async def search(self, keyword: str, limit: int = 3) -> list[dict]:
        if not keyword or not keyword.strip():
            return []
        escaped = (
            keyword.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        kw = f"%{escaped}%"
        result = await self._session.execute(
            select(Faq)
            .where(or_(Faq.question.like(kw), Faq.answer.like(kw)))
            .limit(limit)
        )
        rows = result.scalars().all()
        return [
            {"question": r.question, "answer": r.answer, "category": r.category}
            for r in rows
        ]
