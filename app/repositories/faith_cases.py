from datetime import datetime

from sqlalchemy import select

from app.db.models import FaithCase


class FaithCasesRepo:
    """编造个案台账：一题一行（uk_eval_id），跨轮追溯。

    record 语义：
    - 首判 → 新行（seen_count=1）
    - 复判 → 答案/理由/角标快照更新为最近一次，seen_count+1，last_seen_at 刷新
    - 若此前已标「已解决/无需解决」→ 状态退回「未解决」（复发），清空 resolution、
      保留 resolved_at（有值 + 未解决 = 复发标记）
    """

    def __init__(self, session) -> None:
        self._session = session

    async def record(
        self,
        eval_id: str,
        bucket: str,
        query: str,
        strategy: str,
        answer: str,
        reason: str,
        citations: list | None,
        judge_model: str | None,
    ) -> int:
        existing = await self.get(eval_id)
        if existing is None:
            row = FaithCase(
                eval_id=eval_id,
                bucket=bucket,
                query=query,
                strategy=strategy,
                answer=answer,
                reason=reason,
                citations=citations,
                judge_model=judge_model,
            )
            self._session.add(row)
            await self._session.commit()
            return row.id

        existing.answer = answer
        existing.reason = reason
        existing.citations = citations
        existing.strategy = strategy
        existing.judge_model = judge_model
        existing.seen_count += 1
        existing.last_seen_at = datetime.now()
        if existing.status != "未解决":  # 复发
            existing.status = "未解决"
            existing.resolution = None
        await self._session.commit()
        return existing.id

    async def get(self, eval_id: str) -> FaithCase | None:
        result = await self._session.execute(
            select(FaithCase).where(FaithCase.eval_id == eval_id)
        )
        return result.scalars().one_or_none()

    async def mark_resolved(self, eval_id: str, resolution: str, status: str) -> None:
        """status: 已解决 / 无需解决；resolution 必填——空着的处置等于没有交代。"""
        assert status in ("已解决", "无需解决")
        row = await self.get(eval_id)
        row.status = status
        row.resolution = resolution
        row.resolved_at = datetime.now()
        await self._session.commit()

    async def list_unresolved(self, limit: int = 100) -> list[FaithCase]:
        result = await self._session.execute(
            select(FaithCase)
            .where(FaithCase.status == "未解决")
            .order_by(FaithCase.seen_count.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
