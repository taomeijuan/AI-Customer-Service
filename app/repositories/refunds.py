"""退款单仓库（ch06）：R+yyyyMMdd+当日序号，生成惯例与 TicketsRepo 一致。"""

import asyncio
import datetime
import random

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.models import RefundOrder

# 退款原因固定类目（前后端同源：/api/refunds 校验与前端下拉都用这份）
REFUND_REASONS = ["七天无理由", "质量问题", "少件", "与描述不符", "其他"]


class RefundsRepo:
    MAX_ATTEMPTS = 5

    def __init__(self, session) -> None:
        self._session = session

    async def create(
        self, conversation_id: int, order_no: str, reason_category: str, amount: float
    ) -> str:
        last_err: Exception | None = None
        for attempt in range(self.MAX_ATTEMPTS):
            refund_no = await self._next_refund_no()
            self._session.add(
                RefundOrder(
                    refund_no=refund_no,
                    conversation_id=conversation_id,
                    order_no=order_no,
                    reason_category=reason_category,
                    amount=amount,
                )
            )
            try:
                await self._session.commit()
                return refund_no
            except IntegrityError as e:
                await self._session.rollback()
                errno = e.orig.args[0] if e.orig and e.orig.args else None
                if errno == 1452:  # 会话不存在：重试无意义
                    raise
                last_err = e  # 1062 撞号：退避重取
                await asyncio.sleep(random.uniform(0.01, 0.05) * (attempt + 1))
        raise last_err  # type: ignore[misc]

    async def _next_refund_no(self) -> str:
        today = datetime.date.today().strftime("%Y%m%d")
        prefix = f"R{today}"
        rows = (
            await self._session.execute(
                select(RefundOrder.refund_no).where(RefundOrder.refund_no.like(f"{prefix}%"))
            )
        ).scalars().all()
        max_seq = max(
            (int(no[len(prefix):]) for no in rows if no[len(prefix):].isdigit()),
            default=0,
        )
        return f"{prefix}{max_seq + 1:03d}"
