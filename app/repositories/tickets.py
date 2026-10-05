import asyncio
import datetime
import random

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.models import Ticket


class TicketsRepo:
    """人工工单：T+yyyyMMdd+当日序号（取当日 MAX+1，并发冲突退避重试）。"""

    MAX_ATTEMPTS = 5

    def __init__(self, session) -> None:
        self._session = session

    async def create(
        self, conversation_id: int, description: str, ticket_type: str
    ) -> str:
        last_err: Exception | None = None
        for attempt in range(self.MAX_ATTEMPTS):
            ticket_no = await self._next_ticket_no()
            self._session.add(
                Ticket(
                    ticket_no=ticket_no,
                    conversation_id=conversation_id,
                    description=description,
                    ticket_type=ticket_type,
                )
            )
            try:
                await self._session.commit()
                return ticket_no
            except IntegrityError as e:
                await self._session.rollback()
                errno = e.orig.args[0] if e.orig and e.orig.args else None
                if errno == 1452:  # 外键冲突（会话不存在）：重试无意义
                    raise
                last_err = e  # 1062 撞号：退避后重取
                await asyncio.sleep(random.uniform(0.01, 0.05) * (attempt + 1))
        raise last_err  # type: ignore[misc]

    async def _next_ticket_no(self) -> str:
        today = datetime.date.today().strftime("%Y%m%d")
        prefix = f"T{today}"
        # 取当日最大序号（而非 COUNT）：即使有删单也不回退
        rows = (
            await self._session.execute(
                select(Ticket.ticket_no).where(Ticket.ticket_no.like(f"{prefix}%"))
            )
        ).scalars().all()
        max_seq = max(
            (int(no[len(prefix):]) for no in rows if no[len(prefix):].isdigit()),
            default=0,
        )
        return f"{prefix}{max_seq + 1:03d}"
