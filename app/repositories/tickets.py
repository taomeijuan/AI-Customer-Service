import datetime

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.db.models import Ticket


class TicketsRepo:
    """人工工单：T+yyyyMMdd+当日3位序号，唯一键冲突重试。"""

    def __init__(self, session) -> None:
        self._session = session

    async def create(
        self, conversation_id: int, description: str, ticket_type: str
    ) -> str:
        last_err: Exception | None = None
        for _ in range(3):
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
            except IntegrityError as e:  # 并发撞号：回滚重取
                await self._session.rollback()
                last_err = e
        raise last_err  # type: ignore[misc]

    async def _next_ticket_no(self) -> str:
        today = datetime.date.today().strftime("%Y%m%d")
        prefix = f"T{today}"
        result = await self._session.execute(
            select(func.count())
            .select_from(Ticket)
            .where(Ticket.ticket_no.like(f"{prefix}%"))
        )
        seq = (result.scalar() or 0) + 1
        return f"{prefix}{seq:03d}"
