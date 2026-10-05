from typing import Any, Literal

from langchain.tools import tool
from pydantic import BaseModel, Field

from app.repositories.tickets import TicketsRepo


def build_create_ticket_tool(session: Any, conversation_id: int):
    """建人工工单工具。会话级工厂：conversation_id 由编排层闭包注入，不进模型参数。"""

    class CreateTicketInput(BaseModel):
        """创建人工工单所需信息。"""

        description: str = Field(description="用户问题的完整描述，转人工前整理")
        ticket_type: Literal["售后", "投诉", "咨询"] = Field(description="工单类型")

    @tool(args_schema=CreateTicketInput)
    async def create_ticket(description: str, ticket_type: str) -> dict:
        """创建人工客服工单。用户一旦明确要求转人工/建单/投诉，必须立即调用本工具，用用户已提供的信息填写 description，不要再追问、不要先查询其他信息。"""
        ticket_no = await TicketsRepo(session).create(
            conversation_id=conversation_id,
            description=description,
            ticket_type=ticket_type,
        )
        return {"ok": True, "ticket_no": ticket_no, "message": "工单已创建，人工客服会尽快联系您"}

    return create_ticket
