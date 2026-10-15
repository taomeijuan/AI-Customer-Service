"""内置 create_ticket。确认后才落库；会话从 ctx 取，不进模型参数。"""

from app.repositories.tickets import TicketsRepo
from app.tools.spec import ToolContext, ToolSpec

_DESCRIPTION = (
    "创建人工客服工单。"
    "只有用户明确要求建工单或转人工时才可以调用。"
    "description 必须来自用户原话；用户没有给出问题描述时不要调用，先追问，禁止编造描述后再调用。"
    "ticket_type 只能是用户明确说过的售后、投诉或咨询；用户没说工单类型就不要调用，不要猜测。"
    "问题描述和工单类型都已经出现在用户原话里时，必须调用本工具，不要只回复文字，也不要再追问订单号。"
    "例如「耳机没声音，帮我建个售后工单」要调用，description 用用户说的问题，ticket_type 填售后。"
    "例如用户只说「帮我建个工单」、没有问题和类型时，不要调用。"
    "用户只是表达不满、没有明确要求建单或转人工时，不要调用，先安抚。"
)

_PARAMETERS = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "minLength": 1, "maxLength": 500},
        "ticket_type": {"type": "string", "enum": ["售后", "投诉", "咨询"]},
    },
    "required": ["description", "ticket_type"],
    "additionalProperties": False,
}


def register(registry) -> None:
    async def _create_ticket(args: dict, ctx: ToolContext) -> dict:
        """引擎按 handler(args, ctx) 调用。conversation_id 只来自 ctx。"""
        if not ctx.confirmed:
            raise RuntimeError("写操作需要用户确认")
        session_factory = getattr(registry, "session_factory", None)
        if session_factory is None:
            raise RuntimeError("未配置 session_factory，无法创建工单")
        if ctx.conversation_id is None:
            raise RuntimeError("缺少会话，无法创建工单")
        async with session_factory() as session:
            ticket_no = await TicketsRepo(session).create(
                conversation_id=ctx.conversation_id,
                description=args["description"],
                ticket_type=args["ticket_type"],
            )
        return {
            "ticket_no": ticket_no,
            "message": "工单已创建，人工客服会尽快联系您",
        }

    registry.register(
        ToolSpec(
            name="create_ticket",
            description=_DESCRIPTION,
            parameters=_PARAMETERS,
            permission="write",
            source="builtin",
            mcp_server=None,
            handler=_create_ticket,
        )
    )
