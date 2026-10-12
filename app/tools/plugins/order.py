"""内置 query_order。新工具是新文件，不改引擎。"""

from app.tools.mock_data import order_data
from app.tools.spec import ToolContext, ToolSpec

_DESCRIPTION = "查询用户订单的状态、商品、金额等信息。当用户询问订单相关问题时调用。"

_PARAMETERS = {
    "type": "object",
    "properties": {"order_no": {"type": "string", "minLength": 1}},
    "required": ["order_no"],
    "additionalProperties": False,
}


async def _query_order(args: dict, ctx: ToolContext) -> dict:
    """引擎按 handler(args, ctx) 调用。订单号在参数里，ctx 本工具不读。"""
    del ctx
    return order_data(args["order_no"])


def register(registry) -> None:
    registry.register(
        ToolSpec(
            name="query_order",
            description=_DESCRIPTION,
            parameters=_PARAMETERS,
            permission="read",
            source="builtin",
            mcp_server=None,
            handler=_query_order,
        )
    )
