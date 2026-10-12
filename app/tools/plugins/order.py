"""内置 query_order。新工具是新文件，不改引擎。"""

from app.tools.mock_data import order_data
from app.tools.spec import ToolSpec

_DESCRIPTION = "查询用户订单的状态、商品、金额等信息。当用户询问订单相关问题时调用。"

_PARAMETERS = {
    "type": "object",
    "properties": {"order_no": {"type": "string", "minLength": 1}},
    "required": ["order_no"],
    "additionalProperties": False,
}


def register(registry) -> None:
    registry.register(
        ToolSpec(
            name="query_order",
            description=_DESCRIPTION,
            parameters=_PARAMETERS,
            permission="read",
            source="builtin",
            mcp_server=None,
            handler=order_data,
        )
    )
