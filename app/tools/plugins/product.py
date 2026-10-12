"""内置 query_product。新工具是新文件，不改引擎。"""

from app.tools.mock_data import product_data
from app.tools.spec import ToolContext, ToolSpec

_DESCRIPTION = "查询商品的实时价格、库存和促销信息。当用户询问商品问题时调用。"

_PARAMETERS = {
    "type": "object",
    "properties": {"product_name": {"type": "string", "minLength": 1}},
    "required": ["product_name"],
    "additionalProperties": False,
}


async def _query_product(args: dict, ctx: ToolContext) -> dict:
    """引擎按 handler(args, ctx) 调用。商品名在参数里，ctx 本工具不读。"""
    del ctx
    return product_data(args["product_name"])


def register(registry) -> None:
    registry.register(
        ToolSpec(
            name="query_product",
            description=_DESCRIPTION,
            parameters=_PARAMETERS,
            permission="read",
            source="builtin",
            mcp_server=None,
            handler=_query_product,
        )
    )
