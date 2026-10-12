"""内置 query_product。新工具是新文件，不改引擎。"""

from app.tools.mock_data import product_data
from app.tools.spec import ToolSpec

_DESCRIPTION = "查询商品的实时价格、库存和促销信息。当用户询问商品问题时调用。"

_PARAMETERS = {
    "type": "object",
    "properties": {"product_name": {"type": "string", "minLength": 1}},
    "required": ["product_name"],
    "additionalProperties": False,
}


def register(registry) -> None:
    registry.register(
        ToolSpec(
            name="query_product",
            description=_DESCRIPTION,
            parameters=_PARAMETERS,
            permission="read",
            source="builtin",
            mcp_server=None,
            handler=product_data,
        )
    )
