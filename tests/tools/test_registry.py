"""T4：插件扫描只登记订单/商品；物流假数据种子与旧工具一致。"""

import inspect

from app.tools.ecommerce import query_order
from app.tools.mock_data import logistics, order_data
from app.tools.registry import ToolRegistry, scan_plugins
from app.tools.spec import ToolSpec

_ORDER_SCHEMA = {
    "type": "object",
    "properties": {"order_no": {"type": "string", "minLength": 1}},
    "required": ["order_no"],
    "additionalProperties": False,
}
_PRODUCT_SCHEMA = {
    "type": "object",
    "properties": {"product_name": {"type": "string", "minLength": 1}},
    "required": ["product_name"],
    "additionalProperties": False,
}


def test_scan_registers_order_and_product_not_logistics():
    registry = ToolRegistry()
    scan_plugins(registry)
    names = set(registry.names())
    assert names == {"query_order", "query_product"}
    assert "query_logistics" not in names

    order = registry.get("query_order")
    assert order.permission == "read"
    assert order.source == "builtin"
    assert order.mcp_server is None
    assert order.description == (
        "查询用户订单的状态、商品、金额等信息。当用户询问订单相关问题时调用。"
    )
    assert order.parameters == _ORDER_SCHEMA

    product = registry.get("query_product")
    assert product.permission == "read"
    assert product.source == "builtin"
    assert product.mcp_server is None
    assert product.description == (
        "查询商品的实时价格、库存和促销信息。当用户询问商品问题时调用。"
    )
    assert product.parameters == _PRODUCT_SCHEMA


def test_in_test_register_shows_up_without_editing_engine():
    """新工具只 register 一份 ToolSpec，不必改 engine.py。"""
    registry = ToolRegistry()
    scan_plugins(registry)
    before = set(registry.names())
    registry.register(
        ToolSpec(
            name="ping_probe",
            description="探测新工具能否进表",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            permission="read",
            source="builtin",
            mcp_server=None,
            handler=lambda: {"ok": True},
        )
    )
    assert "ping_probe" in registry.names()
    assert registry.get("ping_probe").name == "ping_probe"
    assert before == {"query_order", "query_product"}


async def _call(handler, *args, **kwargs):
    result = handler(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def test_logistics_stable_and_order_seed_matches_legacy_tool():
    assert logistics("1001") == logistics("1001")
    legacy = await query_order.ainvoke({"order_no": "1001"})
    assert order_data("1001") == legacy

    registry = ToolRegistry()
    scan_plugins(registry)
    assert await _call(registry.get("query_order").handler, "1001") == legacy
