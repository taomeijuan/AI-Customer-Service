"""create_app 把内置插件注册表交给 agent，不要求 MCP Server 在线。"""

from app.main import create_app


def test_create_app_stores_builtin_registry_without_logistics():
    app = create_app()
    names = set(app.state.tool_registry.names())
    assert "query_logistics" not in names
    assert "query_order" in names
    assert "create_ticket" in names
    assert app.state.tool_registry.session_factory is app.state.session_factory
