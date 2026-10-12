import pytest

from app.tools.errors import LookupMiss
from app.tools.spec import ToolContext, ToolSpec
from app.tools.validate import validate_args

TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "minLength": 1},
        "ticket_type": {"type": "string", "enum": ["售后", "投诉", "咨询"]},
    },
    "required": ["description", "ticket_type"],
    "additionalProperties": False,
}


def test_valid_args_return_none():
    assert (
        validate_args(
            TICKET_SCHEMA,
            {"description": "耳机没声音", "ticket_type": "售后"},
        )
        is None
    )


def test_missing_required_names_field():
    msg = validate_args(TICKET_SCHEMA, {"ticket_type": "售后"})
    assert msg == "参数不合法：缺少必填字段 description"


def test_wrong_type_names_field():
    msg = validate_args(TICKET_SCHEMA, {"description": 1, "ticket_type": "售后"})
    assert msg == "参数不合法：description 类型不正确"


def test_enum_violation_names_field():
    msg = validate_args(
        TICKET_SCHEMA,
        {"description": "耳机没声音", "ticket_type": "其他"},
    )
    assert msg == "参数不合法：ticket_type 不在允许的取值内"


def test_min_length_names_field():
    msg = validate_args(TICKET_SCHEMA, {"description": "", "ticket_type": "售后"})
    assert msg == "参数不合法：description 长度不足"


def test_additional_property_names_field():
    msg = validate_args(
        TICKET_SCHEMA,
        {"description": "耳机没声音", "ticket_type": "售后", "extra": 1},
    )
    assert msg == "参数不合法：不允许多余字段 extra"


def test_validation_failure_does_not_raise():
    msg = validate_args(TICKET_SCHEMA, {})
    assert isinstance(msg, str)
    assert msg.startswith("参数不合法：")


def test_tool_context_defaults_and_frozen():
    ctx = ToolContext(conversation_id=1, tool_call_id="call-1")
    assert ctx.conversation_id == 1
    assert ctx.tool_call_id == "call-1"
    assert ctx.confirmed is False
    with pytest.raises(AttributeError):
        ctx.confirmed = True  # frozen dataclass


def test_lookup_miss_signals_empty_result():
    assert issubclass(LookupMiss, Exception)
    assert "查询落空" in (LookupMiss.__doc__ or "")
    assert "不重试" in (LookupMiss.__doc__ or "")


def test_tool_spec_keeps_schema_handler_and_optional_present():
    def handler(args, ctx):
        return {}

    spec = ToolSpec(
        name="create_ticket",
        description="建工单",
        parameters=TICKET_SCHEMA,
        permission="write",
        source="builtin",
        mcp_server=None,
        handler=handler,
    )
    assert spec.parameters["required"] == ["description", "ticket_type"]
    assert spec.handler is handler
    assert spec.present is None
    assert spec.permission == "write"
    assert spec.source == "builtin"
