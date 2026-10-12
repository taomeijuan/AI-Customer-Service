from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


def validate_args(schema: dict, args: dict) -> str | None:
    """校验参数。失败返回一句中文，成功返回 None。不向调用方抛校验异常。"""
    error = next(Draft202012Validator(schema).iter_errors(args), None)
    if error is None:
        return None
    return _chinese_message(error)


def _chinese_message(error: ValidationError) -> str:
    field = _field_name(error)
    if error.validator == "required":
        return f"参数不合法：缺少必填字段 {field}"
    if error.validator == "type":
        return f"参数不合法：{field} 类型不正确"
    if error.validator == "enum":
        return f"参数不合法：{field} 不在允许的取值内"
    if error.validator == "minLength":
        return f"参数不合法：{field} 长度不足"
    if error.validator == "additionalProperties":
        return f"参数不合法：不允许多余字段 {field}"
    return f"参数不合法：{field} 不符合要求"


def _field_name(error: ValidationError) -> str:
    if error.validator == "required":
        missing = [name for name in error.validator_value if name not in error.instance]
        return missing[0] if missing else "参数"
    if error.validator == "additionalProperties":
        properties = error.schema.get("properties", {})
        extras = [name for name in error.instance if name not in properties]
        return extras[0] if extras else "参数"
    if error.path:
        return str(error.path[-1])
    return "参数"
