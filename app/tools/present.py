"""MCP 工具回灌前的字段挑选和枚举人话。引擎在 spec.present 存在时调用。"""

_WARRANTY_STATUS = {
    "IN_WARRANTY": "在保",
    "EXPIRED": "已过保",
}

_RETURN_STATUS = {
    "APPLIED": "已申请",
    "IN_TRANSIT": "寄回途中",
    "REFUNDED": "已退款",
    "NONE": "无退货单",
}


def query_logistics(raw: dict) -> dict:
    traces = []
    for item in raw.get("traces") or []:
        if not isinstance(item, dict):
            continue
        traces.append({"time": item.get("time"), "desc": item.get("desc")})
    return {
        "order_no": raw.get("order_no"),
        "carrier": raw.get("carrier"),
        "tracking_no": raw.get("tracking_no"),
        "traces": traces,
    }


def query_warranty(raw: dict) -> dict:
    status = raw.get("warranty_status")
    return {
        "order_no": raw.get("order_no"),
        "product": raw.get("product"),
        "warranty_status": _WARRANTY_STATUS.get(status, status),
        "warranty_until": raw.get("warranty_until"),
    }


def query_return(raw: dict) -> dict:
    status = raw.get("return_status")
    return {
        "order_no": raw.get("order_no"),
        "return_status": _RETURN_STATUS.get(status, status),
    }
