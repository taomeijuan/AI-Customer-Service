"""ch02 业务工具：模拟实时数据（ch05.6 起为确定性假数据）。

真实性约定：不接 MySQL、不再每调随机。同一订单号/商品名用稳定种子
（md5 派生，跨进程一致）生成同一套数据——同一会话内重复询问、不同
会话的引用卡片，答案永远对得上。时间是种子派生的固定锚点，不随调用
时刻漂移。
"""

import hashlib
import random
from datetime import datetime, timedelta

from langchain.tools import tool

_CARRIERS = ["顺丰速运", "京东物流", "中通快递", "圆通速递"]
_ORDER_STATUS = ["待付款", "已付款", "已发货", "已签收", "售后中"]
_PRODUCTS = ["空气炸锅", "无线蓝牙耳机", "人体工学椅", "扫地机器人", "保温杯"]
_PROMOS = ["限时直降50元", "满300减40", "赠运费险", "无优惠"]
_LOGISTICS_NODES = [
    "包裹已到达{city}转运中心",
    "包裹已从{city}发出",
    "快件已到达{city}网点，正在派送",
    "派送员已揽收包裹",
    "包裹已签收，签收人：本人",
]
_CITIES = ["杭州", "上海", "广州", "成都", "武汉", "北京"]


def _seeded_rng(key: str, salt: str) -> random.Random:
    """同一 (salt, key) 永远得到同一把随机种子。md5 派生，进程间一致。"""
    digest = hashlib.md5(f"{salt}:{key}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _anchor_dt(rng: random.Random) -> datetime:
    """种子派生的固定时间锚点：假数据的创建时间/轨迹时刻不随调用时间漂移。"""
    return datetime(2026, 9, 1) + timedelta(
        days=rng.randint(0, 40), hours=rng.randint(0, 23), minutes=rng.randint(0, 59)
    )


def order_data(order_no: str) -> dict:
    """query_order 纯实现（退款子流程直调；工具对象包一层）。"""
    rng = _seeded_rng(str(order_no), "order")
    return {
        "order_no": order_no,
        "status": rng.choice(_ORDER_STATUS),
        "product": rng.choice(_PRODUCTS),
        "amount": round(rng.uniform(29, 2999), 2),
        "created_at": _anchor_dt(rng).strftime("%Y-%m-%d %H:%M"),
    }


def orders_summary() -> list[dict]:
    """list_orders 纯实现：可操作订单 1001-1005 摘要，数据与 query_order 同源。"""
    rows = []
    for no in ("1001", "1002", "1003", "1004", "1005"):
        o = order_data(no)
        rows.append(
            {"order_no": o["order_no"], "product": o["product"], "amount": o["amount"], "status": o["status"]}
        )
    return rows


@tool
def query_order(order_no: str) -> dict:
    """查询用户订单的状态、商品、金额等信息。当用户询问订单相关问题时调用。"""
    return order_data(order_no)


@tool
def list_orders() -> dict:
    """列出当前用户可操作的订单摘要（订单号/商品/金额/状态）。用户想退款但没说订单号时，用于展示订单选择器。"""
    return {"orders": orders_summary()}


@tool
def query_product(product_name: str) -> dict:
    """查询商品的实时价格、库存和促销信息。当用户询问商品问题时调用。"""
    rng = _seeded_rng(str(product_name), "product")
    return {
        "product": product_name,
        "price": round(rng.uniform(49, 1999), 2),
        "stock": rng.randint(0, 500),
        "promo": rng.choice(_PROMOS),
    }


@tool
def query_logistics(order_no: str) -> dict:
    """查询订单的物流承运公司与轨迹节点。当用户询问快递、物流、到哪了时调用。"""
    rng = _seeded_rng(str(order_no), "logistics")
    anchor = _anchor_dt(rng)
    n = rng.randint(3, 5)
    return {
        "order_no": order_no,
        "carrier": rng.choice(_CARRIERS),
        "tracking_no": f"SF{rng.randint(10**13, 10**14 - 1)}",
        "traces": [
            {
                "time": (anchor - timedelta(hours=3 * (n - i))).strftime("%m-%d %H:%M"),
                "desc": rng.choice(_LOGISTICS_NODES).format(
                    city=rng.choice(_CITIES)
                ),
            }
            for i in range(n)
        ],
    }