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


@tool
def query_order(order_no: str) -> dict:
    """查询用户订单的状态、商品、金额等信息。当用户询问订单相关问题时调用。"""
    return {
        "order_no": order_no,
        "status": random.choice(_ORDER_STATUS),
        "product": random.choice(_PRODUCTS),
        "amount": round(random.uniform(29, 2999), 2),
        "created_at": (
            datetime.now() - timedelta(days=random.randint(0, 10))
        ).strftime("%Y-%m-%d %H:%M"),
    }


@tool
def query_product(product_name: str) -> dict:
    """查询商品的实时价格、库存和促销信息。当用户询问商品问题时调用。"""
    return {
        "product": product_name,
        "price": round(random.uniform(49, 1999), 2),
        "stock": random.randint(0, 500),
        "promo": random.choice(_PROMOS),
    }


@tool
def query_logistics(order_no: str) -> dict:
    """查询订单的物流承运公司与轨迹节点。当用户询问快递、物流、到哪了时调用。"""
    now = datetime.now()
    n = random.randint(3, 5)
    return {
        "order_no": order_no,
        "carrier": random.choice(_CARRIERS),
        "tracking_no": f"SF{random.randint(10**13, 10**14 - 1)}",
        "traces": [
            {
                "time": (now - timedelta(hours=3 * (n - i))).strftime("%m-%d %H:%M"),
                "desc": random.choice(_LOGISTICS_NODES).format(
                    city=random.choice(_CITIES)
                ),
            }
            for i in range(n)
        ],
    }
