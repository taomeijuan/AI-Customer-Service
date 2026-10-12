"""ch02 业务工具外壳。确定性假数据在 mock_data，避免两套种子。

query_logistics 仍留在本模块供旧测试调用，插件扫描不登记它。
"""

from langchain.tools import tool

from app.tools.mock_data import logistics, order_data, product_data


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
def query_product(product_name: str) -> dict:
    """查询商品的实时价格、库存和促销信息。当用户询问商品问题时调用。"""
    return product_data(product_name)


@tool
def query_logistics(order_no: str) -> dict:
    """查询订单的物流承运公司与轨迹节点。当用户询问快递、物流、到哪了时调用。"""
    return logistics(order_no)
