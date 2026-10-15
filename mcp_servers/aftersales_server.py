import sys
from datetime import datetime, timedelta
from pathlib import Path

# 脚本目录在 sys.path[0]，仓库根才有 app。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "aftersales",
    host="127.0.0.1",
    port=8766,
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
def query_warranty(order_no: str) -> dict:
    """查询订单商品的质保状态与到期日。当用户询问保修、质保、过保时调用。"""
    from app.tools.mock_data import _seeded_rng, order_data

    rng = _seeded_rng(str(order_no), "warranty")
    until = datetime(2026, 9, 1) + timedelta(days=rng.randint(0, 400))
    return {
        "order_no": order_no,
        "product": order_data(order_no)["product"],
        "warranty_status": rng.choice(["IN_WARRANTY", "EXPIRED"]),
        "warranty_until": until.strftime("%Y-%m-%d"),
    }


@mcp.tool()
def query_return(order_no: str) -> dict:
    """查询订单退货进度。当用户询问退货、退款到哪了时调用。"""
    from app.tools.mock_data import _seeded_rng

    rng = _seeded_rng(str(order_no), "return")
    return {
        "order_no": order_no,
        "return_status": rng.choice(["APPLIED", "IN_TRANSIT", "REFUNDED", "NONE"]),
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
