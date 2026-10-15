import sys
from pathlib import Path

# 脚本目录在 sys.path[0]，仓库根才有 app。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "logistics",
    host="127.0.0.1",
    port=8765,
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
def query_logistics(order_no: str) -> dict:
    """查询订单的物流承运公司与轨迹节点。当用户询问快递、物流、到哪了时调用。"""
    from app.tools.mock_data import logistics

    return logistics(order_no)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
