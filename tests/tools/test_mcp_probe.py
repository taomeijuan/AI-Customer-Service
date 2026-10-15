"""T6 探针：子进程里的两个 FastMCP Server，经 MultiServerMCPClient 现拉工具。"""

import json
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.tools.mock_data import logistics as logistics_data
from app.tools.mock_data import order_data

ROOT = Path(__file__).resolve().parents[2]
HOST = "127.0.0.1"


def _start(script: str) -> tuple[subprocess.Popen[str], list[str]]:
    proc = subprocess.Popen(
        ["uv", "run", "python", script],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines: list[str] = []

    def _drain() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            lines.append(line)

    threading.Thread(target=_drain, daemon=True).start()
    return proc, lines


def _wait_tcp(proc: subprocess.Popen[str], stderr: list[str], port: int, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            time.sleep(0.2)
            raise AssertionError(
                f"{script_of(proc)} exited {proc.returncode} before {HOST}:{port} accepted TCP\n"
                + "".join(stderr)
            )
        try:
            with socket.create_connection((HOST, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    proc.kill()
    time.sleep(0.2)
    raise AssertionError(
        f"{HOST}:{port} did not accept TCP within {timeout}s\n" + "".join(stderr)
    )


def script_of(proc: subprocess.Popen[str]) -> str:
    args = proc.args
    if isinstance(args, list) and args:
        return str(args[-1])
    return str(args)


def _stop(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=5)


def _payload(result: object) -> dict:
    """适配器把工具 dict 收成 text JSON。content block 的 id 每次调用都变。"""
    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, list):
        raise AssertionError(f"unexpected tool result: {result!r}")
    return json.loads("".join(block["text"] for block in result))


async def test_mcp_probe_lists_tools_and_stable_results():
    logistics_proc, logistics_err = _start("mcp_servers/logistics_server.py")
    aftersales_proc, aftersales_err = _start("mcp_servers/aftersales_server.py")
    try:
        _wait_tcp(logistics_proc, logistics_err, 8765)
        _wait_tcp(aftersales_proc, aftersales_err, 8766)

        client = MultiServerMCPClient(
            {
                "logistics": {"transport": "http", "url": "http://127.0.0.1:8765/mcp"},
                "aftersales": {"transport": "http", "url": "http://127.0.0.1:8766/mcp"},
            },
            tool_name_prefix=False,
            handle_tool_errors=False,
        )

        logistics_tools = await client.get_tools(server_name="logistics")
        logistics_names = {tool.name for tool in logistics_tools}
        assert logistics_names == {"query_logistics"}

        aftersales_tools = await client.get_tools(server_name="aftersales")
        aftersales_names = {tool.name for tool in aftersales_tools}
        assert aftersales_names == {"query_warranty", "query_return"}

        query_logistics = next(tool for tool in logistics_tools if tool.name == "query_logistics")
        first = await query_logistics.ainvoke({"order_no": "1001"})
        second = await query_logistics.ainvoke({"order_no": "1001"})
        assert _payload(first) == _payload(second) == logistics_data("1001")

        query_warranty = next(tool for tool in aftersales_tools if tool.name == "query_warranty")
        warranty_a = await query_warranty.ainvoke({"order_no": "1001"})
        warranty_b = await query_warranty.ainvoke({"order_no": "1001"})
        assert _payload(warranty_a) == _payload(warranty_b)
        warranty = _payload(warranty_a)
        assert warranty["order_no"] == "1001"
        assert warranty["product"] == order_data("1001")["product"]
        assert warranty["warranty_status"] in {"IN_WARRANTY", "EXPIRED"}
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", warranty["warranty_until"])

        query_return = next(tool for tool in aftersales_tools if tool.name == "query_return")
        return_a = await query_return.ainvoke({"order_no": "1001"})
        return_b = await query_return.ainvoke({"order_no": "1001"})
        assert _payload(return_a) == _payload(return_b)
        returned = _payload(return_a)
        assert returned["order_no"] == "1001"
        assert returned["return_status"] in {"APPLIED", "IN_TRANSIT", "REFUNDED", "NONE"}
    finally:
        _stop(logistics_proc)
        _stop(aftersales_proc)
