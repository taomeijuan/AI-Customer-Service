"""T6 探针：子进程里的两个 FastMCP Server，经 MultiServerMCPClient 现拉工具。"""

import json
import os
import re
import signal
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
_SERVER_SCRIPTS = (
    "mcp_servers/logistics_server.py",
    "mcp_servers/aftersales_server.py",
)


def _start(script: str) -> tuple[subprocess.Popen[str], list[str]]:
    proc = subprocess.Popen(
        ["uv", "run", "python", script],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
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
    raise AssertionError(
        f"{HOST}:{port} did not accept TCP within {timeout}s\n" + "".join(stderr)
    )


def script_of(proc: subprocess.Popen[str]) -> str:
    args = proc.args
    if isinstance(args, list) and args:
        return str(args[-1])
    return str(args)


def _listener_pids(port: int) -> list[int]:
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
        capture_output=True,
        text=True,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            pid = int(line[1:])
            if pid not in pids:
                pids.append(pid)
    return pids


def _cmdline(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "args="],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _is_our_mcp_server(cmdline: str) -> bool:
    return "python" in cmdline and any(script in cmdline for script in _SERVER_SCRIPTS)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _kill_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if _wait_until(lambda: not _pid_alive(pid), timeout=5):
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    _wait_until(lambda: not _pid_alive(pid), timeout=5)


def _reap_leftover_servers() -> None:
    """上次探针若只杀掉 uv，Python 子进程还会占着 8765/8766。只清这两个脚本。"""
    for port in (8765, 8766):
        killed = False
        for pid in _listener_pids(port):
            if not _is_our_mcp_server(_cmdline(pid)):
                continue
            _kill_pid(pid)
            killed = True
        if killed:
            _wait_port_free(port)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _wait_until(done, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if done():
            return True
        time.sleep(0.05)
    return bool(done())


def _wait_port_free(port: int, timeout: float = 5) -> None:
    _wait_until(lambda: not _listener_pids(port), timeout)


def _wait_group_dead(proc: subprocess.Popen[str], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    pgid = proc.pid
    while time.monotonic() < deadline:
        proc.poll()
        if not _group_alive(pgid):
            return True
        time.sleep(0.05)
    proc.poll()
    return not _group_alive(pgid)


def _stop(proc: subprocess.Popen[str] | None, port: int | None = None) -> None:
    # uv 不 exec 成 Python，也不转发 SIGKILL。只杀 uv 会留下占端口的子进程，所以杀整个进程组。
    if proc is None:
        return
    pgid = proc.pid
    _signal_group(pgid, signal.SIGTERM)
    if not _wait_group_dead(proc, timeout=5):
        _signal_group(pgid, signal.SIGKILL)
        _wait_group_dead(proc, timeout=5)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    if port is not None:
        _wait_port_free(port)


def _payload(result: object) -> dict:
    """适配器把工具 dict 收成 text JSON。content block 的 id 每次调用都变。"""
    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, list):
        raise AssertionError(f"unexpected tool result: {result!r}")
    return json.loads("".join(block["text"] for block in result))


async def test_mcp_probe_lists_tools_and_stable_results():
    _reap_leftover_servers()
    logistics_proc: subprocess.Popen[str] | None = None
    aftersales_proc: subprocess.Popen[str] | None = None
    try:
        logistics_proc, logistics_err = _start("mcp_servers/logistics_server.py")
        aftersales_proc, aftersales_err = _start("mcp_servers/aftersales_server.py")
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
        _stop(logistics_proc, 8765)
        _stop(aftersales_proc, 8766)
