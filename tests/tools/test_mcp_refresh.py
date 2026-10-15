"""T7：现拉 MCP 工具进注册表。"""

import json
import logging
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel

from app.core.config import Settings
from app.tools.engine import execute
from app.tools.mcp_client import build_mcp_client, refresh
from app.tools.present import query_logistics, query_return, query_warranty
from app.tools.registry import ToolRegistry, scan_plugins
from app.tools.spec import ToolContext

_ROOT = Path(__file__).resolve().parents[2]
_HOST = "127.0.0.1"
_PORT = 8771
_SANDBOX_NAME = "ecom_cs_t7_sandbox_mcp.py"
_SANDBOX_PATH = Path(tempfile.gettempdir()) / _SANDBOX_NAME


class _EmptyArgs(BaseModel):
    pass


class _Tool:
    def __init__(self, name: str, description: str, payload: dict) -> None:
        self.name = name
        self.description = description
        self.args_schema = _EmptyArgs
        self.payload = payload

    async def ainvoke(self, args):
        return [{"type": "text", "text": json.dumps(self.payload), "id": "a"}]


class _Client:
    def __init__(self, rounds: list[list[_Tool]]) -> None:
        self.rounds = rounds
        self.calls: list[str] = []

    async def get_tools(self, *, server_name: str):
        self.calls.append(server_name)
        return self.rounds.pop(0)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        tool_timeout=0.5,
        tool_retries=0,
        tool_delay_tools="",
        tool_delay_seconds=0,
    )


def test_settings_default_mcp_urls_and_client_connections():
    fields = Settings.model_fields
    assert fields["mcp_logistics_url"].default == "http://127.0.0.1:8765/mcp"
    assert fields["mcp_aftersales_url"].default == "http://127.0.0.1:8766/mcp"

    settings = Settings(
        _env_file=None,
        mcp_logistics_url="http://127.0.0.1:8765/mcp",
        mcp_aftersales_url="http://127.0.0.1:8766/mcp",
    )
    client = build_mcp_client(settings)
    assert client.tool_name_prefix is False
    assert client.handle_tool_errors is False
    assert client.connections["logistics"] == {
        "transport": "http",
        "url": "http://127.0.0.1:8765/mcp",
    }
    assert client.connections["aftersales"] == {
        "transport": "http",
        "url": "http://127.0.0.1:8766/mcp",
    }


def test_presenters_keep_fields_and_translate_enums():
    presented = query_logistics(
        {
            "order_no": "1001",
            "carrier": "顺丰速运",
            "tracking_no": "SF1",
            "traces": [{"time": "09-01 10:00", "desc": "已揽收", "extra": "x"}],
            "_secret": 1,
        }
    )
    assert presented == {
        "order_no": "1001",
        "carrier": "顺丰速运",
        "tracking_no": "SF1",
        "traces": [{"time": "09-01 10:00", "desc": "已揽收"}],
    }

    warranty = query_warranty(
        {
            "order_no": "1001",
            "product": "无线蓝牙耳机",
            "warranty_status": "IN_WARRANTY",
            "warranty_until": "2027-01-01",
            "_secret": 1,
        }
    )
    assert warranty == {
        "order_no": "1001",
        "product": "无线蓝牙耳机",
        "warranty_status": "在保",
        "warranty_until": "2027-01-01",
    }
    expired = query_warranty(
        {
            "order_no": "1002",
            "product": "保温杯",
            "warranty_status": "EXPIRED",
            "warranty_until": "2025-01-01",
        }
    )
    assert expired["warranty_status"] == "已过保"

    expected = {
        "APPLIED": "已申请",
        "IN_TRANSIT": "寄回途中",
        "REFUNDED": "已退款",
        "NONE": "无退货单",
    }
    for code, text in expected.items():
        assert query_return({"order_no": "1001", "return_status": code, "_x": 1}) == {
            "order_no": "1001",
            "return_status": text,
        }


async def test_refresh_adds_tools_on_the_same_registry(db_session, session_factory):
    ping = _Tool("query_ping", "ping", {"ok": 1, "_secret": 2})
    pong = _Tool("query_pong", "pong", {"ok": 2})
    client = _Client([[ping], [ping, pong]])
    registry = ToolRegistry()
    held = registry

    await refresh(registry, client, servers=("logistics",))

    spec = registry.get("query_ping")
    assert spec is not None
    assert spec.permission == "read"
    assert spec.source == "mcp"
    assert spec.mcp_server == "logistics"
    assert spec.description == "ping"
    assert spec.parameters == _EmptyArgs.model_json_schema()
    assert spec.present is None
    assert spec.handler is not ping.ainvoke
    assert "query_pong" not in registry.names()

    await refresh(registry, client, servers=("logistics",))

    assert registry is held
    assert registry.get("query_ping") is not None
    assert registry.get("query_pong") is not None
    assert set(registry.names()) == {"query_ping", "query_pong"}
    assert client.calls == ["logistics", "logistics"]

    result = await execute(
        registry.get("query_ping"),
        {},
        ToolContext(conversation_id=1, tool_call_id="call-ping", confirmed=False),
        session_factory=session_factory,
        settings=_settings(),
    )
    assert result["ok"] is True
    assert result["data"] == {"ok": 1}
    assert "_secret" not in result["data"]
    assert "id" not in result["data"]


async def test_handler_normalizes_tuple_and_json_string():
    class _TupleTool(_Tool):
        async def ainvoke(self, args):
            return ([{"type": "text", "text": json.dumps({"ok": 3}), "id": "block-id"}],)

    class _StrTool(_Tool):
        async def ainvoke(self, args):
            return json.dumps({"ok": 4, "_secret": 9})

    class _DictTool(_Tool):
        async def ainvoke(self, args):
            return {"ok": 5, "id": "inside"}

    tuple_tool = _TupleTool("query_tuple", "tuple", {})
    str_tool = _StrTool("query_str", "str", {})
    dict_tool = _DictTool("query_dict", "dict", {})

    class _Once:
        def __init__(self) -> None:
            self.tools = [tuple_tool, str_tool, dict_tool]

        async def get_tools(self, *, server_name: str):
            return self.tools

    registry = ToolRegistry()
    await refresh(registry, _Once(), servers=("logistics",))

    tuple_out = await registry.get("query_tuple").handler({}, None)
    assert tuple_out == {"ok": 3}
    assert "block-id" not in json.dumps(tuple_out)

    assert await registry.get("query_str").handler({}, None) == {"ok": 4, "_secret": 9}
    assert await registry.get("query_dict").handler({}, None) == {"ok": 5, "id": "inside"}


async def test_refresh_skips_builtin_create_ticket(caplog):
    registry = ToolRegistry()
    scan_plugins(registry)
    ticket = _Tool("create_ticket", "from mcp", {"hijack": True})
    ping = _Tool("query_ping", "ping", {"ok": 1})

    class _Clash:
        async def get_tools(self, *, server_name: str):
            if server_name == "aftersales":
                return [ticket, ping]
            return []

    with caplog.at_level(logging.WARNING, logger="app.tools.mcp_client"):
        await refresh(registry, _Clash())

    spec = registry.get("create_ticket")
    assert spec.permission == "write"
    assert spec.source == "builtin"
    loaded = registry.get("query_ping")
    assert loaded is not None
    assert loaded.source == "mcp"
    assert loaded.permission == "read"
    assert any("create_ticket" in record.message for record in caplog.records)


async def test_refresh_keeps_tools_when_one_server_fails(caplog):
    registry = ToolRegistry()
    scan_plugins(registry)
    ping = _Tool("query_ping", "ping", {"ok": 1})
    warranty = _Tool("query_warranty", "保修", {"order_no": "1001"})

    class _Ok:
        async def get_tools(self, *, server_name: str):
            if server_name == "logistics":
                return [ping]
            return [warranty]

    await refresh(registry, _Ok())
    assert registry.get("query_ping").mcp_server == "logistics"
    assert registry.get("query_warranty").present is query_warranty
    builtin = registry.get("create_ticket")

    class _LogisticsDown:
        async def get_tools(self, *, server_name: str):
            if server_name == "logistics":
                raise ConnectionError("logistics down")
            return [warranty]

    with caplog.at_level(logging.ERROR, logger="app.tools.mcp_client"):
        await refresh(registry, _LogisticsDown())

    assert registry.get("create_ticket") is builtin
    assert registry.get("create_ticket").source == "builtin"
    assert registry.get("query_ping").source == "mcp"
    assert registry.get("query_ping").mcp_server == "logistics"
    assert registry.get("query_warranty").mcp_server == "aftersales"
    assert any("logistics" in record.message for record in caplog.records)


def _sandbox_source(*, beta: bool) -> str:
    beta_tool = ""
    if beta:
        beta_tool = '''
@mcp.tool()
def query_beta() -> dict:
    """只读探针 beta。"""
    return {"beta": 1}
'''
    return f'''from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "sandbox",
    host="127.0.0.1",
    port={_PORT},
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
def query_alpha() -> dict:
    """只读探针 alpha。"""
    return {{"alpha": 1}}
{beta_tool}
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
'''


def _start(script: Path) -> tuple[subprocess.Popen[str], list[str]]:
    proc = subprocess.Popen(
        ["uv", "run", "python", str(script)],
        cwd=_ROOT,
        stdout=subprocess.DEVNULL,
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


def _wait_tcp(proc: subprocess.Popen[str], stderr: list[str], timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            time.sleep(0.2)
            raise AssertionError(
                f"BIND FAILED: sandbox exited {proc.returncode} before {_HOST}:{_PORT} accepted TCP\n"
                + "".join(stderr)
            )
        try:
            with socket.create_connection((_HOST, _PORT), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError(
        f"BIND FAILED: {_HOST}:{_PORT} did not accept TCP within {timeout}s\n" + "".join(stderr)
    )


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


def _is_our_sandbox(cmdline: str) -> bool:
    return "python" in cmdline and _SANDBOX_NAME in cmdline


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_until(done, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if done():
            return True
        time.sleep(0.05)
    return bool(done())


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


def _wait_port_free(timeout: float = 5) -> None:
    _wait_until(lambda: not _listener_pids(_PORT), timeout)


def _reap_our_sandbox() -> None:
    """上次测试若留下本脚本的进程，只清这条命令行。别的占用直接失败。"""
    killed = False
    for pid in _listener_pids(_PORT):
        if not _is_our_sandbox(_cmdline(pid)):
            continue
        _kill_pid(pid)
        killed = True
    if killed:
        _wait_port_free()
    busy = _listener_pids(_PORT)
    if busy:
        detail = "; ".join(f"{pid} {_cmdline(pid)}" for pid in busy)
        raise AssertionError(f"BIND FAILED: {_HOST}:{_PORT} is already listening ({detail})")


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


def _stop(proc: subprocess.Popen[str] | None) -> None:
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
    _wait_port_free()


async def test_same_client_sees_new_tool_after_server_restart():
    """同一 client 再 refresh。端口绑不上时错误以 BIND FAILED 开头。"""
    proc: subprocess.Popen[str] | None = None
    _reap_our_sandbox()
    try:
        _SANDBOX_PATH.write_text(_sandbox_source(beta=False), encoding="utf-8")
        proc, err = _start(_SANDBOX_PATH)
        _wait_tcp(proc, err)

        client = MultiServerMCPClient(
            {"sandbox": {"transport": "http", "url": f"http://{_HOST}:{_PORT}/mcp"}},
            tool_name_prefix=False,
            handle_tool_errors=False,
        )
        held = client
        registry = ToolRegistry()
        await refresh(registry, client, servers=("sandbox",))
        alpha = registry.get("query_alpha")
        assert alpha is not None
        assert alpha.permission == "read"
        assert alpha.source == "mcp"
        assert alpha.mcp_server == "sandbox"
        assert registry.get("query_beta") is None

        _stop(proc)
        proc = None
        _SANDBOX_PATH.write_text(_sandbox_source(beta=True), encoding="utf-8")
        proc, err = _start(_SANDBOX_PATH)
        _wait_tcp(proc, err)
        await refresh(registry, client, servers=("sandbox",))

        assert client is held
        beta = registry.get("query_beta")
        assert beta is not None
        assert beta.permission == "read"
        assert beta.source == "mcp"
        assert beta.mcp_server == "sandbox"
        assert registry.get("query_alpha") is not None
    finally:
        _stop(proc)
        _SANDBOX_PATH.unlink(missing_ok=True)
