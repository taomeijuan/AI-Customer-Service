# ch08 实施计划：即插即用工具系统

> **For implementer:** 引擎、注册表、校验、权限、重试、审计、MCP 接线走 TDD：先写失败测试，看它失败，再实现。提示词与工具选型用标注样例，不写单测冒充。工单预览卡片是 Vibe，不进本计划的 TDD 任务。

**Goal:** 把写死的内置工具换成统一工具表和执行引擎，并接入物流、售后两个 MCP Server；`create_ticket` 必须等前端确认才落库。

**Architecture:** 每轮用 `MultiServerMCPClient.get_tools()` 现拉 MCP 工具，与 `app/tools/plugins/` 的内置工具合成一张表。模型看到的是薄外壳，调用进 `execute()`。建工单外壳在真正写入前 `interrupt`。因为主力 Agent 是包在父节点里 `ainvoke` 的独立子图，子图中断要桥接成父图 `interrupt`，聊天接口才能沿用订单选择器的 `Command(resume)`。

**Tech Stack:** 现有 FastAPI / SQLAlchemy / LangGraph `create_react_agent`；新增 `mcp`（官方 SDK 1.x 的 `FastMCP`，用户 2026-10-15 拍板 A）、`langchain-mcp-adapters`（`MultiServerMCPClient`）、`jsonschema`。

设计：`docs/plans/2026-10-12-ch08-tools-design.md`。审计 DDL 以该文档第 5.4 节用户原文为准，不改列。

---

## Context7 核对（2026-10-12）

本会话没有 Context7 MCP 工具。按 Context7 文档的同一套接口查的：`GET /api/v2/libs/search`、`GET /api/v2/context`。库 ID：`/langchain-ai/langchain-mcp-adapters`、`/websites/py_sdk_modelcontextprotocol_io`、`/websites/langchain_oss_python_langgraph`。

**MCP Server（2026-10-15 用户拍板 A）：** 适配器 `langchain-mcp-adapters` 0.3.2 要求 `mcp<2`，解出的是官方 SDK 1.30。v2 的 `MCPServer` 和这个适配器不能一起装。Server 用 1.30 的 `FastMCP`，传输仍是 Streamable HTTP。Context7 上 v1.12.4 的写法与已安装的 1.30 源码一致：`stateless_http`、`json_response`、`host`、`port` 写在构造函数上，`run()` 只收 `transport`。

```python
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
    return {...}

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

- 不要把 `host` / `port` / `stateless_http` / `json_response` 传给 `run()`。1.30 的 `run()` 不收这些参数。
- 不要改用 SDK 2 的 `MCPServer`，也不要改成 stdio。
- 工具返回 `dict` 即可。`stateless_http=True` 与「每轮新拉」一致。这些工具不回调客户端，`json_response=True` 可用。

**MCP Client：**

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "logistics": {"transport": "http", "url": "http://127.0.0.1:8765/mcp"},
        "aftersales": {"transport": "http", "url": "http://127.0.0.1:8766/mcp"},
    },
    tool_name_prefix=False,   # 保持 query_logistics 这个名字，不要变成 logistics_query_logistics
    handle_tool_errors=False,  # 执行错误抛 ToolException，交给引擎分类；传输失败本来就会抛
)
tools = await client.get_tools()
logistics_tools = await client.get_tools(server_name="logistics")
```

`transport` 接受 `http`、`streamable_http`、`streamable-http`。计划用文档示例里的 `"http"`。`get_tools()` 每次现拉，不把进程启动时的清单缓存到死。

**interrupt：** 官方文档允许把 `interrupt()` 写在工具函数里。恢复时用同一 `thread_id`、同一 checkpointer，`Command(resume=值)`，`interrupt()` 的返回值就是这份 resume。`create_react_agent(..., version="v2")` 让每个工具调用单独一个 ToolNode，恢复时不会把同轮已成功的工具再跑一遍。

**和现有代码的接缝（不是换方案）：** `agent_node` 里对 react 子图是 `ainvoke`，子图自己的 checkpointer，每轮一个新 `thread_id`。父图的 `Command(resume)` 不会自动钻进这次 `ainvoke`。做法：子图 `ainvoke` 的结果里若有挂起中断，先把这个子图实例和 `thread_id` 按会话记下，再在父节点调用 `interrupt(同一份 ticket_preview)`。父图恢复时节点会从头跑，第一件事读挂起记录并 `interrupt()` 取回 `{action: confirm|cancel}`，然后对**同一个**子图实例 `ainvoke(Command(resume=...))`。不要在恢复路径上重建子图或重新向模型要一次工具调用。父节点现有的 `except Exception` 不得包住这次 `interrupt()`（`GraphInterrupt` 要原样冒出去）。

T8 先做一支探针：假工具里 `interrupt`，经 `agent_node` 这条 `ainvoke` 看返回值里中断长什么样（`__interrupt__` 或 `.interrupts`）。探针结果写进 `dev-notes/ch08.md` 再接线。探针如果说明子图中断根本出不来，停下来问，不改成手写 ReAct 循环。

---

## 任务

| # | 任务 | 验证 |
|---|---|---|
| T1 | `db/init/10-ch08.sql` 用设计文档原文；`scripts/init_ch08.sh` 双库执行；ORM `ToolAuditLog` 无外键；`conftest` 的 TRUNCATE 加上 `tool_audit_logs` | 单测：模型上没有 ForeignKey；插入一行再读回中文状态 |
| T2 | `app/tools/spec.py` + `app/tools/validate.py`：JSON Schema 校验、`LookupMiss` | 单测：缺字段 / 类型 / 枚举 / 多余字段 |
| T3 | `app/tools/engine.py`：超时、只读才重试、写操作重试 0、三类结果、格式化、审计失败不拦执行 | 单测 |
| T4 | `app/tools/mock_data.py` 抽出确定性种子；插件 `query_order` / `query_product`；内置不再登记 `query_logistics` | 单测：同订单号轨迹一致；扫描只注册新文件 |
| T5 | 插件 `create_ticket`：无确认标记不写库；确认后写 `tickets` | 单测 |
| T6 | `mcp_servers/logistics_server.py`、`mcp_servers/aftersales_server.py`；依赖 `mcp`、`langchain-mcp-adapters`、`jsonschema` | 先探针 `get_tools()`。导入或连不上就停 |
| T7 | `app/tools/mcp_client.py`：按 Server 现拉，权限一律只读，同名内置工具丢弃 | 单测用假 `get_tools`；再加一个真进程刷新测试 |
| T8 | 薄外壳、每轮重建 react、建工单 `interrupt`、父图桥接；`chat.py` 发 `ticket_preview` | 探针 + 图测试：确认落库、取消为权限拒绝 |
| T9 | 提示词改为缺信息先追问；标注样例四条 | `pytest -m eval`，不执行工具 |
| T10 | `main.py` 改走注册表；`TOOL_DELAY_*`；演示命令写入 `scripts/acceptance.sh` 的 ch08 段 | 全量 `pytest` 不带 eval |

依赖：T1 → T3；T2 → T3 → T5；T4 可与 T1 并行；T6 → T7 → T8；T9 可在 T8 之后；T10 收尾。卡片前端在 T8 契约测通后单独做，不占用 TDD 任务号。

---

### T1 审计表

**Files:**
- Create: `db/init/10-ch08.sql`（设计文档 5.4 节原文，一字不改）
- Create: `scripts/init_ch08.sh`（仿 `scripts/init_ch07.sh`，只灌 `10-ch08.sql`，`--default-character-set=utf8mb4`）
- Modify: `app/db/models.py`、`tests/conftest.py`
- Test: `tests/db/test_audit_model.py`

ORM 列与 DDL 对齐：`tool_source` 枚举 `builtin`/`mcp`，`mcp_server` 可空，`status` 五个中文值，`retry_count` 默认 0，`duration_ms` 可空。`conversation_id` 用 `BigInteger`，**不要** `ForeignKey`。

测试：`ToolAuditLog.__table__.foreign_keys` 为空；插入 `status="校验拦下"` 后查回仍是这个字符串。TRUNCATE 列表加入 `tool_audit_logs`，放在 `messages` 附近，不依赖外键顺序。

先在本机跑 `scripts/init_ch08.sh`。库不在就停，不要改 DDL 去迁就。

### T2 校验

**Files:**
- Create: `app/tools/spec.py`、`app/tools/validate.py`、`app/tools/errors.py`
- Test: `tests/tools/test_validate.py`

```python
@dataclass(frozen=True)
class ToolContext:
    conversation_id: int | None
    tool_call_id: str | None
    confirmed: bool = False

class LookupMiss(Exception):
    """查询落空。引擎不重试。"""
```

`validate_args(schema, args) -> str | None`：用 `jsonschema.Draft202012Validator`。失败返回一句中文，`参数不合法：…`，指出字段名。覆盖：必填缺失、类型不对、`enum` 越界、`minLength`、`additionalProperties: false`。成功返回 `None`。不要抛给调用方。

### T3 执行引擎

**Files:**
- Create: `app/tools/engine.py`、`app/repositories/audit.py`
- Modify: `app/core/config.py`（`tool_delay_tools: str = ""`，`tool_delay_seconds: float = 0`）
- Test: `tests/tools/test_engine.py`

`execute(spec, args, ctx, *, session_factory, settings) -> dict` 返回给模型的 JSON 对象，再由外壳 `json.dumps(..., ensure_ascii=False)`。

顺序：未注册 → 校验 → 写操作且 `ctx.confirmed` 为假 → 调用 → 格式化 → 审计。

给模型的形状：

- 成功：`{"ok": true, "data": <格式化结果>}`
- 失败：`{"ok": false, "error_type": "invalid_args"|"not_found"|"fault", "error": "<人话>"}`

重试只针对只读，且异常是 `asyncio.TimeoutError`、`TimeoutError`、`ConnectionError`，或类型名里带 `ConnectError` / `ReadError` / `RemoteProtocolError` / `NetworkError` / `PoolTimeout` 的传输错误。`LookupMiss`、返回值 `{"found": false}`、其他异常：不重试。写操作 `retries = 0`。`retry_count` 只记额外次数。

`TOOL_DELAY_TOOLS` 命中时，在 `wait_for` 内部先 `asyncio.sleep(TOOL_DELAY_SECONDS)` 再调用。延迟本身造成的超时走上面的重试规则。

格式化：`spec.present` 存在则用它；否则去掉键名以 `_` 开头的字段。枚举翻译放在各工具的 `present` 里，不进引擎通用逻辑。`result_summary` 为格式化 JSON，超过 2000 字截断。

审计用 `session_factory()` 新开的会话，`commit`。任何异常 `logger.exception` 后吞掉。列名：`tool_source`、`mcp_server`、`error_message`、`duration_ms`、`retry_count`。

测试要钉死：只读超时 `retry_count==1` 且状态 `超时`；写操作超时 `retry_count==0`；`found: false` 不重试且状态 `成功`、摘要含「未查到」；校验拦下与未确认写操作都不调用 handler；审计仓库抛错时返回值仍在。

### T4 内置插件与假数据

**Files:**
- Create: `app/tools/mock_data.py`、`app/tools/plugins/order.py`、`app/tools/plugins/product.py`、`app/tools/registry.py`
- Modify: `app/tools/ecommerce.py`（改为调用 `mock_data`，避免两套种子）
- Test: `tests/tools/test_registry.py`、既有 `tests/tools/test_ecommerce_determinism.py` 继续绿

`registry.scan_plugins(registry)` 导入 `app.tools.plugins` 里每个模块并调用 `register(registry)`。测试可以再往一个临时包注册，断言新名字出现，且不需要改 `engine.py`。

`query_logistics` 不在插件里。`mock_data.logistics(order_no)` 行为与现在 `ecommerce.query_logistics` 一致，供物流 MCP 使用。订单、商品同样搬到 `mock_data`，插件与旧 `@tool` 都调它，旧测试不用改期望值。

### T5 create_ticket 插件

**Files:**
- Create: `app/tools/plugins/ticket.py`
- Test: `tests/tools/test_ticket_gate.py`

Schema：`description` 字符串 minLength 1 maxLength 500，`ticket_type` 枚举售后/投诉/咨询，`additionalProperties: false`。`permission="write"`，`source="builtin"`。handler 只在 `ctx.confirmed` 时调用 `TicketsRepo.create`。引擎已挡住未确认调用；测试两边都断言 `tickets` 行数不变。确认路径断言工单号形如 `T` + 日期 + 序号。`conversation_id` 来自 `ctx`，不在 Schema 里。

本任务不调用 `interrupt`。中断在 T8。

### T6 两个 MCP Server

**Files:**
- Create: `mcp_servers/logistics_server.py`、`mcp_servers/aftersales_server.py`
- Modify: `pyproject.toml`（`uv add mcp langchain-mcp-adapters jsonschema`）
- Test: `tests/tools/test_mcp_probe.py`

物流工具 `query_logistics(order_no: str) -> dict`，体内调用 `app.tools.mock_data.logistics`。售后：

- `query_warranty` 返回 `order_no`、`product`、`warranty_status`（`IN_WARRANTY` 或 `EXPIRED`）、`warranty_until`
- `query_return` 返回 `order_no`、`return_status`（`APPLIED` / `IN_TRANSIT` / `REFUNDED` / `NONE`）

种子用 `mock_data` 的 `_seeded_rng(order_no, "warranty"|"return")`，同一订单号结果不变。不建表、不访问 MySQL。

探针：子进程启动两个脚本，`MultiServerMCPClient.get_tools(server_name=...)` 能看到上述三个名字，且物流只有一个 `query_logistics`。`ainvoke({"order_no": "1001"})` 连续两次结果相同。失败则停，把异常原文记进 dev-notes。Server 用上面的 `FastMCP`，不要改成 stdio，也不要升到 SDK 2。

端口：物流 8765，售后 8766，路径 `/mcp`。

### T7 现拉进工具表

**Files:**
- Create: `app/tools/mcp_client.py`、`app/tools/present.py`
- Modify: `app/core/config.py`（`mcp_logistics_url`、`mcp_aftersales_url`）
- Test: `tests/tools/test_mcp_refresh.py`

`refresh(registry, client)`：对每个已配置 Server 调用 `get_tools(server_name=...)`。每件工具变成 `ToolSpec`：`permission="read"`，`source="mcp"`，`mcp_server` 为该 Server 名。JSON Schema 取 LangChain 工具的 `args_schema.model_json_schema()`。handler 为 `tool.ainvoke`。名字与内置冲突则跳过并 `logger.warning`。某 Server 连不上：日志后跳过，不清空其他 Server 已拉到的工具，也不清内置工具。

`present.py` 为 `query_warranty` / `query_return` / `query_logistics` 提供字段挑选和枚举人话。新工具没有 present 时走引擎默认（去掉 `_` 字段）。

测试 1：假 client 第一次返回一个工具，第二次多一个，registry 在不重建进程的情况下多出这个名字。测试 2：假工具名叫 `create_ticket`，刷新后表里仍是内置那条，`permission` 仍是 `write`。测试 3（依赖 T6）：真 Server 进程重启并在代码里多一个只读工具后，同一 client 再 `get_tools`，新名字出现。这一条标 `pytest.mark.timeout` 或独立文件，避免没起进程时拖垮全量。

### T8 外壳、中断桥、SSE

**Files:**
- Create: `app/tools/wrappers.py`
- Modify: `app/workflow/agent_node.py`、`app/api/chat.py`、`app/workflow/text.py`（售后两个工具的引用文案）
- Test: `tests/workflow/test_ticket_interrupt.py`、`tests/api/test_chat_stream.py` 增加 `ticket_preview` 用例

外壳：`specs_to_tools(specs, ctx_factory)` 产出 LangChain `@tool`，`args_schema` 由 JSON Schema 转成的 Pydantic 模型（或 `StructuredTool`）。协程体：

1. `create_ticket`：先 `validate_args`。失败则 `engine.execute`（会记校验拦下）并返回，**不要** `interrupt`。
2. 通过后 `decision = interrupt({"type": "ticket_preview", "ticket_type": ..., "description": ...})`。
3. `action=="confirm"` 时 `ctx.confirmed=True` 再 `engine.execute`。否则 `confirmed=False`，引擎记权限拒绝，回灌「用户取消了建单」。
4. 其他工具直接 `engine.execute`，`confirmed=False`。

`build_agent_node` 每轮调用前刷新 MCP 并 `create_react_agent(..., version="v2", checkpointer=进程内同一个 InMemorySaver)`。checkpointer 不能每轮新建，否则恢复找不到子图状态。

桥接用进程内字典 `paused_agents[conversation_id]`。伪代码：

```python
saved = paused_agents.get(cid)
if saved:
    decision = interrupt(saved["payload"])  # 恢复时直接返回 resume 值
    result = await saved["agent"].ainvoke(
        Command(resume=decision),
        config={"configurable": {"thread_id": saved["thread_id"]}, "recursion_limit": settings.max_agent_steps},
    )
    paused_agents.pop(cid, None)
else:
    result = await react.ainvoke(...)
    intr = _first_interrupt(result)
    if intr and intr.get("type") == "ticket_preview":
        paused_agents[cid] = {"agent": react, "thread_id": thread_id, "payload": intr}
        interrupt(intr)  # 让父图停下；不要包在 except Exception 里
```

`_first_interrupt` 由 T8 探针填实：兼容 `result["__interrupt__"][0].value` 和 `result.interrupts[0].value`。

`chat.py` 在现有 `__interrupt__` 分支旁加：`type=="ticket_preview"` 时 `emit("ticket_preview", {"ticket_type", "description"})`。`resume` 体仍是请求里的 JSON，确认 `{"action":"confirm"}`，取消 `{"action":"cancel"}`。不要新开一条恢复 URL。

图测试用假模型：第一轮只调用 `create_ticket` 且参数合法 → 结果含 `ticket_preview`，`tickets` 行数不变。`Command(resume={"action":"confirm"})` 后有工单号。另一用例 resume `cancel`：行数不变，`tool_audit_logs.status=="权限拒绝"`，`retry_count==0`。

写操作超时：`TOOL_DELAY_TOOLS=create_ticket`、`TOOL_DELAY_SECONDS=5`，确认之后审计 `retry_count==0`、`status=="超时"`。只读同样延迟：`retry_count==1`。

### T9 提示词与样例

**Files:**
- Modify: `app/workflow/agent_node.py` 的 `AGENT_SYSTEM_PROMPT`、`app/prompts/templates.py` 的第 5 条、`app/tools/plugins/ticket.py` 的工具描述
- Modify: `tests/data/tool_routing_samples.jsonl`、`tests/prompts/test_templates.py` 里写死的旧句子

新规则写进提示词和工具描述：用户明确说建工单或转人工才调用；问题描述必须来自用户原话；没说工单类型就先问；缺了就追问，不要编造后调用。仅表达不满时先安抚。

样例追加：

```json
{"text": "帮我建个工单", "expected_tool": null}
{"text": "耳机没声音，帮我建个售后工单", "expected_tool": "create_ticket"}
{"text": "订单 1001 还在保吗", "expected_tool": "query_warranty"}
{"text": "订单 1001 的退货到哪一步了", "expected_tool": "query_return"}
```

原有「订单 1001 的物流到哪了」仍期望 `query_logistics`。跑 `pytest -m eval tests/test_eval_routing.py`。选型不稳就改描述后再跑，不放宽断言。

### T10 装配与验收命令

**Files:**
- Modify: `app/main.py`（删掉写死的 `query_logistics` 列表；启动时 `scan_plugins`；`build_agent_node` 收到 registry、mcp client、session_factory）
- Modify: `scripts/acceptance.sh` 或新建 `scripts/ch08_demo.sh`

演示命令写进脚本注释和 dev-notes，至少包括：

```bash
uv run python mcp_servers/logistics_server.py
uv run python mcp_servers/aftersales_server.py
# 另一个终端
uv run uvicorn app.main:app --reload
```

人为超时用环境变量，不改业务代码：

```bash
TOOL_DELAY_TOOLS=query_order TOOL_DELAY_SECONDS=5 uv run uvicorn app.main:app
TOOL_DELAY_TOOLS=create_ticket TOOL_DELAY_SECONDS=5 uv run uvicorn app.main:app
```

全量：`uv run pytest -q -m "not eval"`。eval 单独跑。

---

## 不做

- Skill 机制、仓储系统、改 `POST /api/tickets`
- 工单卡片的 HTML/CSS（T8 的 SSE 契约测通后再 Vibe；浏览器里点确认和取消）

## 风险

- Server 已按用户拍板用 SDK 1.30 `FastMCP`。若 `get_tools()` 仍连不上：停，问用户。不改成 stdio，不升到 SDK 2。
- 父图桥接字典在进程重启后丢失，与现在的 `InMemorySaver` 一样，挂起的预览作废。
- 恢复路径必须复用暂停时的 react 实例。每轮刷新工具表只发生在**没有**挂起记录的新消息上。
