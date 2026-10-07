# Ch05 实施计划：Workflow 编排（LangGraph 骨架 + 祛魅循环 + ReAct 主力 Agent）

> **For implementer:** Use TDD throughout. Write failing test first. Watch it fail. Then implement.

**Goal:** LangGraph 确定性图做骨架（透传→意图→分流→检索→置信度闸→ReAct Agent→日志），主力 Agent 复用 ch02 工具，前端加转人工/建工单。

**Architecture:** bare_loop.py（祛魅对照）→ workflow.py（主图：State/节点/条件边）→ agent.py（create_react_agent 子图）→ api/chat.py 切图（SSE 事件适配）→ /api/tickets 端点 → 前端。

**Tech Stack:** langgraph 1.2.12（已随 langchain 1.4.3 安装，需显式声明依赖）· InMemorySaver · create_react_agent · ToolNode

**Context7 + 本地内省核对结论（2026-10-07）：**
1. `langgraph.checkpoint.memory.InMemorySaver` —— **1.x 把 MemorySaver 改名 InMemorySaver**（设计稿措辞跟改）
2. `langgraph.prebuilt.create_react_agent(model, tools, *, prompt=...)` 可用（model 可为 str 或 Runnable）；返回**编译图**，作为父图节点即成子图；父图 checkpointer 自动传播到子图
3. ⚠️ **父图 stream_mode="messages" 默认不冒出子图内 LLM token，必须 `subgraphs=True`**（SSE delta 的生命线）
4. `StateGraph` / `add_node` / `add_conditional_edges` / `START` / `END`；`recursion_limit` 是**顶层 config 键**（非 configurable）
5. 流模式：`astream(..., stream_mode=["messages", "updates", "custom"], subgraphs=True)`；节点内自定义事件用 `get_stream_writer()`（custom 模式，options 帧走这里）

**约定：** 单测 Fake 模型注入；每任务绿后 commit；意图识别/固定话术文案属 Prompt 类（样例验证替代 TDD）。

---

### Task 1: 依赖声明与图 API 冒烟

**Files:** Modify: `pyproject.toml`（uv add langgraph）
**Step 1:** `uv add langgraph`（显式声明，当前 1.2.12 为传递依赖）
**Step 2:** 冒烟：
```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent, ToolNode
print("langgraph APIs ✓")
```
**Step 3:** **Commit** `ch05: langgraph 显式依赖与图 API 冒烟`

---

### Task 2: 祛魅裸循环 bare_loop.py（TDD）

**Files:** Create: `app/agents/bare_loop.py`；Test: `tests/agents/test_bare_loop.py`

**Step 1: 失败测试**（httpx MockTransport 脚本化 OpenAI 协议响应）
```python
def make_loop(responses):
    """responses: 每轮 assistant 消息的 content/tool_calls 脚本。"""
    ...  # MockTransport: 第 N 次 POST /chat/completions 返回 responses[N-1]


async def test_one_tool_call_then_converge():
    # 第1轮: tool_calls=[get_order(1001)]；第2轮: content="订单已发货"
    # 工具执行结果作为 {"role":"tool","tool_call_id":...} 喂回
    out = await run_bare_loop(client=..., tools=[query_order], messages=[{"role":"user","content":"订单1001呢"}], max_steps=5)
    assert out == "订单已发货"


async def test_multi_step_two_tools():
    # 第1轮查订单、第2轮查物流、第3轮收敛 → 三次 LLM 调用、两次工具执行


async def test_max_steps_stops():
    # 每轮都返回 tool_calls → 跑满 max_steps 后返回最后文本并告警，不死循环


async def test_no_tool_direct_answer():
    # 首轮无 tool_calls → 一次调用收敛
```
**Step 2:** FAIL → **Step 3: 实现**：`run_bare_loop(client, model, tools, messages, tool_executor, max_steps)`——纯 OpenAI 协议手写：`tools` 以 JSON Schema 塞进请求；`tool_calls` 非空→`tool_executor(name, args)` 执行→结果按 `{"role":"tool", "tool_call_id", "content"}` 追加→循环；空→返回 content。教学注释逐行
**Step 4:** PASS → **Commit** `ch05: 祛魅裸 Agent 循环（手写 OpenAI 协议，max_steps 停止）`

---

### Task 3: 意图识别节点（TDD）

**Files:** Create: `app/workflow/intent.py`；Test: `tests/workflow/test_intent.py`

**Step 1: 失败测试**（MockTransport / FakeLLM）
```python
INTENTS = {"物流","订单","商品咨询","退款退货","售后","投诉","闲聊"}

async def test_seven_intents():          # 7 组样例输入→对应 intent（真实 LLM，eval 标记）
async def test_parse_failure_fallback(): # mock 返回非法 JSON → intent="订单"（业务数据类兜底）
async def test_illegal_value_fallback(): # mock 返回 intent="跳舞" → 兜底
```
**Step 2:** FAIL → **Step 3: 实现**：`INTENT_PROMPT`（七类定义+判例）+ `build_intent_classifier(llm)`（build_structured_model 复用，schema={intent: Literal[七类]}，include_raw 处理，失败兜底 "订单"）
**Step 4:** PASS → **Commit**

---

### Task 4: 图骨架 workflow.py（TDD 主战场）

**Files:** Create: `app/workflow/__init__.py`、`app/workflow/state.py`、`app/workflow/graph.py`；Test: `tests/workflow/test_graph.py`

**Step 1: State**（state.py）：
```python
class WorkflowState(TypedDict):
    query: str
    intent: str
    messages: list           # 注入 Agent 的消息（system+历史+知识条目）
    evidence: list[dict]     # citations 全集快照（ch04 语义）
    options: list[str]       # 投诉路径的可选项
    refusal: bool            # gate 拦截标记
    final_text: str          # comfort/chitchat/gate 兜底的固定文本
```

**Step 2: 失败测试**（FakeLLM/FakeRetriever/FakeAgent 注入；真库 MySQL）
```python
async def test_knowledge_route_hits_retrieve_then_gate(db_session): ...
    # intent="商品咨询" → retrieve 被调（FakeRetriever 记录）→ 证据够 → agent 节点被调
async def test_gate_weak_evidence_blocks_agent(db_session): ...
    # FakeRetriever low_confidence=True → agent 未被调、final_text=兜底话术、low_confidence_questions 落行
async def test_business_route_skips_retrieval(db_session): ...
    # intent="物流" → 检索器未被调、agent 被调
async def test_complaint_produces_options(db_session): ...
    # intent="投诉" → final_text=安抚话术、options=["转人工","建工单"]、agent 未被调
async def test_chitchat_fixed_text_zero_llm(db_session): ...
    # intent="闲聊" → 固定话术、 FakeLLM/agent 均未被调
async def test_intent_fallback_routes_to_business(db_session): ...
    # FakeLLM 抛异常 → 兜底"订单"→ 走业务路
async def test_log_node_persists(db_session): ...
    # 图跑完 messages 落 conversations/messages（复用 MessagesRepo）
```
**Step 3:** FAIL → **Step 4: 实现**：
- `build_workflow(retriever, agent_node, intent_classifier, low_repo, msg_repo, settings) -> CompiledGraph`
- 节点：`resolve`(透传) / `intent` / `route`(写死映射：{"物流","订单","售后"}→business；{"商品咨询","退款退货"}→knowledge；"投诉"→complaint；"闲聊"→chitchat) / `retrieve` / `gate`+条件边 / `agent`(注入的子图包装) / `comfort` / `chitchat` / `log`
- agent 未执行的路径 final_text 直接产答案；agent 路径由子图产出 messages
- 编译 `checkpointer=InMemorySaver()`
**Step 5:** PASS → **Commit** `ch05: 图骨架（七类分流/置信度闸/固定话术/日志节点）`

---

### Task 5: ReAct Agent 子图（create_react_agent 接线，TDD）

**Files:** Create: `app/workflow/agent_node.py`；Modify: `app/workflow/graph.py`；Test: `tests/workflow/test_agent_node.py`

**Step 1: 失败测试**（Fake 工具执行记录 + 真实 create_react_agent + Fake 模型 bind_tools 脚本）
```python
async def test_react_multi_step():
    # Fake 模型脚本：第1轮 tool_calls=[query_order]→第2轮 tool_calls=[query_logistics]→第3轮收敛
    # 断言：两次工具执行、最终文本、recursion_limit 生效
async def test_knowledge_injection():
    # 知识类路径：state.messages 含 [n] 编号知识条目消息
```
**Step 2:** FAIL → **Step 3: 实现**：`build_agent_node(model, tools, settings)`：`create_react_agent(model.bind_tools(tools), tools, prompt=客服 System Prompt+负面知识)` 编译为子图；包装函数 `agent_node(state)`：注入 system+历史+知识条目 → invoke 子图（config recursion_limit=settings.agent_max_steps, thread_id=cid:turn）→ 返回 {"final_text", "messages"}；工具执行经 ToolNode（ch02 工具是 async @tool——ToolNode 原生支持）
**Step 4:** PASS → **Commit** `ch05: ReAct Agent 子图（create_react_agent+知识注入+步数上限）`

---

### Task 6: SSE 适配与 api/chat.py 切图

**Files:** Modify: `app/api/chat.py`、`app/main.py`；Delete: `app/agents/orchestrator.py`（等价行为已由图覆盖）+ `tests/api/test_chat_stream.py` 改造；Test: `tests/api/test_chat_stream.py`

**Step 1: 失败测试**（Fake 图/真实图 + Fake 模型，保持既有 SSE 契约）
```python
def test_sse_contract_unchanged(...): ...        # meta/tool/delta/done 序列不变
def test_options_frame_for_complaint(...): ...   # 投诉路径 → event: options data={options:[...]}
def test_done_carries_citations(...): ...        # 知识类 done 带 citations
def test_subgraph_tokens_streamed(...): ...      # astream(subgraphs=True) 冒出子图 token
```
**Step 2:** FAIL → **Step 3: 实现**：`api/chat.py` 组装图输入（MySQL 加载历史+新消息）→ `graph.astream(..., stream_mode=["messages","updates","custom"], subgraphs=True, config={recursion_limit, thread_id})` → 事件映射 SSE（messages→delta；updates 里 agent 工具调用→tool 帧；custom→options 帧；结束→done+citations）；`main.py` 装配图；删 orchestrator.py
**Step 4:** PASS（全量）→ **Commit** `ch05: api/chat 切 LangGraph 图（subgraphs 流式/options 帧/citations）`

---

### Task 7: /api/tickets 端点（TDD）

**Files:** Create: `app/api/tickets.py`；Test: `tests/api/test_tickets.py`
```python
async def test_create_ticket_endpoint(db_session): ...
    # POST /api/tickets {conversation_id, description, ticket_type} → 直写 tickets 表（不调 LLM）
def test_ticket_invalid_type_422(): ...
```
**Step 4:** PASS → **Commit** `ch05: /api/tickets 直写端点`

---

### Task 8: 前端 options 按钮（Vibe）

**Files:** Modify: `app/static/index.html`
options 帧渲染「转人工」「建工单」两独立按钮；转人工=前端模拟（「已转接人工客服」+客服小猫问候）；建工单=调 /api/tickets 后显示工单号；互不绑定。

**Commit** `ch05: 前端转人工/建工单按钮与交互`

---

### Task 9: 五条验收真机验证

1. 政策问题 → 日志断言 retrieve 节点被走到
2. 「订单 1001 的物流」→ Agent 自调工具
3. 「我要投诉」→ 两按钮；转人工模拟；建工单写表
4. 闲聊固定话术
5. 「帮我查订单 1001 然后看它物流到哪了」→ ReAct 多步
结果记 dev-notes → **Commit**

---

### Task 10: 阶段评审 + 完结

评审子代理 → 修复 → 完结留痕 → git push

## 执行方式

沿用「主会话顺序执行 + 阶段评审子代理」。
