# Ch08 设计文档：即插即用工具系统

> 状态：已定稿（2026-10-12，用户逐节批准）
> 范围：不做 Skill 机制；不接仓储等更多外部系统。工单预览卡片的前端不走 brainstorm / TDD / code review，后端契约定稿后按效果直接做
> 流程：Superpowers（brainstorm → 计划 → TDD 构建 → 评审 → finish）
> 技术选型定死：MCP Server 用官方 Python SDK、Streamable HTTP；MCP Client 用 LangChain `langchain-mcp-adapters` 的 `MultiServerMCPClient`。实现前用 Context7 核对这两个库和 LangGraph `interrupt` 的当前接口，对不上就停下来问，不换方案

## 1. 需求与验收

1. 工具注册中心：内置与 MCP 工具一律登记工具名、用途描述、JSON Schema。内置在进程启动时登记；MCP 每轮现拉。新工具能被主力 Agent 用上，不改核心代码。
2. 执行前按 JSON Schema 校验。类型不对、必填缺失、取值越界时不抛异常，把说明包成工具结果回灌。
3. 工具分只读、写。写操作只有 `create_ticket`。客户明确要求建工单才由模型发起；真正执行要等前端确认。未确认的调用引擎拒绝。MCP 用途声明不可信，能不能调只认我们这边的权限规则。
4. 所有工具调用走同一执行引擎：超时、重试、错误处理、结果格式化。重试只给网络抖动这类暂时性故障；业务空结果不重试；写操作不自动重试。错误分参数不合法、查询落空、真故障。结果只留回答用得上的字段，内部枚举码翻成人话，JSON 中文不转义。
5. `tool_audit_logs` 每次调用落一条，含被权限拒、被校验拦的调用。表不挂外键。写审计失败不拦工具执行。
6. 自建物流、售后两个 MCP Server，独立进程，确定性假数据，不接真实系统、不建表。内置 `query_logistics` 下线，不留重名。
7. 建工单确认流：信息不够先追问，不许编造。凑齐后 `interrupt` 推出工单预览。确认才落 `tickets` 并在回复里带工单号；取消不执行，审计状态为权限拒绝。第 5 章投诉按钮 `POST /api/tickets` 保持直接落库。

验收：

1. 新写一个简单工具，只做注册、不动核心代码，Agent 能在对话里用上。
2. 两个 MCP Server 单独起进程，问物流轨迹，Agent 经 MCP 查到数据并回答。
3. 只在 MCP Server 侧加工具并只重启该 Server，客服代码不动、服务不重启，Agent 能用上新工具。
4. 「帮我建个工单」但没说清问题，Agent 先追问；随后聊天页出现预览卡片，点确认后 `tickets` 落一条，回复带工单号。
5. 同样走到预览，点取消，工单不建，审计里这条 `create_ticket` 状态为权限拒绝。
6. 人为让一个工具超时，能看到重试和错误兜底，审计里重试次数、超时状态、耗时都在；写操作超时不被自动重试，审计重试次数为 0。

## 2. 澄清决策（已拍板）

| 决策点 | 结论 |
|---|---|
| 假数据 | 确定性种子。同一订单号跨调用、跨进程结果不变。物流轨迹与现有订单假数据对齐 |
| 接入形态 | 方案 1：每轮刷新工具表，薄外壳统一进执行引擎，本轮重建 `create_react_agent`。`interrupt` 放在建工单外壳里。若官方文档证明工具函数内的 `interrupt` 无法暂停子图，停下来问，不私下改成手写循环 |
| 审计表 | 以用户提供的 DDL 为唯一权威（见第 5 节原文） |

## 3. 架构总览

每一轮对话开始时现拉工具表，再编译这一轮的主力 Agent。模型看到的每个工具都是薄外壳，调用全部进入执行引擎。

```
内置插件（进程启动时登记）──┐
                            ├─→ 工具表 ─→ 薄外壳 ─→ create_react_agent（每轮重建）
物流 / 售后 MCP（每轮现拉）──┘              │
                                           ▼
                                      执行引擎
                         校验 → 权限 → 超时/重试 → 格式化
                                           │
                           写审计（失败只记日志，不拦执行）
```

一条工具记录：`name`、`description`、`parameters`（JSON Schema）、`source`（`builtin` 或 Server 名）、`permission`（`read` / `write`，只由我们填写）、调用目标（内置函数或 MCP 远程调用）。已知工具还可挂我们自己的结果格式化规则。

- 内置工具在 `app/tools/plugins/`，每个文件提供 `register(registry)`，启动时扫描。新工具是新文件，不改 Agent、引擎、图。`query_logistics` 不在内置清单。
- 物流、售后两个 Server 在配置里标成只读。每轮用 `MultiServerMCPClient` 重新拉取；新工具立刻进表，权限一律只读。忽略 Server 自己的只读/写入声明。
- 写操作只有 `create_ticket`。
- 客服进程连不上某个 Server 时照常启动，这一轮工具表里没有它的工具。
- MCP 若带回与内置同名的工具（含 `create_ticket`），这条不进表，打日志。

`POST /api/tickets` 不经过执行引擎。

## 4. 工具表、权限、建工单确认

### 4.1 内置工具

| 工具 | 权限 | 参数 | 说明 |
|---|---|---|---|
| `query_order` | 只读 | `order_no` 必填字符串 | 沿用现在的确定性订单假数据 |
| `query_product` | 只读 | `product_name` 必填字符串 | 沿用现在的商品假数据 |
| `create_ticket` | 写 | `description`（1–500 字）、`ticket_type`（售后 / 投诉 / 咨询），都必填，`additionalProperties: false` | `conversation_id` 由会话注入，不出现在模型参数里 |

### 4.2 MCP 工具

种子算法抽到共享模块，MCP 进程与客服进程都引用它。不接真实系统、不建表。

| Server | 默认地址 | 工具 | 参数 | 原始返回 |
|---|---|---|---|---|
| `logistics` | `http://127.0.0.1:8765/mcp` | `query_logistics` | `order_no` | 与现在相同：承运公司、运单号、轨迹 |
| `aftersales` | `http://127.0.0.1:8766/mcp` | `query_warranty` | `order_no` | `warranty_status`：`IN_WARRANTY` / `EXPIRED`，外加到期日、商品名 |
| `aftersales` | 同上 | `query_return` | `order_no` | `return_status`：`APPLIED` / `IN_TRANSIT` / `REFUNDED` / `NONE` |

### 4.3 权限门

- 配置里的 Server 拉回来的工具全部只读。
- 引擎执行写操作时要看到外壳传来的「已确认」。该标记不是模型参数；Schema 不允许多余字段。
- 没有确认标记就执行写操作：不调用业务函数，回灌「写操作需要用户确认」，审计状态为权限拒绝。

### 4.4 建工单外壳

引擎自己不调用 `interrupt`。外壳顺序：

1. Schema 校验。失败则不弹卡片，回灌校验说明，审计为校验拦下。
2. 通过后 `interrupt`，载荷 `{type: "ticket_preview", ticket_type, description}`。聊天接口发出 SSE 事件 `ticket_preview`，体里只有工单类型和问题描述。
3. 前端 `resume`：`{action: "confirm"}` 或 `{action: "cancel"}`。图从断点重跑，`interrupt` 直接拿到这个值。
4. `confirm`：引擎带着已确认标记落 `tickets`，工具结果带工单号，模型回复说出这个号。
5. `cancel`：不写 `tickets`，回灌「用户取消了建单」，审计为权限拒绝。

暂停那一下不落审计，只在确认或取消的最终结果上落一条。

### 4.5 追问规则

客服提示词替换现在的「立刻建单、不要追问」：用户明确说建工单或转人工才可以调用；问题描述必须来自用户原话；工单类型用户没说就先问；缺了就追问，不许编造后再调用。模型若用空描述硬调，由 Schema 拦住。

仅表达不满时仍先安抚，不建单。

## 5. 执行引擎与审计

`execute(name, args, ctx)` 顺序固定：查表 → JSON Schema 校验 → 权限门 → 执行（超时、按规则重试）→ 结果格式化 → 写审计。给模型的都是工具结果，不把异常抛出图外。

`ctx` 含会话 id、`tool_call_id`、外壳才填得上的「已确认」。

### 5.1 三类结果

| 情况 | 给模型 | 审计状态 | 重试 |
|---|---|---|---|
| 参数不合法 | `invalid_args`，`参数不合法：…` | 校验拦下 | 不重试 |
| 查询落空 | `not_found`，`没有查到对应记录` | 成功，摘要写「未查到」 | 不重试 |
| 真故障 | `fault`，`工具执行失败：…` | 失败 | 见下 |
| 超时且重试耗尽 | `fault`，`工具执行超时` | 超时 | 见下 |
| 写操作没确认，或用户取消 | `写操作需要用户确认` / `用户取消了建单` | 权限拒绝 | 不重试 |
| 工具未注册 | `fault`，`工具未注册：…` | 失败 | 不重试 |

查询落空的信号：工具抛出 `LookupMiss`，或正常返回 `{"found": false}`。现有订单、物流假数据对任意订单号仍返回确定性数据。

### 5.2 重试与超时

- 只读工具：`tool_retries`（默认 1）、`tool_timeout`（默认 3 秒）。只在暂时性故障上重试：连接失败、读中断、超时。业务空结果、参数错误、权限拒绝、认不出的异常都不重试。适配器若包一层异常，解包后仍按这个集合判断。
- 写操作：重试次数强制为 0，超时也只尝试一次。
- 审计 `retry_count` 是额外重试次数，不含第一次。
- 耗时按整次调用计算，包含重试，单位毫秒，写入 `duration_ms`。

人为超时：`TOOL_DELAY_TOOLS`（工具名，逗号分隔）与 `TOOL_DELAY_SECONDS`（默认 0，不生效）。引擎在这一次尝试里先等待再调用。等待大于 3 秒时，只读工具出现重试、超时状态和耗时；`create_ticket` 超时则 `retry_count` 为 0。

### 5.3 结果格式化

回灌前格式化。序列化 `ensure_ascii=False`。翻译表挂在我们的工具记录上。

| 工具 | 留给模型的字段 | 枚举 |
|---|---|---|
| `query_order` | 订单号、状态、商品、金额、下单时间 | 状态本身已是中文 |
| `query_product` | 商品、价格、库存、促销 | — |
| `query_logistics` | 订单号、承运公司、运单号、轨迹的时间与描述 | — |
| `query_warranty` | 订单号、商品、保修状态、到期日 | `IN_WARRANTY`→在保，`EXPIRED`→已过保 |
| `query_return` | 订单号、退货状态 | `APPLIED`→已申请，`IN_TRANSIT`→寄回途中，`REFUNDED`→已退款，`NONE`→无退货单 |
| `create_ticket` | 工单号、一句已创建说明 | — |

没有翻译表的新 MCP 工具：去掉 `_` 开头的字段，其余原样 JSON 交给模型。

`result_summary` 存格式化后的 JSON 文本，超过 2000 字截断。

### 5.4 审计表

用户 DDL 为唯一权威，落地 `db/init/10-ch08.sql`。`scripts/init_ch08.sh` 对 `ecom_cs` 与 `ecom_cs_test` 各执行一次，客户端 `--default-character-set=utf8mb4`。

```sql
-- =============================================================
-- ch08 · 工具系统 · 建表 DDL
-- 本章新建:tool_audit_logs(工具调用审计留痕,统一执行引擎每次调用落一条)
-- 不挂外键:审计写入不能被引用约束拦住,conversation_id 只建普通索引
-- 全库统一 ENGINE=InnoDB、CHARSET=utf8mb4
-- =============================================================

-- 确保中文 ENUM 定义值/DEFAULT/COMMENT 按 utf8mb4 解析
-- (否则 latin1 默认的 mysql client 会把中文 double-encode,ENUM 值存成乱码)
SET NAMES utf8mb4;

-- 工具调用审计:内置和 MCP 工具都记,被权限拒、被校验拦的调用同样落一条
CREATE TABLE tool_audit_logs (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '审计主键',
  conversation_id BIGINT UNSIGNED NULL                    COMMENT '所属会话,无会话上下文的调用为 NULL',
  tool_call_id    VARCHAR(64)     NULL                    COMMENT '模型申请单 id,可对回 messages 流水',
  tool_name       VARCHAR(128)    NOT NULL                COMMENT '工具名',
  tool_source     ENUM('builtin','mcp') NOT NULL          COMMENT '工具来源:内置 / MCP 接入',
  mcp_server      VARCHAR(64)     NULL                    COMMENT '来源 MCP Server 名,内置工具为 NULL',
  arguments       JSON            NULL                    COMMENT '调用参数',
  result_summary  TEXT            NULL                    COMMENT '返回结果,过长截断存摘要',
  status          ENUM('成功','失败','超时','校验拦下','权限拒绝') NOT NULL COMMENT '本次调用结局',
  error_message   VARCHAR(512)    NULL                    COMMENT '失败 / 拦下时的原因说明',
  retry_count     TINYINT UNSIGNED NOT NULL DEFAULT 0     COMMENT '实际重试次数,写操作默认不重试恒为 0',
  duration_ms     INT UNSIGNED    NULL                    COMMENT '耗时毫秒',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '调用时间',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  KEY idx_tool_name (tool_name),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='工具调用审计留痕';
```

列如何填：

- 内置：`tool_source='builtin'`，`mcp_server` 为 NULL
- MCP：`tool_source='mcp'`，`mcp_server` 为 `logistics` 或 `aftersales`
- 失败原因写入 `error_message`
- `duration_ms` 有测量值就写入
- 审计使用独立数据库会话。业务提交成功后审计插入失败只打日志，不回滚工单，也不改变回灌结果

## 6. 测试策略

引擎、注册表、校验、权限、重试、格式化、审计：测试先行。提示词和「该不该建单」：标注样例跑真实模型。工单预览卡片：不进 TDD 与 code review。

单测：

- 插件扫描：只往插件目录加一个带 `register` 的文件，工具表能查到
- Schema：缺字段、类型错、枚举越界、多余字段 → `参数不合法`，函数体不执行，审计为校验拦下
- 权限：写操作无确认标记则函数体不执行，审计为权限拒绝。MCP 权限来自我们的 Server 配置。同名 `create_ticket` 的 MCP 工具不进表
- 重试：只读超时再试一次，`retry_count=1`，状态超时，`duration_ms` 有值。`LookupMiss`、`found: false`、普通业务异常不重试。写操作超时只跑一次，`retry_count=0`
- 格式化：枚举变中文，多余字段去掉，中文不是 `\uXXXX`
- 审计：成功、校验拦下、权限拒绝都有行；审计插入抛错时工具结果照常返回；表上无外键
- 物流假数据：同一订单号两次调用轨迹一致；内置清单没有 `query_logistics`

图和接口：

- 参数不齐不出现 `ticket_preview`
- 参数齐：中断载荷含工单类型和问题描述；`resume {action: confirm}` 后 `tickets` 多一行，结果含工单号
- `resume {action: cancel}`：`tickets` 不增行，审计为权限拒绝
- SSE 发出 `ticket_preview`；`POST /api/tickets` 现有测试继续通过

MCP：

- 测试中拉起两个 Streamable HTTP 进程。刷新后能看到 `query_logistics`、`query_warranty`、`query_return`，且只有一个 `query_logistics`
- 客服进程不重建，再刷一次：Server 侧多一个工具，工具表就多一行

标注样例追加到 `tests/data/tool_routing_samples.jsonl`（真实模型，只看选了谁、不执行）：

- 「帮我建个工单」→ 不调用 `create_ticket`
- 「耳机没声音，帮我建个售后工单」→ `create_ticket`
- 「订单 1001 还在保吗」→ `query_warranty`
- 「订单 1001 的退货到哪一步了」→ `query_return`
- 原有「订单 1001 的物流到哪了」仍期望 `query_logistics`（来自物流 MCP）

人工验收：两个 Server 单独起进程后在聊天里问物流；只重启 MCP Server 后再问新工具；设置 `TOOL_DELAY_*` 后看审计。工单卡片的点击在浏览器里确认。

## 7. 本章不做

- Skill 机制
- 仓储及其他外部系统
- 把投诉按钮改成预览卡片
