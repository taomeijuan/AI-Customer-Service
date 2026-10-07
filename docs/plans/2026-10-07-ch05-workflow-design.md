# Ch05 设计文档：Workflow 编排（LangGraph 骨架 + 祛魅循环 + ReAct 主力 Agent）

> 状态：已定稿（2026-10-07，用户批准）
> 范围：不做意图识别/指代消解正式版、上下文管理升级、MCP、飞轮入库
> 流程：Superpowers（brainstorm → 计划 → TDD 构建 → 评审 → finish）

## 1. 需求与验收

需求：
1. 祛魅热身：手写最裸 Agent 循环（LLM→工具调用执行喂回→收敛），再用 LangGraph 重构
2. 图骨架：指代消解（透传）→ 意图识别（七类 JSON）→ 分流 → 知识检索 → 置信度闸 → 主力 Agent → 日志记录
3. 分流写死：知识类（商品咨询、退款退货）强制先检索过闸再进 Agent；业务数据类（物流、订单、售后）直接进 Agent；投诉不进 Agent（安抚话术+前端自选转人工/建工单）；闲聊固定话术零模型调用
4. 主力 Agent ReAct：一次收敛/多步/追问用户，停止条件+token 控制；复用 ch02 工具
5. State 贯穿 + 自带 checkpointer
6. 指代消解透传、意图识别简单 prompt
7. 置信度闸：卡在知识检索后、Agent 前；证据弱直接兜底话术+记录入池；业务数据类不走闸
8. 转人工/建工单分开、前端自选、不自动执行

验收标准：
1. 政策问题日志可见强制检索节点被走到
2. 「订单 1001 的物流」Agent 自调工具作答
3. 「我要投诉」出两独立按钮；转人工前端模拟；建工单才写表；互不绑定
4. 闲聊固定话术
5. 先查订单再查物流的复杂问题 ReAct 多步可见

## 2. 澄清决策（已拍板）

| 决策点 | 结论 |
|---|---|
| 状态架构 | 分工制：MemorySaver（自带 checkpointer）只管图执行内状态；聊天记录仍归 MySQL messages（ch03 真源不变） |
| Agent 节点 | LangGraph 预构件 create_react_agent，编译后当主图节点 |
| 祛魅循环 | app/agents/bare_loop.py 留仓库可运行（带测试） |

## 3. 图骨架

```
START → resolve（透传）→ intent（LLM 七类 JSON，解析失败兜底→业务数据类）
  → route（分流规则写死代码）
  ├─ 知识类（商品咨询/退款退货）→ retrieve（复用 ch04 HybridRetriever, hybrid_rerank）
  │     → gate（条件边：证据够→agent；弱→fallback：兜底话术+入池）→ agent（ReAct）→ log → END
  ├─ 业务数据类（物流/订单/售后）→ agent → log → END
  ├─ 投诉 → comfort（安抚话术+options=["转人工","建工单"]）→ log → END
  └─ 闲聊 → chitchat（固定话术，零模型调用）→ log → END
```

- State 字段：query / intent / messages / evidence / citations / options / refusal / final_text
- checkpointer：MemorySaver，thread_id = `{cid}:{turn_seq}`（只管轮内 ReAct 多步；跨轮历史从 MySQL 加载）
- 意图识别：`{intent: 七类之一}`，解析失败/非法值 → 业务数据类兜底
- 置信度闸：复用 ch04 信号（精排成功时 relevance 阈值 / 零召回），弱 → 兜底话术 + LowConfidenceRepo.record(source='retrieval_low_conf')；业务数据类无检索证据不走闸
- 知识注入：证据以 [n] 编号条目消息注入 Agent，要求回答带角标；citations 全集快照（ch04 台账语义）经 done 帧透传
- 日志节点：user/assistant 消息落 messages 表（复用 MessagesRepo）+ 意图/路径结构化日志

## 4. 祛魅裸循环（bare_loop.py）

纯 httpx OpenAI 协议手写：调 LLM → tool_calls 非空则本地执行工具并作为 tool 消息喂回 → 循环；空则收敛答案。max_steps 停止条件（超步返回已有最佳文本+日志）。测试：MockTransport 脚本化（一次工具收敛 / 多步 / 超步停止）。

## 5. SSE 与前端

- 事件兼容：meta / tool（ReAct 每次工具调用一帧）/ delta / done（citations）/ **options（新增）** / error
- 投诉路径：comfort 节点产安抚话术（delta 流式）+ options 帧
- 前端：options 帧渲染「转人工」「建工单」两独立按钮；转人工=纯前端模拟（已转接+客服小猫问候）；建工单=POST /api/tickets（直写 tickets，不调 LLM）；互不绑定、不点即普通对话
- 新端点：`POST /api/tickets {conversation_id, description, ticket_type}` 复用 TicketsRepo

## 6. 依赖与配置

- 新增 pip：langgraph（1.x 线）；checkpointer 用内置 MemorySaver（无新包）
- .env 增：`AGENT_MAX_STEPS=6`（ReAct 递归/步数上限）、`INTENT_MODEL`（留空=llm_model）
- 风险预案：LangGraph 1.x 预构件包名/create_react_agent 签名/MemorySaver 路径/astream 事件形态必须 Context7 核对；对不上停下问

## 7. 测试策略

- **TDD**：bare_loop 三场景；意图识别（七类+兜底）；路由映射（七类→四出口）；gate 弱证据（兜底+入池）；comfort/chitchat 节点；/api/tickets 端点；整图 Fake LLM 路径（知识类走检索节点断言）
- **eval/真机**：五条验收逐条真机（日志断言检索节点、ReAct 多步可见、两按钮、闲聊话术）
- **前端**：Vibe

## 8. 流程适配（不变）

Context7 先查后写；选型走不通停下问；dev-notes/ch05.md 逐阶段追记；前端 Vibe。
