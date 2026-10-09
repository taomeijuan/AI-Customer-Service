# ch06 设计：分流器正式版（指代消解 · 意图四件套 · 退款确定性子流程 · 槽位）

日期：2026-10-09 ｜ 前置：ch05 LangGraph Workflow 已完结
拍板记录：退款单新表落库（DDL 用户出）／订单选择器用 list_orders 假数据／「其他」进主力 Agent／消解与意图两次独立调用

## 1. 现状 → 目标

ch05 的 `resolve` 节点是透传占位，意图分类是简化 prompt（无 confidence、无 few-shot、无兜底类）。
ch06 把分流器做成正式版：

```
query+history
   │
resolve（LLM 指代消解+改写，透传规则内置）          ← 新实现
   │
intent（四件套 prompt：选择题+JSON+few-shot+其他）  ← prompt 重写
   │
route（8 类 → 5 出口，写死）                        ← 扩展
   ├─ 退款退货/售后 → refund_prep（确定性子流程）    ← 新增
   │     ①正则提订单号（提不到不猜）
   │     ②无单号 → interrupt(订单选择器) → 用户点选 → Command(resume) 恢复
   │     ③query_order(order_no) 拿订单数据
   │     ④Query 扩写 {queries:[...]} → 多路检索合并去重（强制政策条款）
   │     ⑤证据+订单数据注入主力 Agent → 判「这一单能不能退」
   ├─ 商品咨询 → retrieve → gate → agent（不扩写，维持 ch05）
   ├─ 物流/订单/其他 → agent（其他=兜底，Agent 内自行澄清）
   ├─ 投诉 → comfort（+options）
   └─ 闲聊 → chitchat
```

## 2. 指代消解 + 改写（resolve 节点，一次 LLM 调用）

- 输入：裁剪后对话历史 + 本轮用户消息。
- 输出：强制 JSON `{"query": "<补全后的独立问题>"}`。
- 规则写进 prompt：①指代词（它/这个/那种）必须结合历史替换成具体对象；②口语模糊问法归一为标准问法（如「咋退」→「怎么申请退货退款」）；③**问题已完整、指代已明确 → 原样透传**，禁止改写词序、禁止增删语义。
- 失败兜底：解析失败/超时 → 透传原 query（分流不能因消解挂掉而断）。
- 消解后的 query 作为后续 intent/retrieve/agent 的统一输入；历史注入时剥离陈旧 [n]（沿用 ch05）。

## 3. 意图识别（四件套 prompt）

1. **选择题枚举**：七类业务意图 + 「其他」共 8 个选项，要求只选一个。
2. **强制 JSON**：`{"intent": "<枚举值>", "confidence": <0-1>}`，`with_structured_output(method="function_calling")`（DeepSeek 不支持 json_schema，沿用 ch01 结论）。
3. **few-shot 边界样例**（≥6 条）：「它啥时候到」(历史有订单)→物流；「这个能退吗」(历史有商品)→退款退货；「你们东西是正品吗」→商品咨询；「我要投诉你们」→投诉；「在吗/哈哈」→闲聊；「我说的那个东西怎么样」(指代无法消解)→其他。
4. **「其他」兜底**：拿不准、无法归类 → 其他，禁止硬塞业务意图。
- confidence 阈值：低于 0.6 记入 low_confidence_questions（source="intent_low_conf"，喂 ch09 飞轮）；本期不做小模型降级，分类器接口保持 `classify(query, history)` 不变，降级路后接。
- 失败兜底：解析失败/网络错误 → 沿用 ch05（fallback=订单，进 Agent 最通用）。

## 4. Query 扩写（检索侧现查现用）

- 触发：仅退款子流程内（退款退货/售后）。
- 实现：LLM 强制 JSON `{"queries": ["...", "..."]}`（2-4 条，侧重点不同：政策条款角度/时效角度/场景特例角度）。
- 检索合并：新 `retrieve_multi(queries, strategy="hybrid_rerank")` —— 每条查询独立走 hybrid_rerank，按 chunk_id 去重（同 chunk 取最高 relevance），保持 rerank 分数语义，返回统一 evidence 列表。
- 证据弱 → 沿用置信度闸兜底话术 + low_confidence_questions 入池。
- 知识库只留一份（不拆存多份）；商品咨询等简单 FAQ 不扩。

## 5. 槽位与暂停-恢复（LangGraph 原生 HITL）

- Context7 已核实：节点内 `interrupt(payload)` 暂停图并持久化状态；`astream(Command(resume=value), config同thread_id)` 从检查点恢复；需 checkpointer（主图已有 InMemorySaver）。
- `refund_prep` 提不到订单号（正则 `\b\d{3,}\b` 无命中）→ `interrupt({"type":"order_selector","orders":[...]})` → chat.py 捕获中断 → SSE 发新事件 `order_selector`（含订单卡片数据）→ done 收尾本轮。
- 用户点选卡片 → 前端 POST `/api/chat/stream` 带 `"resume": {"order_no":"1001"}`（无 message）→ 后端取该会话的 pending thread_id → `astream(Command(resume=...))` → 子流程从断点继续。
- pending thread 映射存进程内存 `{conversation_id: thread_id}`（与 InMemorySaver 生命周期一致；重启丢失=检查点同失，行为自洽）。恢复完成（log 节点落库原始问题+最终答复）后清除映射。
- **不猜订单号**：消解/意图/Agent 都不许编单号，选择是唯一回填通道。

## 6. 退款单（新表 + 工具 + 表单）

- 新表 `refund_orders`（订单号/原因类目/金额/状态/会话外键），**表结构以用户 DDL 为准**（开建时提供）。
- 子流程 Agent 判定「能退」后，回复附「申请退款」按钮（复用 ch05 options 机制，新增 action 类型）。
- 点击 → 聊天流内弹退款表单（Vibe Coding）：原因从固定类目下拉（七天无理由/质量问题/少件/与描述不符/其他），**不追问原因**；提交 → `POST /api/refunds` → 落库 → 返回退款单号确认消息。
- 退款原因类目定为常量（前后端同源：后端 `/api/refunds` 校验 Literal，前端下拉同列表）。

## 7. list_orders 工具

- 新 @tool：确定性返回当前可操作订单 1001-1005 摘要列表（商品/金额/状态），数据内部调 `query_order` 生成，与 ch05 md5 种子假数据世界自洽。

## 8. 前端（Vibe Coding，不套流程）

- 订单选择器：聊天流内渲染订单卡片（商品/金额/状态），点选 → 回填 resume 请求 → 展示「已选订单 1001」用户侧回执 → 子流程续跑。
- 退款表单：聊天气泡内嵌简单表单（原因下拉+提交按钮），提交后卡片态展示退款单号。
- 沿用：打字机/工具芯片/引用卡/自诊断行/版本戳（bump v2026-10-XX）。

## 9. 评测集（Prompt/数据任务以标注样例替代 TDD）

- `tests/data/intent_eval.jsonl`：多轮对话用例，每轮断言 `{expected_intent, expected_resolved_query}`；重点覆盖 **物流→退款→物流切换**、指代补全（「它能退吗」「那个多少钱」）、边界怪问题→其他。
- `tests/data/expansion_eval.jsonl`：退款类问题 → 扩写条数 2-4、覆盖点断言（含政策/时效/特例角度）。
- 真实 LLM 跑（.env 存在才跑，沿用 eval skipif 惯例）。

## 10. 本章不做

微调小模型/BERT 分类器；跨会话记忆；意图小模型降级部署（只留接口）； Milvus/LangChain/LangGraph 版本变更。

## 11. 验收标准映射

| 验收 | 对应设计 |
|---|---|
| 1 多轮切换意图全对+指代全补 | §2 resolve + §3 意图 + §9 多轮评测集 |
| 2 JSON 稳定可解析、怪问题落其他 | §3 强制 JSON + 其他兜底 + 失败兜底 |
| 3 「这个能退吗」先补全再走子流程拿订单和政策 | §2 + §4 + §5（SSE 顺序可见：delta→order_selector/工具帧→回答） |
| 4 浏览器无单号问退款弹选择器、点选续跑 | §5 interrupt/resume + §8 卡片 |
