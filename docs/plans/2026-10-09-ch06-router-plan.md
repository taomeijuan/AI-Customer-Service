# ch06 实施计划：分流器正式版

日期：2026-10-09 ｜ 设计：docs/plans/2026-10-09-ch06-router-design.md（已定稿）

## 探针结论（T1 实测，计划前置）

- StateGraph + `astream(stream_mode=["updates",...], subgraphs=True, version="v2")` 下，`interrupt(payload)` 中断以 **updates 块 `{'__interrupt__': (Interrupt(value=…),)}`** 浮出——chat.py 从此提取并发 `order_selector` SSE 事件（不走 writer 帧，恢复重跑天然无重复帧）。
- `astream(Command(resume=…), 同 thread_id config)` 恢复：中断节点**从头重跑**（`interrupt()` 直接返回 resume 值），已完成节点不重跑。
- checkpointer=InMemorySaver 可用（主图已挂）。

## 任务分解

| # | 任务 | 类型 | 验证方式 |
|---|---|---|---|
| T1 | 探针实证（已完成，结论入 dev-notes） | spike | 离线脚本 |
| T2 | resolve 节点：LLM 指代消解+改写（JSON {query}，透传规则，失败兜底透传）| 代码+Prompt | 单测（脚本桩）+ 标注样例 |
| T3 | 意图四件套：8 类选择题 + {intent,confidence} + 6 few-shot + 其他；confidence<0.6 入 low_confidence_questions(source=intent_low_conf) | Prompt+代码 | 单测 + intent_eval |
| T4 | 多轮意图评测集 tests/data/intent_eval.jsonl（物流→退款→物流切换/指代/边界）+ 评测 runner | 数据 | 真实 LLM 跑 |
| T5 | retrieve_multi(queries)：多查询 hybrid_rerank + chunk_id 去重取最高分 | 代码 | 单测（FakeMilvus） |
| T6 | 扩写器 {queries:[...]}（structured output，仅退款子流程用）+ expansion_eval 样例 | Prompt+代码 | 单测 + 标注样例 |
| T7 | refund_prep 子流程节点：正则提单号（不猜）→ query_order 拿单 → 扩写+retrieve_multi 强制政策 → 证据+订单注入 Agent | 代码 | 单测 |
| T8 | 槽位 interrupt/resume：graph 接线（8 类→5 出口）、chat.py `__interrupt__`→order_selector 事件、pending thread 映射、/api/chat/stream resume 分支 | 代码 | 单测（探针式集成） |
| T9 | list_orders 工具（1001-1005，内部调 query_order）+ create_refund 工具 + POST /api/refunds + refund_orders 表（**DDL 待用户提供**） | 代码+DDL | 单测 |
| T10 | 前端（Vibe Coding）：订单卡片（点选→resume 回填+用户侧回执）、退款表单（原因下拉+提交→退款单号）、options 扩展 action、版本戳 | 前端 | 浏览器验收 |
| T11 | 全链路回归：graph 重接线后全量 pytest + 手动 curl 矩阵（7 类意图各一问） | 回归 | pytest+curl |
| T12 | 验收四条：多轮切换评测跑分 / JSON 稳定+其他兜底 / 「这个能退吗」SSE 时序（消解→order_selector/工具帧→回答）/ 浏览器点选续跑 | 验收 | 浏览器+评测 |

依赖：T2/T3 并行 → T4 依赖 T2+T3 → T5/T6 并行 → T7 依赖 T5+T6 → T8 依赖 T7 → T9 可与 T7 并行（DDL 到位后）→ T10 依赖 T8 → T11/T12 收尾。

## 退款原因固定类目（前后端同源常量）

`七天无理由 / 质量问题 / 少件 / 与描述不符 / 其他`

## 风险与对策

- interrupt 在 astream v2 的行为已实证（T1），残余风险：chat.py 对 `__interrupt__` 与既有 updates 分支的并存处理——单测覆盖。
- resume 请求不带 message：ChatRequest 需放宽（message 与 resume 二选一校验）——单测覆盖 422 路径。
- 两次 LLM 调用叠加延迟：resolve 与 intent 都是小 prompt，可接受；扩写仅在退款场景。
- DeepSeek 消解 prompt 偶发改写过度：透传规则写死 + 标注样例回归盯住。
