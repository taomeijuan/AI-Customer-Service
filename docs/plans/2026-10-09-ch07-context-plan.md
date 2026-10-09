# ch07 实施计划：会话上下文管理三层结构

设计：docs/plans/2026-10-09-ch07-context-design.md（定稿）｜ DDL 已应用双库
口径校准点：预算对账按 floor 出 3955/1695（用户「没异议」）；`agent_max_steps→MAX_AGENT_STEPS`、`rerank_top_n→RERANK_TOP_K` 改名（同确认）

## Context7 核对结论

- LangChain 现官方主推 SummarizationMiddleware/before_model（**同步、随图跑**）——与需求「后台异步、一段一行不回炉」路线不符，**不用中间件**，自研编排；`trim_messages`（langchain_core.messages.utils）仍是公开工具，支持自定义 `token_counter`，层1 用它做预算截断（外层套按轮分组保原子）。
- LangGraph 会话级 thread：`aget_state(config).next` 非空即有挂起中断（ch06 探针已实证 Command(resume) 语义）；`add_messages` reducer 自动按序并入。
- checkpoint 选型不变：InMemorySaver（进程内贯穿，重启 messages 表 hydrate 兜底）。

## 任务分解

| # | 任务 | 类型 | 验证 |
|---|---|---|---|
| T1 | 配置改名+新增：MODEL_CONTEXT_WINDOW/MAX_OUTPUT_TOKENS/MAX_USER_INPUT_TOKENS/TOOL_RESULT_MAX_TOKENS/MAX_AGENT_STEPS(改名)/RERANK_TOP_K(改名)/SYSTEM_RESERVE(2000)/EVIDENCE_PER_ITEM_RESERVE(400)/SUMMARY_INJECT_RESERVE(500)/SAFETY_RESERVE(250)/KEEP_TURNS(24)/TURN_STEADY_TOKENS(400)/ASSISTANT_TRUNC_CHARS(80)；全引用点同步（retriever/agent_node/graph/chat）+.env.example | 代码 | 既有测试全绿 |
| T2 | app/memory/tokens.py：`estimate_tokens`（CJK 1字=1tok + 非CJK ⌈/4⌉）；trimmer 换用同一函数；对照校准测试（真实语料与 tiktoken 偏差 <15%） | 代码+数据 | 单测+校准记录进 dev-notes |
| T3 | app/memory/budget.py：compute_budget(settings)→BudgetSpec{peak,fixed,avail,history,layer1,layer2}；启动自检（装不下一轮→log.error + /healthz budget_ok:false） | 代码 | **对账单测：演示配置=5650/3955/1695；默认窗口=9600/6720/2880** |
| T4 | models.py 三列映射 + SummariesRepo（追加段/读段/更新投影+锚点单事务）+ ConversationRepo 读锚点 | 代码 | 单测 |
| T5 | app/memory/layers.py：按锚点切三层（锚点 NULL=全层1）；层2 规则截短渲染（user 不动/assistant 留头 N 字/tool→一行标识）；层1 trim_messages(自定义 counter)+按轮分组原子 | 代码 | 单测（含边界：锚点空/重叠/孤儿 tool 轮） |
| T6 | app/memory/context_builder.py：固定顺序 ①system(人设+红线+工具定义) ②层2 ③层1 ④用户当前句 ⑤单条 user 消息挂「梗概投影+检索证据」；输出 (messages, ctx_report)；`model_ctx`/`history_ctx` 日志（含条数/tokens/预算） | 代码 | 单测 + 日志断言(caplog) |
| T7 | Summarizer 后台任务：trigger(层2 用量>预算)→asyncio.create_task；per-cid 单飞+claim(边界 CAS)；选批(层2 最旧批)→主模型压缩→段追加+summary_upto 前推+投影重组；trigger/start/done/skip/fail 全留痕带耗时；**不回头重写** | 代码 | 单测(mock LLM)+**标注样例**：summary_eval.jsonl（订单号/手机号/诉求/未解决问题保留，寒暄丢弃，40-160 字，禁编造）真实 LLM 跑 |
| T8 | 会话级线程改造：thread_id=`{cid}`；State.messages add_messages 真贯穿（agent_node 回吐 react 本轮消息含工具往返）；hydrate（线程空→messages 表重建）；新消息遇挂起→先静默 `Command(resume={"order_no":""})` 走完取消；**删 pending_resumes/turn_counters**；resolve/intent/agent 全面改吃 context_builder 产物 | 代码 | ch05/ch06 SSE 契约测试重做+新集成测试（劫持防护、hydrate、resume 后续跑） |
| T9 | 只读接口：GET /api/conversations?user_id=（首问预览/已摘要标记/新在前）、GET /api/conversations/{id}/messages（原文、跳过 tool、越权 404） | 代码 | 接口测试 |
| T10 | 前端侧栏（Vibe）：260px 列表(标题+预览+摘要标记)、点选切换回载续聊、「新对话」新建置顶不清旧、接口失败静默降级、版本戳 bump | 前端 | 浏览器验收5 |
| T11 | 三场景集成回归：①演示配置(env 覆盖)连聊 20+ 轮：层1降级→层2 超预算→后台摘要→「最开始那个订单后来怎么说」靠梗概答对 ②默认窗口 20 轮零降级零摘要（日志无 trigger）③grep model_ctx/history_ctx 完整 | 集成 | 脚本+日志核对 |
| T12 | 全量回归+评审修复+验收+完结留痕 | 收尾 | pytest 全绿+五条验收 |

依赖：T1→T2→T3；T4→T5→T6→T7；T8 汇合前两线；T9/T10 独立可后做；T11→T12。
Prompt/数据任务（T7 摘要 prompt、T2 校准）按规矩用标注样例验证替代 TDD；T10 Vibe。

## 风险与对策

- **T8 是本张最重手术**（动 chat.py/graph/agent_node 主链）：ch05/06 的 SSE 测试契约大面积重做——先写目标行为的失败测试再动刀，小步提交。
- 挂起中断静默取消会产生一条「先不继续了」的落库回复：验收语义确认可见（该轮实际终止），不藏。
- 投影 summary 与分段表可能漂移：投影每次摘要完成从分段表整体重组（≤注入预算内取最近段），读路径永不现拼。
- estimate_tokens 保守向可能高估中文→提前降级：与验收 3 冲突时按真实语料校准系数（T2 偏差表），**只调系数不换函数**。
- InMemorySaver 重启丢 State：hydrate 兜底后行为等价 MySQL 真源；测试覆盖。
