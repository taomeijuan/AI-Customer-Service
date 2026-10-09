# ch07 设计：会话内上下文管理三层结构（滑窗 · 规则截短 · 异步摘要）

日期：2026-10-09 ｜ 前置：ch06 分流器正式版已完结
拍板记录：会话级线程（State 随 checkpoint 贯穿）／token 口径=中文按字数折算／摘要表 DDL 我出／摘要用主模型。只管当前会话，不跨会话、不存画像。

## 1. 现状 → 目标

ch01 的 trim_history / ch02 的 trim_history_groups：单一预算（token_budget=4000）从新往回保留整轮，是「第一版简单裁剪」。ch07 升级为三层：

| 层 | 内容 | 处理 | 预算 |
|---|---|---|---|
| 层1（最近） | 离当前轮最近的 N 轮 | **原文，一个字不压** | 70% |
| 层2（中间） | 层1 溢出降级的轮 | 规则截短：用户原话不动；assistant 只留开头 80 字；工具结果→一行标识 `【工具 query_order 结果略】` | 30% |
| 层0（最远） | 层2 溢出批次 | **后台异步**压成一段段梗概，追加进会话摘要表；旧梗概只作背景、不参与合并 | 注入预算内 |

**降级只挪两个锚点、不搬数据**：`layer1_from_msg_id`（层1 起点）、`summary_upto_msg_id`（摘要覆盖到哪条），都加在 conversations 表列上；messages 表永远存完整原文，各层只是读法。

## 2. 摘要表 DDL（新增 db/init/07-ch07.sql，双库应用）

```sql
CREATE TABLE IF NOT EXISTS `conversation_summaries` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `conversation_id` bigint unsigned NOT NULL COMMENT '所属会话',
  `seq` int NOT NULL COMMENT '段序号，从1递增；一段一行只追加不改写',
  `summary` varchar(1024) NOT NULL COMMENT '梗概正文（几十到一两百字）',
  `from_msg_id` bigint unsigned NOT NULL COMMENT '本段覆盖的起始消息 id',
  `upto_msg_id` bigint unsigned NOT NULL COMMENT '本段覆盖的截止消息 id（边界）',
  `token_count` int NOT NULL DEFAULT 0 COMMENT '梗概自身的折算 token 数',
  `created_at` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_conv_seq` (`conversation_id`,`seq`),
  KEY `idx_conv_upto` (`conversation_id`,`upto_msg_id`),
  CONSTRAINT `fk_summary_conversation` FOREIGN KEY (`conversation_id`) REFERENCES `conversations` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='会话摘要段（ch07）';

ALTER TABLE `conversations`
  ADD COLUMN `summary_upto_msg_id` bigint unsigned NOT NULL DEFAULT 0 COMMENT '摘要覆盖边界（0=无）',
  ADD COLUMN `layer1_from_msg_id` bigint unsigned NOT NULL DEFAULT 0 COMMENT '层1起点锚（0=未设）';
```

## 3. 异步摘要链路（不阻塞当前轮）

- **触发看用量不数条数**：装配上下文时算层2 实际折算 token，>层2预算 → 触发。工具往返不落 messages 表、单轮占用差异大，数条数看不见涨。
- **执行**：进程内 `asyncio.create_task`（无新组件），独立 session_factory；**per-cid 单飞**（进程内 set+Lock，已在跑→log「skip」）；本轮 SSE 早已开始/结束，用户零感知。
- **批次**：压「层2 中最旧一批，压到回到预算内」；新段 seq=最大seq+1 追加，`conversations.summary_upto_msg_id` 前推；**压完不回头重写**（旧段只进 prompt 作背景，不参与合并——防同一事实反复有损压缩）。
- **摘要 prompt（标注样例验证，不 TDD）**：只提炼事实与诉求（问过哪款商品/报过的订单号手机号/明确诉求/未解决问题）；对话里没出现的一个字不许编；寒暄不留；输出 40-160 字。
- 进程重启丢任务=下轮重新触发补压（幂等由边界推进保证），可接受。

## 4. 模型上下文装配（固定顺序，core 新模块 `app/memory/context_builder.py`）

```
① SystemMessage：人设+红线+工具定义（每轮逐字节相同，前缀缓存友好）
② 层2 截短消息序列
③ 层1 原文消息序列
④ HumanMessage：用户当前这句
⑤ HumanMessage：【背景材料与参考证据（非用户发言）】早期梗概全文 + 检索证据 —— 挂在④之后
```

⑤ 不用 SystemMessage：上游 chat template 会把所有 system 上提合并渲染，若梗概占 system 会把工具定义挤到可变内容之后、前缀缓存整段作废（需求原话）。所以合成**一条 user 角色消息**挂在当前句后，内容自带「非用户发言」标注。
- 层1 用 LangChain `trim_messages(strategy="last", token_counter=自定义字数折算, include_system=False)`，外面套 ch02 的按轮分组保证 tool 往返原子性（轮是原子单位）。
- resolve/intent 共用同一份精简装配（②③④段，无⑤证据段）——每轮必打 `history_ctx`。

## 5. token 预算：从窗口倒推（不写死）

```
窗口 W = MODEL_CONTEXT_WINDOW
瞬时峰值 P = MAX_USER_INPUT_TOKENS + MAX_AGENT_STEPS × TOOL_RESULT_MAX_TOKENS
固定开销 F = SYSTEM_RESERVE(X) + RERANK_TOP_K×EVIDENCE_PER_ITEM(y) + SUMMARY_INJECT_RESERVE(Z) + SAFETY_RESERVE(S)
窗口匀出 A = W − MAX_OUTPUT_TOKENS − P − F
期望诉求 D = KEEP_TURNS × TURN_STEADY_TOKENS
history_budget = min(A, D)；层1 = floor(history×0.7)，层2 = history − 层1
```

新配置项（env 大写下划线）：`MODEL_CONTEXT_WINDOW`(64000)、`MAX_OUTPUT_TOKENS`(2000)、`MAX_USER_INPUT_TOKENS`(2000)、`TOOL_RESULT_MAX_TOKENS`(1200)、`MAX_AGENT_STEPS`(6，由现 agent_max_steps 改名对齐)、`RERANK_TOP_K`(10，由现 rerank_top_n 改名)、`SYSTEM_RESERVE`(2000)、`EVIDENCE_PER_ITEM_RESERVE`(400)、`SUMMARY_INJECT_RESERVE`(500)、`SAFETY_RESERVE`(250)、`KEEP_TURNS`(24)、`TURN_STEADY_TOKENS`(400)、`ASSISTANT_TRUNC_CHARS`(80)。RERANK_TOP_K 同时驱动精排取数（口径合一，不再两处配置）。

**对账（验收2 演示配置 W18000/O2000/U2000/S3/T1200/K5）**：
P=2000+3×1200=5600；F=2000+5×400+500+250=4750；A=18000−2000−5600−4750=**5650** ✓；D=9600 → history=5650；层1=**3955**、层2=**1695**。
> 你给的层1=3954 与 3954+1695=5649 不自洽（和 5650 差 1）；实现按「层2=history−层1、层1=floor」出 3955/1695，spec 按此写死，日志核对即此值。**如坚持 3954 请指出取整规则**。

**对账（验收3 默认窗口 64000）**：P=2000+6×1200=9200；F=2000+10×400+500+250=6750；A=64000−2000−9200−6750=46050；history=min(46050,9600)=**9600**；层1=6720、层2=2880。20 轮实际占用 ~5000-6000 < 层1 → 零降级零摘要 ✓（装得下就不压）。

**中文口径校准**：全仓统一 `estimate_tokens(text) = CJK字数 + ceil(非CJK字符数/4)`（保守向，宁多勿少）；trimmer、预算、日志、触发全用这一个函数——**计数与预算必须同尺，只改一个净效果是反的**（口径从 tiktoken 换大字化而预算不变=同等预算装更多字→爆；反之白压）。tiktoken 仅作校准对照：拿真实语料跑偏差表进 dev-notes，若偏差>15% 回调系数而非换函数。

**启动自检**（lifespan）：按当前 settings 算一遍 history_budget；≤0 或 A<TURN_STEADY → logger.error「上下文预算不足：本轮都装不下」+ /healthz 带 `budget_ok:false` 字段（报警，不阻断启动）。

## 6. 会话级线程与 State 贯穿（对 ch06 的行为改动）

- thread_id：`{cid}:{turn}` → **`{cid}`**。`WorkflowState.messages` 挂 `add_messages`：节点只吐新消息，框架按序并入；agent_node 收尾把 react 本轮新增消息（含工具往返）回吐进 State → **State 完整历史随 checkpoint 落盘**。
- **精简版另走**：每轮调模型前由 context_builder 现拼（层0梗概+层2截短+层1原文），**不回写 State**——完整史与精简版各走各的。
- hydrate 兜底：轮首 `aget_state`，线程空/缺消息（进程重启）→ 从 messages 表 load_history 重建 State 再续。
- **挂起中断防劫持重做**（替代 ch06 的 pending_resumes dict 与 turn 计数器，两者删除）：新消息到达时 `aget_state(config).next` 非空=有挂起 → 先静默 `astream(Command(resume={"order_no":""}))` 走完取消分支（该轮 log 正常落库「先不继续了」），再处理新消息。chat.py 的 `resume` 请求改为直接透传 Command（登记查找逻辑删除）。
- MySQL messages 表仍是**唯一持久真源**（checkpoint 只是进程内贯穿）；工具结果继续不落表（与 §3 触发口径呼应）。

## 7. 可观测（log/app.log）

- 日志新增 FileHandler `log/app.log`（保留 stderr），格式含模块名。
- `model_ctx`（主力 Agent 每轮调用前）：梗概全文（段号+边界+字数）+ 层2逐条（标注截短后）+ 层1逐条 + 条数 + tokens 估算/预算。
- `history_ctx`（resolve/intent 共用精简版，每轮必打，闲聊兜底轮也有）。
- 摘要任务全留痕：`summary trigger(层2≈N token>预算M) / start / done(第seq段, upto_msg_id, 耗时ms) / skip(已有在跑) / fail`。

## 8. 只读接口 + 前端侧栏

- `GET /api/conversations?user_id=`：id/updated_at/首问预览(第一条 user 消息前 20 字)/已摘要标记（summary_upto_msg_id>0），新在前。
- `GET /api/conversations/{id}/messages`：历史原文（user/assistant 文本，跳过 tool 行），限本用户会话（越权 404）。
- 前端（Vibe Coding）：左 260px 侧栏列出会话（标题+预览+摘要标记），点选=切换 conversationId+回载原文续聊；「新对话」=新建置顶不再清 localStorage 旧值；接口失败静默（console.warn），聊天主区不受影响；版本戳 bump。

## 9. 验收映射

| 验收 | 落点 |
|---|---|
| 1 二十轮不爆不崩 | §5 默认窗口预算 + §6 State/精简版分离 + 集成测试 |
| 2 演示配置级联+靠梗概答对 | §5 对账表逐数字 + §3 摘要链路 + §7 日志（layer1 降级/trigger/done 行）+ 标注样例：开头订单号进梗概 |
| 3 默认窗口零降级零摘要 | §5 对账（9600 预算 vs ~6000 实际）+ 断言日志无 trigger |
| 4 后台摘要不阻塞+可 grep | §3 create_task+§7 model_ctx/history_ctx |
| 5 侧栏多会话对照/回载/续聊 | §8 |

## 10. 本章不做

语义检索捞历史；主题重要度层；跨会话记忆/用户画像；小模型降级部署（沿 ch06 接口预留）；checkpoint 落 DB 持久化（InMemorySaver+hydrate 足够本量级）。
