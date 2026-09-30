# Ch01 设计文档：电商智能客服——纯对话（多轮 + SSE 流式 + Prompt 模板 + 结构化输出）

> 状态：已定稿（2026-09-30，用户批准）
> 范围：本章不做工具调用、Agent 循环；聊天页面不做（纯 API）
> 流程：Superpowers（brainstorm 定稿 → 计划 → TDD 构建 → code review → finish）

## 1. 需求与验收

需求：
1. 对话接口：多轮对话，SSE 流式输出、逐 token 推送
2. Prompt 管理：ChatPromptTemplate 模板化，System Prompt 写清客服角色与行为约束
3. 结构化输出：售后描述 → 订单号 / 诉求类型 / 期望方案，用 with_structured_output
4. 多轮上下文最简版：历史裁剪 + token 预算控制

验收标准：
1. curl 调对话接口能看到流式回复
2. 连续两轮，第二轮能续上第一轮上下文
3. 发一段售后描述，拿到结构化 JSON

## 2. 澄清结论（brainstorm 决策）

| 决策点 | 结论 |
|---|---|
| 会话历史存储 | 服务端内存：`dict[conversation_id, list[BaseMessage]]`，进程重启丢历史（ch01 可接受） |
| token 计数 | tiktoken（cl100k_base）统一近似四家上游；预算默认 4000，可配 |
| 聊天页面 | ch01 不做，纯 API |
| 技术路线 | LangChain 高层 API 全家桶：ChatOpenAI + ChatPromptTemplate + with_structured_output + astream/SSE |

## 3. 架构

单 FastAPI 服务，目录即分层：

```
ecom-cs/
├── app/
│   ├── main.py               # FastAPI 入口
│   ├── core/
│   │   ├── config.py         # pydantic-settings 读 .env
│   │   └── llm.py            # ChatOpenAI 工厂（base_url/model/key 全来自配置）
│   ├── prompts/templates.py  # ChatPromptTemplate：客服 System Prompt
│   ├── memory/session.py     # 内存会话库 + tiktoken 裁剪
│   ├── chains/chat.py        # prompt | llm 组装 + astream 流式
│   └── extraction/
│       ├── schemas.py        # AfterSalesExtraction（Pydantic）
│       └── service.py        # with_structured_output
├── tests/                    # 单测 + 标注样例集
├── scripts/acceptance.sh     # 验收脚本
├── docs/plans/               # 本文档 + 实施计划
├── dev-notes/ch01.md         # 过程留痕
├── .env.example              # 四家上游示例注释
└── pyproject.toml            # uv 管理
```

上游接入：应用侧统一 OpenAI 协议。GPT（官方）/ Claude（Anthropic OpenAI 兼容端点）/ DeepSeek（OpenAI 兼容）/ Ollama（`http://localhost:11434/v1`）——换上游只改 `.env` 的 `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`。

## 4. API 设计

### POST /api/chat/stream
- 请求：`{"conversation_id": "<uuid>|null", "message": "<用户输入>"}`
- 响应：`text/event-stream`，事件序列：
  - `event: meta` `data: {"conversation_id": "..."}`
  - `event: delta` `data: {"text": "<token>"}`（逐 token 多次）
  - `event: done` `data: {"conversation_id": "..."}`
  - 异常时：`event: error` `data: {"message": "<可读信息>"}`
- 会话不存在（传了 id 但内存无此会话）：404；id 为空则新建并在 meta 返回

### POST /api/extract
- 请求：`{"text": "<售后描述>"}`
- 响应：`{"order_no": "...", "issue_type": "...", "expected_resolution": "..."}`
- 失败：422（附可读错误）

### GET /healthz
- `{"status": "ok"}`

## 5. 关键行为

- **System Prompt 约束**：电商客服角色；只答电商范围；不编造订单/物流状态（不知道就说不知道）；超出能力引导转人工；语气礼貌简洁。
- **裁剪规则**：system prompt 永不参与裁剪；从最新消息往回累计 tiktoken 计数，超出 `TOKEN_BUDGET` 的更老消息丢弃；至少保留最近 1 轮。
- **流式**：`chain.astream(...)` 逐 chunk yield `delta` 事件；流结束后完整回复 append 回会话。
- **错误处理**：上游连接失败/超时 → SSE `error` 事件 + 服务端日志；extract 解析失败 → 422；请求校验失败 → FastAPI 默认 422。
- **并发**：会话 dict 读写用 asyncio.Lock 保护（单进程假设，ch01 不做分布式）。

## 6. 结构化输出 schema（拟）

```python
class AfterSalesExtraction(BaseModel):
    order_no: str | None          # 描述中出现的订单号，没有则 null
    issue_type: Literal["退款", "退货", "换货", "维修", "物流投诉", "其他"]
    expected_resolution: str      # 用户期望的处理方案原文概括
```

## 7. 测试策略

- **TDD（可单测代码）**：
  - `memory/session.py`：裁剪逻辑（给定消息列表+预算，断言保留结果）、并发安全、会话不存在分支
  - `api` 层：httpx AsyncClient + mock LLM，断言 SSE 事件序列与格式、404 分支
  - `core/config.py`：env 读取与默认值
- **Prompt/数据类 → 标注样例验证（替代 TDD）**：
  - `tests/data/after_sales_samples.jsonl`：售后描述 → 期望字段标注（≥8 条，覆盖有/无订单号、各诉求类型）
  - 运行样例集逐条断言提取结果；System Prompt 行为约束以少量对话样例人工核对
- **验收**：`scripts/acceptance.sh`——curl SSE 连续两轮（第二轮断言引用第一轮内容）+ extract 真实调用。依赖用户填好 `.env`。

## 8. 依赖

- 运行：`fastapi`、`uvicorn`、`langchain`、`langchain-openai`、`langchain-core`、`pydantic-settings`、`tiktoken`
- 开发：`pytest`、`pytest-asyncio`、`httpx`

## 9. 流程适配（用户工作要求）

- 全程 Superpowers：brainstorm（本稿）→ 计划评审 → 任务循环 → code review → finish
- 计划与实现阶段：凡涉及库/API 用法，先用 Context7 MCP 查最新官方文档再动手
- dev-notes/ch01.md 按阶段追记，不收尾补记
- 技术选型若实现中发现走不通：停下来问用户，不自行换方案
