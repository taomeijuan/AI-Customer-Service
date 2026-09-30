# ecom-cs 电商智能客服系统

教程式逐章开发。当前：**ch01 纯对话**（多轮 + SSE 流式 + Prompt 模板 + 售后结构化抽取）。

## 快速开始

```bash
uv sync                      # 安装依赖（需 uv）
cp .env.example .env         # 填入上游 base_url / api_key / model
uv run uvicorn app.main:app --port 8000
```

## 验收

```bash
bash scripts/acceptance.sh           # 三个端到端场景（另开终端保持服务运行）
uv run pytest -q                     # 单元测试（31 个）
uv run pytest -m eval -v             # 售后抽取标注样例集（需 .env 真实上游）
```

## API

| 端点 | 说明 |
|---|---|
| `POST /api/chat/stream` | `{"conversation_id": null \| "<id>", "message": "..."}` → SSE：`meta`（发 conversation_id）→ `delta`（逐 token）→ `done`；异常走 `error` 事件；不存在的 conversation_id 返回 404 |
| `POST /api/extract` | `{"text": "<售后描述>"}` → `{"order_no", "issue_type", "expected_resolution"}`；解析失败 422 |
| `GET /healthz` | 健康检查 |

## 结构

```
app/
├── core/       config（pydantic-settings）、llm（ChatOpenAI 工厂，OpenAI 协议直连）
├── memory/     session（内存会话库）、trimmer（tiktoken 历史裁剪）
├── prompts/    客服 System Prompt + ChatPromptTemplate
├── chains/     ChatService（prompt 组装 + 流式）
├── extraction/ 售后结构化 schema + with_structured_output
└── api/        SSE 端点、extract 端点
```

设计文档：`docs/plans/2026-09-30-ch01-chat-design.md`；实施计划：`docs/plans/2026-09-30-ch01-chat-plan.md`；开发留痕：`dev-notes/ch01.md`。

> 已知边界（ch01 最简版）：会话为进程内存且无淘汰机制，长期运行内存无界，生产化需替换为带 TTL 的存储；token 预算约束「历史 + 当前输入」之和，system prompt 与回复本身不计入。
