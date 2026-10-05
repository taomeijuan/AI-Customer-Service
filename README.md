# ecom-cs 电商智能客服系统

教程式逐章开发。

- **ch01 纯对话**：多轮 + SSE 流式 + Prompt 模板 + 售后结构化抽取
- **ch02 工具链**：LangChain @tool 五工具（查订单/商品/物流 mock、FAQ 查表、建工单），模型单轮自主选工具，结果回灌流式收敛；会话与消息落 MySQL

## 快速开始

```bash
uv sync                            # 安装依赖（需 uv）
docker-compose up -d               # 起 MySQL（业务库 ecom_cs + 测试库 ecom_cs_test，首启自动建表灌数据）
cp .env.example .env               # 填入上游 base_url / api_key / model
uv run uvicorn app.main:app --port 8000
# 浏览器打开 http://localhost:8000 即聊天页
```

## 验收

```bash
bash scripts/acceptance.sh           # 四个端到端场景（另开终端保持服务运行）
uv run pytest -q -m "not eval"       # 单元测试（53 个，含真 MySQL 集成）
uv run pytest -m eval -v             # 真实上游样例集（抽取 8 条 + 工具选型 9 条）
```

## API

| 端点 | 说明 |
|---|---|
| `POST /api/chat/stream` | `{"conversation_id": null\|int, "message": "...", "user_id": "..."}` → SSE：`meta`（发整型 conversation_id）→ `tool`（工具状态帧，running/done）→ `delta`（逐 token）→ `done`；异常走 `error` 事件 |
| `POST /api/extract` | `{"text": "<售后描述>"}` → `{"order_no", "issue_type", "expected_resolution"}`；解析失败 422 |
| `GET /healthz` | 健康检查 |

## 结构

```
app/
├── core/         config（pydantic-settings）、llm（ChatOpenAI 工厂，OpenAI 协议直连）
├── db/           engine（SQLAlchemy async）、models（faq/conversations/messages/tickets 四表 ORM）
├── repositories/ conversations/messages/faq/tickets 数据访问层
├── memory/       trimmer（tiktoken 历史裁剪）
├── tools/        base（ToolRegistry：校验/超时/重试/错误包装）+ 五个 @tool
├── agents/       orchestrator（bind_tools 单轮编排：选工具→执行→回灌→流式收敛）
├── prompts/      客服 System Prompt + ChatPromptTemplate
├── chains/       ChatService（无工具流式回答单元，ch02 起由 orchestrator 调度）
├── extraction/   售后结构化 schema + with_structured_output
├── api/          SSE 端点、extract 端点
└── static/       聊天页（气泡 + 逐字渲染 + 工具徽章）
db/init/           建库建表与种子脚本（docker-entrypoint 自动执行）
```

设计文档：`docs/plans/2026-09-30-ch01-chat-design.md`；实施计划：`docs/plans/2026-09-30-ch01-chat-plan.md`；开发留痕：`dev-notes/ch01.md`。

> 已知边界（ch01 最简版）：会话为进程内存且无淘汰机制，长期运行内存无界，生产化需替换为带 TTL 的存储；token 预算约束「历史 + 当前输入」之和，system prompt 与回复本身不计入。
