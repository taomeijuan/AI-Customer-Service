# Ch02 设计文档：Function Calling 工具链（查数据 + 工单落库）

> 状态：待定稿（2026-10-05）
> 范围：不做多轮 Agent Loop、不做向量检索/RAG；聊天页改造走 Vibe Coding
> 流程：Superpowers（brainstorm → 计划 → TDD 构建 → 评审 → finish）

## 1. 需求与验收

需求：
1. FastAPI + SQLAlchemy 分层骨架；Docker（colima）起 MySQL；四表灌测试数据
2. LangChain @tool 五工具：query_order / query_product / query_logistics（内部随机造数，不接真实接口不建表）、query_faq（SQL LIKE）、create_ticket（写 tickets）
3. 工具基建：注册管理、参数 Schema 校验、执行错误处理、超时重试、结果回灌
4. 接现有 SSE 聊天：模型定工具 → 执行 → 回灌收敛；工具执行推状态帧；聊天记录（含工具调用与结果）落 conversations/messages；气泡显示工具徽章
5. 只做单轮调用：模型调一次工具就收敛

验收标准：
1. 聊天页问「订单 1001 的物流到哪了」→ 气泡带工具徽章，按工具返回作答
2. 问「退货政策是什么」→ query_faq 查到并作答
3. 问「邮费是多少」→ LIKE 查不到（faq 表故意不灌邮费条目），模型诚实答不知道——**预期漏召回，记入 dev-notes 留给下一步升级**

## 2. 澄清决策（已拍板）

| 决策点 | 结论 |
|---|---|
| 存储架构 | **DB 为唯一真源**：ch01 内存 SessionStore 退役，每轮 DB 读历史→裁剪→调模型→回写 |
| 用户标识 | 前端首次访问生成 uuid 存 localStorage，请求带 user_id |
| 编排路线 | `llm.bind_tools()` + 手写单轮编排（ch03 升级 Agent Loop 只换循环） |
| Docker 运行时 | colima（纯 CLI，本机无 Docker 经用户拍板安装） |

## 3. 数据模型（用户 DDL 为准，原文嵌入）

> 用户 2026-10-05 提供 DDL，原样采用。全库 ENGINE=InnoDB、CHARSET=utf8mb4；
> 建表顺序：先 conversations，再依赖它的 messages / tickets。
> 商品、订单、物流走工具内 mock，不建表。

```sql
-- 会话壳:一通对话的统一身份,messages / tickets 都引用它
CREATE TABLE conversations (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '会话主键',
  user_id     VARCHAR(64)     NOT NULL                COMMENT '用户标识',
  status      ENUM('进行中','已转人工','已结束') NOT NULL DEFAULT '进行中' COMMENT '处理状态',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '开启时间',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客服会话';

-- 消息流水:一通会话底下挂 N 条,role 对齐 Chat Completions 协议
CREATE TABLE messages (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '消息主键',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '所属会话',
  role            ENUM('user','assistant','tool') NOT NULL COMMENT '角色:用户/助手/工具结果',
  content         TEXT            NULL                     COMMENT '消息正文,assistant 纯工具调用时可为空',
  tool_calls      JSON            NULL                     COMMENT 'assistant 消息带的工具调用申请单',
  tool_call_id    VARCHAR(64)     NULL                     COMMENT 'tool 消息对应的申请单 id,回灌时对号入座',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '产生时间',
  PRIMARY KEY (id),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_messages_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话消息流水';

-- FAQ 问答对:query_faq 的数据源;ch03 起检索改走向量库,这张表退居原始录入
CREATE TABLE faq (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'FAQ 主键',
  question    VARCHAR(512)    NOT NULL                COMMENT '问题',
  answer      TEXT            NOT NULL                COMMENT '答案',
  category    VARCHAR(64)     NOT NULL                COMMENT '分类',
  created_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at  DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='常见问答';

-- 人工工单:create_ticket 落地,工单号当业务主键
CREATE TABLE tickets (
  ticket_no       VARCHAR(32)     NOT NULL                COMMENT '工单号,如 T20260701008',
  conversation_id BIGINT UNSIGNED NOT NULL                COMMENT '关联会话,可倒查当时聊了什么',
  description     TEXT            NOT NULL                COMMENT '问题描述',
  ticket_type     ENUM('售后','投诉','咨询') NOT NULL     COMMENT '工单类型',
  status          ENUM('待处理','已处理') NOT NULL DEFAULT '待处理' COMMENT '处理状态',
  created_at      DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (ticket_no),
  KEY idx_conversation_id (conversation_id),
  CONSTRAINT fk_tickets_conversation FOREIGN KEY (conversation_id) REFERENCES conversations (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='人工工单';
```

**DDL 采纳后的适配点**：
1. `conversations.id` 为 BIGINT 自增 → chat API 的 `conversation_id` 改为整型；前端 ch01 存的 uuid 字符串作废（localStorage 版本号兜底重开）
2. 工单号生成：`T + yyyyMMdd + 3位当日序号`（当日最大序号 +1），唯一键冲突重试 3 次
3. messages 无 name 列：回灌时 tool 消息的工具名从配对 assistant 消息的 `tool_calls` JSON 里按 `tool_call_id` 反查，不另存
4. 中文 ENUM 值：SQLAlchemy 用 `Enum('进行中','已转人工','已结束', ...)` 精确映射

## 4. 架构（目录即分层）

```
app/
├── core/          config（+MYSQL_*/TOOL_TIMEOUT/TOOL_RETRIES）、llm（不变）
├── db/            engine.py（async engine/SessionLocal）、models.py（四表 ORM，与 DDL 严格对齐）
├── repositories/  conversations.py / messages.py / faq.py / tickets.py（SQLAlchemy 数据访问）
├── memory/        trimmer.py 保留；session.py 删除
├── tools/         base.py（ToolRegistry：注册/Pydantic 校验/wait_for 超时/重试/错误包装）
│                  ecommerce.py（query_order/query_product/query_logistics 随机造数）
│                  faq.py（query_faq LIKE 查询）、ticket.py（create_ticket 写库）
├── agents/        orchestrator.py（单轮编排，见 §6）
├── chains/        chat.py（ChatService 保留为「无工具流式回答」单元，被编排器复用）
├── api/           chat.py（SSE + tool 帧 + user_id + 全程落库）、extract.py（不变）
└── static/        index.html（Vibe：工具徽章 + user_id + conversation_id 改整型）
docker-compose.yml mysql:8.0 · 库 ecom_cs · 测试伴生库 ecom_cs_test · 3306
```

## 5. 五工具定义（LangChain @tool）

| 工具 | 参数 | 行为 |
|---|---|---|
| query_order | order_no: str | mock：随机状态（已发货/待付款/已签收…）、商品、金额、下单时间 |
| query_product | product_name: str | mock：价格、库存、促销信息 |
| query_logistics | order_no: str | mock：承运公司、运单号、随机 3-5 条轨迹节点 |
| query_faq | keyword: str | `SELECT ... WHERE question LIKE %kw% OR answer LIKE %kw% LIMIT 3` |
| create_ticket | description: str, ticket_type: '售后'\|'投诉'\|'咨询' | 写 tickets 返回工单号；conversation_id 由编排层注入 |

约定：三个 mock 工具**对任何输入都返回合理数据**（演示语义，不模拟不存在）；tool 返回值统一 dict，由 @tool 序列化后回灌。

## 6. 单轮编排（orchestrator.py 核心流程）

```
1. conversations 确保存在（无则建，status=进行中）→ DB 读全量历史 → 重建 BaseMessage 列表
   （assistant.tool_calls JSON → AIMessage(tool_calls=...)；tool 行 → ToolMessage，名字反查）
   → trimmer 裁剪（ch01 复用）→ user 消息落库
2. 第一次调用 llm.bind_tools(tools).astream()：边聚合边判首 chunk
   ├─ 无 tool_call_chunks → 普通回答，delta 流式直出（ch01 老路），完落库 assistant → done
   └─ tool_call_chunks → 工具模式：
       a. SSE 推 tool 帧 {tool, args, status: running}（每个工具调用一条）
       b. 注册表执行（校验→超时→重试→错误包装）
       c. 推 tool 帧 {status: done, summary}；assistant(tool_calls) + tool(结果) 落库
       d. 回灌：原 messages + AIMessage(tool_calls) + ToolMessage(结果)
3. 第二次调用【不绑 tools】→ astream 流式收敛（保证逐 token）→ 完整 assistant 落库 → done 帧
```

SSE 事件：`meta{conversation_id}` → `tool{tool,args,status,summary?}`* → `delta{text}`* → `done` / `error`；异常路径沿用 ch01（error 帧 + 日志，空会话清理改由 DB 无行自然表达）。

已知边界（记录不处理）：个别上游 tool_calls 与正文混发时，正文首段可能被吞。

## 7. 测试策略

- **TDD（真 MySQL）**：Docker 起 `ecom_cs`（开发）与 `ecom_cs_test`（测试，每次 truncate）伴生库
  - 四表 repo：建会话/落消息/重建历史（含 tool_calls JSON 往返）/工单号生成与冲突重试
  - tools：query_faq 命中与不命中、create_ticket 落库且 conversation 外键成立、base 的超时/重试/校验失败/错误包装
  - orchestrator：FakeModel 驱动「调工具→回灌→收敛」与「直答」两路径；断言回灌消息序列、SSE tool 帧序列、落库行数
- **标注样例（Prompt/数据类替代 TDD）**：`tests/data/tool_routing_samples.jsonl` 8 条（问句→期望工具，含一条预期空转），真实模型断言选型，容 1 条波动
- **验收**：聊天页三场景人工过 + acceptance.sh 扩展（curl 检查 tool 帧存在与 done 收敛）

## 8. 依赖与配置

- 新增：`sqlalchemy[asyncio]`、`aiomysql`、`docker-compose.yml`
- .env 新增：`MYSQL_HOST/PORT/USER/PASSWORD/DATABASE`、`TOOL_TIMEOUT=3`、`TOOL_RETRIES=1`
- faq 种子数据：退货政策、退款时限、发货时间、换货流程等 ≥6 条，**不含邮费/运费**

## 9. 流程适配（不变）

Context7 先查后写（SQLAlchemy 2.0 async、@tool、ENUM 映射）；技术选型走不通停下问；dev-notes/ch02.md 逐阶段追记；聊天页改造 Vibe Coding。
