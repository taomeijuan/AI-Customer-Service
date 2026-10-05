# Ch03 设计文档：知识库与向量语义检索（BGE-M3 + Milvus）

> 状态：已定稿（2026-10-05，用户批准）
> 范围：只做 dense 向量单路；不做关键词召回、混合检索、重排
> 流程：Superpowers（brainstorm → 计划 → TDD 构建 → 评审 → finish）

## 1. 需求与验收

需求：
1. 离线建库·文档处理：Markdown 知识文档按标题层级结构感知切分；超长递归切；块间加重叠且裁到最近句号；大表格按行切每块复制表头
2. 离线建库·对话挖知识：定时任务从历史客服对话分批喂 LLM 抽问答对，先进暂存表、再整体去重入库
3. 落库结构：每条知识 category/questions/answer 三格拼文本向量化；questions——商品 FAQ 与挖出 QA 填真实问法，政策手册填所在章节标题、category 填上级标题路径；四类元数据（章节路径/内容类型/是否关键条款/前后块指针）只存不进向量
4. 双写落库：MySQL knowledge_chunks 原文权威源 + Milvus 集合 knowledge；先 MySQL「pending」→ Milvus 回填 vector_id 转「done」；按主键幂等，可重跑
5. 在线检索：问题向量化 → Milvus Top-K → 替换 query_faq 关键词查表内核；入参出参契约不变

验收标准：
1. 「邮费是多少」这类换说法的问题，能召回运费说明并答对
2. 故意中断建库任务再重跑，漏向量化的块能被捡起补齐

## 2. 澄清决策（已拍板）

| 决策点 | 结论 |
|---|---|
| 嵌入模型 | Ollama 本地跑 bge-m3（OpenAI 兼容 /v1/embeddings，1024 维；规避 torch/内嵌库对 py3.14 的 wheel 风险） |
| 向量库 | Milvus standalone docker-compose（milvus+etcd+minio；colima VM 内存扩 6GB） |
| 挖知识任务 | CLI 脚本 `python -m app.jobs.mine_qa`，幂等可重跑；外部调度后续章节再接 |
| 表结构 | 用户 DDL 为准（下文 §3 原文嵌入） |

## 3. 数据模型（用户 DDL 原文，规范来源）

```sql
-- 确保中文 COMMENT 按 utf8mb4 解析(latin1 默认的 mysql client 会把中文 double-encode)
SET NAMES utf8mb4;

CREATE TABLE knowledge_chunks (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'chunk 主键,与 Milvus 集合主键对齐',
  category         VARCHAR(255)    NOT NULL                COMMENT '分类 / 上级标题路径,进向量化文本',
  questions        TEXT            NOT NULL                COMMENT '问法或本节标题,多个问法换行分隔,进向量化文本',
  answer           TEXT            NOT NULL                COMMENT '正文答案,进向量化文本',
  section_path     VARCHAR(512)    NULL                    COMMENT '章节路径,元数据,溯源用,不进向量',
  content_type     VARCHAR(32)     NULL                    COMMENT '内容类型:faq / policy / manual 等,元数据',
  is_key_clause    TINYINT(1)      NOT NULL DEFAULT 0      COMMENT '是否关键条款,0 否 1 是,元数据',
  prev_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '前一块指针,元数据',
  next_chunk_id    BIGINT UNSIGNED NULL                    COMMENT '后一块指针,元数据',
  vector_id        VARCHAR(64)     NULL                    COMMENT 'Milvus 集合 knowledge 里的主键,写入后回填',
  vectorize_status ENUM('pending','done') NOT NULL DEFAULT 'pending' COMMENT '待向量化 / 已向量化,双写幂等靠它',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  updated_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (id),
  KEY idx_category (category),
  KEY idx_vectorize_status (vectorize_status),
  CONSTRAINT fk_chunks_prev FOREIGN KEY (prev_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL,
  CONSTRAINT fk_chunks_next FOREIGN KEY (next_chunk_id) REFERENCES knowledge_chunks (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='知识库 chunk 原文权威源';

CREATE TABLE qa_extraction_staging (
  id               BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '暂存行主键',
  batch_no         VARCHAR(64)     NOT NULL                COMMENT '抽取批次号,一批几十个会话跑一次,分批防串味、按批追溯',
  source_ref       VARCHAR(255)    NULL                    COMMENT '来源会话 / 导出文件标识,溯源用,不入最终知识库',
  question         TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的用户问法',
  answer           TEXT            NOT NULL                COMMENT 'LLM 从会话抽出的客服答案',
  status           ENUM('extracted','kept','discarded') NOT NULL DEFAULT 'extracted' COMMENT '已抽出待去重 / 去重保留 / 去重丢弃',
  created_at       DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '抽取写入时间',
  PRIMARY KEY (id),
  KEY idx_batch_no (batch_no),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='历史对话抽 QA 的离线中转暂存表:分批抽取、整体去重,保留项入 knowledge_chunks,建库完成可清空';
```

适配点：
1. 两表 DDL 通过 `db/init/04-ch03.sql` + `scripts/init_ch03.sh` 手动执行（volume 已存在，entrypoint 不会重跑 init）；测试库同步建表
2. conftest truncate 列表补两表（外键表序：messages/tickets → knowledge_chunks → qa_extraction_staging → conversations/faq）
3. MySQL chunk 主键 = Milvus 集合 knowledge 主键（INT64），双写对齐的锚点

## 4. 架构

```
app/
├── knowledge/
│   ├── splitter.py     # 结构感知切分（纯函数，TDD 主战场）
│   ├── embedder.py     # Ollama /v1/embeddings 客户端（批量、重试）
│   ├── milvus_store.py # pymilvus MilvusClient：建集合/insert/search/upsert
│   └── ingest.py       # 建库编排：切分→MySQL pending→Milvus→回填→done
├── jobs/mine_qa.py     # 挖知识 CLI（python -m）
└── tools/faq.py        # query_faq 内核替换，契约不变
docs/knowledge/         # 演示文档：售后政策.md（含运费条款+关键条款）、商品FAQ.md、售后手册.md（含大表格）
scripts/
├── init_ch03.sh        # 手动执行 04-ch03.sql 到两库
└── seed_dialogs.py     # 造含「邮费/运费」问答的历史会话入 messages（挖知识的料）
docker-compose.yml      # +etcd +minio +milvus-standalone
```

## 5. 切分器规则

1. 按 Markdown 标题层级（#/##/###）切 section；chunk 三格：category=上级标题路径（如 `售后政策>退款>退款时限`）、questions=本节标题、answer=节正文
2. 超长（>800 字）递归切：`\n\n` 段落 → 句号；块间重叠 ~80 字，**重叠起点回退到最近句号**
3. 表格（`|` 连续行）按数据行切，**每块复制表头行**；行数 ≤ 阈值整表一块
4. is_key_clause：正文含「7天/30天/仅此一次/最终解释权/不支持」等 → 1
5. content_type：faq/policy/manual；prev/next 指针同文档相邻 chunk 入库后回填
6. 文档级幂等：(category, questions, answer) 精确匹配复用已有行

## 6. 双写状态机（验收 2 核心）

```
ingest：
1. SELECT * FROM knowledge_chunks WHERE vectorize_status='pending'
2. 逐块：embed(category+questions+answer 拼文本) → Milvus insert(id=chunk.id, 覆盖写)
   → UPDATE 回填 vector_id, status='done'
3. 中断重跑：done 跳过 / pending 续跑 / Milvus 有而 MySQL 未回填 → insert 同 id 覆盖补齐
```

## 7. 挖知识任务（jobs/mine_qa.py）

```
1. messages 表相邻 (user, assistant) 对按会话分组，每批 10 对，batch_no=mine-{yyyymmdd}-{uuid8}
2. LLM 抽取（with_structured_output 复用 ch02 模式）→ [{"question","answer"}]
3. 写 qa_extraction_staging (status='extracted', source_ref=conversation_id)
4. 整体两级去重（对全部 extracted 行）：
   a. 与 knowledge_chunks 现有 (questions,answer) 精确重复 → discarded
   b. 向量近重复：与库内/批内已有问法相似度 >0.95 → discarded
   c. 其余 → kept → INSERT knowledge_chunks(category='对话挖掘', questions=question,
      answer=answer, content_type='mined', status='pending') → staging 置 kept
5. 顺跑 ingest 补向量化
种子数据：scripts/seed_dialogs.py 造几段含「邮费/运费」问答的会话写 messages
```

## 8. 在线检索（query_faq 内核替换）

- 入参 keyword（=用户问题）与出参 `{count, items:[{question, answer, category}]}` 契约不变
- 实现：embed(keyword) → Milvus search top_k=RETRIEVAL_TOP_K（默认 3）、score 阈值过滤（默认 0.45，参数化）→ items（answer 存 Milvus 副本直接返回，MySQL 为权威源）

## 9. 测试策略

- **TDD（Fake embedder/milvus）**：splitter 全规则；双写状态机（回填/断点重跑补齐/同 id 覆盖）；两级去重；契约不变（改造 ch02 faq 工具测试）
- **集成（真连本地 Ollama+Milvus）**：建库→检索闭环；断点重跑演示
- **标注样例**：`tests/data/retrieval_samples.jsonl`（换说法问法→期望召回内容包含），真实链路跑——验收 1 载体
- **风险预案**：pymilvus/grpcio py3.14 无 wheel → 停下问用户是否降 3.12

## 10. 依赖与配置

- 新增：`pymilvus`；brew ollama + `ollama pull bge-m3`；docker-compose 三服务；colima --memory 6
- .env 新增：`OLLAMA_EMBED_BASE_URL=http://localhost:11434/v1`、`EMBED_MODEL=bge-m3`、`MILVUS_URI=http://localhost:19530`、`MILVUS_COLLECTION=knowledge`、`RETRIEVAL_TOP_K=3`、`RETRIEVAL_SCORE_THRESHOLD=0.45`、`MINE_BATCH_SIZE=10`

## 11. 流程适配（不变）

Context7 先查后写（pymilvus MilvusClient、Ollama embeddings API）；选型走不通停下问；dev-notes/ch03.md 逐阶段追记；Prompt/数据类（抽取 prompt、样例集）用标注样例验证替代 TDD。
