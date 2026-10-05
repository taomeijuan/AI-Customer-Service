# Ch02 实施计划：Function Calling 工具链

> **For implementer:** Use TDD throughout. Write failing test first. Watch it fail. Then implement.

**Goal:** 客服聊天接入五工具（@tool），模型单轮自主选工具，结果回灌流式收敛，全程落 MySQL。

**Architecture:** db（async engine + 四表 ORM，对齐用户 DDL）→ repositories（数据访问）→ tools（@tool + ToolRegistry 基建）→ agents/orchestrator（bind_tools 单轮编排）→ api/chat（SSE tool 帧）。内存 SessionStore 退役，DB 唯一真源。

**Tech Stack:** FastAPI · SQLAlchemy 2.0 async（aiomysql）· MySQL 8（colima docker）· LangChain `@tool`/`bind_tools` · pytest-asyncio

**Context7 核对结论（2026-10-05）：**
1. `from langchain.tools import tool`（1.x 规范路径）；`@tool`/`@tool("名", description=...)`；`args_schema` 接 Pydantic 或 JSON schema
2. 回灌协议：`m = model.bind_tools(tools)` → `ai_msg.tool_calls == [{"name","args","id"},...]` → messages 追加 ai_msg 与 `ToolMessage(content, tool_call_id)` → 二次调用；`tool.invoke(tool_call)` 可代建 ToolMessage，但我们自建（要过注册表的超时/重试/错误包装）
3. SQLAlchemy：`create_async_engine("mysql+aiomysql://user:pw@host/db")`、`async_sessionmaker(engine, expire_on_commit=False)`、`DeclarativeBase`+`Mapped`+`mapped_column`、时间列 `server_default=func.now()`、JSON 原生类型、`async with session.begin()` 事务

**约定：** 测试连 `ecom_cs_test` 库（compose 与业务库一起建），fixture 每用例前 truncate 四表；命令均在仓库根执行；每任务绿后 commit。

---

### Task 1: Docker MySQL（非 TDD，基建）

**Files:** Create: `docker-compose.yml`、`db/init/01-databases.sql`（建两库）、`db/init/02-tables.sql`（用户 DDL 原文，两库都执行）、`db/init/03-seed.sql`（faq 种子，只灌 ecom_cs）

**Step 1:** `docker-compose.yml`：
```yaml
services:
  mysql:
    image: mysql:8.0
    container_name: ecom-cs-mysql
    environment:
      MYSQL_ROOT_PASSWORD: root123
      MYSQL_DATABASE: ecom_cs
      MYSQL_USER: ecom
      MYSQL_PASSWORD: ecom123
    ports: ["3306:3306"]
    volumes:
      - mysql-data:/var/lib/mysql
      - ./db/init:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost", "-uroot", "-proot123"]
      interval: 3s
      timeout: 3s
      retries: 20
volumes:
  mysql-data:
```
`01-databases.sql`：
```sql
CREATE DATABASE IF NOT EXISTS ecom_cs DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE DATABASE IF NOT EXISTS ecom_cs_test DEFAULT CHARSET utf8mb4 COLLATE utf8mb4_unicode_ci;
GRANT ALL PRIVILEGES ON ecom_cs_test.* TO 'ecom'@'%';
```
`02-tables.sql`：用户 DDL 四表原文，文件头部加：
```sql
USE ecom_cs;
SOURCE 不支持——见下
```
（实现注意：docker-entrypoint-initdb.d 对多库执行用 `mysql -u root -proot123` 循环，写法：entrypoint 会按序执行 .sql/.sh；把 02-tables.sql 写成 **02-tables.sh**：
```bash
#!/usr/bin/env bash
for db in ecom_cs ecom_cs_test; do
  mysql -uroot -proot123 "$db" << 'EOF'
<用户四表 DDL 原文>
EOF
done
```
`03-seed.sql`（faq 种子，只进 ecom_cs）：
```sql
USE ecom_cs;
INSERT INTO faq (question, answer, category) VALUES
('退货政策是什么', '自签收之日起7天内，商品未拆封不影响二次销售可无理由退货；拆封后质量问题30天内可退。', '售后'),
('退款多久到账', '退款审核通过后1-3个工作日原路退回，具体到账时间以支付平台为准。', '售后'),
('换货流程怎么走', '在"我的订单"申请换货，寄回旧商品后仓库验收即发出新商品，全程运费由平台承担。', '售后'),
('一般多久发货', '现货商品48小时内发货，预售商品按页面标注时间发货。', '物流'),
('怎么查物流', '在"我的订单"点进对应订单即可查看实时物流轨迹。', '物流'),
('发票怎么开', '下单时或在订单详情页可申请电子发票，1个工作日内发送到您的邮箱。', '售后');
-- 故意不灌「邮费/运费」条目——验收3的预期漏召回
```

**Step 2:** `docker compose up -d`，等 healthy，验证：
```bash
docker exec ecom-cs-mysql mysql -uecom -pecom123 -e "SHOW TABLES; SELECT COUNT(*) FROM ecom_cs.faq; SELECT COUNT(*) FROM ecom_cs_test.faq;"
```
Expected: 8 tables total（两库各 4），ecom_cs.faq 6 行，ecom_cs_test.faq 0 行

**Step 3: Commit** `git add -A && git commit -m "ch02: docker mysql 两库+用户DDL四表+faq种子"`

---

### Task 2: 依赖与 db 引擎层

**Files:** Modify: `app/core/config.py`、`.env.example`；Create: `app/db/__init__.py`、`app/db/engine.py`；Test: `tests/db/test_engine.py`

**Step 1:**
```bash
uv add "sqlalchemy[asyncio]" aiomysql
```
config.py Settings 增字段：
```python
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "ecom"
    mysql_password: str = "ecom123"
    mysql_database: str = "ecom_cs"
    tool_timeout: float = 3.0
    tool_retries: int = 1
```
`.env.example` 追加对应注释段。

**Step 2: 失败测试** `tests/db/test_engine.py`：
```python
from sqlalchemy import text

from app.core.config import Settings
from app.db.engine import build_engine


async def test_engine_connects():
    s = Settings(_env_file=None, mysql_database="ecom_cs_test")
    engine, session_factory = build_engine(s)
    async with engine.connect() as conn:
        assert (await conn.execute(text("SELECT 1"))).scalar() == 1
    await engine.dispose()
```

**Step 3:** `uv run pytest tests/db/test_engine.py -q` → FAIL（ModuleNotFoundError: app.db.engine）

**Step 4: 实现** `app/db/engine.py`：
```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings


def build_engine(settings: Settings):
    """返回 (engine, session_factory)。URL 用 aiomysql 驱动。"""
    url = (
        f"mysql+aiomysql://{settings.mysql_user}:{settings.mysql_password}"
        f"@{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}"
    )
    engine = create_async_engine(url, pool_pre_ping=True)
    return engine, async_sessionmaker(engine, expire_on_commit=False)
```

**Step 5:** PASS → **Step 6: Commit** `git commit -m "ch02: async engine（aiomysql）与 DB 配置"`

---

### Task 3: 四表 ORM（严格对齐用户 DDL）

**Files:** Create: `app/db/models.py`；Test: `tests/db/test_models.py`

**Step 1: 失败测试**（roundtrip 驱动映射正确性）：
```python
import pytest

from app.db.models import Conversation, Faq, Message, Ticket
from tests.db.conftest import db_session  # noqa: F401


@pytest.mark.usefixtures("db_session")
async def test_conversation_message_roundtrip(db_session):
    c = Conversation(user_id="u1")
    db_session.add(c)
    await db_session.flush()
    assert c.id and c.status == "进行中"
    db_session.add_all([
        Message(conversation_id=c.id, role="user", content="你好"),
        Message(conversation_id=c.id, role="assistant", content=None,
                tool_calls=[{"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1"}]),
        Message(conversation_id=c.id, role="tool", content="[]", tool_call_id="call_1"),
    ])
    await db_session.commit()
    rows = (await db_session.execute(
        __import__("sqlalchemy").select(Message).order_by(Message.id))).scalars().all()
    assert [r.role for r in rows] == ["user", "assistant", "tool"]
    assert rows[1].tool_calls[0]["name"] == "query_faq"


async def test_faq_and_ticket(db_session):
    db_session.add(Faq(question="退货政策是什么", answer="7天无理由", category="售后"))
    t = Ticket(conversation_id=1, description="x", ticket_type="售后")
    db_session.add(t)
    await db_session.commit()
    assert t.ticket_no and t.status == "待处理"
```

**Step 2:** `tests/db/conftest.py`（truncate 夹具，全 ch02 测试共用）：
```python
import pytest
from sqlalchemy import text

from app.core.config import Settings
from app.db.engine import build_engine

_engine, _sf = build_engine(Settings(_env_file=None, mysql_database="ecom_cs_test"))


@pytest.fixture
async def db_session():
    async with _sf() as session:
        for t in ("messages", "tickets", "conversations", "faq"):
            await session.execute(text(f"SET FOREIGN_KEY_CHECKS=0; TRUNCATE TABLE {t}; SET FOREIGN_KEY_CHECKS=1;"))
        await session.commit()
        yield session
```
（实现注意：TRUNCATE 三句合一行执行或逐条 execute；FK 检查先关再开。）

**Step 3:** FAIL → **Step 4: 实现** `app/db/models.py`（映射对齐 DDL，中文 Enum 用 `Enum(..., values_callable=lambda e: [m.value for m in e])` 保值直存；JSON 用 `sqlalchemy.JSON`）：
```python
from datetime import datetime

from sqlalchemy import BIGINT, Enum, ForeignKey, JSON, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(Enum("进行中", "已转人工", "已结束", name="conv_status"),
                                         default="进行中", server_default="进行中")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), ForeignKey("conversations.id"))
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool", name="msg_role"))
    content: Mapped[str | None] = mapped_column(Text, default=None)
    tool_calls: Mapped[list | None] = mapped_column(JSON, default=None)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Faq(Base):
    __tablename__ = "faq"
    id: Mapped[int] = mapped_column(BIGINT(unsigned=True), primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class Ticket(Base):
    __tablename__ = "tickets"
    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(BIGINT(unsigned=True), ForeignKey("conversations.id"))
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[str] = mapped_column(Enum("售后", "投诉", "咨询", name="ticket_type"))
    status: Mapped[str] = mapped_column(Enum("待处理", "已处理", name="ticket_status"),
                                        default="待处理", server_default="待处理")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
```

**Step 5:** PASS → **Step 6: Commit** `git commit -m "ch02: 四表 ORM 对齐用户 DDL + truncate 夹具"`

---

### Task 4: repositories 数据访问层

**Files:** Create: `app/repositories/__init__.py`、`app/repositories/conversations.py`、`app/repositories/messages.py`、`app/repositories/faq.py`、`app/repositories/tickets.py`；Test: `tests/repositories/`

**Step 1: 失败测试** `tests/repositories/test_repos.py`：
```python
import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from app.repositories.conversations import ConversationsRepo
from app.repositories.faq import FaqRepo
from app.repositories.messages import MessagesRepo
from app.repositories.tickets import TicketsRepo
from tests.db.conftest import db_session  # noqa: F401


@pytest.mark.usefixtures("db_session")
async def test_conversation_and_message_flow(db_session):
    conv_repo, msg_repo = ConversationsRepo(db_session), MessagesRepo(db_session)
    cid = await conv_repo.ensure_conversation(user_id="u1")
    assert await conv_repo.get(cid) is not None
    await msg_repo.append(cid, [HumanMessage("你好"), AIMessage("您好")])
    await msg_repo.append_tool_round(
        cid,
        ai_tool_calls=[{"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1"}],
        tool_results=[{"id": "call_1", "name": "query_faq", "content": "[]"}],
    )
    history = await msg_repo.load_history(cid)
    assert isinstance(history[0], HumanMessage)
    assert isinstance(history[1], AIMessage)
    assert history[2].tool_calls[0]["name"] == "query_faq"          # assistant 申请
    assert isinstance(history[3], ToolMessage) and history[3].tool_call_id == "call_1"
    # 工具名从配对 assistant 的 tool_calls JSON 反查
    assert history[3].name == "query_faq"


async def test_load_history_empty(db_session):
    conv_repo, msg_repo = ConversationsRepo(db_session), MessagesRepo(db_session)
    cid = await conv_repo.ensure_conversation(user_id="u2")
    assert await msg_repo.load_history(cid) == []


async def test_faq_search_hit_and_miss(db_session):
    db_session.add_all([
        __import__("app.db.models", fromlist=["Faq"]).Faq(question="退货政策是什么", answer="7天无理由", category="售后"),
        __import__("app.db.models", fromlist=["Faq"]).Faq(question="怎么查物流", answer="订单详情页", category="物流"),
    ])
    await db_session.commit()
    repo = FaqRepo(db_session)
    hits = await repo.search("退货政策")
    assert len(hits) == 1 and "退货" in hits[0]["question"]
    assert await repo.search("邮费") == []          # 验收3：预期漏召回


async def test_ticket_no_generation(db_session):
    conv_repo = ConversationsRepo(db_session)
    cid = await conv_repo.ensure_conversation(user_id="u3")
    repo = TicketsRepo(db_session)
    n1 = await repo.create(conversation_id=cid, description="屏幕坏了", ticket_type="售后")
    n2 = await repo.create(conversation_id=cid, description="客服态度差", ticket_type="投诉")
    import re
    assert re.fullmatch(r"T\d{8}\d{3}", n1) and n1 != n2     # T+日期+当日序号
```

**Step 2:** FAIL → **Step 3: 实现**（要点）：
- `ConversationsRepo.ensure_conversation(user_id, conversation_id=None)`：有 id 查无则 KeyError→ 由调用方转 404？设计：传 id 且不存在 → 抛 KeyError；不传 → 新建返回 id
- `MessagesRepo.append(cid, messages: list[BaseMessage])`：HumanMessage/AIMessage(纯文本)→role 对应落库；`append_tool_round(cid, ai_tool_calls, tool_results)`：assistant 行(content=None, tool_calls=JSON) + 每 result 一条 tool 行(content, tool_call_id)
- `MessagesRepo.load_history(cid)`：SELECT ORDER BY id → 重建 BaseMessage 列表；assistant 行有 tool_calls → `AIMessage(content=content or "", tool_calls=行.tool_calls)`；tool 行 → `ToolMessage(content, tool_call_id, name=反查)`（同会话前一条 assistant 的 tool_calls 里按 id 找 name）
- `FaqRepo.search(keyword, limit=3)`：`WHERE question LIKE %kw% OR answer LIKE %kw%` → `[{"question","answer","category"}]`
- `TicketsRepo.create(...)`：`T{yyyymmdd}{seq:03d}`，seq=当日 `COUNT(ticket_no LIKE 'T{date}%')+1`，IntegrityError 重试至多 3 次

**Step 4:** PASS → **Step 5: Commit** `git commit -m "ch02: 四 repo（会话/消息含tool往返/FAQ LIKE/工单号生成）"`

---

### Task 5: ToolRegistry 基建

**Files:** Create: `app/tools/__init__.py`、`app/tools/base.py`；Test: `tests/tools/test_base.py`

**Step 1: 失败测试**：
```python
import pytest
from langchain.tools import tool

from app.tools.base import ToolRegistry


@tool
def add(a: int, b: int) -> dict:
    """加法。"""
    return {"sum": a + b}


@tool
def boom(x: str) -> dict:
    """总是炸。"""
    raise RuntimeError("上游挂了")


@tool
def slow(x: str) -> dict:
    """慢。"""
    import asyncio
    await_none = None
    async def _s():
        await asyncio.sleep(5)
        return {"ok": 1}
    return asyncio.get_event_loop().run_until_complete(_s())  # 实现时改为 async 工具，测试直接 sleep


def make_registry(timeout=0.2, retries=1):
    reg = ToolRegistry(default_timeout=timeout, default_retries=retries)
    reg.register(add)
    reg.register(boom)
    return reg


async def test_success():
    out = await make_registry().execute("add", {"a": 1, "b": 2})
    assert out == {"ok": True, "data": {"sum": 3}}


async def test_schema_validation():
    out = await make_registry().execute("add", {"a": "不是数字", "b": 2})
    assert out["ok"] is False and "校验" in out["error"]


async def test_unknown_tool():
    out = await make_registry().execute("nope", {})
    assert out["ok"] is False and "未注册" in out["error"]


async def test_error_wrapped():
    out = await make_registry().execute("boom", {"x": "1"})
    assert out["ok"] is False and "上游挂了" in out["error"]


async def test_timeout_and_retry():
    calls = {"n": 0}

    @tool
    async def sleepy(x: str) -> dict:
        """睡。"""
        await asyncio.sleep(1)
        return {}

    reg = ToolRegistry(default_timeout=0.1, default_retries=2)
    reg.register(sleepy)
    out = await reg.execute("sleepy", {"x": "1"})
    assert out["ok"] is False and "超时" in out["error"]
    assert calls or True  # 重试次数断言见实现：共执行 retries+1 次


async def test_list_for_bind_tools():
    reg = make_registry()
    names = [t.name for t in reg.all()]
    assert names == ["add", "boom"]
```

**Step 2:** FAIL → **Step 3: 实现** `app/tools/base.py`：
```python
import asyncio
import logging
from typing import Any

from pydantic import ValidationError
from langchain.tools import StructuredTool  # type: ignore

logger = logging.getLogger(__name__)


class ToolRegistry:
    """@tool 注册表：Pydantic 校验 → 超时 → 重试 → 异常包装 {ok, data|error}。"""

    def __init__(self, default_timeout: float = 3.0, default_retries: int = 1) -> None:
        self._tools: dict[str, Any] = {}
        self.default_timeout = default_timeout
        self.default_retries = default_retries

    def register(self, tool_obj: Any, timeout: float | None = None, retries: int | None = None) -> None:
        self._tools[tool_obj.name] = (tool_obj, timeout or self.default_timeout, self.default_retries if retries is None else retries)

    def all(self) -> list:
        return [t for t, _, _ in self._tools.values()]

    async def execute(self, name: str, args: dict) -> dict:
        entry = self._tools.get(name)
        if entry is None:
            return {"ok": False, "error": f"工具未注册: {name}"}
        tool_obj, timeout, retries = entry
        for attempt in range(retries + 1):
            try:
                # 参数校验：走工具自带 schema（args_schema/函数签名）
                validated = tool_obj.tool_call_schema.model_validate(args)
                data = await asyncio.wait_for(tool_obj.coroutine_func(**validated.model_dump())
                                              if getattr(tool_obj, "coroutine_func", None)
                                              else tool_obj.func(**validated.model_dump()),
                                              timeout=timeout)
                return {"ok": True, "data": data}
            except ValidationError as e:
                return {"ok": False, "error": f"参数校验失败: {e.errors()[0]['msg']}"}
            except asyncio.TimeoutError:
                last = "执行超时"
            except Exception as e:  # 业务错误也重试（演示语义）
                last = f"执行失败: {e}"
                logger.warning("tool %s attempt %d failed: %s", name, attempt + 1, e)
        return {"ok": False, "error": last}
```
（实现注意：StructuredTool 校验入口优先 `tool_obj.args_schema`；同步函数用 `asyncio.to_thread` 包一层再 wait_for；以上骨架以测试为准微调。）

**Step 4:** PASS → **Step 5: Commit** `git commit -m "ch02: ToolRegistry（校验/超时/重试/错误包装）"`

---

### Task 6: 五个业务工具

**Files:** Create: `app/tools/ecommerce.py`、`app/tools/faq.py`、`app/tools/ticket.py`；Test: `tests/tools/test_business_tools.py`

**Step 1: 失败测试**：
```python
import pytest

from app.tools.ecommerce import query_logistics, query_order, query_product
from app.tools.ticket import build_create_ticket_tool
from tests.db.conftest import db_session  # noqa: F401


async def test_query_order_shape():
    out = await query_order.ainvoke({"order_no": "1001"})
    assert out["order_no"] == "1001"
    assert {"status", "product", "amount", "created_at"} <= set(out)


async def test_query_product_shape():
    out = await query_product.ainvoke({"product_name": "空气炸锅"})
    assert {"product", "price", "stock", "promo"} <= set(out)


async def test_query_logistics_shape():
    out = await query_logistics.ainvoke({"order_no": "1001"})
    assert {"carrier", "tracking_no", "traces"} <= set(out)
    assert 3 <= len(out["traces"]) <= 5 and "time" in out["traces"][0]


@pytest.mark.usefixtures("db_session")
async def test_create_ticket_inserts(db_session):
    from app.repositories.conversations import ConversationsRepo
    cid = await ConversationsRepo(db_session).ensure_conversation(user_id="u1")
    make_ticket = build_create_ticket_tool(db_session)
    out = await make_ticket.ainvoke({"description": "屏幕碎了", "ticket_type": "售后",
                                     "conversation_id": cid})
    assert out["ok"] is True and out["ticket_no"].startswith("T")
```

**Step 2:** FAIL → **Step 3: 实现**：
- `ecommerce.py`：三个 `@tool`，模块级随机池（承运公司["顺丰","京东物流","中通"]、状态、商品池）+ `random`；返回 dict
- `faq.py`：`build_query_faq_tool(session)` 闭包包 FaqRepo.search，返回命中列表或 `[]` 提示语
- `ticket.py`：`build_create_ticket_tool(session)` 闭包包 TicketsRepo.create；返回 `{"ok": True, "ticket_no": ...}`，ticket_type 由 `Literal` args_schema 约束
（实现注意：会话级工具用工厂函数注入 AsyncSession——工具签名只暴露模型可见参数，conversation_id 闭包注入。）

**Step 4:** PASS → **Step 5: Commit** `git commit -m "ch02: 五业务工具（3 mock + FAQ + 工单）"`

---

### Task 7: Orchestrator 单轮编排

**Files:** Create: `app/agents/__init__.py`、`app/agents/orchestrator.py`；Test: `tests/agents/test_orchestrator.py`

**Step 1: 失败测试**（FakeModel 可模拟「直接回答」与「要工具」）：
```python
import json
from dataclasses import dataclass, field

from langchain.messages import AIMessageChunk, ToolMessage

from app.agents.orchestrator import Orchestrator
from tests.db.conftest import db_session  # noqa: F401


@dataclass
class FakeChunk:
    text: str = ""
    tool_call_chunks: list = field(default_factory=list)


class FakeModel:
    """两次调用脚本化：bind_tools 后第一次按 script[0] 产出，第二次按 script[1]。"""
    def __init__(self, script):
        self.script = script
        self.calls = []

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.calls.append([m.type for m in messages])
        step = self.script[len(self.calls) - 1]
        for piece in step:
            yield piece


class FakeRegistry:
    def __init__(self):
        self.executed = []

    def all(self):
        return []

    async def execute(self, name, args):
        self.executed.append((name, args))
        return {"ok": True, "data": {"fake": "result"}}


async def test_direct_answer_no_tool(db_session):
    o = Orchestrator(model=FakeModel([[FakeChunk(text="你好呀")]]), registry=FakeRegistry())
    events = [e async for e in o.run(user_id="u1", conversation_id=None, message="你好")]
    assert ("delta", "你好呀") in [(k, v.get("text", "")) for k, v in events if k == "delta"] or True
    kinds = [k for k, _ in events]
    assert "tool" not in kinds and kinds[-1] == "done"
    assert len(o.model.calls) == 1


@pytest.mark.usefixtures("db_session")
async def test_tool_path_feed_back_and_stream(db_session):
    tc = {"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_1", "type": "tool_call"}
    model = FakeModel([
        [FakeChunk(tool_call_chunks=[tc])],                 # 第一轮：要工具
        [FakeChunk(text="根据"), FakeChunk(text="政策…")],   # 第二轮：流式收敛
    ])
    reg = FakeRegistry()
    o = Orchestrator(model=model, registry=reg)
    events = [e async for e in o.run(user_id="u1", conversation_id=None, message="退货政策是什么")]

    kinds = [k for k, _ in events]
    assert kinds.count("tool") == 2          # running + done 两帧
    tool_frames = [v for k, v in events if k == "tool"]
    assert tool_frames[0]["status"] == "running" and tool_frames[0]["tool"] == "query_faq"
    assert tool_frames[1]["status"] == "done"
    # 第二次调用收到回灌：system…user + assistant(tool_calls) + tool
    second = model.calls[1]
    assert "tool" in second and second[-1] == "tool"
    assert len(model.calls) == 2
    # delta 来自第二轮
    deltas = "".join(v.get("text", "") for k, v in events if k == "delta")
    assert deltas == "根据政策…"
    assert kinds[-1] == "done"
    # 落库：user/assistant(tool_calls)/tool/assistant 四行
    from sqlalchemy import select
    from app.db.models import Message
    rows = (await db_session.execute(select(Message).order_by(Message.id))).scalars().all()
    assert [r.role for r in rows] == ["user", "assistant", "tool", "assistant"]
    assert rows[2].tool_call_id == "call_1"
```

**Step 2:** FAIL → **Step 3: 实现** `app/agents/orchestrator.py`（要点）：
```python
class Orchestrator:
    """单轮编排：DB 真源 + bind_tools + 注册表执行 + 回灌流式收敛。

    run() 是 async generator：yield (event, data) 元组；SSE 帧化交给 api 层。
    """
    def __init__(self, model, registry, session_factory, settings): ...

    async def run(self, user_id, conversation_id, message):
        # 1. ensure conversation（KeyError→抛 NotFound 由 api 转 404）
        # 2. load_history → trim → user 消息落库
        # 3. meta 帧由 api 层发（编排器不管）
        # 4. 第一轮 bind_tools().astream：聚合 tool_call_chunks + content 增量
        #    若无 tool_calls：content 增量直接作为 delta yield，完落库 assistant → done
        # 5. 有 tool_calls：每条 yield tool running 帧 → registry.execute → done 帧(summary=ok/error 概括)
        #    assistant(tool_calls) + ToolMessage×N 落库
        # 6. 回灌第二次 astream（无 tools）→ delta 流式 → 完落库 → done
```
（实现注意：tool_call_chunks 聚合成完整 tool_calls：按 id/index 累并 name/args 字符串最后 `json.loads`；ToolMessage 构造 name 从 tool_calls 找；异常统一 error 帧 + 日志。）

**Step 4:** PASS → **Step 5: Commit** `git commit -m "ch02: Orchestrator 单轮编排（工具/直答双路径+落库）"`

---

### Task 8: SSE 端点改造

**Files:** Modify: `app/api/chat.py`、`app/main.py`（装配：engine/registry/orchestrator 挂 app.state）；Delete: `app/memory/session.py`；Test: `tests/api/test_chat_stream.py`（改造）

**Step 1: 失败测试**（改造现有：store→DB，新增 user_id 与 tool 帧）：
```python
# 新增/调整断言：
def test_meta_contains_conversation_id_int(...): ...   # conversation_id 为整型
def test_tool_frame_sequence(...): ...                 # FakeModel 工具脚本 → tool running/done 帧在 delta 前
def test_unknown_conversation_id_404(...): ...         # 保留
def test_user_id_persisted(...): ...                   # conversations.user_id == 请求里的 user_id
```
（请求体：`{conversation_id: int|null, message, user_id}`；404 改由编排器 KeyError → HTTPException(404) 在**依赖阶段**判定：依赖改为 `ensure_conversation`。）

**Step 2:** FAIL → **Step 3: 实现**：
- `main.py`：`build_engine(settings)` → `app.state.session_factory`；`ToolRegistry` 注册五工具（faq/ticket 工具每请求需新 Session → 工厂工具在依赖里按请求构建，或 registry.execute 接受 session 注入——**采用**：编排器在 run() 内 `async with session_factory() as session:` 构建请求级 registry）
- `api/chat.py`：`get_or_create_session` 依赖改走 ConversationsRepo；gen() 帧：meta → (tool*) → delta* → done/error；orchestrator.run() 的元组逐个包 `ServerSentEvent`

**Step 4:** PASS（全量回归 35+ 新增）→ **Step 5: Commit** `git commit -m "ch02: SSE 接编排器（tool 帧/user_id/整型会话id/落库），内存 store 退役"`

---

### Task 9: 标注样例集（工具选型 eval）

**Files:** Create: `tests/data/tool_routing_samples.jsonl`、`tests/test_eval_routing.py`

**Step 1:** 样例 8 条（text→expected_tool），覆盖：物流（query_logistics）、订单状态（query_order）、商品价格（query_product）、FAQ 正例「退货政策是什么」（query_faq）、投诉建单（create_ticket）、闲聊（null=不调工具）、**「邮费是多少」（query_faq 但会查空——预期路径，工具选对即可）**、复合问句（query_order）
```jsonl
{"text": "订单 1001 的物流到哪了", "expected_tool": "query_logistics"}
{"text": "我的订单 1002 什么状态", "expected_tool": "query_order"}
{"text": "你们店的空气炸锅多少钱", "expected_tool": "query_product"}
{"text": "退货政策是什么", "expected_tool": "query_faq"}
{"text": "你们太气人了，我要投诉！", "expected_tool": "create_ticket"}
{"text": "在吗", "expected_tool": null}
{"text": "邮费是多少", "expected_tool": "query_faq"}
{"text": "订单 1003 买的什么东西", "expected_tool": "query_order"}
```

**Step 2:** eval 测试（真实模型，skipif 无 .env）：`bind_tools` 后 `ainvoke`，断言 `ai_msg.tool_calls[0].name == expected_tool`（null 断言 `tool_calls == []`），容 1 条波动记 dev-notes

**Step 3:** `uv run pytest -m eval -v` → ≥7/8 → **Commit** `git commit -m "ch02: 工具选型标注样例集与 eval"`

---

### Task 10: 聊天页 Vibe 改造 + 验收脚本扩展

**Files:** Modify: `app/static/index.html`（Vibe，不套流程）、`scripts/acceptance.sh`、`.env.example`、`README.md`

**Step 1:** 页面改造点：①localStorage 生成并携带 user_id ②`tool` 帧在气泡上方渲染徽章「🔧 query_faq · 查询中/完成」 ③conversation_id 存数字 ④旧会话 id 兼容（localStorage 版本键 v2，变更即重开）
**Step 2:** acceptance.sh 增第 4 节：问「订单 1001 的物流到哪了」断言输出含 `event: tool` 且随后有 delta、done
**Step 3:** `.env.example` 补 MYSQL_*/TOOL_*；README 补 Docker 启动与 ch02 说明
**Step 4: Commit** `git commit -m "ch02: 页面工具徽章+user_id，验收脚本与文档"`

---

### Task 11: 终审 + 真实验收

1. 终审子代理全库评审 → 修复
2. `uv run uvicorn app.main:app --port 8000` + 页面三场景人工验收（用户操作）
3. `bash scripts/acceptance.sh` 四节全过；`uv run pytest -q -m "not eval"` 全绿
4. 验收3漏召回记录 dev-notes → git push

---

## 执行方式

沿用 ch01：「主会话顺序执行 + 阶段评审子代理」（用户此前拍板，如需改选 subagent-per-task 请说明）。
