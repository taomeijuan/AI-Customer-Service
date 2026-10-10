from datetime import datetime

from sqlalchemy import BigInteger, Boolean, Enum, ForeignKey, JSON, Numeric, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(
        Enum("进行中", "已转人工", "已结束", name="conv_status"),
        default="进行中",
        server_default="进行中",
    )
    # ch07 三层上下文：梗概投影 + 两个边界锚（NULL=未设；边界用消息 id 表达，不搬数据）
    summary: Mapped[str | None] = mapped_column(Text, default=None)
    summary_upto_msg_id: Mapped[int | None] = mapped_column(BigInteger, default=None)
    layer1_from_msg_id: Mapped[int | None] = mapped_column(BigInteger, default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class ConversationSummary(Base):
    """ch07 分段摘要：一段一行只追加，压完不回炉重压。"""

    __tablename__ = "conversation_summaries"
    __table_args__ = (UniqueConstraint("conversation_id", "seq", name="uk_conv_seq"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id")
    )
    seq: Mapped[int] = mapped_column()  # 第几段，从 1 开始
    from_msg_id: Mapped[int] = mapped_column(BigInteger)
    upto_msg_id: Mapped[int] = mapped_column(BigInteger)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id")
    )
    role: Mapped[str] = mapped_column(Enum("user", "assistant", "tool", name="msg_role"))
    content: Mapped[str | None] = mapped_column(Text, default=None)
    tool_calls: Mapped[list | None] = mapped_column(JSON, default=None)
    tool_call_id: Mapped[str | None] = mapped_column(String(64), default=None)
    citations: Mapped[list | None] = mapped_column(JSON, default=None)  # ch07 回复的引用快照（侧栏回载还原可点击引用）
    options: Mapped[dict | None] = mapped_column(JSON, default=None)  # ch07 按钮组快照（回载还原可点按钮）
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Faq(Base):
    __tablename__ = "faq"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(String(512))
    answer: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class Ticket(Base):
    __tablename__ = "tickets"
    ticket_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id")
    )
    description: Mapped[str] = mapped_column(Text)
    ticket_type: Mapped[str] = mapped_column(Enum("售后", "投诉", "咨询", name="ticket_type"))
    status: Mapped[str] = mapped_column(
        Enum("待处理", "已处理", name="ticket_status"),
        default="待处理",
        server_default="待处理",
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class RefundOrder(Base):
    """退款单（ch06，DDL 用户授权自定：db/init/06-ch06.sql）。"""

    __tablename__ = "refund_orders"
    refund_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id")
    )
    order_no: Mapped[str] = mapped_column(String(32))
    reason_category: Mapped[str] = mapped_column(
        Enum("七天无理由", "质量问题", "少件", "与描述不符", "其他", name="refund_reason")
    )
    amount: Mapped[float] = mapped_column(Numeric(10, 2))  # 对齐 DDL decimal(10,2)
    status: Mapped[str] = mapped_column(
        Enum("待审核", "已同意", "已拒绝", "已退款", name="refund_status"),
        default="待审核",
        server_default="待审核",
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class KnowledgeChunk(Base):
    """知识库 chunk 原文权威源（ch03，用户 DDL 对齐）。"""

    __tablename__ = "knowledge_chunks"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(255))
    questions: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str | None] = mapped_column(String(512), default=None)
    content_type: Mapped[str | None] = mapped_column(String(32), default=None)
    is_key_clause: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), default=None
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("knowledge_chunks.id", ondelete="SET NULL"), default=None
    )
    vector_id: Mapped[str | None] = mapped_column(String(64), default=None)
    vectorize_status: Mapped[str] = mapped_column(
        Enum("pending", "done", name="vectorize_status"),
        default="pending",
        server_default="pending",
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class QaExtractionStaging(Base):
    """历史对话抽 QA 的暂存表（ch03，用户 DDL 对齐）。"""

    __tablename__ = "qa_extraction_staging"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    batch_no: Mapped[str] = mapped_column(String(64))
    source_ref: Mapped[str | None] = mapped_column(String(255), default=None)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Enum("extracted", "kept", "discarded", name="qa_staging_status"),
        default="extracted",
        server_default="extracted",
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class LowConfidenceQuestion(Base):
    """低置信度问题池（ch04，用户 DDL 对齐）。ch09 数据飞轮入口。"""

    __tablename__ = "low_confidence_questions"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("conversations.id"), default=None
    )
    raw_question: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(
        Enum(
            "retrieval_low_conf",
            "self_check",
            "user_feedback",
            "intent_low_conf",  # ch06：与 db/init/06-ch06.sql ALTER 保持一致
            name="low_conf_source",
        )
    )
    reason: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class FaithCase(Base):
    """忠实度编造个案台账（ch04，用户 DDL 对齐）：一题一行，跨轮追溯。"""

    __tablename__ = "faith_cases"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    eval_id: Mapped[str] = mapped_column(String(16), unique=True)
    bucket: Mapped[str] = mapped_column(String(24))
    query: Mapped[str] = mapped_column(String(512))
    strategy: Mapped[str] = mapped_column(String(24), default="hybrid_rerank")
    answer: Mapped[str] = mapped_column(Text)
    reason: Mapped[str] = mapped_column(Text)
    citations: Mapped[list | None] = mapped_column(JSON, default=None)
    judge_model: Mapped[str | None] = mapped_column(String(64), default=None)
    status: Mapped[str] = mapped_column(
        Enum("未解决", "已解决", "无需解决", name="faith_case_status"),
        default="未解决",
        server_default="未解决",
    )
    seen_count: Mapped[int] = mapped_column(default=1, server_default="1")
    first_seen_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
    resolution: Mapped[str | None] = mapped_column(String(300), default=None)
    resolved_at: Mapped[datetime | None] = mapped_column(default=None)
