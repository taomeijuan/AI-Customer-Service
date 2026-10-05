from datetime import datetime

from sqlalchemy import BigInteger, Boolean, Enum, ForeignKey, JSON, String, Text, func
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
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


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
