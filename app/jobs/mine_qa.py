import logging
import uuid
from datetime import date

from langchain.messages import HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.knowledge.ingest import ingest_pending
from app.db.models import KnowledgeChunk, QaExtractionStaging
from app.repositories.messages import MessagesRepo

logger = logging.getLogger(__name__)


class QaPair(BaseModel):
    """从历史会话中抽出的问答对。"""

    question: str = Field(description="用户的核心问法，保留原始口语")
    answer: str = Field(description="客服给出的有效答案")


class QaExtraction(BaseModel):
    """一批对话的抽取结果。"""

    qa: list[QaPair] = Field(default_factory=list, description="抽出的问答对列表")

    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict) and "qa" not in obj:  # LLM 可能省略外层键
            obj = {"qa": obj}
        return super().model_validate(obj)


MINE_PROMPT = (
    "你是知识库编辑。下面是一段历史客服对话（多轮 user/assistant）。"
    "从中抽取出「值得沉淀为知识」的问答对：question 用用户的原始问法（口语保留），"
    "answer 用客服的实际有效答案（去除寒暄与订单号等个案信息）。"
    "没有可沉淀内容就返回空列表。"
)


async def mine_qa(session_factory, llm, embedder, milvus, settings) -> dict:
    """挖知识主流程：分批抽取 → 暂存 → 两级去重 → 入库 → 补向量化。幂等可重跑。"""
    stats = {"extracted": 0, "discarded": 0, "kept": 0, "vectorized": 0}
    batch_no = f"mine-{date.today():%Y%m%d}-{uuid.uuid4().hex[:8]}"

    async with session_factory() as session:
        pairs = await MessagesRepo(session).load_dialog_pairs()
        for i in range(0, len(pairs), settings.mine_batch_size):
            batch = pairs[i : i + settings.mine_batch_size]
            transcript = "\n".join(
                f"用户：{u}\n客服：{a}" for u, a in batch
            )
            structured = llm.with_structured_output(QaExtraction)
            result = QaExtraction.model_validate(
                await structured.ainvoke([HumanMessage(f"{MINE_PROMPT}\n\n{transcript}")])
            )
            for qa in result.qa:
                session.add(
                    QaExtractionStaging(
                        batch_no=batch_no,
                        source_ref="messages",
                        question=qa.question,
                        answer=qa.answer,
                    )
                )
            await session.commit()
            stats["extracted"] += len(result.qa)

        # ── 整体去重：全表 extracted 行 ──
        staged = (
            await session.execute(
                select(QaExtractionStaging).where(QaExtractionStaging.status == "extracted")
            )
        ).scalars().all()
        if not staged:
            return stats

        vectors = await embedder.embed([s.question for s in staged])
        for row, vec in zip(staged, vectors):
            if await _exact_duplicate(session, row):
                row.status = "discarded"
                stats["discarded"] += 1
                continue
            if await _vector_near_duplicate(session, embedder, row, vec, settings):
                row.status = "discarded"
                stats["discarded"] += 1
                continue
            session.add(
                KnowledgeChunk(
                    category="对话挖掘",
                    questions=row.question,
                    answer=row.answer,
                    content_type="mined",
                )
            )
            row.status = "kept"
            stats["kept"] += 1
        await session.commit()

    stats["vectorized"] = await ingest_pending(
        KnowledgeRepoAdapter(session_factory), embedder, milvus, settings
    )
    logger.info("mine_qa finished: %s (batch %s)", stats, batch_no)
    return stats


async def _exact_duplicate(session, row) -> bool:
    existing = (
        await session.execute(
            select(KnowledgeChunk.id).where(
                KnowledgeChunk.questions == row.question,
                KnowledgeChunk.answer == row.answer,
            )
        )
    ).scalar()
    return existing is not None


async def _vector_near_duplicate(session, embedder, row, vec, settings, threshold: float = 0.95) -> bool:
    """与库内已有知识的问法做向量近重复判定（demo 规模全量比，量大应走 Milvus）。"""
    from app.knowledge.repository import KnowledgeRepo

    existing = (
        await session.execute(select(KnowledgeChunk).order_by(KnowledgeChunk.id))
    ).scalars().all()
    if not existing:
        return False
    ref_vectors = await embedder.embed([k.questions for k in existing])

    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5 or 1.0
        nb = sum(x * x for x in b) ** 0.5 or 1.0
        return dot / (na * nb)

    return any(_cos(vec, rv) > threshold for rv in ref_vectors)


class KnowledgeRepoAdapter:
    """ingest_pending 需要独立会话：包一层 session_factory 适配。"""

    def __init__(self, session_factory) -> None:
        self._sf = session_factory

    async def list_pending(self, limit: int = 100):
        from app.knowledge.repository import KnowledgeRepo

        async with self._sf() as session:
            repo = KnowledgeRepo(session)
            return await repo.list_pending(limit)

    async def mark_done(self, chunk_id: int, vector_id: str):
        from app.knowledge.repository import KnowledgeRepo

        async with self._sf() as session:
            await KnowledgeRepo(session).mark_done(chunk_id, vector_id)
