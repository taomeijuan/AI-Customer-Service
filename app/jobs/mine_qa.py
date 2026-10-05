import logging
import uuid
from datetime import date

from langchain.messages import HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.models import KnowledgeChunk, QaExtractionStaging
from app.extraction.service import build_structured_model
from app.knowledge.ingest import ingest_pending
from app.repositories.messages import MessagesRepo

logger = logging.getLogger(__name__)


class QaPair(BaseModel):
    """从历史会话中抽出的问答对。"""

    question: str = Field(description="用户的核心问法，保留原始口语")
    answer: str = Field(description="客服给出的有效答案")


class QaExtraction(BaseModel):
    """一段对话的抽取结果。"""

    qa: list[QaPair] = Field(default_factory=list, description="抽出的问答对列表")

    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict) and "qa" not in obj and "parsed" not in obj:
            obj = {"qa": obj}
        return super().model_validate(obj)


MINE_PROMPT = (
    "你是知识库编辑。下面是一段历史客服对话（多轮 user/assistant）。"
    "从中抽取出「值得沉淀为知识」的问答对：question 用用户的原始问法（口语保留），"
    "answer 用客服的实际有效答案（去除寒暄与订单号等个案信息）。"
    "没有可沉淀内容就返回空列表。"
)

NEAR_DUP_THRESHOLD = 0.95


async def mine_qa(session_factory, llm, embedder, milvus) -> dict:
    """挖知识主流程：按会话分批抽取 → 暂存 → 两级去重 → 入库 → 补向量化。幂等可重跑。"""
    stats = {"extracted": 0, "discarded": 0, "kept": 0, "vectorized": 0}
    batch_no = f"mine-{date.today():%Y%m%d}-{uuid.uuid4().hex[:8]}"

    async with session_factory() as session:
        # ── 按会话分组抽取（防跨会话串味），source_ref=会话 id 可溯源 ──
        grouped = await MessagesRepo(session).load_dialog_pairs()
        for cid, pairs in grouped.items():
            transcript = "\n".join(f"用户：{u}\n客服：{a}" for u, a in pairs)
            structured = build_structured_model(llm, QaExtraction)
            result = await structured.ainvoke(
                [HumanMessage(f"{MINE_PROMPT}\n\n{transcript}")]
            )
            if result["parsing_error"] or result["parsed"] is None:
                logger.warning("mine_qa 抽取解析失败（会话 %s）: %s", cid, result["parsing_error"])
                continue
            for qa in result["parsed"].qa:
                session.add(
                    QaExtractionStaging(
                        batch_no=batch_no,
                        source_ref=str(cid),
                        question=qa.question,
                        answer=qa.answer,
                    )
                )
                stats["extracted"] += 1
            await session.commit()

        # ── 整体去重：本批全部 extracted 行 ──
        staged = (
            await session.execute(
                select(QaExtractionStaging).where(
                    QaExtractionStaging.batch_no == batch_no,
                    QaExtractionStaging.status == "extracted",
                )
            )
        ).scalars().all()
        if not staged:
            return stats

        staged_vectors = await embedder.embed([s.question for s in staged])
        existing = (
            await session.execute(select(KnowledgeChunk))
        ).scalars().all()
        existing_vectors = await embedder.embed([k.questions for k in existing])
        kept_vectors = list(existing_vectors)  # 候选与「库内+本轮已保留」都判重

        for row, vec in zip(staged, staged_vectors):
            if await _exact_duplicate(session, row):
                row.status = "discarded"
                stats["discarded"] += 1
                continue
            if _near_duplicate(vec, kept_vectors):
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
            kept_vectors.append(vec)
            stats["kept"] += 1
        await session.commit()

    stats["vectorized"] = await ingest_pending(
        KnowledgeRepoAdapter(session_factory), embedder, milvus
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


def _near_duplicate(vec: list[float], ref_vectors: list[list[float]]) -> bool:
    def _cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5 or 1.0
        nb = sum(x * x for x in b) ** 0.5 or 1.0
        return dot / (na * nb)

    return any(_cos(vec, rv) > NEAR_DUP_THRESHOLD for rv in ref_vectors)


class KnowledgeRepoAdapter:
    """ingest_pending 需要独立会话：包一层 session_factory 适配。"""

    def __init__(self, session_factory) -> None:
        self._sf = session_factory

    async def list_pending(self, limit: int = 100):
        from app.knowledge.repository import KnowledgeRepo

        async with self._sf() as session:
            return await KnowledgeRepo(session).list_pending(limit)

    async def mark_done(self, chunk_id: int, vector_id: str):
        from app.knowledge.repository import KnowledgeRepo

        async with self._sf() as session:
            await KnowledgeRepo(session).mark_done(chunk_id, vector_id)


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    async def _main():
        from app.core.config import get_settings
        from app.core.llm import get_chat_model
        from app.db.engine import build_engine
        from app.knowledge.embedder import build_embedder
        from app.knowledge.milvus_store import MilvusStore

        settings = get_settings()
        engine, session_factory = build_engine(settings)
        try:
            return await mine_qa(
                session_factory=session_factory,
                llm=get_chat_model(),
                embedder=build_embedder(settings),
                milvus=MilvusStore(uri=settings.milvus_uri, collection=settings.milvus_collection),
            )
        finally:
            await engine.dispose()

    print(asyncio.run(_main()))
