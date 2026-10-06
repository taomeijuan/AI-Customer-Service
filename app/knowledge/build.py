import asyncio
import logging
from pathlib import Path

from app.core.config import get_settings
from app.db.engine import build_engine
from app.knowledge.embedder import build_embedder
from app.knowledge.ingest import ingest_pending
from app.knowledge.milvus_store import MilvusStore
from app.knowledge.repository import KnowledgeRepo
from app.knowledge.splitter import split_markdown

logger = logging.getLogger(__name__)
KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "docs" / "knowledge"


def _content_type_for(path: Path) -> str:
    name = path.stem
    if "FAQ" in name:
        return "faq"
    if "手册" in name:
        return "manual"
    return "policy"


async def build(knowledge_dir: Path = KNOWLEDGE_DIR) -> dict:
    """离线建库：遍历 docs/knowledge/*.md → 切分 → MySQL pending → Milvus → done。"""
    settings = get_settings()
    engine, session_factory = build_engine(settings)
    embedder = build_embedder(settings)
    store = MilvusStore(
        uri=settings.milvus_uri,
        collection=settings.milvus_collection,
        dim=1024,
    )
    stats = {"files": 0, "chunks": 0, "reused": 0, "vectorized": 0}
    try:
        async with session_factory() as session:
            repo = KnowledgeRepo(session)
            for md_path in sorted(knowledge_dir.glob("*.md")):
                chunks = split_markdown(
                    md_path.read_text(encoding="utf-8"),
                    content_type=_content_type_for(md_path),
                )
                before = len(chunks)
                stats["reused"] += await repo.count_existing(chunks)
                ids = await repo.upsert_chunks(
                    [
                        {
                            "category": c.category,
                            "questions": c.questions,
                            "answer": c.answer,
                            "section_path": c.section_path,
                            "content_type": c.content_type,
                            "is_key_clause": bool(c.is_key_clause),
                        }
                        for c in chunks
                    ]
                )
                await repo.link_neighbors(ids)
                stats["files"] += 1
                stats["chunks"] += len(ids)
                _ = before
            stats["vectorized"] = await ingest_pending(repo, embedder, store)
    finally:
        await engine.dispose()
    logger.info("build finished: %s", stats)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(asyncio.run(build()))
