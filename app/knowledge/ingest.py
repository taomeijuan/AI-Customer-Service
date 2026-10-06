import logging

logger = logging.getLogger(__name__)


def _vectorize_text(chunk) -> str:
    """三格拼装：category + questions + answer 一起进向量；元数据不进。"""
    return f"{chunk.category}\n{chunk.questions}\n{chunk.answer}"


def _bytes_truncate(text: str, max_bytes: int) -> str:
    """Milvus VARCHAR max_length 按 UTF-8 字节计：按字节截断（中文 1 字 3 字节）。"""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore")


async def ingest_pending(repo, embedder, store, batch_size: int = 50) -> int:
    """把 pending 块向量化写入 Milvus 并回填状态。幂等：可反复跑直至补齐。"""
    done = 0
    while True:
        pending = await repo.list_pending(limit=batch_size)
        if not pending:
            break
        vectors = await embedder.embed([_vectorize_text(c) for c in pending])
        rows = [
            {
                "id": chunk.id,
                "vector": vec,
                "text": _vectorize_text(chunk),
                "questions": _bytes_truncate(chunk.questions, 2000),
                "answer": _bytes_truncate(chunk.answer, 60000),
                "category": _bytes_truncate(chunk.category, 500),
                "content_type": chunk.content_type or "",
            }
            for chunk, vec in zip(pending, vectors)
        ]
        store.upsert(rows)  # 同 id 覆盖 → 中断重跑天然补齐
        for chunk in pending:
            await repo.mark_done(chunk.id, str(chunk.id))
        done += len(pending)
        if len(pending) < batch_size:
            break
    logger.info("ingest done: %d chunks vectorized", done)
    return done
