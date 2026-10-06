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
    """把 pending 块向量化写入 Milvus 并回填状态。幂等：可反复跑直至补齐。

    双写对账：MySQL 已 done 但 Milvus 缺失的块（如集合重建后复用旧行）同样补写，
    否则会出现「状态 done 却无向量」的静默漏块。
    """
    done = 0
    existing_ids = store.all_ids()

    def _needs_vectorize(chunk) -> bool:
        return chunk.vectorize_status == "pending" or chunk.id not in existing_ids

    todo = [c for c in await repo.list_done() if _needs_vectorize(c)]
    todo += [c for c in await repo.list_pending(limit=batch_size) if _needs_vectorize(c)]
    # 去重（同时满足 done-but-missing 与 pending 的块只处理一次）
    by_id = {c.id: c for c in todo}
    todo = [by_id[i] for i in sorted(by_id)]
    if todo:
        vectors = await embedder.embed([_vectorize_text(c) for c in todo])
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
            for chunk, vec in zip(todo, vectors)
        ]
        store.upsert(rows)  # 同 id 覆盖 → 中断重跑天然补齐
        for chunk in todo:
            await repo.mark_done(chunk.id, str(chunk.id))
        done += len(todo)
        logger.info("ingest done: %d chunks vectorized", done)
        return done

    logger.info("ingest done: 0 chunks vectorized (already in sync)")
    return 0
