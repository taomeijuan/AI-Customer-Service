def recall_at_k(relevant: set[int], ranked_ids: list[int], k: int) -> float:
    """Recall@K：前 K 个检索结果覆盖的相关 chunk 比例。"""
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return len(relevant & top) / len(relevant)


def mrr(relevant: set[int], ranked_ids: list[int]) -> float:
    """MRR：首个相关结果排名倒数的均值（单题即为倒数）。"""
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0
