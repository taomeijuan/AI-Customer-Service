import logging
import time
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.evals.dataset import load_eval_set
from app.evals.faithfulness import FaithfulnessJudge, aggregate_report
from app.evals.metrics import mrr, recall_at_k
from app.repositories.faith_cases import FaithCasesRepo
from app.repositories.low_confidence import LowConfidenceRepo

logger = logging.getLogger(__name__)
router = APIRouter()

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")


class RagEvalRequest(BaseModel):
    mode: Literal["retrieval", "e2e"] = "retrieval"
    strategies: list[Literal["dense", "bm25", "hybrid", "hybrid_rerank"]] = [
        "dense",
        "bm25",
        "hybrid",
        "hybrid_rerank",
    ]
    limit: int = Field(default=6, ge=1, le=40)  # 限题数：页面上快速验证用
    bucket: str | None = Field(default=None, description="留空=全部桶")


@router.post("/api/rag-eval/run")
async def run_rag_eval(req: RagEvalRequest, request: Request) -> dict:
    """RAG 评估在线触发：检索排序对照 / 端到端答案覆盖。

    与 CLI runner（app/evals/runner.py）共用同一套组件；页面默认限题数跑，
    完整 32 题四策略仍建议走 CLI。
    """
    started = time.monotonic()
    settings = request.app.state.settings
    retriever = request.app.state.retriever
    answerer = request.app.state.answerer
    model = get_chat_model()
    session_factory = request.app.state.session_factory

    strategies = [s for s in req.strategies if s in STRATEGIES]
    samples = [
        s
        for s in load_eval_set()
        if not s.get("expected_refusal")
        and (req.bucket is None or s["bucket"] == req.bucket)
    ][: req.limit]

    judge = FaithfulnessJudge(
        model, FaithCasesRepoAdapter(session_factory), judge_model=settings.llm_model
    )
    low_repo = LowConfidenceRepoAdapter(session_factory)

    results: dict[str, Any] = {}
    for strategy in strategies:
        rows: list[dict] = []
        for s in samples:
            result = await retriever.retrieve(s["query"], strategy=strategy)
            ranked = [e.chunk_id for e in result.evidences]
            relevant = set(s["relevant_chunk_ids"])
            row: dict = {
                "id": s["id"],
                "bucket": s["bucket"],
                "query": s["query"],
                "ranked_ids": ranked[:10],
                "relevant_ids": sorted(relevant),
                "low_confidence": result.low_confidence,
            }
            for k in (3, 5, 10):
                row[f"recall@{k}"] = round(recall_at_k(relevant, ranked, k), 3)
            row["mrr"] = round(mrr(relevant, ranked), 3)

            if req.mode == "e2e":
                outcome = await answerer.answer(s["query"], result.evidences)
                row["answer"] = outcome.answer
                row["citations"] = outcome.citations
                row["refused"] = not outcome.useful
                if outcome.useful and outcome.citations:
                    verdict = await judge.verify(
                        eval_id=s["id"], bucket=s["bucket"], query=s["query"],
                        strategy=strategy, answer=outcome.answer,
                        citations=outcome.citations,
                    )
                    row["faithfulness"] = 1.0 if verdict.faithful else 0.0
                    row["judge_reason"] = verdict.reason
            rows.append(row)

        results[strategy] = {
            "rows": rows,
            "aggregate": aggregate_report(
                [
                    {
                        "bucket": r["bucket"],
                        **{
                            k: v
                            for k, v in r.items()
                            if isinstance(v, (int, float))
                            and (k.startswith("recall@") or k in ("mrr", "faithfulness"))
                        },
                    }
                    for r in rows
                ]
            ),
        }

    return {
        "mode": req.mode,
        "elapsed_sec": round(time.monotonic() - started, 1),
        "n_questions": len(samples),
        "results": results,
    }


class FaithCasesRepoAdapter:
    def __init__(self, session_factory):
        self._sf = session_factory

    async def record(self, **kwargs):
        async with self._sf() as session:
            return await FaithCasesRepo(session).record(**kwargs)

    async def get(self, eval_id):
        async with self._sf() as session:
            return await FaithCasesRepo(session).get(eval_id)


class LowConfidenceRepoAdapter:
    def __init__(self, session_factory):
        self._sf = session_factory

    async def record(self, **kwargs):
        async with self._sf() as session:
            return await LowConfidenceRepo(session).record(**kwargs)
