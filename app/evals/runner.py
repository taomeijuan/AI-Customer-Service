import argparse
import asyncio
import logging
from datetime import date
from pathlib import Path

from app.core.config import get_settings
from app.core.llm import get_chat_model
from app.db.engine import build_engine
from app.evals.dataset import load_eval_set
from app.evals.faithfulness import FaithfulnessJudge, aggregate_report
from app.evals.metrics import mrr, recall_at_k
from app.generation.answerer import AnswerSchema, Answerer
from app.knowledge.embedder import build_embedder
from app.knowledge.milvus_store import MilvusStore
from app.knowledge.query_rewriter import LangChainRewriter
from app.knowledge.retriever import HybridRetriever
from app.knowledge.reranker import build_reranker
from app.repositories.low_confidence import LowConfidenceRepo

logger = logging.getLogger(__name__)
STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")
REPORT_DIR = Path(__file__).resolve().parents[2] / "docs" / "evals"
KS = (3, 5, 10)


async def run_strategy(strategy: str, samples: list[dict], deps: dict) -> list[dict]:
    """跑一种策略：检索指标（正例）+ 拒答正确率（负例）+ Faithfulness。"""
    retriever: HybridRetriever = deps["retriever"]
    answerer: Answerer = deps["answerer"]
    judge: FaithfulnessJudge = deps["judge"]
    low_repo: LowConfidenceRepo = deps["low_repo"]
    rows: list[dict] = []
    for s in samples:
        if s.get("expected_refusal"):
            result = await retriever.retrieve(s["query"], strategy=strategy)
            refused = result.low_confidence
            if not refused:
                outcome = await answerer.answer(s["query"], result.evidences)
                refused = not outcome.useful
            if refused:
                await low_repo.record(
                    raw_question=s["query"],
                    source="retrieval_low_conf" if result.low_confidence else "self_check",
                    reason=f"评估负例 {s['id']}：知识库无此内容",
                )
            rows.append(
                {"bucket": s["bucket"], "id": s["id"], "refusal_correct": refused}
            )
            continue

        result = await retriever.retrieve(s["query"], strategy=strategy)
        ranked = [e.chunk_id for e in result.evidences]
        relevant = set(s["relevant_chunk_ids"])
        outcome = await answerer.answer(s["query"], result.evidences)
        row = {"bucket": s["bucket"], "id": s["id"]}
        for k in KS:
            row[f"recall@{k}"] = recall_at_k(relevant, ranked, k)
        row["mrr"] = mrr(relevant, ranked)
        if outcome.useful and outcome.citations:
            verdict = await judge.verify(
                eval_id=s["id"], bucket=s["bucket"], query=s["query"],
                strategy=strategy, answer=outcome.answer, citations=outcome.citations,
            )
            row["faithfulness"] = 1.0 if verdict.faithful else 0.0
        rows.append(row)
    return rows


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def write_report(all_rows: dict[str, list[dict]], path: Path) -> None:
    lines = [f"# ch04 检索质量对比报告（{date.today()}）", ""]
    for strategy, rows in all_rows.items():
        lines.append(f"## 策略：{strategy}")
        lines.append("")
        lines.append("| 桶 | n | recall@3 | recall@5 | recall@10 | mrr | faithfulness | 拒答正确 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        report = aggregate_report([r for r in rows if "recall@3" in r])
        neg = [r for r in rows if "refusal_correct" in r]
        for bucket in sorted(report):
            b = report[bucket]
            refused_ok = (
                f"{sum(1 for r in neg if r['bucket'] == bucket and r['refusal_correct'])}/{sum(1 for r in neg if r['bucket'] == bucket)}"
                if neg
                else "-"
            )
            lines.append(
                f"| {bucket} | {b['n']} | {_fmt(b.get('recall@3', 0))} | {_fmt(b.get('recall@5', 0))} "
                f"| {_fmt(b.get('recall@10', 0))} | {_fmt(b.get('mrr', 0))} | {_fmt(b.get('faithfulness', 0))} | {refused_ok} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("report written: %s", path)


async def run_all(strategies: list[str]) -> dict:
    settings = get_settings()
    engine, session_factory = build_engine(settings)
    model = get_chat_model()
    embedder = build_embedder(settings)
    milvus = MilvusStore(uri=settings.milvus_uri, collection=settings.milvus_collection)
    retriever = HybridRetriever(
        milvus=milvus,
        embedder=embedder,
        rewriter=LangChainRewriter(model),
        reranker=build_reranker(settings),
        settings=settings,
    )
    answerer = Answerer(build_structured_model(model, AnswerSchema))
    judge = FaithfulnessJudge(model, FaithCasesRepoAdapter(session_factory), judge_model=settings.llm_model)
    deps = {"retriever": retriever, "answerer": answerer, "judge": judge,
            "low_repo": LowConfidenceRepoAdapter(session_factory)}
    samples = load_eval_set()
    all_rows = {}
    try:
        for strategy in strategies:
            logger.info("running strategy: %s", strategy)
            all_rows[strategy] = await run_strategy(strategy, samples, deps)
    finally:
        await engine.dispose()
    write_report(all_rows, REPORT_DIR / f"ch04-report-{date.today()}.md")
    return all_rows


class FaithCasesRepoAdapter:
    def __init__(self, session_factory):
        self._sf = session_factory

    async def record(self, **kwargs):
        from app.repositories.faith_cases import FaithCasesRepo

        async with self._sf() as session:
            return await FaithCasesRepo(session).record(**kwargs)

    async def get(self, eval_id):
        from app.repositories.faith_cases import FaithCasesRepo

        async with self._sf() as session:
            return await FaithCasesRepo(session).get(eval_id)


class LowConfidenceRepoAdapter:
    def __init__(self, session_factory):
        self._sf = session_factory

    async def record(self, **kwargs):
        from app.repositories.low_confidence import LowConfidenceRepo

        async with self._sf() as session:
            return await LowConfidenceRepo(session).record(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="ch04 检索质量评估")
    parser.add_argument("--strategy", default="all", choices=[*STRATEGIES, "all"])
    args = parser.parse_args()
    strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    result = asyncio.run(run_all(strategies))
    for strategy, rows in result.items():
        pos = [r for r in rows if "recall@3" in r]
        neg = [r for r in rows if "refusal_correct" in r]
        print(
            f"{strategy}: 题 {len(rows)} | "
            f"Recall@5 均值 {sum(r['recall@5'] for r in pos) / max(len(pos), 1):.3f} | "
            f"MRR 均值 {sum(r['mrr'] for r in pos) / max(len(pos), 1):.3f} | "
            f"拒答正确 {sum(1 for r in neg if r['refusal_correct'])}/{len(neg)}"
        )
