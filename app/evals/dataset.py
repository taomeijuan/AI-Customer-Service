import json
from pathlib import Path

DEFAULT_EVAL_SET = Path(__file__).resolve().parents[2] / "tests" / "data" / "eval_set.jsonl"
BUCKETS = {"A_policy", "B_model", "C_colloquial", "E_multi"}


def load_eval_set(path: Path | None = None) -> list[dict]:
    path = path or DEFAULT_EVAL_SET
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(samples: list[dict]) -> None:
    """结构校验：id 唯一、bucket 合法、正例带 relevant_chunk_ids、负例带 expected_refusal。"""
    ids = [s["id"] for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("评估集存在重复 id")
    for s in samples:
        if s["bucket"] not in BUCKETS:
            raise ValueError(f"{s['id']} 非法 bucket: {s['bucket']}")
        if s.get("expected_refusal"):
            continue
        if not s.get("relevant_chunk_ids"):
            raise ValueError(f"{s['id']} 缺 relevant_chunk_ids")
