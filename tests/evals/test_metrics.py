import pytest

from app.evals.dataset import load_eval_set, validate
from app.evals.metrics import mrr, recall_at_k


def test_recall_at_k():
    relevant = {1, 2, 3}
    ranked = [4, 1, 5, 2]
    assert recall_at_k(relevant, ranked, k=1) == 0.0
    assert recall_at_k(relevant, ranked, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(relevant, ranked, k=3) == pytest.approx(1 / 3)
    assert recall_at_k(relevant, ranked, k=4) == pytest.approx(2 / 3)
    assert recall_at_k(relevant, ranked, k=10) == pytest.approx(2 / 3)


def test_mrr():
    assert mrr({1, 2, 3}, [4, 1, 5, 2]) == pytest.approx(1 / 2)  # 首个相关排位 2
    assert mrr({1}, [1, 2, 3]) == pytest.approx(1.0)
    assert mrr({9}, [1, 2, 3]) == 0.0  # 未命中


def test_mrr_empty_inputs():
    assert mrr(set(), [1, 2]) == 0.0
    assert mrr({1}, []) == 0.0


def test_dataset_loads_and_validates():
    samples = load_eval_set()
    assert len(samples) >= 30
    assert validate(samples) is None  # id 唯一、bucket 枚举、负例带 expected_refusal
