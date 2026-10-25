"""Metric tests on small hand-checked rankings."""

from __future__ import annotations

import pytest

from retrieve_then_match.metrics import hit_rate_at_k, mrr, recall_at_k, retrieval_metrics

RANKINGS = {"q1": ["a", "b", "c"], "q2": ["x", "y", "z"], "q3": ["m"]}
GOLD = {
    "q1": frozenset({"b"}),
    "q2": frozenset({"y", "z"}),  # two gold matches
    "q3": frozenset(),  # no match: excluded from retrieval metrics
}


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 0.75), (3, 1.0)])
def test_recall_at_k(k: int, expected: float) -> None:
    assert recall_at_k(RANKINGS, GOLD, k) == pytest.approx(expected)


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 1.0)])
def test_hit_rate_at_k(k: int, expected: float) -> None:
    assert hit_rate_at_k(RANKINGS, GOLD, k) == pytest.approx(expected)


def test_mrr() -> None:
    assert mrr(RANKINGS, GOLD) == pytest.approx(0.5)  # both first hits at rank 2
    assert mrr(RANKINGS, GOLD, k=1) == pytest.approx(0.0)


def test_retrieval_metrics_keys_and_count() -> None:
    metrics = retrieval_metrics(RANKINGS, GOLD, (1, 2))
    assert set(metrics) == {"recall@1", "recall@2", "hit@1", "hit@2", "mrr@2", "evaluated_queries"}
    assert metrics["evaluated_queries"] == 2


def test_unknown_query_raises() -> None:
    with pytest.raises(KeyError):
        recall_at_k({"nope": ["a"]}, GOLD, 1)
