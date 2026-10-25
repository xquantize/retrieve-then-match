"""Ranking metrics for retrieval with set-valued gold matches.

Only queries with at least one gold match are evaluated: a query with no true
match cannot be found, so it says nothing about retrieval quality. (Those
queries still matter for matching precision, which is measured separately.)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

Rankings = Mapping[str, Sequence[str]]
Gold = Mapping[str, frozenset[str]]


def _evaluable(rankings: Rankings, gold: Gold) -> list[str]:
    unknown = [q for q in rankings if q not in gold]
    if unknown:
        raise KeyError(f"rankings contain queries with no gold entry: {unknown[:5]}")
    queries = [q for q in rankings if gold[q]]
    if not queries:
        raise ValueError("no ranked query has a gold match; nothing to evaluate")
    return queries


def recall_at_k(rankings: Rankings, gold: Gold, k: int) -> float:
    """Mean fraction of each query's gold matches found in its top k."""
    queries = _evaluable(rankings, gold)
    total = sum(len(gold[q] & set(rankings[q][:k])) / len(gold[q]) for q in queries)
    return total / len(queries)


def hit_rate_at_k(rankings: Rankings, gold: Gold, k: int) -> float:
    """Fraction of queries with at least one gold match in the top k."""
    queries = _evaluable(rankings, gold)
    return sum(1 for q in queries if gold[q] & set(rankings[q][:k])) / len(queries)


def mrr(rankings: Rankings, gold: Gold, k: int | None = None) -> float:
    """Mean reciprocal rank of the first gold match (0 if none within k)."""
    queries = _evaluable(rankings, gold)
    total = 0.0
    for q in queries:
        ranked = rankings[q] if k is None else rankings[q][:k]
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in gold[q]:
                total += 1.0 / rank
                break
    return total / len(queries)


def retrieval_metrics(rankings: Rankings, gold: Gold, ks: Sequence[int]) -> dict[str, float]:
    """recall@k and hit@k for each k, MRR@max(k), and the evaluated query count."""
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(rankings, gold, k)
    for k in ks:
        metrics[f"hit@{k}"] = hit_rate_at_k(rankings, gold, k)
    metrics[f"mrr@{max(ks)}"] = mrr(rankings, gold, max(ks))
    metrics["evaluated_queries"] = float(len(_evaluable(rankings, gold)))
    return metrics
