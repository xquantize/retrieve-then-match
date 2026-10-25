"""Paired comparison of two saved runs on the same queries."""

from __future__ import annotations

from dataclasses import dataclass

from retrieve_then_match.report import QueryResult, SavedRun
from retrieve_then_match.stats import (
    holm,
    mcnemar_exact,
    paired_bootstrap_ci,
    paired_permutation_test,
)


@dataclass(frozen=True)
class ComparisonRow:
    metric: str
    a: float
    b: float
    ci_low: float
    ci_high: float
    test: str
    p: float
    p_holm: float = 1.0

    @property
    def diff(self) -> float:
        return self.b - self.a


def _shared_queries(a: SavedRun, b: SavedRun) -> list[tuple[QueryResult, QueryResult]]:
    if a.summary["split"] != b.summary["split"]:
        raise ValueError(
            f"runs are on different splits ({a.summary['split']} vs {b.summary['split']})"
        )
    if set(a.queries) != set(b.queries):
        raise ValueError("runs cover different queries; check the split config matches")
    if any(a.queries[q].gold != b.queries[q].gold for q in a.queries):
        raise ValueError("runs disagree on gold matches; they used different data")
    # Queries without gold matches are excluded, matching the retrieval metrics.
    return [(a.queries[q], b.queries[q]) for q in sorted(a.queries) if a.queries[q].gold]


def compare_runs(
    a: SavedRun, b: SavedRun, ks: tuple[int, ...], seed: int = 0
) -> list[ComparisonRow]:
    pairs = _shared_queries(a, b)
    if not pairs:
        raise ValueError("no shared queries with gold matches to compare")
    rows: list[ComparisonRow] = []

    for k in ks:
        hits_a = [qa.hit(k) for qa, _ in pairs]
        hits_b = [qb.hit(k) for _, qb in pairs]
        a_only, b_only, p = mcnemar_exact(hits_a, hits_b)
        low, high = paired_bootstrap_ci(
            [float(x) for x in hits_a], [float(x) for x in hits_b], seed=seed
        )
        rows.append(
            ComparisonRow(
                metric=f"hit@{k}",
                a=sum(hits_a) / len(pairs),
                b=sum(hits_b) / len(pairs),
                ci_low=low,
                ci_high=high,
                test=f"McNemar {a_only}/{b_only}",
                p=p,
            )
        )

    k_max = max(ks)
    rr_a = [qa.reciprocal_rank(k_max) for qa, _ in pairs]
    rr_b = [qb.reciprocal_rank(k_max) for _, qb in pairs]
    low, high = paired_bootstrap_ci(rr_a, rr_b, seed=seed)
    rows.append(
        ComparisonRow(
            metric=f"mrr@{k_max}",
            a=sum(rr_a) / len(pairs),
            b=sum(rr_b) / len(pairs),
            ci_low=low,
            ci_high=high,
            test="permutation",
            p=paired_permutation_test(rr_a, rr_b, seed=seed),
        )
    )

    adjusted = holm([r.p for r in rows])
    return [
        ComparisonRow(
            metric=r.metric,
            a=r.a,
            b=r.b,
            ci_low=r.ci_low,
            ci_high=r.ci_high,
            test=r.test,
            p=r.p,
            p_holm=p_adj,
        )
        for r, p_adj in zip(rows, adjusted, strict=True)
    ]
