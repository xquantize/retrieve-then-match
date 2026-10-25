"""Paired significance tests for comparing two runs on the same queries.

- Binary per-query outcomes (hit@k): exact McNemar test on discordant pairs.
- Continuous per-query scores (reciprocal rank, recall@k): paired sign-flip
  permutation test, plus a paired bootstrap confidence interval for the mean
  difference.
- Several metrics compared at once: Holm-Bonferroni adjustment.

All resampling is seeded, so reported p-values are reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test.

    Returns (a_only, b_only, p): queries only A got right, only B got right, and
    the p-value of the hypothesis that both systems are equally likely to win.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    a_only = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    b_only = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    tail = sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2**n
    return a_only, b_only, min(1.0, 2.0 * tail)


def _diffs(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    if not a:
        raise ValueError("paired samples are empty")
    return np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)


def paired_permutation_test(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 10_000, seed: int = 0
) -> float:
    """Two-sided sign-flip permutation test on the mean paired difference."""
    diffs = _diffs(a, b)
    observed = abs(diffs.mean())
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, diffs.size))
    null_means = np.abs((signs * diffs).mean(axis=1))
    # +1 smoothing: the observed assignment is itself one valid permutation.
    return float((np.count_nonzero(null_means >= observed - 1e-12) + 1) / (n_resamples + 1))


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(b - a), resampling queries with replacement."""
    diffs = _diffs(a, b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    means = diffs[idx].mean(axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(low), float(high)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running_max
    return adjusted
