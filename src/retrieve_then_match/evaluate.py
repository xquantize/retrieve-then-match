"""Run a retriever over a split and measure quality, latency and indexing time."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from retrieve_then_match.data import Dataset, Split
from retrieve_then_match.metrics import retrieval_metrics
from retrieve_then_match.retrieval import Hit, Retriever


@dataclass(frozen=True)
class RetrievalRun:
    method: str
    split: Split
    metrics: dict[str, float]
    latency_ms: dict[str, float]
    index_seconds: float
    rankings: dict[str, list[Hit]]


def run_retrieval(
    retriever: Retriever, ds: Dataset, split: Split, top_k: int, eval_ks: tuple[int, ...]
) -> RetrievalRun:
    start = time.perf_counter()
    retriever.index(ds.corpus)
    index_seconds = time.perf_counter() - start

    rankings: dict[str, list[Hit]] = {}
    latencies: list[float] = []
    for query in ds.split_queries(split):
        t0 = time.perf_counter()
        rankings[query.id] = retriever.search(query, top_k)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    metrics = retrieval_metrics(
        {q: [h.id for h in hits] for q, hits in rankings.items()}, ds.gold, eval_ks
    )
    lat = np.asarray(latencies)
    latency_ms = {
        "mean": float(lat.mean()),
        "p50": float(np.percentile(lat, 50)),
        "p95": float(np.percentile(lat, 95)),
    }
    return RetrievalRun(
        method=retriever.name,
        split=split,
        metrics=metrics,
        latency_ms=latency_ms,
        index_seconds=index_seconds,
        rankings=rankings,
    )
