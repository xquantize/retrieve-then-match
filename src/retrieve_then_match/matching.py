"""Matching stage: rerank retrieved candidates with a cross-encoder, then decide.

Pipeline per query:
  1. the retriever proposes `candidate_k` candidates;
  2. a pair scorer (cross-encoder) scores every (query, candidate) pair;
  3. candidates are reranked by that score;
  4. a decision policy turns scores into predicted match pairs:
       - top1: the best candidate is a match if its score >= threshold;
       - threshold: every candidate scoring >= threshold is a match.

The threshold is tuned on the validation split to maximise pair-level F1 and
then frozen for the test split. Gold matches the retriever never proposed
count as false negatives, so reported recall is end-to-end.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from retrieve_then_match.config import MatchingConfig
from retrieve_then_match.data import Dataset, Split
from retrieve_then_match.retrieval import Hit, Retriever

Policy = Literal["top1", "threshold"]
Pair = tuple[str, str]  # (query id, corpus id)


class PairScorer(Protocol):
    name: str

    def score(self, pairs: Sequence[tuple[str, str]]) -> npt.NDArray[np.float64]: ...


class CrossEncoderScorer:
    """sentence-transformers CrossEncoder; higher score = more likely a match."""

    def __init__(self, cfg: MatchingConfig) -> None:
        from sentence_transformers import CrossEncoder

        device = None if cfg.device == "auto" else cfg.device
        self.name = f"ce-{cfg.model.rsplit('/', 1)[-1]}"
        self._model: Any = CrossEncoder(cfg.model, device=device)
        self._batch_size = cfg.batch_size

    def score(self, pairs: Sequence[tuple[str, str]]) -> npt.NDArray[np.float64]:
        if not pairs:
            return np.zeros(0, dtype=np.float64)
        scores = self._model.predict(
            [[a, b] for a, b in pairs],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float64).reshape(-1)


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reranked:
    rankings: dict[str, list[Hit]]  # query id -> candidates sorted by matcher score
    latency_ms: list[float]  # per query: retrieval + scoring


def rerank_split(
    retriever: Retriever, scorer: PairScorer | None, ds: Dataset, split: Split, k: int
) -> Reranked:
    """Retrieve k candidates per query and reorder them by matcher score.

    With scorer=None the retriever's own ranking and scores are kept (the
    no-reranker baseline). The retriever must already be indexed. Ties keep
    the retriever's order.
    """
    corpus_text = {c.id: c.text for c in ds.corpus}
    rankings: dict[str, list[Hit]] = {}
    latencies: list[float] = []
    for query in ds.split_queries(split):
        t0 = time.perf_counter()
        candidates = retriever.search(query, k)
        if scorer is None:
            rankings[query.id] = candidates
            latencies.append((time.perf_counter() - t0) * 1000.0)
            continue
        scores = scorer.score([(query.text, corpus_text[h.id]) for h in candidates])
        order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
        rankings[query.id] = [Hit(candidates[i].id, float(scores[i])) for i in order]
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return Reranked(rankings=rankings, latency_ms=latencies)


# ---------------------------------------------------------------------------
# Decisions, metrics and threshold tuning
# ---------------------------------------------------------------------------


def gold_pairs(gold: Mapping[str, frozenset[str]], query_ids: Sequence[str]) -> set[Pair]:
    return {(q, c) for q in query_ids for c in gold[q]}


def decision_candidates(
    rankings: Mapping[str, Sequence[Hit]], policy: Policy
) -> list[tuple[Pair, float]]:
    """Pairs that could become matches under the policy, with their scores."""
    out: list[tuple[Pair, float]] = []
    for q, hits in rankings.items():
        chosen = hits[:1] if policy == "top1" else hits
        out.extend(((q, h.id), h.score) for h in chosen)
    return out


def predict(rankings: Mapping[str, Sequence[Hit]], policy: Policy, threshold: float) -> set[Pair]:
    return {pair for pair, score in decision_candidates(rankings, policy) if score >= threshold}


def precision_recall_f1(predicted: set[Pair], gold: set[Pair]) -> dict[str, float]:
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": float(tp),
        "predicted_pairs": float(len(predicted)),
        "gold_pairs": float(len(gold)),
    }


def tune_threshold(candidates: Sequence[tuple[Pair, float]], gold: set[Pair]) -> float:
    """Threshold maximising F1; ties go to the higher (more conservative) threshold.

    Only thresholds equal to an observed score are considered, and tied scores
    are always accepted or rejected together.
    """
    if not candidates or not gold:
        return float("inf")
    scores = np.array([s for _, s in candidates], dtype=np.float64)
    labels = np.array([pair in gold for pair, _ in candidates], dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    tp = np.cumsum(labels)
    predicted = np.arange(1, scores.size + 1, dtype=np.float64)
    f1 = 2 * tp / (predicted + len(gold))  # = 2PR / (P + R)
    # Only cut after the last element of each run of equal scores.
    valid = np.append(scores[1:] != scores[:-1], True)
    f1 = np.where(valid, f1, -1.0)
    best = int(np.argmax(f1))  # first maximum = highest threshold
    return float(scores[best])


@dataclass(frozen=True)
class MatchingResult:
    split: Split
    threshold: float
    rankings: dict[str, list[Hit]]
    metrics: dict[str, float]
    latency_ms: dict[str, float]


def evaluate_matching(
    reranked: Reranked, ds: Dataset, split: Split, policy: Policy, threshold: float
) -> MatchingResult:
    query_ids = list(reranked.rankings)
    gold = gold_pairs(ds.gold, query_ids)
    metrics = precision_recall_f1(predict(reranked.rankings, policy, threshold), gold)
    metrics["threshold"] = threshold
    lat = np.asarray(reranked.latency_ms)
    latency = {
        "mean": float(lat.mean()),
        "p50": float(np.percentile(lat, 50)),
        "p95": float(np.percentile(lat, 95)),
    }
    return MatchingResult(
        split=split,
        threshold=threshold,
        rankings=reranked.rankings,
        metrics=metrics,
        latency_ms=latency,
    )
