"""Matching-stage tests with a deterministic fake scorer (no model download)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from retrieve_then_match.config import BM25Config
from retrieve_then_match.data import Dataset, Record
from retrieve_then_match.matching import (
    decision_candidates,
    evaluate_matching,
    gold_pairs,
    precision_recall_f1,
    predict,
    rerank_split,
    tune_threshold,
)
from retrieve_then_match.retrieval import BM25Retriever, Hit


class JaccardScorer:
    """Token-set Jaccard similarity: a transparent stand-in for a cross-encoder."""

    name = "jaccard"

    def score(self, pairs: Sequence[tuple[str, str]]) -> npt.NDArray[np.float64]:
        out = []
        for a, b in pairs:
            sa, sb = set(a.lower().split()), set(b.lower().split())
            out.append(len(sa & sb) / len(sa | sb) if sa | sb else 0.0)
        return np.asarray(out, dtype=np.float64)


def _dataset() -> Dataset:
    queries = (
        Record("q1", "sony mdrj10 headphones"),
        Record("q2", "bose 161 speakers white"),
        Record("q3", "canon camera sd1000"),
    )
    corpus = (
        Record("c1", "sony mdrj10 headphones blue"),
        Record("c2", "bose 161 speakers"),
        Record("c3", "sony camera"),  # distractor for q3
        Record("c4", "canon powershot sd1000 camera"),
    )
    gold = {"q1": frozenset({"c1"}), "q2": frozenset({"c2"}), "q3": frozenset({"c4"})}
    return Dataset(queries, corpus, gold, train_ids=(), val_ids=("q1", "q2", "q3"), test_ids=())


def test_prf_hand_computed() -> None:
    gold = {("q1", "a"), ("q2", "b")}
    m = precision_recall_f1({("q1", "a"), ("q2", "x")}, gold)
    assert (m["precision"], m["recall"], m["f1"]) == pytest.approx((0.5, 0.5, 0.5))
    assert precision_recall_f1(set(), gold)["f1"] == 0.0


def test_policies() -> None:
    rankings = {"q": [Hit("a", 0.9), Hit("b", 0.8), Hit("c", 0.1)]}
    assert predict(rankings, "top1", 0.5) == {("q", "a")}
    assert predict(rankings, "threshold", 0.5) == {("q", "a"), ("q", "b")}
    assert predict(rankings, "top1", 0.95) == set()


def test_tune_threshold_picks_best_f1_cut() -> None:
    # Sorted: 0.9 TP, 0.8 FP, 0.7 TP, 0.2 FP; gold has 3 pairs (one never proposed).
    candidates = [(("q1", "a"), 0.9), (("q2", "x"), 0.8), (("q3", "c"), 0.7), (("q4", "y"), 0.2)]
    gold = {("q1", "a"), ("q3", "c"), ("q5", "e")}
    # F1 at each cut: 2*1/(1+3)=0.5, 2*1/(2+3)=0.4, 2*2/(3+3)=0.667, 2*2/(4+3)=0.571
    assert tune_threshold(candidates, gold) == pytest.approx(0.7)


def test_tune_threshold_keeps_tied_scores_together() -> None:
    candidates = [(("q1", "a"), 0.5), (("q2", "x"), 0.5)]
    gold = {("q1", "a")}
    # Cutting between the tied pair is impossible: the only cut takes both.
    assert tune_threshold(candidates, gold) == pytest.approx(0.5)
    assert tune_threshold([], gold) == float("inf")


def test_rerank_and_evaluate_end_to_end() -> None:
    ds = _dataset()
    retriever = BM25Retriever(BM25Config(k1=1.5, b=0.75))
    retriever.index(ds.corpus)
    reranked = rerank_split(retriever, JaccardScorer(), ds, "val", k=3)
    assert all(hits[0].id in ds.gold[q] for q, hits in reranked.rankings.items())
    assert len(reranked.latency_ms) == 3

    threshold = tune_threshold(
        decision_candidates(reranked.rankings, "top1"), gold_pairs(ds.gold, ["q1", "q2", "q3"])
    )
    result = evaluate_matching(reranked, ds, "val", "top1", threshold)
    assert result.metrics["f1"] == pytest.approx(1.0)
    assert result.metrics["threshold"] == threshold


def test_no_scorer_keeps_retriever_ranking() -> None:
    ds = _dataset()
    retriever = BM25Retriever(BM25Config(k1=1.5, b=0.75))
    retriever.index(ds.corpus)
    reranked = rerank_split(retriever, None, ds, "val", k=3)
    for query in ds.split_queries("val"):
        assert reranked.rankings[query.id] == retriever.search(query, 3)
