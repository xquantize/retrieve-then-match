"""Hybrid retrieval by Reciprocal Rank Fusion (Cormack et al., 2009).

Each component retriever returns its top `depth` candidates; every candidate
scores sum(1 / (rrf_k + rank)) across the lists it appears in. RRF uses ranks
only, so BM25 and cosine scores never need calibrating against each other.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from retrieve_then_match.data import Record
from retrieve_then_match.retrieval.base import Hit, Retriever


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], rrf_k: int) -> list[Hit]:
    """Fuse ranked id lists; ties broken by first appearance, so output is deterministic."""
    scores: defaultdict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    position = 0
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (rrf_k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = position
                position += 1
    ordered = sorted(scores, key=lambda d: (-scores[d], first_seen[d]))
    return [Hit(id=d, score=scores[d]) for d in ordered]


class HybridRetriever:
    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int, depth: int) -> None:
        if len(retrievers) < 2:
            raise ValueError("hybrid retrieval needs at least two retrievers")
        # Kept short and shell-safe: it becomes part of the run folder name.
        # The components are recorded in the run's config.
        self.name = "hybrid-rrf"
        self._retrievers = list(retrievers)
        self._rrf_k = rrf_k
        self._depth = depth

    def index(self, corpus: Sequence[Record]) -> None:
        for retriever in self._retrievers:
            retriever.index(corpus)

    def search(self, query: Record, k: int) -> list[Hit]:
        depth = max(self._depth, k)
        rankings = [[h.id for h in r.search(query, depth)] for r in self._retrievers]
        return reciprocal_rank_fusion(rankings, self._rrf_k)[:k]
