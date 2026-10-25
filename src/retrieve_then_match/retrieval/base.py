"""Shared retriever interface and ranking helper."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from retrieve_then_match.data import Record


@dataclass(frozen=True)
class Hit:
    id: str
    score: float


class Retriever(Protocol):
    """Index a corpus once, then return the top-k corpus records for a query."""

    name: str

    def index(self, corpus: Sequence[Record]) -> None: ...

    def search(self, query: Record, k: int) -> list[Hit]: ...


def top_k(
    scores: npt.NDArray[np.float64],
    ids: Sequence[str],
    k: int,
    candidates: npt.NDArray[np.intp] | None = None,
) -> list[Hit]:
    """Highest-scoring ids, ties broken by corpus order so rankings are deterministic."""
    pool = np.arange(len(scores)) if candidates is None else candidates
    if pool.size == 0 or k <= 0:
        return []
    pool_scores = scores[pool]
    order = np.lexsort((pool, -pool_scores))[:k]  # primary: score desc; secondary: index
    return [Hit(id=ids[int(pool[i])], score=float(pool_scores[i])) for i in order]
