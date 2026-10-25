"""BM25 (Okapi, Lucene-style IDF) implemented over an inverted index.

Written from scratch rather than wrapped, so the tokeniser is fully under our
control: model numbers are the strongest matching signal in product data, and
how they are tokenised decides whether they match at all.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from retrieve_then_match.config import BM25Config
from retrieve_then_match.data import Record
from retrieve_then_match.retrieval.base import Hit, top_k

_ALNUM = re.compile(r"[a-z0-9]+")
_LETTERS_OR_DIGITS = re.compile(r"[a-z]+|[0-9]+")
# Separators that end a "chunk". Hyphens and dots are deliberately absent, so
# MDR-J10 stays one chunk and can be joined into mdrj10.
_CHUNK_SPLIT = re.compile(r"[\s/|,;:()\[\]]+")


def simple_tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric runs: 'MDR-J10/BLUE' -> ['mdr', 'j10', 'blue']."""
    return _ALNUM.findall(text.lower())


def code_aware_tokenize(text: str) -> list[str]:
    """Simple tokens, plus joined codes and letter/digit sub-tokens.

    'MDR-J10/BLUE' -> ['mdr', 'j10', 'mdrj10', 'j', '10', 'blue']
    '161WH'        -> ['161wh', '161', 'wh']
    """
    tokens: list[str] = []
    for chunk in _CHUNK_SPLIT.split(text.lower()):
        parts = _ALNUM.findall(chunk)
        tokens.extend(parts)
        if len(parts) > 1:
            tokens.append("".join(parts))
        for part in parts:
            pieces = _LETTERS_OR_DIGITS.findall(part)
            if len(pieces) > 1:
                tokens.extend(pieces)
    return tokens


TOKENIZERS: dict[str, Callable[[str], list[str]]] = {
    "simple": simple_tokenize,
    "code_aware": code_aware_tokenize,
}


class BM25Retriever:
    def __init__(self, cfg: BM25Config) -> None:
        self.name = f"bm25-{cfg.tokenizer}"
        self._k1 = cfg.k1
        self._b = cfg.b
        self._tokenize = TOKENIZERS[cfg.tokenizer]
        self._ids: list[str] = []
        # term -> (doc indices, precomputed per-document BM25 weights)
        self._postings: dict[str, tuple[npt.NDArray[np.intp], npt.NDArray[np.float64]]] = {}

    def index(self, corpus: Sequence[Record]) -> None:
        self._ids = [r.id for r in corpus]
        n_docs = len(corpus)
        term_docs: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(n_docs, dtype=np.float64)
        for doc_idx, record in enumerate(corpus):
            counts = Counter(self._tokenize(record.text))
            lengths[doc_idx] = sum(counts.values())
            for term, tf in counts.items():
                term_docs[term].append((doc_idx, tf))

        avg_len = float(lengths.mean()) if n_docs and lengths.mean() > 0 else 1.0
        length_norm = self._k1 * (1.0 - self._b + self._b * lengths / avg_len)

        self._postings = {}
        for term, entries in term_docs.items():
            df = len(entries)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            docs = np.fromiter((d for d, _ in entries), dtype=np.intp, count=df)
            tfs = np.fromiter((tf for _, tf in entries), dtype=np.float64, count=df)
            weights = idf * tfs * (self._k1 + 1.0) / (tfs + length_norm[docs])
            self._postings[term] = (docs, weights)

    def search(self, query: Record, k: int) -> list[Hit]:
        if not self._ids:
            raise RuntimeError("BM25Retriever.search called before index()")
        scores = np.zeros(len(self._ids), dtype=np.float64)
        for term in set(self._tokenize(query.text)):
            posting = self._postings.get(term)
            if posting is not None:
                docs, weights = posting
                scores[docs] += weights
        # Only documents sharing at least one term are real candidates.
        return top_k(scores, self._ids, k, candidates=np.flatnonzero(scores > 0))
