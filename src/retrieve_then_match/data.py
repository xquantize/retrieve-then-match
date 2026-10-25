"""Download and load the Abt-Buy entity-matching benchmark.

Each record in the query source (Abt) is a query; the corpus source (Buy) is
the catalogue we search over. Gold matches come from the perfect mapping file.
A query can have several gold matches, so matches are stored as sets.

Queries are split into validation (for tuning the match threshold) and test
(for reported numbers). The split is seeded and stratified on whether a query
has any gold match, so both splits contain the same mix of matchable and
unmatchable queries.
"""

from __future__ import annotations

import re
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from retrieve_then_match.config import DatasetConfig, SplitConfig

# Expected files (matched case-insensitively, anywhere under raw_dir).
_FILES = {"abt": "abt.csv", "buy": "buy.csv", "mapping": "abt_buy_perfectmapping.csv"}
# Column in the mapping file holding each source's ids.
_MAPPING_ID_COLS = {"abt": "idAbt", "buy": "idBuy"}
_WHITESPACE = re.compile(r"\s+")
# Characters a UTF-8 BOM turns into when a file is read as latin-1.
_BOM_CHARS = "\ufeffï»¿"

Split = Literal["val", "test"]


@dataclass(frozen=True)
class Record:
    id: str
    text: str


@dataclass(frozen=True)
class Dataset:
    queries: tuple[Record, ...]
    corpus: tuple[Record, ...]
    gold: Mapping[str, frozenset[str]]  # query id -> matching corpus ids (may be empty)
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    def split_queries(self, split: Split) -> list[Record]:
        ids = set(self.val_ids if split == "val" else self.test_ids)
        return [q for q in self.queries if q.id in ids]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _locate_files(raw_dir: Path) -> dict[str, Path]:
    """Find the three expected CSVs under raw_dir, case-insensitively."""
    found = {p.name.lower(): p for p in raw_dir.rglob("*.csv")} if raw_dir.exists() else {}
    missing = [name for name in _FILES.values() if name not in found]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} under {raw_dir}. Run `rtm download`, or download the "
            "dataset zip manually and unzip it there."
        )
    return {key: found[name] for key, name in _FILES.items()}


def download(cfg: DatasetConfig, force: bool = False) -> Path:
    """Download and extract the dataset into cfg.raw_dir (skipped if already present)."""
    raw_dir: Path = cfg.raw_dir
    if not force:
        try:
            _locate_files(raw_dir)
            return raw_dir
        except FileNotFoundError:
            pass

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "dataset.zip"
    request = urllib.request.Request(cfg.url, headers={"User-Agent": "retrieve-then-match"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, zip_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(raw_dir)
    except (OSError, zipfile.BadZipFile) as err:
        raise RuntimeError(
            f"Could not download/extract {cfg.url} ({err}). Download the zip manually "
            f"and unzip it into {raw_dir}/"
        ) from err
    finally:
        zip_path.unlink(missing_ok=True)

    _locate_files(raw_dir)  # fail loudly if the archive layout was unexpected
    return raw_dir


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_csv(path: Path, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip().lstrip(_BOM_CHARS) for c in df.columns]
    return df


def build_text(row: Mapping[str, Any], fields: Sequence[str], max_chars: int) -> str:
    """Join the configured fields into one normalised, truncated string."""
    parts = [_WHITESPACE.sub(" ", str(row.get(f, ""))).strip() for f in fields]
    text = " | ".join(p for p in parts if p)
    return text[:max_chars].rstrip()


def _to_records(df: pd.DataFrame, cfg: DatasetConfig, source: str) -> tuple[Record, ...]:
    if "id" not in df.columns:
        raise ValueError(f"{source}: no 'id' column (found {list(df.columns)})")
    records = tuple(
        Record(id=str(row["id"]).strip(), text=build_text(row, cfg.text_fields, cfg.max_chars))
        for row in df.to_dict("records")
    )
    ids = [r.id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source}: duplicate record ids")
    return records


def _build_gold(
    mapping: pd.DataFrame, cfg: DatasetConfig, query_ids: set[str], corpus_ids: set[str]
) -> dict[str, frozenset[str]]:
    q_col = _MAPPING_ID_COLS[cfg.query_source]
    c_col = _MAPPING_ID_COLS[cfg.corpus_source]
    for col in (q_col, c_col):
        if col not in mapping.columns:
            raise ValueError(f"mapping: no '{col}' column (found {list(mapping.columns)})")

    matches: defaultdict[str, set[str]] = defaultdict(set)
    for q_id, c_id in zip(mapping[q_col], mapping[c_col], strict=True):
        q_id, c_id = str(q_id).strip(), str(c_id).strip()
        if q_id not in query_ids or c_id not in corpus_ids:
            raise ValueError(f"mapping references unknown ids: {q_id!r} -> {c_id!r}")
        matches[q_id].add(c_id)
    return {q_id: frozenset(matches.get(q_id, set())) for q_id in sorted(query_ids)}


def split_query_ids(
    gold: Mapping[str, frozenset[str]], cfg: SplitConfig
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Seeded val/test split, stratified on whether each query has a gold match."""
    rng = np.random.default_rng(cfg.seed)
    val: list[str] = []
    test: list[str] = []
    for has_match in (True, False):
        group = sorted(q for q, m in gold.items() if bool(m) == has_match)
        order = rng.permutation(len(group))
        n_val = round(len(group) * cfg.val_fraction)
        val.extend(group[i] for i in order[:n_val])
        test.extend(group[i] for i in order[n_val:])
    return tuple(sorted(val)), tuple(sorted(test))


def load_dataset(dataset_cfg: DatasetConfig, split_cfg: SplitConfig) -> Dataset:
    files = _locate_files(dataset_cfg.raw_dir)
    tables = {
        source: _read_csv(files[source], dataset_cfg.encoding)
        for source in (dataset_cfg.query_source, dataset_cfg.corpus_source)
    }
    for field in dataset_cfg.text_fields:
        if not any(field in df.columns for df in tables.values()):
            raise ValueError(f"text field {field!r} not found in either table")

    queries = _to_records(tables[dataset_cfg.query_source], dataset_cfg, dataset_cfg.query_source)
    corpus = _to_records(tables[dataset_cfg.corpus_source], dataset_cfg, dataset_cfg.corpus_source)
    mapping = _read_csv(files["mapping"], dataset_cfg.encoding)
    gold = _build_gold(mapping, dataset_cfg, {q.id for q in queries}, {c.id for c in corpus})
    val_ids, test_ids = split_query_ids(gold, split_cfg)
    return Dataset(queries=queries, corpus=corpus, gold=gold, val_ids=val_ids, test_ids=test_ids)


def dataset_stats(ds: Dataset) -> dict[str, float]:
    """Summary numbers for `rtm inspect` and run reports."""
    n_matches = [len(m) for m in ds.gold.values()]
    return {
        "queries": len(ds.queries),
        "corpus": len(ds.corpus),
        "gold_pairs": sum(n_matches),
        "queries_with_match": sum(1 for n in n_matches if n > 0),
        "queries_with_multiple_matches": sum(1 for n in n_matches if n > 1),
        "val_queries": len(ds.val_ids),
        "test_queries": len(ds.test_ids),
        "mean_query_chars": float(np.mean([len(q.text) for q in ds.queries])),
        "mean_corpus_chars": float(np.mean([len(c.text) for c in ds.corpus])),
    }
