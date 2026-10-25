"""Typed, validated experiment configuration.

The YAML config is parsed into frozen Pydantic models. Unknown keys and
out-of-range values raise immediately, so a typo can never silently change
an experiment. ``Config.fingerprint()`` gives a short, stable hash of the
fully resolved config, which run reports record for traceability.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Source = Literal["abt", "buy"]
Device = Literal["auto", "cpu", "mps", "cuda"]


class _Strict(BaseModel):
    """Base model: reject unknown keys and forbid mutation after loading."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(_Strict):
    name: str
    url: str
    raw_dir: Path
    encoding: str = "latin-1"
    query_source: Source
    corpus_source: Source
    text_fields: tuple[str, ...] = Field(min_length=1)
    max_chars: int = Field(gt=0)

    @model_validator(mode="after")
    def _distinct_sources(self) -> DatasetConfig:
        if self.query_source == self.corpus_source:
            raise ValueError("query_source and corpus_source must differ")
        return self


class SplitConfig(_Strict):
    seed: int
    val_fraction: float = Field(gt=0.0, lt=1.0)


class BM25Config(_Strict):
    k1: float = Field(gt=0.0)
    b: float = Field(ge=0.0, le=1.0)


class DenseConfig(_Strict):
    model: str
    batch_size: int = Field(gt=0)
    normalize: bool = True
    query_prefix: str = ""
    device: Device = "auto"


class HybridConfig(_Strict):
    rrf_k: int = Field(gt=0)


class RetrievalConfig(_Strict):
    top_k: int = Field(gt=0)
    eval_ks: tuple[int, ...] = Field(min_length=1)
    bm25: BM25Config
    dense: DenseConfig
    hybrid: HybridConfig

    @field_validator("eval_ks")
    @classmethod
    def _ks_positive_increasing(cls, ks: tuple[int, ...]) -> tuple[int, ...]:
        if any(k <= 0 for k in ks):
            raise ValueError("eval_ks must all be positive")
        if list(ks) != sorted(set(ks)):
            raise ValueError("eval_ks must be strictly increasing with no duplicates")
        return ks

    @model_validator(mode="after")
    def _ks_within_top_k(self) -> RetrievalConfig:
        if max(self.eval_ks) > self.top_k:
            raise ValueError(
                f"max(eval_ks)={max(self.eval_ks)} exceeds top_k={self.top_k}; "
                "recall cannot be measured beyond the retrieved candidates"
            )
        return self


class MatchingConfig(_Strict):
    model: str
    batch_size: int = Field(gt=0)
    threshold_metric: Literal["f1"] = "f1"


class OutputConfig(_Strict):
    runs_dir: Path


class Config(_Strict):
    dataset: DatasetConfig
    split: SplitConfig
    retrieval: RetrievalConfig
    matching: MatchingConfig
    output: OutputConfig

    def fingerprint(self, length: int = 12) -> str:
        """Short, stable SHA-256 hash of the resolved config."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return Config.model_validate(raw)
