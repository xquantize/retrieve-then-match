"""Persist retrieval runs as self-describing, diffable reports.

Each run gets its own folder under runs/:

    runs/20261027-031500_bm25-code_aware_val_1056691f277e/
        report.json      # config, fingerprint, metrics, latency, environment
        report.md        # the same, human-readable and diff-friendly
        per_query.jsonl  # one line per query: gold ids, ranked ids, scores

Per-query results are what make paired significance tests possible later.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from retrieve_then_match import __version__
from retrieve_then_match.config import Config
from retrieve_then_match.evaluate import RetrievalRun

SCHEMA_VERSION = 1
REPORT_JSON = "report.json"
REPORT_MD = "report.md"
PER_QUERY = "per_query.jsonl"


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    gold: frozenset[str]
    ranked: tuple[str, ...]
    scores: tuple[float, ...]

    def first_gold_rank(self) -> int | None:
        for rank, doc_id in enumerate(self.ranked, start=1):
            if doc_id in self.gold:
                return rank
        return None

    def hit(self, k: int) -> bool:
        rank = self.first_gold_rank()
        return rank is not None and rank <= k

    def recall(self, k: int) -> float:
        return len(self.gold & set(self.ranked[:k])) / len(self.gold) if self.gold else 0.0

    def reciprocal_rank(self, k: int) -> float:
        rank = self.first_gold_rank()
        return 1.0 / rank if rank is not None and rank <= k else 0.0


@dataclass(frozen=True)
class SavedRun:
    path: Path
    summary: dict[str, Any]
    queries: dict[str, QueryResult]

    @property
    def label(self) -> str:
        return f"{self.summary['method']} ({self.summary['fingerprint']})"


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _environment() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "package_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def _markdown(summary: Mapping[str, Any]) -> str:
    env = summary["environment"]
    git = env["git_commit"] or "unknown"
    if env["git_dirty"]:
        git += " (uncommitted changes)"
    lines = [
        f"# {summary['method']} on {summary['split']}",
        "",
        f"- created: {summary['created_at']}",
        f"- config fingerprint: `{summary['fingerprint']}`",
        f"- git: `{git}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for name, value in summary["metrics"].items():
        shown = f"{int(value)}" if name == "evaluated_queries" else f"{value:.4f}"
        lines.append(f"| {name} | {shown} |")
    lines += ["", "## Latency (ms per query)", "", "| stat | value |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in summary["latency_ms"].items()]
    lines += [
        "",
        f"Index time: {summary['index_seconds']:.3f} s",
        "",
        "## Config",
        "",
        "```yaml",
        yaml.safe_dump(summary["config"], sort_keys=False).rstrip(),
        "```",
        "",
    ]
    return "\n".join(lines)


_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def save_run(
    run: RetrievalRun,
    cfg: Config,
    gold: Mapping[str, frozenset[str]],
    runs_dir: Path,
    tag: str | None = None,
) -> Path:
    if tag is not None and not _TAG.fullmatch(tag):
        raise ValueError(f"tag {tag!r} may only contain letters, digits, '.', '_' and '-'")
    created = datetime.now(UTC)
    fingerprint = cfg.fingerprint()
    stem = f"{created:%Y%m%d-%H%M%S}_{run.method}_{run.split}_{fingerprint}"
    tag_part = f"__{tag}" if tag else ""
    path = runs_dir / f"{stem}{tag_part}"
    suffix = 1
    while path.exists():  # keep the tag last so exact-tag lookup still works
        path = runs_dir / f"{stem}-{suffix}{tag_part}"
        suffix += 1
    path.mkdir(parents=True)

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created.isoformat(timespec="seconds"),
        "method": run.method,
        "split": run.split,
        "fingerprint": fingerprint,
        "tag": tag,
        "metrics": run.metrics,
        "latency_ms": run.latency_ms,
        "index_seconds": run.index_seconds,
        "config": cfg.model_dump(mode="json"),
        "environment": _environment(),
    }
    (path / REPORT_JSON).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (path / REPORT_MD).write_text(_markdown(summary), encoding="utf-8")
    with (path / PER_QUERY).open("w", encoding="utf-8") as f:
        for query_id in sorted(run.rankings):
            hits = run.rankings[query_id]
            record = {
                "query_id": query_id,
                "gold": sorted(gold[query_id]),
                "ranked": [h.id for h in hits],
                "scores": [round(h.score, 6) for h in hits],
            }
            f.write(json.dumps(record) + "\n")
    return path


def load_run(path: Path) -> SavedRun:
    summary: dict[str, Any] = json.loads((path / REPORT_JSON).read_text(encoding="utf-8"))
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported report schema {summary.get('schema_version')}")
    queries: dict[str, QueryResult] = {}
    with (path / PER_QUERY).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[row["query_id"]] = QueryResult(
                query_id=row["query_id"],
                gold=frozenset(row["gold"]),
                ranked=tuple(row["ranked"]),
                scores=tuple(float(s) for s in row["scores"]),
            )
    return SavedRun(path=path, summary=summary, queries=queries)


def list_runs(runs_dir: Path) -> list[Path]:
    """Run folders, oldest first (by write time, so runs in the same second still order)."""
    if not runs_dir.exists():
        return []
    runs = [p for p in runs_dir.iterdir() if (p / REPORT_JSON).is_file()]
    return sorted(runs, key=lambda p: ((p / REPORT_JSON).stat().st_mtime_ns, p.name))


def resolve_run(ref: str, runs_dir: Path) -> Path:
    """Accept a run folder path, 'latest', an exact run tag, or a unique name part."""
    candidate = Path(ref)
    if (candidate / REPORT_JSON).is_file():
        return candidate
    if ref == "latest":
        runs = list_runs(runs_dir)
        if not runs:
            raise FileNotFoundError(f"no runs in {runs_dir}")
        return runs[-1]
    runs = list_runs(runs_dir)
    tagged = [p for p in runs if p.name.endswith(f"__{ref}")]
    if len(tagged) == 1:  # an exact tag wins over substring matches
        return tagged[0]
    matches = [p for p in runs if ref in p.name]
    if not matches:
        raise FileNotFoundError(f"no runs in {runs_dir} match {ref!r}")
    if len(matches) > 1:
        listing = "\n  ".join(p.name for p in matches)
        raise FileNotFoundError(
            f"{len(matches)} runs match {ref!r}; use a longer part of one name:\n  {listing}"
        )
    return matches[0]
