"""Run report round-trip and comparison tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from retrieve_then_match.compare import compare_runs
from retrieve_then_match.config import load_config
from retrieve_then_match.evaluate import RetrievalRun
from retrieve_then_match.report import load_run, resolve_run, save_run
from retrieve_then_match.retrieval import Hit

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
GOLD = {"q1": frozenset({"a"}), "q2": frozenset({"b"}), "q3": frozenset({"c"})}


def _run(method: str, rankings: dict[str, list[str]]) -> RetrievalRun:
    return RetrievalRun(
        method=method,
        split="val",
        metrics={"hit@1": 0.5, "evaluated_queries": 3.0},
        latency_ms={"mean": 0.1, "p50": 0.1, "p95": 0.2},
        index_seconds=0.01,
        rankings={
            q: [Hit(d, 1.0 / (i + 1)) for i, d in enumerate(ids)] for q, ids in rankings.items()
        },
    )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG)
    path = save_run(
        _run("bm25-x", {"q1": ["a", "z"], "q2": ["z", "b"], "q3": ["z"]}), cfg, GOLD, tmp_path
    )
    saved = load_run(path)
    assert saved.summary["fingerprint"] == cfg.fingerprint()
    assert (path / "report.md").read_text(encoding="utf-8").startswith("# bm25-x on val")
    q2 = saved.queries["q2"]
    assert q2.first_gold_rank() == 2
    assert q2.hit(1) is False and q2.hit(2) is True
    assert q2.reciprocal_rank(5) == pytest.approx(0.5)
    assert saved.queries["q3"].first_gold_rank() is None


def test_resolve_run_by_name_part_and_latest(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG)
    first = save_run(_run("bm25-a", {"q1": ["a"]}), cfg, GOLD, tmp_path)
    second = save_run(_run("bm25-b", {"q1": ["a"]}), cfg, GOLD, tmp_path)
    assert resolve_run("bm25-a", tmp_path) == first
    assert resolve_run("latest", tmp_path) == second
    with pytest.raises(FileNotFoundError):
        resolve_run(cfg.fingerprint(), tmp_path)  # ambiguous: both runs share it


def test_compare_detects_direction(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG)
    worse = save_run(
        _run("worse", {"q1": ["z", "a"], "q2": ["z", "b"], "q3": ["z", "c"]}), cfg, GOLD, tmp_path
    )
    better = save_run(_run("better", {"q1": ["a"], "q2": ["b"], "q3": ["c"]}), cfg, GOLD, tmp_path)
    rows = {r.metric: r for r in compare_runs(load_run(worse), load_run(better), (1, 2))}
    assert rows["hit@1"].diff == pytest.approx(1.0)
    assert rows["hit@2"].diff == pytest.approx(0.0)
    assert rows["hit@2"].p == pytest.approx(1.0)
    assert all(r.p_holm >= r.p for r in rows.values())


def test_compare_rejects_different_queries(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG)
    a = save_run(_run("a", {"q1": ["a"], "q2": ["b"]}), cfg, GOLD, tmp_path)
    b = save_run(_run("b", {"q1": ["a"], "q3": ["c"]}), cfg, GOLD, tmp_path)
    with pytest.raises(ValueError, match="different queries"):
        compare_runs(load_run(a), load_run(b), (1,))


def test_ambiguous_reference_lists_matches(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT_CONFIG)
    save_run(_run("dense-a", {"q1": ["a"]}), cfg, GOLD, tmp_path)
    save_run(_run("dense-b", {"q1": ["a"]}), cfg, GOLD, tmp_path)
    with pytest.raises(FileNotFoundError, match="dense-a"):
        resolve_run("dense", tmp_path)
