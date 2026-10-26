"""Command-line interface: download, inspect, retrieve, match, runs, compare."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from retrieve_then_match.compare import compare_runs
from retrieve_then_match.config import Config, load_config
from retrieve_then_match.data import Split, dataset_stats, download, load_dataset
from retrieve_then_match.evaluate import RetrievalRun, run_retrieval
from retrieve_then_match.matching import (
    CrossEncoderScorer,
    decision_candidates,
    evaluate_matching,
    gold_pairs,
    rerank_split,
    tune_threshold,
)
from retrieve_then_match.metrics import recall_at_k, retrieval_metrics
from retrieve_then_match.report import list_runs, load_run, resolve_run, save_run
from retrieve_then_match.retrieval import METHODS, build_retriever
from retrieve_then_match.train import (
    META_FILE,
    build_training_pairs,
    default_model_dir,
    train_cross_encoder,
)

app = typer.Typer(no_args_is_help=True, add_completion=False, help="retrieve-then-match")
console = Console()

DEFAULT_CONFIG = Path("configs/default.yaml")
ConfigOption = Annotated[
    Path, typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config.")
]
SetOption = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        "-s",
        help="Override a config value, e.g. --set retrieval.bm25.tokenizer=simple (repeatable).",
    ),
]
SPLITS = ("val", "test")
TagOption = Annotated[
    str | None,
    typer.Option(help="Short label added to the run folder; usable as a compare reference."),
]


def _load(config: Path, overrides: list[str] | None) -> Config:
    return load_config(config, overrides or [])


@app.command("download")
def download_cmd(
    config: ConfigOption = DEFAULT_CONFIG,
    overrides: SetOption = None,
    force: Annotated[bool, typer.Option(help="Re-download even if present.")] = False,
) -> None:
    """Download and extract the dataset."""
    cfg = _load(config, overrides)
    path = download(cfg.dataset, force=force)
    console.print(f"[green]Dataset ready[/green] in {path}")


@app.command("inspect")
def inspect_cmd(
    config: ConfigOption = DEFAULT_CONFIG,
    overrides: SetOption = None,
    examples: Annotated[int, typer.Option("--examples", "-n", min=0)] = 3,
) -> None:
    """Print dataset statistics and a few gold matching pairs."""
    cfg = _load(config, overrides)
    ds = load_dataset(cfg.dataset, cfg.split)

    stats = Table(title=f"{cfg.dataset.name} (config {cfg.fingerprint()})")
    stats.add_column("statistic")
    stats.add_column("value", justify="right")
    for key, value in dataset_stats(ds).items():
        stats.add_row(key, f"{value:,.1f}" if isinstance(value, float) else f"{value:,}")
    console.print(stats)

    corpus_by_id = {c.id: c for c in ds.corpus}
    shown = 0
    for query in ds.split_queries("val"):
        if shown >= examples:
            break
        if not ds.gold[query.id]:
            continue
        console.rule(f"query {query.id}")
        console.print(f"[bold]{cfg.dataset.query_source}:[/bold] {query.text}")
        for match_id in sorted(ds.gold[query.id]):
            source = cfg.dataset.corpus_source
            console.print(f"[bold]{source}:[/bold] {corpus_by_id[match_id].text}")
        shown += 1


@app.command("retrieve")
def retrieve_cmd(
    method: Annotated[str, typer.Option("--method", "-m", help=f"One of {METHODS}.")] = "bm25",
    split: Annotated[str, typer.Option(help="val or test.")] = "val",
    config: ConfigOption = DEFAULT_CONFIG,
    overrides: SetOption = None,
    save: Annotated[bool, typer.Option(help="Save a run report under runs/.")] = True,
    tag: TagOption = None,
) -> None:
    """Evaluate a retriever: recall@k, hit@k, MRR and per-query latency."""
    if split not in SPLITS:
        raise typer.BadParameter(f"split must be one of {SPLITS}", param_hint="--split")
    cfg = _load(config, overrides)
    ds = load_dataset(cfg.dataset, cfg.split)
    retriever = build_retriever(method, cfg.retrieval)
    run = run_retrieval(
        retriever, ds, cast(Split, split), cfg.retrieval.top_k, cfg.retrieval.eval_ks
    )

    table = Table(title=f"{run.method} on {run.split} (config {cfg.fingerprint()})")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for name, value in run.metrics.items():
        table.add_row(name, f"{int(value):,}" if name == "evaluated_queries" else f"{value:.3f}")
    for name, value in run.latency_ms.items():
        table.add_row(f"latency {name} (ms)", f"{value:.2f}")
    table.add_row("index time (s)", f"{run.index_seconds:.2f}")
    console.print(table)
    if save:
        path = save_run(run, cfg, ds.gold, cfg.output.runs_dir, tag=tag)
        console.print(f"Saved report to [cyan]{path}[/cyan]")


@app.command("match")
def match_cmd(
    split: Annotated[str, typer.Option(help="val or test.")] = "val",
    config: ConfigOption = DEFAULT_CONFIG,
    overrides: SetOption = None,
    save: Annotated[bool, typer.Option(help="Save a run report under runs/.")] = True,
    tag: TagOption = None,
) -> None:
    """Retrieve, optionally rerank with a cross-encoder, and score matches.

    The match threshold is always tuned on the val split, then frozen.
    """
    if split not in SPLITS:
        raise typer.BadParameter(f"split must be one of {SPLITS}", param_hint="--split")
    cfg = _load(config, overrides)
    m = cfg.matching
    ds = load_dataset(cfg.dataset, cfg.split)

    retriever = build_retriever(m.retriever, cfg.retrieval)
    t0 = time.perf_counter()
    retriever.index(ds.corpus)
    index_seconds = time.perf_counter() - t0
    scorer = CrossEncoderScorer(m) if m.scorer == "cross_encoder" else None
    scorer_name = scorer.name if scorer is not None else "retriever"

    # The threshold is always tuned on validation, then frozen.
    with console.status("Scoring validation candidates..."):
        val = rerank_split(retriever, scorer, ds, "val", m.candidate_k)
    threshold = tune_threshold(
        decision_candidates(val.rankings, m.policy), gold_pairs(ds.gold, list(val.rankings))
    )
    target = val
    if split == "test":
        with console.status("Scoring test candidates..."):
            target = rerank_split(retriever, scorer, ds, "test", m.candidate_k)
    result = evaluate_matching(target, ds, cast(Split, split), m.policy, threshold)

    ranked_ids = {q: [h.id for h in hits] for q, hits in result.rankings.items()}
    rerank = retrieval_metrics(ranked_ids, ds.gold, cfg.retrieval.eval_ks)
    rerank["candidate_recall"] = recall_at_k(ranked_ids, ds.gold, m.candidate_k)
    method = f"match-{scorer_name}-{retriever.name}-k{m.candidate_k}-{m.policy}"

    table = Table(title=f"{method} on {split} (config {cfg.fingerprint()})")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for name in ("precision", "recall", "f1"):
        table.add_row(f"[bold]{name}[/bold]", f"[bold]{result.metrics[name]:.3f}[/bold]")
    table.add_row("threshold (tuned on val)", f"{threshold:.4f}")
    for name in ("true_positives", "predicted_pairs", "gold_pairs"):
        table.add_row(name, f"{int(result.metrics[name]):,}")
    table.add_row("candidate recall (ceiling)", f"{rerank['candidate_recall']:.3f}")
    table.add_row("reranked hit@1", f"{rerank['hit@1']:.3f}")
    mrr_key = f"mrr@{max(cfg.retrieval.eval_ks)}"
    table.add_row(f"reranked {mrr_key}", f"{rerank[mrr_key]:.3f}")
    for name, value in result.latency_ms.items():
        table.add_row(f"latency {name} (ms)", f"{value:.2f}")
    console.print(table)
    if split == "val":
        console.print(
            "[yellow]Threshold was tuned on this split, so val F1 is optimistic.[/yellow]"
        )

    if save:
        run = RetrievalRun(
            method=method,
            split=cast(Split, split),
            metrics={**rerank, **result.metrics},
            latency_ms=result.latency_ms,
            index_seconds=index_seconds,
            rankings=result.rankings,
        )
        path = save_run(run, cfg, ds.gold, cfg.output.runs_dir, tag=tag)
        console.print(f"Saved report to [cyan]{path}[/cyan]")


@app.command("train")
def train_cmd(
    config: ConfigOption = DEFAULT_CONFIG,
    overrides: SetOption = None,
    output: Annotated[
        Path | None, typer.Option(help="Model folder (default: derived from the config).")
    ] = None,
    force: Annotated[bool, typer.Option(help="Retrain even if the folder exists.")] = False,
) -> None:
    """Fine-tune the cross-encoder on train-split pairs with BM25 hard negatives."""
    cfg = _load(config, overrides)
    out_dir = output or default_model_dir(cfg)
    if (out_dir / META_FILE).is_file() and not force:
        console.print(f"[yellow]Already trained:[/yellow] {out_dir} (use --force to retrain)")
    else:
        ds = load_dataset(cfg.dataset, cfg.split)
        pairs = build_training_pairs(ds, cfg)
        positives = sum(1 for p in pairs if p.label == 1.0)
        console.print(
            f"Training on {len(pairs):,} pairs ({positives:,} positive, "
            f"{len(pairs) - positives:,} hard negative) from {len(ds.train_ids)} train queries"
        )
        with console.status("Fine-tuning..."):
            meta = train_cross_encoder(pairs, cfg, out_dir)
        losses = ", ".join(f"{loss:.4f}" for loss in meta["epoch_losses"])
        console.print(f"Epoch losses: {losses}")
        console.print(f"Trained in {meta['train_seconds']:.0f}s on {meta['device']}")
        console.print(f"[green]Saved model[/green] to {out_dir}")
    console.print(
        f'Evaluate it with:\n  uv run rtm match --set "matching.model={out_dir}" --tag match-ce-ft'
    )


def _format_p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


RunsDirOption = Annotated[Path, typer.Option("--runs-dir", help="Where run reports live.")]
DEFAULT_RUNS_DIR = Path("runs")


@app.command("runs")
def runs_cmd(runs_dir: RunsDirOption = DEFAULT_RUNS_DIR) -> None:
    """List saved runs, oldest first."""
    paths = list_runs(runs_dir)
    if not paths:
        console.print(f"No runs in {runs_dir}")
        return
    table = Table(title=f"runs in {runs_dir}")
    for column in ("run", "tag", "method", "split", "fingerprint", "hit@1", "mrr"):
        table.add_column(column)
    for path in paths:
        summary = load_run(path).summary
        metrics = summary["metrics"]
        mrr_key = next((k for k in metrics if k.startswith("mrr@")), None)
        table.add_row(
            path.name,
            summary.get("tag") or "-",
            summary["method"],
            summary["split"],
            summary["fingerprint"],
            f"{metrics.get('hit@1', float('nan')):.3f}",
            f"{metrics[mrr_key]:.3f}" if mrr_key else "-",
        )
    console.print(table)


@app.command("compare")
def compare_cmd(
    run_a: Annotated[
        str, typer.Argument(help="Baseline run: folder, 'latest', or unique name part.")
    ],
    run_b: Annotated[str, typer.Argument(help="Candidate run, same forms as run_a.")],
    runs_dir: RunsDirOption = DEFAULT_RUNS_DIR,
    alpha: Annotated[float, typer.Option(help="Significance level after Holm.")] = 0.05,
) -> None:
    """Paired significance test of run B against run A on the same queries."""
    try:
        a = load_run(resolve_run(run_a, runs_dir))
        b = load_run(resolve_run(run_b, runs_dir))
    except FileNotFoundError as err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(1) from err
    ks = tuple(int(k) for k in a.summary["config"]["retrieval"]["eval_ks"])
    rows = compare_runs(a, b, ks)

    table = Table(title=f"A = {a.label}   vs   B = {b.label}")
    for column in ("metric", "A", "B", "B - A", "95% CI", "test", "p", "p (Holm)"):
        table.add_column(column, justify="left" if column in ("metric", "test") else "right")
    for r in rows:
        significant = r.p_holm < alpha
        diff = f"{r.diff:+.3f}"
        table.add_row(
            r.metric,
            f"{r.a:.3f}",
            f"{r.b:.3f}",
            f"[bold]{diff}[/bold]" if significant else diff,
            f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]",
            r.test,
            _format_p(r.p),
            f"[bold]{_format_p(r.p_holm)}[/bold]" if significant else _format_p(r.p_holm),
        )
    console.print(table)
    console.print(
        f"McNemar a/b = queries only A / only B got right. "
        f"Bold = significant at alpha={alpha} after Holm correction."
    )


if __name__ == "__main__":
    app()
