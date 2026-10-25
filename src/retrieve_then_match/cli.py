"""Command-line interface: download, inspect, retrieve, runs, compare."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from retrieve_then_match.compare import compare_runs
from retrieve_then_match.config import Config, load_config
from retrieve_then_match.data import Split, dataset_stats, download, load_dataset
from retrieve_then_match.evaluate import run_retrieval
from retrieve_then_match.report import list_runs, load_run, resolve_run, save_run
from retrieve_then_match.retrieval import METHODS, build_retriever

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
        path = save_run(run, cfg, ds.gold, cfg.output.runs_dir)
        console.print(f"Saved report to [cyan]{path}[/cyan]")


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
    for column in ("run", "method", "split", "fingerprint", "hit@1", "mrr"):
        table.add_column(column)
    for path in paths:
        summary = load_run(path).summary
        metrics = summary["metrics"]
        mrr_key = next((k for k in metrics if k.startswith("mrr@")), None)
        table.add_row(
            path.name,
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
    a = load_run(resolve_run(run_a, runs_dir))
    b = load_run(resolve_run(run_b, runs_dir))
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
            f"{r.p:.4f}",
            f"[bold]{r.p_holm:.4f}[/bold]" if significant else f"{r.p_holm:.4f}",
        )
    console.print(table)
    console.print(
        f"McNemar a/b = queries only A / only B got right. "
        f"Bold = significant at alpha={alpha} after Holm correction."
    )


if __name__ == "__main__":
    app()
