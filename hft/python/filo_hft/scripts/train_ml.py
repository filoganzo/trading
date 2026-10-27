"""CLI for training the baseline ML scoring model.

Usage examples::

    # from live telemetry, all available days
    filo-train --source telemetry \
               --telemetry-root ../telemetry \
               --out ../python/models/baseline.joblib

    # from backtest parquet output
    filo-train --source backtest \
               --trades-path ../python/out/bt_trades.parquet \
               --candidates-path ../python/out/bt_candidates.parquet \
               --out ../python/models/baseline.joblib

    # from shadow_trades.csv — every accepted setup is tracked as a
    # virtual position, so this works even while live trading is gated off
    filo-train --source shadow \
               --telemetry-root ../telemetry \
               --out ../python/models/baseline.joblib

The command loads the requested dataset, builds features, trains the
LightGBM baseline, persists the model, and prints a human-readable report
covering overall AUC, top-quartile precision on the validation fold, and a
per-family precision / recall / AUC breakdown.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from filo_hft.config import load_config
from filo_hft.ml.features import (
    FAMILIES,
    build_dataset,
    load_from_backtest,
    load_from_shadow,
    load_from_telemetry,
)
from filo_hft.ml.train import TrainedModel, save_model, train_baseline


# ----------------------------------------------------------------------
# Report formatting


def _fmt(v: Any) -> str:
    """Compact numeric formatter that tolerates NaN / None / ints."""
    if v is None:
        return "  n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:  # NaN
        return "  nan"
    if abs(f) >= 1000:
        return f"{f:7.0f}"
    return f"{f:6.3f}"


def _print_report(tm: TrainedModel, out_path: Path) -> None:
    """Pretty-print a training report to the terminal."""
    bar = "=" * 72
    click.echo(bar)
    click.echo(" FiloHFT — ML baseline training report")
    click.echo(bar)
    click.echo(f" trained at        : {tm.training_date}")
    click.echo(f" model path        : {out_path}")
    click.echo(f" feature columns   : {len(tm.feature_columns)}")
    click.echo(f" n_train / n_val   : {tm.n_train} / {tm.n_val}")
    click.echo("")

    click.echo(" Train metrics")
    click.echo(f"   auc                    = {_fmt(tm.train_metrics['auc'])}")
    click.echo(f"   avg_precision          = {_fmt(tm.train_metrics['avg_precision'])}")
    click.echo(f"   precision (thr=0.5)    = {_fmt(tm.train_metrics['precision'])}")
    click.echo(f"   recall    (thr=0.5)    = {_fmt(tm.train_metrics['recall'])}")
    click.echo(f"   precision @ top 25%    = {_fmt(tm.train_metrics['precision_top_quartile'])}")
    click.echo(f"   pos_rate               = {_fmt(tm.train_metrics['pos_rate'])}")
    click.echo("")
    click.echo(" Validation metrics (time-ordered hold-out)")
    click.echo(f"   auc                    = {_fmt(tm.val_metrics['auc'])}")
    click.echo(f"   avg_precision          = {_fmt(tm.val_metrics['avg_precision'])}")
    click.echo(f"   precision (thr=0.5)    = {_fmt(tm.val_metrics['precision'])}")
    click.echo(f"   recall    (thr=0.5)    = {_fmt(tm.val_metrics['recall'])}")
    click.echo(f"   precision @ top 25%    = {_fmt(tm.val_metrics['precision_top_quartile'])}")
    click.echo(f"   pos_rate               = {_fmt(tm.val_metrics['pos_rate'])}")
    click.echo("")

    click.echo(" Per-family validation metrics")
    header = f"   {'family':<18}{'support':>10}{'auc':>10}{'prec':>10}{'recall':>10}{'p@25%':>10}"
    click.echo(header)
    click.echo("   " + "-" * (len(header) - 3))
    skipped: list[str] = []
    for fam in FAMILIES:
        row = tm.per_family_metrics.get(fam)
        if row is None:
            skipped.append(fam)
            continue
        click.echo(
            f"   {fam:<18}"
            f"{row['n']:>10d}"
            f"{_fmt(row['auc']):>10}"
            f"{_fmt(row['precision']):>10}"
            f"{_fmt(row['recall']):>10}"
            f"{_fmt(row['precision_top_quartile']):>10}"
        )
    if skipped:
        click.echo("")
        click.echo(
            f"   (families skipped for <10 val samples: {', '.join(skipped)})"
        )
    click.echo(bar)


# ----------------------------------------------------------------------
# CLI


@click.command(name="train-ml")
@click.option(
    "--source",
    type=click.Choice(["telemetry", "backtest", "shadow"], case_sensitive=False),
    required=True,
    help="Where to pull the (trades, candidates) pair from.",
)
@click.option(
    "--telemetry-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Root dir containing YYYYMMDD/ telemetry subdirs (source=telemetry).",
)
@click.option(
    "--days",
    type=str,
    default=None,
    help="Comma-separated YYYYMMDD list; omit to use all days under --telemetry-root.",
)
@click.option(
    "--trades-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Backtest trades parquet (source=backtest).",
)
@click.option(
    "--candidates-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Backtest candidates parquet (source=backtest).",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("../config/default.ini"),
    show_default=True,
    help="FiloHFT ini config (same file the EA reads).",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(path_type=Path),
    default=Path("../python/models/baseline.joblib"),
    show_default=True,
    help="Output path for the joblib-serialized TrainedModel.",
)
@click.option(
    "--min-samples",
    type=int,
    default=200,
    show_default=True,
    help="Abort if the training set has fewer rows than this.",
)
def main(
    source: str,
    telemetry_root: Path | None,
    days: str | None,
    trades_path: Path | None,
    candidates_path: Path | None,
    config_path: Path,
    out_path: Path,
    min_samples: int,
) -> None:
    """Train the baseline ML scoring model and persist it to disk."""
    source = source.lower()

    # --- load config --------------------------------------------------
    if not config_path.exists():
        raise click.ClickException(f"config file not found: {config_path}")
    cfg = load_config(config_path)

    # --- load data ---------------------------------------------------
    if source == "telemetry":
        if telemetry_root is None:
            raise click.ClickException("--telemetry-root is required for --source telemetry")
        day_list: list[str] | None = None
        if days:
            day_list = [d.strip() for d in days.split(",") if d.strip()]
        click.echo(f"[load] telemetry root: {telemetry_root}")
        trades_df, cand_df = load_from_telemetry(telemetry_root, day_list)
    elif source == "shadow":
        if telemetry_root is None:
            raise click.ClickException("--telemetry-root is required for --source shadow")
        day_list = None
        if days:
            day_list = [d.strip() for d in days.split(",") if d.strip()]
        click.echo(f"[load] shadow root: {telemetry_root}")
        trades_df, cand_df = load_from_shadow(telemetry_root, day_list)
    elif source == "backtest":
        if trades_path is None or candidates_path is None:
            raise click.ClickException(
                "--trades-path and --candidates-path are required for --source backtest"
            )
        click.echo(f"[load] trades    : {trades_path}")
        click.echo(f"[load] candidates: {candidates_path}")
        trades_df, cand_df = load_from_backtest(trades_path, candidates_path)
    else:  # pragma: no cover — click already validated the choice
        raise click.ClickException(f"unknown --source {source}")

    click.echo(
        f"[load] trades rows={len(trades_df)}  candidates rows={len(cand_df)}"
    )

    # --- build dataset ------------------------------------------------
    X, y, meta = build_dataset(trades_df, cand_df, cfg)
    click.echo(f"[features] X={X.shape}  y.sum={int(y.sum())}  y.mean={float(y.mean()) if len(y) else float('nan'):.3f}")

    if len(X) < min_samples:
        raise click.ClickException(
            f"training set too small: {len(X)} rows < --min-samples {min_samples}"
        )

    # --- train -------------------------------------------------------
    click.echo("[train] fitting LightGBM baseline...")
    tm = train_baseline(X, y, meta, cfg)

    # --- persist -----------------------------------------------------
    saved = save_model(tm, out_path)
    click.echo(f"[save] wrote {saved}")

    # --- report ------------------------------------------------------
    _print_report(tm, saved)


if __name__ == "__main__":  # pragma: no cover
    main()
