"""CLI: run a parameter sweep over the FiloHFT backtest engine.

Examples
--------
Single-axis sweep of channel length::

    python -m filo_hft.scripts.run_sweep \
        --param channel.length 80,100,120,140 \
        --from 2024-01-01 --to 2024-06-30 \
        --symbol XAUUSD \
        --out ../sweeps_out/chan_len

Cross-sweep of two axes::

    python -m filo_hft.scripts.run_sweep \
        --param channel.length 80,100,120 \
        --param setups.min_setup_score 0.50,0.55,0.60 \
        --metric sharpe \
        --from 2024-01-01 --to 2024-06-30 \
        --out ../sweeps_out/chan_vs_score

Use `--no-cap` to evaluate more than 200 Cartesian combinations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import pandas as pd

from filo_hft.config import load_config
from filo_hft.data.loaders import load_bars
from filo_hft.sweeps.runner import (
    DEFAULT_COMBINATION_CAP,
    DEFAULT_RANK_METRIC,
    ParamGrid,
    run_sweep,
)


TIMEFRAMES: tuple[str, ...] = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")


# ---------------------------------------------------------------------------
# helpers


def _parse_value(raw: str) -> Any:
    """Try to coerce a string value into int / float / bool, else keep str."""
    s = raw.strip()
    lowered = s.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    # int first (so "100" doesn't become 100.0)
    try:
        if "." not in s and "e" not in lowered:
            return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _parse_param_pairs(pairs: tuple[tuple[str, str], ...]) -> dict[str, list[Any]]:
    """Convert click's repeated --param option into a dotted-key -> list dict."""
    out: dict[str, list[Any]] = {}
    for key, csv_vals in pairs:
        values = [_parse_value(v) for v in csv_vals.split(",") if v.strip()]
        if not values:
            raise click.BadParameter(f"no values provided for --param {key}")
        out[key] = values
    return out


def _filter_bars_by_date(
    df: pd.DataFrame,
    start: datetime | None,
    end: datetime | None,
) -> pd.DataFrame:
    if "ts" not in df.columns or (start is None and end is None):
        return df
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    mask = pd.Series(True, index=df.index)
    start_ts = pd.Timestamp(start).tz_convert("UTC") if start is not None and pd.Timestamp(start).tzinfo else (pd.Timestamp(start, tz="UTC") if start is not None else None)
    end_ts   = pd.Timestamp(end).tz_convert("UTC")   if end   is not None and pd.Timestamp(end).tzinfo   else (pd.Timestamp(end,   tz="UTC") if end   is not None else None)
    if start_ts is not None:
        mask &= ts >= start_ts
    if end_ts is not None:
        mask &= ts < end_ts
    return df[mask].reset_index(drop=True)


def _load_primary_bars(
    data_root: Path,
    symbol: str,
    start: datetime | None,
    end: datetime | None,
) -> dict[str, pd.DataFrame]:
    """Load every TF for `symbol`, silently skipping any missing files."""
    out: dict[str, pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        try:
            df = load_bars(data_root, symbol, tf)
        except FileNotFoundError:
            click.echo(f"[sweep] no bars for {symbol} {tf} — skipping", err=True)
            continue
        out[tf] = _filter_bars_by_date(df, start, end)
    if "M1" not in out:
        raise click.ClickException(
            f"primary {symbol} M1 bars are required but were not found under {data_root}"
        )
    return out


def _load_corr_bars(
    data_root: Path,
    symbols: tuple[str, ...],
    start: datetime | None,
    end: datetime | None,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = load_bars(data_root, sym, "M1")
        except FileNotFoundError:
            click.echo(f"[sweep] correlation symbol {sym} M1 missing — skipping", err=True)
            continue
        out[sym] = _filter_bars_by_date(df, start, end)
    return out


# ---------------------------------------------------------------------------
# CLI


@click.command()
@click.option(
    "--param",
    "param_pairs",
    nargs=2,
    type=str,
    multiple=True,
    required=True,
    metavar="KEY VALUES",
    help=(
        "Dotted-key override + comma-separated values, e.g. "
        "`--param channel.length 80,100,120`. Repeatable for cross-sweeps."
    ),
)
@click.option(
    "--metric",
    default=DEFAULT_RANK_METRIC,
    show_default=True,
    help="Metric column used to rank the top-N summary.",
)
@click.option(
    "--config",
    "config_path",
    default="../config/default.ini",
    show_default=True,
    type=click.Path(exists=True),
)
@click.option(
    "--data",
    "data_root",
    default="../data",
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("--from", "start", default=None, type=click.DateTime(["%Y-%m-%d"]))
@click.option("--to", "end", default=None, type=click.DateTime(["%Y-%m-%d"]))
@click.option("--symbol", default=None, help="Override primary symbol (default from config).")
@click.option(
    "--out",
    "out_dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Output directory. Defaults to ../sweeps_out/<timestamp>.",
)
@click.option("--jobs", "n_jobs", default=-1, show_default=True, type=int)
@click.option(
    "--no-cap",
    is_flag=True,
    default=False,
    help=f"Allow more than {DEFAULT_COMBINATION_CAP} Cartesian combinations.",
)
@click.option(
    "--top-n",
    default=10,
    show_default=True,
    type=int,
    help="Rows to print / include in the markdown summary.",
)
def main(
    param_pairs: tuple[tuple[str, str], ...],
    metric: str,
    config_path: str,
    data_root: str,
    start: datetime | None,
    end: datetime | None,
    symbol: str | None,
    out_dir: str | None,
    n_jobs: int,
    no_cap: bool,
    top_n: int,
) -> None:
    cfg = load_config(config_path)

    primary = symbol or cfg.symbols.primary
    data_root_path = Path(data_root)

    if start is not None:
        start = start.replace(tzinfo=timezone.utc)
    if end is not None:
        end = end.replace(tzinfo=timezone.utc)

    click.echo(f"[sweep] loading bars for {primary} from {data_root_path}")
    bars_by_tf = _load_primary_bars(data_root_path, primary, start, end)

    corr_symbols: list[str] = []
    if cfg.correlation.enable_xagusd:
        corr_symbols.append(cfg.symbols.silver_proxy)
    if cfg.correlation.enable_dxy:
        corr_symbols.append(cfg.symbols.dxy_proxy)
    click.echo(f"[sweep] loading correlation bars: {corr_symbols or '(none)'}")
    corr_bars = _load_corr_bars(data_root_path, tuple(corr_symbols), start, end)

    axes = _parse_param_pairs(param_pairs)
    grid = ParamGrid(axes)

    if out_dir is None:
        ts_tag = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_dir = f"../sweeps_out/{ts_tag}"
    out_path = Path(out_dir)

    click.echo(
        f"[sweep] grid has {grid.size} points across {len(axes)} axis/axes; "
        f"out={out_path}"
    )

    results = run_sweep(
        bars_by_tf=bars_by_tf,
        corr_bars=corr_bars,
        cfg=cfg,
        param_grid=grid,
        n_jobs=n_jobs,
        out_dir=out_path,
        rank_metric=metric,
        cap=None if no_cap else DEFAULT_COMBINATION_CAP,
        top_n=top_n,
    )

    rank_col = metric if metric in results.columns else "total_pnl_usd"
    if rank_col not in results.columns and not results.empty:
        rank_col = results.columns[-1]

    click.echo("")
    click.echo(f"[sweep] top {top_n} by {rank_col}:")
    if results.empty:
        click.echo("  (no results)")
        return

    display_cols = [c for c in results.columns if c.startswith("param.")]
    for extra in ("total_pnl_usd", "trade_count", "win_rate", "expectancy_pips", "sharpe", "max_dd_pct"):
        if extra in results.columns and extra not in display_cols:
            display_cols.append(extra)
    if rank_col not in display_cols:
        display_cols.append(rank_col)

    ranked = results.sort_values(rank_col, ascending=False).head(top_n)
    with pd.option_context("display.max_columns", None, "display.width", 180):
        click.echo(ranked[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
