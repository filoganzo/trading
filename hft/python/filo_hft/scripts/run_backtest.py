"""`filo-backtest` CLI — offline bar-level replay over historical parquet bars.

Example:
    filo-backtest \
        --config config/default.ini \
        --data   data \
        --symbol XAUUSD \
        --from   2025-09-01 \
        --to     2025-10-01 \
        --out    artifacts/bt_2025_09
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from filo_hft.backtest.replay import run_from_disk
from filo_hft.config import load_config


def _fmt_usd(v: float) -> str:
    return f"${v:,.2f}"


def _print_summary(symbol: str, metrics: dict, trade_count: int, cand_count: int) -> None:
    click.echo()
    click.echo("=" * 72)
    click.echo(f" FiloHFT backtest summary — {symbol}")
    click.echo("=" * 72)
    click.echo(f" total pnl          : {_fmt_usd(metrics.get('total_pnl_usd', 0.0))}")
    click.echo(f" trade count        : {metrics.get('trade_count', 0)}")
    click.echo(f" win rate           : {metrics.get('win_rate', 0.0) * 100:.2f}%")
    click.echo(f" expectancy (pips)  : {metrics.get('expectancy_pips', 0.0):.3f}")
    click.echo(f" expectancy (usd)   : {_fmt_usd(metrics.get('expectancy_usd', 0.0))}")
    click.echo(f" profit factor      : {metrics.get('profit_factor', 0.0):.2f}")
    click.echo(f" sharpe (per-trade) : {metrics.get('sharpe', 0.0):.2f}")
    click.echo(f" max drawdown       : {metrics.get('max_dd_pct', 0.0):.2f}%")
    click.echo(f" candidates logged  : {cand_count}")
    click.echo("-" * 72)
    pf = metrics.get("per_family", {}) or {}
    if pf:
        click.echo(" per-family breakdown:")
        click.echo(
            f"   {'family':<18}{'count':>8}{'pnl_usd':>14}"
            f"{'win%':>10}{'exp_pips':>12}"
        )
        for fam, sub in pf.items():
            click.echo(
                f"   {fam:<18}{sub.get('trade_count', 0):>8}"
                f"{_fmt_usd(sub.get('total_pnl_usd', 0.0)):>14}"
                f"{sub.get('win_rate', 0.0) * 100:>9.2f}%"
                f"{sub.get('expectancy_pips', 0.0):>12.3f}"
            )
    click.echo("=" * 72)


@click.command("filo-backtest")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("../config/default.ini"),
    show_default=True,
    help="Path to default.ini",
)
@click.option(
    "--data",
    "data_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("../data"),
    show_default=True,
    help="Root directory of extracted parquet bars (one subdir per symbol).",
)
@click.option(
    "--symbol",
    type=str,
    default=None,
    help="Primary symbol (defaults to cfg.symbols.primary).",
)
@click.option("--from", "from_str", type=str, default=None, help="Start date (YYYY-MM-DD).")
@click.option("--to", "to_str", type=str, default=None, help="End date (YYYY-MM-DD).")
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("./bt_out"),
    show_default=True,
    help="Directory to write trades.parquet, candidates.parquet, equity.parquet, metrics.json.",
)
@click.option(
    "--no-correlations",
    is_flag=True,
    default=False,
    help="Skip loading XAGUSD / dxy_m6 correlation bars.",
)
def main(
    config_path: Path,
    data_root: Path,
    symbol: str | None,
    from_str: str | None,
    to_str: str | None,
    out_dir: Path,
    no_correlations: bool,
) -> None:
    """Run an offline backtest and write trade / candidate / equity artifacts."""
    cfg = load_config(config_path)
    sym = symbol or cfg.symbols.primary

    from_ts = pd.Timestamp(from_str) if from_str else None
    to_ts = pd.Timestamp(to_str) if to_str else None

    click.echo(f"[backtest] loading bars from {data_root} — symbol={sym}")
    result = run_from_disk(
        data_root=data_root,
        cfg=cfg,
        symbol=sym,
        from_ts=from_ts,
        to_ts=to_ts,
        load_correlations=not no_correlations,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / "trades.parquet"
    cands_path = out_dir / "candidates.parquet"
    eq_path = out_dir / "equity.parquet"
    metrics_path = out_dir / "metrics.json"

    # Parquet does not accept an empty DataFrame with object dtype without
    # a schema — coerce the empty-case to a one-row sentinel-free frame by
    # keeping the columns only. We write CSV fallback if empty.
    if len(result.trades) > 0:
        result.trades.to_parquet(trades_path, index=False)
    else:
        result.trades.to_csv(trades_path.with_suffix(".csv"), index=False)

    if len(result.candidates) > 0:
        result.candidates.to_parquet(cands_path, index=False)
    else:
        result.candidates.to_csv(cands_path.with_suffix(".csv"), index=False)

    if len(result.equity_curve) > 0:
        result.equity_curve.to_parquet(eq_path, index=False)
    else:
        result.equity_curve.to_csv(eq_path.with_suffix(".csv"), index=False)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result.metrics, f, indent=2, default=str)

    click.echo(f"[backtest] wrote {trades_path.name}, {cands_path.name}, {eq_path.name}, metrics.json")
    _print_summary(sym, result.metrics, len(result.trades), len(result.candidates))


if __name__ == "__main__":
    main()
