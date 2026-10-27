"""Replay driver — loads bars from disk and feeds them to the engine.

This is the thin "glue" layer between `filo_hft.data.loaders.load_bars`
and `filo_hft.backtest.engine.run_backtest`. It knows which timeframes
the engine wants and how to fetch and (optionally) date-clip them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from filo_hft.backtest.engine import TF_ORDER, BacktestResult, run_backtest
from filo_hft.config import FiloConfig
from filo_hft.data.loaders import load_bars

__all__ = ["load_replay_bars", "run_from_disk"]


# Timeframes we always try to load for the primary symbol.
PRIMARY_TFS: tuple[str, ...] = ("M1",) + TF_ORDER


def _to_utc(ts: pd.Timestamp | None) -> pd.Timestamp | None:
    if ts is None:
        return None
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _clip(df: pd.DataFrame, from_ts: pd.Timestamp | None, to_ts: pd.Timestamp | None) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df
    from_ts = _to_utc(from_ts)
    to_ts = _to_utc(to_ts)
    if from_ts is not None:
        out = out[out["ts"] >= from_ts]
    if to_ts is not None:
        out = out[out["ts"] <= to_ts]
    return out.reset_index(drop=True)


def load_replay_bars(
    data_root: Path,
    cfg: FiloConfig,
    symbol: str | None = None,
    from_ts: pd.Timestamp | None = None,
    to_ts: pd.Timestamp | None = None,
    load_correlations: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load primary TFs + optional correlation M1 bars.

    Returns a tuple `(bars_by_tf, corr_bars)` ready for `run_backtest`.
    Missing timeframes (on disk) are silently skipped — the engine copes
    with missing higher TFs and missing correlation symbols.
    """
    sym = symbol or cfg.symbols.primary
    bars_by_tf: dict[str, pd.DataFrame] = {}
    for tf in PRIMARY_TFS:
        try:
            df = load_bars(data_root, sym, tf)
        except FileNotFoundError:
            continue
        df = _clip(df, from_ts, to_ts)
        if len(df) > 0:
            bars_by_tf[tf] = df

    corr_bars: dict[str, pd.DataFrame] = {}
    if load_correlations:
        for sec in (cfg.symbols.silver_proxy, cfg.symbols.dxy_proxy):
            try:
                df = load_bars(data_root, sec, "M1")
            except FileNotFoundError:
                continue
            df = _clip(df, from_ts, to_ts)
            if len(df) > 0:
                corr_bars[sec] = df

    if "M1" not in bars_by_tf:
        raise FileNotFoundError(
            f"No M1 bars found for {sym} under {data_root} — cannot run backtest"
        )

    return bars_by_tf, corr_bars


def run_from_disk(
    data_root: Path,
    cfg: FiloConfig,
    symbol: str | None = None,
    from_ts: pd.Timestamp | None = None,
    to_ts: pd.Timestamp | None = None,
    load_correlations: bool = True,
) -> BacktestResult:
    """End-to-end: load bars, run the engine, return the result."""
    bars_by_tf, corr_bars = load_replay_bars(
        data_root, cfg, symbol=symbol, from_ts=from_ts, to_ts=to_ts,
        load_correlations=load_correlations,
    )
    return run_backtest(bars_by_tf, cfg, corr_bars)
