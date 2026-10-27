"""
Aggregate all MetricRows from the sweep and write output files.
Also runs robustness tests on the top-N variants after the main sweep.
"""
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from stocks.backtest.config import RESULTS_DIR
from stocks.backtest.simulation.param_grid import TRADE_VARIANTS, ENTRY_GRID

log = logging.getLogger(__name__)


def save_results(
    all_metrics: List[Dict],
    portfolio_metrics: List[Dict],
    entry_metrics: List[Dict],
    run_tag: str = "",
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{run_tag}" if run_tag else ""

    # --- Trade-level metrics ---
    trade_df = pd.DataFrame(all_metrics)
    trade_path = RESULTS_DIR / f"trade_metrics{tag}.csv"
    trade_df.to_csv(trade_path, index=False)

    # --- Portfolio-level metrics ---
    port_df   = pd.DataFrame(portfolio_metrics)
    port_path = RESULTS_DIR / f"portfolio_metrics{tag}.csv"
    port_df.to_csv(port_path, index=False)

    # --- Entry param metrics ---
    entry_df   = pd.DataFrame(entry_metrics)
    entry_path = RESULTS_DIR / f"entry_metrics{tag}.csv"
    entry_df.to_csv(entry_path, index=False)

    log.info(f"Results saved: {trade_path}")
    return trade_path


def _calmar_composite(df: pd.DataFrame) -> pd.Series:
    """
    Calmar-penalized composite = sharpe_R × profit_factor × win_rate
                                  / (1 + |max_loss_R| / 5)
    Rewards return consistency while penalising large single-trade losses.
    """
    return (
        df["sharpe_R"].clip(lower=0) *
        df["profit_factor"].clip(lower=0, upper=10) *
        df["win_rate"].clip(lower=0) /
        (1 + df["max_loss_R"].abs().clip(upper=5) / 5)
    )


def top_variants(
    trade_df: pd.DataFrame,
    n: int = 10,
    min_trades: int = 20,
) -> pd.DataFrame:
    """
    Rank variants by Calmar-penalized composite on segment_value='ALL'.
    """
    if trade_df.empty:
        return pd.DataFrame()

    all_seg = trade_df[
        (trade_df["segment_value"] == "ALL") &
        (trade_df["total_trades"] >= min_trades)
    ].copy()

    if all_seg.empty:
        return pd.DataFrame()

    all_seg["composite"] = _calmar_composite(all_seg)
    top = all_seg.nlargest(n, "composite").reset_index(drop=True)

    vmap = {v["variant_id"]: v["variant_name"] for v in TRADE_VARIANTS}
    top["variant_name"] = top["variant_id"].map(vmap)
    return top


def best_by_segment(
    trade_df: pd.DataFrame,
    min_trades: int = 15,
) -> pd.DataFrame:
    """
    For each exchange and sector segment, return the best trade variant
    ranked by Calmar-penalized composite.
    """
    if trade_df.empty:
        return pd.DataFrame()

    rows = []
    vmap = {v["variant_id"]: v["variant_name"] for v in TRADE_VARIANTS}

    for seg_type in ("exchange", "sector", "vol_tier"):
        seg_data = trade_df[
            (trade_df["segment_type"] == seg_type) &
            (trade_df["total_trades"] >= min_trades) &
            trade_df["segment_value"].notna() &
            (trade_df["segment_value"].astype(str).str.strip() != "")
        ].copy()
        if seg_data.empty:
            continue
        seg_data["composite"] = _calmar_composite(seg_data)
        best = (
            seg_data.sort_values("composite", ascending=False)
            .groupby("segment_value")
            .first()
            .reset_index()
        )
        best["variant_name"] = best["variant_id"].map(vmap)
        best["seg_type"] = seg_type
        rows.append(best)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def top_entry_configs(
    entry_df: pd.DataFrame,
    n: int = 5,
    min_trades: int = 20,
) -> pd.DataFrame:
    if entry_df.empty:
        return pd.DataFrame()
    filt = entry_df[entry_df["total_trades"] >= min_trades].copy()
    if filt.empty:
        return pd.DataFrame()
    filt["composite"] = (
        filt["sharpe_R"].clip(lower=0) *
        filt["profit_factor"].clip(lower=0, upper=10) *
        filt["win_rate"].clip(lower=0)
    )
    top = filt.nlargest(n, "composite").reset_index(drop=True)
    emap = {e["entry_id"]: e for e in ENTRY_GRID}
    for k in ["lrc_length", "sigma_mult", "min_score", "touch_pct", "vel_required"]:
        top[k] = top["entry_id"].map(lambda eid, _k=k: emap.get(eid, {}).get(_k))
    return top


def print_summary(trade_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  BACKTEST SUMMARY — TOP 10 TRADE MANAGEMENT VARIANTS (ALL segments)")
    print("=" * 70)
    tv = top_variants(trade_df)
    if tv.empty:
        print("  No results."); return
    cols = ["variant_name", "total_trades", "win_rate", "avg_R",
            "profit_factor", "sharpe_R", "avg_bars_held", "composite"]
    tv2 = tv[[c for c in cols if c in tv.columns]].copy()
    tv2["win_rate"]     = tv2["win_rate"].map(lambda x: f"{x:.1%}")
    tv2["avg_R"]        = tv2["avg_R"].map(lambda x: f"{x:.3f}")
    tv2["profit_factor"]= tv2["profit_factor"].map(lambda x: f"{x:.2f}")
    tv2["sharpe_R"]     = tv2["sharpe_R"].map(lambda x: f"{x:.2f}")
    tv2["composite"]    = tv2["composite"].map(lambda x: f"{x:.3f}")
    print(tv2.to_string(index=False))

    print("\n" + "=" * 70)
    print("  PORTFOLIO METRICS — TOP 5 by CAGR")
    print("=" * 70)
    if not portfolio_df.empty:
        top5 = portfolio_df.nlargest(5, "cagr_%")[
            ["variant_id", "max_concurrent", "final_pnl_usd", "cagr_%", "sharpe", "max_drawdown_%"]
        ]
        print(top5.to_string(index=False))
