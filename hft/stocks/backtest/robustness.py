"""
Quick robustness tests run automatically on the top-N variants after the main sweep.

Tests:
  1. Walk-forward (3 folds: Y1, Y2, Y3)
  2. Sensitivity: ±10% / ±20% perturbation of SL/TP parameters
  3. Symbol-drop: exclude top-5 contributor symbols and re-run
  4. Segment stability: metrics must hold across exchange + sector slices
"""
import logging
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from stocks.backtest.reporting.metrics import compute_trade_metrics, compute_portfolio_metrics
from stocks.backtest.reporting.report_builder import top_variants
from stocks.backtest.simulation.portfolio_sim import run_portfolio_sim
from stocks.backtest.simulation.param_grid import TRADE_VARIANTS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Walk-forward test
# ---------------------------------------------------------------------------

def _date_folds(candidates_df: pd.DataFrame, n_folds: int = 3) -> List[pd.DataFrame]:
    """Split CandidateFrame into n equal time slices."""
    dates  = pd.to_datetime(candidates_df["date"])
    dmin, dmax = dates.min(), dates.max()
    span   = (dmax - dmin) / n_folds
    folds  = []
    for i in range(n_folds):
        start = dmin + i * span
        end   = dmin + (i + 1) * span
        mask  = (dates >= start) & (dates < end)
        folds.append(candidates_df[mask].copy())
    return folds


def walk_forward(
    variant: Dict[str, Any],
    candidates_df: pd.DataFrame,
    all_bars: Dict,
    max_concurrent: int,
    n_folds: int = 3,
) -> pd.DataFrame:
    folds   = _date_folds(candidates_df, n_folds)
    rows    = []
    for i, fold in enumerate(folds):
        if fold.empty:
            continue
        trades, equity = run_portfolio_sim(fold, all_bars, variant, max_concurrent)
        m = compute_trade_metrics(trades, "fold", f"fold_{i+1}", variant["variant_id"])
        m["fold"] = i + 1
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sensitivity test: perturb SL/TP multipliers
# ---------------------------------------------------------------------------

def _perturb(variant: Dict[str, Any], factor: float) -> Dict[str, Any]:
    v = deepcopy(variant)
    for key in ("sl_atr_mult", "sl_pct", "tp_rr", "tp1_rr", "tp2_rr", "trail_atr"):
        if key in v and isinstance(v[key], (int, float)):
            v[key] = v[key] * factor
    return v


def sensitivity(
    variant: Dict[str, Any],
    candidates_df: pd.DataFrame,
    all_bars: Dict,
    max_concurrent: int,
) -> pd.DataFrame:
    factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    rows    = []
    for f in factors:
        v      = _perturb(variant, f)
        trades, _ = run_portfolio_sim(candidates_df, all_bars, v, max_concurrent)
        m = compute_trade_metrics(trades, "sensitivity", f"x{f:.1f}", variant["variant_id"])
        m["perturb_factor"] = f
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Symbol-drop test
# ---------------------------------------------------------------------------

def symbol_drop(
    variant: Dict[str, Any],
    candidates_df: pd.DataFrame,
    all_bars: Dict,
    max_concurrent: int,
    drop_n: int = 5,
) -> pd.DataFrame:
    """Re-run after excluding the N symbols with highest total P&L contribution."""
    trades_full, _ = run_portfolio_sim(candidates_df, all_bars, variant, max_concurrent)
    if trades_full.empty:
        return pd.DataFrame()

    top_syms = (
        trades_full.groupby("symbol")["pnl_usd"].sum()
        .nlargest(drop_n).index.tolist()
    )
    reduced   = candidates_df[~candidates_df["symbol"].isin(top_syms)]
    trades_r, _ = run_portfolio_sim(reduced, all_bars, variant, max_concurrent)
    m = compute_trade_metrics(trades_r, "sym_drop", f"drop_top{drop_n}", variant["variant_id"])
    m["dropped_symbols"] = ",".join(top_syms)
    return pd.DataFrame([m])


# ---------------------------------------------------------------------------
# Segment stability
# ---------------------------------------------------------------------------

def segment_stability(
    variant: Dict[str, Any],
    candidates_df: pd.DataFrame,
    all_bars: Dict,
    max_concurrent: int,
) -> pd.DataFrame:
    rows = []
    for seg_col in ("exchange", "sector"):
        for seg_val in candidates_df[seg_col].unique():
            sub = candidates_df[candidates_df[seg_col] == seg_val]
            if len(sub) < 10:
                continue
            trades, _ = run_portfolio_sim(sub, all_bars, variant, max_concurrent)
            m = compute_trade_metrics(trades, seg_col, seg_val, variant["variant_id"])
            rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Master robustness runner
# ---------------------------------------------------------------------------

def run_robustness(
    trade_metrics_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    all_bars: Dict,
    max_concurrent: int = 5,
    top_n: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Run all robustness tests on the top_n variants.
    Returns a dict of result DataFrames keyed by test name.
    """
    log.info(f"Running robustness tests on top {top_n} variants...")
    top = top_variants(trade_metrics_df, n=top_n, min_trades=10)
    if top.empty:
        log.warning("No variants qualify for robustness testing.")
        return {}

    vmap = {v["variant_id"]: v for v in TRADE_VARIANTS}
    results: Dict[str, List] = {
        "walk_forward":      [],
        "sensitivity":       [],
        "symbol_drop":       [],
        "segment_stability": [],
    }

    for _, row in top.iterrows():
        vid     = int(row["variant_id"])
        variant = vmap[vid]
        vname   = variant["variant_name"]
        log.info(f"  Robustness — variant {vid} ({vname})")

        wf = walk_forward(variant, candidates_df, all_bars, max_concurrent)
        wf["variant_name"] = vname
        results["walk_forward"].append(wf)

        se = sensitivity(variant, candidates_df, all_bars, max_concurrent)
        se["variant_name"] = vname
        results["sensitivity"].append(se)

        sd = symbol_drop(variant, candidates_df, all_bars, max_concurrent)
        if not sd.empty:
            sd["variant_name"] = vname
            results["symbol_drop"].append(sd)

        ss = segment_stability(variant, candidates_df, all_bars, max_concurrent)
        ss["variant_name"] = vname
        results["segment_stability"].append(ss)

    return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame()
            for k, v in results.items()}


def print_robustness_summary(rob: Dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 70)
    print("  ROBUSTNESS RESULTS")
    print("=" * 70)

    wf = rob.get("walk_forward", pd.DataFrame())
    if not wf.empty:
        print("\n--- Walk-Forward (fold consistency) ---")
        pivot = wf.pivot_table(
            index="variant_name", columns="fold",
            values="sharpe_R", aggfunc="first"
        )
        print(pivot.to_string())

    se = rob.get("sensitivity", pd.DataFrame())
    if not se.empty:
        print("\n--- Sensitivity (±20% parameter perturbation) ---")
        cols = ["variant_name", "perturb_factor", "win_rate", "avg_R", "sharpe_R"]
        print(se[[c for c in cols if c in se.columns]].to_string(index=False))

    sd = rob.get("symbol_drop", pd.DataFrame())
    if not sd.empty:
        print("\n--- Symbol-Drop (exclude top-5 contributors) ---")
        cols = ["variant_name", "total_trades", "win_rate", "avg_R", "sharpe_R", "dropped_symbols"]
        print(sd[[c for c in cols if c in sd.columns]].to_string(index=False))

    ss = rob.get("segment_stability", pd.DataFrame())
    if not ss.empty:
        print("\n--- Segment Stability (exchange + sector breakdown) ---")
        cols = ["variant_name", "segment_type", "segment_value",
                "total_trades", "win_rate", "avg_R", "sharpe_R"]
        print(ss[[c for c in cols if c in ss.columns]].head(40).to_string(index=False))
