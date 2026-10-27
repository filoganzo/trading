"""
FTSE MIB Pairs — M15 Staggered Entry Analysis
===============================================
Replicates the trader's approach:
  - M15 bars, 15-day rolling window for key level detection
  - Ratio chart key levels found via local extrema clustering
  - Staggered entry: open 1 unit at L1, add at L2, add at L3
  - Exit all units when ratio reverts to center key level
  - Compare staggered vs single-entry on same signals

Run from project root:
    python -m filo_hft.scripts.pairs_staggered
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Top pairs confirmed from H1 screen ────────────────────────────
CANDIDATE_PAIRS = [
    ("ENEL", "TRN"),   # best WF Sharpe, utility grid pair
    ("IG",   "TRN"),   # IG = Italgas, TRN = Terna — same network
    ("ENEL", "IG"),    # both Italian grid regulated
    ("ENEL", "SRG"),   # Snam gas + Enel electric
    ("BAMI", "BZU"),   # bank + cement — check if structural
    ("HER",  "RACE"),  # Hera utility vs Ferrari — likely macro beta only
    ("TIT",  "UNI"),   # TIM + Unipol — check
    ("BMPS", "MB"),    # Monte Paschi + Mediobanca — financials
    ("ISP",  "UCG"),   # trader's original pair
    ("ISP",  "G"),     # Intesa + Generali
    ("UCG",  "G"),     # UniCredit + Generali
    ("ISP",  "MB"),    # Intesa + Mediobanca
]

YF_SUFFIX = ".MI"
TF = "M15"
LEVEL_WINDOW_DAYS = 15      # rolling window for key level detection
HISTORY_DAYS = 58           # yfinance M15 cap
N_LEVELS = 6                # key levels to cluster per window
MAX_UNITS = 3               # max stagger depth
CAPITAL_PER_UNIT = 5_000.0  # EUR per unit per leg
SPREAD_PCT = 0.04           # 4 bps per leg


# ─────────────────────────────────────────────────────────────────
# Data fetch
# ─────────────────────────────────────────────────────────────────
def fetch_yf(sym: str, interval: str = "15m", days: int = 58) -> pd.Series | None:
    import yfinance as yf
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    try:
        raw = yf.download(
            sym + YF_SUFFIX, start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"), interval=interval,
            progress=False, auto_adjust=True, actions=False,
        )
    except Exception:
        return None
    if raw is None or len(raw) < 50:
        return None
    raw.index = pd.to_datetime(raw.index, utc=True)
    col = [c for c in raw.columns if str(c).lower() == "close"]
    if not col:
        col = [raw.columns[3]]  # 4th col is close in OHLCV
    s = raw[col[0]].rename(sym).dropna()
    return s if len(s) >= 50 else None


# ─────────────────────────────────────────────────────────────────
# Key level detection on ratio series
# ─────────────────────────────────────────────────────────────────
def detect_key_levels(ratio: np.ndarray, n_levels: int = N_LEVELS,
                      order: int = 3) -> np.ndarray:
    """Cluster local extrema of ratio into n_levels support/resistance lines."""
    hi = argrelextrema(ratio, np.greater_equal, order=order)[0]
    lo = argrelextrema(ratio, np.less_equal,    order=order)[0]
    pivots = ratio[np.concatenate([hi, lo])]
    pivots = pivots[np.isfinite(pivots)]
    if len(pivots) < n_levels:
        return np.percentile(ratio[np.isfinite(ratio)], np.linspace(5, 95, n_levels))
    km = KMeans(n_clusters=n_levels, n_init=10, random_state=42)
    km.fit(pivots.reshape(-1, 1))
    return np.sort(km.cluster_centers_.flatten())


# ─────────────────────────────────────────────────────────────────
# STAGGERED ENTRY back-test
# ─────────────────────────────────────────────────────────────────
def backtest_staggered(
    ratio: pd.Series,
    level_window: int,          # bars to use for key level computation
    n_levels: int = N_LEVELS,
    max_units: int = MAX_UNITS,
    capital_per_unit: float = CAPITAL_PER_UNIT,
    spread_pct: float = SPREAD_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (trade_log, bar_log).

    Logic:
      - Roll a 'level_window' lookback; at each bar recompute key levels.
      - Levels are sorted: L[0] < L[1] < ... < L[n-1]; median ~ L[n//2].
      - If ratio > L[n//2]: ratio is HIGH  -> short ratio (short A, long B)
        Stagger: open 1 unit at L[n//2+1], add at L[n//2+2], add at L[n//2+3]
      - If ratio < L[n//2]: ratio is LOW   -> long ratio (long A, short B)
        Stagger: open 1 unit at L[n//2-1], add at L[n//2-2], add at L[n//2-3]
      - Exit: ratio crosses back through L[n//2] (the median level)
      - Stop: ratio goes past L[0] or L[-1] (extreme outlier)
    """
    arr = ratio.values
    idx = ratio.index
    n = len(arr)
    bars_per_day = 26  # ~6.5 h × 4 bars/h for Italian mkt 09:00-17:30

    trades = []
    bar_records = []

    units: list[dict] = []   # each: {direction, entry_ratio, entry_bar}
    exit_level = np.nan
    stop_hi = np.nan
    stop_lo = np.nan

    for i in range(level_window, n):
        r = arr[i]
        if not np.isfinite(r):
            continue

        # Recompute levels on rolling window (every 'bars_per_day' bars = once/day)
        if (i - level_window) % bars_per_day == 0 or i == level_window:
            window_ratio = arr[i - level_window: i]
            levels = detect_key_levels(window_ratio, n_levels)
            mid_idx = n_levels // 2
            exit_level = levels[mid_idx]
            # Stop beyond outermost entry levels by 1x the inter-level spacing
            spacing = float(np.diff(levels).mean()) if len(levels) > 1 else float(levels[-1] * 0.01)
            stop_lo = levels[0]  - spacing
            stop_hi = levels[-1] + spacing
            # Entry level thresholds above/below mid
            entry_levels_hi = levels[mid_idx + 1: mid_idx + 1 + max_units]  # to short ratio
            entry_levels_lo = levels[mid_idx - max_units: mid_idx][::-1]     # to long ratio

        # ── Exit check ──
        if units:
            direction = units[0]["direction"]
            hit_exit = (direction == -1 and r <= exit_level) or \
                       (direction == +1 and r >= exit_level)
            hit_stop = r <= stop_lo or r >= stop_hi

            if hit_exit or hit_stop:
                for u in units:
                    pnl_ratio = direction * (u["entry_ratio"] - r)
                    pnl_pct = pnl_ratio / u["entry_ratio"]
                    gross = pnl_pct * capital_per_unit * 2  # 2 legs
                    cost = spread_pct / 100.0 * capital_per_unit * 2
                    trades.append({
                        "ts_open":      idx[u["entry_bar"]],
                        "ts_close":     idx[i],
                        "direction":    direction,
                        "unit":         u["unit_num"],
                        "entry_ratio":  u["entry_ratio"],
                        "exit_ratio":   r,
                        "gross_pnl":    gross,
                        "net_pnl":      gross - cost,
                        "exit_type":    "stop" if hit_stop else "tp",
                        "duration_bars": i - u["entry_bar"],
                    })
                units = []

        # ── Entry / stagger check ──
        if len(units) < max_units:
            direction = units[0]["direction"] if units else 0
            unit_entries = [u["entry_level_idx"] for u in units]

            # Short ratio (ratio too high)
            for lvl_idx, lvl in enumerate(entry_levels_hi):
                if lvl_idx in unit_entries:
                    continue
                if r >= lvl and (direction == 0 or direction == -1):
                    units.append({
                        "direction":     -1,
                        "entry_ratio":   r,
                        "entry_bar":     i,
                        "entry_level_idx": lvl_idx,
                        "unit_num":      len(units) + 1,
                    })
                    break

            # Long ratio (ratio too low)
            for lvl_idx, lvl in enumerate(entry_levels_lo):
                if lvl_idx in unit_entries:
                    continue
                if r <= lvl and (direction == 0 or direction == +1):
                    units.append({
                        "direction":     +1,
                        "entry_ratio":   r,
                        "entry_bar":     i,
                        "entry_level_idx": lvl_idx,
                        "unit_num":      len(units) + 1,
                    })
                    break

        bar_records.append({
            "ts":        idx[i],
            "ratio":     r,
            "n_units":   len(units),
            "direction": units[0]["direction"] if units else 0,
        })

    trade_df = pd.DataFrame(trades)
    bar_df   = pd.DataFrame(bar_records)
    return trade_df, bar_df


# ─────────────────────────────────────────────────────────────────
# SINGLE ENTRY back-test (same key level, only first unit)
# ─────────────────────────────────────────────────────────────────
def backtest_single(
    ratio: pd.Series,
    level_window: int,
    n_levels: int = N_LEVELS,
    capital_per_unit: float = CAPITAL_PER_UNIT,
    spread_pct: float = SPREAD_PCT,
) -> pd.DataFrame:
    """Same logic but max_units=1 — baseline comparison."""
    t, _ = backtest_staggered(
        ratio, level_window, n_levels,
        max_units=1, capital_per_unit=capital_per_unit, spread_pct=spread_pct,
    )
    return t


# ─────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────
def metrics(trades: pd.DataFrame, label: str = "") -> dict:
    if trades is None or len(trades) == 0:
        return {"label": label, "n": 0, "pnl": 0, "win_rate": 0,
                "sharpe": 0, "max_dd_pct": 0, "profit_factor": 0,
                "avg_dur_bars": 0, "pnl_per_trade": 0}
    # Aggregate by trade-group (all units of same entry event) using ts_close
    g = trades.groupby("ts_open")["net_pnl"].sum().reset_index()
    p = g["net_pnl"]
    wins = p[p > 0]
    losses = p[p < 0]
    equity = p.cumsum()
    peak = equity.cummax()
    dd = (peak - equity) / peak.where(peak > 0, 1.0) * 100
    gw, gl = wins.sum(), abs(losses.sum())
    dur = trades.groupby("ts_open")["duration_bars"].max().mean()
    return {
        "label": label,
        "n_events": len(p),
        "n_units_total": len(trades),
        "avg_units_per_event": len(trades) / max(len(p), 1),
        "pnl": float(p.sum()),
        "pnl_per_event": float(p.mean()),
        "win_rate": float(len(wins) / len(p)),
        "sharpe": float(p.mean() / p.std() * np.sqrt(len(p))) if p.std() > 0 else 0,
        "max_dd_pct": float(dd.max()),
        "profit_factor": float(gw / gl) if gl > 0 else (99.0 if gw > 0 else 0.0),
        "avg_dur_bars": float(dur) if pd.notna(dur) else 0,
        "n_stops": int((trades["exit_type"] == "stop").sum()),
    }


# ─────────────────────────────────────────────────────────────────
# Walk-forward on staggered vs single
# ─────────────────────────────────────────────────────────────────
def walk_forward_compare(ratio: pd.Series, level_window: int,
                         n_folds: int = 4) -> pd.DataFrame:
    n = len(ratio)
    fold_size = n // (n_folds + 1)
    rows = []
    for i in range(n_folds):
        start = fold_size * (i + 1)
        end = start + fold_size
        if end > n:
            break
        fold_ratio = ratio.iloc[start:end]
        ts = backtest_staggered(fold_ratio, level_window)[0]
        ss = backtest_single(fold_ratio, level_window)
        ms = metrics(ts, "staggered")
        ms["fold"] = i
        ms["method"] = "staggered"
        rows.append(ms)
        ms2 = metrics(ss, "single")
        ms2["fold"] = i
        ms2["method"] = "single"
        rows.append(ms2)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# Stagger depth analysis — how often is each unit # profitable?
# ─────────────────────────────────────────────────────────────────
def stagger_depth_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    if len(trades) == 0:
        return pd.DataFrame()
    rows = []
    for u in range(1, MAX_UNITS + 1):
        sub = trades[trades["unit"] == u]
        if len(sub) == 0:
            continue
        p = sub["net_pnl"]
        rows.append({
            "unit": u,
            "count": len(sub),
            "pnl": float(p.sum()),
            "pnl_per_trade": float(p.mean()),
            "win_rate": float((p > 0).mean()),
            "n_stops": int((sub["exit_type"] == "stop").sum()),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────
def plot_pair_staggered(
    ratio: pd.Series,
    levels: np.ndarray,
    stagger_trades: pd.DataFrame,
    single_trades: pd.DataFrame,
    bar_df: pd.DataFrame,
    sym_a: str, sym_b: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.3)

    ax_ratio  = fig.add_subplot(gs[0, :])
    ax_units  = fig.add_subplot(gs[1, :], sharex=ax_ratio)
    ax_eq_s   = fig.add_subplot(gs[2, 0])
    ax_eq_1   = fig.add_subplot(gs[2, 1])
    ax_depth  = fig.add_subplot(gs[3, 0])
    ax_dur    = fig.add_subplot(gs[3, 1])

    x = np.arange(len(ratio))

    # Panel 1: ratio + key levels
    ax_ratio.plot(x, ratio.values, color="steelblue", lw=0.7, label="Ratio A/B")
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(levels)))
    for lv, c in zip(levels, colors):
        ax_ratio.axhline(lv, color=c, lw=0.9, ls="--", alpha=0.8)
    mid_lv = levels[len(levels) // 2]
    ax_ratio.axhline(mid_lv, color="black", lw=1.2, ls="-", alpha=0.5, label="Mid level")
    ax_ratio.set_title(f"{sym_a} / {sym_b}  — M15 Ratio + Key Levels", fontsize=11, fontweight="bold")
    ax_ratio.legend(fontsize=8)
    ax_ratio.set_ylabel("Ratio")

    # Panel 2: units open
    if len(bar_df) > 0:
        bx = np.arange(len(bar_df))
        ax_units.fill_between(bx, 0, bar_df["n_units"].values,
                              where=(bar_df["direction"].values == -1),
                              color="red", alpha=0.5, label="Short ratio units")
        ax_units.fill_between(bx, 0, bar_df["n_units"].values,
                              where=(bar_df["direction"].values == +1),
                              color="green", alpha=0.5, label="Long ratio units")
        ax_units.set_ylabel("Open units")
        ax_units.set_yticks([0, 1, 2, 3])
        ax_units.legend(fontsize=8)
        ax_units.set_title("Stagger depth over time")

    # Panel 3 & 4: equity curves
    def _equity_plot(ax, trades, title):
        if len(trades) == 0:
            ax.set_title(title + " (no trades)")
            return
        pnl_by_event = trades.groupby("ts_open")["net_pnl"].sum()
        eq = pnl_by_event.cumsum().values
        ax.plot(eq, color="green" if eq[-1] > 0 else "red", lw=1.2)
        ax.fill_between(range(len(eq)), eq, 0,
                        where=(eq >= 0), alpha=0.2, color="green")
        ax.fill_between(range(len(eq)), eq, 0,
                        where=(eq < 0), alpha=0.2, color="red")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Cumul. PnL (EUR)")
        ax.set_xlabel("Trade #")

    _equity_plot(ax_eq_s, stagger_trades, f"Staggered (max {MAX_UNITS} units)")
    _equity_plot(ax_eq_1, single_trades,  "Single entry (1 unit)")

    # Panel 5: per-unit breakdown
    depth = stagger_depth_breakdown(stagger_trades)
    if len(depth) > 0:
        bars = ax_depth.bar(depth["unit"], depth["pnl"], color=["#2ecc71", "#f39c12", "#e74c3c"])
        ax_depth.axhline(0, color="black", lw=0.5)
        ax_depth.set_xlabel("Stagger unit #")
        ax_depth.set_ylabel("Total PnL (EUR)")
        ax_depth.set_title("PnL contribution by stagger depth")
        ax_depth.set_xticks(depth["unit"])
        for bar, wr in zip(bars, depth["win_rate"]):
            ax_depth.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + (abs(bar.get_height()) * 0.05),
                          f"WR {wr:.0%}", ha="center", va="bottom", fontsize=8)

    # Panel 6: trade duration histogram
    if len(stagger_trades) > 0:
        dur_h = stagger_trades["duration_bars"] * 15 / 60  # convert to hours
        ax_dur.hist(dur_h, bins=20, color="steelblue", alpha=0.8, edgecolor="white")
        ax_dur.axvline(dur_h.median(), color="red", lw=1.5, ls="--", label=f"Median {dur_h.median():.1f}h")
        ax_dur.set_xlabel("Trade duration (hours)")
        ax_dur.set_ylabel("Count")
        ax_dur.set_title("Duration distribution (staggered units)")
        ax_dur.legend(fontsize=8)

    plt.suptitle(f"FTSE MIB Pairs Arb — {sym_a}/{sym_b} — M15 Staggered Entry Analysis",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_summary_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    """Bar chart comparing staggered vs single across all pairs."""
    if len(summary) == 0:
        return
    pairs = summary["pair"].unique()
    x = np.arange(len(pairs))
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics_to_plot = [
        ("sharpe",      "Sharpe ratio"),
        ("profit_factor", "Profit factor"),
        ("pnl",         "Total PnL (EUR)"),
    ]
    for ax, (col, title) in zip(axes, metrics_to_plot):
        s_vals = summary[summary["method"] == "staggered"].set_index("pair")[col].reindex(pairs).fillna(0)
        e_vals = summary[summary["method"] == "single"].set_index("pair")[col].reindex(pairs).fillna(0)
        bars1 = ax.bar(x - w / 2, s_vals, w, label="Staggered", color="#2ecc71", alpha=0.85)
        bars2 = ax.bar(x + w / 2, e_vals, w, label="Single",    color="#3498db", alpha=0.85)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(pairs, rotation=45, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    out = Path("data/pairs_staggered")
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    print("\n" + "=" * 62)
    print("  FTSE MIB Pairs — M15 Staggered Entry  |  15-day key levels")
    print("=" * 62 + "\n")

    bars_per_day = 26
    level_window = LEVEL_WINDOW_DAYS * bars_per_day   # ~390 bars

    # ── Load data ──────────────────────────────────────────────
    print("[1/4] Fetching M15 data from Yahoo Finance...")
    all_syms = list({s for pair in CANDIDATE_PAIRS for s in pair})
    price_cache: dict[str, pd.Series] = {}
    for sym in all_syms:
        s = fetch_yf(sym, interval="15m", days=HISTORY_DAYS)
        if s is not None:
            price_cache[sym] = s
            print(f"  {sym}: {len(s)} bars")
        else:
            print(f"  {sym}: FAILED")

    # ── Per-pair analysis ──────────────────────────────────────
    print("\n[2/4] Running staggered vs single backtest per pair...")
    summary_rows = []
    wf_rows = []
    depth_rows = []

    for sym_a, sym_b in CANDIDATE_PAIRS:
        if sym_a not in price_cache or sym_b not in price_cache:
            print(f"  {sym_a}/{sym_b}: skip (missing data)")
            continue

        # Align
        merged = pd.DataFrame({"a": price_cache[sym_a], "b": price_cache[sym_b]}).dropna()
        if len(merged) < level_window + 50:
            print(f"  {sym_a}/{sym_b}: skip (only {len(merged)} aligned bars)")
            continue

        ratio = (merged["a"] / merged["b"]).rename(f"{sym_a}/{sym_b}")

        # Cointegration check on full sample
        try:
            _, coint_p, _ = coint(merged["a"].values, merged["b"].values)
        except Exception:
            coint_p = 1.0

        # Full-sample backtest
        stagger_t, bar_df = backtest_staggered(ratio, level_window)
        single_t          = backtest_single(ratio, level_window)
        ms = metrics(stagger_t, "staggered")
        me = metrics(single_t,  "single")

        # Key levels on last 15 days
        last_window = ratio.values[-level_window:]
        levels = detect_key_levels(last_window)

        pair_label = f"{sym_a}/{sym_b}"
        print(f"\n  {pair_label}  (coint p={coint_p:.3f})")
        print(f"    Staggered: n={ms['n_events']} events, "
              f"avg_units={ms['avg_units_per_event']:.2f}, "
              f"PnL={ms['pnl']:.0f}, Sharpe={ms['sharpe']:.2f}, "
              f"PF={ms['profit_factor']:.2f}, stops={ms['n_stops']}")
        print(f"    Single:    n={me['n_events']} events, "
              f"PnL={me['pnl']:.0f}, Sharpe={me['sharpe']:.2f}, "
              f"PF={me['profit_factor']:.2f}, stops={me['n_stops']}")

        for m, method in [(ms, "staggered"), (me, "single")]:
            r = dict(m)
            r["pair"] = pair_label
            r["method"] = method
            r["coint_p"] = coint_p
            summary_rows.append(r)

        # Stagger depth breakdown
        db = stagger_depth_breakdown(stagger_t)
        if len(db) > 0:
            db["pair"] = pair_label
            depth_rows.append(db)

        # Walk-forward
        wf = walk_forward_compare(ratio, level_window)
        if len(wf) > 0:
            wf["pair"] = pair_label
            wf_rows.append(wf)

        # Plot
        plot_pair_staggered(
            ratio, levels, stagger_t, single_t, bar_df,
            sym_a, sym_b,
            plots / f"{sym_a}_{sym_b}_staggered.png"
        )

    # ── Aggregate results ──────────────────────────────────────
    print("\n[3/4] Aggregating results...")
    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        summary_df.to_csv(out / "summary.csv", index=False)

    if depth_rows:
        depth_df = pd.concat(depth_rows).reset_index(drop=True)
        depth_df.to_csv(out / "stagger_depth.csv", index=False)

    if wf_rows:
        wf_df = pd.concat(wf_rows).reset_index(drop=True)
        wf_df.to_csv(out / "walk_forward.csv", index=False)
        wf_sum_cols = [c for c in ["pnl", "sharpe", "win_rate", "n_events"] if c in wf_df.columns]
        wf_sum = (wf_df.groupby(["pair", "method"])[wf_sum_cols]
                  .mean().reset_index())
        wf_sum["pct_prof_folds"] = (
            wf_df.groupby(["pair", "method"])["pnl"]
            .apply(lambda x: f"{(x > 0).mean():.0%}")
            .values
        )
        wf_sum.to_csv(out / "walk_forward_summary.csv", index=False)

    # ── Comparison chart ──────────────────────────────────────
    print("[4/4] Generating comparison chart...")
    if len(summary_df) > 0:
        plot_summary_comparison(summary_df, plots / "_comparison_staggered_vs_single.png")

    # ── Print final tables ────────────────────────────────────
    print("\n" + "=" * 62)
    print("FULL SAMPLE — STAGGERED vs SINGLE (sorted by staggered Sharpe)")
    print("=" * 62)

    if len(summary_df) > 0:
        cols = ["pair", "method", "coint_p", "n_events", "avg_units_per_event",
                "pnl", "win_rate", "sharpe", "max_dd_pct", "profit_factor", "n_stops"]
        avail = [c for c in cols if c in summary_df.columns]
        stagger_order = (summary_df[summary_df["method"] == "staggered"]
                         .sort_values("sharpe", ascending=False)["pair"].tolist())
        ordered = pd.concat([
            summary_df[summary_df["pair"] == p] for p in stagger_order
        ])
        print(ordered[avail].to_string(index=False))

    if wf_rows:
        print("\n" + "=" * 62)
        print("WALK-FORWARD AVERAGE — STAGGERED vs SINGLE")
        print("=" * 62)
        print(wf_sum.sort_values(["method", "sharpe"], ascending=[True, False]).to_string(index=False))

    if depth_rows:
        print("\n" + "=" * 62)
        print("STAGGER DEPTH — PnL BY UNIT NUMBER (aggregated across pairs)")
        print("=" * 62)
        agg_depth = (pd.concat(depth_rows)
                     .groupby("unit")[["count", "pnl", "pnl_per_trade", "win_rate", "n_stops"]]
                     .sum())
        agg_depth["win_rate"] = agg_depth["win_rate"] / agg_depth["count"]  # re-average
        print(agg_depth.to_string())

    print(f"\n[done] Saved to {out}/")
    print(f"       summary.csv | stagger_depth.csv | walk_forward_summary.csv | plots/")


if __name__ == "__main__":
    project_root = Path(__file__).parents[3]
    os.chdir(project_root)
    main()
