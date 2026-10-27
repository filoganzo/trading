"""
FTSE MIB Pairs — Alpha Hunt + Live Screener
============================================
1. Re-run M15 / M30 backtests on top pairs with correct parameters
   (window scaled to match the winning H1 60-bar lookback)
2. Slice alpha by: time-of-day, weekday, ratio vol regime, trend/range
3. Print a live screener table: current z-score, distance to key levels,
   economic rationale, action signal

Run:
    python -m filo_hft.scripts.pairs_alpha            # full analysis
    python -m filo_hft.scripts.pairs_alpha --screen   # screener only (fast)
"""

from __future__ import annotations

import argparse
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
from statsmodels.tsa.stattools import coint, adfuller

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────
# Pair universe — econometric rationale baked in
# ─────────────────────────────────────────────────────────────────
PAIRS = {
    # Pair              : (sym_a, sym_b, economic_rationale, sector)
    "ENEL/TRN"  : ("ENEL", "TRN",
                   "Both Italian electricity: ENEL generates, TRN transmits. "
                   "Same ARERA regulatory tariff framework -> shared discount rate.",
                   "Utilities"),
    "IG/TRN"    : ("IG",   "TRN",
                   "Italgas (gas distribution) + Terna (electricity grid). "
                   "Both ARERA-regulated infrastructure, same WACC methodology.",
                   "Utilities"),
    "ENEL/IG"   : ("ENEL", "IG",
                   "Electricity + gas Italian regulated infra. "
                   "Tariff resets are correlated; both trade on regulated yield spread.",
                   "Utilities"),
    "ENEL/SRG"  : ("ENEL", "SRG",
                   "Enel (electricity) + Snam (gas pipelines). "
                   "Both Italian energy infrastructure with regulated returns.",
                   "Utilities"),
    "TIT/UNI"   : ("TIT",  "UNI",
                   "TIM + Unipol: large Italian domestics with heavy retail investor base. "
                   "Move in tandem on Italian risk-on/off and BTP spread.",
                   "Domestic macro"),
    "ISP/UCG"   : ("ISP",  "UCG",
                   "Italy's two largest banks. Same ECB rate sensitivity, "
                   "same Italian NPA legacy exposure, same BTP mark-to-market risk.",
                   "Banks"),
    "ISP/MB"    : ("ISP",  "MB",
                   "Intesa (retail bank) + Mediobanca (investment bank). "
                   "Correlated on Italian credit cycle; MB is ISP's largest single shareholder.",
                   "Banks"),
    "UCG/G"     : ("UCG",  "G",
                   "UniCredit + Generali. Both historically linked via shareholder structure "
                   "and Italian financial sector sentiment.",
                   "Financials"),
    "BMPS/MB"   : ("BMPS", "MB",
                   "Monte Paschi + Mediobanca. Both sensitive to Italian banking M&A flow "
                   "and ECB supervisory decisions.",
                   "Banks"),
    "BAMI/ISP"  : ("BAMI", "ISP",
                   "Banco BPM + Intesa: mid-cap vs large-cap Italian retail banks. "
                   "Near-perfect rate sensitivity match; diverge on M&A premium.",
                   "Banks"),
    "HER/A2A"   : ("HER",  "A2A",
                   "Hera + A2A: both Italian multi-utility operators. "
                   "Share regulated grid tariffs and waste management revenues.",
                   "Utilities"),
    "SRG/TRN"   : ("SRG",  "TRN",
                   "Snam + Terna: gas and electricity transmission monopolies. "
                   "Both 'bond proxies', move together on Italian long-rate changes.",
                   "Utilities"),
}

YF_SUFFIX = ".MI"
CAPITAL    = 10_000.0   # EUR per pair (5k per leg)
SPREAD_BPS = 4          # per leg (8 bps round trip)

# H1 winning params: window=60, entry_z=1.5, exit_z=0.5
# Scale for M15: 60 H1 bars = 240 M15 bars = 120 M30 bars
TIMEFRAMES = {
    "M15": {"interval": "15m",  "bars_per_h": 4,  "window": 240, "entry_z": 1.5, "exit_z": 0.5},
    "M30": {"interval": "30m",  "bars_per_h": 2,  "window": 120, "entry_z": 1.5, "exit_z": 0.5},
    "H1":  {"interval": "1h",   "bars_per_h": 1,  "window":  60, "entry_z": 1.5, "exit_z": 0.5},
}

# Italian market hours (CET = UTC+1, CEST = UTC+2 in summer)
# We'll work in bar-hour from open
SESSION_BUCKETS = {
    "Open (09-11)":     (9, 11),
    "Mid-AM (11-13)":   (11, 13),
    "Lunch (13-15)":    (13, 15),
    "Afternoon (15-17)": (15, 17),
    "Close (17-18)":    (17, 18),
}


# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────
def fetch_yf(sym: str, interval: str, days: int = 58) -> pd.Series | None:
    import yfinance as yf
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    try:
        raw = yf.download(
            sym + YF_SUFFIX,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False, auto_adjust=True, actions=False,
        )
    except Exception:
        return None
    if raw is None or len(raw) < 50:
        return None
    raw.index = pd.to_datetime(raw.index, utc=True)
    col = next((c for c in raw.columns if str(c).lower() == "close"), None)
    if col is None:
        col = raw.columns[3]
    return raw[col].rename(sym).dropna()


def load_all(interval: str, days: int = 58) -> dict[str, pd.Series]:
    syms = list({s for _, (a, b, *_) in PAIRS.items() for s in (a, b)})
    cache: dict[str, pd.Series] = {}
    for sym in syms:
        s = fetch_yf(sym, interval, days)
        if s is not None:
            cache[sym] = s
    return cache


# ─────────────────────────────────────────────────────────────────
# Z-score engine
# ─────────────────────────────────────────────────────────────────
def zscore_series(ratio: pd.Series, window: int) -> pd.DataFrame:
    df = pd.DataFrame({"ratio": ratio})
    df["ma"]    = df["ratio"].rolling(window).mean()
    df["std"]   = df["ratio"].rolling(window).std()
    df["z"]     = (df["ratio"] - df["ma"]) / df["std"]
    df["atr"]   = df["ratio"].diff().abs().rolling(window).mean()  # ratio ATR
    return df


# ─────────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────────
def backtest(ratio_df: pd.DataFrame, entry_z: float, exit_z: float,
             stop_z: float = 4.0) -> pd.DataFrame:
    trades = []
    pos = 0
    entry_z_val = entry_bar = entry_ratio = 0.0

    for i, row in ratio_df.iterrows():
        z, r = row["z"], row["ratio"]
        if pd.isna(z):
            continue
        if pos == 0:
            if z >= entry_z:
                pos, entry_z_val, entry_ratio, entry_bar = -1, z, r, i
            elif z <= -entry_z:
                pos, entry_z_val, entry_ratio, entry_bar = +1, z, r, i
        else:
            tp   = (pos == -1 and z <= exit_z) or (pos == +1 and z >= -exit_z)
            stop = abs(z) >= stop_z
            if tp or stop:
                pnl_pct = pos * (entry_ratio - r) / entry_ratio
                gross   = pnl_pct * CAPITAL
                cost    = SPREAD_BPS / 10_000 * CAPITAL * 2
                trades.append({
                    "ts_open":   entry_bar,
                    "ts_close":  i,
                    "pos":       pos,
                    "entry_z":   entry_z_val,
                    "exit_z":    z,
                    "entry_r":   entry_ratio,
                    "exit_r":    r,
                    "gross":     gross,
                    "net":       gross - cost,
                    "exit_type": "stop" if stop else "tp",
                    "hour_open": pd.Timestamp(entry_bar).hour if hasattr(entry_bar, 'hour') else 0,
                    "dow_open":  pd.Timestamp(entry_bar).dayofweek,
                })
                pos = 0

    return pd.DataFrame(trades)


def metrics(trades: pd.DataFrame) -> dict:
    if len(trades) == 0:
        return dict(n=0, pnl=0, wr=0, sharpe=0, pf=0, mdd=0, exp=0)
    p = trades["net"]
    w = p[p > 0]
    l = p[p < 0]
    eq = p.cumsum()
    pk = eq.cummax()
    mdd = float(((pk - eq) / pk.where(pk > 0, 1)).max() * 100)
    gw, gl = w.sum(), abs(l.sum())
    return dict(
        n=len(p), pnl=round(p.sum(), 1), wr=round(len(w)/len(p), 3),
        sharpe=round(float(p.mean()/p.std()*np.sqrt(len(p))) if p.std()>0 else 0, 2),
        pf=round(gw/gl if gl>0 else 99.0, 2),
        mdd=round(mdd, 1),
        exp=round(p.mean(), 1),
        n_stops=int((trades["exit_type"] == "stop").sum()),
    )


# ─────────────────────────────────────────────────────────────────
# Alpha slicing
# ─────────────────────────────────────────────────────────────────
def slice_by_hour(trades: pd.DataFrame) -> pd.DataFrame:
    if len(trades) == 0:
        return pd.DataFrame()
    rows = []
    for h in range(9, 18):
        sub = trades[trades["hour_open"] == h]
        if len(sub) == 0:
            continue
        m = metrics(sub)
        m["hour"] = h
        rows.append(m)
    return pd.DataFrame(rows).set_index("hour") if rows else pd.DataFrame()


def slice_by_dow(trades: pd.DataFrame) -> pd.DataFrame:
    if len(trades) == 0:
        return pd.DataFrame()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    rows = []
    for d in range(5):
        sub = trades[trades["dow_open"] == d]
        if len(sub) == 0:
            continue
        m = metrics(sub)
        m["day"] = days[d]
        rows.append(m)
    return pd.DataFrame(rows).set_index("day") if rows else pd.DataFrame()


def slice_by_vol_regime(trades: pd.DataFrame, ratio_df: pd.DataFrame) -> pd.DataFrame:
    """Split by whether ratio ATR was above/below median at entry."""
    if len(trades) == 0 or "atr" not in ratio_df.columns:
        return pd.DataFrame()
    med_atr = ratio_df["atr"].median()
    results = []
    for regime, cond in [("low_vol", lambda z: z < med_atr),
                          ("high_vol", lambda z: z >= med_atr)]:
        atr_at_entry = ratio_df["atr"].reindex(trades["ts_open"]).values
        mask = cond(pd.Series(atr_at_entry, index=trades.index))
        sub = trades[mask.values]
        if len(sub) == 0:
            continue
        m = metrics(sub)
        m["regime"] = regime
        results.append(m)
    return pd.DataFrame(results).set_index("regime") if results else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────
# Cointegration diagnostics
# ─────────────────────────────────────────────────────────────────
def coint_diagnostics(sa: pd.Series, sb: pd.Series) -> dict:
    merged = pd.DataFrame({"a": sa, "b": sb}).dropna()
    if len(merged) < 100:
        return {}
    _, p, _ = coint(merged["a"], merged["b"])
    ratio = merged["a"] / merged["b"]
    adf_stat, adf_p, *_ = adfuller(ratio.dropna(), maxlag=5)
    half_life = _half_life(ratio.dropna())
    return {
        "coint_p":   round(float(p), 4),
        "adf_p":     round(float(adf_p), 4),
        "half_life_bars": round(float(half_life), 1) if half_life else None,
        "corr":      round(float(merged["a"].corr(merged["b"])), 3),
        "n_bars":    len(merged),
    }


def _half_life(series: pd.Series) -> float | None:
    """Ornstein-Uhlenbeck half-life of mean reversion in bars."""
    try:
        s = series.dropna()
        lag = s.shift(1)
        delta = s - lag
        lag = lag.dropna()
        delta = delta.iloc[1:]
        beta = np.polyfit(lag, delta, 1)[0]
        if beta >= 0:
            return None
        return float(-np.log(2) / beta)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────
# Live screener
# ─────────────────────────────────────────────────────────────────
def build_screener(cache: dict[str, pd.Series], tf_cfg: dict) -> pd.DataFrame:
    window   = tf_cfg["window"]
    entry_z  = tf_cfg["entry_z"]
    rows = []

    for name, (sym_a, sym_b, rationale, sector) in PAIRS.items():
        if sym_a not in cache or sym_b not in cache:
            continue
        merged = pd.DataFrame({"a": cache[sym_a], "b": cache[sym_b]}).dropna()
        if len(merged) < window + 10:
            continue

        ratio_df = zscore_series(merged["a"] / merged["b"], window)
        last = ratio_df.dropna().iloc[-1]

        z     = float(last["z"])
        ratio = float(last["ratio"])
        ma    = float(last["ma"])
        std   = float(last["std"])

        # Distance to entry level in sigma units
        dist_to_entry = entry_z - abs(z)   # negative = already past entry

        # Direction of signal
        if z >= entry_z:
            signal = "SHORT ratio (short A, long B)"
            signal_code = "SHORT"
        elif z <= -entry_z:
            signal = "LONG ratio (long A, short B)"
            signal_code = "LONG"
        elif dist_to_entry < 0.3:
            signal = f"WATCH — {dist_to_entry:.2f}z to entry"
            signal_code = "WATCH"
        else:
            signal = "neutral"
            signal_code = "—"

        # Cointegration quick check on last window
        sub_a = merged["a"].iloc[-window:]
        sub_b = merged["b"].iloc[-window:]
        diag = coint_diagnostics(sub_a, sub_b)
        hl = diag.get("half_life_bars")

        rows.append({
            "Pair":       name,
            "Sector":     sector,
            "Z-score":    round(z, 2),
            "Ratio":      round(ratio, 4),
            "MA":         round(ma, 4),
            "to_entry":   round(dist_to_entry, 2),
            "Signal":     signal_code,
            "coint_p":    diag.get("coint_p", "—"),
            "half_life":  f"{hl:.0f}b" if hl else "—",
            "corr":       diag.get("corr", "—"),
            "Rationale":  rationale[:80] + "…",
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    # Sort: actionable first, then by abs(z-score) desc
    order = {"SHORT": 0, "LONG": 1, "WATCH": 2, "—": 3}
    df["_ord"] = df["Signal"].map(order)
    df = df.sort_values(["_ord", "Z-score"], key=lambda x: x.abs() if x.name == "Z-score" else x,
                        ascending=[True, False]).drop(columns=["_ord"])
    return df


def print_screener(df: pd.DataFrame, tf: str) -> None:
    print(f"\n{'='*90}")
    print(f"  FTSE MIB PAIRS SCREENER  |  {tf}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*90}")
    cols = ["Pair", "Sector", "Z-score", "to_entry", "Signal", "coint_p", "half_life", "corr"]
    print(df[cols].to_string(index=False))
    print()

    active = df[df["Signal"].isin(["SHORT", "LONG", "WATCH"])]
    if len(active) > 0:
        print("--- ACTION LIST ---")
        for _, r in active.iterrows():
            marker = ">>>" if r["Signal"] in ("SHORT", "LONG") else "  ~"
            print(f"{marker} {r['Pair']:12s}  Z={r['Z-score']:+.2f}  {r['Signal']:5s}  "
                  f"coint_p={r['coint_p']}  HL={r['half_life']}")
            print(f"    WHY: {r['Rationale']}")
        print()


# ─────────────────────────────────────────────────────────────────
# Full alpha hunt — run all TFs, slice results
# ─────────────────────────────────────────────────────────────────
def run_alpha_hunt(out: Path, tf_name: str = "M15") -> None:
    tf_cfg = TIMEFRAMES[tf_name]
    print(f"\n[alpha] {tf_name} | window={tf_cfg['window']} bars "
          f"(={tf_cfg['window']/tf_cfg['bars_per_h']:.0f}h) | "
          f"entry_z={tf_cfg['entry_z']} exit_z={tf_cfg['exit_z']}")

    cache = load_all(tf_cfg["interval"])
    loaded = list(cache.keys())
    print(f"  Loaded: {loaded}")

    results = {}
    all_trades_list = []

    for name, (sym_a, sym_b, rationale, sector) in PAIRS.items():
        if sym_a not in cache or sym_b not in cache:
            continue
        merged = pd.DataFrame({"a": cache[sym_a], "b": cache[sym_b]}).dropna()
        if len(merged) < tf_cfg["window"] + 50:
            continue

        ratio      = merged["a"] / merged["b"]
        ratio_df   = zscore_series(ratio, tf_cfg["window"])
        diag       = coint_diagnostics(merged["a"], merged["b"])
        trades     = backtest(ratio_df.dropna(), tf_cfg["entry_z"], tf_cfg["exit_z"])
        m          = metrics(trades)

        hour_df    = slice_by_hour(trades)
        dow_df     = slice_by_dow(trades)
        vol_df     = slice_by_vol_regime(trades, ratio_df)

        hl = diag.get("half_life_bars")
        print(f"\n  {name:12s}  coint_p={diag.get('coint_p','?'):<6}  "
              f"HL={f'{hl:.0f}b' if hl else '—':6s}  "
              f"n={m['n']:3d}  pnl={m['pnl']:7.0f}  "
              f"sharpe={m['sharpe']:5.2f}  pf={m['pf']:5.2f}  "
              f"wr={m['wr']:.0%}")

        # Best hour
        if len(hour_df) > 0 and "sharpe" in hour_df.columns:
            best_h = hour_df["sharpe"].idxmax()
            print(f"    Best hour: {best_h:02d}:00  "
                  f"sharpe={hour_df.loc[best_h,'sharpe']:.2f}  "
                  f"n={hour_df.loc[best_h,'n']}")
        if len(vol_df) > 0 and "pnl" in vol_df.columns:
            print(f"    Vol regime: {vol_df[['pnl','sharpe']].to_dict()}")

        if len(trades) > 0:
            trades["pair"]   = name
            trades["sector"] = sector
            all_trades_list.append(trades)

        results[name] = {
            "diag": diag, "metrics": m, "ratio_df": ratio_df,
            "trades": trades, "hour": hour_df, "dow": dow_df, "vol": vol_df,
        }

    if not results:
        print("  No pairs had enough data.")
        return

    # ── Global alpha slices ──────────────────────────────────────
    if all_trades_list:
        all_t = pd.concat(all_trades_list, ignore_index=True)
        all_t.to_csv(out / f"trades_{tf_name}.csv", index=False)

        print(f"\n{'='*60}")
        print(f"  GLOBAL HOUR BREAKDOWN ({tf_name}) — all pairs combined")
        print(f"{'='*60}")
        gh = slice_by_hour(all_t)
        if len(gh) > 0:
            print(gh[["n", "pnl", "sharpe", "wr", "pf"]].to_string())

        print(f"\n  WEEKDAY BREAKDOWN")
        gd = slice_by_dow(all_t)
        if len(gd) > 0:
            print(gd[["n", "pnl", "sharpe", "wr"]].to_string())

        print(f"\n  VOL REGIME BREAKDOWN")
        ratio_all = pd.concat(
            [pd.DataFrame({"a": cache[PAIRS[nm][0]], "b": cache[PAIRS[nm][1]]}).dropna().pipe(
                lambda d: zscore_series(d["a"]/d["b"], tf_cfg["window"]))
             for nm in results], ignore_index=True
        )
        gv = slice_by_vol_regime(all_t, ratio_all)
        if len(gv) > 0:
            print(gv[["n", "pnl", "sharpe", "wr"]].to_string())

    # ── Plots ────────────────────────────────────────────────────
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    _plot_alpha_summary(results, tf_name, plots)
    _plot_hour_heatmap(results, tf_name, plots)
    _plot_half_life_bar(results, tf_name, plots)

    return results


def _plot_alpha_summary(results: dict, tf_name: str, plots: Path) -> None:
    pairs = list(results.keys())
    sharpes = [results[p]["metrics"]["sharpe"] for p in pairs]
    coint_ps = [results[p]["diag"].get("coint_p", 1.0) for p in pairs]
    hls = [results[p]["diag"].get("half_life_bars") or 0 for p in pairs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sharpe
    colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in sharpes]
    axes[0].barh(pairs, sharpes, color=colors)
    axes[0].axvline(0, color="black", lw=0.7)
    axes[0].set_title(f"Sharpe ratio ({tf_name})", fontweight="bold")
    axes[0].set_xlabel("Sharpe")

    # Cointegration p-value
    ccolors = ["#2ecc71" if p < 0.05 else "#f39c12" if p < 0.10 else "#e74c3c"
               for p in coint_ps]
    axes[1].barh(pairs, coint_ps, color=ccolors)
    axes[1].axvline(0.05, color="green", lw=1, ls="--", label="p=0.05")
    axes[1].axvline(0.10, color="orange", lw=1, ls="--", label="p=0.10")
    axes[1].set_title("Cointegration p-value", fontweight="bold")
    axes[1].set_xlabel("p-value")
    axes[1].legend(fontsize=8)

    # Half-life in bars
    hl_h = [h / (4 if tf_name == "M15" else 2 if tf_name == "M30" else 1)
            for h in hls]
    axes[2].barh(pairs, hl_h, color="steelblue")
    axes[2].set_title("Mean-reversion half-life (hours)", fontweight="bold")
    axes[2].set_xlabel("Hours")
    axes[2].axvline(8, color="orange", lw=1, ls="--", label="1 day")
    axes[2].axvline(40, color="red", lw=1, ls="--", label="1 week")
    axes[2].legend(fontsize=8)

    plt.suptitle(f"FTSE MIB Pairs — Alpha Summary  [{tf_name}]",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(plots / f"alpha_summary_{tf_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_hour_heatmap(results: dict, tf_name: str, plots: Path) -> None:
    pairs = [p for p in results if len(results[p]["trades"]) > 5]
    if not pairs:
        return
    hours = list(range(9, 18))
    data = np.full((len(pairs), len(hours)), np.nan)
    for i, p in enumerate(pairs):
        h_df = results[p]["hour"]
        if len(h_df) == 0:
            continue
        for j, h in enumerate(hours):
            if h in h_df.index and "sharpe" in h_df.columns:
                data[i, j] = h_df.loc[h, "sharpe"]

    fig, ax = plt.subplots(figsize=(12, max(4, len(pairs) * 0.5 + 2)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-3, vmax=3)
    ax.set_xticks(range(len(hours)))
    ax.set_xticklabels([f"{h:02d}:00" for h in hours])
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs)
    ax.set_title(f"Sharpe by entry hour  [{tf_name}]", fontweight="bold")
    fig.colorbar(im, ax=ax, label="Sharpe")

    # Annotate cells with trade count
    for i, p in enumerate(pairs):
        h_df = results[p]["hour"]
        for j, h in enumerate(hours):
            if h in h_df.index and not np.isnan(data[i, j]):
                n = h_df.loc[h, "n"] if "n" in h_df.columns else ""
                ax.text(j, i, str(n), ha="center", va="center", fontsize=7, color="black")

    plt.tight_layout()
    fig.savefig(plots / f"hour_heatmap_{tf_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_half_life_bar(results: dict, tf_name: str, plots: Path) -> None:
    """Equity curve per pair on the same canvas."""
    pairs_with_trades = [(p, results[p]["trades"]) for p in results
                         if len(results[p]["trades"]) > 0]
    if not pairs_with_trades:
        return

    n = len(pairs_with_trades)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3 + 1))
    axes = np.array(axes).flatten()

    for idx, (name, t) in enumerate(pairs_with_trades):
        ax = axes[idx]
        eq = t["net"].cumsum().values
        ax.plot(eq, color="#2ecc71" if eq[-1] > 0 else "#e74c3c", lw=1.2)
        ax.fill_between(range(len(eq)), eq, 0,
                        where=(eq >= 0), alpha=0.2, color="#2ecc71")
        ax.fill_between(range(len(eq)), eq, 0,
                        where=(eq < 0), alpha=0.2, color="#e74c3c")
        ax.axhline(0, color="black", lw=0.5)
        m = metrics(t)
        diag = results[name]["diag"]
        ax.set_title(f"{name}  S={m['sharpe']:.2f}  PF={m['pf']:.2f}  "
                     f"cp={diag.get('coint_p','?')}", fontsize=8, fontweight="bold")
        ax.set_xlabel("trade #", fontsize=7)
        ax.set_ylabel("EUR", fontsize=7)
        ax.tick_params(labelsize=7)

    for idx in range(len(pairs_with_trades), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"Equity curves per pair  [{tf_name}]", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(plots / f"equity_curves_{tf_name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# HTML screener
# ─────────────────────────────────────────────────────────────────
def build_html_screener(df: pd.DataFrame, tf: str, results: dict | None,
                        out_path: Path) -> None:
    signal_color = {
        "SHORT": "#e74c3c", "LONG": "#2ecc71",
        "WATCH": "#f39c12", "—": "#95a5a6"
    }

    rows_html = ""
    for _, r in df.iterrows():
        sc = signal_color.get(r["Signal"], "#95a5a6")
        z  = float(r["Z-score"])
        bar_pct = min(abs(z) / 4 * 100, 100)
        bar_col = "#e74c3c" if z > 0 else "#2ecc71"

        # extra diagnostics from results dict
        extra = ""
        if results and r["Pair"] in results:
            m = results[r["Pair"]]["metrics"]
            d = results[r["Pair"]]["diag"]
            hl = d.get("half_life_bars")
            hl_h = f"{hl/4:.1f}h" if hl and tf == "M15" else (f"{hl/2:.1f}h" if hl and tf == "M30" else (f"{hl:.1f}h" if hl else "—"))
            extra = (f"<br><small style='color:#888'>backtest: n={m['n']} | "
                     f"sharpe={m['sharpe']:.2f} | pf={m['pf']:.2f} | "
                     f"wr={m['wr']:.0%} | half-life={hl_h}</small>")

        # Get rationale from PAIRS dict
        pair_key = r["Pair"]
        rationale = PAIRS.get(pair_key, ("", "", "", ""))[2] if pair_key in PAIRS else ""

        rows_html += f"""
        <tr>
          <td><strong>{r['Pair']}</strong></td>
          <td>{r['Sector']}</td>
          <td style="text-align:center">
            <div style="background:#eee;border-radius:4px;overflow:hidden;height:16px;width:120px;display:inline-block">
              <div style="background:{bar_col};width:{bar_pct:.0f}%;height:100%"></div>
            </div>
            <br><span style="font-weight:bold;font-size:1.1em">{z:+.2f}</span>
          </td>
          <td style="text-align:center">{r['to_entry']:+.2f} &sigma;</td>
          <td style="text-align:center">
            <span style="background:{sc};color:white;padding:3px 8px;border-radius:4px;font-weight:bold">
              {r['Signal']}
            </span>
          </td>
          <td>{r.get('coint_p','—')}</td>
          <td>{r.get('half_life','—')}</td>
          <td>{r.get('corr','—')}</td>
          <td style="font-size:0.85em;color:#444">{rationale}{extra}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="300">
  <title>FTSE MIB Pairs Screener</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background:#f5f6fa; margin:0; padding:20px }}
    h1   {{ color:#2c3e50; margin-bottom:4px }}
    .sub {{ color:#7f8c8d; margin-bottom:20px; font-size:0.9em }}
    table {{ border-collapse:collapse; width:100%; background:white;
             border-radius:8px; overflow:hidden;
             box-shadow:0 2px 8px rgba(0,0,0,0.08) }}
    th {{ background:#2c3e50; color:white; padding:10px 12px;
          text-align:left; font-size:0.9em }}
    td {{ padding:10px 12px; border-bottom:1px solid #ecf0f1; vertical-align:top }}
    tr:hover td {{ background:#f8f9fa }}
    .legend {{ margin-top:20px; font-size:0.85em; color:#666 }}
    .legend span {{ display:inline-block; padding:2px 8px; border-radius:4px;
                    color:white; margin-right:8px }}
  </style>
</head>
<body>
  <h1>FTSE MIB Pairs Screener</h1>
  <div class="sub">
    Timeframe: <strong>{tf}</strong> &nbsp;|&nbsp;
    Window: <strong>{TIMEFRAMES[tf]['window']} bars
    ({TIMEFRAMES[tf]['window']/TIMEFRAMES[tf]['bars_per_h']:.0f}h)</strong> &nbsp;|&nbsp;
    Entry Z: <strong>&plusmn;{TIMEFRAMES[tf]['entry_z']}</strong> &nbsp;|&nbsp;
    Generated: <strong>{datetime.now().strftime('%Y-%m-%d %H:%M')}</strong>
    &nbsp;(auto-refreshes every 5 min)
  </div>
  <table>
    <thead>
      <tr>
        <th>Pair</th><th>Sector</th><th>Z-score</th>
        <th>Distance to entry</th><th>Signal</th>
        <th>Coint p</th><th>Half-life</th><th>Corr</th>
        <th>Why it works</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  <div class="legend">
    <span style="background:#e74c3c">SHORT</span> Short ratio (short A, long B) — ratio too high<br>
    <span style="background:#2ecc71">LONG</span>  Long ratio (long A, short B) — ratio too low<br>
    <span style="background:#f39c12">WATCH</span> Within 0.3&sigma; of entry level<br>
    <em>Distance to entry</em>: how many &sigma; until the entry level is triggered
    (negative = already past it).<br>
    <em>Half-life</em>: Ornstein-Uhlenbeck estimated time for spread to revert 50%
    (in bars of the chosen timeframe).<br>
    <em>Coint p</em>: Engle-Granger cointegration p-value on the lookback window.
    Green &lt;0.05, amber &lt;0.10.
  </div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"\n  HTML screener -> {out_path}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main(screen_only: bool = False, tf_name: str = "M15") -> None:
    out = Path("data/pairs_alpha")
    out.mkdir(parents=True, exist_ok=True)

    tf_cfg = TIMEFRAMES[tf_name]

    if screen_only:
        print(f"\n[screener] Loading {tf_name} data...")
        cache = load_all(tf_cfg["interval"])
        df = build_screener(cache, tf_cfg)
        print_screener(df, tf_name)
        build_html_screener(df, tf_name, None, out / f"screener_{tf_name}.html")
        return

    # Full alpha hunt across all three timeframes
    print("\n" + "="*62)
    print("  FTSE MIB PAIRS — ALPHA HUNT")
    print("="*62)

    all_results = {}
    for tf in ["M15", "M30", "H1"]:
        results = run_alpha_hunt(out, tf)
        if results:
            all_results[tf] = results

    # Cross-timeframe summary table
    print("\n" + "="*70)
    print("  CROSS-TIMEFRAME SHARPE SUMMARY")
    print("="*70)
    rows = []
    for pair_name, (sym_a, sym_b, rationale, sector) in PAIRS.items():
        row = {"Pair": pair_name, "Sector": sector}
        for tf in ["M15", "M30", "H1"]:
            if tf in all_results and pair_name in all_results[tf]:
                m = all_results[tf][pair_name]["metrics"]
                d = all_results[tf][pair_name]["diag"]
                row[f"{tf}_sharpe"] = m["sharpe"]
                row[f"{tf}_pf"]     = m["pf"]
                row[f"{tf}_n"]      = m["n"]
                if "H1_coint_p" not in row:
                    row["coint_p"] = d.get("coint_p", "—")
                    row["HL_h1b"]  = d.get("half_life_bars")
            else:
                row[f"{tf}_sharpe"] = None
                row[f"{tf}_n"]      = 0
        rows.append(row)

    xt_df = pd.DataFrame(rows)
    if len(xt_df) > 0:
        xt_df = xt_df.sort_values("H1_sharpe", ascending=False)
        cols_show = ["Pair", "Sector", "coint_p",
                     "M15_sharpe", "M15_n",
                     "M30_sharpe", "M30_n",
                     "H1_sharpe",  "H1_n"]
        available_cols = [c for c in cols_show if c in xt_df.columns]
        print(xt_df[available_cols].to_string(index=False))
        xt_df.to_csv(out / "cross_tf_summary.csv", index=False)

    # Live screener on M15 (the trader's timeframe)
    print(f"\n[screener] Building live M15 screener...")
    cache_m15 = load_all(TIMEFRAMES["M15"]["interval"])
    screen_df = build_screener(cache_m15, TIMEFRAMES["M15"])
    print_screener(screen_df, "M15")
    results_m15 = all_results.get("M15")
    build_html_screener(screen_df, "M15", results_m15, out / "screener_M15.html")

    print(f"\n[done] All outputs in {out}/")
    print(f"       screener_M15.html  (open in browser, auto-refreshes)")
    print(f"       cross_tf_summary.csv")
    print(f"       plots/alpha_summary_*.png")
    print(f"       plots/hour_heatmap_*.png")
    print(f"       plots/equity_curves_*.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", action="store_true",
                        help="Run screener only (fast, no backtest)")
    parser.add_argument("--tf", default="M15", choices=list(TIMEFRAMES.keys()))
    args = parser.parse_args()

    project_root = Path(__file__).parents[3]
    os.chdir(project_root)
    main(screen_only=args.screen, tf_name=args.tf)
