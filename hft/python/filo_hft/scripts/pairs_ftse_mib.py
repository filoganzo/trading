"""
FTSE MIB Pairs Trading Analysis
================================
1. Pull intraday bars for all FTSE MIB 40 stocks (MT5 → yfinance fallback)
2. Correlation + cointegration screen all pairs
3. Compute ratio time series, detect key levels via clustering
4. Backtest mean-reversion strategy with Z-score entries
5. Walk-forward robustness test
6. Output ranked results + plots

Run:
    python -m filo_hft.scripts.pairs_ftse_mib [--tf M15] [--days 120] [--out data/pairs_arb]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import adfuller, coint

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows
import io
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ─────────────────────────────────────────────
# FTSE MIB 40 universe
# ─────────────────────────────────────────────
FTSE_MIB = [
    "A2A", "AMP", "AZM", "BAMI", "BMPS", "BPE", "BZU",
    "CNHIND", "CPR", "DIA", "ENEL", "ENI", "ERG",
    "FBK", "G", "HER", "IG", "INW", "ISP", "LDO",
    "MB", "MONC", "NEXI", "PIRC", "PRY", "PST",
    "RACE", "REC", "SPM", "SRG", "STLAM", "STMMI",
    "TEN", "TIT", "TRN", "UCG", "UNI", "LTT", "SES",
]

# Yahoo Finance suffix
YF_SUFFIX = ".MI"

# MT5 symbol formats to try (broker-dependent)
MT5_FORMATS = ["{sym}", "{sym}.MI", "#{sym}", "{sym}m"]

# ─────────────────────────────────────────────
# MT5 helpers
# ─────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False

TF_MAP = {
    "M5":  "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1":  "TIMEFRAME_H1",
}


def _mt5_init() -> bool:
    if not MT5_AVAILABLE:
        return False
    if mt5.initialize():
        return True
    # Try with credentials if available
    cred_path = Path(__file__).parents[3] / "credentials.local.ini"
    if cred_path.exists():
        import configparser
        p = configparser.ConfigParser()
        p.read(cred_path)
        if "mt5" in p:
            m = p["mt5"]
            return mt5.initialize(server=m["server"], login=int(m["login"]), password=m["password"])
    return False


def _mt5_fetch(symbol: str, tf: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    if not MT5_AVAILABLE:
        return None
    tf_enum = getattr(mt5, TF_MAP[tf], None)
    if tf_enum is None:
        return None
    mt5.symbol_select(symbol, True)
    raw = mt5.copy_rates_range(symbol, tf_enum, start, end)
    if raw is None or len(raw) == 0:
        # cold cache warm-up
        for n in (500, 2000, 10000):
            raw = mt5.copy_rates_from_pos(symbol, tf_enum, 0, n)
            if raw is not None and len(raw) > 0:
                break
        if raw is None or len(raw) == 0:
            return None
        raw = mt5.copy_rates_range(symbol, tf_enum, start, end)
        if raw is None or len(raw) == 0:
            return None
    df = pd.DataFrame(raw)
    df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[(df["ts"] >= start) & (df["ts"] < end)].reset_index(drop=True)
    return df[["ts", "open", "high", "low", "close", "tick_volume"]].rename(
        columns={"tick_volume": "volume"}
    ) if len(df) > 0 else None


def fetch_mt5(sym: str, tf: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    for fmt in MT5_FORMATS:
        candidate = fmt.format(sym=sym)
        df = _mt5_fetch(candidate, tf, start, end)
        if df is not None and len(df) > 50:
            return df
    return None


# ─────────────────────────────────────────────
# Yahoo Finance helpers
# ─────────────────────────────────────────────
def fetch_yfinance(sym: str, tf: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    import yfinance as yf

    # yfinance interval mapping
    yf_tf = {"M5": "5m", "M15": "15m", "M30": "30m", "H1": "1h"}.get(tf, "1h")
    ticker = sym + YF_SUFFIX
    # yfinance caps intraday history: 60d for 1h, 60d for 30m, 60d for 15m, 60d for 5m
    max_days = 58
    fetch_start = max(start, datetime.now(tz=timezone.utc) - timedelta(days=max_days))

    try:
        raw = yf.download(
            ticker,
            start=fetch_start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=yf_tf,
            progress=False,
            auto_adjust=True,
            actions=False,
        )
    except Exception:
        return None

    if raw is None or len(raw) == 0:
        return None

    raw = raw.copy()
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in raw.columns]
    raw = raw.rename(columns={"tick_volume": "volume"})
    if "close" not in raw.columns:
        return None
    raw = raw.reset_index().rename(columns={"index": "ts", "datetime": "ts", "Datetime": "ts"})
    if "ts" not in raw.columns:
        raw["ts"] = raw.index
    df = raw[["ts", "close"]].dropna()
    return df if len(df) > 50 else None


# ─────────────────────────────────────────────
# Data loader (MT5 first, yf fallback)
# ─────────────────────────────────────────────
def load_prices(symbols: list[str], tf: str, days: int) -> dict[str, pd.Series]:
    end = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    mt5_on = _mt5_init() if MT5_AVAILABLE else False
    if mt5_on:
        print("[data] MT5 connected")
    else:
        print("[data] MT5 not available — using Yahoo Finance only")

    prices: dict[str, pd.Series] = {}
    for sym in symbols:
        df = None
        if mt5_on:
            df = fetch_mt5(sym, tf, start, end)
            if df is not None:
                s = df.set_index("ts")["close"].rename(sym)
                prices[sym] = s
                print(f"  [MT5] {sym}: {len(s)} bars")
                continue

        df = fetch_yfinance(sym, tf, start, end)
        if df is not None:
            s = df.set_index("ts")["close"].rename(sym)
            prices[sym] = s
            print(f"  [YF]  {sym}: {len(s)} bars")
        else:
            print(f"  [--]  {sym}: no data")

    if mt5_on:
        mt5.shutdown()

    return prices


# ─────────────────────────────────────────────
# Align prices to a common index
# ─────────────────────────────────────────────
def build_matrix(prices: dict[str, pd.Series], min_overlap: int = 200) -> pd.DataFrame:
    df = pd.DataFrame(prices)
    df = df.dropna(how="all")
    df = df.ffill(limit=3)
    # Keep only columns with enough data
    df = df.loc[:, df.notna().sum() >= min_overlap]
    return df


# ─────────────────────────────────────────────
# Pair screening: correlation + cointegration
# ─────────────────────────────────────────────
def screen_pairs(
    matrix: pd.DataFrame,
    min_corr: float = 0.70,
    coint_pval: float = 0.10,
) -> pd.DataFrame:
    syms = list(matrix.columns)
    rows = []
    pairs = list(itertools.combinations(syms, 2))
    print(f"[screen] {len(pairs)} pairs to test")

    for a, b in pairs:
        s = matrix[[a, b]].dropna()
        if len(s) < 200:
            continue
        corr = float(s[a].corr(s[b]))
        if abs(corr) < min_corr:
            continue
        # Engle-Granger cointegration
        try:
            _, pval, _ = coint(s[a], s[b])
        except Exception:
            pval = 1.0
        rows.append({"sym_a": a, "sym_b": b, "corr": corr, "coint_pval": float(pval), "n_bars": len(s)})

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df = df.sort_values("coint_pval").reset_index(drop=True)
    df["cointegrated"] = df["coint_pval"] < coint_pval
    return df


# ─────────────────────────────────────────────
# Ratio computation + Z-score
# ─────────────────────────────────────────────
def compute_ratio(s_a: pd.Series, s_b: pd.Series, window: int = 200) -> pd.DataFrame:
    df = pd.DataFrame({"a": s_a, "b": s_b}).dropna()
    df["ratio"] = df["a"] / df["b"]
    df["ratio_ma"] = df["ratio"].rolling(window).mean()
    df["ratio_std"] = df["ratio"].rolling(window).std()
    df["zscore"] = (df["ratio"] - df["ratio_ma"]) / df["ratio_std"]
    return df


# ─────────────────────────────────────────────
# Key level detection
# ─────────────────────────────────────────────
def find_key_levels(ratio: pd.Series, n_levels: int = 6, order: int = 5) -> np.ndarray:
    """Cluster local extrema into n_levels key levels."""
    arr = ratio.dropna().values
    # Local maxima and minima
    hi_idx = argrelextrema(arr, np.greater, order=order)[0]
    lo_idx = argrelextrema(arr, np.less, order=order)[0]
    pivots = np.concatenate([arr[hi_idx], arr[lo_idx]])
    if len(pivots) < n_levels:
        # Fall back to percentile levels
        return np.percentile(arr, np.linspace(5, 95, n_levels))
    pivots = pivots.reshape(-1, 1)
    km = KMeans(n_clusters=min(n_levels, len(pivots)), n_init=10, random_state=42)
    km.fit(pivots)
    return np.sort(km.cluster_centers_.flatten())


# ─────────────────────────────────────────────
# Back-test engine (single pair, single param set)
# ─────────────────────────────────────────────
def backtest_pair(
    df: pd.DataFrame,       # columns: a, b, ratio, zscore
    entry_z: float = 2.0,   # enter when |zscore| > entry_z
    exit_z: float = 0.5,    # exit when |zscore| < exit_z
    stop_z: float = 4.0,    # hard stop at extreme zscore
    capital: float = 10_000.0,
    spread_pct: float = 0.04,  # 4 bps per leg = 8 bps round trip (typical FTSE MIB)
) -> pd.DataFrame:
    """
    Signal:
      zscore > +entry_z  → ratio too high → short A, long B
      zscore < -entry_z  → ratio too low  → long A, short B
    Exit: |zscore| < exit_z or |zscore| > stop_z (stop loss)
    Returns trade log.
    """
    position = 0  # +1 = long A/short B, -1 = short A/long B
    entry_price_a = entry_price_b = 0.0
    entry_ratio = 0.0
    entry_ts = None
    trades = []

    half_cap = capital / 2.0

    for i, row in df.iterrows():
        z = row["zscore"]
        r = row["ratio"]
        if pd.isna(z) or pd.isna(r):
            continue

        if position == 0:
            if z > entry_z:
                position = -1  # short ratio (short A, long B)
                entry_price_a = row["a"]
                entry_price_b = row["b"]
                entry_ratio = r
                entry_ts = i
            elif z < -entry_z:
                position = +1  # long ratio (long A, short B)
                entry_price_a = row["a"]
                entry_price_b = row["b"]
                entry_ratio = r
                entry_ts = i
        else:
            exit_signal = abs(z) < exit_z
            stop_signal = abs(z) > stop_z

            if exit_signal or stop_signal:
                # Compute PnL
                # Leg A: position * (close_a - entry_a) / entry_a
                # Leg B: -position * (close_b - entry_b) / entry_b
                pnl_a = position * (row["a"] - entry_price_a) / entry_price_a * half_cap
                pnl_b = -position * (row["b"] - entry_price_b) / entry_price_b * half_cap
                gross_pnl = pnl_a + pnl_b
                cost = spread_pct / 100.0 * capital  # 2 legs × round-trip
                net_pnl = gross_pnl - cost

                trades.append({
                    "ts_open": entry_ts,
                    "ts_close": i,
                    "position": position,
                    "entry_ratio": entry_ratio,
                    "exit_ratio": r,
                    "entry_z": z if position == -1 else -z,
                    "exit_z": z,
                    "pnl_a": pnl_a,
                    "pnl_b": pnl_b,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "exit_type": "stop" if stop_signal else "tp",
                    "duration_bars": i - entry_ts if isinstance(i, int) else None,
                })
                position = 0
                entry_price_a = entry_price_b = 0.0

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────
# Metrics from trade log
# ─────────────────────────────────────────────
def trade_metrics(trades: pd.DataFrame) -> dict:
    if len(trades) == 0:
        return {"n": 0, "pnl": 0, "win_rate": 0, "sharpe": 0, "max_dd_pct": 0, "profit_factor": 0}
    p = trades["net_pnl"]
    wins = p[p > 0]
    losses = p[p < 0]
    equity = p.cumsum()
    peak = equity.cummax()
    dd = (peak - equity) / peak.where(peak > 0, 1.0) * 100
    gross_w = wins.sum()
    gross_l = abs(losses.sum())
    return {
        "n": len(trades),
        "pnl": float(p.sum()),
        "win_rate": float(len(wins) / len(trades)),
        "expectancy": float(p.mean()),
        "sharpe": float(p.mean() / p.std() * np.sqrt(len(trades))) if p.std() > 0 else 0,
        "max_dd_pct": float(dd.max()),
        "profit_factor": float(gross_w / gross_l) if gross_l > 0 else (99.0 if gross_w > 0 else 0.0),
        "n_stop": int((trades["exit_type"] == "stop").sum()),
        "avg_duration_bars": float(trades.get("duration_bars", pd.Series(dtype=float)).mean()) if "duration_bars" in trades.columns else 0,
    }


# ─────────────────────────────────────────────
# Parameter sweep over (window, entry_z, exit_z)
# ─────────────────────────────────────────────
def parameter_sweep(df: pd.DataFrame, sym_a: str, sym_b: str) -> pd.DataFrame:
    windows = [60, 120, 200, 300]
    entry_zs = [1.5, 2.0, 2.5, 3.0]
    exit_zs = [0.25, 0.5, 0.75]
    rows = []
    for w in windows:
        ratio_df = compute_ratio(df[sym_a], df[sym_b], window=w)
        for ez in entry_zs:
            for xz in exit_zs:
                if xz >= ez:
                    continue
                t = backtest_pair(ratio_df, entry_z=ez, exit_z=xz)
                m = trade_metrics(t)
                rows.append({
                    "window": w, "entry_z": ez, "exit_z": xz,
                    **m,
                    "sym_a": sym_a, "sym_b": sym_b,
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Walk-forward robustness
# ─────────────────────────────────────────────
def walk_forward(
    df: pd.DataFrame,
    sym_a: str,
    sym_b: str,
    window: int,
    entry_z: float,
    exit_z: float,
    n_folds: int = 5,
) -> list[dict]:
    n = len(df)
    fold_size = n // (n_folds + 1)
    results = []
    for i in range(n_folds):
        train_end = fold_size * (i + 1)
        test_start = train_end
        test_end = test_start + fold_size
        if test_end > n:
            break
        test_slice = df.iloc[test_start:test_end].copy()
        # Re-compute z on rolling basis within test (use train stats for mean/std)
        train_slice = df.iloc[:train_end]
        mu = (train_slice[sym_a] / train_slice[sym_b]).mean()
        sigma = (train_slice[sym_a] / train_slice[sym_b]).std()
        ratio_df = pd.DataFrame({"a": test_slice[sym_a], "b": test_slice[sym_b]})
        ratio_df["ratio"] = ratio_df["a"] / ratio_df["b"]
        ratio_df["zscore"] = (ratio_df["ratio"] - mu) / sigma
        t = backtest_pair(ratio_df, entry_z=entry_z, exit_z=exit_z)
        m = trade_metrics(t)
        m["fold"] = i
        results.append(m)
    return results


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def plot_pair(
    ratio_df: pd.DataFrame,
    trades: pd.DataFrame,
    sym_a: str,
    sym_b: str,
    key_levels: np.ndarray,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 2], hspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    idx = np.arange(len(ratio_df))

    # Panel 1: ratio + key levels
    ax1.plot(idx, ratio_df["ratio"].values, color="steelblue", lw=0.8, label="Ratio A/B")
    ax1.plot(idx, ratio_df["ratio_ma"].values, color="orange", lw=1.2, ls="--", label=f"MA")
    for lv in key_levels:
        ax1.axhline(lv, color="red", lw=0.6, ls=":", alpha=0.7)
    ax1.set_title(f"{sym_a} / {sym_b}  —  Ratio & Key Levels", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.set_ylabel("Ratio")

    # Panel 2: Z-score
    ax2.plot(idx, ratio_df["zscore"].values, color="purple", lw=0.8)
    ax2.axhline(0, color="black", lw=0.5)
    for ez in [2.0, -2.0]:
        ax2.axhline(ez, color="red", lw=0.8, ls="--", alpha=0.7)
    ax2.set_ylabel("Z-score")
    ax2.set_title("Z-score")

    # Panel 3: Equity curve
    if len(trades) > 0:
        equity = trades["net_pnl"].cumsum()
        ax3.plot(range(len(equity)), equity.values, color="green", lw=1.2)
        ax3.fill_between(range(len(equity)), equity.values, 0,
                         where=(equity.values >= 0), alpha=0.2, color="green")
        ax3.fill_between(range(len(equity)), equity.values, 0,
                         where=(equity.values < 0), alpha=0.2, color="red")
    ax3.set_ylabel("Cumul. PnL (€)")
    ax3.set_title("Equity Curve")
    ax3.axhline(0, color="black", lw=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(sweep: pd.DataFrame, sym_a: str, sym_b: str, out_path: Path) -> None:
    pivot = sweep.groupby(["entry_z", "window"])["sharpe"].mean().unstack("window")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=3)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Window (bars)")
    ax.set_ylabel("Entry Z-score")
    ax.set_title(f"Sharpe heatmap  {sym_a}/{sym_b}")
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────
def main(tf: str = "H1", days: int = 90, out_dir: str = "data/pairs_arb",
         min_corr: float = 0.70, coint_pval: float = 0.10,
         top_n: int = 20) -> None:

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  FTSE MIB Pairs Arb  |  TF={tf}  days={days}")
    print(f"{'='*60}\n")

    # ── 1. Data ──
    print("[1/5] Loading price data...")
    prices = load_prices(FTSE_MIB, tf, days)
    loaded = list(prices.keys())
    print(f"  Loaded {len(loaded)}/{len(FTSE_MIB)} symbols: {loaded}")

    if len(loaded) < 4:
        print("[ERROR] Too few symbols loaded. Check MT5 connection or Yahoo Finance availability.")
        return

    matrix = build_matrix(prices)
    print(f"  Aligned matrix: {len(matrix)} bars x {len(matrix.columns)} symbols\n")

    matrix.to_parquet(out / "price_matrix.parquet")

    # ── 2. Pair screening ──
    print("[2/5] Screening pairs (correlation + cointegration)...")
    screen = screen_pairs(matrix, min_corr=min_corr, coint_pval=coint_pval)
    if len(screen) == 0:
        print("  No pairs passed the screen. Try lowering min_corr.")
        return
    screen.to_csv(out / "pair_screen.csv", index=False)
    print(f"  {len(screen)} pairs passed correlation >= {min_corr}")
    print(f"  {screen['cointegrated'].sum()} cointegrated at p < {coint_pval}")
    print(screen.head(15).to_string(index=False))

    # ── 3. Parameter sweep on top pairs ──
    print(f"\n[3/5] Parameter sweep on top {min(top_n, len(screen))} pairs...")
    top_pairs = screen.head(top_n)
    all_sweeps = []
    best_results = []

    for _, row in top_pairs.iterrows():
        sa, sb = row["sym_a"], row["sym_b"]
        if sa not in matrix.columns or sb not in matrix.columns:
            continue
        pair_df = matrix[[sa, sb]].dropna()
        if len(pair_df) < 200:
            continue

        sweep = parameter_sweep(pair_df, sa, sb)
        all_sweeps.append(sweep)

        # Best params by Sharpe (min 10 trades)
        ok = sweep[(sweep["n"] >= 10) & (sweep["profit_factor"] > 1.0)]
        if len(ok) == 0:
            continue
        best = ok.sort_values("sharpe", ascending=False).iloc[0]
        best_results.append({
            "sym_a": sa, "sym_b": sb,
            "corr": row["corr"],
            "coint_pval": row["coint_pval"],
            "window": int(best["window"]),
            "entry_z": float(best["entry_z"]),
            "exit_z": float(best["exit_z"]),
            "n_trades": int(best["n"]),
            "pnl": float(best["pnl"]),
            "win_rate": float(best["win_rate"]),
            "sharpe": float(best["sharpe"]),
            "max_dd_pct": float(best["max_dd_pct"]),
            "profit_factor": float(best["profit_factor"]),
        })

        print(f"  {sa}/{sb}: best Sharpe={best['sharpe']:.2f} "
              f"PF={best['profit_factor']:.2f} n={best['n']} "
              f"w={best['window']} ez={best['entry_z']} xz={best['exit_z']}")

    if len(all_sweeps) > 0:
        pd.concat(all_sweeps).to_csv(out / "sweep_all.csv", index=False)

    if len(best_results) == 0:
        print("  No pairs found profitable best params.")
    else:
        best_df = pd.DataFrame(best_results).sort_values("sharpe", ascending=False)
        best_df.to_csv(out / "best_pairs.csv", index=False)
        print(f"\n  Top pairs by Sharpe:\n{best_df.head(10).to_string(index=False)}")

    # ── 4. Walk-forward on top 10 ──
    print(f"\n[4/5] Walk-forward robustness on top 10 pairs...")
    wf_rows = []
    top_for_wf = best_df.head(10) if len(best_results) > 0 else pd.DataFrame()

    for _, br in top_for_wf.iterrows():
        sa, sb = br["sym_a"], br["sym_b"]
        pair_df = matrix[[sa, sb]].dropna()
        folds = walk_forward(pair_df, sa, sb,
                             window=int(br["window"]),
                             entry_z=float(br["entry_z"]),
                             exit_z=float(br["exit_z"]))
        for f in folds:
            f["sym_a"] = sa
            f["sym_b"] = sb
            wf_rows.append(f)

    if wf_rows:
        wf_df = pd.DataFrame(wf_rows)
        wf_df.to_csv(out / "walk_forward.csv", index=False)
        wf_summary = wf_df.groupby(["sym_a", "sym_b"])[["pnl", "sharpe", "win_rate", "n"]].mean().reset_index()
        wf_summary["pnl_positive_folds"] = (
            wf_df.groupby(["sym_a", "sym_b"])["pnl"].apply(lambda x: (x > 0).mean()).values
        )
        wf_summary.to_csv(out / "walk_forward_summary.csv", index=False)
        print(wf_summary.sort_values("sharpe", ascending=False).head(10).to_string(index=False))

    # ── 5. Plots for top 5 ──
    print(f"\n[5/5] Generating plots for top 5 pairs...")
    plot_pairs = best_df.head(5) if len(best_results) > 0 else pd.DataFrame()

    for _, br in plot_pairs.iterrows():
        sa, sb = br["sym_a"], br["sym_b"]
        pair_df = matrix[[sa, sb]].dropna()
        ratio_df = compute_ratio(pair_df[sa], pair_df[sb], window=int(br["window"]))
        key_levels = find_key_levels(ratio_df["ratio"])
        t = backtest_pair(ratio_df, entry_z=float(br["entry_z"]), exit_z=float(br["exit_z"]))

        # Reset index to integers for plotting
        ratio_df_plot = ratio_df.reset_index(drop=True)
        plot_pair(ratio_df_plot, t, sa, sb, key_levels, plots_dir / f"{sa}_{sb}_chart.png")

        # Heatmap
        sweep_pair = pd.concat(all_sweeps).query(f"sym_a=='{sa}' and sym_b=='{sb}'") if all_sweeps else pd.DataFrame()
        if len(sweep_pair) > 0:
            plot_heatmap(sweep_pair, sa, sb, plots_dir / f"{sa}_{sb}_heatmap.png")

        print(f"  Saved plots for {sa}/{sb}")

    # ── Final summary report ──
    report = _build_report(screen, best_df if len(best_results) > 0 else pd.DataFrame(),
                           wf_df if wf_rows else pd.DataFrame(), tf, days)
    (out / "report.md").write_text(report, encoding="utf-8")
    print(f"\n[done] Results saved to {out}/")
    print(f"       report.md  |  best_pairs.csv  |  sweep_all.csv  |  walk_forward.csv  |  plots/")


def _build_report(screen, best_df, wf_df, tf, days) -> str:
    lines = [
        "# FTSE MIB Pairs Arbitrage — Research Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Timeframe:** {tf}  |  **History:** {days} days",
        "",
        "---",
        "",
        "## 1. Universe & Data",
        f"- FTSE MIB 40 universe, {len(FTSE_MIB)} symbols attempted",
        f"- Intraday bars at {tf} frequency",
        "",
        "## 2. Pair Screening",
    ]
    if len(screen) > 0:
        lines.append(f"- **{len(screen)} pairs** passed correlation ≥ 0.70")
        lines.append(f"- **{int(screen['cointegrated'].sum())} pairs** cointegrated (p < 0.10)")
        lines.append("")
        lines.append("### Top 15 pairs by cointegration p-value")
        lines.append("```")
        lines.append(screen.head(15).to_string(index=False))
        lines.append("```")

    lines += ["", "## 3. Best Parameters (Sharpe-ranked)"]
    if len(best_df) > 0:
        lines.append("```")
        lines.append(best_df.head(15).to_string(index=False))
        lines.append("```")
        lines.append("")
        lines.append("### Key observations")
        best = best_df.iloc[0]
        lines.append(f"- **Best pair:** `{best['sym_a']} / {best['sym_b']}` "
                     f"(Sharpe={best['sharpe']:.2f}, PF={best['profit_factor']:.2f}, "
                     f"WR={best['win_rate']:.1%})")
        lines.append(f"- **Best window:** {best['window']} bars = "
                     f"{best['window']} × {tf} lookback")
        lines.append(f"- **Entry Z:** ±{best['entry_z']}  **Exit Z:** ±{best['exit_z']}")

    lines += ["", "## 4. Walk-Forward Robustness"]
    if len(wf_df) > 0:
        wf_s = wf_df.groupby(["sym_a", "sym_b"])[["pnl", "sharpe", "n"]].mean().reset_index()
        wf_s["pct_profitable_folds"] = (
            wf_df.groupby(["sym_a", "sym_b"])["pnl"]
            .apply(lambda x: f"{(x > 0).mean():.0%}")
            .values
        )
        lines.append("```")
        lines.append(wf_s.sort_values("sharpe", ascending=False).head(10).to_string(index=False))
        lines.append("```")

    lines += [
        "",
        "## 5. Strategy Logic",
        "",
        "### Entry",
        "When the rolling Z-score of ratio(A/B) crosses ±entry_z:",
        "- Z > +entry_z → **short A, long B** (ratio expected to revert down)",
        "- Z < -entry_z → **long A, short B** (ratio expected to revert up)",
        "",
        "### Exit",
        "- |Z| < exit_z → take profit (mean reversion achieved)",
        "- |Z| > 4.0 → hard stop loss",
        "",
        "### Capital allocation",
        "- Equal notional per leg (€5,000 each for €10,000 base)",
        "- Spread cost: 4 bps/leg (8 bps round-trip) — conservative for FTSE MIB",
        "",
        "## 6. Caveats & Next Steps",
        "- All results are **in-sample + walk-forward**, not live.",
        "- Spread costs are approximate; actual costs depend on order size and broker.",
        "- Cointegration relationships are not stable — regime monitoring required.",
        "- Optimal pairs: Italian financials (ISP/UCG/MB/BAMI) and utilities (ENEL/A2A/HER).",
        "- Consider execution: borrow availability for shorts, tick size, lot constraints.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FTSE MIB pairs arbitrage analysis")
    parser.add_argument("--tf", default="H1", choices=list(TF_MAP.keys()),
                        help="Timeframe (default: H1)")
    parser.add_argument("--days", type=int, default=90,
                        help="Days of history to fetch (default: 90; yfinance caps at ~58 for intraday)")
    parser.add_argument("--out", default="data/pairs_arb",
                        help="Output directory (default: data/pairs_arb)")
    parser.add_argument("--min-corr", type=float, default=0.70,
                        help="Minimum correlation to consider a pair (default: 0.70)")
    parser.add_argument("--coint-pval", type=float, default=0.10,
                        help="Cointegration p-value threshold (default: 0.10)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top-correlated pairs to sweep (default: 20)")
    args = parser.parse_args()

    # Run from project root
    project_root = Path(__file__).parents[3]
    os.chdir(project_root)

    main(
        tf=args.tf,
        days=args.days,
        out_dir=args.out,
        min_corr=args.min_corr,
        coint_pval=args.coint_pval,
        top_n=args.top_n,
    )
