"""
Backtest sweep for the v3 LRC strategy (LRC candle close + scored level + velocity).
No WaveTrend -- replaced with daily-bar velocity/acceleration gate.

Usage:
    python -m stocks.backtest.runner_v3
    python -m stocks.backtest.runner_v3 --skip-extend --workers 6

Output:
    stocks/data/backtest_results/v3_sweep_{ts}.csv
    stocks/data/backtest_results/v3_best_by_regime_{ts}.csv
"""
import sys, warnings, logging, argparse
from pathlib import Path
from itertools import product
from multiprocessing import Pool, cpu_count
from datetime import datetime

sys.path.insert(0, r'path/to/repo/hft')
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from stocks.backtest.config import BT_CACHE_DIR, RESULTS_DIR
from stocks.backtest.data.extender import load_bt_bars
from stocks.backtest.data.universe import build_universe
from stocks.backtest.signals.wt_lrc_signal import (
    _lrc_vectorized, _lrc_candle_close_vectorized, _atr_vectorized,
)
from stocks.backtest.signals.lrc_vel_signal import _velocity_arrays, find_signals_v3

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
US_EXCHANGES      = {"NYSE", "NAS"}
MIN_BARS          = 200
LRC_CANDLE_LEN    = 15
LRC_CANDLE_SMOOTH = 9
ATR_P             = 14
VEL_LOOKBACK      = 5    # bars for average-range computation

# ── Sweep grid (no wt_ob dimension) ───────────────────────────────────────────
SWEEP = {
    "lrc_length":    [80, 100, 120],
    "sigma_mult":    [1.0, 1.5, 2.0],
    "touch_pct":     [0.30, 0.50, 0.75],
    "sl_atr_mult":   [0.3, 0.5, 1.0],
    "tp_rr":         [3.0, 5.0, 7.0],
    "max_concurrent":[3, 5, 10],
}
# 729 combos. Fixed: vel_lookback=5, min_score=40, lookback=120, lrc_candle_len=15


def load_us_liquid() -> dict:
    """Load US liquid bars (top-25% dollar vol). Returns {sym: df_lowercase}."""
    univ = build_universe()
    us   = univ[univ["exchange"].isin(US_EXCHANGES)]

    bars = {}; dvols = {}
    for _, row in us.iterrows():
        sym = row["symbol"]
        df  = load_bt_bars(sym)
        if df.empty or len(df) < MIN_BARS:
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        dvol = (df["close"] * df["volume"]).replace(0, np.nan).dropna()
        if len(dvol) >= MIN_BARS:
            dvols[sym] = float(dvol.median())
            bars[sym]  = df

    if not dvols:
        return {}
    s      = pd.Series(dvols)
    thresh = float(s.quantile(0.75))
    liquid = set(s[s >= thresh].index)
    log.info(f"Universe: {len(liquid)} US liquid symbols")
    return {sym: bars[sym] for sym in liquid if sym in bars}


def precompute_symbol_v3(df: pd.DataFrame, lrc_lengths, sigma_mults) -> dict:
    """
    Compute all indicator arrays once per symbol for v3.
    Velocity arrays replace WaveTrend relative to v2.
    """
    c = df["close"].values; h = df["high"].values
    l = df["low"].values;   o = df["open"].values

    lrc_data = {}
    for ll in lrc_lengths:
        for sm in sigma_mults:
            mid, upper, lower, _ = _lrc_vectorized(c, ll, sm)
            lrc_data[(ll, sm)] = (mid, upper, lower)

    vel_short, vel_long = _velocity_arrays(h, l, c, o, VEL_LOOKBACK)

    return {
        "close":     c,
        "high":      h,
        "low":       l,
        "open":      o,
        "n":         len(c),
        "index":     df.index,
        "lrc_c":     _lrc_candle_close_vectorized(c, LRC_CANDLE_LEN, LRC_CANDLE_SMOOTH),
        "atr":       _atr_vectorized(h, l, c, ATR_P),
        "vel_short": vel_short,
        "vel_long":  vel_long,
        "lrc_data":  lrc_data,
    }


def compute_metrics(all_r: list, avg_bars: int) -> dict:
    if not all_r:
        return {k: 0 for k in ("n_trades", "win_rate", "avg_r", "total_r",
                                "profit_factor", "max_dd_r", "sharpe_r", "expectancy")}
    r      = np.array(all_r)
    wins   = r > 0; losses = ~wins
    win_rt = wins.mean()
    gross_w = r[wins].sum()  if wins.any()  else 0.0
    gross_l = abs(r[losses].sum()) if losses.any() else 1e-9
    equity  = np.cumsum(np.insert(r, 0, 0))
    peak    = np.maximum.accumulate(equity)
    max_dd  = (equity - peak).min()

    n_years = avg_bars / 252
    ann     = np.sqrt(len(r) / n_years) if n_years > 0 else 0
    sharpe  = (r.mean() / r.std() * ann) if r.std() > 0 else 0.0

    return {
        "n_trades":      len(r),
        "win_rate":      round(win_rt * 100, 1),
        "avg_r":         round(r.mean(), 3),
        "total_r":       round(r.sum(), 2),
        "profit_factor": round(gross_w / gross_l, 2),
        "max_dd_r":      round(max_dd, 2),
        "sharpe_r":      round(sharpe, 2),
        "expectancy":    round(r.mean(), 3),
    }


# ── Worker (multiprocessing) ───────────────────────────────────────────────────
_PRECOMP: dict = {}

def _init_worker(precomp):
    global _PRECOMP
    _PRECOMP = precomp


def _find_exit(pc, entry_bar, sl, tp, direction):
    h = pc["high"]; l = pc["low"]; n = pc["n"]
    for i in range(entry_bar, n):
        sl_hit = (l[i] <= sl) if direction == "LONG" else (h[i] >= sl)
        tp_hit = (h[i] >= tp) if direction == "LONG" else (l[i] <= tp)
        if sl_hit: return i, sl, "SL"
        if tp_hit: return i, tp, "TP"
    return n - 1, float(pc["close"][n - 1]), "END"


def _run_combo(params: dict) -> dict:
    global _PRECOMP
    ll  = params["lrc_length"]; sm = params["sigma_mult"]
    tp  = params["touch_pct"];  sl = params["sl_atr_mult"]
    rr  = params["tp_rr"];      mc = int(params["max_concurrent"])

    signals_all = {}
    for sym, pc in _PRECOMP.items():
        sigs = find_signals_v3(pc, ll, sm, tp, sl, rr)
        if sigs: signals_all[sym] = sigs

    candidates = []
    for sym, sigs in signals_all.items():
        pc = _PRECOMP[sym]; n = pc["n"]
        for sig in sigs:
            bi, direction, sl_raw, tp_rr_s, _, readiness = sig
            if bi + 1 >= n: continue
            eb   = bi + 1
            ep   = float(pc["open"][eb])
            risk = abs(ep - sl_raw)
            if risk < 1e-6: continue
            pos    = +1 if direction == "SHORT" else -1
            tp_px  = ep - pos * tp_rr_s * risk
            xb, xp, _ = _find_exit(pc, eb, sl_raw, tp_px, direction)
            r = ((xp - ep) / risk if direction == "LONG" else (ep - xp) / risk)
            candidates.append({
                "sym": sym, "readiness": readiness,
                "entry_date": pc["index"][eb], "exit_date": pc["index"][xb],
                "r": round(r, 3),
            })

    candidates.sort(key=lambda t: (t["entry_date"], -t["readiness"]))

    accepted = []; open_pos = []
    for t in candidates:
        ed = t["entry_date"]
        open_pos = [(xd, s) for xd, s in open_pos if xd > ed]
        if len(open_pos) >= mc: continue
        if t["sym"] in {s for _, s in open_pos}: continue
        accepted.append(t)
        open_pos.append((t["exit_date"], t["sym"]))

    all_r    = [t["r"] for t in accepted]
    avg_bars = sum(pc["n"] for pc in _PRECOMP.values()) // max(1, len(_PRECOMP))
    return {**params, **compute_metrics(all_r, avg_bars)}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extend", action="store_true",
                    help="Skip MT5 cache extension")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()

    t0 = datetime.now()

    if not args.skip_extend:
        log.info("Extending MT5 cache (pass --skip-extend if already done)...")
        try:
            from stocks.backtest.data.extender import extend_cache
            univ  = build_universe()
            syms  = univ[univ["exchange"].isin(US_EXCHANGES)]["symbol"].tolist()
            benchmarks = list({r["benchmark"] for _, r in univ.iterrows()
                               if r.get("benchmark")})
            extend_cache(syms + benchmarks)
        except Exception as e:
            log.warning(f"Cache extension failed: {e} -- continuing with existing cache")

    log.info("Loading US liquid bars...")
    bars = load_us_liquid()
    if not bars:
        log.error("No bars loaded. Run cache extension first.")
        return

    lrc_lengths = SWEEP["lrc_length"]; sigma_mults = SWEEP["sigma_mult"]
    log.info(f"Precomputing indicators for {len(bars)} symbols "
             f"({len(lrc_lengths)*len(sigma_mults)} LRC combos each)...")
    precomp = {}
    for sym, df in bars.items():
        try:
            precomp[sym] = precompute_symbol_v3(df, lrc_lengths, sigma_mults)
        except Exception as e:
            log.debug(f"  {sym}: precompute failed -- {e}")

    log.info(f"Precomputed {len(precomp)} symbols in {(datetime.now()-t0).seconds}s")

    keys   = list(SWEEP.keys())
    combos = [dict(zip(keys, c)) for c in product(*[SWEEP[k] for k in keys])]
    log.info(f"Sweep: {len(combos)} combos x {len(precomp)} symbols "
             f"using {args.workers} workers")

    with Pool(args.workers, initializer=_init_worker, initargs=(precomp,)) as pool:
        results = pool.map(_run_combo, combos)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M")
    df_res = pd.DataFrame(results).sort_values("sharpe_r", ascending=False)

    out = RESULTS_DIR / f"v3_sweep_{ts}.csv"
    df_res.to_csv(out, index=False)
    log.info(f"Full sweep -> {out}")

    regime_best = (df_res.groupby(["sigma_mult", "touch_pct"])
                         .first()
                         .reset_index())
    regime_path = RESULTS_DIR / f"v3_best_by_regime_{ts}.csv"
    regime_best.to_csv(regime_path, index=False)
    log.info(f"Best by regime -> {regime_path}")

    elapsed = (datetime.now() - t0).total_seconds()
    log.info(f"Total time: {elapsed:.0f}s")

    print("\n--- Top 20 combos by Sharpe (v3: LRC + velocity + level) ---")
    cols = ["lrc_length", "sigma_mult", "touch_pct", "sl_atr_mult", "tp_rr",
            "max_concurrent", "n_trades", "win_rate", "avg_r", "total_r",
            "profit_factor", "max_dd_r", "sharpe_r"]
    print(df_res[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
