"""
Backtest sweep for the v2 LRC strategy (WaveTrend + LRC candle close + scored level).

Usage:
    python -m stocks.backtest.runner_v2
    python -m stocks.backtest.runner_v2 --skip-extend --workers 6

Output:
    stocks/data/backtest_results/v2_sweep_{ts}.csv       — all 729 combos
    stocks/data/backtest_results/v2_best_by_regime_{ts}.csv
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
    _lrc_vectorized, _lrc_candle_close_vectorized,
    _wavetrend_vectorized, _atr_vectorized, find_signals,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
US_EXCHANGES      = {"NYSE", "NAS"}
MIN_BARS          = 200
LRC_CANDLE_LEN    = 15
LRC_CANDLE_SMOOTH = 9
WT_N1, WT_N2      = 10, 21
ATR_P             = 14

# ── Sweep grid ────────────────────────────────────────────────────────────────
SWEEP = {
    "lrc_length":    [80, 100, 120],
    "sigma_mult":    [1.0, 1.5, 2.0],
    "wt_ob":         [40.0, 53.0, 65.0],
    "touch_pct":     [0.30, 0.50, 0.75],
    "sl_atr_mult":   [0.3, 0.5, 1.0],
    "tp_rr":         [3.0, 5.0, 7.0],
    "max_concurrent":[3, 5, 10],
}
# Fixed: min_score=40, lookback=120, wt_n1=10, wt_n2=21

# ─────────────────────────────────────────────────────────────────────────────

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


def precompute_symbol(df: pd.DataFrame, lrc_lengths, sigma_mults) -> dict:
    """
    Compute all indicator arrays for one symbol once.
    Expensive indicators (LRC per combo, WT, ATR) are computed here and reused
    across all 729 sweep combos.
    """
    c = df["close"].values; h = df["high"].values; l = df["low"].values

    lrc_data = {}
    for ll in lrc_lengths:
        for sm in sigma_mults:
            mid, upper, lower, _ = _lrc_vectorized(c, ll, sm)
            lrc_data[(ll, sm)] = (mid, upper, lower)

    return {
        "close":    c,
        "high":     h,
        "low":      l,
        "open":     df["open"].values,
        "n":        len(c),
        "index":    df.index,
        "lrc_c":    _lrc_candle_close_vectorized(c, LRC_CANDLE_LEN, LRC_CANDLE_SMOOTH),
        "wt1":      _wavetrend_vectorized(c, h, l, WT_N1, WT_N2)[0],
        "atr":      _atr_vectorized(h, l, c, ATR_P),
        "lrc_data": lrc_data,
    }


def simulate_symbol(precomp: dict, signals: list) -> list:
    """
    Walk-forward simulation: signal at bar-close → enter at next bar-open.
    One position at a time per symbol (realistic for daily signals).
    Returns list of R multiples.
    """
    if not signals:
        return []

    o = precomp["open"]; h = precomp["high"]; l = precomp["low"]
    n = precomp["n"]

    sig_map = {s[0]: s for s in signals}   # bar_idx → (idx, dir, sl, tp_rr, close, rdns)

    trades     = []
    in_trade   = False
    entry = sl = tp = direction = None

    for i in range(1, n):
        if in_trade:
            hi = h[i]; lo = l[i]
            sl_hit = (lo <= sl) if direction == "LONG" else (hi >= sl)
            tp_hit = (hi >= tp) if direction == "LONG" else (lo <= tp)

            if sl_hit or tp_hit:
                exit_px = sl if (sl_hit and tp_hit) or sl_hit else tp
                result  = "SL" if sl_hit else "TP"
                risk    = abs(entry - sl)
                r_mult  = ((exit_px - entry) / risk if direction == "LONG"
                           else (entry - exit_px) / risk)
                trades.append(round(r_mult, 3))
                in_trade = False

        if not in_trade and (i - 1) in sig_map:
            _, direction_s, sl_raw, tp_rr_s, _, _ = sig_map[i - 1]
            entry_px = float(o[i])
            risk     = abs(entry_px - sl_raw)
            if risk < 1e-6:
                continue
            pos      = +1 if direction_s == "SHORT" else -1
            tp_px    = entry_px - pos * tp_rr_s * risk

            in_trade = True
            entry    = entry_px
            sl       = sl_raw
            tp       = tp_px
            direction = direction_s

    return trades


def compute_metrics(all_r: list, avg_bars: int) -> dict:
    if not all_r:
        return {k: 0 for k in ("n_trades","win_rate","avg_r","total_r",
                               "profit_factor","max_dd_r","sharpe_r","expectancy")}
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


# ── Worker (multiprocessing) ──────────────────────────────────────────────────
_PRECOMP: dict = {}

def _init_worker(precomp):
    global _PRECOMP
    _PRECOMP = precomp


def _find_exit(pc, entry_bar, entry_px, sl, tp, direction):
    h = pc["high"]; l = pc["low"]; n = pc["n"]
    for i in range(entry_bar, n):
        sl_hit = (l[i] <= sl) if direction == "LONG" else (h[i] >= sl)
        tp_hit = (h[i] >= tp) if direction == "LONG" else (l[i] <= tp)
        if sl_hit: return i, sl, "SL"
        if tp_hit: return i, tp, "TP"
    return n - 1, float(pc["close"][n - 1]), "END"


def _run_combo(params: dict) -> dict:
    global _PRECOMP
    ll   = params["lrc_length"];    sm   = params["sigma_mult"]
    wob  = params["wt_ob"];         tp   = params["touch_pct"]
    sl   = params["sl_atr_mult"];   rr   = params["tp_rr"]
    mc   = int(params["max_concurrent"])

    # Collect signals per symbol
    signals_all = {}
    for sym, pc in _PRECOMP.items():
        sigs = find_signals(pc, ll, sm, wob, tp, sl, rr)
        if sigs: signals_all[sym] = sigs

    # Build candidate trade list with pre-computed exits
    candidates = []
    for sym, sigs in signals_all.items():
        pc = _PRECOMP[sym]; n = pc["n"]
        for sig in sigs:
            bi, direction, sl_raw, tp_rr_s, _, readiness = sig
            if bi + 1 >= n: continue
            eb      = bi + 1
            ep      = float(pc["open"][eb])
            risk    = abs(ep - sl_raw)
            if risk < 1e-6: continue
            pos     = +1 if direction == "SHORT" else -1
            tp_px   = ep - pos * tp_rr_s * risk
            xb, xp, _ = _find_exit(pc, eb, ep, sl_raw, tp_px, direction)
            r = ((xp - ep) / risk if direction == "LONG" else (ep - xp) / risk)
            candidates.append({
                "sym": sym, "readiness": readiness,
                "entry_date": pc["index"][eb], "exit_date": pc["index"][xb],
                "r": round(r, 3),
            })

    candidates.sort(key=lambda t: (t["entry_date"], -t["readiness"]))

    # Apply concurrency cap
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extend", action="store_true",
                    help="Skip MT5 cache extension (bars already cached)")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()

    t0 = datetime.now()

    if not args.skip_extend:
        log.info("Extending MT5 cache (pass --skip-extend if already done)...")
        try:
            from stocks.backtest.data.extender import extend_cache
            from stocks.backtest.data.universe import build_universe
            univ  = build_universe()
            syms  = univ[univ["exchange"].isin(US_EXCHANGES)]["symbol"].tolist()
            benchmarks = list({r["benchmark"] for _, r in univ.iterrows()
                               if r.get("benchmark")})
            extend_cache(syms + benchmarks)
        except Exception as e:
            log.warning(f"Cache extension failed: {e} — continuing with existing cache")

    log.info("Loading US liquid bars...")
    bars = load_us_liquid()
    if not bars:
        log.error("No bars loaded. Run cache extension first.")
        return

    # Precompute indicators for all symbols once
    lrc_lengths = SWEEP["lrc_length"]; sigma_mults = SWEEP["sigma_mult"]
    log.info(f"Precomputing indicators for {len(bars)} symbols "
             f"({len(lrc_lengths)*len(sigma_mults)} LRC combos each)...")
    precomp = {}
    for sym, df in bars.items():
        try:
            precomp[sym] = precompute_symbol(df, lrc_lengths, sigma_mults)
        except Exception as e:
            log.debug(f"  {sym}: precompute failed — {e}")

    log.info(f"Precomputed {len(precomp)} symbols in "
             f"{(datetime.now()-t0).seconds}s")

    # Build sweep grid
    keys   = list(SWEEP.keys())
    combos = [dict(zip(keys, c)) for c in product(*[SWEEP[k] for k in keys])]
    log.info(f"Sweep: {len(combos)} combos × {len(precomp)} symbols "
             f"using {args.workers} workers")

    with Pool(args.workers, initializer=_init_worker, initargs=(precomp,)) as pool:
        results = pool.map(_run_combo, combos)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M")
    df_res  = pd.DataFrame(results).sort_values("sharpe_r", ascending=False)

    out = RESULTS_DIR / f"v2_sweep_{ts}.csv"
    df_res.to_csv(out, index=False)
    log.info(f"Full sweep → {out}")

    # Best per regime (sigma_mult × wt_ob)
    regime_best = (df_res.groupby(["sigma_mult", "wt_ob"])
                         .first()
                         .reset_index())
    regime_path = RESULTS_DIR / f"v2_best_by_regime_{ts}.csv"
    regime_best.to_csv(regime_path, index=False)
    log.info(f"Best by regime → {regime_path}")

    elapsed = (datetime.now() - t0).total_seconds()
    log.info(f"Total time: {elapsed:.0f}s")

    print("\n--- Top 20 combos by Sharpe ---")
    cols = ["lrc_length","sigma_mult","wt_ob","touch_pct","sl_atr_mult","tp_rr",
            "n_trades","win_rate","avg_r","total_r","profit_factor","max_dd_r","sharpe_r"]
    print(df_res[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
