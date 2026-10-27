"""
Comprehensive v4 backtest sweep.
Strategy: LRC candle close + velocity gate + scored key level (no WaveTrend).

Two-phase architecture (fast):
  Phase A — precompute raw signals for 27 (lrc, sigma, touch) groups once.
  Phase B — sweep execution params (sl, tp, breakeven, max_hold) x portfolio
             variants (max_concurrent, universe) cheaply.

Universe sweep: US-All, US-Top10, EU-All, + US sector filters.
Dollar PnL:  $1,000 capital x 20x leverage = $20,000 effective per trade.

Usage:
    python -m stocks.backtest.runner_v4 --skip-extend --workers 4
    python -m stocks.backtest.runner_v4 --skip-extend   # auto 40% CPU

Output:
    stocks/data/backtest_results/v4_sweep_{ts}.csv
    stocks/data/backtest_results/v4_top_combos_{ts}.csv   (top 200 by total_pnl_usd)
"""
import sys, warnings, logging, argparse, math
from pathlib import Path
from itertools import product
from multiprocessing import Pool, cpu_count
from datetime import datetime

sys.path.insert(0, r'path/to/repo/hft')
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from stocks.backtest.config import RESULTS_DIR
from stocks.backtest.data.extender import load_bt_bars
from stocks.backtest.data.universe import build_universe
from stocks.backtest.signals.wt_lrc_signal import (
    _lrc_vectorized, _lrc_candle_close_vectorized, _atr_vectorized, _scored_level_at,
)
from stocks.backtest.signals.lrc_vel_signal import _velocity_arrays

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
US_EXCH   = {"NYSE", "NAS"}
EU_EXCH   = {"LSE", "ETR", "PAR", "IT", "MAD", "AMS", "IE", "SWX"}
ALL_EXCH  = US_EXCH | EU_EXCH
MIN_BARS  = 200
LRC_CL    = 15
LRC_CS    = 9
ATR_P     = 14
VEL_LB    = 5

LEVERAGE  = 20
CAPITAL   = 1_000.0        # USD per trade
EFFECTIVE = CAPITAL * LEVERAGE   # $20,000 effective position

# ── Sweep grids ───────────────────────────────────────────────────────────────
SIGNAL_GRID = {
    "lrc_length": [80, 100, 120],    # 3  — channel length
    "sigma_mult": [1.0, 1.5, 2.0],   # 3  — channel width
    "touch_pct":  [0.30, 0.50, 0.75], # 3  — key-level proximity %
}   # 27 signal groups

EXEC_GRID = {
    "sl_atr_mult":  [0.3, 0.5, 1.0],                # 3  — stop = N x ATR from level
    "tp_rr":        [2.0, 3.0, 5.0, 7.0, 10.0],     # 5  — take-profit R:R ratio
    "breakeven_rr": [0.0, 1.0, 1.5],                 # 3  — move SL to entry at +N R (0=off)
    "max_hold":     [0, 30],                          # 2  — max bars held (0=off)
}   # 90 exec combos per signal group => 2430 total

MC_VALUES = [1, 3, 5, 10, 20, 0]   # max concurrent positions (0 = unlimited)

US_SECTORS = ["Technology", "Healthcare", "Financials",
              "Energy", "Consumer", "Industrial"]
MIN_SECTOR_SYMS = 5


# ── Universe building ─────────────────────────────────────────────────────────

def load_all_bars(univ: pd.DataFrame) -> tuple:
    """
    Load bars for all supported exchanges. Returns (bars_dict, dvols_dict, sym_meta).
    sym_meta: {sym: {"exchange": ..., "sector": ...}}
    """
    bars = {}; dvols = {}; sym_meta = {}
    for _, row in univ.iterrows():
        sym  = row["symbol"]
        exch = row["exchange"]
        if exch not in ALL_EXCH:
            continue
        df = load_bt_bars(sym)
        if df.empty or len(df) < MIN_BARS:
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        dvol = (df["close"] * df["volume"]).replace(0, np.nan).dropna()
        if len(dvol) < MIN_BARS:
            continue
        bars[sym]     = df
        dvols[sym]    = float(dvol.median())
        sym_meta[sym] = {"exchange": exch, "sector": row.get("sector", "Unknown")}

    return bars, dvols, sym_meta


def build_universe_sets(bars: dict, dvols: dict, sym_meta: dict) -> dict:
    """
    Returns {universe_name: frozenset(syms)}.
    Universe names: US-All, US-Top10, EU-All, US-Tech, US-Health, US-Fin,
                    US-Energy, US-Consumer, US-Industrial
    """
    def top_q(syms, q):
        sub = {s: dvols[s] for s in syms if s in dvols}
        if not sub:
            return frozenset()
        s = pd.Series(sub)
        return frozenset(s[s >= s.quantile(q)].index)

    us_all = {s for s, m in sym_meta.items() if m["exchange"] in US_EXCH}
    eu_all = {s for s, m in sym_meta.items() if m["exchange"] in EU_EXCH}

    uni = {
        "US-All":   top_q(us_all, 0.75),   # top 25% liquid US
        "US-Top10": top_q(us_all, 0.90),   # top 10% liquid US
        "EU-All":   top_q(eu_all, 0.75),   # top 25% liquid EU
    }

    us_top25 = uni["US-All"]
    for sec in US_SECTORS:
        sec_syms = {s for s in us_top25 if sym_meta.get(s, {}).get("sector") == sec}
        key = f"US-{sec[:3]}"           # US-Tec, US-Hea, US-Fin, US-Ene, US-Con, US-Ind
        if len(sec_syms) >= MIN_SECTOR_SYMS:
            uni[key] = frozenset(sec_syms)

    active = {k: v for k, v in uni.items() if len(v) >= MIN_SECTOR_SYMS}
    log.info("Universe sets: " +
             ", ".join(f"{k}={len(v)}" for k, v in active.items()))
    return active


# ── Indicator precompute ──────────────────────────────────────────────────────

def precompute_symbol(df: pd.DataFrame, lrc_lengths, sigma_mults) -> dict:
    c = df["close"].values; h = df["high"].values
    l = df["low"].values;   o = df["open"].values

    lrc_data = {}
    for ll in lrc_lengths:
        for sm in sigma_mults:
            mid, upper, lower, _ = _lrc_vectorized(c, ll, sm)
            lrc_data[(ll, sm)] = (mid, upper, lower)

    vel_s, vel_l = _velocity_arrays(h, l, c, o, VEL_LB)

    return {
        "close": c, "high": h, "low": l, "open": o,
        "n":     len(c), "index": df.index,
        "lrc_c": _lrc_candle_close_vectorized(c, LRC_CL, LRC_CS),
        "atr":   _atr_vectorized(h, l, c, ATR_P),
        "vel_short": vel_s, "vel_long": vel_l,
        "lrc_data":  lrc_data,
    }


# ── Raw signal finder (defers SL check to exec phase) ────────────────────────

def _find_raw_signals(precomp: dict, lrc_length: int, sigma_mult: float,
                      touch_pct: float, min_score: float = 40.0,
                      lookback: int = 120) -> list:
    """
    Returns raw signal candidates without SL/TP application.
    Each entry: (bar_idx, entry_bar, entry_px, direction, level_px, atr, readiness)
    """
    c  = precomp["close"]; h = precomp["high"]; l = precomp["low"]
    o  = precomp["open"]
    lrc_c  = precomp["lrc_c"]; atr = precomp["atr"]
    vs = precomp["vel_short"]; vl  = precomp["vel_long"]
    mid, upper, lower = precomp["lrc_data"][(lrc_length, sigma_mult)]
    n  = precomp["n"]

    touch_frac = touch_pct / 100.0
    min_i = max(lrc_length, LRC_CL + LRC_CS + 5, ATR_P + 5, VEL_LB + 5)
    if min_i >= n:
        return []

    idx     = np.arange(min_i, n)
    lrc_ci  = lrc_c[idx]; ui = upper[idx]; li = lower[idx]
    short_m = lrc_ci > ui; long_m = lrc_ci < li
    vel_m   = (short_m & vs[idx]) | (long_m & vl[idx])
    atr_i   = atr[idx]
    valid_m = (short_m | long_m) & vel_m \
              & ~np.isnan(lrc_ci) & ~np.isnan(ui) & ~np.isnan(atr_i)

    mid_i = mid[idx]
    bs    = np.where(sigma_mult > 0, (ui - mid_i) / sigma_mult, 0.0)
    dsig  = np.where(bs > 0, (lrc_ci - mid_i) / bs, 0.0)
    ext_s = np.clip((np.abs(dsig) / sigma_mult - 1.0) * 100.0, 0, 100)
    pre_r = ext_s * 0.5
    valid_m &= (pre_r + 50.0) >= min_score

    cands = idx[valid_m]
    if len(cands) == 0:
        return []

    sigs = []
    for i in cands:
        j   = int(i - min_i)
        dir_ = "SHORT" if lrc_c[i] > upper[i] else "LONG"
        lv, _, lv_score = _scored_level_at(h, l, c, int(i), dir_, touch_frac, lookback)
        if math.isnan(lv):
            continue
        readiness = float(pre_r[j]) + lv_score * 0.5
        if readiness < min_score:
            continue
        eb = int(i) + 1
        if eb >= n:
            continue
        ep = float(o[eb])
        atr_v = float(atr[i])
        sigs.append((int(i), eb, ep, dir_, lv, atr_v, readiness))

    return sigs


# ── Phase A worker ────────────────────────────────────────────────────────────
_PC_A: dict = {}

def _init_a(precomp):
    global _PC_A
    _PC_A = precomp


def _run_sig_group(sig_key: tuple) -> tuple:
    """Worker A: compute raw signals for one (lrc, sigma, touch) group, all symbols."""
    lrc, sigma, touch = sig_key
    out = {}
    for sym, pc in _PC_A.items():
        try:
            s = _find_raw_signals(pc, lrc, sigma, touch)
            if s:
                out[sym] = s
        except Exception:
            pass
    return sig_key, out


# ── Exit simulation ───────────────────────────────────────────────────────────

def _find_exit(h, l, c, eb, ep, sl_init, tp_px, direction,
               breakeven_rr, risk, max_hold, n):
    """Walk-forward exit with optional breakeven and max-hold."""
    sl = sl_init
    for i in range(eb, n):
        if max_hold and (i - eb) >= max_hold:
            return i, float(c[i]), "TIME"

        hi = h[i]; lo = l[i]

        # Update SL to breakeven (optimistic: HI triggers BE before LO checks SL)
        if breakeven_rr > 0:
            if direction == "LONG"  and hi >= ep + breakeven_rr * risk:
                sl = max(sl, ep)
            elif direction == "SHORT" and lo <= ep - breakeven_rr * risk:
                sl = min(sl, ep)

        sl_hit = (lo <= sl) if direction == "LONG" else (hi >= sl)
        tp_hit = (hi >= tp_px) if direction == "LONG" else (lo <= tp_px)

        if sl_hit:
            return i, sl, "SL"
        if tp_hit:
            return i, tp_px, "TP"

    return n - 1, float(c[n - 1]), "END"


# ── Phase B worker ────────────────────────────────────────────────────────────
_PC_B:  dict = {}
_RS_B:  dict = {}
_UNI_B: dict = {}

def _init_b(precomp, raw_signals, uni_syms):
    global _PC_B, _RS_B, _UNI_B
    _PC_B = precomp; _RS_B = raw_signals; _UNI_B = uni_syms


def _run_exec_combo(params: dict) -> list:
    """
    Worker B: for one exec combo, compute all exits once, then apply all
    (max_concurrent x universe) portfolio variants cheaply.
    """
    lrc   = params["lrc_length"];  sigma = params["sigma_mult"]
    touch = params["touch_pct"];   sl    = params["sl_atr_mult"]
    tp_r  = params["tp_rr"];       be    = params["breakeven_rr"]
    mh    = params["max_hold"]

    raw = _RS_B.get((lrc, sigma, touch), {})

    # Compute all trade candidates with actual exits
    candidates = []
    for sym, sigs in raw.items():
        pc = _PC_B.get(sym)
        if pc is None:
            continue
        h = pc["high"]; l = pc["low"]; c = pc["close"]
        n = pc["n"];     idx = pc["index"]

        for (bi, eb, ep, dir_, lv, atr_v, readiness) in sigs:
            pos  = +1 if dir_ == "SHORT" else -1
            sl_px = lv + pos * sl * atr_v
            if pos * (sl_px - ep) <= 0:     # SL on wrong side of entry
                continue
            risk = abs(ep - sl_px)
            if risk < 1e-6:
                continue
            tp_px = ep - pos * tp_r * risk

            xb, xp, _ = _find_exit(h, l, c, eb, ep, sl_px, tp_px, dir_,
                                    be, risk, mh, n)

            r = (xp - ep) / risk if dir_ == "LONG" else (ep - xp) / risk

            # Dollar PnL: $1k at 20x => $20k effective, full position
            shares     = EFFECTIVE / ep
            dollar_pnl = r * shares * sl * atr_v

            candidates.append({
                "sym":        sym,
                "readiness":  readiness,
                "entry_date": idx[eb],
                "exit_date":  idx[xb],
                "r":          round(r, 3),
                "pnl":        round(dollar_pnl, 2),
            })

    candidates.sort(key=lambda t: (t["entry_date"], -t["readiness"]))

    avg_bars = sum(pc["n"] for pc in _PC_B.values()) // max(1, len(_PC_B))

    results = []
    for mc in MC_VALUES:
        mc_cap = mc if mc > 0 else 999_999
        for uni_key, uni_syms in _UNI_B.items():
            filtered = [t for t in candidates if t["sym"] in uni_syms]
            accepted = []; open_pos = []
            for t in filtered:
                ed = t["entry_date"]
                open_pos = [(xd, s) for xd, s in open_pos if xd > ed]
                if len(open_pos) >= mc_cap:
                    continue
                if t["sym"] in {s for _, s in open_pos}:
                    continue
                accepted.append(t)
                open_pos.append((t["exit_date"], t["sym"]))

            all_r   = [t["r"]   for t in accepted]
            all_pnl = [t["pnl"] for t in accepted]
            results.append({
                **params,
                "max_concurrent": mc,
                "universe":       uni_key,
                "n_uni_syms":     len(uni_syms),
                **_metrics(all_r, all_pnl, avg_bars),
            })

    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def _metrics(all_r: list, all_pnl: list, avg_bars: int) -> dict:
    if not all_r:
        return dict(n_trades=0, win_rate=0, avg_r=0, total_r=0,
                    profit_factor=0, max_dd_r=0, sharpe_r=0,
                    total_pnl_usd=0, avg_pnl_usd=0, max_dd_usd=0, sharpe_usd=0)
    r   = np.array(all_r, dtype=float)
    pnl = np.array(all_pnl, dtype=float)
    wins  = r > 0; losses = ~wins
    gw = r[wins].sum()  if wins.any()  else 0.0
    gl = abs(r[losses].sum()) if losses.any() else 1e-9

    def _dd(arr):
        eq = np.cumsum(np.insert(arr, 0, 0.0))
        return float((eq - np.maximum.accumulate(eq)).min())

    n_years = avg_bars / 252
    ann     = np.sqrt(len(r) / n_years) if n_years > 0 else 0

    def _sharpe(arr):
        s = arr.std()
        return float(arr.mean() / s * ann) if s > 0 else 0.0

    return dict(
        n_trades      = len(r),
        win_rate      = round(float(wins.mean() * 100), 1),
        avg_r         = round(float(r.mean()), 3),
        total_r       = round(float(r.sum()), 2),
        profit_factor = round(gw / gl, 2),
        max_dd_r      = round(_dd(r), 2),
        sharpe_r      = round(_sharpe(r), 2),
        total_pnl_usd = round(float(pnl.sum()), 0),
        avg_pnl_usd   = round(float(pnl.mean()), 2),
        max_dd_usd    = round(_dd(pnl), 0),
        sharpe_usd    = round(_sharpe(pnl), 2),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extend", action="store_true")
    ap.add_argument("--workers", type=int,
                    default=max(1, int(cpu_count() * 0.4)))
    args = ap.parse_args()

    workers = args.workers
    t0 = datetime.now()
    log.info(f"v4 sweep starting | workers={workers} (40% CPU cap)")

    # ── Cache extension ──
    if not args.skip_extend:
        log.info("Extending MT5 cache...")
        try:
            from stocks.backtest.data.extender import extend_cache
            univ_ = build_universe()
            syms_ = univ_[univ_["exchange"].isin(ALL_EXCH)]["symbol"].tolist()
            benchmarks_ = list({r["benchmark"] for _, r in univ_.iterrows()
                                if r.get("benchmark")})
            extend_cache(syms_ + benchmarks_)
        except Exception as e:
            log.warning(f"Cache extension failed: {e}")

    # ── Load bars & universe ──
    log.info("Loading bars (US + EU exchanges)...")
    univ = build_universe()
    bars, dvols, sym_meta = load_all_bars(univ)
    log.info(f"Bars loaded: {len(bars)} symbols")

    uni_sets = build_universe_sets(bars, dvols, sym_meta)
    uni_keys = list(uni_sets.keys())

    # Use the union of all universe sets as the bar set to precompute
    all_syms_needed = set().union(*uni_sets.values())
    bars = {s: bars[s] for s in bars if s in all_syms_needed}
    log.info(f"Symbols after universe filter: {len(bars)}")

    # ── Indicator precompute ──
    lrc_lengths = SIGNAL_GRID["lrc_length"]
    sigma_mults = SIGNAL_GRID["sigma_mult"]
    log.info(f"Precomputing indicators for {len(bars)} symbols...")
    precomp = {}
    for sym, df in bars.items():
        try:
            precomp[sym] = precompute_symbol(df, lrc_lengths, sigma_mults)
        except Exception as e:
            log.debug(f"  {sym} precompute failed: {e}")
    log.info(f"Precomputed {len(precomp)} symbols in {(datetime.now()-t0).seconds}s")

    # ── Phase A: signal precompute ──
    sig_keys = list(product(lrc_lengths, sigma_mults, SIGNAL_GRID["touch_pct"]))
    log.info(f"Phase A: finding raw signals for {len(sig_keys)} signal groups "
             f"x {len(precomp)} symbols...")
    t_a = datetime.now()
    with Pool(workers, initializer=_init_a, initargs=(precomp,)) as pool:
        a_results = pool.map(_run_sig_group, sig_keys)
    raw_signals = {key: syms for key, syms in a_results}
    n_raw = sum(len(s) for d in raw_signals.values() for s in d.values())
    log.info(f"Phase A done in {(datetime.now()-t_a).seconds}s — {n_raw:,} raw signals")

    # ── Build exec combos ──
    exec_keys  = list(EXEC_GRID.keys())
    exec_combos = [dict(zip(["lrc_length","sigma_mult","touch_pct"] + exec_keys, c))
                   for c in product(
                       lrc_lengths, sigma_mults, SIGNAL_GRID["touch_pct"],
                       *[EXEC_GRID[k] for k in exec_keys]
                   )]
    n_portfolio = len(MC_VALUES) * len(uni_keys)
    log.info(f"Phase B: {len(exec_combos)} exec combos x {n_portfolio} portfolio "
             f"variants = {len(exec_combos)*n_portfolio:,} result rows")

    # ── Phase B: execution sweep ──
    t_b = datetime.now()
    with Pool(workers,
              initializer=_init_b,
              initargs=(precomp, raw_signals, uni_sets)) as pool:
        b_results = pool.map(_run_exec_combo, exec_combos)

    flat = [row for combo_rows in b_results for row in combo_rows]
    log.info(f"Phase B done in {(datetime.now()-t_b).seconds}s — {len(flat):,} rows")

    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M")
    df_res = pd.DataFrame(flat).sort_values("total_pnl_usd", ascending=False)

    full_path = RESULTS_DIR / f"v4_sweep_{ts}.csv"
    df_res.to_csv(full_path, index=False)
    log.info(f"Full sweep -> {full_path}  ({len(df_res):,} rows)")

    top_path = RESULTS_DIR / f"v4_top_combos_{ts}.csv"
    df_res.head(200).to_csv(top_path, index=False)
    log.info(f"Top 200    -> {top_path}")

    elapsed = (datetime.now() - t0).total_seconds()
    log.info(f"Total time: {elapsed:.0f}s")

    # ── Print top 20 by $ PnL ──
    cols = ["universe", "lrc_length", "sigma_mult", "touch_pct",
            "sl_atr_mult", "tp_rr", "breakeven_rr", "max_hold", "max_concurrent",
            "n_trades", "win_rate", "avg_r", "total_pnl_usd", "max_dd_usd", "sharpe_usd"]
    print("\n--- Top 20 combos by total dollar PnL ---")
    print(df_res[cols].head(20).to_string(index=False))

    # ── Best by universe ──
    print("\n--- Best combo per universe (by total_pnl_usd) ---")
    best_uni = df_res.groupby("universe").first().reset_index()
    print(best_uni[["universe","n_uni_syms","lrc_length","sigma_mult",
                    "touch_pct","sl_atr_mult","tp_rr","breakeven_rr","max_hold",
                    "max_concurrent","n_trades","win_rate","total_pnl_usd",
                    "max_dd_usd","sharpe_usd"]].to_string(index=False))


if __name__ == "__main__":
    main()
