"""
V11 backtest sweep: three new signal components on top of v8 validated region.

What's new vs v8:
  - Multi-timeframe entry (entry_tf): M15 / M30 / H1 — resampled from M15 data.
    M15 = existing behaviour; M30 = resample to 30-min; H1 = resample to 60-min.
    Same strict pin-bar logic applied to whichever TF is selected.
  - Relative-performance filter (rel_perf_thresh): require the stock to have
    underperformed SPX over the prior 5 bars (LONG) or outperformed (SHORT),
    capturing the "price fell hard relative to own history AND the index" edge.
  - ATR contraction filter (atr_contract_ratio): require current ATR <= ratio *
    peak ATR over previous 10 bars, capturing the "stalls / compresses before
    reversal" edge.
  - sig_key is now an 8-tuple:
      (lrc_length, sigma_mult, htf_tf_frac, entry_touch_frac,
       min_htf_score, entry_tf_str, rel_perf_thresh, atr_contract_ratio)
  - Phase C symbol discovery: outputs v11_symbol_discovery and v11_live_universe.
  - swing_wing fixed at 2, kept in grid for compatibility.
  - Single-threaded throughout (no multiprocessing — Windows pickling / memory).
  - m15_touch_pct parameter renamed "entry_touch_pct" in grid for clarity,
    but stored as m15_touch_pct for sweep CSV compat.

Sweep size:
  Signal groups: 2 sigma × 3 entry_touch × 3 entry_tf × 2 atr × 3 rel = 108
  Wait — lrc=1, htf_touch=1, min_score=1, swing_wing=1 are fixed; active dims:
    sigma_mult(2) × m15_touch_pct(3) × entry_tf(3) × rel_perf_thresh(3) ×
    atr_contract_ratio(2) = 108
  Exec combos:  108 × 4 (tp_rr) = 432
  Portfolio:    3 MC × ~8 universes = 24
  Total rows ≈ 10,368   (well within memory budget)

Usage:
    python -m stocks.backtest.runner_v11 --skip-extend

Output (RESULTS_DIR):
    v11_sweep_{ts}.csv
    v11_top_{ts}.csv
    v11_symbol_stats_{ts}.csv
    v11_trade_log_{ts}.csv
    v11_monthly_{ts}.csv
    v11_symbol_discovery_{ts}.csv
    v11_live_universe_{ts}.csv
"""
import sys, warnings, logging, argparse
from itertools import product
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, r'path/to/repo/hft')
warnings.filterwarnings("ignore")

from stocks.backtest.config import (
    RESULTS_DIR, BT_CACHE_DIR, BT_CACHE_M15_DIR, BT_CACHE_W1_DIR,
)
from stocks.backtest.data.extender import load_bt_bars, load_bt_bars_m15, load_bt_bars_w1
from stocks.backtest.data.universe import build_universe
from stocks.backtest.signals.wt_lrc_signal import (
    _lrc_vectorized, _lrc_candle_close_vectorized, _atr_vectorized,
)
from stocks.backtest.signals.htf_signal import (
    precompute_weekly_levels, find_htf_level, find_m15_entry,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
US_EXCH  = {"NYSE", "NAS"}
EU_EXCH  = {"LSE", "ETR", "PAR", "IT", "MAD", "AMS", "IE", "SWX"}
ALL_EXCH = US_EXCH | EU_EXCH
MIN_BARS = 200
LRC_CL   = 15; LRC_CS = 9; ATR_P = 14

LEVERAGE  = 20
CAPITAL   = 1_000.0
EFFECTIVE = CAPITAL * LEVERAGE   # $20,000

# ── Sweep grids ────────────────────────────────────────────────────────────────
SIGNAL_GRID = {
    "lrc_length":         [80],
    "sigma_mult":         [1.0, 1.5],
    "htf_touch_pct":      [0.5],
    "m15_touch_pct":      [0.3, 0.5, 1.0],   # "entry_touch_pct"; applies to whichever TF
    "min_htf_score":      [60],
    "swing_wing":         [2],                 # fixed at 2, kept for compat
    "entry_tf":           ["M15", "M30", "H1"],
    "rel_perf_thresh":    [0.0, -0.02, -0.05],
    "atr_contract_ratio": [1.0, 0.85],
}

EXEC_GRID = {
    "sl_atr_mult":  [1.0],
    "tp_rr":        [1.0, 1.5, 2.0, 3.0],
    "breakeven_rr": [0.0],
}

MC_VALUES = [0, 5, 10]

# Phase C live-universe filter thresholds
DISCOVERY_MIN_TRADES    = 5
DISCOVERY_MIN_WIN_RATE  = 58.0
DISCOVERY_MIN_SHARPE    = 0.8


# ── SPX / US500 relative-performance map ──────────────────────────────────────

def load_spx_rets() -> dict:
    """
    Load US500 daily closes and return {date: 5-bar pct-change} map.
    date keys are pd.Timestamp normalised to midnight (no time component).
    Falls back gracefully if the file is missing.
    """
    candidates = [
        BT_CACHE_DIR / "US500.parquet",
        BT_CACHE_DIR / "US500.NAS.parquet",
        BT_CACHE_DIR / "US500.NYSE.parquet",
    ]
    for path in candidates:
        if path.exists():
            try:
                df = pd.read_parquet(path)
                df.columns = [c.lower() for c in df.columns]
                close = df["close"]
                ret5  = close.pct_change(5)
                out   = {}
                for d, r in ret5.items():
                    if pd.notna(r):
                        out[pd.Timestamp(d).normalize()] = float(r)
                log.info(f"SPX rets loaded from {path.name}: {len(out):,} dates")
                return out
            except Exception as e:
                log.warning(f"Could not load {path}: {e}")
    log.warning("US500 data not found — rel_perf filter will be inactive for all bars")
    return {}


# ── Universe ───────────────────────────────────────────────────────────────────

def load_all_bars(univ: pd.DataFrame) -> tuple:
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
    Build named universe sets — same logic as v10:
      US-Broad  = top 50% dvol of all US symbols
      US-All    = top 75% dvol of US
      US-Top10  = top 90% dvol of US
      EU-All    = top 75% dvol of EU
      Sector sets: Technology, Healthcare, Financials, Energy (if >= 5 symbols)
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
        "US-Broad": top_q(us_all, 0.50),
        "US-All":   top_q(us_all, 0.75),
        "US-Top10": top_q(us_all, 0.90),
        "EU-All":   top_q(eu_all, 0.75),
    }

    us_broad = uni["US-Broad"]
    for sec in ["Technology", "Healthcare", "Financials", "Energy"]:
        sec_syms = {s for s in us_broad if sym_meta.get(s, {}).get("sector") == sec}
        key = f"US-{sec[:3]}"
        if len(sec_syms) >= 5:
            uni[key] = frozenset(sec_syms)

    active = {k: v for k, v in uni.items() if len(v) >= 5}
    log.info("Universe sets: " + ", ".join(f"{k}={len(v)}" for k, v in active.items()))
    return active


# ── Precompute ─────────────────────────────────────────────────────────────────

def _resample_m15_to_bars(m15_df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample an M15 DataFrame to a coarser timeframe.
    rule = '30min' for M30, '1h' for H1.
    Returns a DataFrame with columns: open, high, low, close.
    """
    agg = m15_df.resample(rule, closed='left', label='left').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    ).dropna()
    return agg


def precompute_symbol_v11(df: pd.DataFrame, m15_df: pd.DataFrame,
                          w1_df: pd.DataFrame,
                          lrc_lengths: list, sigma_mults: list,
                          htf_touch_pcts: list, swing_wings: list) -> dict:
    """
    Precompute all indicator data for one symbol.

    Returns dict with keys:
      close, high, low, open, n, index,
      lrc_c, atr, lrc_data,
      w_levels, m15_by_date, m30_by_date, h1_by_date,
      ret5_map    — {date: 5-bar stock return} for rel-perf filter
    """
    c = df["close"].values; h = df["high"].values
    l = df["low"].values;   o = df["open"].values
    idx = df.index

    # LRC indicators
    lrc_data = {}
    for ll in lrc_lengths:
        for sm in sigma_mults:
            mid, upper, lower, _ = _lrc_vectorized(c, ll, sm)
            lrc_data[(ll, sm)] = (mid, upper, lower)

    # Weekly levels keyed by (swing_wing, touch_frac)
    touch_fracs = [p / 100.0 for p in htf_touch_pcts]
    w_levels = {}
    if w1_df is not None and not w1_df.empty and len(w1_df) >= 4:
        w1_cols = {col.lower(): col for col in w1_df.columns}
        w_h = w1_df[w1_cols.get("high", "High")].values
        w_l = w1_df[w1_cols.get("low",  "Low")].values
        for sw in swing_wings:
            lvls = precompute_weekly_levels(w_h, w_l, touch_fracs, swing_wing=sw)
            for tf, lv_list in lvls.items():
                w_levels[(sw, tf)] = lv_list
    else:
        for sw in swing_wings:
            for tf in touch_fracs:
                w_levels[(sw, tf)] = []

    # M15 / M30 / H1 bars grouped by date
    m15_by_date = {}
    m30_by_date = {}
    h1_by_date  = {}

    if m15_df is not None and not m15_df.empty:
        m15 = m15_df.copy()
        m15.columns = [cc.lower() for cc in m15.columns]

        # M15 — group directly
        m15["_date"] = m15.index.normalize()
        for d, grp in m15.groupby("_date"):
            m15_by_date[d] = list(zip(
                grp["high"].values, grp["low"].values,
                grp["close"].values, grp["open"].values))

        # M30 — resample first, then group by date
        m30 = _resample_m15_to_bars(m15.drop(columns=["_date"]), '30min')
        m30["_date"] = m30.index.normalize()
        for d, grp in m30.groupby("_date"):
            m30_by_date[d] = list(zip(
                grp["high"].values, grp["low"].values,
                grp["close"].values, grp["open"].values))

        # H1 — resample first, then group by date
        h1 = _resample_m15_to_bars(m15.drop(columns=["_date"]), '1h')
        h1["_date"] = h1.index.normalize()
        for d, grp in h1.groupby("_date"):
            h1_by_date[d] = list(zip(
                grp["high"].values, grp["low"].values,
                grp["close"].values, grp["open"].values))

    # 5-bar stock return map for rel-perf filter
    ret5_map = {}
    for i in range(5, len(c)):
        if c[i - 5] != 0:
            date_key = pd.Timestamp(idx[i]).normalize()
            ret5_map[date_key] = float((c[i] - c[i - 5]) / c[i - 5])

    return {
        "close": c, "high": h, "low": l, "open": o,
        "n": len(c), "index": idx,
        "lrc_c":      _lrc_candle_close_vectorized(c, LRC_CL, LRC_CS),
        "atr":        _atr_vectorized(h, l, c, ATR_P),
        "lrc_data":   lrc_data,
        "w_levels":   w_levels,
        "m15_by_date": m15_by_date,
        "m30_by_date": m30_by_date,
        "h1_by_date":  h1_by_date,
        "ret5_map":    ret5_map,
    }


# ── Phase A: signal finding ────────────────────────────────────────────────────

def _find_signals_v11(pc: dict,
                      lrc_length: int,
                      sigma_mult: float,
                      htf_tf: float,
                      entry_touch_tf: float,
                      min_htf_score: float,
                      swing_wing: int,
                      entry_tf_str: str,
                      rel_perf_thresh: float,
                      atr_contract_ratio: float,
                      spx_ret5_map: dict) -> list:
    """
    LRC extension → HTF swing level → entry-TF pin-bar.
    New filters: relative performance vs SPX, ATR contraction.

    Returns list of:
        (signal_bar_i, entry_bar_i, entry_px, direction, level_px, atr_v, htf_score)
    """
    n         = pc["n"]
    lrc_c     = pc["lrc_c"]
    atr       = pc["atr"]
    h         = pc["high"]; l = pc["low"]; c = pc["close"]
    idx       = pc["index"]
    w_levels  = pc["w_levels"].get((swing_wing, htf_tf), [])
    ret5_map  = pc["ret5_map"]

    # Select entry-TF bar dict
    if entry_tf_str == "M30":
        m15_by_dt = pc["m30_by_date"]
    elif entry_tf_str == "H1":
        m15_by_dt = pc["h1_by_date"]
    else:
        m15_by_dt = pc["m15_by_date"]

    if not w_levels:
        return []

    mid, upper, lower = pc["lrc_data"][(lrc_length, sigma_mult)]
    min_i = max(lrc_length, LRC_CL + LRC_CS + 1, ATR_P + 1)
    if min_i >= n - 1:
        return []

    sigs = []
    for i in range(min_i, n - 1):
        ci = c[i]
        if np.isnan(ci) or np.isnan(upper[i]) or np.isnan(atr[i]):
            continue

        if ci > upper[i]:
            direction = "SHORT"
        elif ci < lower[i]:
            direction = "LONG"
        else:
            continue

        # ── ATR contraction filter ───────────────────────────────────────────
        if atr_contract_ratio < 1.0:
            if i >= 1:
                atr_peak = float(np.max(atr[max(0, i - 10):i]))
                if atr_peak > 0:
                    if atr[i] / atr_peak > atr_contract_ratio:
                        continue
                # if atr_peak <= 0: skip filter (don't drop signal)
            # if i < 1: skip filter (don't drop signal)

        # ── Relative performance filter ──────────────────────────────────────
        if rel_perf_thresh != 0.0 and i >= 5:
            date_key = pd.Timestamp(idx[i]).normalize()
            stock_ret5 = ret5_map.get(date_key)
            spx_ret5   = spx_ret5_map.get(date_key)
            # Only apply filter if both values are available
            if stock_ret5 is not None and spx_ret5 is not None:
                rel_ret5 = stock_ret5 - spx_ret5
                if direction == "LONG":
                    if rel_ret5 > rel_perf_thresh:   # rel_perf_thresh is negative
                        continue
                else:  # SHORT
                    if rel_ret5 < -rel_perf_thresh:  # require rel_ret5 >= -rel_perf_thresh
                        continue

        # ── HTF level proximity ──────────────────────────────────────────────
        price = float(c[i])
        lv, score = find_htf_level(w_levels, price, htf_tf, min_htf_score)
        if np.isnan(lv):
            continue

        # ── Entry-TF pin-bar ─────────────────────────────────────────────────
        eb = i + 1
        day_key  = pd.Timestamp(idx[eb]).normalize()
        tf_bars  = m15_by_dt.get(day_key)
        if not tf_bars:
            continue

        _, entry_px = find_m15_entry(tf_bars, direction, lv, entry_touch_tf)
        if entry_px is None:
            continue

        sigs.append((i, eb, entry_px, direction, lv, float(atr[i]), score))

    return sigs


# ── Exit simulation ────────────────────────────────────────────────────────────

def _find_exit(h, l, c, eb, ep, sl_init, tp_px, direction,
               breakeven_rr, risk, n):
    sl    = sl_init
    mfe_r = 0.0
    mae_r = 0.0

    for i in range(eb, n):
        hi = h[i]; lo = l[i]

        if direction == "LONG":
            fav = (hi - ep) / risk
            adv = (lo - ep) / risk
        else:
            fav = (ep - lo) / risk
            adv = (ep - hi) / risk
        if fav > mfe_r: mfe_r = fav
        if adv < mae_r: mae_r = adv

        if breakeven_rr > 0:
            if direction == "LONG"  and hi >= ep + breakeven_rr * risk:
                sl = max(sl, ep)
            elif direction == "SHORT" and lo <= ep - breakeven_rr * risk:
                sl = min(sl, ep)

        sl_hit = (lo <= sl) if direction == "LONG" else (hi >= sl)
        tp_hit = (hi >= tp_px) if direction == "LONG" else (lo <= tp_px)

        if sl_hit:
            return i, sl, "SL", round(mfe_r, 3), round(mae_r, 3)
        if tp_hit:
            return i, tp_px, "TP", round(mfe_r, 3), round(mae_r, 3)

    return n - 1, float(c[n - 1]), "END", round(mfe_r, 3), round(mae_r, 3)


# ── Phase B: exec simulation ───────────────────────────────────────────────────
#
# v11 keeps Phase B single-threaded (no multiprocessing) for two reasons:
#   1. Windows pickling of raw_signals (1M+ entries) causes MemoryError.
#   2. CPU constraint: keep usage ≤ 40%.
#
# We use module-level globals exactly as in v8.

_PC_B:  dict = {}
_RS_B:  dict = {}
_UNI_B: dict = {}


def _init_b(precomp, raw_signals, uni_syms):
    global _PC_B, _RS_B, _UNI_B
    _PC_B = precomp; _RS_B = raw_signals; _UNI_B = uni_syms


def _run_exec_combo(params: dict) -> list:
    lrc        = params["lrc_length"]
    sigma      = params["sigma_mult"]
    htf        = params["htf_touch_pct"]
    m15_t      = params["m15_touch_pct"]
    mscore     = params["min_htf_score"]
    swing_w    = params["swing_wing"]
    entry_tf_s = params["entry_tf"]
    rel_thresh = params["rel_perf_thresh"]
    atr_ratio  = params["atr_contract_ratio"]
    sl         = params["sl_atr_mult"]
    tp_r       = params["tp_rr"]
    be         = params["breakeven_rr"]

    # Reconstruct sig_key — must exactly match Phase A key
    htf_tf_frac    = htf   / 100.0
    entry_touch_frac = m15_t / 100.0
    sig_key = (lrc, sigma, htf_tf_frac, entry_touch_frac,
               mscore, entry_tf_s, rel_thresh, atr_ratio)

    raw = _RS_B.get(sig_key, {})

    candidates = []
    for sym, sigs in raw.items():
        pc = _PC_B.get(sym)
        if pc is None:
            continue
        h_ = pc["high"]; l_ = pc["low"]; c_ = pc["close"]
        n_ = pc["n"];    idx_ = pc["index"]

        for (bi, eb, ep, dir_, lv, atr_v, readiness) in sigs:
            pos   = +1 if dir_ == "SHORT" else -1
            sl_px = lv + pos * sl * atr_v
            if pos * (sl_px - ep) <= 0:
                continue
            risk = abs(ep - sl_px)
            if risk < 1e-6:
                continue
            tp_px = ep - pos * tp_r * risk

            xb, xp, xtype, mfe_r, mae_r = _find_exit(
                h_, l_, c_, eb, ep, sl_px, tp_px, dir_, be, risk, n_)

            r = (xp - ep) / risk if dir_ == "LONG" else (ep - xp) / risk
            dollar_pnl = r * (EFFECTIVE / ep) * risk

            candidates.append({
                "sym":        sym,
                "readiness":  readiness,
                "entry_date": idx_[eb],
                "exit_date":  idx_[xb],
                "direction":  dir_,
                "r":          round(r, 3),
                "pnl":        round(dollar_pnl, 2),
                "mfe_r":      mfe_r,
                "mae_r":      mae_r,
                "exit_type":  xtype,
            })

    candidates.sort(key=lambda t: (t["entry_date"], -t["readiness"]))
    avg_bars = sum(pc["n"] for pc in _PC_B.values()) // max(1, len(_PC_B))

    results = []
    for mc in MC_VALUES:
        mc_cap = mc if mc > 0 else 999_999
        for uni_key, uni_syms in _UNI_B.items():
            filtered = [t for t in candidates if t["sym"] in uni_syms]
            accepted  = []; open_pos = []
            for t in filtered:
                ed = t["entry_date"]
                open_pos = [(xd, s) for xd, s in open_pos if xd > ed]
                if len(open_pos) >= mc_cap:
                    continue
                if t["sym"] in {s for _, s in open_pos}:
                    continue
                accepted.append(t)
                open_pos.append((t["exit_date"], t["sym"]))

            all_r   = [t["r"]     for t in accepted]
            all_pnl = [t["pnl"]   for t in accepted]
            all_mfe = [t["mfe_r"] for t in accepted]
            all_mae = [t["mae_r"] for t in accepted]

            results.append({
                **params,
                "max_concurrent": mc,
                "universe":       uni_key,
                "n_uni_syms":     len(uni_syms),
                **_metrics(all_r, all_pnl, all_mfe, all_mae, avg_bars),
            })

    return results


# ── Metrics ────────────────────────────────────────────────────────────────────

def _metrics(all_r, all_pnl, all_mfe, all_mae, avg_bars) -> dict:
    empty = dict(
        n_trades=0, win_rate=0, avg_r=0, total_r=0,
        profit_factor=0, max_dd_r=0, sharpe_r=0,
        total_pnl_usd=0, max_dd_usd=0, sharpe_usd=0,
        avg_mfe_r=0, avg_mae_r=0,
        pct_mfe_gt_1r=0, pct_mae_lt_half=0, pct_loss_had_mfe=0,
    )
    if not all_r:
        return empty

    r   = np.array(all_r,   dtype=float)
    pnl = np.array(all_pnl, dtype=float)
    mfe = np.array(all_mfe, dtype=float)
    mae = np.array(all_mae, dtype=float)

    wins   = r > 0; losses = ~wins
    gw = r[wins].sum()  if wins.any()   else 0.0
    gl = abs(r[losses].sum()) if losses.any() else 1e-9

    def _dd(arr):
        eq = np.cumsum(np.insert(arr, 0, 0.0))
        return float((eq - np.maximum.accumulate(eq)).min())

    n_years = avg_bars / 252
    ann     = np.sqrt(len(r) / n_years) if n_years > 0 else 0

    def _sharpe(arr):
        s = arr.std()
        return float(arr.mean() / s * ann) if s > 0 else 0.0

    pct_mfe_gt_1r    = round(float((mfe >= 1.0).mean() * 100), 1)
    pct_mae_lt_half  = round(float((mae > -0.5).mean() * 100), 1)
    pct_loss_had_mfe = round(
        float((mfe[losses] > 0.0).mean() * 100) if losses.any() else 0.0, 1)

    return dict(
        n_trades         = len(r),
        win_rate         = round(float(wins.mean() * 100), 1),
        avg_r            = round(float(r.mean()), 3),
        total_r          = round(float(r.sum()), 2),
        profit_factor    = round(gw / gl, 2),
        max_dd_r         = round(_dd(r), 2),
        sharpe_r         = round(_sharpe(r), 2),
        total_pnl_usd    = round(float(pnl.sum()), 0),
        max_dd_usd       = round(_dd(pnl), 0),
        sharpe_usd       = round(_sharpe(pnl), 2),
        avg_mfe_r        = round(float(mfe.mean()), 3),
        avg_mae_r        = round(float(mae.mean()), 3),
        pct_mfe_gt_1r    = pct_mfe_gt_1r,
        pct_mae_lt_half  = pct_mae_lt_half,
        pct_loss_had_mfe = pct_loss_had_mfe,
    )


# ── Per-symbol stats for a given combo ────────────────────────────────────────

def compute_symbol_stats(best_row: dict, precomp: dict, raw_signals: dict,
                         uni_sets: dict) -> tuple:
    lrc        = best_row["lrc_length"]
    sigma      = best_row["sigma_mult"]
    htf        = best_row["htf_touch_pct"]
    m15_t      = best_row["m15_touch_pct"]
    mscore     = best_row["min_htf_score"]
    swing_w    = int(best_row.get("swing_wing", 2))
    entry_tf_s = best_row["entry_tf"]
    rel_thresh = best_row["rel_perf_thresh"]
    atr_ratio  = best_row["atr_contract_ratio"]
    sl         = best_row["sl_atr_mult"]
    tp_r       = best_row["tp_rr"]
    be         = best_row["breakeven_rr"]
    universe   = best_row["universe"]
    mc         = int(best_row["max_concurrent"])

    htf_tf_frac     = htf   / 100.0
    entry_touch_frac = m15_t / 100.0
    sig_key  = (lrc, sigma, htf_tf_frac, entry_touch_frac,
                mscore, entry_tf_s, rel_thresh, atr_ratio)
    raw      = raw_signals.get(sig_key, {})
    uni_syms = uni_sets.get(universe, frozenset())

    candidates = []
    for sym, sigs in raw.items():
        if sym not in uni_syms:
            continue
        pc = precomp.get(sym)
        if pc is None:
            continue
        h_ = pc["high"]; l_ = pc["low"]; c_ = pc["close"]
        n_ = pc["n"];    idx_ = pc["index"]

        for (bi, eb, ep, dir_, lv, atr_v, readiness) in sigs:
            pos   = +1 if dir_ == "SHORT" else -1
            sl_px = lv + pos * sl * atr_v
            if pos * (sl_px - ep) <= 0:
                continue
            risk = abs(ep - sl_px)
            if risk < 1e-6:
                continue
            tp_px = ep - pos * tp_r * risk
            xb, xp, xtype, mfe_r, mae_r = _find_exit(
                h_, l_, c_, eb, ep, sl_px, tp_px, dir_, be, risk, n_)
            r = (xp - ep) / risk if dir_ == "LONG" else (ep - xp) / risk
            dollar_pnl = r * (EFFECTIVE / ep) * risk
            candidates.append({
                "sym": sym, "readiness": readiness,
                "entry_date": idx_[eb], "exit_date": idx_[xb],
                "direction": dir_, "r": round(r, 3),
                "pnl": round(dollar_pnl, 2),
                "mfe_r": mfe_r, "mae_r": mae_r, "exit_type": xtype,
            })

    candidates.sort(key=lambda t: (t["entry_date"], -t["readiness"]))
    mc_cap = mc if mc > 0 else 999_999
    accepted = []; open_pos = []
    for t in candidates:
        ed = t["entry_date"]
        open_pos = [(xd, s) for xd, s in open_pos if xd > ed]
        if len(open_pos) >= mc_cap:
            continue
        if t["sym"] in {s for _, s in open_pos}:
            continue
        accepted.append(t)
        open_pos.append((t["exit_date"], t["sym"]))

    if not accepted:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(accepted)
    grp = df.groupby("sym")
    stats = pd.DataFrame({
        "n_trades":  grp["r"].count(),
        "win_rate":  grp["r"].apply(lambda x: round((x > 0).mean() * 100, 1)),
        "avg_r":     grp["r"].mean().round(3),
        "total_r":   grp["r"].sum().round(2),
        "total_pnl": grp["pnl"].sum().round(0),
        "avg_mfe_r": grp["mfe_r"].mean().round(3),
        "avg_mae_r": grp["mae_r"].mean().round(3),
        "pct_tp":    grp["exit_type"].apply(lambda x: round((x == "TP").mean() * 100, 1)),
        "pct_sl":    grp["exit_type"].apply(lambda x: round((x == "SL").mean() * 100, 1)),
    }).reset_index().sort_values("total_pnl", ascending=False)

    return stats, df


# ── Phase C: symbol discovery ──────────────────────────────────────────────────

def discover_symbols(df_sweep: pd.DataFrame, precomp: dict, raw_signals: dict,
                     uni_sets: dict, sym_meta: dict, ts: str) -> None:
    """
    For each symbol, find the best-param combo (by sharpe_r) and record stats.
    Filter into a live universe using DISCOVERY_MIN_TRADES / WIN_RATE / SHARPE.
    Outputs:
      v11_symbol_discovery_{ts}.csv  — all symbols × best combo
      v11_live_universe_{ts}.csv     — filtered candidates
    """
    log.info("Phase C: symbol discovery...")

    # Use best signal combo by sharpe_r (across all universes/MC)
    best_row = df_sweep.iloc[0].to_dict()

    lrc        = best_row["lrc_length"]
    sigma      = best_row["sigma_mult"]
    htf        = best_row["htf_touch_pct"]
    m15_t      = best_row["m15_touch_pct"]
    mscore     = best_row["min_htf_score"]
    swing_w    = int(best_row.get("swing_wing", 2))
    entry_tf_s = best_row["entry_tf"]
    rel_thresh = best_row["rel_perf_thresh"]
    atr_ratio  = best_row["atr_contract_ratio"]
    sl         = best_row["sl_atr_mult"]
    tp_r       = best_row["tp_rr"]
    be         = best_row["breakeven_rr"]

    htf_tf_frac      = htf   / 100.0
    entry_touch_frac = m15_t / 100.0
    sig_key = (lrc, sigma, htf_tf_frac, entry_touch_frac,
               mscore, entry_tf_s, rel_thresh, atr_ratio)
    raw = raw_signals.get(sig_key, {})

    all_syms_with_data = set(raw.keys())
    rows = []

    for sym in all_syms_with_data:
        pc = precomp.get(sym)
        if pc is None:
            continue
        sigs = raw.get(sym, [])
        if not sigs:
            continue

        h_ = pc["high"]; l_ = pc["low"]; c_ = pc["close"]
        n_ = pc["n"];    idx_ = pc["index"]

        sym_trades = []
        for (bi, eb, ep, dir_, lv, atr_v, readiness) in sigs:
            pos   = +1 if dir_ == "SHORT" else -1
            sl_px = lv + pos * sl * atr_v
            if pos * (sl_px - ep) <= 0:
                continue
            risk = abs(ep - sl_px)
            if risk < 1e-6:
                continue
            tp_px = ep - pos * tp_r * risk
            xb, xp, xtype, mfe_r, mae_r = _find_exit(
                h_, l_, c_, eb, ep, sl_px, tp_px, dir_, be, risk, n_)
            r = (xp - ep) / risk if dir_ == "LONG" else (ep - xp) / risk
            dollar_pnl = r * (EFFECTIVE / ep) * risk
            sym_trades.append({
                "r": r, "pnl": dollar_pnl,
                "entry_date": idx_[eb], "exit_date": idx_[xb],
                "direction": dir_, "mfe_r": mfe_r, "mae_r": mae_r,
                "readiness": readiness,
            })

        if not sym_trades:
            continue

        sym_trades.sort(key=lambda t: t["entry_date"])
        all_r   = [t["r"]   for t in sym_trades]
        all_pnl = [t["pnl"] for t in sym_trades]
        all_mfe = [t["mfe_r"] for t in sym_trades]
        all_mae = [t["mae_r"] for t in sym_trades]

        avg_bars = pc["n"]
        m = _metrics(all_r, all_pnl, all_mfe, all_mae, avg_bars)
        meta = sym_meta.get(sym, {})
        rows.append({
            "sym":       sym,
            "exchange":  meta.get("exchange", ""),
            "sector":    meta.get("sector",   "Unknown"),
            **m,
        })

    if not rows:
        log.warning("Phase C: no symbols with trades.")
        return

    disc_df = pd.DataFrame(rows).sort_values("sharpe_r", ascending=False)
    disc_path = RESULTS_DIR / f"v11_symbol_discovery_{ts}.csv"
    disc_df.to_csv(disc_path, index=False)
    log.info(f"Symbol discovery -> {disc_path}  ({len(disc_df)} symbols)")

    live_df = disc_df[
        (disc_df["n_trades"]  >= DISCOVERY_MIN_TRADES)  &
        (disc_df["win_rate"]  >= DISCOVERY_MIN_WIN_RATE) &
        (disc_df["sharpe_r"]  >= DISCOVERY_MIN_SHARPE)
    ].copy()
    live_path = RESULTS_DIR / f"v11_live_universe_{ts}.csv"
    live_df.to_csv(live_path, index=False)
    log.info(f"Live universe    -> {live_path}  ({len(live_df)} symbols)")

    print(f"\n--- Phase C: Top 40 symbols by Sharpe (n_trades>={DISCOVERY_MIN_TRADES}, "
          f"WR>={DISCOVERY_MIN_WIN_RATE}%, Sharpe>={DISCOVERY_MIN_SHARPE}) ---")
    show_cols = ["sym", "exchange", "sector", "n_trades", "win_rate", "sharpe_r",
                 "avg_r", "total_pnl_usd", "profit_factor"]
    avail = [cc for cc in show_cols if cc in live_df.columns]
    print(live_df[avail].head(40).to_string(index=False))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extend", action="store_true",
                    help="Skip daily cache extension")
    ap.add_argument("--fetch-m15",  action="store_true",
                    help="Pull M15 bars from MT5 before sweep")
    ap.add_argument("--fetch-w1",   action="store_true",
                    help="Pull weekly bars from MT5 before sweep")
    args = ap.parse_args()

    t0 = datetime.now()
    log.info("v11 sweep starting (single-threaded)")

    # ── Daily cache extension ──
    if not args.skip_extend:
        log.info("Extending daily MT5 cache...")
        try:
            from stocks.backtest.data.extender import extend_cache
            univ_ = build_universe()
            syms_ = univ_[univ_["exchange"].isin(ALL_EXCH)]["symbol"].tolist()
            extend_cache(syms_, [])
        except Exception as e:
            log.warning(f"Daily cache extension failed: {e}")

    # ── W1 / M15 cache fetch ──
    if args.fetch_w1:
        log.info("Fetching weekly bars from MT5...")
        try:
            from stocks.backtest.data.extender import extend_cache_w1
            univ_ = build_universe()
            syms_ = univ_[univ_["exchange"].isin(ALL_EXCH)]["symbol"].tolist()
            extend_cache_w1(syms_)
        except Exception as e:
            log.warning(f"W1 fetch failed: {e}")

    if args.fetch_m15:
        log.info("Fetching M15 bars from MT5...")
        try:
            from stocks.backtest.data.extender import extend_cache_m15
            univ_ = build_universe()
            syms_ = univ_[univ_["exchange"].isin(ALL_EXCH)]["symbol"].tolist()
            extend_cache_m15(syms_)
        except Exception as e:
            log.warning(f"M15 fetch failed: {e}")

    # ── Data availability checks ──
    w1_available  = (BT_CACHE_W1_DIR.exists()
                     and any(BT_CACHE_W1_DIR.glob("*.parquet")))
    m15_available = (BT_CACHE_M15_DIR.exists()
                     and any(BT_CACHE_M15_DIR.glob("*.parquet")))
    log.info(f"W1 data:  {'available' if w1_available  else 'MISSING — weekly levels disabled'}")
    log.info(f"M15 data: {'available' if m15_available else 'MISSING — entry TF disabled'}")

    # ── Load SPX returns (once, global) ──
    spx_ret5_map = load_spx_rets()

    # ── Load daily bars ──
    log.info("Loading daily bars...")
    univ = build_universe()
    bars, dvols, sym_meta = load_all_bars(univ)
    log.info(f"Bars loaded: {len(bars)} symbols")

    uni_sets = build_universe_sets(bars, dvols, sym_meta)
    all_syms = set().union(*uni_sets.values())
    bars = {s: bars[s] for s in bars if s in all_syms}
    log.info(f"Symbols after universe filter: {len(bars)}")

    # ── Precompute ──
    lrc_lengths    = SIGNAL_GRID["lrc_length"]
    sigma_mults    = SIGNAL_GRID["sigma_mult"]
    htf_touch_pcts = SIGNAL_GRID["htf_touch_pct"]
    swing_wings    = SIGNAL_GRID["swing_wing"]

    log.info(f"Precomputing indicators for {len(bars)} symbols "
             f"(W1={'yes' if w1_available else 'no'}, "
             f"M15={'yes' if m15_available else 'no'})...")

    precomp = {}
    w1_loaded = m15_loaded = 0
    for sym, df in bars.items():
        try:
            w1_df  = load_bt_bars_w1(sym)  if w1_available  else pd.DataFrame()
            m15_df = load_bt_bars_m15(sym) if m15_available else pd.DataFrame()
            if not w1_df.empty:  w1_loaded  += 1
            if not m15_df.empty: m15_loaded += 1
            precomp[sym] = precompute_symbol_v11(
                df, m15_df, w1_df,
                lrc_lengths, sigma_mults, htf_touch_pcts, swing_wings,
            )
        except Exception as e:
            log.debug(f"  {sym} precompute failed: {e}")

    log.info(f"Precomputed {len(precomp)} symbols "
             f"({w1_loaded} with W1, {m15_loaded} with M15) "
             f"in {(datetime.now()-t0).seconds}s")

    # ── Phase A: signal finding ──
    # sig_key = 8-tuple:
    #   (lrc_length, sigma_mult, htf_tf_frac, entry_touch_frac,
    #    min_htf_score, entry_tf_str, rel_perf_thresh, atr_contract_ratio)
    sig_keys = list(product(
        lrc_lengths,
        sigma_mults,
        [p / 100.0 for p in htf_touch_pcts],      # htf_tf_frac
        [p / 100.0 for p in SIGNAL_GRID["m15_touch_pct"]],  # entry_touch_frac
        SIGNAL_GRID["min_htf_score"],
        SIGNAL_GRID["entry_tf"],
        SIGNAL_GRID["rel_perf_thresh"],
        SIGNAL_GRID["atr_contract_ratio"],
    ))
    # Note: swing_wing is fixed at 2 — we bake it in rather than sweeping
    SWING_WING_FIXED = SIGNAL_GRID["swing_wing"][0]

    log.info(f"Phase A: {len(sig_keys)} signal groups x {len(precomp)} symbols "
             f"(single-threaded)...")
    t_a = datetime.now()
    raw_signals = {}
    for i, sig_key in enumerate(sig_keys, 1):
        (lrc, sigma, htf_tf, entry_touch_tf,
         min_score, entry_tf_str, rel_perf_thresh, atr_contract_ratio) = sig_key

        group_out = {}
        for sym, pc in precomp.items():
            try:
                s = _find_signals_v11(
                    pc,
                    lrc_length=lrc,
                    sigma_mult=sigma,
                    htf_tf=htf_tf,
                    entry_touch_tf=entry_touch_tf,
                    min_htf_score=min_score,
                    swing_wing=SWING_WING_FIXED,
                    entry_tf_str=entry_tf_str,
                    rel_perf_thresh=rel_perf_thresh,
                    atr_contract_ratio=atr_contract_ratio,
                    spx_ret5_map=spx_ret5_map,
                )
                if s:
                    group_out[sym] = s
            except Exception:
                pass
        raw_signals[sig_key] = group_out

        if i % 10 == 0 or i == len(sig_keys):
            log.info(f"  Phase A {i}/{len(sig_keys)} groups")

    n_raw = sum(len(s) for d in raw_signals.values() for s in d.values())
    log.info(f"Phase A done in {(datetime.now()-t_a).seconds}s -- {n_raw:,} raw signals")

    # Strip bulky per-bar dicts from precomp before Phase B
    # (not needed for exit sim; keeping them would waste memory)
    for pc in precomp.values():
        pc.pop("m15_by_date", None)
        pc.pop("m30_by_date", None)
        pc.pop("h1_by_date",  None)
        pc.pop("w_levels",    None)

    if n_raw == 0:
        log.warning("No signals found. Ensure W1 and M15 caches are populated "
                    "(run with --fetch-w1 --fetch-m15).")
        return

    # ── Phase B: exec combos ──
    exec_keys   = list(EXEC_GRID.keys())
    exec_combos = []
    for sk in sig_keys:
        (lrc, sigma, htf_tf_frac, entry_touch_frac,
         min_score, entry_tf_str, rel_perf_thresh, atr_contract_ratio) = sk
        # Convert back to pct for the params dict (CSV-friendly)
        htf_pct  = round(htf_tf_frac * 100.0, 6)
        m15_pct  = round(entry_touch_frac * 100.0, 6)
        for combo in product(*[EXEC_GRID[k] for k in exec_keys]):
            d = dict(zip(exec_keys, combo))
            d.update(
                lrc_length=lrc,
                sigma_mult=sigma,
                htf_touch_pct=htf_pct,
                m15_touch_pct=m15_pct,
                min_htf_score=min_score,
                swing_wing=SWING_WING_FIXED,
                entry_tf=entry_tf_str,
                rel_perf_thresh=rel_perf_thresh,
                atr_contract_ratio=atr_contract_ratio,
            )
            exec_combos.append(d)

    n_portfolio = len(MC_VALUES) * len(uni_sets)
    log.info(f"Phase B: {len(exec_combos)} exec combos x {n_portfolio} portfolio "
             f"variants = {len(exec_combos) * n_portfolio:,} result rows")

    _init_b(precomp, raw_signals, uni_sets)
    t_b = datetime.now()
    flat = []
    for i, combo in enumerate(exec_combos, 1):
        flat.extend(_run_exec_combo(combo))
        if i % 100 == 0 or i == len(exec_combos):
            log.info(f"  Phase B {i}/{len(exec_combos)} combos")
    log.info(f"Phase B done in {(datetime.now()-t_b).seconds}s -- {len(flat):,} rows")

    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M")
    df_res = pd.DataFrame(flat).sort_values("sharpe_r", ascending=False)

    full_path = RESULTS_DIR / f"v11_sweep_{ts}.csv"
    df_res.to_csv(full_path, index=False)
    log.info(f"Full sweep -> {full_path}  ({len(df_res):,} rows)")

    top_path = RESULTS_DIR / f"v11_top_{ts}.csv"
    df_res.head(200).to_csv(top_path, index=False)
    log.info(f"Top 200    -> {top_path}")

    # ── Per-symbol stats + per-trade log for best combo ──
    best = df_res.iloc[0].to_dict()
    log.info(f"Computing per-symbol stats for best combo "
             f"(Sharpe={best['sharpe_r']}, WR={best['win_rate']}%)...")
    sym_stats, trade_log = compute_symbol_stats(best, precomp, raw_signals, uni_sets)

    if not sym_stats.empty:
        sym_path = RESULTS_DIR / f"v11_symbol_stats_{ts}.csv"
        sym_stats.to_csv(sym_path, index=False)
        log.info(f"Symbol stats -> {sym_path}")

    monthly = pd.DataFrame()
    if not trade_log.empty:
        tlog_path = RESULTS_DIR / f"v11_trade_log_{ts}.csv"
        trade_log.to_csv(tlog_path, index=False)
        log.info(f"Trade log    -> {tlog_path}")

        # Monthly PnL breakdown
        trade_log["month"] = pd.to_datetime(trade_log["entry_date"]).dt.to_period("M")
        monthly = (trade_log.groupby("month")
                   .agg(n_trades=("r", "count"),
                        win_rate=("r", lambda x: round((x > 0).mean() * 100, 1)),
                        total_r=("r", "sum"),
                        total_pnl=("pnl", "sum"))
                   .round(3).reset_index())
        monthly["month"] = monthly["month"].astype(str)
        mon_path = RESULTS_DIR / f"v11_monthly_{ts}.csv"
        monthly.to_csv(mon_path, index=False)
        log.info(f"Monthly PnL  -> {mon_path}")

    elapsed = (datetime.now() - t0).total_seconds()
    log.info(f"Total time: {elapsed:.0f}s")

    # ── Phase C: symbol discovery ──
    discover_symbols(df_res, precomp, raw_signals, uni_sets, sym_meta, ts)

    # ── Print results ──
    cols_main = [
        "universe", "lrc_length", "sigma_mult",
        "htf_touch_pct", "m15_touch_pct", "min_htf_score", "swing_wing",
        "entry_tf", "rel_perf_thresh", "atr_contract_ratio",
        "sl_atr_mult", "tp_rr", "breakeven_rr", "max_concurrent",
        "n_trades", "win_rate", "avg_r",
        "total_pnl_usd", "max_dd_usd", "sharpe_r", "sharpe_usd",
        "avg_mfe_r", "avg_mae_r", "pct_mfe_gt_1r", "pct_mae_lt_half",
    ]

    print("\n--- Top 20 combos by Sharpe (R) ---")
    avail_main = [c for c in cols_main if c in df_res.columns]
    print(df_res[avail_main].head(20).to_string(index=False))

    if not sym_stats.empty:
        print(f"\n--- Top 15 symbols (best combo: {best['universe']} | "
              f"LRC-{int(best['lrc_length'])} | sigma={best['sigma_mult']} | "
              f"htf={best['htf_touch_pct']}% | entry_touch={best['m15_touch_pct']}% | "
              f"entry_tf={best['entry_tf']} | min_score={int(best['min_htf_score'])} | "
              f"rel_perf={best['rel_perf_thresh']} | atr_ratio={best['atr_contract_ratio']}) ---")
        print(sym_stats.head(15).to_string(index=False))

    if not trade_log.empty:
        print("\n--- Direction split (LONG vs SHORT) ---")
        dir_split = (trade_log.groupby("direction")
                     .agg(n_trades=("r", "count"),
                          win_rate=("r", lambda x: round((x > 0).mean() * 100, 1)),
                          avg_r=("r", "mean"),
                          total_pnl=("pnl", "sum"))
                     .round(3).reset_index())
        print(dir_split.to_string(index=False))

        if not monthly.empty:
            print("\n--- Monthly PnL (best combo) ---")
            print(monthly.to_string(index=False))

    # ── Param contribution: entry_tf × rel_perf_thresh × atr_contract_ratio ──
    print("\n--- Param contribution: entry_tf x rel_perf_thresh x atr_contract_ratio "
          "(avg Sharpe) ---")
    grp_cols = ["entry_tf", "rel_perf_thresh", "atr_contract_ratio",
                "sharpe_r", "win_rate", "n_trades"]
    avail_g = [cc for cc in grp_cols if cc in df_res.columns]
    if len(avail_g) >= 4:
        param_grp = (
            df_res[avail_g]
            .groupby(["entry_tf", "rel_perf_thresh", "atr_contract_ratio"])
            .agg(sharpe_r=("sharpe_r", "mean"),
                 win_rate=("win_rate", "mean"),
                 n_trades=("n_trades", "mean"))
            .round(2).reset_index()
            .sort_values("sharpe_r", ascending=False)
        )
        print(param_grp.to_string(index=False))

    # ── Classic param contribution (min_score × m15_touch × sigma) ──
    print("\n--- Param contribution: sigma_mult x m15_touch_pct x entry_tf "
          "(avg Sharpe) ---")
    grp2 = ["sigma_mult", "m15_touch_pct", "entry_tf",
             "sharpe_r", "win_rate", "n_trades"]
    avail_g2 = [cc for cc in grp2 if cc in df_res.columns]
    if len(avail_g2) >= 4:
        param_grp2 = (
            df_res[avail_g2]
            .groupby(["sigma_mult", "m15_touch_pct", "entry_tf"])
            .agg(sharpe_r=("sharpe_r", "mean"),
                 win_rate=("win_rate", "mean"),
                 n_trades=("n_trades", "mean"))
            .round(2).reset_index()
            .sort_values("sharpe_r", ascending=False)
        )
        print(param_grp2.to_string(index=False))


if __name__ == "__main__":
    main()
