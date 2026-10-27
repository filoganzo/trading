"""
V10 backtest sweep: full cross-universe search + Phase C symbol discovery.

What's new vs v8:
  - Expanded SIGNAL_GRID: lrc=[60,80,100], sigma=[1.0,1.5,2.0],
    htf=[0.5,1.0], m15=[0.3,0.5,1.0], score=[40,60], wing=[1,2].
  - US-Broad universe added (top 50pct by dvol, vs v8's top 75pct) so
    mid-cap symbols that might respond well are included.
  - Phase C — Symbol Discovery: after the sweep, take the best signal+exec
    params and run them on EVERY precomputed symbol independently
    (ignoring portfolio/universe membership).  Ranks all symbols by
    per-symbol Sharpe.  Output: v10_symbol_discovery_{ts}.csv +
    a recommended live universe (n_trades>=5, WR>=55%, sharpe>=0.8).

Usage:
    python -m stocks.backtest.runner_v10 --skip-extend

Output (RESULTS_DIR):
    v10_sweep_{ts}.csv
    v10_top_{ts}.csv
    v10_symbol_stats_{ts}.csv
    v10_trade_log_{ts}.csv
    v10_monthly_{ts}.csv
    v10_symbol_discovery_{ts}.csv   ← Phase C: all symbols ranked
    v10_live_universe_{ts}.csv      ← curated shortlist
"""
import sys, warnings, logging, argparse
from itertools import product
from multiprocessing import Pool, cpu_count
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, r'path/to/repo/hft')
warnings.filterwarnings("ignore")

from stocks.backtest.config import RESULTS_DIR, BT_CACHE_M15_DIR, BT_CACHE_W1_DIR
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

# ── Constants ─────────────────────────────────────────────────────────────────
US_EXCH  = {"NYSE", "NAS"}
EU_EXCH  = {"LSE", "ETR", "PAR", "IT", "MAD", "AMS", "IE", "SWX"}
ALL_EXCH = US_EXCH | EU_EXCH
MIN_BARS = 200
LRC_CL   = 15; LRC_CS = 9; ATR_P = 14

LEVERAGE  = 20
CAPITAL   = 1_000.0
EFFECTIVE = CAPITAL * LEVERAGE   # $20,000

# ── Sweep grids ───────────────────────────────────────────────────────────────
SIGNAL_GRID = {
    "lrc_length":    [60, 80, 100],       # v8 confirmed 80 best; test neighbours
    "sigma_mult":    [1.0, 1.5, 2.0],    # wider band = fewer but higher-quality signals
    "htf_touch_pct": [0.5, 1.0],         # 0.5% confirmed; 1.0% for less liquid symbols
    "m15_touch_pct": [0.3, 0.5, 1.0],
    "min_htf_score": [40, 60],            # 60 confirmed superior; 40 as baseline
    "swing_wing":    [1, 2],             # 2 confirmed superior; keep 1 for comparison
}

EXEC_GRID = {
    "sl_atr_mult":  [1.0],
    "tp_rr":        [0.75, 1.0, 1.5, 2.0, 3.0],
    "breakeven_rr": [0.0],
}

MC_VALUES = [0, 5, 10]

# Phase C quality filters for live universe recommendation
DISC_MIN_TRADES = 5
DISC_MIN_WR     = 55.0
DISC_MIN_SHARPE = 0.8


# ── Universe ──────────────────────────────────────────────────────────────────

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
    def top_q(syms, q):
        sub = {s: dvols[s] for s in syms if s in dvols}
        if not sub:
            return frozenset()
        s = pd.Series(sub)
        return frozenset(s[s >= s.quantile(q)].index)

    us_all = {s for s, m in sym_meta.items() if m["exchange"] in US_EXCH}
    eu_all = {s for s, m in sym_meta.items() if m["exchange"] in EU_EXCH}

    uni = {
        "US-Broad":  top_q(us_all, 0.50),   # new: top-50pct by dvol
        "US-All":    top_q(us_all, 0.75),
        "US-Top10":  top_q(us_all, 0.90),
        "EU-All":    top_q(eu_all, 0.75),
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


# ── Precompute ────────────────────────────────────────────────────────────────

def precompute_symbol(df: pd.DataFrame, m15_df: pd.DataFrame,
                      w1_df: pd.DataFrame,
                      lrc_lengths, sigma_mults,
                      htf_touch_pcts, swing_wings) -> dict:
    c = df["close"].values; h = df["high"].values
    l = df["low"].values;   o = df["open"].values

    lrc_data = {}
    for ll in lrc_lengths:
        for sm in sigma_mults:
            mid, upper, lower, _ = _lrc_vectorized(c, ll, sm)
            lrc_data[(ll, sm)] = (mid, upper, lower)

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

    m15_by_date = {}
    if m15_df is not None and not m15_df.empty:
        m15 = m15_df.copy()
        m15.columns = [cc.lower() for cc in m15.columns]
        m15["_date"] = m15.index.normalize()
        for d, grp in m15.groupby("_date"):
            brs = list(zip(grp["high"].values, grp["low"].values,
                           grp["close"].values, grp["open"].values))
            m15_by_date[d] = brs

    return {
        "close": c, "high": h, "low": l, "open": o,
        "n": len(c), "index": df.index,
        "lrc_c":    _lrc_candle_close_vectorized(c, LRC_CL, LRC_CS),
        "atr":      _atr_vectorized(h, l, c, ATR_P),
        "lrc_data": lrc_data,
        "w_levels":    w_levels,
        "m15_by_date": m15_by_date,
    }


# ── Phase A: signal finding ───────────────────────────────────────────────────

def _find_signals_v10(pc: dict, lrc_length: int, sigma_mult: float,
                      htf_tf: float, m15_tf: float,
                      min_htf_score: float = 0.0,
                      swing_wing: int = 2) -> list:
    n         = pc["n"]
    lrc_c     = pc["lrc_c"]
    atr       = pc["atr"]
    h         = pc["high"]; l = pc["low"]; c = pc["close"]
    idx       = pc["index"]
    w_levels  = pc["w_levels"].get((swing_wing, htf_tf), [])
    m15_by_dt = pc["m15_by_date"]

    if not w_levels:
        return []

    mid, upper, lower = pc["lrc_data"].get((lrc_length, sigma_mult), (None, None, None))
    if mid is None:
        return []

    min_i = max(lrc_length, LRC_CL + LRC_CS + 1, ATR_P + 1)
    if min_i >= n - 1:
        return []

    sigs = []
    for i in range(min_i, n - 1):
        lrc_ci = lrc_c[i]
        if np.isnan(lrc_ci) or np.isnan(upper[i]) or np.isnan(atr[i]):
            continue

        if lrc_ci > upper[i]:
            direction = "SHORT"
        elif lrc_ci < lower[i]:
            direction = "LONG"
        else:
            continue

        price = float(c[i])
        lv, score = find_htf_level(w_levels, price, htf_tf, min_htf_score)
        if np.isnan(lv):
            continue

        eb = i + 1
        day_key = pd.Timestamp(idx[eb]).normalize()
        m15_bars = m15_by_dt.get(day_key)
        if not m15_bars:
            continue

        _, entry_px = find_m15_entry(m15_bars, direction, lv, m15_tf)
        if entry_px is None:
            continue

        sigs.append((i, eb, entry_px, direction, lv, float(atr[i]), score))

    return sigs


# ── Exit simulation with MFE/MAE ─────────────────────────────────────────────

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


# ── Phase B worker ────────────────────────────────────────────────────────────
_PC_B:  dict = {}
_RS_B:  dict = {}
_UNI_B: dict = {}

def _init_b(precomp, raw_signals, uni_syms):
    global _PC_B, _RS_B, _UNI_B
    _PC_B = precomp; _RS_B = raw_signals; _UNI_B = uni_syms


def _run_exec_combo(params: dict) -> list:
    lrc    = params["lrc_length"];  sigma   = params["sigma_mult"]
    htf    = params["htf_touch_pct"]; m15_t = params["m15_touch_pct"]
    mscore = params["min_htf_score"]; swing_w = params["swing_wing"]
    sl     = params["sl_atr_mult"];  tp_r   = params["tp_rr"]
    be     = params["breakeven_rr"]

    raw = _RS_B.get((lrc, sigma, htf, m15_t, mscore, swing_w), {})

    candidates = []
    for sym, sigs in raw.items():
        pc = _PC_B.get(sym)
        if pc is None:
            continue
        h = pc["high"]; l = pc["low"]; c = pc["close"]
        n = pc["n"];     idx = pc["index"]

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
                h, l, c, eb, ep, sl_px, tp_px, dir_, be, risk, n)

            r = (xp - ep) / risk if dir_ == "LONG" else (ep - xp) / risk
            dollar_pnl = r * (EFFECTIVE / ep) * risk

            candidates.append({
                "sym":        sym,
                "readiness":  readiness,
                "entry_date": idx[eb],
                "exit_date":  idx[xb],
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


# ── Metrics ───────────────────────────────────────────────────────────────────

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
        n_trades        = len(r),
        win_rate        = round(float(wins.mean() * 100), 1),
        avg_r           = round(float(r.mean()), 3),
        total_r         = round(float(r.sum()), 2),
        profit_factor   = round(gw / gl, 2),
        max_dd_r        = round(_dd(r), 2),
        sharpe_r        = round(_sharpe(r), 2),
        total_pnl_usd   = round(float(pnl.sum()), 0),
        max_dd_usd      = round(_dd(pnl), 0),
        sharpe_usd      = round(_sharpe(pnl), 2),
        avg_mfe_r       = round(float(mfe.mean()), 3),
        avg_mae_r       = round(float(mae.mean()), 3),
        pct_mfe_gt_1r   = pct_mfe_gt_1r,
        pct_mae_lt_half = pct_mae_lt_half,
        pct_loss_had_mfe= pct_loss_had_mfe,
    )


# ── Per-symbol stats for best combo (portfolio-level) ────────────────────────

def compute_symbol_stats(best_row: dict, precomp: dict, raw_signals: dict,
                         uni_sets: dict) -> tuple:
    lrc    = best_row["lrc_length"];  sigma   = best_row["sigma_mult"]
    htf    = best_row["htf_touch_pct"]; m15_t = best_row["m15_touch_pct"]
    mscore  = best_row["min_htf_score"]; swing_w = int(best_row.get("swing_wing", 2))
    sl      = best_row["sl_atr_mult"];  tp_r   = best_row["tp_rr"]
    be      = best_row["breakeven_rr"]
    universe = best_row["universe"]
    mc       = int(best_row["max_concurrent"])

    raw      = raw_signals.get((lrc, sigma, htf, m15_t, mscore, swing_w), {})
    uni_syms = uni_sets.get(universe, frozenset())

    candidates = []
    for sym, sigs in raw.items():
        if sym not in uni_syms:
            continue
        pc = precomp.get(sym)
        if pc is None:
            continue
        h = pc["high"]; l = pc["low"]; c = pc["close"]
        n = pc["n"];     idx = pc["index"]

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
                h, l, c, eb, ep, sl_px, tp_px, dir_, be, risk, n)
            r = (xp - ep) / risk if dir_ == "LONG" else (ep - xp) / risk
            dollar_pnl = r * (EFFECTIVE / ep) * risk
            candidates.append({
                "sym": sym, "readiness": readiness,
                "entry_date": idx[eb], "exit_date": idx[xb],
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


# ── Phase C: Symbol Discovery ─────────────────────────────────────────────────

def discover_symbols(best_row: dict, precomp: dict, raw_signals: dict,
                     sym_meta: dict) -> pd.DataFrame:
    """Run best params on EVERY precomputed symbol independently (no portfolio
    max_concurrent filter).  Returns all symbols ranked by per-symbol Sharpe."""
    lrc    = best_row["lrc_length"];  sigma   = best_row["sigma_mult"]
    htf    = best_row["htf_touch_pct"]; m15_t = best_row["m15_touch_pct"]
    mscore  = best_row["min_htf_score"]; swing_w = int(best_row.get("swing_wing", 2))
    sl      = best_row["sl_atr_mult"];  tp_r   = best_row["tp_rr"]
    be      = best_row["breakeven_rr"]

    raw = raw_signals.get((lrc, sigma, htf, m15_t, mscore, swing_w), {})

    rows = []
    for sym, sigs in raw.items():
        pc = precomp.get(sym)
        if pc is None:
            continue
        h = pc["high"]; l = pc["low"]; c = pc["close"]
        n = pc["n"]; idx = pc["index"]

        trades = []
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
                h, l, c, eb, ep, sl_px, tp_px, dir_, be, risk, n)
            r_val = (xp - ep) / risk if dir_ == "LONG" else (ep - xp) / risk
            dollar_pnl = r_val * (EFFECTIVE / ep) * risk
            trades.append({
                "r":          round(r_val, 3),
                "pnl":        round(dollar_pnl, 2),
                "direction":  dir_,
                "exit_type":  xtype,
                "mfe_r":      mfe_r,
                "mae_r":      mae_r,
                "entry_date": idx[eb],
            })

        if not trades:
            continue

        r_arr   = np.array([t["r"]   for t in trades], dtype=float)
        pnl_arr = np.array([t["pnl"] for t in trades], dtype=float)
        mfe_arr = np.array([t["mfe_r"] for t in trades], dtype=float)
        mae_arr = np.array([t["mae_r"] for t in trades], dtype=float)
        wins    = r_arr > 0

        n_years = pc["n"] / 252
        ann     = np.sqrt(len(r_arr) / n_years) if n_years > 0 else 0
        sharpe  = float(r_arr.mean() / r_arr.std() * ann) if r_arr.std() > 0 else 0.0

        longs  = [t for t in trades if t["direction"] == "LONG"]
        shorts = [t for t in trades if t["direction"] == "SHORT"]

        def _wr(ts):
            if not ts:
                return None
            return round((np.array([t["r"] for t in ts]) > 0).mean() * 100, 1)

        meta = sym_meta.get(sym, {})
        rows.append({
            "sym":       sym,
            "sector":    meta.get("sector", "?"),
            "exchange":  meta.get("exchange", "?"),
            "n_trades":  len(trades),
            "win_rate":  round(float(wins.mean() * 100), 1),
            "avg_r":     round(float(r_arr.mean()), 3),
            "total_r":   round(float(r_arr.sum()), 2),
            "total_pnl": round(float(pnl_arr.sum()), 0),
            "sharpe_r":  round(sharpe, 2),
            "max_dd_r":  round(float((np.cumsum(np.insert(r_arr, 0, 0))
                                      - np.maximum.accumulate(np.cumsum(np.insert(r_arr, 0, 0)))).min()), 2),
            "avg_mfe_r": round(float(mfe_arr.mean()), 3),
            "avg_mae_r": round(float(mae_arr.mean()), 3),
            "n_long":    len(longs),
            "n_short":   len(shorts),
            "long_wr":   _wr(longs),
            "short_wr":  _wr(shorts),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("sharpe_r", ascending=False).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extend", action="store_true")
    ap.add_argument("--fetch-m15",   action="store_true")
    ap.add_argument("--fetch-w1",    action="store_true")
    ap.add_argument("--workers", type=int,
                    default=max(1, int(cpu_count() * 0.4)))
    args = ap.parse_args()

    workers = args.workers
    t0 = datetime.now()
    log.info(f"v10 sweep starting | workers={workers}")

    if not args.skip_extend:
        log.info("Extending daily MT5 cache...")
        try:
            from stocks.backtest.data.extender import extend_cache
            univ_ = build_universe()
            syms_ = univ_[univ_["exchange"].isin(ALL_EXCH)]["symbol"].tolist()
            bmarks = list({r["benchmark"] for _, r in univ_.iterrows() if r.get("benchmark")})
            extend_cache(syms_, bmarks)
        except Exception as e:
            log.warning(f"Daily cache extension failed: {e}")

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

    w1_available  = (BT_CACHE_W1_DIR.exists()
                     and any(BT_CACHE_W1_DIR.glob("*.parquet")))
    m15_available = (BT_CACHE_M15_DIR.exists()
                     and any(BT_CACHE_M15_DIR.glob("*.parquet")))
    log.info(f"W1 data:  {'available' if w1_available else 'MISSING'}")
    log.info(f"M15 data: {'available' if m15_available else 'MISSING'}")

    log.info("Loading daily bars...")
    univ = build_universe()
    bars, dvols, sym_meta = load_all_bars(univ)
    log.info(f"Bars loaded: {len(bars)} symbols")

    uni_sets = build_universe_sets(bars, dvols, sym_meta)
    all_syms = set().union(*uni_sets.values())
    bars = {s: bars[s] for s in bars if s in all_syms}
    log.info(f"Symbols in universe: {len(bars)}")

    lrc_lengths    = SIGNAL_GRID["lrc_length"]
    sigma_mults    = SIGNAL_GRID["sigma_mult"]
    htf_touch_pcts = SIGNAL_GRID["htf_touch_pct"]

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
            precomp[sym] = precompute_symbol(
                df, m15_df, w1_df, lrc_lengths, sigma_mults, htf_touch_pcts,
                SIGNAL_GRID["swing_wing"])
        except Exception as e:
            log.debug(f"  {sym} precompute failed: {e}")

    log.info(f"Precomputed {len(precomp)} symbols "
             f"({w1_loaded} with W1, {m15_loaded} with M15) "
             f"in {(datetime.now()-t0).seconds}s")

    # ── Phase A ──
    sig_keys = list(product(
        lrc_lengths, sigma_mults,
        SIGNAL_GRID["htf_touch_pct"],
        SIGNAL_GRID["m15_touch_pct"],
        SIGNAL_GRID["min_htf_score"],
        SIGNAL_GRID["swing_wing"],
    ))
    log.info(f"Phase A: {len(sig_keys)} signal groups x {len(precomp)} symbols "
             f"(single-threaded)...")
    t_a = datetime.now()
    raw_signals = {}
    for i, sig_key in enumerate(sig_keys, 1):
        lrc, sigma, htf_pct, m15_pct, min_score, swing_wing = sig_key
        htf_tf = htf_pct / 100.0
        m15_tf = m15_pct / 100.0
        group_out = {}
        for sym, pc in precomp.items():
            try:
                s = _find_signals_v10(pc, lrc, sigma, htf_tf, m15_tf, min_score, swing_wing)
                if s:
                    group_out[sym] = s
            except Exception:
                pass
        raw_signals[sig_key] = group_out
        if i % 12 == 0 or i == len(sig_keys):
            log.info(f"  Phase A {i}/{len(sig_keys)} groups")
    n_raw = sum(len(s) for d in raw_signals.values() for s in d.values())
    log.info(f"Phase A done in {(datetime.now()-t_a).seconds}s -- {n_raw:,} raw signals")

    for pc in precomp.values():
        pc.pop("m15_by_date", None)
        pc.pop("w_levels", None)

    if n_raw == 0:
        log.warning("No signals found. Check W1 and M15 caches.")
        return

    # ── Phase B ──
    exec_keys   = list(EXEC_GRID.keys())
    exec_combos = []
    for sk in sig_keys:
        lrc, sigma, htf, m15_t, min_score, swing_wing = sk
        for combo in product(*[EXEC_GRID[k] for k in exec_keys]):
            d = dict(zip(exec_keys, combo))
            d.update(lrc_length=lrc, sigma_mult=sigma,
                     htf_touch_pct=htf, m15_touch_pct=m15_t,
                     min_htf_score=min_score, swing_wing=swing_wing)
            exec_combos.append(d)

    n_portfolio = len(MC_VALUES) * len(uni_sets)
    log.info(f"Phase B: {len(exec_combos)} exec combos x {n_portfolio} portfolio "
             f"variants = {len(exec_combos)*n_portfolio:,} result rows")

    _init_b(precomp, raw_signals, uni_sets)
    t_b = datetime.now()
    flat = []
    for i, combo in enumerate(exec_combos, 1):
        flat.extend(_run_exec_combo(combo))
        if i % 100 == 0 or i == len(exec_combos):
            log.info(f"  Phase B {i}/{len(exec_combos)} combos")
    log.info(f"Phase B done in {(datetime.now()-t_b).seconds}s -- {len(flat):,} rows")

    # ── Save sweep results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M")
    df_res = pd.DataFrame(flat).sort_values("sharpe_r", ascending=False)

    full_path = RESULTS_DIR / f"v10_sweep_{ts}.csv"
    df_res.to_csv(full_path, index=False)
    log.info(f"Full sweep -> {full_path}  ({len(df_res):,} rows)")

    top_path = RESULTS_DIR / f"v10_top_{ts}.csv"
    df_res.head(200).to_csv(top_path, index=False)
    log.info(f"Top 200    -> {top_path}")

    best = df_res.iloc[0].to_dict()
    log.info(f"Computing per-symbol stats for best combo "
             f"(Sharpe={best['sharpe_r']}, WR={best['win_rate']}%)...")
    sym_stats, trade_log = compute_symbol_stats(best, precomp, raw_signals, uni_sets)

    if not sym_stats.empty:
        sym_path = RESULTS_DIR / f"v10_symbol_stats_{ts}.csv"
        sym_stats.to_csv(sym_path, index=False)
        log.info(f"Symbol stats -> {sym_path}")

    monthly = pd.DataFrame()
    if not trade_log.empty:
        tlog_path = RESULTS_DIR / f"v10_trade_log_{ts}.csv"
        trade_log.to_csv(tlog_path, index=False)
        log.info(f"Trade log    -> {tlog_path}")

        trade_log["month"] = pd.to_datetime(trade_log["entry_date"]).dt.to_period("M")
        monthly = (trade_log.groupby("month")
                   .agg(n_trades=("r", "count"),
                        win_rate=("r", lambda x: round((x > 0).mean() * 100, 1)),
                        total_r=("r", "sum"),
                        total_pnl=("pnl", "sum"))
                   .round(3).reset_index())
        monthly["month"] = monthly["month"].astype(str)
        mon_path = RESULTS_DIR / f"v10_monthly_{ts}.csv"
        monthly.to_csv(mon_path, index=False)
        log.info(f"Monthly PnL  -> {mon_path}")

    # ── Phase C: Symbol Discovery ──
    log.info("Phase C: symbol discovery across all precomputed symbols...")
    t_c = datetime.now()
    disc = discover_symbols(best, precomp, raw_signals, sym_meta)
    log.info(f"Phase C done in {(datetime.now()-t_c).seconds}s -- "
             f"{len(disc)} symbols with signals")

    if not disc.empty:
        disc_path = RESULTS_DIR / f"v10_symbol_discovery_{ts}.csv"
        disc.to_csv(disc_path, index=False)
        log.info(f"Symbol discovery -> {disc_path}")

        live = disc[
            (disc["n_trades"]  >= DISC_MIN_TRADES) &
            (disc["win_rate"]  >= DISC_MIN_WR) &
            (disc["sharpe_r"]  >= DISC_MIN_SHARPE)
        ].copy()
        live_path = RESULTS_DIR / f"v10_live_universe_{ts}.csv"
        live.to_csv(live_path, index=False)
        log.info(f"Live universe    -> {live_path}  ({len(live)} symbols)")

    elapsed = (datetime.now() - t0).total_seconds()
    log.info(f"Total time: {elapsed:.0f}s")

    # ── Print results ──
    cols_main = ["universe", "lrc_length", "sigma_mult",
                 "htf_touch_pct", "m15_touch_pct", "min_htf_score", "swing_wing",
                 "sl_atr_mult", "tp_rr", "max_concurrent",
                 "n_trades", "win_rate", "avg_r",
                 "total_pnl_usd", "max_dd_usd", "sharpe_r",
                 "avg_mfe_r", "avg_mae_r", "pct_mfe_gt_1r", "pct_mae_lt_half"]

    print("\n--- Top 20 combos by Sharpe (R) ---")
    print(df_res[[c for c in cols_main if c in df_res.columns]].head(20).to_string(index=False))

    if not sym_stats.empty:
        print(f"\n--- Top 15 symbols (best combo: {best['universe']} | "
              f"LRC-{int(best['lrc_length'])} | sigma={best['sigma_mult']} | "
              f"wing={int(best['swing_wing'])} | score={int(best['min_htf_score'])}) ---")
        print(sym_stats.head(15).to_string(index=False))

    if not trade_log.empty:
        print("\n--- Direction split ---")
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

    if not disc.empty:
        print(f"\n--- Phase C: Top 40 symbols by Sharpe (all universes, best params) ---")
        show_cols = ["sym", "sector", "exchange", "n_trades", "win_rate",
                     "avg_r", "total_pnl", "sharpe_r", "n_long", "n_short",
                     "long_wr", "short_wr"]
        print(disc[[c for c in show_cols if c in disc.columns]].head(40).to_string(index=False))

        if not live.empty:
            print(f"\n--- Recommended live universe "
                  f"(n>={DISC_MIN_TRADES}, WR>={DISC_MIN_WR}%, "
                  f"Sharpe>={DISC_MIN_SHARPE}) ---")
            print(live[[c for c in show_cols if c in live.columns]].to_string(index=False))
            syms_list = ", ".join(live["sym"].tolist())
            print(f"\nSymbols ({len(live)}): {syms_list}")

    print("\n--- Param contribution (avg Sharpe by lrc x sigma x min_score x wing) ---")
    grp_cols = ["lrc_length", "sigma_mult", "min_htf_score", "swing_wing",
                "sharpe_r", "win_rate", "n_trades"]
    avail = [cc for cc in grp_cols if cc in df_res.columns]
    param_grp = (df_res[avail]
                 .groupby(["lrc_length", "sigma_mult", "min_htf_score", "swing_wing"])
                 .agg(sharpe_r=("sharpe_r", "mean"),
                      win_rate=("win_rate", "mean"),
                      n_trades=("n_trades", "mean"))
                 .round(2).reset_index()
                 .sort_values("sharpe_r", ascending=False))
    print(param_grp.to_string(index=False))


if __name__ == "__main__":
    main()
