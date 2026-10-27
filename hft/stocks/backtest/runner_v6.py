"""
V6 backtest sweep: HTF weekly levels + M15 intraday entry.

What's new vs v5:
  - HTF level detection: weekly bar S/R zones replace 120-day daily levels.
    Zones scored by touch count (5+ touches = 100 pts).
  - M15 intraday entry: after LRC extension fires on daily, walk the NEXT
    session's M15 bars; enter at close of first reversal bar testing the
    weekly level. Replicates "wait for M15 to confirm the level holds".
  - Entry price = M15 close of confirmation bar; exit via daily SL/TP.
  - Dollar PnL uses actual risk (abs(entry - SL)), not just ATR buffer.

Usage:
    python -m stocks.backtest.runner_v6 --skip-extend --fetch-w1 --fetch-m15
    python -m stocks.backtest.runner_v6 --skip-extend          # skip all fetches

Output (RESULTS_DIR):
    v6_sweep_{ts}.csv          all result rows
    v6_top_{ts}.csv            top 200 by sharpe_r
    v6_symbol_stats_{ts}.csv   per-symbol breakdown for best combo
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
    "lrc_length":    [80, 120],
    "sigma_mult":    [1.0, 1.5],
    "htf_touch_pct": [1.0, 2.0],    # % proximity to weekly level for detection
    "m15_touch_pct": [0.3, 0.5, 1.0],  # % proximity for M15 entry trigger
}

EXEC_GRID = {
    "sl_atr_mult":  [0.3, 0.5, 1.0],
    "tp_rr":        [1.0, 1.5, 2.0, 3.0],
    "breakeven_rr": [0.0, 0.5],
}

MC_VALUES = [5, 10, 20, 0]   # 0 = unlimited


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
        "US-All":   top_q(us_all, 0.75),
        "US-Top10": top_q(us_all, 0.90),
        "EU-All":   top_q(eu_all, 0.75),
    }

    us_top25 = uni["US-All"]
    for sec in ["Technology", "Healthcare", "Financials", "Energy"]:
        sec_syms = {s for s in us_top25 if sym_meta.get(s, {}).get("sector") == sec}
        key = f"US-{sec[:3]}"
        if len(sec_syms) >= 5:
            uni[key] = frozenset(sec_syms)

    active = {k: v for k, v in uni.items() if len(v) >= 5}
    log.info("Universe sets: " + ", ".join(f"{k}={len(v)}" for k, v in active.items()))
    return active


# ── Precompute ────────────────────────────────────────────────────────────────

def precompute_symbol(df: pd.DataFrame, m15_df: pd.DataFrame,
                      w1_df: pd.DataFrame,
                      lrc_lengths, sigma_mults, htf_touch_pcts) -> dict:
    c = df["close"].values; h = df["high"].values
    l = df["low"].values;   o = df["open"].values

    lrc_data = {}
    for ll in lrc_lengths:
        for sm in sigma_mults:
            mid, upper, lower, _ = _lrc_vectorized(c, ll, sm)
            lrc_data[(ll, sm)] = (mid, upper, lower)

    # Weekly levels — precompute for all htf_touch_pct fractions at once
    touch_fracs = [p / 100.0 for p in htf_touch_pcts]
    if w1_df is not None and not w1_df.empty and len(w1_df) >= 4:
        w1_cols = {col.lower(): col for col in w1_df.columns}
        w_h = w1_df[w1_cols.get("high", "High")].values
        w_l = w1_df[w1_cols.get("low",  "Low")].values
        w_levels = precompute_weekly_levels(w_h, w_l, touch_fracs)
    else:
        w_levels = {tf: [] for tf in touch_fracs}

    # M15 by date: all bars per trading day (used for intraday entry)
    m15_by_date = {}
    if m15_df is not None and not m15_df.empty:
        m15 = m15_df.copy()
        m15.columns = [cc.lower() for cc in m15.columns]
        m15["_date"] = m15.index.normalize()
        for d, grp in m15.groupby("_date"):
            bars = list(zip(grp["high"].values, grp["low"].values,
                            grp["close"].values, grp["open"].values))
            m15_by_date[d] = bars

    return {
        "close": c, "high": h, "low": l, "open": o,
        "n": len(c), "index": df.index,
        "lrc_c":    _lrc_candle_close_vectorized(c, LRC_CL, LRC_CS),
        "atr":      _atr_vectorized(h, l, c, ATR_P),
        "lrc_data": lrc_data,
        "w_levels":    w_levels,      # {touch_frac: [(level_px, score)...]}
        "m15_by_date": m15_by_date,   # {date: [(h,l,c,o)...]}
    }


# ── Phase A: signal finding ───────────────────────────────────────────────────

def _find_signals_v6(pc: dict, lrc_length: int, sigma_mult: float,
                     htf_tf: float, m15_tf: float) -> list:
    """
    For each daily bar where the LRC candle is outside the sigma band:
      1. Check if a weekly level exists within htf_tf% of closing price.
      2. Walk the next session's M15 bars to find a reversal at the level.
      3. Entry = M15 confirmation bar close.

    Returns list of (signal_bar_i, entry_bar_i, entry_px, direction,
                     level_px, atr_v, htf_score)
    """
    n         = pc["n"]
    lrc_c     = pc["lrc_c"]
    atr       = pc["atr"]
    h         = pc["high"]; l = pc["low"]; c = pc["close"]
    idx       = pc["index"]
    w_levels  = pc["w_levels"].get(htf_tf, [])
    m15_by_dt = pc["m15_by_date"]

    if not w_levels:
        return []

    mid, upper, lower = pc["lrc_data"][(lrc_length, sigma_mult)]
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

        # HTF weekly level check
        lv, score = find_htf_level(w_levels, price, htf_tf)
        if np.isnan(lv):
            continue

        # M15 entry in next session
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
    lrc   = params["lrc_length"];  sigma = params["sigma_mult"]
    htf   = params["htf_touch_pct"]; m15_t = params["m15_touch_pct"]
    sl    = params["sl_atr_mult"];  tp_r  = params["tp_rr"]
    be    = params["breakeven_rr"]

    raw = _RS_B.get((lrc, sigma, htf, m15_t), {})

    candidates = []
    for sym, sigs in raw.items():
        pc = _PC_B.get(sym)
        if pc is None:
            continue
        h = pc["high"]; l = pc["low"]; c = pc["close"]
        n = pc["n"];     idx = pc["index"]

        for (bi, eb, ep, dir_, lv, atr_v, readiness) in sigs:
            # SL: level_px ± sl_atr_mult*ATR (behind the weekly level)
            pos   = +1 if dir_ == "SHORT" else -1
            sl_px = lv + pos * sl * atr_v
            if pos * (sl_px - ep) <= 0:   # SL must be on wrong side of entry
                continue
            risk = abs(ep - sl_px)
            if risk < 1e-6:
                continue
            tp_px = ep - pos * tp_r * risk

            xb, xp, xtype, mfe_r, mae_r = _find_exit(
                h, l, c, eb, ep, sl_px, tp_px, dir_, be, risk, n)

            r = (xp - ep) / risk if dir_ == "LONG" else (ep - xp) / risk
            # dollar_pnl uses actual risk per share (entry to SL)
            dollar_pnl = r * (EFFECTIVE / ep) * risk

            candidates.append({
                "sym":        sym,
                "readiness":  readiness,
                "entry_date": idx[eb],
                "exit_date":  idx[xb],
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

    pct_mfe_gt_1r   = round(float((mfe >= 1.0).mean() * 100), 1)
    pct_mae_lt_half = round(float((mae > -0.5).mean() * 100), 1)
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


# ── Per-symbol stats for best combo ──────────────────────────────────────────

def compute_symbol_stats(best_row: dict, precomp: dict, raw_signals: dict,
                         uni_sets: dict) -> pd.DataFrame:
    lrc   = best_row["lrc_length"];  sigma = best_row["sigma_mult"]
    htf   = best_row["htf_touch_pct"]; m15_t = best_row["m15_touch_pct"]
    sl    = best_row["sl_atr_mult"];  tp_r  = best_row["tp_rr"]
    be    = best_row["breakeven_rr"]
    universe = best_row["universe"]
    mc       = int(best_row["max_concurrent"])

    raw      = raw_signals.get((lrc, sigma, htf, m15_t), {})
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
        return pd.DataFrame()

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

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extend", action="store_true",
                    help="Skip daily cache extension")
    ap.add_argument("--fetch-m15", action="store_true",
                    help="Pull M15 bars from MT5 before sweep")
    ap.add_argument("--fetch-w1", action="store_true",
                    help="Pull weekly bars from MT5 before sweep")
    ap.add_argument("--workers", type=int,
                    default=max(1, int(cpu_count() * 0.4)))
    args = ap.parse_args()

    workers = args.workers
    t0 = datetime.now()
    log.info(f"v6 sweep starting | workers={workers}")

    # ── Daily cache extension ──
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

    # ── W1 cache fetch ──
    if args.fetch_w1:
        log.info("Fetching weekly bars from MT5...")
        try:
            from stocks.backtest.data.extender import extend_cache_w1
            univ_ = build_universe()
            syms_ = univ_[univ_["exchange"].isin(ALL_EXCH)]["symbol"].tolist()
            extend_cache_w1(syms_)
        except Exception as e:
            log.warning(f"W1 fetch failed: {e}")

    # ── M15 cache fetch ──
    if args.fetch_m15:
        log.info("Fetching M15 bars from MT5...")
        try:
            from stocks.backtest.data.extender import extend_cache_m15
            univ_ = build_universe()
            syms_ = univ_[univ_["exchange"].isin(ALL_EXCH)]["symbol"].tolist()
            extend_cache_m15(syms_)
        except Exception as e:
            log.warning(f"M15 fetch failed: {e}")

    # ── Availability checks ──
    w1_available  = (BT_CACHE_W1_DIR.exists()
                     and any(BT_CACHE_W1_DIR.glob("*.parquet")))
    m15_available = (BT_CACHE_M15_DIR.exists()
                     and any(BT_CACHE_M15_DIR.glob("*.parquet")))
    log.info(f"W1 data:  {'available' if w1_available else 'MISSING — weekly levels disabled'}")
    log.info(f"M15 data: {'available' if m15_available else 'MISSING — M15 entry disabled'}")

    # ── Load daily bars ──
    log.info("Loading daily bars...")
    univ = build_universe()
    bars, dvols, sym_meta = load_all_bars(univ)
    log.info(f"Bars loaded: {len(bars)} symbols")

    uni_sets = build_universe_sets(bars, dvols, sym_meta)
    all_syms = set().union(*uni_sets.values())
    bars = {s: bars[s] for s in bars if s in all_syms}
    log.info(f"Symbols in universe: {len(bars)}")

    # ── Precompute ──
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
                df, m15_df, w1_df, lrc_lengths, sigma_mults, htf_touch_pcts)
        except Exception as e:
            log.debug(f"  {sym} precompute failed: {e}")

    log.info(f"Precomputed {len(precomp)} symbols "
             f"({w1_loaded} with W1, {m15_loaded} with M15) "
             f"in {(datetime.now()-t0).seconds}s")

    # ── Phase A (single-threaded — precomp with M15 dicts is too large to pickle) ──
    sig_keys = list(product(
        lrc_lengths, sigma_mults,
        SIGNAL_GRID["htf_touch_pct"],
        SIGNAL_GRID["m15_touch_pct"],
    ))
    log.info(f"Phase A: {len(sig_keys)} signal groups x {len(precomp)} symbols "
             f"(single-threaded)...")
    t_a = datetime.now()
    raw_signals = {}
    for i, sig_key in enumerate(sig_keys, 1):
        lrc, sigma, htf_pct, m15_pct = sig_key
        htf_tf = htf_pct / 100.0
        m15_tf = m15_pct / 100.0
        group_out = {}
        for sym, pc in precomp.items():
            try:
                s = _find_signals_v6(pc, lrc, sigma, htf_tf, m15_tf)
                if s:
                    group_out[sym] = s
            except Exception:
                pass
        raw_signals[sig_key] = group_out
        if i % 6 == 0 or i == len(sig_keys):
            log.info(f"  Phase A {i}/{len(sig_keys)} groups")
    n_raw = sum(len(s) for d in raw_signals.values() for s in d.values())
    log.info(f"Phase A done in {(datetime.now()-t_a).seconds}s -- {n_raw:,} raw signals")

    # Strip m15_by_date and w_levels from precomp — not needed for exit simulation
    # and would cause MemoryError when pickling for Phase B workers
    for pc in precomp.values():
        pc.pop("m15_by_date", None)
        pc.pop("w_levels", None)

    if n_raw == 0:
        log.warning("No signals found. Check that W1 and M15 caches are populated "
                    "(run with --fetch-w1 --fetch-m15).")
        return

    # ── Phase B ──
    exec_keys   = list(EXEC_GRID.keys())
    exec_combos = []
    for sk in sig_keys:
        lrc, sigma, htf, m15_t = sk
        for combo in product(*[EXEC_GRID[k] for k in exec_keys]):
            d = dict(zip(exec_keys, combo))
            d.update(lrc_length=lrc, sigma_mult=sigma,
                     htf_touch_pct=htf, m15_touch_pct=m15_t)
            exec_combos.append(d)

    n_portfolio = len(MC_VALUES) * len(uni_sets)
    log.info(f"Phase B: {len(exec_combos)} exec combos x {n_portfolio} portfolio "
             f"variants = {len(exec_combos)*n_portfolio:,} result rows")

    # Phase B single-threaded — raw_signals (1M+ entries) too large to pickle
    # to multiple workers simultaneously on Windows
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

    full_path = RESULTS_DIR / f"v6_sweep_{ts}.csv"
    df_res.to_csv(full_path, index=False)
    log.info(f"Full sweep -> {full_path}  ({len(df_res):,} rows)")

    top_path = RESULTS_DIR / f"v6_top_{ts}.csv"
    df_res.head(200).to_csv(top_path, index=False)
    log.info(f"Top 200    -> {top_path}")

    # ── Per-symbol stats for best combo ──
    best = df_res.iloc[0].to_dict()
    log.info(f"Computing per-symbol stats for best combo "
             f"(Sharpe={best['sharpe_r']}, WR={best['win_rate']}%)...")
    sym_stats = compute_symbol_stats(best, precomp, raw_signals, uni_sets)
    if not sym_stats.empty:
        sym_path = RESULTS_DIR / f"v6_symbol_stats_{ts}.csv"
        sym_stats.to_csv(sym_path, index=False)
        log.info(f"Symbol stats -> {sym_path}")

    elapsed = (datetime.now() - t0).total_seconds()
    log.info(f"Total time: {elapsed:.0f}s")

    # ── Print results ──
    cols_main = ["universe", "lrc_length", "sigma_mult",
                 "htf_touch_pct", "m15_touch_pct",
                 "sl_atr_mult", "tp_rr", "breakeven_rr", "max_concurrent",
                 "n_trades", "win_rate", "avg_r",
                 "total_pnl_usd", "max_dd_usd", "sharpe_r", "sharpe_usd",
                 "avg_mfe_r", "avg_mae_r", "pct_mfe_gt_1r", "pct_mae_lt_half"]

    print("\n--- Top 20 combos by Sharpe (R) ---")
    print(df_res[[c for c in cols_main if c in df_res.columns]].head(20).to_string(index=False))

    if not sym_stats.empty:
        print(f"\n--- Top 15 symbols (best combo: {best['universe']} | "
              f"LRC-{int(best['lrc_length'])} | sigma={best['sigma_mult']} | "
              f"htf={best['htf_touch_pct']}% | m15={best['m15_touch_pct']}%) ---")
        print(sym_stats.head(15).to_string(index=False))

    # ── HTF/M15 param contribution summary ──
    print("\n--- Param contribution (avg Sharpe by htf_touch_pct x m15_touch_pct) ---")
    grp_cols = ["htf_touch_pct", "m15_touch_pct", "sharpe_r", "win_rate", "n_trades"]
    avail = [cc for cc in grp_cols if cc in df_res.columns]
    param_grp = (df_res[avail]
                 .groupby(["htf_touch_pct", "m15_touch_pct"])
                 .agg(sharpe_r=("sharpe_r", "mean"),
                      win_rate=("win_rate", "mean"),
                      n_trades=("n_trades", "mean"))
                 .round(2).reset_index()
                 .sort_values("sharpe_r", ascending=False))
    print(param_grp.to_string(index=False))


if __name__ == "__main__":
    main()
