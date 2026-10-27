"""
Comprehensive v5 backtest sweep.

What's new vs v4:
  - M15 confirmation gate: last 8 M15 bars of each session must show velocity
    confirming the reversal (bearish at resistance, bullish at support).
  - Component sweep: use_level (on/off) x vel_min (0,1,2,3) x use_m15 (on/off)
    to isolate which strategy elements actually add edge.
  - TP range 1.0-3.0 (focus on high win-rate regime, not 10RR lottery tickets).
  - MFE/MAE per trade: avg max-favorable / max-adverse excursion in R.
  - Per-symbol breakdown for the best combo: top 10-15 stocks.

Usage:
    python -m stocks.backtest.runner_v5 --skip-extend           # skip daily cache
    python -m stocks.backtest.runner_v5 --fetch-m15             # pull M15 data first
    python -m stocks.backtest.runner_v5 --skip-extend --no-m15  # daily-only mode

Output (RESULTS_DIR):
    v5_sweep_{ts}.csv          all result rows
    v5_top_{ts}.csv            top 200 by sharpe_r
    v5_symbol_stats_{ts}.csv   per-symbol breakdown for best combo
"""
import sys, warnings, logging, argparse, math
from itertools import product
from multiprocessing import Pool, cpu_count
from datetime import datetime
from pathlib import Path

sys.path.insert(0, r'path/to/repo/hft')
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from stocks.backtest.config import RESULTS_DIR, BT_CACHE_M15_DIR
from stocks.backtest.data.extender import load_bt_bars, load_bt_bars_m15
from stocks.backtest.data.universe import build_universe
from stocks.backtest.signals.wt_lrc_signal import (
    _lrc_vectorized, _lrc_candle_close_vectorized, _atr_vectorized, _scored_level_at,
)
from stocks.backtest.signals.lrc_v5_signal import (
    vel_component_arrays, m15_check, find_signals_v5,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
US_EXCH  = {"NYSE", "NAS"}
EU_EXCH  = {"LSE", "ETR", "PAR", "IT", "MAD", "AMS", "IE", "SWX"}
ALL_EXCH = US_EXCH | EU_EXCH
MIN_BARS = 200
LRC_CL   = 15; LRC_CS = 9; ATR_P = 14; VEL_LB = 5
M15_CONF_BARS = 8       # last N M15 bars of each day used for confirmation

LEVERAGE  = 20
CAPITAL   = 1_000.0
EFFECTIVE = CAPITAL * LEVERAGE   # $20,000

# ── Sweep grids ───────────────────────────────────────────────────────────────
SIGNAL_GRID = {
    "lrc_length": [80, 120],
    "sigma_mult": [1.0, 1.5],
    "touch_pct":  [0.30, 0.50],   # only relevant when use_level=True
}

COMPONENT_GRID = {
    "use_level": [True, False],
    "vel_min":   [0, 1, 2, 3],    # 0=none 1=any 2=two-of-3 3=all
    "use_m15":   [True, False],   # resolved at runtime against M15 availability
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


# ── M15 precompute ────────────────────────────────────────────────────────────

def _precompute_m15_gates(daily_index, m15_df: pd.DataFrame,
                          n_bars: int = M15_CONF_BARS):
    """
    Align M15 bars with daily bars and precompute SHORT/LONG confirmation bool.
    Returns (conf_short, conf_long): np.ndarray[bool] aligned with daily_index.
    """
    n = len(daily_index)
    conf_short = np.zeros(n, dtype=bool)
    conf_long  = np.zeros(n, dtype=bool)

    if m15_df is None or m15_df.empty:
        return conf_short, conf_long

    try:
        m15 = m15_df.copy()
        m15.columns = [c.lower() for c in m15.columns]
        m15["_date"] = m15.index.floor("D")

        # Build date -> last-n_bars arrays
        by_date = {}
        for d, grp in m15.groupby("_date"):
            if len(grp) < n_bars:
                continue
            last = grp.iloc[-n_bars:]
            by_date[d] = (last["high"].values, last["low"].values,
                          last["close"].values, last["open"].values)

        for i, dt in enumerate(daily_index):
            key = pd.Timestamp(dt).floor("D")
            if key in by_date:
                h, l, c, o = by_date[key]
                conf_short[i], conf_long[i] = m15_check(h, l, c, o)
    except Exception:
        pass

    return conf_short, conf_long


# ── Indicator precompute ──────────────────────────────────────────────────────

def precompute_symbol(df: pd.DataFrame, m15_df: pd.DataFrame,
                      lrc_lengths, sigma_mults) -> dict:
    c = df["close"].values; h = df["high"].values
    l = df["low"].values;   o = df["open"].values

    lrc_data = {}
    for ll in lrc_lengths:
        for sm in sigma_mults:
            mid, upper, lower, _ = _lrc_vectorized(c, ll, sm)
            lrc_data[(ll, sm)] = (mid, upper, lower)

    fad, be, bu, uw, lw = vel_component_arrays(h, l, c, o, VEL_LB)
    m15_cs, m15_cl = _precompute_m15_gates(df.index, m15_df)

    return {
        "close": c, "high": h, "low": l, "open": o,
        "n":     len(c), "index": df.index,
        "lrc_c": _lrc_candle_close_vectorized(c, LRC_CL, LRC_CS),
        "atr":   _atr_vectorized(h, l, c, ATR_P),
        "lrc_data":      lrc_data,
        "vel_fading":    fad,
        "vel_bearish":   be,
        "vel_bullish":   bu,
        "vel_upper_wick": uw,
        "vel_lower_wick": lw,
        "m15_conf_short": m15_cs,
        "m15_conf_long":  m15_cl,
    }


# ── Phase A worker ────────────────────────────────────────────────────────────
_PC_A: dict = {}

def _init_a(precomp):
    global _PC_A
    _PC_A = precomp


def _run_sig_group(sig_key: tuple) -> tuple:
    """
    sig_key = (lrc_length, sigma_mult, touch_pct, use_level, vel_min, use_m15)
    Returns (sig_key, {sym: [signal_tuples]})
    """
    lrc, sigma, touch, use_level, vel_min, use_m15 = sig_key
    out = {}
    for sym, pc in _PC_A.items():
        try:
            s = find_signals_v5(pc, lrc, sigma, touch,
                                use_level=use_level, vel_min=vel_min,
                                use_m15=use_m15)
            if s:
                out[sym] = s
        except Exception:
            pass
    return sig_key, out


# ── Exit simulation with MFE/MAE ─────────────────────────────────────────────

def _find_exit(h, l, c, eb, ep, sl_init, tp_px, direction,
               breakeven_rr, risk, n):
    """
    Walk-forward exit. Returns (exit_bar, exit_price, exit_type, mfe_r, mae_r).
    mfe_r: max favorable excursion in R multiples (always >= 0)
    mae_r: max adverse excursion in R multiples (always <= 0)
    """
    sl    = sl_init
    mfe_r = 0.0
    mae_r = 0.0

    for i in range(eb, n):
        hi = h[i]; lo = l[i]

        # Track excursion before applying SL/TP
        if direction == "LONG":
            fav = (hi - ep) / risk
            adv = (lo - ep) / risk
        else:
            fav = (ep - lo) / risk
            adv = (ep - hi) / risk
        if fav > mfe_r: mfe_r = fav
        if adv < mae_r: mae_r = adv

        # Breakeven move
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

    return n-1, float(c[n-1]), "END", round(mfe_r, 3), round(mae_r, 3)


# ── Phase B worker ────────────────────────────────────────────────────────────
_PC_B:  dict = {}
_RS_B:  dict = {}
_UNI_B: dict = {}

def _init_b(precomp, raw_signals, uni_syms):
    global _PC_B, _RS_B, _UNI_B
    _PC_B = precomp; _RS_B = raw_signals; _UNI_B = uni_syms


def _run_exec_combo(params: dict) -> list:
    """
    Worker B: one exec combo -> all (mc x universe) portfolio variants.
    Also accumulates per-trade MFE/MAE for aggregate metrics.
    """
    lrc       = params["lrc_length"];  sigma = params["sigma_mult"]
    touch     = params["touch_pct"];   sl    = params["sl_atr_mult"]
    tp_r      = params["tp_rr"];       be    = params["breakeven_rr"]
    use_level = params["use_level"];   vel_min = params["vel_min"]
    use_m15   = params["use_m15"]

    raw = _RS_B.get((lrc, sigma, touch, use_level, vel_min, use_m15), {})

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
            shares     = EFFECTIVE / ep
            dollar_pnl = r * shares * sl * atr_v

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
    if not all_r:
        return dict(
            n_trades=0, win_rate=0, avg_r=0, total_r=0,
            profit_factor=0, max_dd_r=0, sharpe_r=0,
            total_pnl_usd=0, max_dd_usd=0, sharpe_usd=0,
            avg_mfe_r=0, avg_mae_r=0,
            pct_mfe_gt_tp=0, pct_mae_lt_half=0,
        )
    r   = np.array(all_r,   dtype=float)
    pnl = np.array(all_pnl, dtype=float)
    mfe = np.array(all_mfe, dtype=float)
    mae = np.array(all_mae, dtype=float)

    wins   = r > 0; losses = ~wins
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

    # MFE/MAE insights
    # pct_mfe_gt_1r: % of trades where price moved at least 1R in our favour
    # pct_mae_lt_half: % of trades with max adverse < 0.5R (clean entries, barely tested SL)
    # pct_loss_had_mfe: % of losing trades that were profitable at some point (near-misses)
    pct_mfe_gt_1r    = round(float((mfe >= 1.0).mean() * 100), 1) if len(mfe) > 0 else 0.0
    pct_mae_lt_half  = round(float((mae > -0.5).mean() * 100), 1) if len(mae) > 0 else 0.0
    if losses.any():
        pct_loss_had_mfe = round(float((mfe[losses] > 0.0).mean() * 100), 1)
    else:
        pct_loss_had_mfe = 0.0

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
    """Re-run the best combo and collect per-symbol trade breakdown."""
    lrc   = best_row["lrc_length"];  sigma = best_row["sigma_mult"]
    touch = best_row["touch_pct"];   sl    = best_row["sl_atr_mult"]
    tp_r  = best_row["tp_rr"];       be    = best_row["breakeven_rr"]
    use_level = best_row["use_level"]
    vel_min   = best_row["vel_min"]
    use_m15   = best_row["use_m15"]
    universe  = best_row["universe"]
    mc        = int(best_row["max_concurrent"])

    raw      = raw_signals.get((lrc, sigma, touch, use_level, vel_min, use_m15), {})
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
            shares     = EFFECTIVE / ep
            dollar_pnl = r * shares * sl * atr_v
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
        "n_trades":    grp["r"].count(),
        "win_rate":    grp["r"].apply(lambda x: round((x > 0).mean() * 100, 1)),
        "avg_r":       grp["r"].mean().round(3),
        "total_r":     grp["r"].sum().round(2),
        "total_pnl":   grp["pnl"].sum().round(0),
        "avg_mfe_r":   grp["mfe_r"].mean().round(3),
        "avg_mae_r":   grp["mae_r"].mean().round(3),
        "pct_tp":      grp["exit_type"].apply(lambda x: round((x == "TP").mean() * 100, 1)),
        "pct_sl":      grp["exit_type"].apply(lambda x: round((x == "SL").mean() * 100, 1)),
    }).reset_index().sort_values("total_pnl", ascending=False)

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extend", action="store_true",
                    help="Skip daily cache extension")
    ap.add_argument("--fetch-m15", action="store_true",
                    help="Fetch M15 bars from MT5 before sweep")
    ap.add_argument("--no-m15", action="store_true",
                    help="Disable M15 gate entirely (faster, daily-only)")
    ap.add_argument("--workers", type=int,
                    default=max(1, int(cpu_count() * 0.4)))
    args = ap.parse_args()

    workers = args.workers
    t0 = datetime.now()
    log.info(f"v5 sweep starting | workers={workers}")

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

    # ── M15 cache fetch ──
    if args.fetch_m15 and not args.no_m15:
        log.info("Fetching M15 bars from MT5...")
        try:
            from stocks.backtest.data.extender import extend_cache_m15
            univ_ = build_universe()
            syms_ = univ_[univ_["exchange"].isin(ALL_EXCH)]["symbol"].tolist()
            extend_cache_m15(syms_)
        except Exception as e:
            log.warning(f"M15 fetch failed: {e}")

    # ── Check M15 availability ──
    m15_available = (not args.no_m15
                     and BT_CACHE_M15_DIR.exists()
                     and any(BT_CACHE_M15_DIR.glob("*.parquet")))
    log.info(f"M15 data: {'available' if m15_available else 'not available (daily-only mode)'}")

    # ── Load bars ──
    log.info("Loading daily bars...")
    univ = build_universe()
    bars, dvols, sym_meta = load_all_bars(univ)
    log.info(f"Bars loaded: {len(bars)} symbols")

    uni_sets = build_universe_sets(bars, dvols, sym_meta)
    all_syms = set().union(*uni_sets.values())
    bars = {s: bars[s] for s in bars if s in all_syms}
    log.info(f"Symbols in universe: {len(bars)}")

    # ── Precompute ──
    lrc_lengths = SIGNAL_GRID["lrc_length"]
    sigma_mults = SIGNAL_GRID["sigma_mult"]
    log.info(f"Precomputing indicators for {len(bars)} symbols"
             f"{' + M15 gates' if m15_available else ''}...")

    precomp = {}
    m15_loaded = 0
    for sym, df in bars.items():
        try:
            m15_df = load_bt_bars_m15(sym) if m15_available else pd.DataFrame()
            if not m15_df.empty:
                m15_loaded += 1
            precomp[sym] = precompute_symbol(df, m15_df, lrc_lengths, sigma_mults)
        except Exception as e:
            log.debug(f"  {sym} precompute failed: {e}")

    if m15_available:
        log.info(f"Precomputed {len(precomp)} symbols "
                 f"({m15_loaded} with M15 data) in {(datetime.now()-t0).seconds}s")
    else:
        log.info(f"Precomputed {len(precomp)} symbols in {(datetime.now()-t0).seconds}s")

    # ── Build Phase A keys ──
    use_m15_vals = [True, False] if m15_available else [False]

    sig_keys = []
    for lrc, sigma, use_m15 in product(lrc_lengths, sigma_mults, use_m15_vals):
        for vel_min in COMPONENT_GRID["vel_min"]:
            for use_level in COMPONENT_GRID["use_level"]:
                if use_level:
                    for touch in SIGNAL_GRID["touch_pct"]:
                        sig_keys.append((lrc, sigma, touch, use_level, vel_min, use_m15))
                else:
                    # touch_pct irrelevant when no level; use 0.50 as canonical
                    sig_keys.append((lrc, sigma, 0.50, use_level, vel_min, use_m15))

    sig_keys = list(dict.fromkeys(sig_keys))   # deduplicate
    log.info(f"Phase A: {len(sig_keys)} signal groups x {len(precomp)} symbols...")
    t_a = datetime.now()
    with Pool(workers, initializer=_init_a, initargs=(precomp,)) as pool:
        a_results = pool.map(_run_sig_group, sig_keys)
    raw_signals = {key: syms for key, syms in a_results}
    n_raw = sum(len(s) for d in raw_signals.values() for s in d.values())
    log.info(f"Phase A done in {(datetime.now()-t_a).seconds}s -- {n_raw:,} raw signals")

    # ── Build Phase B exec combos ──
    exec_keys   = list(EXEC_GRID.keys())
    exec_combos = []
    for sk in sig_keys:
        lrc, sigma, touch, use_level, vel_min, use_m15 = sk
        for combo in product(*[EXEC_GRID[k] for k in exec_keys]):
            d = dict(zip(exec_keys, combo))
            d.update(lrc_length=lrc, sigma_mult=sigma, touch_pct=touch,
                     use_level=use_level, vel_min=vel_min, use_m15=use_m15)
            exec_combos.append(d)

    n_portfolio = len(MC_VALUES) * len(uni_sets)
    log.info(f"Phase B: {len(exec_combos)} exec combos x {n_portfolio} portfolio "
             f"variants = {len(exec_combos)*n_portfolio:,} result rows")

    t_b = datetime.now()
    with Pool(workers, initializer=_init_b,
              initargs=(precomp, raw_signals, uni_sets)) as pool:
        b_results = pool.map(_run_exec_combo, exec_combos)

    flat = [row for combo_rows in b_results for row in combo_rows]
    log.info(f"Phase B done in {(datetime.now()-t_b).seconds}s -- {len(flat):,} rows")

    # ── Save results ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M")
    df_res = pd.DataFrame(flat).sort_values("sharpe_r", ascending=False)

    full_path = RESULTS_DIR / f"v5_sweep_{ts}.csv"
    df_res.to_csv(full_path, index=False)
    log.info(f"Full sweep -> {full_path}  ({len(df_res):,} rows)")

    top_path = RESULTS_DIR / f"v5_top_{ts}.csv"
    df_res.head(200).to_csv(top_path, index=False)
    log.info(f"Top 200    -> {top_path}")

    # ── Per-symbol stats for best combo ──
    best = df_res.iloc[0].to_dict()
    log.info(f"Computing per-symbol stats for best combo "
             f"(Sharpe={best['sharpe_r']}, WR={best['win_rate']}%)...")
    sym_stats = compute_symbol_stats(best, precomp, raw_signals, uni_sets)
    if not sym_stats.empty:
        sym_path = RESULTS_DIR / f"v5_symbol_stats_{ts}.csv"
        sym_stats.to_csv(sym_path, index=False)
        log.info(f"Symbol stats -> {sym_path}")

    elapsed = (datetime.now() - t0).total_seconds()
    log.info(f"Total time: {elapsed:.0f}s")

    # ── Print results ──
    cols_main = ["universe", "lrc_length", "sigma_mult", "touch_pct",
                 "use_level", "vel_min", "use_m15",
                 "sl_atr_mult", "tp_rr", "breakeven_rr", "max_concurrent",
                 "n_trades", "win_rate", "avg_r",
                 "total_pnl_usd", "max_dd_usd", "sharpe_r", "sharpe_usd",
                 "avg_mfe_r", "avg_mae_r", "pct_mfe_gt_1r", "pct_mae_lt_half"]

    print("\n--- Top 20 combos by Sharpe (R) ---")
    print(df_res[[c for c in cols_main if c in df_res.columns]].head(20).to_string(index=False))

    if not sym_stats.empty:
        print(f"\n--- Top 15 symbols (best combo: {best['universe']} | "
              f"LRC-{int(best['lrc_length'])} | sigma={best['sigma_mult']} | "
              f"vel_min={int(best['vel_min'])} | m15={best['use_m15']}) ---")
        print(sym_stats.head(15).to_string(index=False))

    # ── Component contribution summary ──
    print("\n--- Component contribution (avg Sharpe by use_level x vel_min x use_m15) ---")
    comp_cols = ["use_level", "vel_min", "use_m15", "sharpe_r", "win_rate", "n_trades"]
    avail = [c for c in comp_cols if c in df_res.columns]
    comp_grp = (df_res[avail]
                .groupby(["use_level", "vel_min", "use_m15"])
                .agg(sharpe_r=("sharpe_r", "mean"),
                     win_rate=("win_rate", "mean"),
                     n_trades=("n_trades", "mean"))
                .round(2).reset_index()
                .sort_values("sharpe_r", ascending=False))
    print(comp_grp.to_string(index=False))


if __name__ == "__main__":
    main()
