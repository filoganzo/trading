"""
V9 regime analysis: correlate US-Ene trade outcomes with QQQ/Nasdaq market state.

Fixed at best v8 params:
  lrc=80, sigma=1.0, htf=0.5%, m15=0.5%, min_score=60, wing=2, sl=1.0, tp=1.0

New vs v8:
  For every accepted trade, attach QQQ market state at signal date:
    qqq_ret_5d       — Nasdaq 5d return (positive=rally, negative=falling)
    qqq_ret_20d      — Nasdaq 20d return (trend)
    qqq_above_ma50   — QQQ above its 50-bar MA (broad bull regime)
    qqq_lrc_sigma    — QQQ LRC extension in sigma units
                       +1 = QQQ at upper band (stretched bullish)
                       -1 = QQQ at lower band (stretched bearish / falling knife zone)
    stock_lrc_sigma  — Stock own LRC extension at signal bar (absolute stretch)
    stock_rel_ret_20d — Stock 20d return minus QQQ 20d return
                       positive = stock outran index (candidate for reversal)
                       negative = stock lagged index

Analysis:
  1. WR by direction x QQQ 5d return regime
  2. WR by direction x QQQ LRC sigma quartile
  3. WR by direction x stock vs QQQ 20d relative performance quartile
  4. LONG filter tests — which QQQ condition most improves WR?
  5. SHORT filter tests
  6. Combined long+short filter with best regime

Reads:  v8_trade_log_*.csv  (latest run)
Writes: v9_trade_log_{ts}.csv    (trade log + QQQ regime columns)
        v9_regime_{ts}.txt       (full analysis report)
        v9_filter_results_{ts}.csv

Usage:
    python -m stocks.backtest.runner_v9
"""
import sys, warnings, logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r'path/to/repo/hft')
warnings.filterwarnings("ignore")

from stocks.backtest.config import RESULTS_DIR, BT_CACHE_DIR, BT_CACHE_M15_DIR, BT_CACHE_W1_DIR
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
LEVERAGE  = 20; CAPITAL = 1_000.0; EFFECTIVE = CAPITAL * LEVERAGE

# ── Fixed best params (v8 confirmed) ─────────────────────────────────────────
LRC_LEN    = 80
SIGMA_MULT = 1.0
HTF_PCT    = 0.5    # proximity % (will divide by 100 internally)
M15_PCT    = 0.5
MIN_SCORE  = 60
SWING_WING = 2
SL_ATR     = 1.0
TP_RR      = 1.0
BE_RR      = 0.0
TARGET_UNI = "US-Ene"

SPX_SYM    = "US500"        # S&P 500 CFD (benchmark used by the universe builder)
QQQ_SYM    = "QQQ.NAS"     # Nasdaq 100 ETF (secondary)
MKT_LRC    = 80            # same LRC length as stocks for consistency


# ── Market regime features ────────────────────────────────────────────────────

def _load_market_df(sym: str) -> pd.DataFrame:
    """Try SPX first (US500), fall back to QQQ."""
    path = BT_CACHE_DIR / f"{sym}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]
    return df


def build_qqq_features() -> pd.DataFrame:
    """Load US500 (S&P 500) → fallback QQQ from parquet, compute regime features."""
    mkt = _load_market_df(SPX_SYM)
    src = SPX_SYM
    if mkt.empty:
        mkt = _load_market_df(QQQ_SYM)
        src = QQQ_SYM
    if mkt.empty:
        log.warning("Neither US500 nor QQQ.NAS in cache — regime features will be NaN")
        return pd.DataFrame()
    log.info(f"Market proxy: {src}")

    qqq = mkt
    qqq.columns = [c.lower() for c in qqq.columns]

    c   = qqq["close"].values
    qqq_s = pd.Series(c, index=qqq.index)

    # Returns
    ret_5d  = qqq_s.pct_change(5)
    ret_20d = qqq_s.pct_change(20)

    # MAs
    ma20 = qqq_s.rolling(20).mean()
    ma50 = qqq_s.rolling(50).mean()

    # LRC sigma position: (close - midline) / half_band
    # +1 = at upper band (bullish stretch), -1 = at lower band (bearish)
    mid, upper, lower, _ = _lrc_vectorized(c, MKT_LRC, 1.0)
    half_band = (upper - lower) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        lrc_sigma = np.where(half_band > 0, (c - mid) / half_band, np.nan)

    feats = pd.DataFrame({
        "qqq_close":      c,
        "qqq_ret_5d":     ret_5d.values,
        "qqq_ret_20d":    ret_20d.values,
        "qqq_above_ma20": (c > ma20.values).astype(float),
        "qqq_above_ma50": (c > ma50.values).astype(float),
        "qqq_lrc_sigma":  lrc_sigma,
    }, index=qqq.index)

    # Normalise to tz-naive date for merging
    feats.index = pd.to_datetime(feats.index).normalize().tz_localize(None)
    return feats


# ── Universe helpers ──────────────────────────────────────────────────────────

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


def build_universe_sets(bars, dvols, sym_meta):
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
    return {k: v for k, v in uni.items() if len(v) >= 5}


# ── Precompute ────────────────────────────────────────────────────────────────

def precompute_symbol(df, m15_df, w1_df) -> dict:
    c = df["close"].values; h = df["high"].values
    l = df["low"].values;   o = df["open"].values

    htf_tf = HTF_PCT / 100.0
    mid, upper, lower, _ = _lrc_vectorized(c, LRC_LEN, SIGMA_MULT)
    half_band = (upper - lower) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        stock_lrc_sigma = np.where(half_band > 0, (c - mid) / half_band, np.nan)

    stock_ret_20 = pd.Series(c, index=df.index).pct_change(20).values

    w_levels = {}
    if w1_df is not None and not w1_df.empty and len(w1_df) >= 4:
        w1_cols  = {col.lower(): col for col in w1_df.columns}
        w_h = w1_df[w1_cols.get("high", "High")].values
        w_l = w1_df[w1_cols.get("low",  "Low")].values
        lvls = precompute_weekly_levels(w_h, w_l, [htf_tf], swing_wing=SWING_WING)
        for tf, lv_list in lvls.items():
            w_levels[(SWING_WING, tf)] = lv_list
    else:
        w_levels[(SWING_WING, htf_tf)] = []

    m15_by_date = {}
    if m15_df is not None and not m15_df.empty:
        m15 = m15_df.copy()
        m15.columns = [cc.lower() for cc in m15.columns]
        m15["_date"] = m15.index.normalize()
        for d, grp in m15.groupby("_date"):
            m15_by_date[d] = list(zip(grp["high"].values, grp["low"].values,
                                      grp["close"].values, grp["open"].values))

    return {
        "close": c, "high": h, "low": l, "open": o,
        "n": len(c), "index": df.index,
        "lrc_c":            _lrc_candle_close_vectorized(c, LRC_CL, LRC_CS),
        "atr":              _atr_vectorized(h, l, c, ATR_P),
        "lrc_data":         {(LRC_LEN, SIGMA_MULT): (mid, upper, lower)},
        "w_levels":         w_levels,
        "m15_by_date":      m15_by_date,
        "stock_lrc_sigma":  stock_lrc_sigma,
        "stock_ret_20":     stock_ret_20,
    }


# ── Signal finding ────────────────────────────────────────────────────────────

def _find_signals(pc: dict, qqq_feats: pd.DataFrame) -> list:
    """
    Find signals with fixed v8 best params and attach QQQ regime at signal date.
    Returns list of tuples:
      (bi, eb, entry_px, direction, lv, atr_v, score,
       qqq_ret_5d, qqq_ret_20d, qqq_above_ma50, qqq_lrc_sigma,
       stock_lrc_sigma_val, stock_rel_ret_20d)
    """
    htf_tf    = HTF_PCT / 100.0
    m15_tf    = M15_PCT / 100.0
    n         = pc["n"]
    lrc_c     = pc["lrc_c"]
    atr       = pc["atr"]
    h, l, c   = pc["high"], pc["low"], pc["close"]
    idx       = pc["index"]
    w_levels  = pc["w_levels"].get((SWING_WING, htf_tf), [])
    m15_by_dt = pc["m15_by_date"]
    slrc      = pc["stock_lrc_sigma"]
    sret20    = pc["stock_ret_20"]

    if not w_levels:
        return []

    mid, upper, lower = pc["lrc_data"][(LRC_LEN, SIGMA_MULT)]
    min_i = max(LRC_LEN, LRC_CL + LRC_CS + 1, ATR_P + 1)
    if min_i >= n - 1:
        return []

    sigs = []
    for i in range(min_i, n - 1):
        lrc_ci = lrc_c[i]
        if np.isnan(lrc_ci) or np.isnan(upper[i]) or np.isnan(atr[i]):
            continue
        if   lrc_ci > upper[i]: direction = "SHORT"
        elif lrc_ci < lower[i]: direction = "LONG"
        else: continue

        price = float(c[i])
        lv, score = find_htf_level(w_levels, price, htf_tf, MIN_SCORE)
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

        # QQQ regime at signal bar date
        sig_date = pd.Timestamp(idx[i]).normalize()
        if sig_date.tzinfo is not None:
            sig_date = sig_date.tz_localize(None)

        if not qqq_feats.empty and sig_date in qqq_feats.index:
            qrow         = qqq_feats.loc[sig_date]
            qqq_r5d      = float(qrow["qqq_ret_5d"])
            qqq_r20d     = float(qrow["qqq_ret_20d"])
            qqq_ma50     = float(qrow["qqq_above_ma50"])
            qqq_lrc      = float(qrow["qqq_lrc_sigma"])
            qqq_r20_val  = qqq_r20d
        else:
            qqq_r5d = qqq_r20d = qqq_ma50 = qqq_lrc = qqq_r20_val = np.nan

        stk_lrc  = float(slrc[i])    if not np.isnan(slrc[i])    else np.nan
        stk_r20  = float(sret20[i])  if not np.isnan(sret20[i])  else np.nan
        rel_ret  = (stk_r20 - qqq_r20_val
                    if not (np.isnan(stk_r20) or np.isnan(qqq_r20_val))
                    else np.nan)

        sigs.append((
            i, eb, entry_px, direction, lv, float(atr[i]), score,
            qqq_r5d, qqq_r20d, qqq_ma50, qqq_lrc,
            stk_lrc, rel_ret,
        ))

    return sigs


# ── Exit simulation ───────────────────────────────────────────────────────────

def _find_exit(h, l, c, eb, ep, sl_px, tp_px, direction, risk, n):
    mfe_r = mae_r = 0.0
    for i in range(eb, n):
        hi, lo = h[i], l[i]
        fav = (hi - ep) / risk if direction == "LONG" else (ep - lo) / risk
        adv = (lo - ep) / risk if direction == "LONG" else (ep - hi) / risk
        if fav > mfe_r: mfe_r = fav
        if adv < mae_r: mae_r = adv
        sl_hit = (lo <= sl_px) if direction == "LONG" else (hi >= sl_px)
        tp_hit = (hi >= tp_px) if direction == "LONG" else (lo <= tp_px)
        if sl_hit: return i, sl_px, "SL",  round(mfe_r, 3), round(mae_r, 3)
        if tp_hit: return i, tp_px, "TP",  round(mfe_r, 3), round(mae_r, 3)
    return n - 1, float(c[n - 1]), "END", round(mfe_r, 3), round(mae_r, 3)


# ── Trade log builder ─────────────────────────────────────────────────────────

def build_trade_log(precomp: dict, raw_signals: dict,
                    uni_syms: frozenset) -> pd.DataFrame:
    rows = []
    for sym, sigs in raw_signals.items():
        if sym not in uni_syms:
            continue
        pc = precomp.get(sym)
        if pc is None:
            continue
        h, l, c, n, idx = pc["high"], pc["low"], pc["close"], pc["n"], pc["index"]

        for sig in sigs:
            (bi, eb, ep, dir_, lv, atr_v, score,
             qqq_r5d, qqq_r20d, qqq_ma50, qqq_lrc,
             stk_lrc, rel_ret) = sig

            pos   = +1 if dir_ == "SHORT" else -1
            sl_px = lv + pos * SL_ATR * atr_v
            if pos * (sl_px - ep) <= 0:
                continue
            risk = abs(ep - sl_px)
            if risk < 1e-6:
                continue
            tp_px = ep - pos * TP_RR * risk

            xb, xp, xtype, mfe_r, mae_r = _find_exit(h, l, c, eb, ep, sl_px, tp_px, dir_, risk, n)
            r          = (xp - ep) / risk if dir_ == "LONG" else (ep - xp) / risk
            dollar_pnl = r * (EFFECTIVE / ep) * risk

            rows.append({
                "sym":       sym,
                "direction": dir_,
                "entry_date": idx[eb],
                "exit_date":  idx[xb],
                "r":          round(r, 3),
                "pnl":        round(dollar_pnl, 2),
                "mfe_r":      mfe_r,
                "mae_r":      mae_r,
                "exit_type":  xtype,
                "htf_score":  score,
                # QQQ regime
                "qqq_ret_5d":       round(qqq_r5d,  4) if not np.isnan(qqq_r5d)  else np.nan,
                "qqq_ret_20d":      round(qqq_r20d, 4) if not np.isnan(qqq_r20d) else np.nan,
                "qqq_above_ma50":   qqq_ma50,
                "qqq_lrc_sigma":    round(qqq_lrc,  3) if not np.isnan(qqq_lrc)  else np.nan,
                # Stock relative
                "stock_lrc_sigma":      round(stk_lrc,  3) if not np.isnan(stk_lrc)  else np.nan,
                "stock_rel_ret_20d":    round(rel_ret,  4) if not np.isnan(rel_ret)  else np.nan,
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    return df.sort_values("entry_date").reset_index(drop=True)


# ── Regime analysis ───────────────────────────────────────────────────────────

def _smry(df: pd.DataFrame) -> pd.Series:
    if len(df) == 0:
        return pd.Series({"n": 0, "wr%": np.nan, "avg_r": np.nan, "total_pnl": 0})
    return pd.Series({
        "n":         len(df),
        "wr%":       round((df["r"] > 0).mean() * 100, 1),
        "avg_r":     round(df["r"].mean(), 3),
        "total_pnl": round(df["pnl"].sum(), 0),
    })


def regime_analysis(df: pd.DataFrame) -> str:
    lines = []

    def S(title, tbl):
        lines.append(f"\n{'='*64}")
        lines.append(title)
        lines.append("="*64)
        lines.append(tbl.to_string())

    # ─ 1. Direction split ─────────────────────────────────────────
    S("1.  Direction split — all trades",
      df.groupby("direction").apply(_smry).reset_index())

    # ─ 2. QQQ 5d return regime x direction ────────────────────────
    df2 = df.dropna(subset=["qqq_ret_5d"]).copy()
    df2["qqq_regime_5d"] = pd.cut(
        df2["qqq_ret_5d"],
        bins=[-np.inf, -0.03, -0.01, 0.01, 0.03, np.inf],
        labels=["crash(<-3%)", "falling(-3to-1%)", "flat(±1%)",
                "rising(1→3%)", "rally(>3%)"],
    )
    S("2.  WR by direction × QQQ 5d return regime\n"
      "    (Are LONGs hurt when Nasdaq is falling?)",
      df2.groupby(["direction", "qqq_regime_5d"]).apply(_smry).reset_index())

    # ─ 3. QQQ LRC sigma quartiles x direction ─────────────────────
    df3 = df.dropna(subset=["qqq_lrc_sigma"]).copy()
    if len(df3) >= 8:
        df3["qqq_lrc_q"] = pd.qcut(
            df3["qqq_lrc_sigma"], q=4,
            labels=["Q1(QQQ-bearish-stretch)", "Q2", "Q3",
                    "Q4(QQQ-bullish-stretch)"],
        )
        S("3.  WR by direction × QQQ LRC sigma quartile\n"
          "    Q1=QQQ most extended below LRC midline, Q4=above",
          df3.groupby(["direction", "qqq_lrc_q"]).apply(_smry).reset_index())

    # ─ 4. Stock relative performance vs QQQ ───────────────────────
    df4 = df.dropna(subset=["stock_rel_ret_20d"]).copy()
    if len(df4) >= 8:
        df4["rel_q"] = pd.qcut(
            df4["stock_rel_ret_20d"], q=4,
            labels=["Q1(stock lagged QQQ)", "Q2", "Q3",
                    "Q4(stock outran QQQ)"],
        )
        S("4.  WR by direction × stock 20d return relative to QQQ\n"
          "    Positive = stock outperformed Nasdaq over 20d",
          df4.groupby(["direction", "rel_q"]).apply(_smry).reset_index())

    # ─ 5. Stock LRC sigma quartiles x direction ───────────────────
    df5 = df.dropna(subset=["stock_lrc_sigma"]).copy()
    if len(df5) >= 8:
        df5["slrc_q"] = pd.qcut(
            df5["stock_lrc_sigma"], q=4,
            labels=["Q1(most-bearish-stretch)", "Q2", "Q3",
                    "Q4(most-bullish-stretch)"],
        )
        S("5.  WR by direction × stock LRC extension magnitude\n"
          "    How stretched is the stock itself at signal time?",
          df5.groupby(["direction", "slrc_q"]).apply(_smry).reset_index())

    # ─ 6. LONG filter tests ───────────────────────────────────────
    longs = df[df["direction"] == "LONG"].copy()
    lines.append(f"\n{'='*64}")
    lines.append("6.  LONG filter tests")
    lines.append("="*64)

    frows = []
    def fl(name, mask=None):
        sub = longs if mask is None else longs[mask]
        frows.append({
            "filter": name, "n": len(sub),
            "wr%":   round((sub["r"] > 0).mean() * 100, 1) if len(sub) else 0,
            "avg_r": round(sub["r"].mean(), 3)              if len(sub) else 0,
            "total_pnl": round(sub["pnl"].sum(), 0)         if len(sub) else 0,
        })

    fl("no filter (baseline)")
    fl("QQQ 5d ret > -1%",      longs["qqq_ret_5d"] > -0.01)
    fl("QQQ 5d ret > -2%",      longs["qqq_ret_5d"] > -0.02)
    fl("QQQ 5d ret > -3%",      longs["qqq_ret_5d"] > -0.03)
    fl("QQQ above MA50",         longs["qqq_above_ma50"] == 1.0)
    fl("QQQ lrc_sigma > -0.3",  longs["qqq_lrc_sigma"] > -0.3)
    fl("QQQ lrc_sigma > 0",     longs["qqq_lrc_sigma"] > 0.0)
    fl("stock outperformed QQQ (rel>0)",  longs["stock_rel_ret_20d"] > 0)
    fl("stock lagged QQQ (rel<0)",        longs["stock_rel_ret_20d"] <= 0)
    fl("QQQ above MA50 + 5d > -2%",
       (longs["qqq_above_ma50"] == 1.0) & (longs["qqq_ret_5d"] > -0.02))
    fl("QQQ above MA50 + lrc > -0.3",
       (longs["qqq_above_ma50"] == 1.0) & (longs["qqq_lrc_sigma"] > -0.3))
    fl("QQQ 5d > -2% + lrc > -0.3",
       (longs["qqq_ret_5d"] > -0.02) & (longs["qqq_lrc_sigma"] > -0.3))
    lines.append(pd.DataFrame(frows).to_string(index=False))

    # ─ 7. SHORT filter tests ──────────────────────────────────────
    shorts = df[df["direction"] == "SHORT"].copy()
    lines.append(f"\n{'='*64}")
    lines.append("7.  SHORT filter tests")
    lines.append("="*64)

    srows = []
    def fs(name, mask=None):
        sub = shorts if mask is None else shorts[mask]
        srows.append({
            "filter": name, "n": len(sub),
            "wr%":   round((sub["r"] > 0).mean() * 100, 1) if len(sub) else 0,
            "avg_r": round(sub["r"].mean(), 3)              if len(sub) else 0,
            "total_pnl": round(sub["pnl"].sum(), 0)         if len(sub) else 0,
        })

    fs("no filter (baseline)")
    fs("QQQ 5d ret < 0",          shorts["qqq_ret_5d"] < 0)
    fs("QQQ 5d ret < -1%",        shorts["qqq_ret_5d"] < -0.01)
    fs("QQQ below MA50",           shorts["qqq_above_ma50"] == 0.0)
    fs("QQQ lrc_sigma < 0",       shorts["qqq_lrc_sigma"] < 0.0)
    fs("QQQ lrc_sigma > 0.3 (QQQ overbought)", shorts["qqq_lrc_sigma"] > 0.3)
    fs("stock outperformed QQQ (good SHORT candidate)", shorts["stock_rel_ret_20d"] > 0)
    lines.append(pd.DataFrame(srows).to_string(index=False))

    # ─ 8. Best combined filter: LONG + SHORT together ─────────────
    lines.append(f"\n{'='*64}")
    lines.append("8.  Combined LONG+SHORT: regime-filtered vs unfiltered")
    lines.append("="*64)

    combo_rows = []
    def combo(name, long_mask, short_mask):
        sub_l = longs if long_mask is None else longs[long_mask]
        sub_s = shorts if short_mask is None else shorts[short_mask]
        sub   = pd.concat([sub_l, sub_s])
        combo_rows.append({
            "filter": name, "n": len(sub), "n_long": len(sub_l), "n_short": len(sub_s),
            "wr%":   round((sub["r"] > 0).mean() * 100, 1) if len(sub) else 0,
            "avg_r": round(sub["r"].mean(), 3)              if len(sub) else 0,
            "total_pnl": round(sub["pnl"].sum(), 0)         if len(sub) else 0,
        })

    combo("unfiltered",
          None, None)
    combo("LONG: QQQ above MA50 | SHORT: all",
          longs["qqq_above_ma50"] == 1.0, None)
    combo("LONG: QQQ 5d > -2%   | SHORT: all",
          longs["qqq_ret_5d"] > -0.02, None)
    combo("LONG: QQQ lrc > -0.3 | SHORT: all",
          longs["qqq_lrc_sigma"] > -0.3, None)
    combo("LONG: QQQ above MA50 | SHORT: QQQ 5d < 0",
          longs["qqq_above_ma50"] == 1.0,
          shorts["qqq_ret_5d"] < 0)
    combo("LONG: QQQ 5d > -2% + above MA50 | SHORT: QQQ lrc > 0.3",
          (longs["qqq_ret_5d"] > -0.02) & (longs["qqq_above_ma50"] == 1.0),
          shorts["qqq_lrc_sigma"] > 0.3)
    lines.append(pd.DataFrame(combo_rows).to_string(index=False))

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = datetime.now()
    log.info("v9 regime analysis starting")
    log.info(f"Fixed params: lrc={LRC_LEN}, sigma={SIGMA_MULT}, "
             f"htf={HTF_PCT}%, m15={M15_PCT}%, score>={MIN_SCORE}, "
             f"wing={SWING_WING}, sl={SL_ATR}, tp={TP_RR}")

    # ── QQQ features ──
    log.info(f"Loading {QQQ_SYM} regime features...")
    qqq_feats = build_qqq_features()
    if qqq_feats.empty:
        log.warning("No QQQ data — analysis will proceed but regime columns will be NaN")
    else:
        log.info(f"QQQ: {len(qqq_feats)} dates "
                 f"({qqq_feats.index.min().date()} to {qqq_feats.index.max().date()})")

    # ── Daily bars ──
    log.info("Loading daily bars...")
    univ = build_universe()
    bars, dvols, sym_meta = load_all_bars(univ)
    log.info(f"Bars loaded: {len(bars)} symbols")

    uni_sets = build_universe_sets(bars, dvols, sym_meta)
    log.info("Universe sets: " + ", ".join(f"{k}={len(v)}" for k, v in uni_sets.items()))

    ene_syms = uni_sets.get(TARGET_UNI, frozenset())
    if not ene_syms:
        log.error(f"{TARGET_UNI} not found or too small")
        return
    log.info(f"{TARGET_UNI}: {len(ene_syms)} symbols: {sorted(ene_syms)}")

    # ── Precompute ──
    w1_avail  = BT_CACHE_W1_DIR.exists()  and any(BT_CACHE_W1_DIR.glob("*.parquet"))
    m15_avail = BT_CACHE_M15_DIR.exists() and any(BT_CACHE_M15_DIR.glob("*.parquet"))
    log.info(f"Precomputing {len(ene_syms)} symbols "
             f"(W1={'yes' if w1_avail else 'no'}, M15={'yes' if m15_avail else 'no'})...")
    precomp = {}
    for sym in ene_syms:
        if sym not in bars:
            continue
        try:
            w1_df  = load_bt_bars_w1(sym)  if w1_avail  else pd.DataFrame()
            m15_df = load_bt_bars_m15(sym) if m15_avail else pd.DataFrame()
            precomp[sym] = precompute_symbol(bars[sym], m15_df, w1_df)
        except Exception as e:
            log.debug(f"  {sym} failed: {e}")
    log.info(f"Precomputed {len(precomp)} symbols")

    # ── Phase A: find signals ──
    log.info("Finding signals with fixed best params + QQQ regime enrichment...")
    raw_signals = {}
    for sym, pc in precomp.items():
        try:
            sigs = _find_signals(pc, qqq_feats)
            if sigs:
                raw_signals[sym] = sigs
        except Exception as e:
            log.debug(f"  {sym} signal error: {e}")

    n_raw = sum(len(s) for s in raw_signals.values())
    log.info(f"Raw signals: {n_raw} across {len(raw_signals)} symbols")

    # Strip heavy M15/level data
    for pc in precomp.values():
        pc.pop("m15_by_date", None)
        pc.pop("w_levels",    None)

    # ── Build trade log ──
    log.info("Building trade log...")
    trade_log = build_trade_log(precomp, raw_signals, ene_syms)
    log.info(f"Trades: {len(trade_log)} total "
             f"({(trade_log['direction']=='LONG').sum()} LONG, "
             f"{(trade_log['direction']=='SHORT').sum()} SHORT)")

    if trade_log.empty:
        log.error("No trades. Check W1 and M15 caches.")
        return

    # ── Save trade log ──
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    tlog_path = RESULTS_DIR / f"v9_trade_log_{ts}.csv"
    trade_log.to_csv(tlog_path, index=False)
    log.info(f"Trade log -> {tlog_path}")

    # ── Regime analysis ──
    log.info("Running regime analysis...")
    report = regime_analysis(trade_log)

    report_path = RESULTS_DIR / f"v9_regime_{ts}.txt"
    report_path.write_text(report, encoding="utf-8")
    log.info(f"Regime report -> {report_path}")

    # Print safely to stdout (replace any non-ASCII for Windows console)
    print(report.encode("ascii", errors="replace").decode("ascii"))

    log.info(f"Total time: {(datetime.now()-t0).total_seconds():.0f}s")


if __name__ == "__main__":
    main()
