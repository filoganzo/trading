"""
discover_all.py — Full param sweep across ALL available symbols.

For each symbol, sweeps:
  sigma × rel_perf × atr_contract × entry_tf × tp  (96 combos/symbol)
Reports the best combo per symbol, ranked by Sharpe.
No dvol filter — every symbol with daily + M15 + W1 data is tested.

Usage:
    python -m stocks.backtest.discover_all
"""
import sys, warnings, logging
from datetime import datetime
from itertools import product as iproduct

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

# ── Fixed (confirmed across all prior sweeps) ─────────────────────────────────
LRC_LENGTH    = 80
HTF_TOUCH_PCT = 0.5
HTF_FRAC      = HTF_TOUCH_PCT / 100.0
MIN_HTF_SCORE = 60
SWING_WING    = 2
ENTRY_TOUCH   = 0.005        # 0.5% proximity for intraday entry
SL_MULT       = 1.0
BE_RR         = 0.0

# ── Sweep params ──────────────────────────────────────────────────────────────
SIGMA_MULTS       = [1.0, 1.5]
REL_PERF_THRESHS  = [0.0, -0.02, -0.05]
ATR_CONTRACTS     = [1.0, 0.85]
ENTRY_TFS         = ["M15", "M30"]
TP_VALUES         = [1.0, 1.5, 2.0, 3.0]

# ── Capital ───────────────────────────────────────────────────────────────────
LEVERAGE  = 20
CAPITAL   = 1_000.0
EFFECTIVE = CAPITAL * LEVERAGE   # $20,000

LRC_CL = 15; LRC_CS = 9; ATR_P = 14
MIN_BARS = 200
ALL_EXCH = {"NYSE", "NAS", "LSE", "ETR", "PAR", "IT", "MAD", "AMS", "IE", "SWX"}

# ── Live universe quality filters ─────────────────────────────────────────────
MIN_TRADES = 4
MIN_WR     = 55.0
MIN_SHARPE = 0.6


# ── SPX returns ───────────────────────────────────────────────────────────────

def load_spx_rets() -> dict:
    base = BT_CACHE_DIR
    for name in ["US500.parquet", "US500.NAS.parquet", "US500.NYSE.parquet"]:
        p = base / name
        if p.exists():
            df = pd.read_parquet(p)
            df.columns = [c.lower() for c in df.columns]
            r5 = df["close"].pct_change(5)
            result = {pd.Timestamp(d).normalize(): float(v)
                      for d, v in r5.items() if pd.notna(v)}
            log.info(f"SPX rets loaded: {len(result)} dates")
            return result
    log.warning("US500 not found — rel_perf filter will be inactive")
    return {}


# ── M15→M30 resample ─────────────────────────────────────────────────────────

def resample_to_bars(m15_df: pd.DataFrame, rule: str) -> dict:
    rs = (m15_df.resample(rule, closed='left', label='left')
          .agg({'open':'first','high':'max','low':'min','close':'last'})
          .dropna())
    rs["_date"] = rs.index.normalize()
    out = {}
    for d, grp in rs.groupby("_date"):
        out[d] = list(zip(grp["high"].values, grp["low"].values,
                          grp["close"].values, grp["open"].values))
    return out


# ── Precompute one symbol (all sigma values) ──────────────────────────────────

def precompute(df: pd.DataFrame, m15_df: pd.DataFrame, w1_df: pd.DataFrame) -> dict | None:
    c = df["close"].values; h = df["high"].values
    l = df["low"].values; n = len(c)
    if n < MIN_BARS:
        return None

    # LRC for all sigma values
    lrc_data = {}
    for sm in SIGMA_MULTS:
        _, upper, lower, _ = _lrc_vectorized(c, LRC_LENGTH, sm)
        lrc_data[sm] = (upper, lower)

    lrc_c = _lrc_candle_close_vectorized(c, LRC_CL, LRC_CS)
    atr   = _atr_vectorized(h, l, c, ATR_P)

    # Weekly levels (single HTF proximity)
    w_levels = []
    if w1_df is not None and not w1_df.empty and len(w1_df) >= 4:
        w1c = {col.lower(): col for col in w1_df.columns}
        wh = w1_df[w1c.get("high", "High")].values
        wl = w1_df[w1c.get("low",  "Low")].values
        try:
            lvl_dict = precompute_weekly_levels(wh, wl, [HTF_FRAC], swing_wing=SWING_WING)
            w_levels = lvl_dict.get(HTF_FRAC, [])
        except Exception:
            pass

    # M15 and M30 by date
    m15_by_date = {}
    m30_by_date = {}
    if m15_df is not None and not m15_df.empty:
        m15 = m15_df.copy()
        m15.columns = [cc.lower() for cc in m15.columns]
        m15["_date"] = m15.index.normalize()
        for d, grp in m15.groupby("_date"):
            m15_by_date[d] = list(zip(grp["high"].values, grp["low"].values,
                                      grp["close"].values, grp["open"].values))
        try:
            m30_by_date = resample_to_bars(m15[["open","high","low","close"]], "30min")
        except Exception:
            pass

    # Stock 5-day returns
    ret5_map = {}
    for i in range(5, n):
        if c[i-5] > 0:
            ret5_map[pd.Timestamp(df.index[i]).normalize()] = float((c[i]-c[i-5])/c[i-5])

    return {
        "close": c, "high": h, "low": l, "n": n, "index": df.index,
        "lrc_c": lrc_c, "atr": atr,
        "lrc_data":     lrc_data,       # {sigma: (upper, lower)}
        "w_levels":     w_levels,
        "m15_by_date":  m15_by_date,
        "m30_by_date":  m30_by_date,
        "ret5_map":     ret5_map,
    }


# ── Signal finding ────────────────────────────────────────────────────────────

def find_signals(pc: dict, sigma: float, entry_tf: str,
                 rel_perf_thresh: float, atr_contract: float,
                 spx_rets: dict) -> list:
    n      = pc["n"]; idx = pc["index"]
    lrc_c  = pc["lrc_c"]; atr = pc["atr"]
    c      = pc["close"]
    w_lv   = pc["w_levels"]
    upper, lower = pc["lrc_data"].get(sigma, (None, None))
    if upper is None:
        return []

    m_by_dt = pc["m15_by_date"] if entry_tf == "M15" else pc["m30_by_date"]
    if not m_by_dt:
        return []
    if not w_lv:
        return []

    min_i = max(LRC_LENGTH, LRC_CL + LRC_CS + 1, ATR_P + 1, 10)
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

        # ATR contraction check (optional)
        if atr_contract < 1.0:
            lb = max(0, i - 10)
            if lb < i:
                peak = np.max(atr[lb:i])   # excludes bar i — no look-ahead
                if peak > 0 and atr[i] / peak > atr_contract:
                    continue

        price = float(c[i])
        lv, score = find_htf_level(w_lv, price, HTF_FRAC, MIN_HTF_SCORE)
        if np.isnan(lv):
            continue

        # Relative performance filter (optional)
        if rel_perf_thresh != 0.0 and spx_rets:
            dk    = pd.Timestamp(idx[i]).normalize()
            spx_r = spx_rets.get(dk)
            stk_r = pc["ret5_map"].get(dk)
            if spx_r is not None and stk_r is not None:
                rel = stk_r - spx_r
                if direction == "LONG"  and rel > rel_perf_thresh:
                    continue
                if direction == "SHORT" and rel < -rel_perf_thresh:
                    continue

        eb = i + 1
        day_key = pd.Timestamp(idx[eb]).normalize()
        m_bars  = m_by_dt.get(day_key)
        if not m_bars:
            continue

        _, entry_px = find_m15_entry(m_bars, direction, lv, ENTRY_TOUCH)
        if entry_px is None:
            continue

        sigs.append((i, eb, entry_px, direction, lv, float(atr[i]), score))

    return sigs


# ── Exit ──────────────────────────────────────────────────────────────────────

def find_exit(h, l, c, eb, ep, sl_px, tp_px, direction, n):
    for i in range(eb, n):
        sl_hit = (l[i] <= sl_px) if direction == "LONG" else (h[i] >= sl_px)
        tp_hit = (h[i] >= tp_px) if direction == "LONG" else (l[i] <= tp_px)
        if sl_hit: return i, sl_px, "SL"
        if tp_hit: return i, tp_px, "TP"
    return n-1, float(c[n-1]), "END"


# ── Per-symbol sweep: find best combo ─────────────────────────────────────────

def sweep_symbol(sym: str, pc: dict, spx_rets: dict, sym_meta: dict) -> dict | None:
    """Try all 96 combos, return stats for the best Sharpe combo."""
    h = pc["high"]; l = pc["low"]; c_arr = pc["close"]; n = pc["n"]
    best = None

    for sigma, rel_thresh, atr_c, entry_tf, tp_rr in iproduct(
            SIGMA_MULTS, REL_PERF_THRESHS, ATR_CONTRACTS, ENTRY_TFS, TP_VALUES):

        sigs = find_signals(pc, sigma, entry_tf, rel_thresh, atr_c, spx_rets)
        if not sigs:
            continue

        trades = []
        for (bi, eb, ep, dir_, lv, atr_v, readiness) in sigs:
            pos   = +1 if dir_ == "SHORT" else -1
            sl_px = lv + pos * SL_MULT * atr_v
            if pos * (sl_px - ep) <= 0:
                continue
            risk = abs(ep - sl_px)
            if risk < 1e-6:
                continue
            tp_px = ep - pos * tp_rr * risk
            _, xp, xtype = find_exit(h, l, c_arr, eb, ep, sl_px, tp_px, dir_, n)
            r   = (xp - ep) / risk if dir_ == "LONG" else (ep - xp) / risk
            pnl = r * (EFFECTIVE / ep) * risk
            trades.append({"r": round(r,3), "pnl": round(pnl,2),
                           "direction": dir_, "exit_type": xtype,
                           "entry_date": pc["index"][eb]})

        if not trades:
            continue

        r_arr   = np.array([t["r"]   for t in trades])
        pnl_arr = np.array([t["pnl"] for t in trades])
        wins    = r_arr > 0

        n_years = n / 252
        ann     = np.sqrt(len(r_arr) / n_years) if n_years > 0 else 0
        sharpe  = float(r_arr.mean() / r_arr.std() * ann) if r_arr.std() > 0 else 0.0

        if best is None or sharpe > best["sharpe_r"]:
            longs  = [t for t in trades if t["direction"] == "LONG"]
            shorts = [t for t in trades if t["direction"] == "SHORT"]

            def _wr(ts):
                if not ts: return None
                return round((np.array([t["r"] for t in ts]) > 0).mean() * 100, 1)

            # Compute equity drawdown
            eq = np.cumsum(np.insert(pnl_arr, 0, 0.0))
            max_dd = float((eq - np.maximum.accumulate(eq)).min())

            best = {
                "sym":               sym,
                "exchange":          sym_meta.get("exchange", "?"),
                "sector":            sym_meta.get("sector",   "?"),
                "sigma":             sigma,
                "rel_perf_thresh":   rel_thresh,
                "atr_contract":      atr_c,
                "entry_tf":          entry_tf,
                "tp_rr":             tp_rr,
                "n_trades":          len(trades),
                "win_rate":          round(float(wins.mean() * 100), 1),
                "avg_r":             round(float(r_arr.mean()), 3),
                "total_r":           round(float(r_arr.sum()),  2),
                "total_pnl":         round(float(pnl_arr.sum()), 0),
                "max_dd_usd":        round(max_dd, 0),
                "sharpe_r":          round(sharpe, 2),
                "n_long":            len(longs),
                "n_short":           len(shorts),
                "long_wr":           _wr(longs),
                "short_wr":          _wr(shorts),
                "trades_yr":         round(len(trades) / n_years, 1),
            }

    return best


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = datetime.now()
    log.info("discover_all: full param sweep across ALL symbols")
    log.info(f"Sweep: sigma={SIGMA_MULTS} | rel_perf={REL_PERF_THRESHS} | "
             f"atr_contract={ATR_CONTRACTS} | entry_tf={ENTRY_TFS} | tp={TP_VALUES}")
    log.info(f"Fixed: LRC={LRC_LENGTH} | htf={HTF_TOUCH_PCT}% | "
             f"score={MIN_HTF_SCORE} | wing={SWING_WING} | sl={SL_MULT}")

    w1_ok  = BT_CACHE_W1_DIR.exists()  and any(BT_CACHE_W1_DIR.glob("*.parquet"))
    m15_ok = BT_CACHE_M15_DIR.exists() and any(BT_CACHE_M15_DIR.glob("*.parquet"))
    log.info(f"W1={'ok' if w1_ok else 'MISSING'}  M15={'ok' if m15_ok else 'MISSING'}")

    spx_rets = load_spx_rets()

    # Load ALL symbols — no dvol quantile filter
    log.info("Loading daily bars for all symbols...")
    univ     = build_universe()
    sym_rows = []
    for _, row in univ.iterrows():
        sym  = row["symbol"]
        exch = row["exchange"]
        if exch not in ALL_EXCH:
            continue
        df = load_bt_bars(sym)
        if df.empty or len(df) < MIN_BARS:
            continue
        df = df[["Open","High","Low","Close","Volume"]].copy()
        df.columns = ["open","high","low","close","volume"]
        sym_rows.append((sym, exch, row.get("sector","?"), df))

    log.info(f"Loaded {len(sym_rows)} symbols")

    # Precompute
    log.info("Precomputing indicators (LRC×2 sigma + M15/M30 resample)...")
    precomp  = {}
    sym_meta = {}
    w1_cnt = m15_cnt = 0
    for sym, exch, sector, df in sym_rows:
        try:
            w1_df  = load_bt_bars_w1(sym)  if w1_ok  else pd.DataFrame()
            m15_df = load_bt_bars_m15(sym) if m15_ok else pd.DataFrame()
            if not w1_df.empty:  w1_cnt  += 1
            if not m15_df.empty: m15_cnt += 1
            pc = precompute(df, m15_df, w1_df)
            if pc is not None:
                precomp[sym]  = pc
                sym_meta[sym] = {"exchange": exch, "sector": sector}
        except Exception as e:
            log.debug(f"  {sym} precompute failed: {e}")

    log.info(f"Precomputed {len(precomp)} symbols "
             f"({w1_cnt} W1, {m15_cnt} M15) in {(datetime.now()-t0).seconds}s")

    # Per-symbol sweep
    n_combos = (len(SIGMA_MULTS) * len(REL_PERF_THRESHS) * len(ATR_CONTRACTS)
                * len(ENTRY_TFS) * len(TP_VALUES))
    log.info(f"Sweeping {n_combos} combos × {len(precomp)} symbols...")
    results = []
    n_with  = 0
    for i, (sym, pc) in enumerate(precomp.items(), 1):
        best = sweep_symbol(sym, pc, spx_rets, sym_meta.get(sym, {}))
        if best:
            results.append(best)
            n_with += 1
        if i % 250 == 0:
            log.info(f"  {i}/{len(precomp)} symbols | {n_with} with signals so far")

    log.info(f"Done: {n_with} symbols produced >=1 signal")

    if not results:
        log.warning("No signals found across any symbol.")
        return

    df_all = pd.DataFrame(results).sort_values("sharpe_r", ascending=False).reset_index(drop=True)

    live = df_all[
        (df_all["n_trades"] >= MIN_TRADES) &
        (df_all["win_rate"] >= MIN_WR) &
        (df_all["sharpe_r"] >= MIN_SHARPE)
    ].copy()

    # Frequency estimate
    n_bars_median = np.median([pc["n"] for pc in precomp.values()])
    years_est     = n_bars_median / 252
    all_pw  = df_all["n_trades"].sum() / (years_est * 52)
    live_pw = live["n_trades"].sum()   / (years_est * 52)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    disc_path = RESULTS_DIR / f"discover_all_{ts}.csv"
    live_path = RESULTS_DIR / f"discover_live_{ts}.csv"
    df_all.to_csv(disc_path, index=False)
    live.to_csv(live_path,   index=False)
    log.info(f"Full discovery -> {disc_path}  ({len(df_all)} symbols)")
    log.info(f"Live universe  -> {live_path}  ({len(live)} symbols)")
    log.info(f"Total time: {(datetime.now()-t0).total_seconds():.0f}s")

    show = ["sym","exchange","sector","sigma","entry_tf","tp_rr","rel_perf_thresh",
            "atr_contract","n_trades","win_rate","avg_r","total_pnl","max_dd_usd",
            "sharpe_r","n_long","n_short","long_wr","short_wr","trades_yr"]

    print(f"\n{'='*80}")
    print(f"ALL-SYMBOL DISCOVERY  |  {len(df_all)} symbols with signals")
    print(f"Backtest window: ~{years_est:.1f} years")
    print(f"Total raw signals/week (all {n_with} symbols): {all_pw:.1f}")
    print(f"Live universe signals/week ({len(live)} symbols, quality filter): {live_pw:.1f}")
    print(f"{'='*80}\n")

    print("--- Top 60 symbols by Sharpe (best combo per symbol) ---")
    print(df_all[[c for c in show if c in df_all.columns]].head(60).to_string(index=False))

    print(f"\n--- Live universe  (n>={MIN_TRADES}, WR>={MIN_WR}%, Sharpe>={MIN_SHARPE}) ---")
    print(f"  {len(live)} symbols | {int(live['n_trades'].sum())} trades over ~{years_est:.1f} yr "
          f"= {live_pw:.1f} setups/week")
    print(live[[c for c in show if c in live.columns]].to_string(index=False))

    print("\n--- By sector (top 15) ---")
    if "sector" in df_all.columns:
        sec = (df_all.groupby("sector")
               .agg(n_syms=("sym","count"),
                    live_syms=("sym", lambda x: live["sym"].isin(x).sum()),
                    total_trades=("n_trades","sum"),
                    avg_wr=("win_rate","mean"),
                    avg_sharpe=("sharpe_r","mean"),
                    trades_yr_sum=("trades_yr","sum"))
               .round(2).sort_values("avg_sharpe", ascending=False))
        sec["setups_wk"] = (sec["trades_yr_sum"] / 52).round(2)
        print(sec.head(15).to_string())

    print("\n--- Best params by entry_tf ---")
    tf_grp = (df_all.groupby("entry_tf")
              .agg(n_syms=("sym","count"),
                   total_trades=("n_trades","sum"),
                   avg_wr=("win_rate","mean"),
                   avg_sharpe=("sharpe_r","mean"))
              .round(2).sort_values("avg_sharpe", ascending=False))
    print(tf_grp.to_string())

    print("\n--- Best params by sigma ---")
    s_grp = (df_all.groupby("sigma")
             .agg(n_syms=("sym","count"),
                  avg_wr=("win_rate","mean"),
                  avg_sharpe=("sharpe_r","mean"))
             .round(2).sort_values("avg_sharpe", ascending=False))
    print(s_grp.to_string())

    print("\n--- Best params by rel_perf_thresh ---")
    rp_grp = (df_all.groupby("rel_perf_thresh")
              .agg(n_syms=("sym","count"),
                   avg_wr=("win_rate","mean"),
                   avg_sharpe=("sharpe_r","mean"))
              .round(2).sort_values("avg_sharpe", ascending=False))
    print(rp_grp.to_string())

    print(f"\n--- Live universe symbol list ({len(live)}) ---")
    if not live.empty:
        print(", ".join(live["sym"].tolist()))


if __name__ == "__main__":
    main()
