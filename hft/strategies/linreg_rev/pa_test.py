"""
Evaluate price-action models against the same control framework.

Variants tested, all LONG, all measured as forward return in ATR units versus a
control of valid bars where the variant did NOT fire:

  linreg          the LinReg/WaveTrend setup alone
  linreg_htf      + higher-timeframe overextension gate
  hammer          a bullish hammer, alone
  br              break-and-reclaim of a confirmed HTF support level, alone
  br_hammer       break-and-reclaim whose reclaim bar is a hammer
  near_level      price merely within N ATR of a confirmed support
  linreg_level    LinReg setup occurring near a support
  linreg_br       LinReg setup plus a break-and-reclaim
  linreg_hammer   LinReg setup whose trigger bar is a hammer

Two statistics per variant, because they fail in different ways:
  pooled t   Welch t on all signals against all control bars. Powerful, but
             inflated by overlapping forward windows.
  sign test  how many of the N symbols individually show a positive lift. Far
             more conservative and immune to the overlap problem -- this is the
             one to believe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from mtf_runner import BaseParams, base_indicators, base_signals
from mtf import htf_gate, HtfParams, ALIGN_CLOSED
from price_action import (htf_levels_for_base, break_reclaim, hammer,
                          distance_to_level)
from stocks_test import stock_symbols


def variants_for(sym: str, base_tf: str, htf_tf: str, bp: BaseParams,
                 hp: HtfParams, level_tf: str, horizons=(6, 12, 24),
                 near_atr: float = 0.75, br_lookback: int = 6):
    try:
        d = base_indicators(sym, base_tf, bp)
    except Exception:
        return None
    if len(d) < bp.chan_len + 200:
        return None

    a = d["atr"].to_numpy(float)
    valid = (np.isfinite(a) & np.isfinite(d["lrc_lower"].to_numpy(float))
             & np.isfinite(d["wt2"].to_numpy(float)))
    valid[-(max(horizons) + 5):] = False
    if valid.sum() < 500:
        return None

    lv = htf_levels_for_base(d, level_tf)
    sup = lv["support"]
    g = htf_gate(d, hp)
    sig = base_signals(d, bp)

    lr = sig["long_sig"].to_numpy(bool)
    gate = g["htf_long_ok"].to_numpy(bool)
    hm = hammer(d, +1, atr=a, min_range_atr=0.3)
    br = break_reclaim(d, sup, +1, lookback=br_lookback,
                       min_pierce_atr=0.10, atr=a)
    near = np.isfinite(sup) & (distance_to_level(d, sup, a) <= near_atr)

    V = {
        "linreg":         lr,
        "linreg_htf":     lr & gate,
        "hammer":         hm,
        "br":             br,
        "br_hammer":      br & hm,
        "near_level":     near,
        "linreg_level":   lr & near,
        "linreg_br":      lr & br,
        "linreg_hammer":  lr & hm,
        "br_htf":         br & gate,
        "br_hammer_htf":  br & hm & gate,
    }
    V = {k: (v & valid) for k, v in V.items()}

    c = d["close"].to_numpy(float); o = d["open"].to_numpy(float)
    n = len(c)
    nxt = np.r_[o[1:], np.nan]
    fwd = {}
    for h in horizons:
        fc = np.full(n, np.nan); fc[:n - h] = c[h:]
        fwd[h] = (fc - nxt) / a
    return {"symbol": sym, "V": V, "valid": valid, "fwd": fwd}


def evaluate(base_tf="H1", htf_tf="D1", level_tf="D1", symbols=None,
             horizons=(6, 12, 24), bp: BaseParams | None = None,
             hp: HtfParams | None = None, verbose=True):
    bp = bp or BaseParams(chan_len=100, min_red_run=3, touch_src="raw",
                          touch_lookback=3, wt_level=40.0, wt_cross_lookback=3)
    hp = hp or HtfParams(tf=htf_tf, align=ALIGN_CLOSED, chan_len=50,
                         wt_level=50.0, band_src="linreg")
    symbols = symbols or stock_symbols(base_tf)

    acc = {}
    per_sym = {}
    for s in symbols:
        r = variants_for(s, base_tf, htf_tf, bp, hp, level_tf, horizons)
        if r is None:
            continue
        for name, mask in r["V"].items():
            if mask.sum() < 8:
                continue
            ctrl = r["valid"] & ~mask
            d = acc.setdefault(name, {h: ([], []) for h in horizons})
            ps = per_sym.setdefault(name, {h: [] for h in horizons})
            for h in horizons:
                f = r["fwd"][h]
                A = f[mask]; B = f[ctrl]
                A = A[np.isfinite(A)]; B = B[np.isfinite(B)]
                if len(A) < 8 or len(B) < 100:
                    continue
                d[h][0].append(A); d[h][1].append(B)
                ps[h].append(A.mean() - B.mean())

    rows = []
    for name, byh in acc.items():
        for h in horizons:
            As, Bs = byh[h]
            if not As:
                continue
            A = np.concatenate(As); B = np.concatenate(Bs)
            t, p = sps.ttest_ind(A, B, equal_var=False)
            lifts = np.array(per_sym[name][h])
            npos = int((lifts > 0).sum())
            st = sps.binomtest(npos, len(lifts), 0.5).pvalue if len(lifts) else np.nan
            rows.append({"variant": name, "h": h, "nA": len(A), "nB": len(B),
                         "meanA": A.mean(), "meanB": B.mean(),
                         "lift": A.mean() - B.mean(), "t": t, "p": p,
                         "syms_pos": npos, "syms": len(lifts), "sign_p": st,
                         "median_sym_lift": float(np.median(lifts)) if len(lifts) else np.nan})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import warnings, time
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 220)
    t0 = time.time()

    print("=" * 100)
    print("STOCKS  base H1 / HTF gate D1 / key levels from D1  -- LONG")
    print("=" * 100)
    r = evaluate("H1", "D1", "D1")
    for h in (6, 12, 24):
        print(f"\n--- horizon {h} bars ---")
        sub = r[r.h == h].sort_values("lift", ascending=False)
        print(sub[["variant", "nA", "meanA", "meanB", "lift", "t", "p",
                   "syms_pos", "syms", "sign_p", "median_sym_lift"]].round(4).to_string(index=False))
    r.to_parquet("sweeps/pa_stocks_H1_D1.parquet")
    print(f"\n[{time.time()-t0:.0f}s]")

    print("\n" + "=" * 100)
    print("GOLD  base M5 / HTF gate M30 / key levels from H4  -- LONG")
    print("=" * 100)
    rg = evaluate("M5", "M30", "H4", symbols=["XAUUSD"], horizons=(12, 24, 48))
    print(rg[["variant", "h", "nA", "meanA", "meanB", "lift", "t", "p"]]
          .sort_values(["h", "lift"], ascending=[True, False]).round(4).to_string(index=False))
    rg.to_parquet("sweeps/pa_gold_M5_M30.parquet")
    print(f"\n[{time.time()-t0:.0f}s]")
