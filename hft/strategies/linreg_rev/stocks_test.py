"""
Stocks: the same MTF reversal test, pooled across the equity cross-section.

Why stocks matter here beyond being the next asset class
--------------------------------------------------------
Gold's verdict was not "no signal" but "not enough trades to prove one": 215
unique trades, an expectancy whose bootstrap CI spans zero, and a result that
flips negative if two winners are removed. That is a POWER problem, and the
cross-section is the fix -- 41 symbols with largely independent idiosyncratic
moves supply far more effectively independent trades than one instrument can.

Per the user's spec: H1 entry with H4/D1 confirmation, and D1 entry with W1.
"""
from __future__ import annotations

import glob
import itertools
import os
import time

import numpy as np
import pandas as pd
from scipy import stats as sps

from mtf_runner import (DATA_DIR, BaseParams, base_indicators, base_signals,
                        load_tf)
from mtf import htf_gate, HtfParams, ALIGN_CLOSED
from management import Mgmt
from mgmt_sweep import run_on_indices


def stock_symbols(tf: str = "H1") -> list[str]:
    fs = sorted(glob.glob(os.path.join(DATA_DIR, f"*_{tf}.parquet")))
    out = []
    for f in fs:
        s = os.path.basename(f).replace(f"_{tf}.parquet", "")
        if s.startswith(("XAU", "XAG")):
            continue
        out.append(s)
    return out


def did_symbol(sym, base_tf, htf_tf, bp: BaseParams, hp: HtfParams,
               horizons=(6, 12, 24), side=+1):
    """A/B forward returns for one symbol, in ATR units."""
    try:
        d = base_indicators(sym, base_tf, bp)
    except Exception:
        return None
    if len(d) < bp.chan_len + 150:
        return None
    g = htf_gate(d, hp)
    sig = base_signals(d, bp)
    trig = (sig["long_sig"] if side > 0 else sig["short_sig"]).to_numpy(bool)
    gate = (g["htf_long_ok"] if side > 0 else g["htf_short_ok"]).to_numpy(bool)
    valid = (np.isfinite(d["atr"].to_numpy(float))
             & np.isfinite(d["lrc_lower"].to_numpy(float))
             & np.isfinite(d["wt2"].to_numpy(float)))
    valid[-(max(horizons) + 5):] = False

    c = d["close"].to_numpy(float); o = d["open"].to_numpy(float)
    a = d["atr"].to_numpy(float); n = len(c)
    nxt = np.r_[o[1:], np.nan]
    rows = {}
    A = valid & trig & gate
    B = valid & (~trig) & gate
    if A.sum() < 8 or B.sum() < 60:
        return None
    for h in horizons:
        fc = np.full(n, np.nan); fc[:n - h] = c[h:]
        fwd = side * (fc - nxt) / a
        rows[h] = (fwd[A], fwd[B])
    return {"symbol": sym, "nA": int(A.sum()), "nB": int(B.sum()), "fwd": rows}


def run_did(base_tf="H1", htf_tf="D1", side=+1, horizons=(6, 12, 24),
            bp: BaseParams | None = None, hp: HtfParams | None = None,
            verbose=True):
    bp = bp or BaseParams(chan_len=100, min_red_run=3, touch_src="raw",
                          touch_lookback=3, wt_level=40.0, wt_cross_lookback=3)
    hp = hp or HtfParams(tf=htf_tf, align=ALIGN_CLOSED, chan_len=50,
                         wt_level=50.0, band_src="linreg")
    syms = stock_symbols(base_tf)
    per, poolA, poolB = [], {h: [] for h in horizons}, {h: [] for h in horizons}
    for s in syms:
        r = did_symbol(s, base_tf, htf_tf, bp, hp, horizons, side)
        if r is None:
            continue
        row = {"symbol": s, "nA": r["nA"], "nB": r["nB"]}
        for h in horizons:
            a, b = r["fwd"][h]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            poolA[h].append(a); poolB[h].append(b)
            row[f"A{h}"] = a.mean() if len(a) else np.nan
            row[f"B{h}"] = b.mean() if len(b) else np.nan
            row[f"d{h}"] = row[f"A{h}"] - row[f"B{h}"]
        per.append(row)
    per = pd.DataFrame(per)
    if len(per) == 0:
        if verbose:
            print("  no symbols produced enough signals at these parameters")
        return per, pd.DataFrame()
    if verbose:
        print(f"  {len(per)} symbols usable, total A={per['nA'].sum()} "
              f"B={per['nB'].sum()}")
    summary = []
    for h in horizons:
        a = np.concatenate(poolA[h]); b = np.concatenate(poolB[h])
        t, p = sps.ttest_ind(a, b, equal_var=False)
        dcol = per[f"d{h}"].dropna()
        # symbol-level sign test: how many of the 41 names point the same way
        st = sps.binomtest(int((dcol > 0).sum()), len(dcol), 0.5) if len(dcol) else None
        summary.append({"h": h, "nA": len(a), "nB": len(b),
                        "meanA": a.mean(), "meanB": b.mean(),
                        "A_minus_B": a.mean() - b.mean(), "t": t, "p": p,
                        "symbols_positive": int((dcol > 0).sum()),
                        "symbols": len(dcol),
                        "sign_test_p": st.pvalue if st else np.nan})
    return per, pd.DataFrame(summary)


def run_mgmt(base_tf="H1", htf_tf="D1", side=+1, mg: Mgmt | None = None,
             bp: BaseParams | None = None, hp: HtfParams | None = None,
             n_control_per_sym=600, seed=23, verbose=True):
    """Pooled trades plus a matched random-entry control, across all symbols."""
    rng = np.random.default_rng(seed)
    bp = bp or BaseParams(chan_len=100, min_red_run=3, touch_src="raw",
                          touch_lookback=3, wt_level=40.0, wt_cross_lookback=3)
    hp = hp or HtfParams(tf=htf_tf, align=ALIGN_CLOSED, chan_len=50,
                         wt_level=50.0, band_src="linreg")
    mg = mg or Mgmt(stop_mode="atr", sl_atr=1.5, tp_mode="none", max_bars=48)
    real, ctrl = [], []
    for s in stock_symbols(base_tf):
        try:
            d = base_indicators(s, base_tf, bp)
        except Exception:
            continue
        if len(d) < bp.chan_len + 400:
            continue
        g = htf_gate(d, hp)
        sig = base_signals(d, bp)
        trig = (sig["long_sig"] if side > 0 else sig["short_sig"]).to_numpy(bool)
        gate = (g["htf_long_ok"] if side > 0 else g["htf_short_ok"]).to_numpy(bool)
        valid = (np.isfinite(d["atr"].to_numpy(float))
                 & np.isfinite(d["lrc_lower"].to_numpy(float))
                 & np.isfinite(d["wt2"].to_numpy(float)))
        valid[-(mg.max_bars + 5):] = False
        ridx = np.where(trig & gate & valid)[0]
        pool = np.where((~trig) & gate & valid)[0]
        if len(ridx) < 10 or len(pool) < 200:
            continue
        rr = run_on_indices(d, ridx, side, mg, s, commission_per_lot_rt=0.0,
                            extra_slippage_points=0.0)
        cidx = np.sort(rng.choice(pool, size=min(n_control_per_sym, len(pool)),
                                  replace=False))
        cc = run_on_indices(d, cidx, side, mg, s, commission_per_lot_rt=0.0)
        if not rr or not cc:
            continue
        real.append(pd.DataFrame({"symbol": s, "r": rr["r"], "bars": rr["bars"],
                                  "reason": rr["reason"],
                                  "utc": d["utc"].to_numpy()[rr["idx"]]}))
        ctrl.append(cc["r"])
    if not real:
        return pd.DataFrame(), np.array([])
    return pd.concat(real).reset_index(drop=True), np.concatenate(ctrl)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200)
    t0 = time.time()
    for base_tf, htf_tf, hz in (("H1", "D1", (6, 12, 24)),
                                ("H1", "H4", (6, 12, 24)),
                                ("D1", "W1", (3, 5, 10))):
        print("=" * 78)
        print(f"STOCKS DiD -- base {base_tf} / confirmation {htf_tf}, LONG")
        print("=" * 78)
        per, summ = run_did(base_tf, htf_tf, +1, hz)
        print(summ.round(4).to_string(index=False))
        print(f"  [{time.time()-t0:.0f}s]")
