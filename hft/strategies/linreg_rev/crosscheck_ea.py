"""
Reconcile the MQL5 EA against the Python engine.

This re-implements LinRegRevGold.mq5's ComputeSetup() in Python, following the
EA's loop structure literally (scan window, run counting, sinceLo/sinceHi, the
WaveTrend cross window) rather than the vectorised formulation used by
engine.make_signals. If the two agree bar for bar, the port is faithful and any
tester discrepancy is an execution-model difference, not a signal bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import Params, load, add_indicators, make_signals
from indicators import _reg_parts


def _reg_at(y: np.ndarray, end: int, L: int):
    """Scalar OLS endpoint/slope/stdev over y[end-L+1 .. end] -- mirrors RegEndpoint()."""
    if L < 2 or end < L - 1:
        return 0.0, 0.0, 0.0
    w = y[end - L + 1:end + 1]
    n = float(L)
    x = np.arange(L, dtype=float)
    sx, sy = x.sum(), w.sum()
    sxx, sxy, syy = (x * x).sum(), (x * w).sum(), (w * w).sum()
    sxx_c = sxx - sx * sx / n
    if abs(sxx_c) < 1e-12:
        return sy / n, 0.0, 0.0
    slope = (sxy - sx * sy / n) / sxx_c
    intercept = sy / n - slope * (sx / n)
    endpoint = intercept + slope * (n - 1.0)
    sse = max(syy - sy * sy / n - slope * slope * sxx_c, 0.0)
    return endpoint, slope, np.sqrt(sse / (n - 1.0))


def ea_setup_at(o, h, l, c, wt1, wt2, cur: int, p: Params, scan: int):
    """Return (lower_setup, upper_setup) exactly as the EA computes them."""
    if cur - scan + 1 < max(p.chan_len, p.cand_len) - 1:
        return False, False

    green = np.empty(scan, bool)
    blow = np.empty(scan)
    bhigh = np.empty(scan)
    lowB = np.empty(scan)
    upB = np.empty(scan)
    slp = np.empty(scan)
    for k in range(scan):
        idx = cur - k
        bo = _reg_at(o, idx, p.cand_len)[0]
        bc = _reg_at(c, idx, p.cand_len)[0]
        green[k] = bo < bc
        blow[k] = _reg_at(l, idx, p.cand_len)[0]
        bhigh[k] = _reg_at(h, idx, p.cand_len)[0]
        mid, s, sd = _reg_at(c, idx, p.chan_len)
        lowB[k] = mid - p.chan_mult * sd
        upB[k] = mid + p.chan_mult * sd
        slp[k] = s

    red_run = 0
    for k in range(1, scan):
        if green[k]:
            break
        red_run += 1
    green_run = 0
    for k in range(1, scan):
        if not green[k]:
            break
        green_run += 1

    since_lo = since_hi = -1
    for k in range(scan):
        idx = cur - k
        lo_src = l[idx] if p.touch_src == "raw" else blow[k]
        hi_src = h[idx] if p.touch_src == "raw" else bhigh[k]
        if since_lo < 0 and lo_src <= lowB[k]:
            since_lo = k
        if since_hi < 0 and hi_src >= upB[k]:
            since_hi = k
    lo_lim = p.touch_lookback if p.touch_lookback > 0 else red_run
    hi_lim = p.touch_lookback if p.touch_lookback > 0 else green_run
    lo_ok = 0 <= since_lo <= lo_lim
    hi_ok = 0 <= since_hi <= hi_lim

    lb = max(1, p.wt_cross_lookback)
    bull = bear = False
    for k in range(lb):
        i = cur - k
        if i < 1 or not np.isfinite([wt1[i], wt2[i], wt1[i - 1], wt2[i - 1]]).all():
            continue
        if wt1[i] > wt2[i] and wt1[i - 1] <= wt2[i - 1] and wt1[i] <= -p.wt_level:
            bull = True
        if wt1[i] < wt2[i] and wt1[i - 1] >= wt2[i - 1] and wt1[i] >= p.wt_level:
            bear = True

    lower = bool(green[0] and red_run >= p.min_red_run and lo_ok and bull)
    upper = bool((not green[0]) and green_run >= p.min_red_run and hi_ok and bear)
    if p.require_counter_slope:
        if slp[0] >= 0:
            lower = False
        if slp[0] <= 0:
            upper = False
    return lower, upper


def reconcile(p: Params, symbol: str = "XAUUSD", n_bars: int = 12000) -> dict:
    d_full = add_indicators(load(symbol, "M5"), p, symbol)
    sig = make_signals(d_full, p)
    scan = p.min_red_run + max(p.touch_lookback, p.wt_cross_lookback) + 80

    o, h, l, c = (d_full[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    wt1, wt2 = d_full["wt1"].to_numpy(float), d_full["wt2"].to_numpy(float)
    n = len(d_full)
    lo_v = sig["long_sig"].to_numpy(bool)
    up_v = sig["short_sig"].to_numpy(bool)

    start = max(n - n_bars, p.chan_len + scan + 5)
    mism_lo = mism_up = 0
    checked = 0
    examples = []
    for cur in range(start, n - 1):
        el, eu = ea_setup_at(o, h, l, c, wt1, wt2, cur, p, scan)
        checked += 1
        if el != lo_v[cur]:
            mism_lo += 1
            if len(examples) < 5:
                examples.append(("lower", cur, str(d_full["utc"].iloc[cur]), el, bool(lo_v[cur])))
        if eu != up_v[cur]:
            mism_up += 1
            if len(examples) < 5:
                examples.append(("upper", cur, str(d_full["utc"].iloc[cur]), eu, bool(up_v[cur])))
    return {"bars_checked": checked,
            "vec_lower_signals": int(lo_v[start:n - 1].sum()),
            "vec_upper_signals": int(up_v[start:n - 1].sum()),
            "mismatch_lower": mism_lo, "mismatch_upper": mism_up,
            "examples": examples}


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    p = Params(chan_len=200, min_red_run=6, touch_src="raw", touch_lookback=3,
               wt_level=50.0, wt_cross_lookback=3, require_counter_slope=True,
               mode="momentum", allow_long=True, allow_short=False,
               max_bars=24, sl_atr=2.0, tp_atr=3.0,
               use_silver=True, silver_mode="zone", silver_lookback=6)
    print("Reconciling EA logic vs vectorised engine (deployed config)...")
    r = reconcile(p, "XAUUSD", n_bars=12000)
    for k, v in r.items():
        if k != "examples":
            print(f"  {k:>20}: {v}")
    if r["examples"]:
        print("  first mismatches (side, bar, utc, ea, vectorised):")
        for e in r["examples"]:
            print("   ", e)
    else:
        print("  -> EXACT MATCH on every bar checked.")

    print("\nAlso checking a touch_lookback=0 config (run-length path):")
    p2 = Params(chan_len=100, min_red_run=3, touch_src="linreg", touch_lookback=0,
                wt_level=40.0, wt_cross_lookback=1, require_counter_slope=False)
    r2 = reconcile(p2, "XAUUSD", n_bars=8000)
    for k, v in r2.items():
        if k != "examples":
            print(f"  {k:>20}: {v}")
    if r2["examples"]:
        print("  first mismatches:", r2["examples"])
    else:
        print("  -> EXACT MATCH on every bar checked.")
