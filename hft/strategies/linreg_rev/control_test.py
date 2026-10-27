"""
Difference-in-differences identification of where (if anywhere) the edge lives.

The problem with plain excess drift
-----------------------------------
`sweep_mtf.py` scores a signal against ONE unconditional drift constant for the
whole sample. Over Apr-2025..Aug-2026 gold rose 42%, so any filter that
preferentially fires in bullish regimes prints positive long "excess" for free.
The HTF confirmation is exactly such a filter -- the long/short decomposition
showed its benefit is almost entirely the anti-symmetric (directional) component.
So plain excess drift cannot distinguish "the setup predicts reversion" from
"the filter found the uptrend".

The fix: condition on the regime the filter selects
---------------------------------------------------
Four cells, all measured on the same instrument over the same bars:

                        base setup FIRES      base setup does NOT fire
    HTF gate ON              A                          B
    HTF gate OFF             C                          D

  trigger_value_gated   = A - B    the base trigger's marginal contribution
                                   WITH the HTF regime held constant
  trigger_value_ungated = C - D    the same, outside that regime
  DiD                   = (A-B) - (C-D)
                                   how much the HTF context changes what the
                                   trigger is worth

B and D are the load-bearing part: they are bars in the same regime, on the same
side, where the trigger did NOT fire. If A - B is zero, the trigger carries no
information beyond the regime it fired in, and no exit rule can fix that.

A reversion trade also needs its target to be worth the risk, so this module
additionally records the distance from entry to the channel MIDLINE in ATR at
signal time. Without it, "P(reach midline before 1 ATR against)" is unreadable:
at a 0.5 ATR target against a 1.0 ATR stop, breakeven is p = 0.667, so a
measured 0.60 is a LOSING bet, not a winning one.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats as sps

from mtf_runner import BaseParams, base_indicators, base_signals, POINT, DEFAULT_POINT
from mtf import htf_gate, HtfParams, ALIGN_CLOSED, ALIGN_FORMING


def forward_returns(d: pd.DataFrame, horizons=(12, 24, 48)) -> dict:
    """Signed forward return in ATR units, per bar, measured from the NEXT open."""
    c = d["close"].to_numpy(float)
    o = d["open"].to_numpy(float)
    a = d["atr"].to_numpy(float)
    n = len(c)
    nxt = np.r_[o[1:], np.nan]              # fill price for a signal on bar i
    out = {}
    for h in horizons:
        fc = np.full(n, np.nan)
        fc[:n - h] = c[h:]
        out[h] = (fc - nxt) / a             # positive = price rose
    return out


def midline_distance(d: pd.DataFrame) -> np.ndarray:
    """|entry - channel midline| in ATR at signal time. The reversion target."""
    o = d["open"].to_numpy(float)
    nxt = np.r_[o[1:], np.nan]
    mid = d["lrc_mid"].to_numpy(float)
    a = d["atr"].to_numpy(float)
    return np.abs(mid - nxt) / a


def _mean_diff(x: np.ndarray, y: np.ndarray):
    """Welch mean difference with t and p. x = treatment, y = control."""
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 10 or len(y) < 10:
        return np.nan, np.nan, np.nan, len(x), len(y)
    t, p = sps.ttest_ind(x, y, equal_var=False)
    return float(x.mean() - y.mean()), float(t), float(p), len(x), len(y)


def did_one(d, fwd, mdist, base_sig, gate, side, horizons=(12, 24, 48)) -> dict:
    """
    One (base config, HTF config, direction) -> the 2x2 and its contrasts.

    ``base_sig`` and ``gate`` are boolean arrays over base bars.
    """
    valid = np.isfinite(d["atr"].to_numpy(float)) & np.isfinite(
        d["lrc_lower"].to_numpy(float)) & np.isfinite(d["wt2"].to_numpy(float))
    A = valid & base_sig & gate
    B = valid & ~base_sig & gate
    C = valid & base_sig & ~gate
    D = valid & ~base_sig & ~gate

    out = {"nA": int(A.sum()), "nB": int(B.sum()),
           "nC": int(C.sum()), "nD": int(D.sum())}
    if out["nA"] < 30 or out["nC"] < 30:
        return {}
    out["mid_dist_atr"] = float(np.nanmedian(mdist[A]))

    for h in horizons:
        r = side * fwd[h]                    # signed so positive = trade wins
        ab, tab, pab, na, nb = _mean_diff(r[A], r[B])
        cd, tcd, pcd, nc, nd = _mean_diff(r[C], r[D])
        out[f"trig_gated_{h}"] = ab
        out[f"t_gated_{h}"] = tab
        out[f"p_gated_{h}"] = pab
        out[f"trig_ungated_{h}"] = cd
        out[f"t_ungated_{h}"] = tcd
        out[f"did_{h}"] = (ab - cd) if (np.isfinite(ab) and np.isfinite(cd)) else np.nan
        out[f"meanA_{h}"] = float(np.nanmean(r[A]))
        out[f"meanB_{h}"] = float(np.nanmean(r[B]))
    return out


def reversion_expectancy(d, idx, side, stop_atr: float = 1.0,
                         horizon: int = 24) -> dict:
    """
    The reversion bet priced properly.

    Walks the path to see whether the channel midline is reached before the stop,
    and pairs that probability with the ACTUAL target distance, so the result is
    an expectancy in ATR rather than a bare probability.
    """
    n = len(d)
    h = d["high"].to_numpy(float); l = d["low"].to_numpy(float)
    o = d["open"].to_numpy(float); mid = d["lrc_mid"].to_numpy(float)
    a = d["atr"].to_numpy(float)
    entry = np.r_[o[1:], np.nan][idx]
    atr_e = a[idx]
    stop = entry - side * stop_atr * atr_e
    outcome = np.zeros(len(idx), dtype=np.int8)
    gain = np.full(len(idx), np.nan)

    for j in range(horizon):
        src = idx + 1 + j
        m = (src < n) & (outcome == 0)
        if not m.any():
            break
        s = src[m]
        tgt = mid[s]
        if side > 0:
            reach, adverse = h[s] >= tgt, l[s] <= stop[m]
        else:
            reach, adverse = l[s] <= tgt, h[s] >= stop[m]
        res = np.where(adverse, -1, np.where(reach, 1, 0)).astype(np.int8)
        g = np.where(reach & ~adverse, side * (tgt - entry[m]) / atr_e[m], np.nan)
        oo = outcome.copy(); oo[m] = res; outcome = oo
        gg = gain.copy(); gg[m] = np.where(np.isfinite(g), g, gg[m]); gain = gg

    dec = outcome != 0
    if dec.sum() < 20:
        return {}
    p_win = float((outcome[dec] == 1).mean())
    avg_gain = float(np.nanmean(gain[outcome == 1])) if (outcome == 1).any() else np.nan
    exp_atr = p_win * avg_gain - (1 - p_win) * stop_atr if np.isfinite(avg_gain) else np.nan
    breakeven_p = stop_atr / (avg_gain + stop_atr) if np.isfinite(avg_gain) and (avg_gain + stop_atr) > 0 else np.nan
    return {"p_mid_first": p_win, "avg_target_atr": avg_gain,
            "breakeven_p": breakeven_p, "expectancy_atr": exp_atr,
            "n_decided": int(dec.sum()),
            "edge_vs_breakeven": (p_win - breakeven_p) if np.isfinite(breakeven_p) else np.nan}


BASE_GRID = {
    "chan_len":          [50, 100, 200],
    "min_red_run":       [2, 3, 4, 6],
    "touch_src":         ["linreg", "raw"],
    "wt_level":          [30.0, 40.0, 50.0],
    "wt_cross_lookback": [1, 3],
}
HTF_GRID = {
    "chan_len":  [50, 100, 200],
    "wt_level":  [40.0, 50.0, 60.0],
    "band_src":  ["linreg", "close"],
    "align":     [ALIGN_CLOSED, ALIGN_FORMING],
}


def run(symbol: str, base_tf: str, htf_tf: str, out_path: str | None = None,
        horizons=(12, 24, 48), verbose=True) -> pd.DataFrame:
    import os, time
    t0 = time.time()
    ref = base_indicators(symbol, base_tf, BaseParams())
    bkeys, hkeys = list(BASE_GRID), list(HTF_GRID)
    hcombos = list(itertools.product(*(HTF_GRID[k] for k in hkeys)))
    bcombos = list(itertools.product(*(BASE_GRID[k] for k in bkeys)))

    gates = {}
    for hc in hcombos:
        hp = HtfParams(tf=htf_tf, **dict(zip(hkeys, hc)))
        g = htf_gate(ref, hp)
        gates[hc] = (g["htf_long_ok"].to_numpy(bool), g["htf_short_ok"].to_numpy(bool))
    if verbose:
        print(f"  {len(gates)} gates in {time.time()-t0:.0f}s", flush=True)

    rows = []
    for bi, bc in enumerate(bcombos):
        bkw = dict(zip(bkeys, bc))
        bp = BaseParams(**bkw)
        d = base_indicators(symbol, base_tf, bp)
        sig = base_signals(d, bp)
        fwd = forward_returns(d, horizons)
        mdist = midline_distance(d)
        ls = sig["long_sig"].to_numpy(bool)
        ss = sig["short_sig"].to_numpy(bool)
        for hc in hcombos:
            gl, gs = gates[hc]
            hkw = {f"htf_{k}": v for k, v in zip(hkeys, hc)}
            for side, bs, gt, dn in ((+1, ls, gl, "long"), (-1, ss, gs, "short")):
                m = did_one(d, fwd, mdist, bs, gt, side, horizons)
                if not m:
                    continue
                rv = reversion_expectancy(d, np.where(bs & gt)[0], side)
                rows.append({"symbol": symbol, "base_tf": base_tf, "htf_tf": htf_tf,
                             "dir": dn, **bkw, **hkw, **m, **rv})
        if verbose and (bi + 1) % 36 == 0:
            print(f"  {bi+1}/{len(bcombos)} rows={len(rows)} {time.time()-t0:.0f}s",
                  flush=True)
    df = pd.DataFrame(rows)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path)
    if verbose:
        print(f"  DONE {base_tf}/{htf_tf}: {len(df)} rows in {time.time()-t0:.0f}s")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--base", default="M5")
    ap.add_argument("--htf", default="M30")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"sweeps/did_{a.symbol}_{a.base}_{a.htf}.parquet"
    print(f"Difference-in-differences: {a.symbol} {a.base} / {a.htf}")
    run(a.symbol, a.base, a.htf, out)
