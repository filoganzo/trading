"""
Entry-space sweep, scored on EXCESS drift rather than on P&L.

Why excess drift
----------------
Gold rose ~42% across this sample. Any rule that is net long makes money and any
rule that is net short loses, regardless of whether it carries information. So a
parameter set is scored on

    excess_h = mean(signal drift over h bars)  -  side * mean(unconditional drift)

both in ATR units. excess_h > 0 with a supporting t-stat is the only thing that
distinguishes a real signal from directional beta. P&L sweeps come later, and
only for configurations that clear this bar.

Every score is reported IS and OOS on one chronological split.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from engine import (Params, load, add_indicators, make_signals, silver_gate,
                    POINT)

HZ = (6, 12, 24, 48)
SPLIT = pd.Timestamp("2026-02-01")


# --------------------------------------------------- precomputed forward paths

class Forward:
    """Forward max/min/close over each horizon; independent of any parameter."""

    def __init__(self, d: pd.DataFrame, horizons=HZ):
        h = d["high"].to_numpy(float)
        l = d["low"].to_numpy(float)
        c = d["close"].to_numpy(float)
        n = len(d)
        self.n = n
        self.open = d["open"].to_numpy(float)
        self.spread = d["spread"].to_numpy(float)
        self.close = c
        self.fmax, self.fmin, self.fclose = {}, {}, {}
        hs, ls = pd.Series(h), pd.Series(l)
        for k in horizons:
            # max/min over bars i+1 .. i+k
            self.fmax[k] = hs.rolling(k).max().shift(-k).to_numpy()
            self.fmin[k] = ls.rolling(k).min().shift(-k).to_numpy()
            fc = np.full(n, np.nan)
            fc[:n - k] = c[k:]
            self.fclose[k] = fc


def uncond_drift(d: pd.DataFrame, horizons=HZ, mask: np.ndarray | None = None) -> dict:
    """Mean unconditional forward drift in ATR units, per horizon."""
    c = d["close"].to_numpy(float)
    a = d["atr"].to_numpy(float)
    n = len(c)
    out = {}
    for k in horizons:
        fwd = np.full(n, np.nan)
        fwd[:n - k] = c[k:] - c[:n - k]
        z = fwd / a
        if mask is not None:
            z = z[mask]
        z = z[np.isfinite(z)]
        out[k] = float(z.mean()) if len(z) else np.nan
    return out


# ------------------------------------------------------------------- scoring

def score(d: pd.DataFrame, fwd: Forward, idx: np.ndarray, side: int,
          symbol: str, base: dict, utc: np.ndarray, horizons=HZ) -> dict:
    """Excess-drift metrics for one set of signal indices on one side."""
    if len(idx) == 0:
        return {}
    point = POINT[symbol]
    a = d["atr"].to_numpy(float)[idx]
    nxt = np.minimum(idx + 1, fwd.n - 1)
    entry = fwd.open[nxt] + side * fwd.spread[nxt] * point
    is_mask = utc[idx] < SPLIT

    out = {"n": int(len(idx)), "n_is": int(is_mask.sum()),
           "n_oos": int((~is_mask).sum())}
    for k in horizons:
        fc = fwd.fclose[k][idx]
        drift = side * (fc - entry) / a
        fx, fn = fwd.fmax[k][idx], fwd.fmin[k][idx]
        if side > 0:
            mfe, mae = (fx - entry) / a, (fn - entry) / a
        else:
            mfe, mae = (entry - fn) / a, (entry - fx) / a
        b = side * base[k]
        for tag, m in (("", np.isfinite(drift)),
                       ("_is", is_mask & np.isfinite(drift)),
                       ("_oos", (~is_mask) & np.isfinite(drift))):
            z = drift[m]
            if len(z) < 6 or z.std(ddof=1) == 0:
                out[f"exc{k}{tag}"] = np.nan
                out[f"t{k}{tag}"] = np.nan
                continue
            exc = z.mean() - b
            out[f"exc{k}{tag}"] = float(exc)
            out[f"t{k}{tag}"] = float(exc / (z.std(ddof=1) / np.sqrt(len(z))))
        mm = np.isfinite(mfe) & np.isfinite(mae)
        out[f"edge{k}"] = (float(np.mean(mfe[mm]) / abs(np.mean(mae[mm])))
                           if mm.sum() > 5 and abs(np.mean(mae[mm])) > 1e-12 else np.nan)
    return out


# --------------------------------------------------------------------- sweep

GRID = {
    "chan_len":            [30, 50, 75, 100, 150, 200, 300],
    "min_red_run":         [2, 3, 4, 5, 6, 8],
    "touch_src":           ["linreg", "raw"],
    "touch_lookback":      [0, 3],
    "wt_level":            [30.0, 40.0, 50.0],
    "wt_cross_lookback":   [1, 3],
    "require_counter_slope": [False, True],
}


def run_sweep(symbol: str = "XAUUSD", grid: dict | None = None,
              horizons=HZ, verbose: bool = True) -> pd.DataFrame:
    grid = grid or GRID
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    base_df = load(symbol, "M5")

    # forward paths and the unconditional benchmark depend only on price
    d_ref = add_indicators(base_df, Params(), symbol)
    fwd = Forward(d_ref, horizons)
    base = uncond_drift(d_ref, horizons)
    utc = d_ref["utc"].to_numpy()

    rows = []
    for ci, combo in enumerate(combos):
        kw = dict(zip(keys, combo))
        p = Params(**kw)
        d = add_indicators(base_df, p, symbol)
        sig = make_signals(d, p)
        for side, col, dname in ((+1, "long_sig", "long"), (-1, "short_sig", "short")):
            idx = np.where(sig[col].to_numpy(bool))[0]
            m = score(d, fwd, idx, side, symbol, base, utc, horizons)
            if m:
                rows.append({**kw, "dir": dname, **m})
        if verbose and (ci + 1) % 200 == 0:
            print(f"  ... {ci + 1}/{len(combos)} combos", flush=True)
    return pd.DataFrame(rows)
