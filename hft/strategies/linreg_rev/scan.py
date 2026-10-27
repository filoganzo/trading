"""
Exit-agnostic entry-quality scan.

Sweeping stop/target first is the fastest way to overfit: with ~1,200 signals
there is always some (SL, TP) pair that looks good. So the first question is
whether the ENTRY carries any directional information at all, measured without
any exit rule:

    drift_h    mean signed forward return h bars after the fill, in ATR units
    mfe_h      mean max favourable excursion within h bars, in ATR units
    mae_h      mean max adverse excursion within h bars, in ATR units
    edge_h     mfe_h / |mae_h|   (>1 means the move goes our way first)
    t_stat     drift_h / stderr  -- is the drift distinguishable from zero

Everything is reported IS / OOS on a single chronological split so a filter that
only works on the fitting half is visible immediately.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd

from engine import (Params, load, add_indicators, make_signals, build_paths,
                    silver_gate, POINT)

warnings.filterwarnings("ignore")

HORIZONS = (3, 6, 12, 24, 48)


def entry_quality(symbol: str, p: Params, horizons=HORIZONS,
                  split_utc: str | None = "2026-02-01") -> dict:
    """
    One row of entry diagnostics for a parameter set, IS/OOS split.

    Forward returns are measured from the realistic fill (next bar's open plus
    the full spread) and normalised by ATR at the signal bar, so the number is
    comparable across the whole 17-month sample.
    """
    d = add_indicators(load(symbol, "M5"), p, symbol)
    sig = make_signals(d, p)
    long_sig = sig["long_sig"].to_numpy(bool).copy()
    short_sig = sig["short_sig"].to_numpy(bool).copy()
    if p.use_silver:
        g = silver_gate(d, p)
        long_sig &= g["sv_long"].to_numpy(bool)
        short_sig &= g["sv_short"].to_numpy(bool)

    hz = max(horizons)
    point = POINT[symbol]
    atr_arr = d["atr"].to_numpy(float)
    spread = d["spread"].to_numpy(float)
    utc = d["utc"].to_numpy()
    o = d["open"].to_numpy(float)

    recs = []
    for side, mask in ((+1, long_sig), (-1, short_sig)):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        H, L, C, eo = build_paths(d, idx, hz)
        entry = eo + side * spread[np.minimum(idx + 1, len(d) - 1)] * point
        a = atr_arr[idx]
        base = {"side": side, "utc": utc[idx], "atr": a, "entry": entry}
        for h in horizons:
            hh, ll, cc = H[:, :h], L[:, :h], C[:, :h]
            with np.errstate(invalid="ignore"):
                last = np.where(np.isfinite(cc), cc, np.nan)
                # last finite close within the horizon
                fwd = np.full(len(idx), np.nan)
                for j in range(h - 1, -1, -1):
                    need = ~np.isfinite(fwd)
                    if not need.any():
                        break
                    fwd[need] = last[need, j]
                base[f"drift_{h}"] = side * (fwd - entry) / a
                if side > 0:
                    base[f"mfe_{h}"] = (np.nanmax(hh, axis=1) - entry) / a
                    base[f"mae_{h}"] = (np.nanmin(ll, axis=1) - entry) / a
                else:
                    base[f"mfe_{h}"] = (entry - np.nanmin(ll, axis=1)) / a
                    base[f"mae_{h}"] = (entry - np.nanmax(hh, axis=1)) / a
        recs.append(pd.DataFrame(base))

    if not recs:
        return {}
    t = pd.concat(recs).reset_index(drop=True)
    t["dir"] = np.where(t["side"] > 0, "long", "short")
    return t


def summarise(t: pd.DataFrame, horizons=HORIZONS, label: str = "") -> dict:
    out = {"label": label, "n": int(len(t))}
    if len(t) == 0:
        return out
    for h in horizons:
        dr = t[f"drift_{h}"].to_numpy(float)
        dr = dr[np.isfinite(dr)]
        mfe = np.nanmean(t[f"mfe_{h}"])
        mae = np.nanmean(t[f"mae_{h}"])
        out[f"drift_{h}"] = float(np.mean(dr)) if len(dr) else np.nan
        out[f"t_{h}"] = (float(np.mean(dr) / (np.std(dr, ddof=1) / np.sqrt(len(dr))))
                         if len(dr) > 5 and np.std(dr, ddof=1) > 0 else np.nan)
        out[f"edge_{h}"] = float(mfe / abs(mae)) if mae and abs(mae) > 1e-9 else np.nan
    return out


def is_oos(t: pd.DataFrame, split: str = "2026-02-01", horizons=HORIZONS):
    cut = pd.Timestamp(split)
    a = summarise(t[t["utc"] < cut], horizons, "IS")
    b = summarise(t[t["utc"] >= cut], horizons, "OOS")
    return a, b
