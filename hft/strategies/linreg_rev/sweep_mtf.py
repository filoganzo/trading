"""
Stage A: entry + HTF-confirmation sweep for one (base, htf) timeframe pair.

Scored EXIT-AGNOSTICALLY on three things, because with a grid this size any
P&L ranking would mostly be selecting luck:

  exc{h}   excess drift: mean forward move h base-bars out, in ATR units, minus
           the unconditional drift of the instrument over the same horizon.
           If this is zero the post-signal path is a martingale and no exit rule
           can extract expectancy from it.
  edge{h}  mean MFE / mean |MAE| -- path asymmetry, which pure drift misses and
           which trims and trailing stops can monetise.
  pmid{h}  P(price reaches the channel MIDLINE before going 1R against). This is
           the strategy's actual thesis stated as a measurable quantity: a
           reversion trade is a bet on getting back inside the channel.

Everything is reported IS / OOS on one chronological split, and every config
carries its signal count so thin winners can be discarded on sight.

Usage:
    python sweep_mtf.py --symbol XAUUSD --base M5 --htf M15 --out out.parquet
"""
from __future__ import annotations

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

from mtf_runner import (BaseParams, base_indicators, base_signals, load_tf,
                        POINT, DEFAULT_POINT)
from mtf import htf_gate, HtfParams, ALIGN_FORMING, ALIGN_CLOSED

HZ = (6, 12, 24, 48)


class Fwd:
    """Forward max/min/close per horizon plus the channel path. Price-only."""

    def __init__(self, d: pd.DataFrame, horizons=HZ):
        h, l, c = (d[x].to_numpy(float) for x in ("high", "low", "close"))
        n = len(d)
        self.n = n
        self.open = d["open"].to_numpy(float)
        self.spread = d["spread"].to_numpy(float) if "spread" in d else np.zeros(n)
        self.atr = d["atr"].to_numpy(float)
        self.mid = d["lrc_mid"].to_numpy(float)
        hs, ls = pd.Series(h), pd.Series(l)
        self.fmax, self.fmin, self.fclose, self.fmidmax, self.fmidmin = {}, {}, {}, {}, {}
        ms = pd.Series(self.mid)
        for k in horizons:
            self.fmax[k] = hs.rolling(k).max().shift(-k).to_numpy()
            self.fmin[k] = ls.rolling(k).min().shift(-k).to_numpy()
            fc = np.full(n, np.nan); fc[:n - k] = c[k:]
            self.fclose[k] = fc
            # the midline itself drifts; track its range over the window so the
            # "did we get back to the middle" test uses a reachable level
            self.fmidmax[k] = ms.rolling(k).max().shift(-k).to_numpy()
            self.fmidmin[k] = ms.rolling(k).min().shift(-k).to_numpy()


def uncond(d: pd.DataFrame, horizons=HZ) -> dict:
    c = d["close"].to_numpy(float); a = d["atr"].to_numpy(float); n = len(c)
    out = {}
    for k in horizons:
        f = np.full(n, np.nan); f[:n - k] = c[k:] - c[:n - k]
        z = (f / a); z = z[np.isfinite(z)]
        out[k] = float(z.mean()) if len(z) else np.nan
    return out


def pmid_before_adverse(d, idx, side, entry, atr, hz_bars: int):
    """
    P(touch the channel midline before trading 1 ATR against the entry).

    Walks the path bar by bar for the signal set only, so it stays cheap.
    """
    n = len(d)
    h = d["high"].to_numpy(float); l = d["low"].to_numpy(float)
    mid = d["lrc_mid"].to_numpy(float)
    stop = entry - side * atr
    hit = np.zeros(len(idx), dtype=np.int8)     # 1 mid first, -1 adverse first
    for j in range(hz_bars):
        src = idx + 1 + j
        m = (src < n) & (hit == 0)
        if not m.any():
            break
        s = src[m]
        tgt = mid[s]
        if side > 0:
            reach = h[s] >= tgt
            adverse = l[s] <= stop[m]
        else:
            reach = l[s] <= tgt
            adverse = h[s] >= stop[m]
        res = np.where(adverse, -1, np.where(reach, 1, 0)).astype(np.int8)
        hh = hit.copy(); hh[m] = res; hit = hh
    decided = hit != 0
    return float((hit[decided] == 1).mean()) if decided.any() else np.nan


def score(d, fwd, idx, side, base, utc, split, symbol, horizons=HZ,
          want_pmid=True) -> dict:
    if len(idx) == 0:
        return {}
    point = POINT.get(symbol, DEFAULT_POINT)
    a = fwd.atr[idx]
    nxt = np.minimum(idx + 1, fwd.n - 1)
    entry = fwd.open[nxt] + side * fwd.spread[nxt] * point
    ism = utc[idx] < split
    out = {"n": int(len(idx)), "n_is": int(ism.sum()), "n_oos": int((~ism).sum())}
    for k in horizons:
        drift = side * (fwd.fclose[k][idx] - entry) / a
        fx, fn = fwd.fmax[k][idx], fwd.fmin[k][idx]
        if side > 0:
            mfe, mae = (fx - entry) / a, (fn - entry) / a
        else:
            mfe, mae = (entry - fn) / a, (entry - fx) / a
        b = side * base[k]
        for tag, msk in (("", np.isfinite(drift)),
                         ("_is", ism & np.isfinite(drift)),
                         ("_oos", (~ism) & np.isfinite(drift))):
            z = drift[msk]
            if len(z) < 6 or z.std(ddof=1) == 0:
                out[f"exc{k}{tag}"] = np.nan; out[f"t{k}{tag}"] = np.nan
                continue
            e = z.mean() - b
            out[f"exc{k}{tag}"] = float(e)
            out[f"t{k}{tag}"] = float(e / (z.std(ddof=1) / np.sqrt(len(z))))
        mm = np.isfinite(mfe) & np.isfinite(mae)
        out[f"edge{k}"] = (float(np.mean(mfe[mm]) / abs(np.mean(mae[mm])))
                           if mm.sum() > 5 and abs(np.mean(mae[mm])) > 1e-12 else np.nan)
    if want_pmid:
        out["pmid24"] = pmid_before_adverse(d, idx, side, entry, a, 24)
    return out


BASE_GRID = {
    "chan_len":          [50, 100, 200],
    "min_red_run":       [2, 3, 4, 6],
    "touch_src":         ["linreg", "raw"],
    "wt_level":          [30.0, 40.0, 50.0],
    "wt_cross_lookback": [1, 3],
}
HTF_GRID = {
    "chan_len":  [50, 100, 200],
    "wt_level":  [30.0, 40.0, 50.0, 60.0],
    "band_src":  ["linreg", "raw", "close"],
    "align":     [ALIGN_FORMING, ALIGN_CLOSED],
}


def run_pair(symbol: str, base_tf: str, htf_tf: str | None, split: str,
             out_path: str, verbose: bool = True) -> pd.DataFrame:
    t0 = time.time()
    ref = base_indicators(symbol, base_tf, BaseParams())
    fwd = Fwd(ref)
    base_u = uncond(ref)
    utc = ref["utc"].to_numpy()
    split_ts = np.datetime64(pd.Timestamp(split))

    bkeys = list(BASE_GRID); hkeys = list(HTF_GRID)
    bcombos = list(itertools.product(*(BASE_GRID[k] for k in bkeys)))
    hcombos = [()] if htf_tf is None else list(
        itertools.product(*(HTF_GRID[k] for k in hkeys)))

    # HTF gates depend only on the HTF grid, so compute each once and reuse it
    gates = {}
    if htf_tf is not None:
        for hc in hcombos:
            hkw = dict(zip(hkeys, hc))
            hp = HtfParams(tf=htf_tf, **hkw)
            g = htf_gate(ref, hp)
            gates[hc] = (g["htf_long_ok"].to_numpy(bool),
                         g["htf_short_ok"].to_numpy(bool))
        if verbose:
            print(f"  built {len(gates)} HTF gates in {time.time()-t0:.1f}s", flush=True)

    rows = []
    for bi, bc in enumerate(bcombos):
        bkw = dict(zip(bkeys, bc))
        bp = BaseParams(**bkw)
        d = base_indicators(symbol, base_tf, bp)
        sig = base_signals(d, bp)
        ls = sig["long_sig"].to_numpy(bool)
        ss = sig["short_sig"].to_numpy(bool)
        fwd_b = fwd if bp.chan_len == BaseParams().chan_len else Fwd(d)
        bu = base_u if bp.chan_len == BaseParams().chan_len else uncond(d)
        for hc in hcombos:
            if htf_tf is None:
                L, S = ls, ss
                hkw = {}
            else:
                gl, gs = gates[hc]
                L, S = ls & gl, ss & gs
                hkw = {f"htf_{k}": v for k, v in zip(hkeys, hc)}
            for side, mask, dname in ((+1, L, "long"), (-1, S, "short")):
                idx = np.where(mask)[0]
                if len(idx) < 30:
                    continue
                m = score(d, fwd_b, idx, side, bu, utc, split_ts, symbol)
                if m:
                    rows.append({"symbol": symbol, "base_tf": base_tf,
                                 "htf_tf": htf_tf or "none", "dir": dname,
                                 **bkw, **hkw, **m})
        if verbose and (bi + 1) % 24 == 0:
            print(f"  base combo {bi+1}/{len(bcombos)}  rows={len(rows)}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path)
    if verbose:
        print(f"  DONE {base_tf}/{htf_tf}: {len(df)} rows in {time.time()-t0:.0f}s "
              f"-> {out_path}", flush=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--base", required=True)
    ap.add_argument("--htf", default=None)
    ap.add_argument("--split", default=None,
                    help="IS/OOS cut; default = 60%% through the sample")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    htf = None if a.htf in (None, "none", "None") else a.htf
    if a.split is None:
        d0 = load_tf(a.symbol, a.base)
        t0, t1 = d0["utc"].iloc[0], d0["utc"].iloc[-1]
        a.split = str((t0 + (t1 - t0) * 0.6).date())
    out = a.out or os.path.join("sweeps", f"{a.symbol}_{a.base}_{htf or 'none'}.parquet")
    print(f"Stage A sweep: {a.symbol} base={a.base} htf={htf} split={a.split}")
    run_pair(a.symbol, a.base, htf, a.split, out)
