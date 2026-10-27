"""
Multi-timeframe reversal runner.

Base timeframe supplies the TRIGGER (the full setup, exactly as originally
specified). The higher timeframe supplies CONTEXT (overextended in the same
direction; no colour flip and no cross required).

This is the reversal reading throughout -- the lower-band setup buys.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

from indicators import linreg_candles, lrc_channel, wavetrend, atr as atr_fn
from engine import _bars_since, _run_length_of_false, session_of, SERVER_UTC_OFFSET
from mtf import htf_gate, HtfParams, TF_MINUTES, ALIGN_FORMING
from management import Mgmt, simulate_managed, REASON_NAMES
import os

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "linreg_rev")

CONTRACT = {"XAUUSD": 100.0, "XAGUSD": 1000.0}
POINT = {"XAUUSD": 0.01, "XAGUSD": 0.001}
DEFAULT_CONTRACT, DEFAULT_POINT = 1.0, 0.01


@dataclass
class BaseParams:
    cand_len: int = 11
    chan_len: int = 100
    chan_mult: float = 1.3
    wt_n1: int = 10
    wt_n2: int = 21
    atr_period: int = 14
    min_red_run: int = 3
    touch_src: str = "linreg"        # 'linreg' | 'raw'
    touch_lookback: int = 0          # 0 => touch must lie inside the run
    wt_level: float = 40.0
    wt_cross_lookback: int = 1
    require_counter_slope: bool = False


# Sized to hold one full equity cross-section (41 symbols) for a given
# chan_len, so a sweep that varies the other parameters does not recompute
# indicators for every symbol on every config.
_BASE_CACHE: dict = {}
_RAW_CACHE: dict = {}


def load_tf(symbol: str, tf: str) -> pd.DataFrame:
    k = (symbol, tf)
    if k not in _RAW_CACHE:
        df = pd.read_parquet(os.path.join(DATA_DIR, f"{symbol}_{tf}.parquet"))
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="s")
        df["utc"] = df["dt"] - pd.Timedelta(hours=SERVER_UTC_OFFSET)
        df["utc_hour"] = df["utc"].dt.hour
        df["session"] = session_of(df["utc_hour"].to_numpy())
        _RAW_CACHE[k] = df
    return _RAW_CACHE[k]


def base_indicators(symbol: str, tf: str, bp: BaseParams) -> pd.DataFrame:
    k = (symbol, tf, bp.cand_len, bp.chan_len, bp.chan_mult, bp.wt_n1, bp.wt_n2,
         bp.atr_period)
    if k in _BASE_CACHE:
        return _BASE_CACHE[k]
    d = load_tf(symbol, tf).copy()
    d = pd.concat([d, linreg_candles(d, length=bp.cand_len)], axis=1)
    d = pd.concat([d, lrc_channel(d, length=bp.chan_len, mult=bp.chan_mult)], axis=1)
    d = pd.concat([d, wavetrend(d, bp.wt_n1, bp.wt_n2)], axis=1)
    d["atr"] = atr_fn(d, bp.atr_period)
    if len(_BASE_CACHE) > 60:
        _BASE_CACHE.clear()
    _BASE_CACHE[k] = d
    return d


def base_signals(d: pd.DataFrame, bp: BaseParams) -> pd.DataFrame:
    """The full setup on the entry timeframe. Reversal reading."""
    green = d["lrc_green"].to_numpy(bool)
    red = ~green
    red_run = _run_length_of_false(green)
    green_run = _run_length_of_false(red)

    if bp.touch_src == "linreg":
        lo_src, hi_src = d["blow"].to_numpy(float), d["bhigh"].to_numpy(float)
    else:
        lo_src, hi_src = d["low"].to_numpy(float), d["high"].to_numpy(float)
    lower, upper = d["lrc_lower"].to_numpy(float), d["lrc_upper"].to_numpy(float)
    since_lo = _bars_since(lo_src <= lower)
    since_hi = _bars_since(hi_src >= upper)
    if bp.touch_lookback > 0:
        lo_ok, hi_ok = since_lo <= bp.touch_lookback, since_hi <= bp.touch_lookback
    else:
        lo_ok, hi_ok = since_lo <= red_run, since_hi <= green_run

    wt1, wt2 = d["wt1"].to_numpy(float), d["wt2"].to_numpy(float)
    pw1, pw2 = np.r_[np.nan, wt1[:-1]], np.r_[np.nan, wt2[:-1]]
    bull = (wt1 > wt2) & (pw1 <= pw2) & (wt1 <= -bp.wt_level)
    bear = (wt1 < wt2) & (pw1 >= pw2) & (wt1 >= bp.wt_level)
    lb = max(1, bp.wt_cross_lookback)
    wl = _bars_since(bull) <= lb - 1
    ws = _bars_since(bear) <= lb - 1

    long_sig = green & (red_run >= bp.min_red_run) & lo_ok & wl
    short_sig = red & (green_run >= bp.min_red_run) & hi_ok & ws
    if bp.require_counter_slope:
        sl = d["lrc_slope"].to_numpy(float)
        long_sig &= sl < 0
        short_sig &= sl > 0

    ok = (np.isfinite(d["atr"].to_numpy(float)) & np.isfinite(lower)
          & np.isfinite(wt2))
    ok[-1] = False
    return pd.DataFrame({"long_sig": long_sig & ok, "short_sig": short_sig & ok},
                        index=d.index)


# --------------------------------------------------------------- path matrices

def build_paths(d: pd.DataFrame, idx: np.ndarray, hz: int, want_bands=True,
                swing_lookback: int = 12):
    """Forward OHLC (+ channel) matrices starting on the FILL bar."""
    n = len(d)
    o, h, l, c = (d[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    k = len(idx)
    H = np.full((k, hz), np.nan); L = np.full((k, hz), np.nan)
    C = np.full((k, hz), np.nan); OP = np.full((k, hz), np.nan)
    MID = UP = LO = None
    mid0 = up0 = lo0 = None
    if want_bands:
        mid = d["lrc_mid"].to_numpy(float)
        up = d["lrc_upper"].to_numpy(float)
        lw = d["lrc_lower"].to_numpy(float)
        MID = np.full((k, hz), np.nan); UP = np.full((k, hz), np.nan)
        LO = np.full((k, hz), np.nan)
        # band values at the SIGNAL bar -- the last ones observable when the
        # order is placed, needed for band stops and for the lagged targets
        mid0, up0, lo0 = mid[idx], up[idx], lw[idx]
    for j in range(hz):
        src = idx + 1 + j
        m = src < n
        s = src[m]
        H[m, j] = h[s]; L[m, j] = l[s]; C[m, j] = c[s]; OP[m, j] = o[s]
        if want_bands:
            MID[m, j] = mid[s]; UP[m, j] = up[s]; LO[m, j] = lw[s]
    entry_open = np.full(k, np.nan)
    m = (idx + 1) < n
    entry_open[m] = o[idx[m] + 1]

    # structural stop: extreme of the preceding swing_lookback bars
    sw_lo = np.full(k, np.nan); sw_hi = np.full(k, np.nan)
    for a, i in enumerate(idx):
        s = max(0, i - swing_lookback + 1)
        sw_lo[a] = l[s:i + 1].min()
        sw_hi[a] = h[s:i + 1].max()
    return (H, L, C, OP, MID, UP, LO, entry_open, sw_lo, sw_hi,
            mid0, up0, lo0)


# ------------------------------------------------------------------- backtest

def run(symbol: str, base_tf: str, bp: BaseParams, hp: HtfParams | None,
        mg: Mgmt, commission_per_lot_rt: float = 7.0,
        extra_slippage_points: float = 0.0,
        allow_long: bool = True, allow_short: bool = True,
        sessions: tuple = ()) -> pd.DataFrame:
    """One full configuration -> a trades frame."""
    d = base_indicators(symbol, base_tf, bp)
    sig = base_signals(d, bp)
    long_sig = sig["long_sig"].to_numpy(bool).copy()
    short_sig = sig["short_sig"].to_numpy(bool).copy()

    if hp is not None:
        g = htf_gate(d, hp)
        long_sig &= g["htf_long_ok"].to_numpy(bool)
        short_sig &= g["htf_short_ok"].to_numpy(bool)
    if not allow_long:
        long_sig[:] = False
    if not allow_short:
        short_sig[:] = False
    if sessions:
        insess = d["session"].isin(list(sessions)).to_numpy(bool)
        long_sig &= insess
        short_sig &= insess

    point = POINT.get(symbol, DEFAULT_POINT)
    contract = CONTRACT.get(symbol, DEFAULT_CONTRACT)
    spread = d["spread"].to_numpy(float) if "spread" in d else np.zeros(len(d))
    atr_a = d["atr"].to_numpy(float)
    hz = int(mg.max_bars)

    rows = []
    for side, mask in ((+1, long_sig), (-1, short_sig)):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        (H, L, C, OP, MID, UP, LO, eo, sw_lo, sw_hi,
         mid0, up0, lo0) = build_paths(d, idx, hz,
                                       swing_lookback=mg.swing_lookback)
        cost = spread[np.minimum(idx + 1, len(d) - 1)] + extra_slippage_points
        entry = eo + side * cost * point
        a = atr_a[idx]
        swing = sw_lo if side > 0 else sw_hi

        # commission expressed in R needs the R unit, which depends on the stop
        res = simulate_managed(H, L, C, entry, a, side, mg,
                               MID=MID, UPPER=UP, LOWER=LO, SWING=swing, O=OP,
                               MID0=mid0, UPPER0=up0, LOWER0=lo0)
        comm_r = commission_per_lot_rt / (res["r_unit"] * contract)
        r_net = res["r"] - comm_r

        rows.append(pd.DataFrame({
            "i": idx,
            "utc": d["utc"].to_numpy()[idx],
            "session": d["session"].to_numpy()[idx],
            "side": side,
            "dir": "long" if side > 0 else "short",
            "entry": entry,
            "atr": a,
            "r_unit": res["r_unit"],
            "r_gross": res["r"],
            "r": r_net,
            "usd": res["r"] * res["r_unit"] * contract - commission_per_lot_rt,
            "bars": res["bars"],
            "reason": pd.Series(res["reason"]).map(REASON_NAMES).to_numpy(),
            "mfe_r": res["mfe_r"],
            "mae_r": res["mae_r"],
            "n_partials": res["n_partials"],
        }))
    if not rows:
        return pd.DataFrame()
    t = pd.concat(rows).sort_values("i").reset_index(drop=True)
    return t.dropna(subset=["r"]).reset_index(drop=True)


def stats(t: pd.DataFrame, label: str = "") -> dict:
    if t is None or len(t) == 0:
        return {"label": label, "n": 0, "expectancy_r": np.nan, "total_r": 0.0,
                "win_rate": np.nan, "profit_factor": np.nan, "max_dd_r": np.nan,
                "sharpe_pt": np.nan, "total_usd": 0.0}
    r = t["r"].to_numpy(float)
    eq = np.cumsum(r)
    dd = eq - np.maximum.accumulate(np.r_[0.0, eq])[1:]
    gp = r[r > 0].sum(); gl = -r[r <= 0].sum()
    sd = r.std(ddof=1) if len(r) > 2 else np.nan
    return {"label": label, "n": int(len(r)), "expectancy_r": float(r.mean()),
            "total_r": float(r.sum()), "win_rate": float((r > 0).mean()),
            "profit_factor": float(gp / gl) if gl > 0 else np.inf,
            "max_dd_r": float(dd.min()) if len(dd) else 0.0,
            "sharpe_pt": float(r.mean() / sd) if sd and sd > 0 else np.nan,
            "total_usd": float(t["usd"].sum()),
            "avg_bars": float(t["bars"].mean())}
