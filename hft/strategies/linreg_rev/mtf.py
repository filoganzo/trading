"""
Higher-timeframe confirmation, computed without look-ahead.

The trap this module exists to avoid
------------------------------------
At the close of the 10:05 M5 bar, the M15 bar covering 10:00-10:15 is one third
formed. Reading that bar's FINAL high/low/close - which is what naive MTF
backtests do when they join a base series to a downloaded HTF series on
timestamp - imports data from up to one full HTF bar into the future. It makes
almost any MTF filter look excellent and it does not survive contact with a live
market.

There are exactly two honest readings, and this module implements both:

  ALIGN_CLOSED   use the last HTF bar that has actually closed at or before the
                 base bar's close. Safe, but stale by up to one HTF bar.
  ALIGN_FORMING  use the partially-formed HTF bar, built only from base bars up
                 to and including the signal bar. This is what a trader sees on
                 the screen, and it is still strictly causal.

HTF bars are RESAMPLED FROM THE BASE SERIES rather than downloaded separately.
That removes a whole class of alignment bug (phantom bars, differing session
boundaries, broker gap handling) and guarantees the HTF and base views are
mutually consistent by construction.

Forming-bar indicators are computed incrementally: rolling sums over the
completed HTF bars are combined with the single partial bar's contribution, so
the whole thing stays O(n) and exact rather than O(n*L).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators import _roll_sum

ALIGN_CLOSED = "closed"
ALIGN_FORMING = "forming"

# timeframe label -> minutes. M45 has no native MT5 equivalent and is built here.
TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "M45": 45, "H1": 60,
              "H2": 120, "H4": 240, "D1": 1440, "W1": 10080}


# ------------------------------------------------------------------ bucketing

def bucket_ids(epoch_seconds: np.ndarray, htf_minutes: int) -> np.ndarray:
    """
    Calendar-aligned HTF bucket index for each base bar.

    Bucketing on absolute time (rather than counting N base bars) is what makes
    this correct across weekend gaps and missing bars: a bucket is a wall-clock
    window, so a gap simply produces a bucket with fewer base bars in it, never a
    silently mis-aligned window.
    """
    return (np.asarray(epoch_seconds, dtype=np.int64) // (int(htf_minutes) * 60))


def completed_htf(base: pd.DataFrame, htf_minutes: int):
    """
    Aggregate the base series into completed HTF bars.

    Returns (htf frame indexed 0..M-1, bucket_pos per base bar), where
    ``bucket_pos[i]`` is the ORDINAL position of base bar i's bucket in the HTF
    frame (not the raw bucket id), so it can index the HTF arrays directly.
    """
    b = bucket_ids(base["time"].to_numpy(), htf_minutes)
    g = pd.DataFrame({"b": b,
                      "open": base["open"].to_numpy(float),
                      "high": base["high"].to_numpy(float),
                      "low": base["low"].to_numpy(float),
                      "close": base["close"].to_numpy(float)})
    agg = g.groupby("b", sort=True).agg(open=("open", "first"),
                                        high=("high", "max"),
                                        low=("low", "min"),
                                        close=("close", "last"))
    uniq = agg.index.to_numpy()
    pos = np.searchsorted(uniq, b)            # base bar -> ordinal HTF index
    agg = agg.reset_index(drop=True)
    return agg, pos


def running_partial(base: pd.DataFrame, bucket_pos: np.ndarray):
    """
    The partially-formed HTF bar as of each base bar: open of the bucket so far,
    running high/low, and the current close. Strictly causal.
    """
    df = pd.DataFrame({"p": bucket_pos,
                       "o": base["open"].to_numpy(float),
                       "h": base["high"].to_numpy(float),
                       "l": base["low"].to_numpy(float),
                       "c": base["close"].to_numpy(float)})
    grp = df.groupby("p", sort=False)
    return (grp["o"].transform("first").to_numpy(),
            grp["h"].cummax().to_numpy(),
            grp["l"].cummin().to_numpy(),
            df["c"].to_numpy())


# ------------------------------------------------- forming-bar regression math

def _ramp_sum(a: np.ndarray, W: int) -> np.ndarray:
    """sum_{j=0..W-1} j * a[i-W+1+j]  -- trailing window of length W."""
    n = len(a)
    out = np.full(n, np.nan)
    if W < 1 or W > n:
        return out
    x = np.arange(W, dtype=float)
    az = np.where(np.isfinite(a), a, 0.0)
    conv = np.convolve(az, x[::-1], mode="full")[W - 1:n]
    out[W - 1:] = conv
    return out


def reg_forming(completed: np.ndarray, partial: np.ndarray, pos: np.ndarray,
                L: int):
    """
    Rolling regression over a window whose newest element is the PARTIAL bar.

    ``completed``  HTF series over completed bars (length M)
    ``partial``    the partial bar's value for each base bar (length n)
    ``pos``        base bar -> ordinal index of its (still forming) HTF bucket
    ``L``          regression length in HTF bars

    Window = completed[pos-L+1 .. pos-1]  (L-1 bars, x = 0..L-2)
             + partial                      (x = L-1)

    Returns (endpoint, slope, stdev) per base bar.
    """
    M = len(completed)
    W = L - 1                                    # completed bars in the window
    n = float(L)
    if W < 1:
        raise ValueError("L must be >= 2")

    off = float(np.nanmean(completed)) if M else 0.0
    cc = completed - off                         # condition the sums
    S1 = _roll_sum(cc, W)                        # sum of completed part
    SR = _ramp_sum(cc, W)                        # ramp-weighted sum, weights 0..W-1
    S2 = _roll_sum(cc * cc, W)

    k = pos - 1                                  # last COMPLETED bucket
    ok = (k >= W - 1) & (k < M)
    ki = np.where(ok, k, 0)

    p = partial - off
    Sy = np.where(ok, S1[ki] + p, np.nan)
    Sxy = np.where(ok, SR[ki] + (L - 1) * p, np.nan)
    Syy = np.where(ok, S2[ki] + p * p, np.nan)

    Sx = (L - 1) * L / 2.0
    Sxx = (L - 1) * L * (2 * L - 1) / 6.0
    Sxx_c = Sxx - Sx * Sx / n

    slope = (Sxy - Sx * Sy / n) / Sxx_c
    ybar = Sy / n
    endpoint = ybar - slope * (Sx / n) + slope * (L - 1) + off
    sse = np.maximum(Syy - Sy * Sy / n - slope * slope * Sxx_c, 0.0)
    stdev = np.sqrt(sse / (L - 1))
    return endpoint, slope, stdev


def reg_closed(completed: np.ndarray, pos: np.ndarray, L: int):
    """Same, but evaluated on the last CLOSED HTF bar (index pos-1)."""
    from indicators import _reg_parts
    slope_c, end_c, sse_c = _reg_parts(completed, L)
    std_c = np.sqrt(sse_c / max(L - 1, 1))
    M = len(completed)
    k = pos - 1
    ok = (k >= 0) & (k < M)
    ki = np.where(ok, k, 0)
    nan = np.nan
    return (np.where(ok, end_c[ki], nan),
            np.where(ok, slope_c[ki], nan),
            np.where(ok, std_c[ki], nan))


# ----------------------------------------------------- forming-bar WaveTrend

def wavetrend_forming(chlc: dict, completed: pd.DataFrame, pos: np.ndarray,
                      n1: int, n2: int):
    """
    WaveTrend on the forming HTF bar.

    The EMA chain is recursive, so the forming value is one more EMA step taken
    from the completed-bar state -- exactly what a live chart shows.
    """
    from indicators import _ema, _sma

    ap_c = (completed["high"].to_numpy(float) + completed["low"].to_numpy(float)
            + completed["close"].to_numpy(float)) / 3.0
    esa_c = _ema(ap_c, n1)
    d_c = _ema(np.abs(ap_c - esa_c), n1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ci_c = (ap_c - esa_c) / (0.015 * d_c)
    ci_c = np.where(np.isfinite(ci_c), ci_c, np.nan)
    wt1_c = _ema(ci_c, n2)
    wt2_c = _sma(wt1_c, 4)
    s3 = _roll_sum(wt1_c, 3)                    # for the forming SMA(4)

    M = len(completed)
    k = pos - 1
    ok = (k >= 0) & (k < M)
    ki = np.where(ok, k, 0)

    a1 = 2.0 / (n1 + 1.0)
    a2 = 2.0 / (n2 + 1.0)
    ap_p = (chlc["h"] + chlc["l"] + chlc["c"]) / 3.0

    esa_prev = np.where(ok, esa_c[ki], np.nan)
    d_prev = np.where(ok, d_c[ki], np.nan)
    wt1_prev = np.where(ok, wt1_c[ki], np.nan)
    wt2_prev = np.where(ok, wt2_c[ki], np.nan)

    esa_p = a1 * ap_p + (1 - a1) * esa_prev
    d_p = a1 * np.abs(ap_p - esa_p) + (1 - a1) * d_prev
    with np.errstate(invalid="ignore", divide="ignore"):
        ci_p = (ap_p - esa_p) / (0.015 * d_p)
    ci_p = np.where(np.isfinite(ci_p), ci_p, np.nan)
    wt1_p = a2 * ci_p + (1 - a2) * wt1_prev
    wt2_p = (np.where(ok, s3[ki], np.nan) + wt1_p) / 4.0
    return wt1_p, wt2_p, wt1_prev, wt2_prev


# ----------------------------------------------------------------- the gate

@dataclass
class HtfParams:
    tf: str = "M15"
    align: str = ALIGN_FORMING
    cand_len: int = 11
    chan_len: int = 100
    chan_mult: float = 1.3
    wt_n1: int = 10
    wt_n2: int = 21
    band_src: str = "linreg"      # 'linreg' (blow/bhigh) | 'raw' | 'close'
    band_mode: str = "touch"      # 'touch' (wick beyond) | 'close_beyond'
    wt_level: float = 40.0        # |wt1| must be at least this extreme
    require_color: bool = False   # user: NOT required
    require_cross: bool = False   # user: NOT required


def htf_gate(base: pd.DataFrame, hp: HtfParams) -> pd.DataFrame:
    """
    Per base bar: is the higher timeframe overextended in the direction that
    would support a reversal trade?

    long_ok  : HTF stretched DOWN  (at/below the lower band) and WaveTrend low
    short_ok : HTF stretched UP    (at/above the upper band) and WaveTrend high

    Deliberately NOT required (per the strategy definition): an opposite-colour
    HTF candle close, or an HTF WaveTrend cross. The higher timeframe supplies
    context; the base timeframe supplies the trigger.
    """
    mins = TF_MINUTES[hp.tf]
    comp, pos = completed_htf(base, mins)
    po, ph, pl, pc = running_partial(base, pos)
    M = len(comp)

    co = comp["open"].to_numpy(float)
    ch = comp["high"].to_numpy(float)
    cl = comp["low"].to_numpy(float)
    cc = comp["close"].to_numpy(float)

    if hp.align == ALIGN_FORMING:
        bo = reg_forming(co, po, pos, hp.cand_len)[0]
        bh = reg_forming(ch, ph, pos, hp.cand_len)[0]
        bl = reg_forming(cl, pl, pos, hp.cand_len)[0]
        bc = reg_forming(cc, pc, pos, hp.cand_len)[0]
        mid, slope, sd = reg_forming(cc, pc, pos, hp.chan_len)
        wt1, wt2, wt1_prev, wt2_prev = wavetrend_forming(
            {"h": ph, "l": pl, "c": pc}, comp, pos, hp.wt_n1, hp.wt_n2)
        raw_hi, raw_lo, raw_cl = ph, pl, pc
    else:
        bo = reg_closed(co, pos, hp.cand_len)[0]
        bh = reg_closed(ch, pos, hp.cand_len)[0]
        bl = reg_closed(cl, pos, hp.cand_len)[0]
        bc = reg_closed(cc, pos, hp.cand_len)[0]
        mid, slope, sd = reg_closed(cc, pos, hp.chan_len)
        from indicators import wavetrend as _wt
        w = _wt(comp, hp.wt_n1, hp.wt_n2)
        w1a, w2a = w["wt1"].to_numpy(float), w["wt2"].to_numpy(float)
        k = pos - 1
        ok = (k >= 0) & (k < M)
        ki = np.where(ok, k, 0)
        kj = np.where(ok & (k >= 1), k - 1, 0)
        wt1 = np.where(ok, w1a[ki], np.nan)
        wt2 = np.where(ok, w2a[ki], np.nan)
        wt1_prev = np.where(ok & (k >= 1), w1a[kj], np.nan)
        wt2_prev = np.where(ok & (k >= 1), w2a[kj], np.nan)
        idx = np.where(ok, ki, 0)
        raw_hi = np.where(ok, ch[idx], np.nan)
        raw_lo = np.where(ok, cl[idx], np.nan)
        raw_cl = np.where(ok, cc[idx], np.nan)

    upper = mid + hp.chan_mult * sd
    lower = mid - hp.chan_mult * sd

    if hp.band_src == "linreg":
        lo_probe, hi_probe = bl, bh
    elif hp.band_src == "raw":
        lo_probe, hi_probe = raw_lo, raw_hi
    else:
        lo_probe = hi_probe = raw_cl

    if hp.band_mode == "close_beyond":
        stretched_dn = raw_cl < lower
        stretched_up = raw_cl > upper
    else:
        stretched_dn = lo_probe <= lower
        stretched_up = hi_probe >= upper

    wt_dn = wt1 <= -hp.wt_level
    wt_up = wt1 >= hp.wt_level

    long_ok = stretched_dn & wt_dn
    short_ok = stretched_up & wt_up

    if hp.require_color:
        green = bo < bc
        long_ok &= green
        short_ok &= ~green
    if hp.require_cross:
        long_ok &= (wt1 > wt2) & (wt1_prev <= wt2_prev)
        short_ok &= (wt1 < wt2) & (wt1_prev >= wt2_prev)

    valid = np.isfinite(mid) & np.isfinite(wt1) & np.isfinite(sd) & (sd > 0)
    long_ok &= valid
    short_ok &= valid

    return pd.DataFrame({"htf_long_ok": long_ok, "htf_short_ok": short_ok,
                         "htf_mid": mid, "htf_upper": upper, "htf_lower": lower,
                         "htf_wt1": wt1, "htf_slope": slope,
                         "htf_bucket": pos}, index=base.index)
