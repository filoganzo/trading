"""
Indicators for the LinReg-Reversal strategy, ported 1:1 from the TradingView Pine
sources supplied by the user:

  * Humble LinReg Candles          (v4)  -> linreg_candles()
  * Linear Regression Channel      (v6)  -> lrc_channel()
  * WaveTrend [LazyBear]           (v4)  -> wavetrend()

Fidelity notes
--------------
- Pine ``ta.linreg(src, length, 0)`` is the least-squares fitted value AT THE
  CURRENT BAR of a regression over the trailing ``length`` bars. That is exactly
  what ``_reg_endpoint`` returns.
- The Pine channel indicator guards its maths behind ``barstate.islast`` so it
  only draws once, on the final bar. A backtest needs the same channel evaluated
  at *every* bar, so ``lrc_channel`` computes it on a rolling basis. This is the
  faithful per-bar generalisation, not a change of formula: at any given bar the
  numbers equal what Pine would print if that bar were the last one.
- Pine's ``stdDev`` divides the residual sum of squares by ``length - 1``
  (``periods``), i.e. ddof=1. Reproduced here. (Note: lrc_bot._lrc uses ddof=0 --
  a negligible but real difference at length 100; we follow Pine.)
- Pine's ``ta.ema`` seeds the recursion with an SMA of the first ``length``
  values rather than with the raw first value. Reproduced in ``_ema``.

All rolling regressions are computed with cumulative sums (O(n), not O(n*L)).
Prices are demeaned first so the float64 cumsum stays well-conditioned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- primitives

def _roll_sum(a: np.ndarray, L: int) -> np.ndarray:
    """
    Trailing rolling sum of window L. NaN where the window is not fully
    populated with finite samples.

    NaNs must be zero-filled before the cumsum: a single NaN poisons every
    later cumsum entry, which would silently turn downstream series such as
    ``wt2 = sma(wt1, 4)`` into all-NaN (and therefore produce zero signals).
    """
    a = np.asarray(a, dtype=float)
    n = len(a)
    out = np.full(n, np.nan, dtype=float)
    if L > n:
        return out
    valid = np.isfinite(a)
    cs = np.cumsum(np.insert(np.where(valid, a, 0.0), 0, 0.0))
    cv = np.cumsum(np.insert(valid.astype(np.int64), 0, 0))
    tot = cs[L:] - cs[:-L]
    cnt = cv[L:] - cv[:-L]
    out[L - 1:] = np.where(cnt == L, tot, np.nan)
    return out


def _reg_parts(y: np.ndarray, L: int):
    """
    Rolling OLS of y on x = 0..L-1 (x=L-1 is the current bar).

    Returns (slope, endpoint, sse) where ``endpoint`` is the fitted value at the
    current bar and ``sse`` is the residual sum of squares over the window.
    """
    y = np.asarray(y, dtype=float)
    off = float(np.nanmean(y)) if len(y) else 0.0
    yd = y - off                                    # condition the cumsums

    x = np.arange(L, dtype=float)
    Sx, Sxx = x.sum(), (x * x).sum()
    Sxx_c = Sxx - Sx * Sx / L                       # centred sum of squares in x

    Sy = _roll_sum(yd, L)
    Syy = _roll_sum(yd * yd, L)
    # Sxy needs sum(j * y[i-L+1+j]); a correlation against the ramp kernel.
    Sxy = np.full(len(yd), np.nan)
    ydz = np.where(np.isfinite(yd), yd, 0.0)   # NaN-safe convolution
    conv = np.convolve(ydz, x[::-1], mode="full")[L - 1:len(yd)]
    Sxy[L - 1:] = conv
    Sxy = np.where(np.isfinite(Sy), Sxy, np.nan)

    slope = (Sxy - Sx * Sy / L) / Sxx_c
    ybar = Sy / L
    intercept = ybar - slope * (Sx / L)
    endpoint = intercept + slope * (L - 1) + off    # undo the demean

    Syy_c = Syy - Sy * Sy / L
    sse = np.maximum(Syy_c - slope * slope * Sxx_c, 0.0)
    return slope, endpoint, sse


def _reg_endpoint(y: np.ndarray, L: int) -> np.ndarray:
    """Pine ``ta.linreg(y, L, 0)``."""
    return _reg_parts(y, L)[1]


def _ema(a: np.ndarray, L: int) -> np.ndarray:
    """
    Pine ``ta.ema``: alpha = 2/(L+1), recursion seeded with the SMA.

    Leading NaNs are skipped rather than poisoning the whole series -- the
    WaveTrend chain feeds one EMA into the next, so ``d = ema(abs(ap - esa))``
    starts with n1-1 NaNs by construction and a naive seed would return all-NaN.
    """
    a = np.asarray(a, dtype=float)
    n = len(a)
    out = np.full(n, np.nan)
    valid = np.isfinite(a)
    if valid.sum() < L:
        return out
    start = int(np.argmax(valid))            # first finite sample
    # first index at which L consecutive finite samples are available
    seed = None
    run = 0
    for i in range(start, n):
        run = run + 1 if valid[i] else 0
        if run >= L:
            seed = i
            break
    if seed is None:
        return out
    alpha = 2.0 / (L + 1.0)
    out[seed] = a[seed - L + 1:seed + 1].mean()
    for i in range(seed + 1, n):
        out[i] = (alpha * a[i] + (1.0 - alpha) * out[i - 1]
                  if valid[i] else out[i - 1])
    return out


def _sma(a: np.ndarray, L: int) -> np.ndarray:
    return _roll_sum(np.asarray(a, dtype=float), L) / L


# ---------------------------------------------------------------- indicators

def linreg_candles(df: pd.DataFrame, length: int = 11, signal_length: int = 11,
                   sma_signal: bool = True, lin_reg: bool = True) -> pd.DataFrame:
    """Humble LinReg Candles. Adds bopen/bhigh/blow/bclose, lrc_green, signal."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    if lin_reg:
        bo, bh, bl, bc = (_reg_endpoint(v, length) for v in (o, h, l, c))
    else:
        bo, bh, bl, bc = o, h, l, c
    sig = _sma(bc, signal_length) if sma_signal else _ema(bc, signal_length)
    return pd.DataFrame(
        {"bopen": bo, "bhigh": bh, "blow": bl, "bclose": bc,
         # Pine: r = bopen < bclose -> green, everything else red.
         "lrc_green": bo < bc,
         "lrc_signal": sig},
        index=df.index,
    )


def lrc_channel(df: pd.DataFrame, length: int = 100, mult: float = 1.3,
                source: str = "close") -> pd.DataFrame:
    """
    Rolling Linear Regression Channel (Pine v6 indicator, per-bar).

    Returns mid / upper / lower / stdev / dev_sigma / pearson_r, where
    ``dev_sigma`` is (source - mid) / stdev -- how many sigmas price sits from
    the regression line, the natural continuous form of the band test.
    """
    src = df[source].to_numpy(float)
    L = int(length)
    slope, mid, sse = _reg_parts(src, L)
    stdev = np.sqrt(sse / max(L - 1, 1))

    with np.errstate(invalid="ignore", divide="ignore"):
        dev_sigma = (src - mid) / stdev
        # Pearson r of the fit: r^2 = 1 - SSE/SST, signed by the slope.
        Sy = _roll_sum(src - np.nanmean(src), L)
        Syy = _roll_sum((src - np.nanmean(src)) ** 2, L)
        sst = Syy - Sy * Sy / L
        r2 = np.clip(1.0 - sse / np.where(sst > 0, sst, np.nan), 0.0, 1.0)
        pearson_r = np.sign(slope) * np.sqrt(r2)

    return pd.DataFrame(
        {"lrc_mid": mid, "lrc_upper": mid + mult * stdev,
         "lrc_lower": mid - mult * stdev, "lrc_stdev": stdev,
         "lrc_slope": slope, "dev_sigma": dev_sigma, "pearson_r": pearson_r},
        index=df.index,
    )


def wavetrend(df: pd.DataFrame, n1: int = 10, n2: int = 21) -> pd.DataFrame:
    """WaveTrend [LazyBear]. Returns wt1, wt2, wt_hist."""
    hlc3 = (df["high"].to_numpy(float) + df["low"].to_numpy(float)
            + df["close"].to_numpy(float)) / 3.0
    esa = _ema(hlc3, n1)
    d = _ema(np.abs(hlc3 - esa), n1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ci = (hlc3 - esa) / (0.015 * d)
    ci = np.where(np.isfinite(ci), ci, np.nan)
    wt1 = _ema(ci, n2)
    wt2 = _sma(wt1, 4)
    return pd.DataFrame({"wt1": wt1, "wt2": wt2, "wt_hist": wt1 - wt2},
                        index=df.index)


def atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """Wilder ATR."""
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    n = len(c)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    pc = np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    out[period] = tr[1:period + 1].mean()
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out
