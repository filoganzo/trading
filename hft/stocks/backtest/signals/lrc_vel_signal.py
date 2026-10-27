"""
Signal engine for the v3 LRC strategy (no WaveTrend).

Conditions:
  1. LRC candle close (linreg-15 -> EMA-9) outside 120-bar sigma channel
  2. Price AT a significant historical level (within touch_pct%, scored by touches)
  3. Daily-bar velocity gate: range fading OR reversal candle OR wick rejection

Shared computation helpers imported from wt_lrc_signal to avoid duplication.
"""
import math
import numpy as np

from stocks.backtest.signals.wt_lrc_signal import (
    _lrc_vectorized,
    _lrc_candle_close_vectorized,
    _atr_vectorized,
    _scored_level_at,
)


def _velocity_arrays(high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, open_: np.ndarray,
                     lookback: int = 5):
    """
    Vectorised daily-bar velocity / acceleration confirmation arrays.

    Returns (vel_short, vel_long): per-bar bool — True when the bar confirms
    a reversal at the level in that direction.

    SHORT confirmed (price at resistance):
        fading range  OR  bearish candle  OR  upper-wick rejection
    LONG confirmed (price at support):
        fading range  OR  bullish candle  OR  lower-wick rejection
    """
    n          = len(close)
    bar_range  = high - low
    body       = np.abs(close - open_)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low

    # Prior-N-bar average range (excludes current bar)
    avg_range = np.full(n, np.nan)
    if n > lookback:
        # sliding_window_view(arr, k)[j] covers arr[j : j+k]
        # avg_range[i] = mean(bar_range[i-lookback : i]) = avgs[i-lookback]
        avgs = np.lib.stride_tricks.sliding_window_view(bar_range, lookback).mean(axis=1)
        avg_range[lookback:] = avgs[:n - lookback]

    fading = np.zeros(n, dtype=bool)
    valid  = ~np.isnan(avg_range) & (avg_range > 1e-8)
    fading[valid] = bar_range[valid] < 0.75 * avg_range[valid]

    short_rev = close < open_   # bearish close = reversal at resistance
    long_rev  = close > open_   # bullish close = reversal at support

    # Wick threshold: 1.5x body, floored at 10% of bar_range for near-doji
    wick_thresh = np.maximum(1.5 * body, 0.10 * bar_range)
    short_wick  = upper_wick > wick_thresh   # upper wick = higher prices rejected
    long_wick   = lower_wick > wick_thresh   # lower wick = lower prices rejected

    vel_short = fading | short_rev | short_wick
    vel_long  = fading | long_rev  | long_wick

    return vel_short, vel_long


def find_signals_v3(precomp: dict, lrc_length: int, sigma_mult: float,
                    touch_pct: float, sl_atr_mult: float,
                    tp_rr: float,
                    min_score: float = 40.0, lookback: int = 120):
    """
    V3 fast signal finder: LRC candle gate + velocity gate + scored level.

    precomp must contain: close, high, low, open, lrc_c, atr,
                          vel_short, vel_long,
                          lrc_data[(lrc_length, sigma_mult)] = (mid, upper, lower)

    Returns list of (bar_idx, direction, sl, tp_rr, signal_close, readiness).
    Readiness = ext_score * 0.5 + level_score * 0.5   (both 0-100 scale).
    """
    c         = precomp["close"];  h = precomp["high"]; l = precomp["low"]
    lrc_c     = precomp["lrc_c"]
    atr       = precomp["atr"]
    vel_short = precomp["vel_short"]
    vel_long  = precomp["vel_long"]
    mid, upper, lower = precomp["lrc_data"][(lrc_length, sigma_mult)]
    n = precomp["n"]

    touch_frac = touch_pct / 100.0
    min_i      = max(lrc_length, 15 + 9 + 5, 14 + 5)
    if min_i >= n:
        return []

    idx = np.arange(min_i, n)

    # Gate 1: LRC candle outside channel
    lrc_ci = lrc_c[idx]; ui = upper[idx]; li = lower[idx]
    short_m = lrc_ci > ui
    long_m  = lrc_ci < li

    # Gate 2: velocity confirms direction
    vel_m = (short_m & vel_short[idx]) | (long_m & vel_long[idx])

    atr_i   = atr[idx]
    valid_m = (short_m | long_m) & vel_m \
              & ~np.isnan(lrc_ci) & ~np.isnan(ui) & ~np.isnan(atr_i)

    # Pre-readiness from extension only (level adds up to 50 pts)
    mid_i = mid[idx]
    bs    = np.where(sigma_mult > 0, (ui - mid_i) / sigma_mult, 0.0)
    dsig  = np.where(bs > 0, (lrc_ci - mid_i) / bs, 0.0)
    ext_s = np.clip((np.abs(dsig) / sigma_mult - 1.0) * 100.0, 0, 100)
    pre_r = ext_s * 0.5

    valid_m &= (pre_r + 50.0) >= min_score   # prune bars that can't reach threshold

    cands = idx[valid_m]
    if len(cands) == 0:
        return []

    signals = []
    for i in cands:
        j         = int(i - min_i)
        direction = "SHORT" if lrc_c[i] > upper[i] else "LONG"

        level_px, _, level_score = _scored_level_at(h, l, c, int(i), direction,
                                                    touch_frac, lookback)
        if math.isnan(level_px):
            continue

        readiness = float(pre_r[j]) + level_score * 0.5
        if readiness < min_score:
            continue

        pos      = +1 if direction == "SHORT" else -1
        atr_val  = float(atr[i])
        sl_price = level_px + pos * sl_atr_mult * atr_val

        if pos * (sl_price - c[i]) <= 0:
            continue
        risk = abs(c[i] - sl_price)
        if risk < 1e-6:
            continue

        signals.append((int(i), direction, sl_price, tp_rr, float(c[i]), readiness))

    return signals
