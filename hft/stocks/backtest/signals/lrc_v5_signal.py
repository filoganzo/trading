"""
Signal engine for v5: LRC channel + configurable velocity strictness + M15 confirmation gate.

Component toggles (swept in runner_v5):
  use_level  — require scored key level (False = LRC extension only, SL from entry)
  vel_min    — 0=no vel gate, 1=any-1-of-3, 2=two-of-3, 3=all-3
  use_m15    — require M15 confirmation precomputed in precomp dict

M15 precompute (called during symbol precompute, stored in precomp):
  precomp["m15_conf_short"] — bool array, True when last N M15 bars of that day
                               show bearish rejection at resistance
  precomp["m15_conf_long"]  — same for bullish acceptance at support
"""
import math
import numpy as np

from stocks.backtest.signals.wt_lrc_signal import (
    _lrc_vectorized,
    _lrc_candle_close_vectorized,
    _atr_vectorized,
    _scored_level_at,
)


# ── Velocity component arrays ─────────────────────────────────────────────────

def vel_component_arrays(high: np.ndarray, low: np.ndarray,
                         close: np.ndarray, open_: np.ndarray,
                         lookback: int = 5):
    """
    Returns the three individual velocity components as separate bool arrays.
    Direction-neutral: SHORT uses (fading, bearish, upper_wick_sig);
                       LONG  uses (fading, bullish, lower_wick_sig).

    Returns: (fading, bearish, bullish, upper_wick_sig, lower_wick_sig)
    """
    n         = len(close)
    bar_range = high - low
    body      = np.abs(close - open_)
    u_wick    = high - np.maximum(close, open_)
    l_wick    = np.minimum(close, open_) - low

    avg_range = np.full(n, np.nan)
    if n > lookback:
        avgs = np.lib.stride_tricks.sliding_window_view(bar_range, lookback).mean(axis=1)
        avg_range[lookback:] = avgs[:n - lookback]

    fading = np.zeros(n, dtype=bool)
    valid  = ~np.isnan(avg_range) & (avg_range > 1e-8)
    fading[valid] = bar_range[valid] < 0.75 * avg_range[valid]

    bearish = close < open_
    bullish = close > open_
    wick_thresh  = np.maximum(1.5 * body, 0.10 * bar_range)
    upper_wick_s = u_wick > wick_thresh
    lower_wick_s = l_wick > wick_thresh

    return fading, bearish, bullish, upper_wick_s, lower_wick_s


# ── M15 gate ─────────────────────────────────────────────────────────────────

def m15_check(h: np.ndarray, l: np.ndarray, c: np.ndarray, o: np.ndarray):
    """
    Returns (confirm_short, confirm_long) for the last N M15 bars of a session.

    Requires at least 2 of 3 criteria to confirm:
      SHORT (resistance rejection): bearish majority, upper-wick count >= 2, fading range
      LONG  (support acceptance):   bullish majority, lower-wick count >= 2, fading range
    """
    n = len(c)
    if n == 0:
        return False, False

    body      = np.abs(c - o)
    bar_range = h - l
    u_wick    = h - np.maximum(c, o)
    l_wick    = np.minimum(c, o) - l
    wt        = np.maximum(1.5 * body, 0.10 * bar_range)

    avg_rng      = bar_range.mean()
    fading_range = bool(bar_range[-1] < 0.75 * avg_rng) if avg_rng > 1e-8 else False

    bear_maj = int((c < o).sum() > n * 0.5)
    bull_maj = int((c > o).sum() > n * 0.5)
    uw_count = int((u_wick > wt).sum() >= 2)
    lw_count = int((l_wick > wt).sum() >= 2)
    fading   = int(fading_range)

    conf_short = (bear_maj + uw_count + fading) >= 2
    conf_long  = (bull_maj + lw_count + fading) >= 2

    return bool(conf_short), bool(conf_long)


# ── Signal finder ─────────────────────────────────────────────────────────────

def find_signals_v5(precomp: dict,
                    lrc_length: int, sigma_mult: float, touch_pct: float,
                    use_level: bool = True,
                    vel_min: int = 1,
                    use_m15: bool = False,
                    min_score: float = 40.0,
                    lookback: int = 120) -> list:
    """
    V5 signal finder with configurable component toggles.

    use_level=False: SL placed sl_atr_mult x ATR from entry (not from level).
                     level_px returned as entry_px so Phase B code is unchanged.
    vel_min:  0=no gate, 1=any-1-of-3, 2=two-of-3, 3=all-3
    use_m15:  gate on precomp["m15_conf_short/long"] (must be precomputed)

    Returns list of (bar_idx, entry_bar, entry_px, direction, level_px, atr_v, readiness)
    """
    c     = precomp["close"]; h = precomp["high"]; l = precomp["low"]
    o     = precomp["open"]
    lrc_c = precomp["lrc_c"]
    atr   = precomp["atr"]
    mid, upper, lower = precomp["lrc_data"][(lrc_length, sigma_mult)]
    n     = precomp["n"]

    fading     = precomp["vel_fading"]
    bearish    = precomp["vel_bearish"]
    bullish    = precomp["vel_bullish"]
    u_wick_s   = precomp["vel_upper_wick"]
    l_wick_s   = precomp["vel_lower_wick"]
    m15_cs     = precomp.get("m15_conf_short")
    m15_cl     = precomp.get("m15_conf_long")

    touch_frac = touch_pct / 100.0
    min_i = max(lrc_length, 15 + 9 + 5, 14 + 5)
    if min_i >= n:
        return []

    idx    = np.arange(min_i, n)
    lrc_ci = lrc_c[idx]; ui = upper[idx]; li = lower[idx]
    atr_i  = atr[idx]
    short_m = lrc_ci > ui
    long_m  = lrc_ci < li

    # Velocity gate
    if vel_min == 0:
        vel_m = np.ones(len(idx), dtype=bool)
    else:
        f = fading[idx]; be = bearish[idx]; bu = bullish[idx]
        uw = u_wick_s[idx]; lw = l_wick_s[idx]
        if vel_min == 1:
            vs = f | be | uw
            vl = f | bu | lw
        elif vel_min == 2:
            vs = (f & be) | (f & uw) | (be & uw)
            vl = (f & bu) | (f & lw) | (bu & lw)
        else:  # 3
            vs = f & be & uw
            vl = f & bu & lw
        vel_m = (short_m & vs) | (long_m & vl)

    # M15 gate
    if use_m15 and m15_cs is not None and m15_cl is not None:
        m15_m = (short_m & m15_cs[idx]) | (long_m & m15_cl[idx])
    else:
        m15_m = np.ones(len(idx), dtype=bool)

    valid_m = ((short_m | long_m) & vel_m & m15_m
               & ~np.isnan(lrc_ci) & ~np.isnan(ui) & ~np.isnan(atr_i))

    # Extension score
    mid_i = mid[idx]
    bs    = np.where(sigma_mult > 0, (ui - mid_i) / sigma_mult, 0.0)
    dsig  = np.where(bs > 0, (lrc_ci - mid_i) / bs, 0.0)
    ext_s = np.clip((np.abs(dsig) / sigma_mult - 1.0) * 100.0, 0, 100)

    # Readiness pre-score (level adds up to 50 pts when use_level=True)
    pre_r = ext_s * 0.5 if use_level else ext_s
    valid_m &= (pre_r + (50.0 if use_level else 0.0)) >= min_score

    cands = idx[valid_m]
    if len(cands) == 0:
        return []

    sigs = []
    for i in cands:
        j    = int(i - min_i)
        dir_ = "SHORT" if lrc_c[i] > upper[i] else "LONG"
        eb   = int(i) + 1
        if eb >= n:
            continue
        ep    = float(o[eb])
        atr_v = float(atr[i])

        if use_level:
            lv, _, lv_score = _scored_level_at(h, l, c, int(i), dir_, touch_frac, lookback)
            if math.isnan(lv):
                continue
            readiness = float(pre_r[j]) + lv_score * 0.5
            if readiness < min_score:
                continue
        else:
            lv        = ep      # SL placed from entry in Phase B (lv == ep)
            readiness = float(pre_r[j])

        sigs.append((int(i), eb, ep, dir_, lv, atr_v, readiness))

    return sigs
