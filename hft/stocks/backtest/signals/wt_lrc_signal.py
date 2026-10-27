"""
Signal engine for the v2 LRC strategy (ported from LRC_Bot v1.0).

Conditions:
  1. LRC candle close (linreg-15 → EMA-9) outside 120-bar 1.5σ channel
  2. WaveTrend WT1 (n1=10, n2=21) > OB for SHORT, < OS for LONG
  3. Price AT a significant historical level (within touch_pct%), scored by touches

All heavy functions are vectorized via stride_tricks.
_scored_level_at() has a Python loop but is only called for candidate bars.
"""
import math
import numpy as np
import pandas as pd


# ── EMA ───────────────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(arr), np.nan)
    alpha = 2.0 / (period + 1)
    start = next((i for i, v in enumerate(arr) if not np.isnan(v)), None)
    if start is None:
        return result
    result[start] = arr[start]
    for i in range(start + 1, len(arr)):
        prev = result[i - 1]
        result[i] = alpha * arr[i] + (1 - alpha) * prev if not np.isnan(arr[i]) else prev
    return result


# ── LRC channel (vectorized) ──────────────────────────────────────────────────

def _lrc_vectorized(close: np.ndarray, length: int, sigma_mult: float):
    """Returns (mid, upper, lower, dsig), all shape (n,)."""
    n = len(close)
    mid = np.full(n, np.nan); upper = np.full(n, np.nan)
    lower = np.full(n, np.nan); dsig = np.full(n, np.nan)

    windows = np.lib.stride_tricks.sliding_window_view(close, length)  # (n-length+1, length)
    x  = np.arange(length, dtype=float)
    xm = x.mean(); xd = x - xm; xd2 = (xd ** 2).sum()

    ym      = windows.mean(axis=1)
    sl      = ((windows - ym[:, None]) * xd).sum(axis=1) / xd2
    mid_v   = sl * (length - 1) + (ym - sl * xm)
    yh      = sl[:, None] * x + (ym - sl * xm)[:, None]
    std_v   = (windows - yh).std(axis=1, ddof=0)

    valid = std_v > 0
    vi = np.where(valid)[0]
    base = length - 1
    mid  [base + vi] = mid_v[vi]
    upper[base + vi] = mid_v[vi] + sigma_mult * std_v[vi]
    lower[base + vi] = mid_v[vi] - sigma_mult * std_v[vi]
    dsig [base + vi] = (close[base + vi] - mid_v[vi]) / std_v[vi]

    # bars where std==0: set mid only
    nv = np.where(~valid)[0]
    mid[base + nv] = mid_v[nv]

    return mid, upper, lower, dsig


# ── LRC candle close (vectorized) ─────────────────────────────────────────────

def _lrc_candle_close_vectorized(close: np.ndarray, length: int, smooth: int) -> np.ndarray:
    """Endpoint of short LRC per bar, EMA-smoothed."""
    n = len(close)
    windows = np.lib.stride_tricks.sliding_window_view(close, length)
    x  = np.arange(length, dtype=float)
    xm = x.mean(); xd = x - xm; xd2 = (xd ** 2).sum()
    ym  = windows.mean(axis=1)
    sl  = ((windows - ym[:, None]) * xd).sum(axis=1) / xd2
    ep  = sl * (length - 1) + (ym - sl * xm)
    reg = np.full(n, np.nan)
    reg[length - 1:] = ep
    return _ema(reg, smooth)


# ── WaveTrend (vectorized) ────────────────────────────────────────────────────

def _wavetrend_vectorized(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                          n1: int = 10, n2: int = 21):
    """WaveTrend (LazyBear). Returns (wt1, wt2)."""
    ap  = (high + low + close) / 3
    esa = _ema(ap, n1)
    d   = _ema(np.abs(ap - esa), n1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ci = np.where(d > 1e-10, (ap - esa) / (0.015 * d), 0.0)
    wt1 = _ema(ci, n2)
    wt2 = pd.Series(wt1).rolling(4).mean().values
    return wt1, wt2


# ── ATR (Wilder smoothing) ────────────────────────────────────────────────────

def _atr_vectorized(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    p: int = 14) -> np.ndarray:
    n = len(close); res = np.full(n, np.nan)
    if n < p + 1:
        return res
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    res[p] = tr[:p].mean()
    for i in range(p + 1, n):
        res[i] = (res[i - 1] * (p - 1) + tr[i - 1]) / p
    return res


# ── Scored level ──────────────────────────────────────────────────────────────

def _scored_level_at(high, low, close, i, direction, touch_pct_frac, lookback=120):
    """
    Find best historically-tested swing level at bar i.
    Returns (level_px, dist_pct, score_0_to_100).
    Only called for actual candidate bars, so the Python loop is fine.
    """
    end = i + 1
    start = max(0, end - lookback)
    h = high[start:end]; l = low[start:end]; c_last = close[i]
    w = len(h)

    best_px = float("nan"); best_score = 0.0; best_dist = float("nan")

    for j in range(2, w - 2):
        if direction == "SHORT":
            px = h[j]
            lo = max(0, j - 2); hi = min(w, j + 3)
            if px != h[lo:hi].max():
                continue
        else:
            px = l[j]
            lo = max(0, j - 2); hi = min(w, j + 3)
            if px != l[lo:hi].min():
                continue

        dist_frac = abs(px - c_last) / c_last
        if dist_frac > touch_pct_frac:
            continue

        arr     = h if direction == "SHORT" else l
        touches = int(np.sum(np.abs(arr - px) / px <= touch_pct_frac))
        if touches > best_score:
            best_score = touches; best_px = px; best_dist = dist_frac * 100

    if math.isnan(best_px):
        return float("nan"), float("nan"), 0.0
    return best_px, best_dist, min(100.0, best_score * 20.0)


# ── Vectorized candidate finder ───────────────────────────────────────────────

def find_signals(precomp: dict, lrc_length: int, sigma_mult: float,
                 wt_ob: float, touch_pct: float, sl_atr_mult: float,
                 tp_rr: float, min_score: float = 40.0, lookback: int = 120):
    """
    Fast signal finder. Vectorizes LRC/WT checks, then calls _scored_level_at
    only for the small fraction of bars that pass both.

    precomp must contain: close, high, low, lrc_c, wt1, atr,
                          lrc_data[(lrc_length, sigma_mult)] = (mid, upper, lower)

    Returns list of (bar_idx, direction, sl, tp_rr, signal_close, readiness).
    """
    c = precomp["close"]; h = precomp["high"]; l = precomp["low"]
    lrc_c = precomp["lrc_c"]; wt1 = precomp["wt1"]; atr = precomp["atr"]
    mid, upper, lower = precomp["lrc_data"][(lrc_length, sigma_mult)]
    n = precomp["n"]

    touch_frac = touch_pct / 100.0
    wt_os = -wt_ob

    min_i = max(lrc_length,
                15 + 9 + 5,          # lrc_candle_len + smooth
                10 + 21 + 10,        # wt_n1 + wt_n2
                14 + 5)              # atr period
    if min_i >= n:
        return []

    idx = np.arange(min_i, n)

    # Vectorized gate checks
    lrc_ci = lrc_c[idx]; ui = upper[idx]; li = lower[idx]
    short_m = lrc_ci > ui
    long_m  = lrc_ci < li
    dir_m   = short_m | long_m

    wt1_i   = wt1[idx]
    wt_m    = ((short_m & (wt1_i > wt_ob)) | (long_m & (wt1_i < wt_os)))

    atr_i   = atr[idx]
    valid_m = dir_m & wt_m & ~np.isnan(lrc_ci) & ~np.isnan(ui) & ~np.isnan(atr_i)

    # Pre-readiness without level (max level contribution = 20 pts)
    mid_i   = mid[idx]
    bs      = np.where(sigma_mult > 0, (ui - mid_i) / sigma_mult, 0.0)
    dsig    = np.where(bs > 0, (lrc_ci - mid_i) / bs, 0.0)
    ext_s   = np.clip((np.abs(dsig) / sigma_mult - 1.0) * 100.0, 0, 100)
    wt_s    = np.where(wt_m, np.clip((np.abs(wt1_i) - wt_ob) * 2.0, 0, 100), 0.0)
    pre_r   = ext_s * 0.4 + wt_s * 0.4

    valid_m &= (pre_r + 20.0) >= min_score   # prune hopeless bars

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

        readiness = float(pre_r[j]) + level_score * 0.2
        if readiness < min_score:
            continue

        pos       = +1 if direction == "SHORT" else -1
        atr_val   = float(atr[i])
        sl_price  = level_px + pos * sl_atr_mult * atr_val

        if pos * (sl_price - c[i]) <= 0:
            continue
        risk = abs(c[i] - sl_price)
        if risk < 1e-6:
            continue

        signals.append((int(i), direction, sl_price, tp_rr, float(c[i]), readiness))

    return signals
