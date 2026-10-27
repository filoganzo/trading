"""
HTF signal components for v7.

Weekly level detection:
    Only uses SWING POINT highs/lows (local extrema) as level candidates —
    a weekly bar's high is a swing high if it exceeds all bars within `wing`
    bars on either side.  This cuts candidates from 500+ to 30-60 per symbol,
    eliminating the dense grid of noise-levels that plagued v6.

    Levels clustered within touch_frac%, scored by touch count.
    Filterable by min_score at signal-finding time.

M15 entry trigger:
    Strict pin-bar: requires BOTH body direction AND wick rejection (not OR).
    Bar must commit — bullish close AND meaningful lower wick for LONG;
    bearish close AND upper wick for SHORT.
"""
import numpy as np


# ── Swing-point extraction ────────────────────────────────────────────────────

def _swing_px(w_high: np.ndarray, w_low: np.ndarray, wing: int) -> np.ndarray:
    """Return prices of weekly swing highs and lows (local extrema)."""
    n  = len(w_high)
    px = []
    for i in range(wing, n - wing):
        if (np.all(w_high[i - wing:i] < w_high[i]) and
                np.all(w_high[i + 1:i + wing + 1] < w_high[i])):
            px.append(float(w_high[i]))
        if (np.all(w_low[i - wing:i] > w_low[i]) and
                np.all(w_low[i + 1:i + wing + 1] > w_low[i])):
            px.append(float(w_low[i]))
    return np.array(px) if px else np.array([])


# ── Weekly level precompute ───────────────────────────────────────────────────

def precompute_weekly_levels(w_high: np.ndarray, w_low: np.ndarray,
                             touch_fracs: list, swing_wing: int = 2) -> dict:
    """
    Build {touch_frac: [(level_px, score), ...]} sorted by score desc.
    Called once per symbol during precompute.

    Only swing points are candidates. Score = min(100, touches/5*100).
    """
    if len(w_high) < 4 + swing_wing * 2:
        return {tf: [] for tf in touch_fracs}

    all_px = _swing_px(w_high, w_low, swing_wing)
    if len(all_px) == 0:
        return {tf: [] for tf in touch_fracs}

    result = {}
    for tf in touch_fracs:
        seen   = np.zeros(len(all_px), dtype=bool)
        levels = []
        for i in range(len(all_px)):
            if seen[i]:
                continue
            lv   = all_px[i]
            near = np.abs(all_px - lv) / (lv + 1e-9) <= tf
            seen |= near
            rep     = float(all_px[near].mean())
            touches = int(near.sum())
            score   = min(100.0, touches / 5.0 * 100.0)
            levels.append((rep, score))
        result[tf] = sorted(levels, key=lambda x: -x[1])

    return result


def find_htf_level(w_levels: list, price: float,
                   touch_frac: float, min_score: float = 0.0):
    """
    Find closest swing level within touch_frac% of `price` with
    score >= min_score. Returns (level_px, score) or (nan, 0.0).
    """
    best_lv    = float('nan')
    best_score = 0.0
    zone       = price * touch_frac

    for lv, sc in w_levels:
        if sc < min_score:
            continue
        if abs(lv - price) <= zone and sc > best_score:
            best_lv    = lv
            best_score = sc

    return best_lv, best_score


# ── M15 entry trigger ─────────────────────────────────────────────────────────

def find_m15_entry(m15_bars_day: list, direction: str,
                   level_px: float, touch_frac: float):
    """
    Walk M15 bars; return (bar_idx, entry_px) on first strict pin-bar at level.

    LONG:  l <= level_px + zone  AND  c > o  AND  lower_wick >= wick_thresh
    SHORT: h >= level_px - zone  AND  c < o  AND  upper_wick >= wick_thresh

    Both body direction AND wick required (strict, not OR).
    Entry price = close of confirmation bar.
    """
    if not m15_bars_day or level_px <= 0:
        return None, None

    zone = level_px * touch_frac

    for i, (h, l, c, o) in enumerate(m15_bars_day):
        body        = abs(c - o)
        bar_range   = h - l
        upper_wick  = h - max(c, o)
        lower_wick  = min(c, o) - l
        wick_thresh = max(1.5 * body, 0.08 * bar_range)

        if direction == "LONG":
            if l <= level_px + zone and (c > o) and (lower_wick >= wick_thresh):
                return i, float(c)
        else:  # SHORT
            if h >= level_px - zone and (c < o) and (upper_wick >= wick_thresh):
                return i, float(c)

    return None, None
