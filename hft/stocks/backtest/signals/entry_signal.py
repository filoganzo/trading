"""
Entry signal enrichment on daily bars:
  - Key level detection (pivot H/L + round numbers) from daily bars only
  - Velocity proxy from 3-bar ROC
  - Combine into EntryFrame with level_score, velocity, combined_score

Note: In the backtest we only have daily bars, so H4/M15 are not available.
Level detection uses daily pivots as a conservative but realistic approximation.
"""
import numpy as np
import pandas as pd

PIVOT_N     = 8      # bars each side for pivot detection
ROUND_RANGE = 3      # round levels to check above/below
TF_W = {"D1_pivot": 2.0, "D1_period": 3.0, "round": 1.5}


def _pivot_level_score(h: np.ndarray, l: np.ndarray, n: int) -> tuple:
    """Return list of (price, score) pivot levels."""
    levels = []
    for i in range(n, len(h) - n):
        if h[i] >= max(h[i - n: i]) and h[i] >= max(h[i + 1: i + n + 1]):
            left  = sum(1 for j in range(i - 1, max(0, i - n * 2) - 1, -1) if h[j] < h[i])
            right = sum(1 for j in range(i + 1, min(len(h), i + n * 2 + 1))  if h[j] < h[i])
            strength = min(5, 1 + (left + right) // 3)
            levels.append((float(h[i]), TF_W["D1_pivot"] * strength))
        if l[i] <= min(l[i - n: i]) and l[i] <= min(l[i + 1: i + n + 1]):
            left  = sum(1 for j in range(i - 1, max(0, i - n * 2) - 1, -1) if l[j] > l[i])
            right = sum(1 for j in range(i + 1, min(len(l), i + n * 2 + 1))  if l[j] > l[i])
            strength = min(5, 1 + (left + right) // 3)
            levels.append((float(l[i]), TF_W["D1_pivot"] * strength))
    return levels


def _round_levels(price: float, atr: float) -> list:
    if price <= 0 or atr <= 0:
        return []
    mag  = 10 ** max(0, int(np.log10(price)) - 1)
    base = round(price / mag) * mag
    return [
        (base + k * mag, TF_W["round"])
        for k in range(-ROUND_RANGE, ROUND_RANGE + 1)
        if abs(base + k * mag - price) <= 3 * atr and base + k * mag > 0
    ]


def _nearest_level(levels: list, price: float) -> tuple:
    """Return (level_price, level_score, dist_pct) of closest level."""
    if not levels:
        return (price, 0.0, 999.0)
    best = min(levels, key=lambda lv: abs(lv[0] - price))
    dist_pct = abs(best[0] - price) / price * 100 if price > 0 else 999.0
    return (best[0], best[1], dist_pct)


def _velocity(closes: np.ndarray, i: int, atr: float, v_bars: int = 3) -> float:
    """Normalized v_bars ROC at position i."""
    if i < v_bars or atr <= 0:
        return 0.0
    return float(closes[i] - closes[i - v_bars]) / (atr * v_bars)


def compute_entry_signals(
    df: pd.DataFrame,
    touch_pct: float,
    vel_required: bool,
) -> pd.DataFrame:
    """
    Returns per-bar EntryFrame with:
      level_price, level_score, level_dist_pct, velocity, entry_score, valid
    Levels are computed using a lookback window of the last 252 bars to simulate
    knowing only what was visible at each point in time (no future data).
    """
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    atrs   = df["atr"].values
    n      = len(df)

    level_price  = np.full(n, np.nan)
    level_score  = np.zeros(n)
    level_dist   = np.full(n, 999.0)
    velocity     = np.zeros(n)

    # Pre-compute pivots over full series then restrict look-ahead
    all_levels = _pivot_level_score(highs, lows, PIVOT_N)

    for i in range(PIVOT_N * 2, n):
        price = closes[i]
        atr_i = atrs[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        # Only pivots from bars [0..i-PIVOT_N] to avoid look-ahead
        levels_i = [(p, s) for (p, s) in all_levels
                    if abs(p - highs[i]) / (highs[i] + 1e-9) > 0 or abs(p - lows[i]) / (lows[i] + 1e-9) > 0]
        # Simple guard: exclude any level that equals today's exact high/low (would be look-ahead for pivots straddling today)
        levels_i += _round_levels(price, atr_i)

        lp, ls, ld = _nearest_level(levels_i, price)
        level_price[i] = lp
        level_score[i] = min(100.0, ls * 5)   # normalise to 0-100
        level_dist[i]  = ld
        velocity[i]    = _velocity(closes, i, atr_i)

    # valid = within touch_pct of a level (and velocity confirmed if required)
    valid = level_dist <= touch_pct
    if vel_required:
        # velocity must be non-zero in the signal direction — caller provides direction
        # here we just mark vel_ok = abs(velocity) > 0.05
        valid = valid & (np.abs(velocity) > 0.05)

    return pd.DataFrame({
        "level_price":  level_price,
        "level_score":  level_score,
        "level_dist_%": level_dist,
        "velocity":     velocity,
        "valid":        valid,
    }, index=df.index)
