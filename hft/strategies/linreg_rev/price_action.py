"""
Higher-timeframe key levels, break-and-reclaim, and hammer candles.

The causality trap in pivot levels
----------------------------------
A swing high at higher-timeframe bar k is only a swing high once ``right`` bars
have printed after it without exceeding it. Marking the level as "known" at bar
k is look-ahead of exactly ``right`` HTF bars -- on a D1 pivot with right=2 that
is two full days of hindsight, and it is the single most common way a
support/resistance backtest fabricates an edge. Here a pivot becomes usable only
at HTF bar ``k + right``, and that confirmation time is then mapped onto base
bars, so a base bar can only see levels that were already confirmed when it
closed.

Levels supplied
---------------
  pivot      confirmed swing highs / lows on the confirmation timeframe
  prior      previous completed HTF bar's high / low / close
  round      psychological round numbers at a configurable step

Patterns
--------
  break_reclaim   price traded through a level and then closed back on the
                  original side within a lookback window -- the failed
                  breakdown / spring. For a long: traded BELOW support, closed
                  back ABOVE it.
  hammer          long lower wick, small body, close in the upper part of the
                  range (mirrored for the bearish shooting star).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mtf import completed_htf, TF_MINUTES
from engine import _bars_since, BIG


# ------------------------------------------------------------------- pivots

def pivots(high: np.ndarray, low: np.ndarray, left: int = 2, right: int = 2):
    """
    Swing highs/lows with a confirmation delay.

    Returns (is_ph, is_pl, confirm_idx) where ``confirm_idx[k]`` is the bar at
    which a pivot at k becomes KNOWN (k + right). Positions with no pivot are
    False.
    """
    n = len(high)
    is_ph = np.zeros(n, dtype=bool)
    is_pl = np.zeros(n, dtype=bool)
    for k in range(left, n - right):
        w_h = high[k - left:k + right + 1]
        w_l = low[k - left:k + right + 1]
        if high[k] >= w_h.max():
            is_ph[k] = True
        if low[k] <= w_l.min():
            is_pl[k] = True
    confirm = np.arange(n) + right
    return is_ph, is_pl, confirm


def htf_levels_for_base(base: pd.DataFrame, htf_tf: str, left: int = 2,
                        right: int = 2, max_levels: int = 6,
                        include_prior: bool = True) -> dict:
    """
    For every base bar, the nearest confirmed key level above and below.

    Strictly causal: a level is visible to base bar i only if its confirming HTF
    bar closed at or before bar i.
    """
    mins = TF_MINUTES[htf_tf]
    comp, pos = completed_htf(base, mins)
    ch = comp["high"].to_numpy(float)
    cl = comp["low"].to_numpy(float)
    cc = comp["close"].to_numpy(float)
    M = len(comp)
    is_ph, is_pl, confirm = pivots(ch, cl, left, right)

    # For each HTF bar index m, the set of levels confirmed at or before m.
    # Walk forward keeping a rolling list of the most recent confirmed levels.
    sup = np.full(M, np.nan)      # nearest confirmed support below the close
    res = np.full(M, np.nan)      # nearest confirmed resistance above
    recent_hi: list[float] = []
    recent_lo: list[float] = []
    ph_idx = np.where(is_ph)[0]
    pl_idx = np.where(is_pl)[0]
    next_ph = 0
    next_pl = 0
    for m in range(M):
        while next_ph < len(ph_idx) and confirm[ph_idx[next_ph]] <= m:
            recent_hi.append(ch[ph_idx[next_ph]])
            next_ph += 1
        while next_pl < len(pl_idx) and confirm[pl_idx[next_pl]] <= m:
            recent_lo.append(cl[pl_idx[next_pl]])
            next_pl += 1
        hi_pool = recent_hi[-max_levels:]
        lo_pool = recent_lo[-max_levels:]
        if include_prior and m >= 1:
            hi_pool = hi_pool + [ch[m - 1]]
            lo_pool = lo_pool + [cl[m - 1]]
        px = cc[m]
        above = [x for x in hi_pool if x > px]
        below = [x for x in lo_pool if x < px]
        res[m] = min(above) if above else np.nan
        sup[m] = max(below) if below else np.nan

    # map HTF bar -> base bars, using the LAST CLOSED htf bar (pos - 1)
    k = pos - 1
    ok = (k >= 0) & (k < M)
    ki = np.where(ok, k, 0)
    return {"support": np.where(ok, sup[ki], np.nan),
            "resistance": np.where(ok, res[ki], np.nan),
            "htf_prev_high": np.where(ok, ch[ki], np.nan),
            "htf_prev_low": np.where(ok, cl[ki], np.nan)}


def round_levels(close: np.ndarray, step: float) -> dict:
    """Nearest round numbers above and below, at a fixed step."""
    below = np.floor(close / step) * step
    above = below + step
    return {"round_below": below, "round_above": above}


# ------------------------------------------------------------- the patterns

def break_reclaim(df: pd.DataFrame, level: np.ndarray, side: int = +1,
                  lookback: int = 6, min_pierce_atr: float = 0.0,
                  atr: np.ndarray | None = None) -> np.ndarray:
    """
    Failed breakdown (long) / failed breakout (short).

    Long: within the last ``lookback`` bars price traded BELOW the support
    level, and this bar CLOSES back above it. Optionally the pierce must have
    been at least ``min_pierce_atr`` deep, which filters out a level merely
    grazed by a wick.
    """
    close = df["close"].to_numpy(float)
    low = df["low"].to_numpy(float)
    high = df["high"].to_numpy(float)
    n = len(close)
    lvl = np.asarray(level, dtype=float)

    if side > 0:
        pierced = low < lvl
        if min_pierce_atr > 0 and atr is not None:
            pierced &= (lvl - low) >= min_pierce_atr * atr
        reclaimed = close > lvl
    else:
        pierced = high > lvl
        if min_pierce_atr > 0 and atr is not None:
            pierced &= (high - lvl) >= min_pierce_atr * atr
        reclaimed = close < lvl

    pierced = pierced & np.isfinite(lvl)
    since = _bars_since(pierced)
    # the pierce must be recent AND not on this very bar only -- a bar that
    # pierces and reclaims within itself still counts (since == 0 allowed)
    return reclaimed & np.isfinite(lvl) & (since <= lookback)


def hammer(df: pd.DataFrame, side: int = +1, wick_body: float = 2.0,
           close_frac: float = 0.66, max_opp_wick: float = 0.35,
           min_range_atr: float = 0.0,
           atr: np.ndarray | None = None) -> np.ndarray:
    """
    Bullish hammer (side=+1) or bearish shooting star (side=-1).

    wick_body      lower wick must be at least this multiple of the body
    close_frac     close must sit in the top ``close_frac`` of the bar's range
    max_opp_wick   upper wick at most this fraction of the range
    min_range_atr  ignore doji-sized bars whose whole range is noise
    """
    o = df["open"].to_numpy(float); c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
    rng = np.maximum(h - l, 1e-12)
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l

    if side > 0:
        long_wick, opp_wick = lower, upper
        pos_in_range = (c - l) / rng
    else:
        long_wick, opp_wick = upper, lower
        pos_in_range = (h - c) / rng

    ok = (long_wick >= wick_body * np.maximum(body, 1e-12)) \
        & (pos_in_range >= close_frac) \
        & (opp_wick <= max_opp_wick * rng)
    if min_range_atr > 0 and atr is not None:
        ok &= rng >= min_range_atr * atr
    return ok


def distance_to_level(df: pd.DataFrame, level: np.ndarray,
                      atr: np.ndarray) -> np.ndarray:
    """|close - level| in ATR units. Used to require proximity to a level."""
    return np.abs(df["close"].to_numpy(float) - np.asarray(level, float)) / atr


# ----------------------------------------------------------------- self-test

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from mtf_runner import base_indicators, BaseParams, load_tf

    print("VALIDATION")
    print("\n1. pivot confirmation delay is respected")
    rng = np.random.default_rng(0)
    h = np.cumsum(rng.normal(size=400)) + 100
    l = h - 1.0
    is_ph, is_pl, conf = pivots(h, l, 2, 2)
    bad = 0
    for k in np.where(is_ph)[0][:50]:
        # a pivot at k must not be visible before k+2
        if conf[k] != k + 2:
            bad += 1
    print(f"   pivots found {is_ph.sum()} highs / {is_pl.sum()} lows, "
          f"confirmation-offset errors {bad}")
    print("   PASS" if bad == 0 else "   FAIL")

    print("\n2. key levels are causal (truncate-and-recompute)")
    d = base_indicators("XAUUSD", "M5", BaseParams()).iloc[-20000:].reset_index(drop=True)
    full = htf_levels_for_base(d, "H4")
    probes = rng.choice(np.arange(8000, len(d) - 10), 12, replace=False)
    mism = 0
    for i in probes:
        tr = htf_levels_for_base(d.iloc[:i + 1].reset_index(drop=True), "H4")
        for key in ("support", "resistance"):
            a, b = full[key][i], tr[key][i]
            if not (np.isnan(a) and np.isnan(b)) and not np.isclose(a, b, equal_nan=True):
                mism += 1
    print(f"   probed {len(probes)} bars x 2 level types, mismatches {mism}")
    print("   PASS - no look-ahead" if mism == 0 else "   FAIL - levels leak")

    print("\n3. hammer detection sanity")
    test = pd.DataFrame({"open": [100.0, 100.0], "high": [100.5, 105.0],
                         "low": [95.0, 99.5], "close": [100.2, 104.5]})
    hm = hammer(test, +1)
    print(f"   textbook hammer -> {hm[0]} (expect True); "
          f"strong green bar -> {hm[1]} (expect False)")
    print("   PASS" if hm[0] and not hm[1] else "   FAIL")

    print("\n4. break_reclaim fires on a failed breakdown")
    t2 = pd.DataFrame({"open": [100, 99, 98, 99.5], "high": [101, 100, 99, 100.5],
                       "low": [99.5, 97, 96, 99], "close": [100, 98, 97, 100.2]},
                      dtype=float)
    lvl = np.full(4, 99.0)
    br = break_reclaim(t2, lvl, +1, lookback=3)
    print(f"   sequence -> {br.tolist()} (expect the last bar True)")
    print("   PASS" if br[-1] and not br[1] else "   FAIL")
