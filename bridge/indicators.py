r"""WaveTrend, LinReg candles and the regression channel.

Ported from "prova finale wowow.mq5". Two conventions from the MQL5 source are
preserved deliberately, because getting either wrong silently shifts every
signal:

1. SHIFT INDEXING. In the EA, shift 0 is the most recent evaluated bar and
   larger shifts go further back. Here arrays are chronological (index -1 is
   newest), so `_at(a, s)` maps shift s to `a[len(a)-1-s]`.

2. LINREG ORIENTATION. LinRegAt sums x = 1..len against y = PxByType(s+i),
   i.e. x increases as we go BACK in time, and returns
   `mean(y) - slope*mean(x) + slope`, which is the fitted value at x = 1 -
   the NEWEST bar in the window. A textbook "fit the last N bars" that runs x
   forward in time gives the fitted value at the oldest bar instead, which is
   a different number with the opposite slope sign. The tests pin this down.
"""
from __future__ import annotations

import numpy as np

# regression source codes, matching ENUM_SRC in the EA
SRC_CLOSE, SRC_OPEN, SRC_HIGH, SRC_LOW, SRC_HL2, SRC_HLC3, SRC_OHLC4 = range(7)


def _at(a: np.ndarray, s: int) -> float:
    """Value at EA shift s (0 = newest). Returns 0.0 when out of range, which
    is what the MQL5 iOpen/iClose family does for a missing bar."""
    i = len(a) - 1 - s
    return float(a[i]) if 0 <= i < len(a) else 0.0


def px_by_type(o, h, l, c, t: int, s: int) -> float:
    if t == 0:
        return _at(o, s)
    if t == 1:
        return _at(h, s)
    if t == 2:
        return _at(l, s)
    return _at(c, s)


def src_at(o, h, l, c, source: int, s: int) -> float:
    if source == SRC_OPEN:
        return _at(o, s)
    if source == SRC_HIGH:
        return _at(h, s)
    if source == SRC_LOW:
        return _at(l, s)
    if source == SRC_HL2:
        return (_at(h, s) + _at(l, s)) / 2.0
    if source == SRC_HLC3:
        return (_at(h, s) + _at(l, s) + _at(c, s)) / 3.0
    if source == SRC_OHLC4:
        return (_at(o, s) + _at(h, s) + _at(l, s) + _at(c, s)) / 4.0
    return _at(c, s)


def linreg_at(o, h, l, c, type_: int, length: int, s: int) -> float:
    """Fitted value at the newest bar of the window ending at shift s."""
    if length <= 1:
        return px_by_type(o, h, l, c, type_, s)
    sx = sy = sxx = sxy = 0.0
    for i in range(length):
        y = px_by_type(o, h, l, c, type_, s + i)
        x = i + 1.0
        sx += x
        sy += y
        sxx += x * x
        sxy += y * x
    den = length * sxx - sx * sx
    slope = (length * sxy - sx * sy) / den if den != 0.0 else 0.0
    return sy / length - slope * sx / length + slope


def regression_stats(o, h, l, c, length: int, source: int = SRC_CLOSE):
    """(ok, basis, sd) for the raw-price channel.

    `sd` is the RMS of residuals around the fitted LINE (not around a flat
    mean), walked bar by bar exactly as the EA does, with the n-1 divisor.
    """
    if length <= 1 or len(c) < length + 2:
        return False, 0.0, 0.0
    sx = sy = sxx = sxy = 0.0
    for i in range(length):
        y = src_at(o, h, l, c, source, i)
        x = i + 1.0
        sx += x
        sy += y
        sxx += x * x
        sxy += y * x
    den = length * sxx - sx * sx
    slope = (length * sxy - sx * sy) / den if den != 0.0 else 0.0
    intercept = sy / length - slope * sx / length + slope

    acc = 0.0
    fitted = intercept
    for i in range(length):
        r = src_at(o, h, l, c, source, i) - fitted
        acc += r * r
        fitted += slope
    sd = float(np.sqrt(acc / (length - 1)))
    return True, float(intercept), sd


def wavetrend(h, l, c, channel_length: int = 10, average_length: int = 21):
    """(ok, wt). EMA-of-CI oscillator, seeded from the oldest bar in the
    window and walked forward, capped at 600 bars as in the EA."""
    avail = len(c) - 1
    if avail < 60:
        return False, 0.0
    n = min(600, avail)
    kc = 2.0 / (channel_length + 1.0)
    ka = 2.0 / (average_length + 1.0)
    esa = d = wt = 0.0
    init = False
    for s in range(n - 1, -1, -1):
        hi, lo, cl = _at(h, s), _at(l, s), _at(c, s)
        if hi == 0 or lo == 0:
            continue
        ap = (hi + lo + cl) / 3.0
        if not init:
            esa, d, wt = ap, 0.0, 0.0
            init = True
            continue
        esa = kc * ap + (1.0 - kc) * esa
        d = kc * abs(ap - esa) + (1.0 - kc) * d
        ci = (ap - esa) / (0.015 * d) if d != 0.0 else 0.0
        wt = ka * ci + (1.0 - ka) * wt
    return (True, float(wt)) if init else (False, 0.0)


def linreg_signal(o, h, l, c, lr_len: int, signal_length: int,
                  sma: bool, use_linreg: bool):
    """(ok, signal) - the smoothing line the LinReg close is compared against."""
    if signal_length <= 0:
        return False, 0.0
    if sma:
        total = 0.0
        for i in range(signal_length):
            total += linreg_at(o, h, l, c, 3, lr_len, i) if use_linreg else _at(c, i)
        return True, total / signal_length
    warm = signal_length * 6
    k = 2.0 / (signal_length + 1.0)
    e = 0.0
    init = False
    for i in range(warm - 1, -1, -1):
        v = linreg_at(o, h, l, c, 3, lr_len, i) if use_linreg else _at(c, i)
        if not init:
            e, init = v, True
        else:
            e = k * v + (1.0 - k) * e
    return init, e
