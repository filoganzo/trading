"""
Given a candidate row and a variant spec, compute:
  - initial SL / TP prices
  - per-bar exit checks (SL hit, TP hit, trail updates)
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Compute initial trade specs
# ---------------------------------------------------------------------------

def compute_sl(
    entry: float,
    direction: int,         # +1 = short, -1 = long
    level_price: float,
    atr: float,
    variant: Dict[str, Any],
) -> float:
    mode = variant["sl_mode"]
    if mode == "level_atr":
        offset = variant["sl_atr_mult"] * atr
        return level_price + offset if direction == +1 else level_price - offset
    elif mode == "entry_pct":
        pct = variant["sl_pct"]
        return entry * (1 + pct) if direction == +1 else entry * (1 - pct)
    elif mode == "entry_atr":
        offset = variant["sl_atr_mult"] * atr
        return entry + offset if direction == +1 else entry - offset
    return entry   # fallback


def compute_tp(
    entry: float,
    sl: float,
    lrc_mid: float,
    direction: int,
    variant: Dict[str, Any],
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Returns (tp_main, tp1, tp2).
    For non-multi variants, tp1=tp2=None.
    For multi variants, tp_main is a sentinel (nan), tp1 and tp2 are set.
    """
    risk = abs(entry - sl)
    if risk == 0:
        return (entry, None, None)

    mode = variant["tp_mode"]
    if mode == "lrc_mid":
        tp = lrc_mid
        return (tp, None, None)
    elif mode == "rr":
        rr = variant["tp_rr"]
        tp = entry - direction * risk * rr
        return (tp, None, None)
    elif mode == "multi":
        tp1 = entry - direction * risk * variant["tp1_rr"]
        tp2 = entry - direction * risk * variant["tp2_rr"]
        return (float("nan"), tp1, tp2)
    return (entry, None, None)


# ---------------------------------------------------------------------------
# Per-bar exit check
# ---------------------------------------------------------------------------

class PositionState:
    __slots__ = [
        "symbol", "variant_id", "direction", "entry_date", "entry_price",
        "sl", "tp_main", "tp1", "tp2", "tp1_frac", "tp2_frac",
        "notional", "be_hit", "trailing_high", "trailing_low",
        "partial_closed", "partial_frac", "atr",
    ]
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.be_hit        = False
        self.trailing_high = kw.get("entry_price")
        self.trailing_low  = kw.get("entry_price")
        self.partial_closed = False
        self.partial_frac   = 0.0


def check_and_update(
    pos: PositionState,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    bar_date,
    variant: Dict[str, Any],
) -> List[Dict]:
    """
    Check if position exits on this bar.  Returns list of ClosedTrade dicts
    (may be 2 for partial TP hit then SL).
    Assumes SL checked before TP (conservative).
    Updates trailing stop in-place.
    """
    direction = pos.direction    # +1 short, -1 long
    results   = []

    # --- Update trailing high/low ---
    if direction == -1:   # long: track highest close reached
        pos.trailing_high = max(pos.trailing_high, bar_high)
    else:                 # short: track lowest close reached
        pos.trailing_low  = min(pos.trailing_low, bar_low)

    risk = abs(pos.entry_price - pos.sl)

    # --- Trail mode: be_then_atr ---
    if variant["trail_mode"] == "be_then_atr" and risk > 0:
        be_trigger  = variant["be_trigger_r"] * risk
        trail_atr   = variant["trail_atr"] * pos.atr

        if not pos.be_hit:
            # Has price moved 0.5R in our favour?
            fav_move = (pos.entry_price - bar_low) if direction == -1 else (bar_high - pos.entry_price)
            if fav_move >= be_trigger:
                pos.sl     = pos.entry_price   # move to breakeven
                pos.be_hit = True
        else:
            # Trail by 1 ATR from extreme
            if direction == -1:
                new_sl = pos.trailing_high - trail_atr
                if new_sl > pos.sl:
                    pos.sl = new_sl
            else:
                new_sl = pos.trailing_low + trail_atr
                if new_sl < pos.sl:
                    pos.sl = new_sl

    # --- SL check (before TP, conservative) ---
    sl_hit = (direction == -1 and bar_low  <= pos.sl) or \
             (direction == +1 and bar_high >= pos.sl)

    # --- TP check ---
    is_multi = not np.isnan(pos.tp_main) if not np.isnan(float(pos.tp_main if pos.tp_main is not None else float('nan'))) else False
    is_multi = (pos.tp1 is not None)

    if is_multi:
        # TP1 check (partial)
        tp1_hit = (not pos.partial_closed) and (
            (direction == -1 and bar_high >= pos.tp1) or
            (direction == +1 and bar_low  <= pos.tp1)
        )
        tp2_hit = pos.partial_closed and (
            (direction == -1 and bar_high >= pos.tp2) or
            (direction == +1 and bar_low  <= pos.tp2)
        )

        if sl_hit:
            frac = (1.0 - pos.partial_frac)
            results.append(_close_record(pos, pos.sl, "sl", bar_date, frac))
            pos.partial_frac = 1.0
        elif tp1_hit and not pos.partial_closed:
            frac = variant.get("tp1_frac", 0.5)
            results.append(_close_record(pos, pos.tp1, "tp1", bar_date, frac))
            pos.partial_frac   = frac
            pos.partial_closed = True
            pos.sl             = pos.entry_price   # move to BE after first TP
        elif tp2_hit:
            frac = 1.0 - pos.partial_frac
            results.append(_close_record(pos, pos.tp2, "tp2", bar_date, frac))
            pos.partial_frac = 1.0
    else:
        tp_hit = (direction == -1 and bar_high >= pos.tp_main) or \
                 (direction == +1 and bar_low  <= pos.tp_main)
        if sl_hit:
            results.append(_close_record(pos, pos.sl, "sl", bar_date, 1.0))
            pos.partial_frac = 1.0
        elif tp_hit:
            results.append(_close_record(pos, pos.tp_main, "tp", bar_date, 1.0))
            pos.partial_frac = 1.0

    return results


def is_fully_closed(pos: PositionState) -> bool:
    return pos.partial_frac >= 1.0


def _close_record(pos: PositionState, exit_price: float, reason: str, date, frac: float) -> Dict:
    risk_pts = abs(pos.entry_price - pos.sl)
    if risk_pts == 0:
        pnl_r = 0.0
    else:
        raw_pnl = (exit_price - pos.entry_price) * (-pos.direction)
        pnl_r   = raw_pnl / risk_pts

    alloc = pos.notional * frac
    pnl_usd = alloc * (pnl_r * risk_pts / pos.entry_price) if pos.entry_price else 0.0

    return {
        "symbol":      pos.symbol,
        "variant_id":  pos.variant_id,
        "direction":   pos.direction,
        "entry_date":  pos.entry_date,
        "exit_date":   date,
        "entry_price": pos.entry_price,
        "exit_price":  exit_price,
        "exit_reason": reason,
        "notional":    alloc,
        "pnl_r":       pnl_r,
        "pnl_usd":     pnl_usd,
    }
