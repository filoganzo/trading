"""
50 trade-management variants (5 SL × 5 TP × 2 trail = 50)
+ 72-combo entry parameter grid.
"""
from itertools import product
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# SL definitions
# ---------------------------------------------------------------------------
SL_CONFIGS: List[Dict[str, Any]] = [
    {"sl_id": "lev+0.3atr", "sl_mode": "level_atr", "sl_atr_mult": 0.3},
    {"sl_id": "lev+0.7atr", "sl_mode": "level_atr", "sl_atr_mult": 0.7},
    {"sl_id": "lev+1.2atr", "sl_mode": "level_atr", "sl_atr_mult": 1.2},
    {"sl_id": "ent-1.5pct", "sl_mode": "entry_pct", "sl_pct":      0.015},
    {"sl_id": "ent-2.5atr", "sl_mode": "entry_atr", "sl_atr_mult": 2.5},
]

# ---------------------------------------------------------------------------
# TP definitions
# ---------------------------------------------------------------------------
TP_CONFIGS: List[Dict[str, Any]] = [
    {"tp_id": "lrc_mid",     "tp_mode": "lrc_mid"},
    {"tp_id": "rr2",         "tp_mode": "rr",    "tp_rr": 2.0},
    {"tp_id": "rr3",         "tp_mode": "rr",    "tp_rr": 3.0},
    {"tp_id": "rr5",         "tp_mode": "rr",    "tp_rr": 5.0},
    # multi: 50 % at 1.5R, 50 % at 3R — represented as two half-rows in candidates
    {"tp_id": "multi1.5+3",  "tp_mode": "multi", "tp1_rr": 1.5, "tp1_frac": 0.50,
                                                  "tp2_rr": 3.0, "tp2_frac": 0.50},
]

# ---------------------------------------------------------------------------
# Trail definitions
# ---------------------------------------------------------------------------
TRAIL_CONFIGS: List[Dict[str, Any]] = [
    {"trail_id": "none",      "trail_mode": "none"},
    # Move SL to BE after 0.5R profit, then trail 1 ATR once 1R is reached
    {"trail_id": "be_trail",  "trail_mode": "be_then_atr",
     "be_trigger_r": 0.5, "trail_atr": 1.0},
]

# ---------------------------------------------------------------------------
# Build 50 variants
# ---------------------------------------------------------------------------
TRADE_VARIANTS: List[Dict[str, Any]] = []
for _i, (_sl, _tp, _tr) in enumerate(product(SL_CONFIGS, TP_CONFIGS, TRAIL_CONFIGS)):
    v: Dict[str, Any] = {**_sl, **_tp, **_tr}
    v["variant_id"]   = _i
    v["variant_name"] = f"{_sl['sl_id']}|{_tp['tp_id']}|{_tr['trail_id']}"
    TRADE_VARIANTS.append(v)

assert len(TRADE_VARIANTS) == 50, f"Expected 50 variants, got {len(TRADE_VARIANTS)}"

# ---------------------------------------------------------------------------
# Entry parameter grid (72 combos)
# ---------------------------------------------------------------------------
from stocks.backtest.config import LRC_LENGTHS, SIGMA_MULTS, MIN_SCORES, TOUCH_PCTS, VEL_REQUIRED

ENTRY_GRID: List[Dict[str, Any]] = []
for _j, combo in enumerate(product(LRC_LENGTHS, SIGMA_MULTS, MIN_SCORES, TOUCH_PCTS, VEL_REQUIRED)):
    lrc_len, sigma, min_sc, touch, vel = combo
    ENTRY_GRID.append({
        "entry_id":    _j,
        "lrc_length":  lrc_len,
        "sigma_mult":  sigma,
        "min_score":   min_sc,
        "touch_pct":   touch,
        "vel_required": vel,
    })
