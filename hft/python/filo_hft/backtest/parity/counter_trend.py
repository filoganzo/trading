"""Counter-trend velocity + trend-state vetoes.

Backtest-parity port of the MQL5 logic in
``mql5/Include/FiloHFT/Setups.mqh`` (~lines 283-320).  Only applies to
reversal families (mean_reversion / rejection / fakeout / failed_auction);
continuation families (breakout / retest / engulfing) are unaffected.

This is parity code -- do NOT import from live execution paths.
"""

from dataclasses import dataclass
from typing import Any

# Families that the counter-trend vetoes apply to.  Mirrors the
# ``is_counter_trend`` predicate in Setups.mqh.
COUNTER_TREND_FAMILIES = frozenset({
    "mean_reversion",
    "rejection",
    "fakeout",
    "failed_auction",
})


@dataclass
class CounterTrendInputs:
    direction: int                 # +1 long, -1 short
    velocity_pips_per_sec: float   # signed, + = up
    trend_state: str               # "strong_up"|"up"|"neutral"|"down"|"strong_down"
    family: str                    # setup family name
    rev_confirm_passed: bool       # if rev_confirm already said ok, trend veto may relax


def _cfg_get(cfg: Any, name: str, default):
    """Tolerant attribute/dict lookup so this works with dataclass cfg
    objects and plain dicts alike."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def check_counter_trend(inp: CounterTrendInputs, cfg: Any) -> tuple[bool, str]:
    """Apply counter-trend velocity + trend-state vetoes.

    Returns ``(ok, reason)``.  ``ok == True`` means the setup passes and
    ``reason == ""``; ``ok == False`` means vetoed and ``reason`` matches
    the MQL5 ``block_reason`` string as closely as practical.
    """
    # Family gate -- only reversal families are subject to these vetoes.
    block_families = frozenset(
        _cfg_get(cfg, "counter_trend_block_families", COUNTER_TREND_FAMILIES)
    )
    if inp.family not in block_families:
        return True, ""

    # (B) Velocity veto: price accelerating hard against the setup.
    v_thr = float(_cfg_get(cfg, "counter_trend_velocity_pips_per_sec", 3.0))
    if v_thr > 0.0:
        v = inp.velocity_pips_per_sec
        if inp.direction == +1 and v < -v_thr:
            return False, f"counter-trend veto vel={v:.2f} < -{v_thr:.2f}"
        if inp.direction == -1 and v > v_thr:
            return False, f"counter-trend veto vel={v:.2f} > {v_thr:.2f}"

    # (C) Trend-state veto: M1 regression actively pointing the other way.
    # MQL5 only keys off TREND_UP / TREND_DOWN; here we also treat the
    # "strong_*" buckets as vetoing because they are a strict superset of
    # "against".  If rev_confirm already passed and cfg allows it, a plain
    # (non-strong) opposing trend is relaxed to a pass -- the strong
    # variant still hard-vetoes.
    trend_veto_on = bool(_cfg_get(cfg, "counter_trend_trend_veto", True))
    if trend_veto_on:
        ts = (inp.trend_state or "").lower()
        relax_on_rev = bool(
            _cfg_get(cfg, "counter_trend_trend_veto_relax_on_rev_confirm", True)
        )

        if inp.direction == +1:  # fresh long
            if ts == "strong_down":
                return False, "counter-trend veto trend=strong_down"
            if ts == "down":
                if not (relax_on_rev and inp.rev_confirm_passed):
                    return False, "counter-trend veto trend=down"

        elif inp.direction == -1:  # fresh short
            if ts == "strong_up":
                return False, "counter-trend veto trend=strong_up"
            if ts == "up":
                if not (relax_on_rev and inp.rev_confirm_passed):
                    return False, "counter-trend veto trend=up"

    return True, ""
