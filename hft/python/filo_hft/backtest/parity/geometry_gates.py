"""Geometry / proximity gates ported from mql5/Include/FiloHFT/Setups.mqh.

Backtest-parity only, not live code. Mirrors:
  at_zone               -> Setups.mqh L246-L255
  continuation extreme  -> Setups.mqh L263-L269
  htf_clearance         -> new Python-side gate (no MQL counterpart yet)
Family strings match Setups.mqh canonical names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

# Families that must sit *at* a KL zone to be valid (reversion thesis).
_REVERSION_FAMILIES: frozenset[str] = frozenset({
    "mean_reversion",
    "failed_auction",
    "rejection",
})

# Families that ride an extension — we use the looser continuation extreme mult.
_CONTINUATION_FAMILIES: frozenset[str] = frozenset({
    "breakout",
    "retest",
    "engulfing_accel",
    "momentum_cont",
})

_HTF_TIMEFRAMES: frozenset[str] = frozenset({"D1", "H4"})


@dataclass
class KLZoneRef:
    zone_id: str
    center: float
    half_width: float
    timeframe: str  # "D1"|"H4"|"H1"|"M30"|"M15"|"M5"


def _get(cfg: Any, name: str, default: float) -> float:
    """Look up a cfg field via attribute or mapping, falling back to default."""
    if cfg is None:
        return default
    val = getattr(cfg, name, None)
    if val is None and isinstance(cfg, dict):
        val = cfg.get(name)
    return float(val) if val is not None else default


def _pip_size(cfg: Any) -> float:
    # Backtest cfg carries pip_size explicitly; default to FX 0.0001.
    return _get(cfg, "pip_size", 0.0001)


def check_at_zone(
    mid: float,
    family: str,
    zones: Iterable[KLZoneRef],
    cfg: Any,
) -> tuple[bool, str, str]:
    """Return (ok, zone_id, reason). Non-reversion families always pass."""
    if family not in _REVERSION_FAMILIES:
        return True, "", ""

    mult = _get(cfg, "at_zone_mult", 1.5)
    min_pips = _get(cfg, "at_zone_min_pips", 3.0)
    pip = _pip_size(cfg)

    best_id = ""
    best_slack_pips = float("inf")
    for z in zones:
        # Distance from mid to the nearer edge of the zone.
        edge_dist = max(0.0, abs(mid - z.center) - z.half_width)
        edge_dist_pips = edge_dist / pip if pip > 0 else edge_dist
        # Mirror of Setups.mqh: tolerance = max(half_width_pips * mult, min_pips).
        half_pips = (z.half_width / pip) if pip > 0 else z.half_width
        tol_pips = max(half_pips * mult, min_pips)
        slack = edge_dist_pips - tol_pips
        if slack < best_slack_pips:
            best_slack_pips = slack
            best_id = z.zone_id
            if slack <= 0.0:
                return True, z.zone_id, ""

    if best_id == "":
        return False, "", "no KL zones available"
    return False, best_id, (
        f"not at zone ({best_slack_pips:+.1f} pips beyond tolerance)"
    )


def check_htf_clearance(
    mid: float,
    direction: int,
    zones: Iterable[KLZoneRef],
    cfg: Any,
) -> tuple[bool, str]:
    """Ensure clear space to nearest D1/H4 obstacle in trade direction.

    direction: +1 long, -1 short. Obstacle for a long is the next HTF zone
    *above* mid (its near edge), for a short the next one *below*.
    """
    if direction not in (+1, -1):
        return True, ""

    need_pips = _get(cfg, "htf_clearance_pips", 8.0)
    if need_pips <= 0.0:
        return True, ""
    pip = _pip_size(cfg)

    nearest: float = float("inf")
    for z in zones:
        if z.timeframe not in _HTF_TIMEFRAMES:
            continue
        if direction > 0:
            near_edge = z.center - z.half_width
            if near_edge <= mid:
                continue  # behind or straddling price, not a ceiling
            dist = near_edge - mid
        else:
            near_edge = z.center + z.half_width
            if near_edge >= mid:
                continue
            dist = mid - near_edge
        if dist < nearest:
            nearest = dist

    if nearest == float("inf"):
        return True, ""  # no HTF obstacle in front -> clear

    nearest_pips = nearest / pip if pip > 0 else nearest
    if nearest_pips < need_pips:
        return False, (
            f"htf_clearance {nearest_pips:.1f} < {need_pips:.1f} pips"
        )
    return True, ""


def continuation_extreme_block_mult_for(family: str, cfg: Any) -> float:
    """Multiplier on channel extreme_block_mult for continuation families.
    Returns 1.0 for non-continuation families. Mirrors the fallback in
    Setups.mqh: if the setup-level mult is unset (<=0), use the channel one.
    """
    if family not in _CONTINUATION_FAMILIES:
        return 1.0
    mult = _get(cfg, "continuation_extreme_block_mult", 1.5)
    if mult <= 0.0:
        mult = _get(cfg, "chan_extreme_block_mult", 1.1)
    return mult
