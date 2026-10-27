"""Python parity port of SETUP_MOMENTUM_CONT from mql5/Include/FiloHFT/Setups.mqh.

Source: _DetectMomentumCont (Setups.mqh ~L685-816).

Thesis (from MQL5 comments):
    After a rocket M1 candle (single-bar body >= min_move_pips), catch the
    first shallow pullback in the rocket's direction, provided the move has
    persisted (velocity sanity), partner (XAG) agrees, and there is no
    opposing D1 KL obstacle within veto distance.

This module is strictly for backtest parity. It does NOT read MT5 bars or
correlation state -- the caller is expected to have derived:
    * a direction_hint and impulse extremes from the equivalent of the
      rocket candle,
    * pullback_depth_pips vs that extreme,
    * velocity_pips_per_sec as the persistence proxy,
    * partner_velocity as the signed XAG agreement proxy,
    * trend_bars_in_state as the "trend_state has held N bars" proxy.

The MQL5 version does not itself read trend_bars_in_state or a burst
counter -- those inputs are required by the spec so that parity callers
can layer additional gating identical to the live engine's upstream
context. We honour them as extra gates when the cfg exposes matching
fields; otherwise they are ignored.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class MomentumContInputs:
    mid: float
    ts_ms: int
    direction_hint: int               # +1 or -1 from trend_state
    trend_state: str
    trend_bars_in_state: int
    velocity_pips_per_sec: float
    last_impulse_high: float
    last_impulse_low: float
    pullback_depth_pips: float
    partner_velocity: float
    burst_count_up: int
    burst_count_dn: int


@dataclass
class MomentumContSignal:
    fired: bool
    direction: int
    score: float
    stop_pips: float
    target_pips: float
    lot_mult: float
    reason: str


def _g(cfg: Any, name: str, default):
    """cfg may be a dataclass, dict, or SimpleNamespace — read tolerantly."""
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _miss(reason: str) -> MomentumContSignal:
    return MomentumContSignal(
        fired=False, direction=0, score=0.0,
        stop_pips=0.0, target_pips=0.0, lot_mult=0.0, reason=reason,
    )


def detect_momentum_cont(inp: MomentumContInputs, cfg: Any) -> MomentumContSignal:
    if not _g(cfg, "setup_momentum_cont_enable", True):
        return _miss("disabled")

    # --- direction from rocket / trend_state hint ---
    d = int(inp.direction_hint)
    if d == 0:
        if inp.trend_state == "up":
            d = +1
        elif inp.trend_state == "down":
            d = -1
    if d not in (+1, -1):
        return _miss("no_direction")

    # --- persistence gate: trend must have held N bars ---
    min_persist_bars = int(_g(cfg, "setup_momentum_cont_min_persistence_bars", 0))
    if min_persist_bars > 0 and inp.trend_bars_in_state < min_persist_bars:
        return _miss("persistence_bars")

    # --- rocket body bounds: abs(impulse_high - impulse_low) in pips ---
    # Replaces MQL5's single-bar body-size gate. Caller is expected to have
    # populated last_impulse_high/low from the rocket candle (or equivalent).
    pip_size = float(_g(cfg, "pip_size", 0.0001))
    if pip_size <= 0.0:
        pip_size = 0.0001
    rocket_range_pips = abs(inp.last_impulse_high - inp.last_impulse_low) / pip_size
    min_move = float(_g(cfg, "setup_momentum_cont_min_move_pips", 50.0))
    max_move = float(_g(cfg, "setup_momentum_cont_max_move_pips", 120.0))
    if rocket_range_pips < min_move:
        return _miss("rocket_too_small")
    if rocket_range_pips > max_move:
        return _miss("rocket_too_big")

    # --- velocity persistence sanity ---
    # MQL5 checks: price `persistence_ms` ago is still >= persist_min_delta_pips
    # on the pre-rocket side. We approximate with velocity_pips_per_sec * window.
    pers_ms = int(_g(cfg, "setup_momentum_cont_persistence_ms", 75000))
    if pers_ms < 1000:
        pers_ms = 1000
    pers_min_delta = float(_g(cfg, "setup_momentum_cont_persist_min_delta_pips", 30.0))
    delta_pips = inp.velocity_pips_per_sec * (pers_ms / 1000.0)
    if d > 0 and delta_pips < pers_min_delta:
        return _miss("persist_velocity_long")
    if d < 0 and delta_pips > -pers_min_delta:
        return _miss("persist_velocity_short")

    # --- shallow pullback bounds from rocket's leading extreme ---
    # The caller should have computed pullback_depth_pips from the directional
    # extreme; we gate the bounds directly.
    min_pb = float(_g(cfg, "setup_momentum_cont_pullback_min_pips",
                       _g(cfg, "setup_momentum_cont_min_pullback_pips", 3.0)))
    max_pb = float(_g(cfg, "setup_momentum_cont_pullback_max_pips",
                       _g(cfg, "setup_momentum_cont_max_pullback_pips", 8.0)))
    pb = float(inp.pullback_depth_pips)
    if pb < min_pb:
        return _miss("pullback_shallow")
    if pb > max_pb:
        return _miss("pullback_deep")

    # --- partner-instrument (XAG) agreement ---
    partner_min = float(_g(cfg, "setup_momentum_cont_partner_min_ag", 0.40))
    partner_signed = inp.partner_velocity if d > 0 else -inp.partner_velocity
    if partner_signed < partner_min:
        return _miss("partner_disagree")

    # --- optional burst-count gate (extra beyond MQL5) ---
    min_burst = int(_g(cfg, "setup_momentum_cont_min_burst_count", 0))
    if min_burst > 0:
        have = inp.burst_count_up if d > 0 else inp.burst_count_dn
        if have < min_burst:
            return _miss("burst_count")

    # --- scoring: base 0.55, +0.005/pip over 50 (cap 40), -0.02 per pip
    #     away from the 5-pip pullback sweet spot. Matches MQL5. ---
    base = 0.55
    bonus = 0.005 * max(0.0, min(rocket_range_pips - 50.0, 40.0))
    penalty = 0.02 * abs(pb - 5.0)
    score = max(0.0, min(1.0, base + bonus - penalty))

    stop_pips = float(_g(cfg, "setup_momentum_cont_stop_pips", 50.0))
    target_pips = float(_g(cfg, "setup_momentum_cont_target_pips", 50.0))
    lot_mult = float(_g(cfg, "setup_momentum_cont_lot_mult", 0.2))

    return MomentumContSignal(
        fired=True,
        direction=d,
        score=score,
        stop_pips=stop_pips,
        target_pips=target_pips,
        lot_mult=lot_mult,
        reason="ok",
    )
