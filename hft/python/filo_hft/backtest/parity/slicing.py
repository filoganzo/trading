"""Backtest parity port of the KL slicing-penalty logic from
KeyLevels.mqh (ObservePrice / EffectiveWeight).

Semantics (mirrors the MQL5 header comments verbatim):
  - Each zone has an asymmetric "dead as ceiling" and "dead as floor"
    directional cooldown.
  - A body commit UP past `center + k*half_width` marks the zone dead
    as a ceiling (price sliced UP through it, so it can no longer be
    trusted as a ceiling for a SHORT entry).
  - A body commit DOWN past `center - k*half_width` marks it dead as
    a floor (hurts LONG entries trying to bounce off it).
  - Only NEW transitions re-arm the cooldown — dwelling on the far
    side does not keep re-stamping the deadline (non-cumulative).
  - During the cooldown window the weight decays linearly from
    `floor_factor` (at the moment of the slice) back up to 1.0 at the
    deadline.

NOTE on signature vs MQL5:
  The MQL5 uses a configurable `kl_slice_commitment_mult` (default 2.0)
  to compute the far-edge commit threshold as `center +/- k*half_width`.
  This Python port takes `zone_half_width` directly and uses k=1.0
  implicitly; callers that want full MQL5 parity should pass
  `kl_slice_commitment_mult * raw_half_width` as `zone_half_width`.
  Both `commit_window_ms` and `decay_ms` correspond to the same
  `kl_slice_cooldown_ms` in Config.mqh — the first is the deadline
  stamp applied in observe_price, the second is the denominator of
  the linear decay curve in effective_weight. Keeping them as
  separate params lets callers experiment without breaking parity.
"""

from dataclasses import dataclass


@dataclass
class KLZoneSliceState:
    last_committed_side: int = 0       # -1 = below, +1 = above, 0 = none
    dead_as_ceiling_until_ms: int = 0
    dead_as_floor_until_ms: int = 0
    ceiling_armed_at_ms: int = 0
    floor_armed_at_ms: int = 0


def observe_price(
    state: KLZoneSliceState,
    zone_center: float,
    zone_half_width: float,
    mid: float,
    ts_ms: int,
    commit_window_ms: int,
    decay_ms: int,
) -> None:
    """Update slice state when a body commits past a zone's far edge.

    Mirrors KeyLevels.mqh::ObservePrice. Noop if the cooldown window
    is non-positive (matches the MQL5 early-out when
    `k <= 0 || cd <= 0`).
    """
    if commit_window_ms <= 0 or decay_ms <= 0:
        return

    top_commit = zone_center + zone_half_width
    bot_commit = zone_center - zone_half_width

    new_side = state.last_committed_side
    if mid >= top_commit:
        new_side = +1
    elif mid <= bot_commit:
        new_side = -1

    # Only stamp on a NEW transition — non-cumulative semantics.
    if new_side != state.last_committed_side:
        if new_side == +1:
            # Price sliced UP through the zone → dead as ceiling
            # (hurts SHORT entries that would use it as resistance).
            state.dead_as_ceiling_until_ms = ts_ms + commit_window_ms
            state.ceiling_armed_at_ms = ts_ms
        elif new_side == -1:
            # Price sliced DOWN through the zone → dead as floor
            # (hurts LONG entries that would bounce off it as support).
            state.dead_as_floor_until_ms = ts_ms + commit_window_ms
            state.floor_armed_at_ms = ts_ms
        state.last_committed_side = new_side


def effective_weight(
    state: KLZoneSliceState,
    base_weight: float,
    direction: int,
    ts_ms: int,
    decay_ms: int,
    floor_factor: float,
) -> float:
    """Return the weight scaled by any active slicing penalty.

    direction: +1 = LONG (wants the zone as a FLOOR/support → consults
    `dead_as_floor_until_ms`), -1 = SHORT (wants the zone as a
    CEILING/resistance → consults `dead_as_ceiling_until_ms`).
    Mirrors KeyLevels.mqh::EffectiveWeight.
    """
    if decay_ms <= 0:
        return base_weight

    if direction > 0:
        deadline = state.dead_as_floor_until_ms
    else:
        deadline = state.dead_as_ceiling_until_ms

    if deadline <= 0 or ts_ms >= deadline:
        return base_weight

    # remaining/cd in [0,1]; progress 0 = just sliced, 1 = fully recovered.
    remaining = float(deadline - ts_ms)
    progress = 1.0 - (remaining / float(decay_ms))
    if progress < 0.0:
        progress = 0.0
    if progress > 1.0:
        progress = 1.0
    scale = floor_factor + (1.0 - floor_factor) * progress
    return base_weight * scale
