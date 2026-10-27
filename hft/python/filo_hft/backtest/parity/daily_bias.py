"""DailyBias parity port of mql5/Include/FiloHFT/DailyBias.mqh.

Tracks D1 break/bounce events and exposes an exponentially-decayed
directional bias. Backtest parity only — the live engine scans KeyLevel
zones on M5 bar closes; this standalone port operates on the D1 levels
passed into `step()` (prev_high / prev_low) and detects break/bounce
crossings from the mid-price stream. Decay, compounding and clipping
math mirror the MQL5 implementation exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List

_LN2 = 0.6931471805599453
_CAP = 16  # FILO_DAILY_BIAS_CAP

# Pip size for a 5-digit FX symbol (EURUSD-style). Parity harness uses
# this conversion; override via cfg.pip_size if present.
_DEFAULT_PIP = 0.0001


def _pip_size(cfg: Any) -> float:
    return float(getattr(cfg, "pip_size", _DEFAULT_PIP))


def _price_from_pips(cfg: Any, pips: float) -> float:
    return float(pips) * _pip_size(cfg)


@dataclass
class _Event:
    level_id: str
    level_price: float
    etype: str        # "break" | "bounce"
    direction: int    # +1 / -1
    ts_ms: int
    alive: bool = True


@dataclass
class DailyBiasState:
    current_bias: float
    last_event_ts_ms: int
    last_event_kind: str  # "break_up"|"break_dn"|"bounce_up"|"bounce_dn"|"none"


class DailyBiasEngine:
    """Python port of DailyBias from DailyBias.mqh."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._events: List[_Event] = []
        self._last_event_ts_ms: int = 0
        self._last_event_kind: str = "none"
        # Previous-tick mid for crossing detection; seeded lazily.
        self._prev_mid: float | None = None
        self._prev_day_key: tuple[float, float] | None = None
        # Per-(day, level) latch so we don't re-fire the same event repeatedly
        # from consecutive ticks straddling the level.
        self._fired: set[tuple[float, float, str, int]] = set()

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------
    def step(
        self,
        ts_ms: int,
        mid: float,
        d1_high: float,
        d1_low: float,
        d1_open: float,
        d1_prev_high: float,
        d1_prev_low: float,
    ) -> None:
        if not bool(getattr(self.cfg, "daily_bias_enable", True)):
            return

        day_key = (float(d1_prev_high), float(d1_prev_low))
        if self._prev_day_key != day_key:
            # New D1 bar → forget per-day latches; events themselves persist
            # and keep decaying until max_age.
            self._fired.clear()
            self._prev_day_key = day_key
            self._prev_mid = None

        tol = _price_from_pips(self.cfg, getattr(self.cfg, "daily_bias_break_tolerance_pips", 1.5))
        min_wick = _price_from_pips(self.cfg, getattr(self.cfg, "daily_bias_bounce_min_wick_pips", 4.0))

        prev = self._prev_mid
        self._prev_mid = mid
        if prev is None:
            self._prune(ts_ms)
            return

        # Two candidate D1 levels: prev-day high and prev-day low.
        for level_id, level_px in (("prev_high", d1_prev_high), ("prev_low", d1_prev_low)):
            if level_px <= 0.0:
                continue

            # BREAK up: crossed upward through level + tol.
            key_bu = (day_key[0], day_key[1], level_id, +1)
            if prev <= (level_px - tol) and mid >= (level_px + tol):
                if key_bu not in self._fired:
                    self._fired.add(key_bu)
                    self._add(_Event(level_id, level_px, "break", +1, ts_ms))
                    self._last_event_ts_ms = ts_ms
                    self._last_event_kind = "break_up"
                continue

            # BREAK down
            key_bd = (day_key[0], day_key[1], level_id, -1)
            if prev >= (level_px + tol) and mid <= (level_px - tol):
                if key_bd not in self._fired:
                    self._fired.add(key_bd)
                    self._add(_Event(level_id, level_px, "break", -1, ts_ms))
                    self._last_event_ts_ms = ts_ms
                    self._last_event_kind = "break_dn"
                continue

            # BOUNCE up: intraday low dipped under level by >= min_wick,
            # price now back above level.
            if d1_low <= (level_px - min_wick) and mid > level_px:
                key = (day_key[0], day_key[1], f"{level_id}:bounce", +1)
                if key not in self._fired:
                    self._fired.add(key)
                    self._add(_Event(level_id, level_px, "bounce", +1, ts_ms))
                    self._last_event_ts_ms = ts_ms
                    self._last_event_kind = "bounce_up"
                continue

            # BOUNCE down
            if d1_high >= (level_px + min_wick) and mid < level_px:
                key = (day_key[0], day_key[1], f"{level_id}:bounce", -1)
                if key not in self._fired:
                    self._fired.add(key)
                    self._add(_Event(level_id, level_px, "bounce", -1, ts_ms))
                    self._last_event_ts_ms = ts_ms
                    self._last_event_kind = "bounce_dn"

        # Silence unused-local warning for d1_high/low/open when untouched.
        _ = (d1_high, d1_low, d1_open)
        self._prune(ts_ms)

    # ------------------------------------------------------------------
    # Event bookkeeping
    # ------------------------------------------------------------------
    def _add(self, e: _Event) -> None:
        # Dedup: identical (level_id, ts_ms, type) — mirrors MQL5 _Add.
        for ex in self._events:
            if ex.alive and ex.level_id == e.level_id and ex.ts_ms == e.ts_ms and ex.etype == e.etype:
                return
        if len(self._events) < _CAP:
            self._events.append(e)
            return
        # Full: evict oldest by ts_ms.
        oldest = min(range(len(self._events)), key=lambda i: self._events[i].ts_ms)
        self._events[oldest] = e

    def _prune(self, now_ms: int) -> None:
        max_age = int(getattr(self.cfg, "daily_bias_max_age_ms", 900_000))
        for ex in self._events:
            if ex.alive and (now_ms - ex.ts_ms) > max_age:
                ex.alive = False

    # ------------------------------------------------------------------
    # Bias computation — byte-for-byte equivalent of DailyBias::Bias.
    # ------------------------------------------------------------------
    def bias_at(self, ts_ms: int) -> float:
        if not bool(getattr(self.cfg, "daily_bias_enable", True)):
            return 0.0
        if not self._events:
            return 0.0
        hl = float(getattr(self.cfg, "daily_bias_half_life_ms", 360_000))
        if hl <= 0.0:
            return 0.0
        max_age = int(getattr(self.cfg, "daily_bias_max_age_ms", 900_000))
        near_px = _price_from_pips(self.cfg, getattr(self.cfg, "daily_bias_nearby_d1_compound_pips", 10.0))

        w: List[float] = []
        s: List[int] = []
        p: List[float] = []
        for ex in self._events:
            if not ex.alive:
                continue
            age = ts_ms - ex.ts_ms
            if age < 0:
                age = 0
            if age > max_age:
                continue
            w.append(math.exp(-_LN2 * age / hl))
            s.append(ex.direction)
            p.append(ex.level_price)

        n = len(w)
        if n == 0:
            return 0.0
        compound = [False] * n
        for i in range(n):
            for j in range(i + 1, n):
                if s[i] == s[j] and abs(p[i] - p[j]) <= near_px:
                    compound[i] = True
                    compound[j] = True
        total = 0.0
        for i in range(n):
            c = 2.0 if compound[i] else 1.0
            total += s[i] * w[i] * c
        if total > 1.0:
            total = 1.0
        if total < -1.0:
            total = -1.0
        return total

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def state(self) -> DailyBiasState:
        return DailyBiasState(
            current_bias=self.bias_at(self._last_event_ts_ms or 0),
            last_event_ts_ms=self._last_event_ts_ms,
            last_event_kind=self._last_event_kind,
        )

    def score_adjustment(self, direction: int, ts_ms: int) -> float:
        """Additive score nudge for a candidate setup.

        Mirrors Setups._TryFamily behaviour: `Bias() * daily_bias_score_weight`,
        signed by candidate direction so aligned setups get +weight*|bias|
        and opposed setups get -weight*|bias|.
        """
        weight = float(getattr(self.cfg, "daily_bias_score_weight", 0.20))
        b = self.bias_at(ts_ms)
        return float(direction) * b * weight
