"""Echo-count tracker — backtest parity port of Setups.mqh::_ComputeEchoCount.

Mirrors the MQL5 circular buffer of the last FILO_ECHO_RING (=16) accepted
setup candidates. On each new candidate, counts how many prior entries in the
ring share the same (zone_id, direction) AND were emitted within
`window_ms` of `now_ts_ms`. That count is the echo_count tag logged to
setups.csv / intents — purely observational, does NOT veto.

Source of truth (READ-ONLY reference):
  mql5/Include/FiloHFT/Setups.mqh  — _ComputeEchoCount / _PushEcho / FILO_ECHO_RING
  mql5/Include/FiloHFT/Config.mqh  — setup_echo_enable, setup_echo_window_ms

Semantics preserved exactly:
  - ring size is a fixed constant (FILO_ECHO_RING = 16)
  - empty slots (ts_ms <= 0) are skipped
  - inclusion rule matches MQL5: entries where ts_ms >= (now_ms - window_ms)
    are considered in-window (i.e. MQL5's `ts_ms < cutoff` exclusion)
  - echo_count is computed BEFORE appending the new candidate
  - head advances modulo ring size, overwriting the oldest slot
"""

from dataclasses import dataclass, field
from typing import List

# Matches #define FILO_ECHO_RING 16 in Setups.mqh.
FILO_ECHO_RING: int = 16


@dataclass
class EchoCandidate:
    ts_ms: int
    zone_id: str
    direction: int  # +1 long / -1 short
    family: str


@dataclass
class _EchoEntry:
    ts_ms: int = 0
    zone_id: str = ""
    direction: int = 0


class EchoTracker:
    """Rolling ring buffer of recently-accepted setup candidates."""

    def __init__(self, ring_size: int = FILO_ECHO_RING) -> None:
        if ring_size <= 0:
            raise ValueError("ring_size must be positive")
        self._ring_size: int = ring_size
        self._ring: List[_EchoEntry] = [_EchoEntry() for _ in range(ring_size)]
        self._head: int = 0

    def count_for(
        self, zone_id: str, direction: int, now_ts_ms: int, window_ms: int
    ) -> int:
        """Count prior ring entries matching (zone_id, direction) within window.

        Mirrors _ComputeEchoCount: entries with ts_ms <= 0 are empty and skipped;
        an entry is in-window iff ts_ms >= (now_ts_ms - window_ms).
        """
        cutoff = now_ts_ms - window_ms
        n = 0
        for e in self._ring:
            if e.ts_ms <= 0:
                continue
            if e.ts_ms < cutoff:
                continue
            if e.direction != direction:
                continue
            if e.zone_id != zone_id:
                continue
            n += 1
        return n

    def record(self, cand: EchoCandidate, window_ms: int = 120_000) -> int:
        """Compute echo_count for `cand` against prior ring state, then append.

        Returns the count of prior same-zone, same-direction candidates emitted
        within `window_ms` of cand.ts_ms. The append happens AFTER the count,
        matching Setups.mqh where _ComputeEchoCount runs before _PushEcho.
        """
        echo = self.count_for(cand.zone_id, cand.direction, cand.ts_ms, window_ms)
        slot = self._ring[self._head]
        slot.ts_ms = cand.ts_ms
        slot.zone_id = cand.zone_id
        slot.direction = cand.direction
        self._head = (self._head + 1) % self._ring_size
        return echo
