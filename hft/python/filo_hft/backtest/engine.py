"""Event-driven bar-level replay engine.

Mirrors the MQL5 hot-path but operates on M1 bars rather than ticks. The
engine walks forward one M1 bar at a time, maintaining:

- KL zone state across D1/H4/H1/M30/M15/M5 (incremental refresh on TF flip)
- regression channel recomputed on M1 bar close
- rolling per-symbol correlation state
- open-position bookkeeping with hidden SL / TP / break-even / trailing

Each M1 bar simulates intrabar price behavior via a canonical OHLC traversal
(open -> low -> high -> close for up bars, open -> high -> low -> close for
down bars). This is explicitly "good enough for parameter sweeps", not a
tick-accurate replay.

The single public entry point is `run_backtest(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from filo_hft.config import FiloConfig, pips_to_usd
from filo_hft.backtest.parity.slicing import KLZoneSliceState, observe_price, effective_weight

if TYPE_CHECKING:  # pragma: no cover — typing only
    pass


# ======================================================================
# Public result type
# ======================================================================


@dataclass
class BacktestResult:
    """Everything a downstream consumer (sweeps, ML, UI) needs."""

    trades: pd.DataFrame
    candidates: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict


# ======================================================================
# Constants
# ======================================================================


TF_ORDER: tuple[str, ...] = ("D1", "H4", "H1", "M30", "M15", "M5")

SETUP_FAMILIES: tuple[str, ...] = (
    "breakout",
    "retest",
    "rejection",
    "fakeout",
    "failed_auction",
    "engulfing_accel",
    "mean_reversion",
)

# Families treated as "continuation" for channel-extreme and exhaustion gates.
_CONTINUATION: frozenset[str] = frozenset({"breakout", "retest", "engulfing_accel"})

# Families treated as "reversion" for spread gates.
_REVERSION: frozenset[str] = frozenset({"mean_reversion", "failed_auction", "rejection"})

# Families that fire AT a KL expecting rejection — eligible for zone-anchored
# entry triggers when execution.reversal_anchor == "zone".
_REVERSAL_ENTRY: frozenset[str] = frozenset(
    {"rejection", "fakeout", "failed_auction", "mean_reversion"}
)

# Cooldown (bars) per family to avoid re-firing on every bar after a hit.
# MQL5 uses 800ms; at M1 the smallest meaningful cooldown is 1 bar.
_FAMILY_COOLDOWN_BARS: int = 2


# ======================================================================
# Key-level engine
# ======================================================================


@dataclass
class _KLZone:
    center: float
    half_width: float
    origin_tf: str
    weight: float
    role: str
    zone_id: str
    alive: bool = True
    slice_state: KLZoneSliceState = field(default_factory=KLZoneSliceState)


class _KeyLevels:
    """Pythonic port of KeyLevels.mqh.

    Builds zones off of per-TF OHLC, applies body-cross invalidation from
    later closed bars in the same TF, and merges overlapping zones keeping
    the highest-weight origin. Rebuilds only when the relevant TF flips.
    """

    def __init__(self, cfg: FiloConfig) -> None:
        self._cfg = cfg
        self._zones: list[_KLZone] = []
        self._next_id: int = 0
        self._last_ts_by_tf: dict[str, pd.Timestamp | None] = {tf: None for tf in TF_ORDER}
        # Per-TF bar frames and "current index" (index of the most recent
        # CLOSED bar, i.e. the bar whose close_ts <= current M1 close_ts).
        self._tf_bars: dict[str, pd.DataFrame] = {}
        self._tf_np: dict[str, dict[str, np.ndarray]] = {}
        self._tf_cursor: dict[str, int] = {tf: -1 for tf in TF_ORDER}

    # ---- setup ----------------------------------------------------
    def attach_tf(self, tf: str, bars: pd.DataFrame) -> None:
        """Store per-TF bars. `bars` must be sorted ascending by ts."""
        self._tf_bars[tf] = bars
        if len(bars) == 0:
            self._tf_np[tf] = {
                "ts": np.empty(0, dtype="datetime64[ns]"),
                "open": np.empty(0),
                "high": np.empty(0),
                "low": np.empty(0),
                "close": np.empty(0),
            }
        else:
            ts_col = bars["ts"]
            if getattr(ts_col.dt, "tz", None) is not None:
                ts_col = ts_col.dt.tz_convert("UTC").dt.tz_localize(None)
            self._tf_np[tf] = {
                "ts": ts_col.to_numpy(dtype="datetime64[ns]"),
                "open": bars["open"].to_numpy(dtype=float),
                "high": bars["high"].to_numpy(dtype=float),
                "low": bars["low"].to_numpy(dtype=float),
                "close": bars["close"].to_numpy(dtype=float),
            }

    # ---- helpers --------------------------------------------------
    def _zone_half_width(self) -> float:
        pip_price = self._cfg.instrument.pip_in_points * 0.01
        return self._cfg.keylevels.zone_half_width_pips * pip_price

    def _merge_tol(self) -> float:
        pip_price = self._cfg.instrument.pip_in_points * 0.01
        return self._cfg.keylevels.merge_overlap_pips * pip_price

    def _weight_for(self, tf: str) -> float:
        kl = self._cfg.keylevels
        return {
            "D1": kl.weight_d1,
            "H4": kl.weight_h4,
            "H1": kl.weight_h1,
            "M30": kl.weight_m30,
            "M15": kl.weight_m15,
            "M5": kl.weight_m5,
        }.get(tf, 0.0)

    def _tf_lookback(self, tf: str) -> int:
        d = self._cfg.keylevels.lookback_days
        return {
            "D1": d,
            "H4": d * 6,
            "H1": d * 24,
            "M30": d * 48,
            "M15": d * 96,
            "M5": d * 288,
        }.get(tf, 0)

    def _make_id(self) -> str:
        self._next_id += 1
        return f"kl{self._next_id:06d}"

    def _is_invalidated(self, tf: str, origin_idx: int, end_idx: int, price: float) -> bool:
        """Scalar fallback — fast path is the vectorized `_refresh_tf` below.

        Returns True iff any bar strictly after `origin_idx` and up to
        `end_idx` has a body (min(o,c)..max(o,c)) that straddles `price`.
        """
        np_tf = self._tf_np[tf]
        o = np_tf["open"]
        c = np_tf["close"]
        lo = origin_idx + 1
        hi = end_idx + 1  # slice-exclusive
        if lo >= hi:
            return False
        body_max = np.maximum(o[lo:hi], c[lo:hi])
        body_min = np.minimum(o[lo:hi], c[lo:hi])
        return bool(np.any((body_max > price) & (body_min < price)))

    def _find_merge_idx(self, center: float) -> int:
        tol = self._merge_tol()
        for i, z in enumerate(self._zones):
            if not z.alive:
                continue
            if abs(z.center - center) <= tol:
                return i
        return -1

    def _add_or_merge(self, price: float, tf: str, role: str) -> None:
        w = self._weight_for(tf)
        half = self._zone_half_width()
        existing = self._find_merge_idx(price)
        if existing >= 0:
            z = self._zones[existing]
            if w <= z.weight:
                return
            z.center = price
            z.half_width = half
            z.origin_tf = tf
            z.weight = w
            z.role = "merged"
            return
        self._zones.append(
            _KLZone(
                center=price,
                half_width=half,
                origin_tf=tf,
                weight=w,
                role=role,
                zone_id=self._make_id(),
                alive=True,
            )
        )

    def _refresh_tf(self, tf: str, end_idx: int) -> None:
        """Rebuild zones originating from `tf` using bars [end_idx-lookback, end_idx].

        Vectorized: instead of calling `_is_invalidated` per candidate
        price (which slices numpy arrays O(lookback) times per bar), we
        compute the full set of candidate (price, origin_idx) pairs in
        bulk and eliminate any whose price falls inside the body of a
        STRICTLY LATER bar.
        """
        if tf not in self._tf_np:
            return
        ts_arr = self._tf_np[tf]["ts"]
        if end_idx < 1 or len(ts_arr) == 0:
            return
        lookback = min(self._tf_lookback(tf), end_idx)
        if lookback < 2:
            return
        start = max(1, end_idx - lookback + 1)

        high = self._tf_np[tf]["high"]
        low = self._tf_np[tf]["low"]
        open_ = self._tf_np[tf]["open"]
        close = self._tf_np[tf]["close"]

        idxs = np.arange(start, end_idx + 1, dtype=np.int64)
        h_slice = high[start : end_idx + 1]
        l_slice = low[start : end_idx + 1]
        valid = (h_slice > 0) & (l_slice > 0)

        # Build candidate arrays: price, origin index, role flag
        prices = np.concatenate([h_slice[valid], l_slice[valid]])
        origins = np.concatenate([idxs[valid], idxs[valid]])
        roles = np.concatenate(
            [np.zeros(int(valid.sum()), dtype=np.int8), np.ones(int(valid.sum()), dtype=np.int8)]
        )

        if len(prices) == 0:
            return

        # Body intervals for bars in (start+1 .. end] — any later bar that
        # spans the origin's price invalidates it.
        body_max = np.maximum(open_[start : end_idx + 1], close[start : end_idx + 1])
        body_min = np.minimum(open_[start : end_idx + 1], close[start : end_idx + 1])

        # For each candidate i, survive iff NO bar j where j-index > origin[i]
        # has body_min[j] < prices[i] < body_max[j].
        # We do this with a per-bar-j mask and accumulate a "killed" boolean.
        killed = np.zeros(len(prices), dtype=bool)
        # map local idx (0..len-1) -> global idx via start offset
        rel_origin = origins - start  # in [0 .. lookback-1]
        for j in range(1, end_idx + 1 - start):
            bmax = body_max[j]
            bmin = body_min[j]
            if bmax <= bmin:
                continue
            # candidates with origin strictly before this bar j, whose price
            # lies strictly inside the body, are killed.
            mask = (~killed) & (rel_origin < j) & (prices > bmin) & (prices < bmax)
            if mask.any():
                killed |= mask

        survivors_idx = np.where(~killed)[0]
        for k in survivors_idx:
            self._add_or_merge(
                float(prices[k]), tf, "high" if roles[k] == 0 else "low"
            )

    def _compact(self) -> None:
        self._zones = [z for z in self._zones if z.alive]

    def _full_rebuild(self, now_ts: pd.Timestamp) -> None:
        self._zones = []
        for tf in TF_ORDER:
            if tf not in self._tf_np:
                continue
            idx = self._tf_cursor[tf]
            if idx >= 1:
                self._refresh_tf(tf, idx)
        self._compact()

    def _incremental_update(self, tf: str, new_idx: int) -> None:
        """Cheap per-flip update: invalidate existing zones the new bar
        body crosses, then add the new bar's high/low candidates.
        """
        if tf not in self._tf_np:
            return
        if new_idx < 1:
            return
        o = float(self._tf_np[tf]["open"][new_idx])
        c = float(self._tf_np[tf]["close"][new_idx])
        h = float(self._tf_np[tf]["high"][new_idx])
        lo = float(self._tf_np[tf]["low"][new_idx])
        if h <= 0 or lo <= 0:
            return

        bmin = min(o, c)
        bmax = max(o, c)
        if bmax > bmin:
            # invalidate any existing zone whose center is strictly inside
            # the new bar's body (regardless of origin TF — matches the
            # spirit of body-cross invalidation)
            for z in self._zones:
                if z.alive and bmin < z.center < bmax:
                    z.alive = False

        # add new high/low candidates from THIS bar
        self._add_or_merge(h, tf, "high")
        self._add_or_merge(lo, tf, "low")

    # ---- external API --------------------------------------------
    def update_if_due(self, now_ts: pd.Timestamp) -> None:
        """Advance per-TF cursors to the most recent bar closed at or before `now_ts`,
        and refresh zones for any TF that flipped since last call.
        """
        flipped: list[str] = []
        for tf in TF_ORDER:
            if tf not in self._tf_np:
                continue
            ts = self._tf_np[tf]["ts"]
            if len(ts) == 0:
                continue
            # find the latest index whose ts <= now_ts (that bar is "closed")
            now_naive = now_ts.tz_convert("UTC").tz_localize(None) if now_ts.tzinfo is not None else now_ts
            pos = int(np.searchsorted(ts, np.datetime64(now_naive), side="right")) - 1
            if pos < 0:
                continue
            if pos != self._tf_cursor[tf]:
                self._tf_cursor[tf] = pos
                last_ts = pd.Timestamp(ts[pos])
                if self._last_ts_by_tf[tf] is not None and last_ts != self._last_ts_by_tf[tf]:
                    flipped.append(tf)
                self._last_ts_by_tf[tf] = last_ts
            else:
                # cursor unchanged — already aligned
                pass

        if not flipped:
            # first call: force a full rebuild so we start with something
            if not self._zones and any(self._tf_cursor[tf] >= 1 for tf in TF_ORDER):
                self._full_rebuild(now_ts)
            return

        # D1 flip triggers full rebuild; otherwise incrementally update
        # affected TFs only (invalidate-on-body-cross + add new bar's
        # high/low). This is much cheaper than a full per-TF rescan and
        # preserves the same "which zones are alive" semantics.
        if "D1" in flipped:
            self._full_rebuild(now_ts)
            return
        for tf in flipped:
            idx = self._tf_cursor[tf]
            if idx >= 1:
                self._incremental_update(tf, idx)
        self._compact()

    def find_nearest(
        self, price: float, max_dist_pips: float, min_weight: float,
        direction: int = 0, ts_ms: int = 0,
    ) -> _KLZone | None:
        if not self._zones:
            return None
        pip_price = self._cfg.instrument.pip_in_points * 0.01
        kl = self._cfg.keylevels
        best: _KLZone | None = None
        best_eff: float = 0.0
        best_d = float("inf")
        for z in self._zones:
            eff_w = (
                effective_weight(
                    z.slice_state, z.weight, direction, ts_ms,
                    kl.slice_cooldown_ms, kl.slice_floor_factor,
                )
                if direction != 0
                else z.weight
            )
            if eff_w < min_weight:
                continue
            d = abs(z.center - price)
            if d < best_d:
                best_d = d
                best = z
                best_eff = eff_w
        if best is None:
            return None
        if (best_d / pip_price) > max_dist_pips:
            return None
        # stash effective weight on the zone object for downstream scoring
        best._effective_weight = best_eff  # type: ignore[attr-defined]
        return best

    def price_in_any(self, price: float, min_weight: float,
                     direction: int = 0, ts_ms: int = 0) -> _KLZone | None:
        kl = self._cfg.keylevels
        for z in self._zones:
            eff_w = (
                effective_weight(
                    z.slice_state, z.weight, direction, ts_ms,
                    kl.slice_cooldown_ms, kl.slice_floor_factor,
                )
                if direction != 0
                else z.weight
            )
            if eff_w < min_weight:
                continue
            if z.center - z.half_width <= price <= z.center + z.half_width:
                z._effective_weight = eff_w  # type: ignore[attr-defined]
                return z
        return None

    def observe_all(self, mid: float, ts_ms: int) -> None:
        """Step slicing state across every live zone on each M1 bar close."""
        kl = self._cfg.keylevels
        commit_half = self._zone_half_width() * kl.slice_commitment_mult
        for z in self._zones:
            if not z.alive:
                continue
            observe_price(
                z.slice_state, z.center, commit_half, mid, ts_ms,
                kl.slice_cooldown_ms, kl.slice_cooldown_ms,
            )

    def count(self) -> int:
        return len(self._zones)


# ======================================================================
# Feature engine (channel + trend + lightweight "burst" proxy)
# ======================================================================


@dataclass
class _FeatureState:
    chan_mid: float = 0.0
    chan_upper: float = 0.0
    chan_lower: float = 0.0
    chan_sigma: float = 0.0
    chan_slope_pips_per_bar: float = 0.0
    chan_pos_norm: float = 0.0
    trend_state: str = "flat"  # "up" | "down" | "flat"
    trend_persistence: float = 0.0
    # Bar-level burst proxy: count same-direction bars in the last 3 M1
    # closes whose per-bar pip move exceeds a sigma threshold.
    burst_up: int = 0
    burst_dn: int = 0


class _Features:
    """Bar-level port of Features.mqh.

    Tick-level velocity/burst logic is replaced with a bar-level analog:
    burst_up = number of last-3 M1 bars with bullish body > sigma*pip_move,
    burst_dn likewise. Uses M1 close-to-close moves; the semantics ("at
    least min_bursts same-direction strong moves recently") are preserved.
    """

    def __init__(self, cfg: FiloConfig) -> None:
        self._cfg = cfg
        self._state = _FeatureState()
        self._last_recompute_idx: int = -1
        self._pip_price = cfg.instrument.pip_in_points * 0.01

    @property
    def state(self) -> _FeatureState:
        return self._state

    def recompute(
        self,
        idx: int,
        m1_close: np.ndarray,
        m1_open: np.ndarray,
        current_mid: float,
    ) -> None:
        """Recompute channel/trend/burst state using bars [..idx] inclusive.

        `idx` is the index of the most recent CLOSED M1 bar.
        """
        N = max(4, int(self._cfg.channel.length))
        if idx + 1 < N + 2:
            return

        # Regression over closes[idx-N+1 .. idx]
        y = m1_close[idx - N + 1 : idx + 1].astype(float)
        x = np.arange(N, dtype=float)  # 0..N-1, N-1 is newest

        # polyfit: slope_pb = price per bar (positive = bullish)
        slope, intercept_at_0 = np.polyfit(x, y, 1)
        pred = slope * x + intercept_at_0
        residuals = y - pred
        sigma = float(np.sqrt(np.mean(residuals * residuals)))
        chan_mid = float(pred[-1])  # model value at most recent bar

        self._state.chan_mid = chan_mid
        self._state.chan_sigma = sigma
        self._state.chan_upper = chan_mid + self._cfg.channel.stdev_mult * sigma
        self._state.chan_lower = chan_mid - self._cfg.channel.stdev_mult * sigma
        self._state.chan_slope_pips_per_bar = float(slope) / self._pip_price
        denom = max(1e-9, sigma * self._cfg.channel.stdev_mult)
        self._state.chan_pos_norm = (current_mid - chan_mid) / denom

        # Trend
        tN = max(4, int(self._cfg.trend.lookback_bars_m1))
        if idx + 1 >= tN + 2:
            o_slice = m1_open[idx - tN + 1 : idx + 1]
            c_slice = m1_close[idx - tN + 1 : idx + 1]
            up_bars = int(np.sum(c_slice > o_slice))
            dn_bars = int(np.sum(c_slice < o_slice))
            ratio_up = up_bars / tN
            ratio_dn = dn_bars / tN
            pmin = self._cfg.trend.persistence_min_ratio
            slope_pips = self._state.chan_slope_pips_per_bar
            if slope_pips > 0.05 and ratio_up >= pmin:
                self._state.trend_state = "up"
                self._state.trend_persistence = ratio_up
            elif slope_pips < -0.05 and ratio_dn >= pmin:
                self._state.trend_state = "down"
                self._state.trend_persistence = ratio_dn
            else:
                self._state.trend_state = "flat"
                self._state.trend_persistence = max(ratio_up, ratio_dn)

        # Burst proxy: last 3 closed M1 bars, a "burst" bar is one whose
        # body exceeds burst_sigma * rolling_stddev(body) over last 20 bars.
        win_lo = max(0, idx - 19)
        body_window = np.abs(m1_close[win_lo : idx + 1] - m1_open[win_lo : idx + 1])
        if len(body_window) >= 5:
            body_sigma = float(np.std(body_window))
        else:
            body_sigma = 0.0
        threshold = max(
            self._cfg.acceleration.burst_sigma * body_sigma,
            0.5 * self._pip_price,  # floor: half a pip
        )
        burst_up = 0
        burst_dn = 0
        look = min(3, idx + 1)
        for k in range(look):
            j = idx - k
            diff = m1_close[j] - m1_open[j]
            if diff >= threshold:
                burst_up += 1
            elif diff <= -threshold:
                burst_dn += 1
        self._state.burst_up = burst_up
        self._state.burst_dn = burst_dn
        self._last_recompute_idx = idx

    def has_min_bursts(self, direction: int) -> bool:
        need = self._cfg.acceleration.min_bursts
        if direction > 0:
            return self._state.burst_up >= need
        return self._state.burst_dn >= need

    def at_channel_extreme(self, direction: int) -> bool:
        p = self._state.chan_pos_norm
        m = self._cfg.channel.extreme_block_mult
        if direction > 0:
            return p >= m
        return p <= -m


# ======================================================================
# Correlation
# ======================================================================


class _Correlation:
    """Bar-level port of Correlation.mqh.

    Holds per-symbol M1 close arrays (aligned to the primary M1 timeline)
    and computes signed-return agreement over the last N seconds (converted
    to bars at M1 = 60s).
    """

    def __init__(
        self,
        cfg: FiloConfig,
        primary_close: np.ndarray,
        xag_close: np.ndarray | None,
        dxy_close: np.ndarray | None,
    ) -> None:
        self._cfg = cfg
        self._xau = primary_close
        self._xag = xag_close
        self._dxy = dxy_close
        self._lookback_bars = max(
            1, cfg.correlation.lookback_seconds // 60 if cfg.correlation.lookback_seconds >= 60 else 1
        )
        self._min_agreement = cfg.correlation.min_agreement

    @staticmethod
    def _agreement(ra: float, rb: float) -> float:
        if abs(ra) < 1e-8 or abs(rb) < 1e-8:
            return 0.0
        sign = 1.0 if ra * rb >= 0.0 else -1.0
        denom = max(abs(ra), abs(rb))
        mag = min(1.0, min(abs(ra), abs(rb)) / denom) if denom > 0 else 0.0
        return sign * mag

    def _ret(self, arr: np.ndarray | None, idx: int) -> float:
        if arr is None:
            return 0.0
        past_idx = idx - self._lookback_bars
        if past_idx < 0 or idx >= len(arr):
            return 0.0
        past = float(arr[past_idx])
        if abs(past) < 1e-12:
            return 0.0
        return (float(arr[idx]) - past) / past

    def directional_ok(self, idx: int, direction: int) -> tuple[bool, str]:
        r_xau = self._ret(self._xau, idx)
        ag_xag = 0.0
        ag_dxy = 0.0
        block = self._min_agreement * 0.5

        if self._cfg.correlation.enable_xagusd and self._xag is not None:
            r_xag = self._ret(self._xag, idx)
            ag_xag = self._agreement(r_xau, r_xag)
            if ag_xag < -block:
                return False, "xagusd disagreeing"

        if self._cfg.correlation.enable_dxy and self._dxy is not None:
            r_dxy = self._ret(self._dxy, idx)
            if self._cfg.correlation.dxy_inverse:
                r_dxy = -r_dxy
            ag_dxy = self._agreement(r_xau, r_dxy)
            if ag_dxy < -block:
                return False, "dxy agreeing with xau (inverse expected)"

        return True, ""


# ======================================================================
# Setup detectors
# ======================================================================


@dataclass
class _Candidate:
    family: str
    direction: int  # +1 long, -1 short
    zone: _KLZone | None
    kl_distance_pips: float
    intrinsic_score: float
    block_reason: str
    accepted: bool


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class _SetupEngine:
    """Faithful port of Setups.mqh (7 families + common gates).

    Ports MQL5 tick-level semantics to bar-level by using the latest CLOSED
    M1 bar as "bar shift 1" and the current M1 bar (partially known via
    OHLC) for the reference price / mid used in detection.
    """

    def __init__(
        self,
        cfg: FiloConfig,
        kl: _KeyLevels,
        features: _Features,
        correlation: _Correlation,
    ) -> None:
        self._cfg = cfg
        self._kl = kl
        self._features = features
        self._correlation = correlation
        self._pip_price = cfg.instrument.pip_in_points * 0.01
        self._cooldown: dict[str, int] = dict.fromkeys(SETUP_FAMILIES, -10**9)

    def _pip_from_price(self, p: float) -> float:
        return p / self._pip_price

    def _price_from_pips(self, p: float) -> float:
        return p * self._pip_price

    # ---- family detectors (each returns Candidate or None) ------

    def _detect_breakout(self, mid: float, spread_pips: float) -> _Candidate | None:
        if not self._cfg.setups.enable_breakout:
            return None
        zone = self._kl.find_nearest(
            mid, self._cfg.keylevels.max_setup_distance_pips, self._cfg.keylevels.min_kl_weight
        )
        if zone is None:
            return None
        break_tol = self._price_from_pips(self._cfg.execution.trigger_tolerance_pips)
        top = zone.center + zone.half_width
        bot = zone.center - zone.half_width
        direction = 0
        if mid > top + break_tol and self._features.has_min_bursts(1):
            direction = 1
        elif mid < bot - break_tol and self._features.has_min_bursts(-1):
            direction = -1
        else:
            return None
        # score: 0.4 + 0.5*weight + 0.3*vscore - spread penalty
        # bar-level vscore proxy: channel slope pips/bar scaled
        slope_pb = self._features.state.chan_slope_pips_per_bar
        vscore = _clamp(abs(slope_pb) / 2.0, 0.0, 1.0)
        penalty = _clamp(spread_pips / 1.0, 0.0, 0.5)
        score = _clamp(0.4 + 0.5 * zone.weight + 0.3 * vscore - penalty, 0.0, 1.0)
        return _Candidate(
            family="breakout",
            direction=direction,
            zone=zone,
            kl_distance_pips=self._pip_from_price(abs(mid - zone.center)),
            intrinsic_score=score,
            block_reason="",
            accepted=False,
        )

    def _detect_rejection(
        self,
        mid: float,
        prev_o: float,
        prev_c: float,
        prev_h: float,
        prev_l: float,
    ) -> _Candidate | None:
        if not self._cfg.setups.enable_rejection:
            return None
        zone = self._kl.find_nearest(
            mid, self._cfg.keylevels.max_setup_distance_pips, self._cfg.keylevels.min_kl_weight
        )
        if zone is None:
            return None
        body = abs(prev_c - prev_o)
        top = zone.center + zone.half_width
        bot = zone.center - zone.half_width

        wick_top = (
            prev_h >= bot
            and max(prev_o, prev_c) < zone.center
            and (prev_h - max(prev_o, prev_c)) > body * 1.2
        )
        wick_bot = (
            prev_l <= top
            and min(prev_o, prev_c) > zone.center
            and (min(prev_o, prev_c) - prev_l) > body * 1.2
        )

        direction = 0
        if wick_top and self._features.has_min_bursts(-1):
            direction = -1
        elif wick_bot and self._features.has_min_bursts(1):
            direction = 1
        else:
            return None
        score = _clamp(0.35 + 0.55 * zone.weight, 0.0, 1.0)
        return _Candidate(
            family="rejection",
            direction=direction,
            zone=zone,
            kl_distance_pips=self._pip_from_price(abs(mid - zone.center)),
            intrinsic_score=score,
            block_reason="",
            accepted=False,
        )

    def _detect_fakeout(
        self,
        mid: float,
        m1_open: np.ndarray,
        m1_close: np.ndarray,
        m1_high: np.ndarray,
        m1_low: np.ndarray,
        idx: int,
    ) -> _Candidate | None:
        if not self._cfg.setups.enable_fakeout:
            return None
        zone = self._kl.find_nearest(
            mid, self._cfg.keylevels.max_setup_distance_pips, self._cfg.keylevels.min_kl_weight
        )
        if zone is None:
            return None
        top = zone.center + zone.half_width
        bot = zone.center - zone.half_width
        tol = self._price_from_pips(self._cfg.execution.trigger_tolerance_pips)
        # look at last 3 closed M1 bars
        for k in range(1, 4):
            j = idx - (k - 1)  # idx is most-recent closed bar
            if j < 0:
                break
            h = float(m1_high[j])
            lo = float(m1_low[j])
            c = float(m1_close[j])
            if h > top + tol and c < zone.center and self._features.has_min_bursts(-1):
                score = _clamp(0.30 + 0.60 * zone.weight, 0.0, 1.0)
                return _Candidate(
                    family="fakeout",
                    direction=-1,
                    zone=zone,
                    kl_distance_pips=self._pip_from_price(abs(mid - zone.center)),
                    intrinsic_score=score,
                    block_reason="",
                    accepted=False,
                )
            if lo < bot - tol and c > zone.center and self._features.has_min_bursts(1):
                score = _clamp(0.30 + 0.60 * zone.weight, 0.0, 1.0)
                return _Candidate(
                    family="fakeout",
                    direction=1,
                    zone=zone,
                    kl_distance_pips=self._pip_from_price(abs(mid - zone.center)),
                    intrinsic_score=score,
                    block_reason="",
                    accepted=False,
                )
        return None

    def _detect_retest(self, mid: float) -> _Candidate | None:
        if not self._cfg.setups.enable_retest:
            return None
        zone = self._kl.price_in_any(mid, self._cfg.keylevels.min_kl_weight)
        if zone is None:
            return None
        slope = self._features.state.chan_slope_pips_per_bar
        direction = 0
        if slope > 0.05 and self._features.has_min_bursts(1):
            direction = 1
        elif slope < -0.05 and self._features.has_min_bursts(-1):
            direction = -1
        else:
            return None
        score = _clamp(0.35 + 0.55 * zone.weight, 0.0, 1.0)
        return _Candidate(
            family="retest",
            direction=direction,
            zone=zone,
            kl_distance_pips=self._pip_from_price(abs(mid - zone.center)),
            intrinsic_score=score,
            block_reason="",
            accepted=False,
        )

    def _detect_failed_auction(
        self,
        mid: float,
        prev_h: float,
        prev_l: float,
        prev_c: float,
    ) -> _Candidate | None:
        if not self._cfg.setups.enable_failed_auction:
            return None
        fs = self._features.state
        if fs.chan_sigma <= 0:
            return None

        direction = 0
        if prev_h > fs.chan_upper and prev_c < fs.chan_upper - 0.3 * fs.chan_sigma:
            if self._features.has_min_bursts(-1):
                direction = -1
        elif prev_l < fs.chan_lower and prev_c > fs.chan_lower + 0.3 * fs.chan_sigma:
            if self._features.has_min_bursts(1):
                direction = 1
        if direction == 0:
            return None
        zone = self._kl.find_nearest(
            mid, self._cfg.keylevels.max_setup_distance_pips, self._cfg.keylevels.min_kl_weight
        )
        weight = zone.weight if zone is not None else 0.0
        score = _clamp(0.45 + 0.35 * weight, 0.0, 1.0)
        return _Candidate(
            family="failed_auction",
            direction=direction,
            zone=zone,
            kl_distance_pips=(
                self._pip_from_price(abs(mid - zone.center)) if zone is not None else 0.0
            ),
            intrinsic_score=score,
            block_reason="",
            accepted=False,
        )

    def _detect_engulfing_accel(
        self,
        mid: float,
        m1_open: np.ndarray,
        m1_close: np.ndarray,
        idx: int,
    ) -> _Candidate | None:
        if not self._cfg.setups.enable_engulfing_accel:
            return None
        if idx < 1:
            return None
        o1 = float(m1_open[idx])
        c1 = float(m1_close[idx])
        o2 = float(m1_open[idx - 1])
        c2 = float(m1_close[idx - 1])
        bull = c2 < o2 and c1 > o1 and c1 >= o2 and o1 <= c2
        bear = c2 > o2 and c1 < o1 and c1 <= o2 and o1 >= c2
        zone = self._kl.find_nearest(
            mid, self._cfg.keylevels.max_setup_distance_pips, self._cfg.keylevels.min_kl_weight
        )
        if zone is None:
            return None
        direction = 0
        if bull and self._features.has_min_bursts(1):
            direction = 1
        elif bear and self._features.has_min_bursts(-1):
            direction = -1
        else:
            return None
        score = _clamp(0.40 + 0.50 * zone.weight, 0.0, 1.0)
        return _Candidate(
            family="engulfing_accel",
            direction=direction,
            zone=zone,
            kl_distance_pips=self._pip_from_price(abs(mid - zone.center)),
            intrinsic_score=score,
            block_reason="",
            accepted=False,
        )

    def _detect_mean_reversion(self, mid: float) -> _Candidate | None:
        if not self._cfg.setups.enable_mean_reversion:
            return None
        p = self._features.state.chan_pos_norm
        direction = 0
        if p >= 0.95 and self._features.has_min_bursts(-1):
            direction = -1
        elif p <= -0.95 and self._features.has_min_bursts(1):
            direction = 1
        if direction == 0:
            return None
        zone = self._kl.find_nearest(
            mid, self._cfg.keylevels.max_setup_distance_pips, self._cfg.keylevels.min_kl_weight
        )
        weight = zone.weight if zone is not None else 0.0
        score = _clamp(0.40 + 0.35 * weight, 0.0, 1.0)
        return _Candidate(
            family="mean_reversion",
            direction=direction,
            zone=zone,
            kl_distance_pips=(
                self._pip_from_price(abs(mid - zone.center)) if zone is not None else 0.0
            ),
            intrinsic_score=score,
            block_reason="",
            accepted=False,
        )

    # ---- common gates --------------------------------------------
    def _common_gates(
        self, cand: _Candidate, spread_pips: float, bar_idx: int,
        m1_open: np.ndarray, m1_close: np.ndarray, ts_ms: int,
    ) -> bool:
        # spread gate by family class
        cfg = self._cfg
        if cand.family in ("breakout", "retest"):
            max_spread = cfg.execution.max_spread_breakout
        elif cand.family in _REVERSION:
            max_spread = cfg.execution.max_spread_reversion
        else:
            max_spread = cfg.execution.max_spread_default

        if spread_pips > max_spread:
            cand.block_reason = f"spread {spread_pips:.2f} > {max_spread:.2f}"
            return False

        if cand.zone is None:
            cand.block_reason = "no kl zone"
            return False
        eff_w = effective_weight(
            cand.zone.slice_state, cand.zone.weight, cand.direction, ts_ms,
            cfg.keylevels.slice_cooldown_ms, cfg.keylevels.slice_floor_factor,
        )
        cand.kl_effective_weight = eff_w  # type: ignore[attr-defined]
        if eff_w < cfg.keylevels.min_kl_weight:
            cand.block_reason = (
                f"kl eff-weight {eff_w:.2f} < min {cfg.keylevels.min_kl_weight:.2f} (sliced)"
            )
            return False

        if cand.family in _CONTINUATION and self._features.at_channel_extreme(cand.direction):
            cand.block_reason = "at channel extreme (continuation blocked)"
            return False

        # correlation
        ok, reason = self._correlation.directional_ok(bar_idx, cand.direction)
        if not ok:
            cand.block_reason = reason
            return False

        # exhaustion
        bars = 0
        pips_moved = 0.0
        for k in range(1, cfg.setups.exhaustion_bars + 1):
            j = bar_idx - (k - 1)
            if j < 0:
                break
            o = float(m1_open[j])
            c = float(m1_close[j])
            if cand.direction > 0 and c > o:
                bars += 1
            elif cand.direction < 0 and c < o:
                bars += 1
            else:
                break
            pips_moved += self._pip_from_price(abs(c - o))
        if (
            bars >= cfg.setups.exhaustion_bars or pips_moved >= cfg.setups.exhaustion_pips
        ) and cand.family in _CONTINUATION:
            cand.block_reason = f"exhaustion {bars} bars / {pips_moved:.1f} pips"
            return False

        return True

    # ---- main entry point -----------------------------------------
    def scan(
        self,
        bar_idx: int,
        now_ts: pd.Timestamp,
        mid: float,
        spread_pips: float,
        m1_open: np.ndarray,
        m1_high: np.ndarray,
        m1_low: np.ndarray,
        m1_close: np.ndarray,
    ) -> tuple[list[_Candidate], _Candidate | None]:
        """Run all 7 families. Returns (all_candidates, first_accepted)."""

        if self._kl.count() == 0:
            return [], None

        # "Previous closed bar" in MQL5 is shift=1. In our bar-level replay,
        # the most recent CLOSED bar is bar_idx (we process post-close).
        prev_idx = bar_idx
        if prev_idx < 1:
            return [], None
        prev_o = float(m1_open[prev_idx])
        prev_c = float(m1_close[prev_idx])
        prev_h = float(m1_high[prev_idx])
        prev_l = float(m1_low[prev_idx])

        candidates: list[_Candidate] = []
        accepted: _Candidate | None = None

        detectors: list[tuple[str, callable]] = [
            ("breakout", lambda: self._detect_breakout(mid, spread_pips)),
            ("retest", lambda: self._detect_retest(mid)),
            ("rejection", lambda: self._detect_rejection(mid, prev_o, prev_c, prev_h, prev_l)),
            ("fakeout", lambda: self._detect_fakeout(mid, m1_open, m1_close, m1_high, m1_low, prev_idx)),
            ("failed_auction", lambda: self._detect_failed_auction(mid, prev_h, prev_l, prev_c)),
            (
                "engulfing_accel",
                lambda: self._detect_engulfing_accel(mid, m1_open, m1_close, prev_idx),
            ),
            ("mean_reversion", lambda: self._detect_mean_reversion(mid)),
        ]

        ts_ms_now = int(pd.Timestamp(now_ts).value // 1_000_000)
        for fam, det in detectors:
            if bar_idx - self._cooldown[fam] < _FAMILY_COOLDOWN_BARS:
                continue
            cand = det()
            if cand is None:
                continue

            if not self._common_gates(
                cand, spread_pips, prev_idx, m1_open, m1_close, ts_ms_now
            ):
                candidates.append(cand)
                continue

            if cand.intrinsic_score < self._cfg.setups.min_setup_score:
                cand.block_reason = (
                    f"score {cand.intrinsic_score:.2f} < {self._cfg.setups.min_setup_score:.2f}"
                )
                candidates.append(cand)
                continue

            cand.accepted = True
            candidates.append(cand)
            if accepted is None:
                accepted = cand
                self._cooldown[fam] = bar_idx

        return candidates, accepted


# ======================================================================
# Trade management
# ======================================================================


@dataclass
class _ManagedTrade:
    family: str
    direction: int
    entry_price: float
    sl: float
    tp: float
    initial_sl: float
    initial_tp: float
    opened_ts: pd.Timestamp
    at_be: bool = False
    trailing: bool = False
    kl_zone_id: str = ""
    kl_weight: float = 0.0
    intrinsic_score: float = 0.0
    entry_spread_pips: float = 0.0


class _TradeBook:
    """Mirrors TradeManagement.mqh semantics — hidden SL/TP, BE, trailing."""

    def __init__(self, cfg: FiloConfig) -> None:
        self._cfg = cfg
        self._open: list[_ManagedTrade] = []
        self._pip_price = cfg.instrument.pip_in_points * 0.01
        self.closed: list[dict] = []

    @property
    def open_count(self) -> int:
        return len(self._open)

    def can_open(self, direction: int) -> bool:
        if len(self._open) >= self._cfg.sizing.max_concurrent_trades:
            return False
        for t in self._open:
            if t.direction == direction and not t.at_be:
                return False
        return True

    def _stop_target_for(self, family: str) -> tuple[float, float]:
        tm = self._cfg.trade_management
        if family == "breakout" or family == "retest":
            return tm.stop_breakout_pips, tm.target_breakout_pips
        if family == "rejection":
            return tm.stop_rejection_pips, tm.target_rejection_pips
        # reversion family (mean_reversion, failed_auction, fakeout, engulfing_accel)
        return tm.stop_reversion_pips, tm.target_reversion_pips

    def open(
        self,
        cand: _Candidate,
        entry_price: float,
        ts: pd.Timestamp,
        spread_pips: float,
    ) -> None:
        stop_pips, target_pips = self._stop_target_for(cand.family)
        if cand.direction > 0:
            sl = entry_price - stop_pips * self._pip_price
            tp = entry_price + target_pips * self._pip_price
        else:
            sl = entry_price + stop_pips * self._pip_price
            tp = entry_price - target_pips * self._pip_price
        zone = cand.zone
        self._open.append(
            _ManagedTrade(
                family=cand.family,
                direction=cand.direction,
                entry_price=entry_price,
                sl=sl,
                tp=tp,
                initial_sl=sl,
                initial_tp=tp,
                opened_ts=ts,
                at_be=False,
                trailing=False,
                kl_zone_id=zone.zone_id if zone is not None else "",
                kl_weight=zone.weight if zone is not None else 0.0,
                intrinsic_score=cand.intrinsic_score,
                entry_spread_pips=spread_pips,
            )
        )

    def step_intrabar(
        self,
        bar_ts: pd.Timestamp,
        o: float,
        h: float,
        l: float,
        c: float,
    ) -> None:
        """Walk price through canonical intrabar path, updating SL/BE/trail
        and closing trades that touch hidden SL or TP.
        """
        if not self._open:
            return

        # canonical path: up-bar goes O-L-H-C; down-bar goes O-H-L-C
        up_bar = c >= o
        path = [o, l, h, c] if up_bar else [o, h, l, c]

        tm = self._cfg.trade_management
        be_trigger = tm.be_trigger_pips * self._pip_price
        be_buffer = tm.be_buffer_pips * self._pip_price
        trail_start = tm.trail_start_pips * self._pip_price
        trail_dist = tm.trail_distance_pips * self._pip_price

        survivors: list[_ManagedTrade] = []
        for t in self._open:
            closed = False
            close_price = 0.0
            close_reason = ""

            for px in path:
                # in-bar price level
                if t.direction > 0:
                    gain = px - t.entry_price
                    if px <= t.sl:
                        close_price = t.sl
                        close_reason = "sl" if not t.at_be and not t.trailing else (
                            "be" if t.at_be and not t.trailing else "trail"
                        )
                        closed = True
                        break
                    if px >= t.tp:
                        close_price = t.tp
                        close_reason = "tp"
                        closed = True
                        break
                    # BE and trail move the SL upwards
                    if not t.at_be and gain >= be_trigger:
                        t.sl = max(t.sl, t.entry_price + be_buffer)
                        t.at_be = True
                    if gain >= trail_start:
                        new_sl = px - trail_dist
                        if new_sl > t.sl:
                            t.sl = new_sl
                            t.trailing = True
                else:
                    gain = t.entry_price - px
                    if px >= t.sl:
                        close_price = t.sl
                        close_reason = "sl" if not t.at_be and not t.trailing else (
                            "be" if t.at_be and not t.trailing else "trail"
                        )
                        closed = True
                        break
                    if px <= t.tp:
                        close_price = t.tp
                        close_reason = "tp"
                        closed = True
                        break
                    if not t.at_be and gain >= be_trigger:
                        t.sl = min(t.sl, t.entry_price - be_buffer)
                        t.at_be = True
                    if gain >= trail_start:
                        new_sl = px + trail_dist
                        if new_sl < t.sl:
                            t.sl = new_sl
                            t.trailing = True

            if closed:
                pip_move = (close_price - t.entry_price) / self._pip_price
                if t.direction < 0:
                    pip_move = -pip_move
                gross_usd = pips_to_usd(pip_move, self._cfg.sizing.fixed_lot, self._cfg)
                commission = (
                    self._cfg.sizing.fixed_lot
                    * self._cfg.instrument.commission_usd_per_lot_round_turn
                )
                usd = gross_usd - commission
                self.closed.append(
                    {
                        "ts_open": t.opened_ts,
                        "ts_close": bar_ts,
                        "family": t.family,
                        "direction": "long" if t.direction > 0 else "short",
                        "entry": t.entry_price,
                        "exit": close_price,
                        "sl": t.initial_sl,
                        "tp": t.initial_tp,
                        "pips": pip_move,
                        "gross_usd": gross_usd,
                        "commission_usd": commission,
                        "usd": usd,
                        "kl_zone_id": t.kl_zone_id,
                        "kl_weight": t.kl_weight,
                        "intrinsic_score": t.intrinsic_score,
                        "close_reason": close_reason,
                        "entry_spread_pips": t.entry_spread_pips,
                    }
                )
            else:
                survivors.append(t)
        self._open = survivors

    def force_close_all(self, final_ts: pd.Timestamp, final_price: float) -> None:
        """Close anything still open at the final bar — mark-to-market."""
        for t in self._open:
            pip_move = (final_price - t.entry_price) / self._pip_price
            if t.direction < 0:
                pip_move = -pip_move
            gross_usd = pips_to_usd(pip_move, self._cfg.sizing.fixed_lot, self._cfg)
            commission = (
                self._cfg.sizing.fixed_lot
                * self._cfg.instrument.commission_usd_per_lot_round_turn
            )
            usd = gross_usd - commission
            self.closed.append(
                {
                    "ts_open": t.opened_ts,
                    "ts_close": final_ts,
                    "family": t.family,
                    "direction": "long" if t.direction > 0 else "short",
                    "entry": t.entry_price,
                    "exit": final_price,
                    "sl": t.initial_sl,
                    "tp": t.initial_tp,
                    "pips": pip_move,
                    "gross_usd": gross_usd,
                    "commission_usd": commission,
                    "usd": usd,
                    "kl_zone_id": t.kl_zone_id,
                    "kl_weight": t.kl_weight,
                    "intrinsic_score": t.intrinsic_score,
                    "close_reason": "eod",
                    "entry_spread_pips": t.entry_spread_pips,
                }
            )
        self._open = []


# ======================================================================
# Public entry point
# ======================================================================


def _classify_session(hour: int) -> str:
    """Very rough London/NY/Asia breakdown on an exchange clock (UTC+0)."""
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 16:
        return "overlap_ln_ny"
    if 16 <= hour < 21:
        return "ny"
    return "offhours"


def run_backtest(
    bars_by_tf: dict[str, pd.DataFrame],
    cfg: FiloConfig,
    corr_bars: dict[str, pd.DataFrame] | None = None,
) -> BacktestResult:
    """Run the offline bar-level replay.

    Parameters
    ----------
    bars_by_tf
        Dict with at least key "M1" (primary symbol, the driving timeline).
        Optional higher TFs "D1","H4","H1","M30","M15","M5" used by the KL
        engine. Missing higher TFs are simply not used — the engine still
        runs but produces fewer zones.
    cfg
        Loaded FiloConfig.
    corr_bars
        Optional dict with keys matching `cfg.symbols.silver_proxy` and/or
        `cfg.symbols.dxy_proxy`, each mapped to an M1 bars DataFrame aligned
        (by ts) to the primary M1 frame. Missing entries = neutral correlation.
    """
    if "M1" not in bars_by_tf:
        raise ValueError("bars_by_tf must include 'M1'")
    m1 = bars_by_tf["M1"].sort_values("ts").reset_index(drop=True)
    if len(m1) == 0:
        return BacktestResult(
            trades=pd.DataFrame(),
            candidates=pd.DataFrame(),
            equity_curve=pd.DataFrame(columns=["ts", "equity_usd", "drawdown_pct"]),
            metrics={},
        )

    ts_arr = m1["ts"].to_numpy()
    open_arr = m1["open"].to_numpy(dtype=float)
    high_arr = m1["high"].to_numpy(dtype=float)
    low_arr = m1["low"].to_numpy(dtype=float)
    close_arr = m1["close"].to_numpy(dtype=float)
    spread_arr = (
        m1["spread"].to_numpy(dtype=float) if "spread" in m1.columns else np.zeros(len(m1))
    )
    pip_in_points = cfg.instrument.pip_in_points
    pip_price = pip_in_points * 0.01

    kl = _KeyLevels(cfg)
    for tf in TF_ORDER:
        if tf in bars_by_tf:
            kl.attach_tf(tf, bars_by_tf[tf].sort_values("ts").reset_index(drop=True))

    features = _Features(cfg)

    # correlation frames aligned to primary M1 timeline
    def _align(arr: pd.DataFrame | None) -> np.ndarray | None:
        if arr is None or len(arr) == 0:
            return None
        merged = pd.merge_asof(
            m1[["ts"]],
            arr[["ts", "close"]].sort_values("ts"),
            on="ts",
            direction="backward",
        )
        return merged["close"].to_numpy(dtype=float)

    xag = None
    dxy = None
    if corr_bars:
        xag_df = corr_bars.get(cfg.symbols.silver_proxy)
        dxy_df = corr_bars.get(cfg.symbols.dxy_proxy)
        xag = _align(xag_df) if xag_df is not None else None
        dxy = _align(dxy_df) if dxy_df is not None else None

    correlation = _Correlation(cfg, close_arr, xag, dxy)
    setups = _SetupEngine(cfg, kl, features, correlation)
    book = _TradeBook(cfg)

    # -----------------------------------------------------------------
    # Warmup — we need at least `channel.length` closed M1 bars before the
    # regression channel becomes meaningful. Start the active loop once the
    # channel can be computed.
    warmup = max(int(cfg.channel.length) + 5, int(cfg.trend.lookback_bars_m1) + 5)

    # candidates log
    cand_rows: list[dict] = []

    n = len(m1)
    for i in range(n):
        ts_i = pd.Timestamp(ts_arr[i])
        o_i = float(open_arr[i])
        h_i = float(high_arr[i])
        l_i = float(low_arr[i])
        c_i = float(close_arr[i])
        spread_points = float(spread_arr[i])
        spread_pips = spread_points / pip_in_points if pip_in_points > 0 else 0.0

        # 1) manage open trades through THIS bar's intrabar path
        book.step_intrabar(ts_i, o_i, h_i, l_i, c_i)

        if i < warmup:
            continue

        # 2) update KL engine (only recalculates on TF flip)
        kl.update_if_due(ts_i)
        # step slicing-penalty state across every live zone
        ts_ms_i = int(pd.Timestamp(ts_i).value // 1_000_000)
        kl.observe_all(c_i, ts_ms_i)

        # 3) recompute features using closes up to THIS bar
        features.recompute(i, close_arr, open_arr, c_i)

        # 4) scan setups — current mid = close of bar just observed
        all_cands, accepted = setups.scan(
            i, ts_i, c_i, spread_pips, open_arr, high_arr, low_arr, close_arr
        )

        # log every candidate with block_reason
        for cand in all_cands:
            cand_rows.append(
                {
                    "ts": ts_i,
                    "family": cand.family,
                    "direction": "long" if cand.direction > 0 else "short",
                    "kl_zone_id": cand.zone.zone_id if cand.zone is not None else "",
                    "kl_weight": cand.zone.weight if cand.zone is not None else 0.0,
                    "kl_effective_weight": getattr(cand, "kl_effective_weight", cand.zone.weight if cand.zone is not None else 0.0),
                    "kl_distance_pips": cand.kl_distance_pips,
                    "intrinsic_score": cand.intrinsic_score,
                    "spread_pips": spread_pips,
                    "chan_pos_norm": features.state.chan_pos_norm,
                    "trend_state": features.state.trend_state,
                    "accepted": cand.accepted,
                    "block_reason": cand.block_reason,
                }
            )

        # 5) execute the first accepted candidate (if any) at the NEXT bar's open,
        #    to avoid look-ahead. Equivalent in our loop: open now at current close,
        #    but with spread added to the entry side.
        if accepted is not None and book.can_open(accepted.direction):
            half_spread_price = 0.5 * spread_pips * pip_price
            zone_anchor = (
                cfg.execution.reversal_anchor == "zone"
                and accepted.family in _REVERSAL_ENTRY
                and accepted.zone is not None
            )
            fill_price: float | None
            if zone_anchor:
                if accepted.direction > 0:
                    edge = accepted.zone.center - accepted.zone.half_width
                    fill_price = edge + half_spread_price
                else:
                    edge = accepted.zone.center + accepted.zone.half_width
                    fill_price = edge - half_spread_price
                dist_pips = abs(c_i - edge) / pip_price if pip_price > 0 else 0.0
                if dist_pips > cfg.execution.reversal_zone_max_dist_pips:
                    fill_price = None  # too far from zone — drop the fill
            else:
                if accepted.direction > 0:
                    fill_price = c_i + half_spread_price  # ask for long
                else:
                    fill_price = c_i - half_spread_price  # bid for short
            if fill_price is not None:
                book.open(accepted, fill_price, ts_i, spread_pips)

    # close whatever is still open on the final bar
    if n > 0:
        book.force_close_all(pd.Timestamp(ts_arr[-1]), float(close_arr[-1]))

    # -----------------------------------------------------------------
    # Assemble outputs
    trades_df = pd.DataFrame(book.closed)
    if len(trades_df) > 0:
        trades_df["ts_open"] = pd.to_datetime(trades_df["ts_open"])
        trades_df["ts_close"] = pd.to_datetime(trades_df["ts_close"])
        trades_df["hour"] = trades_df["ts_open"].dt.hour
        trades_df["day_of_week"] = trades_df["ts_open"].dt.dayofweek
        trades_df["session"] = trades_df["hour"].map(_classify_session)
    else:
        trades_df = pd.DataFrame(
            columns=[
                "ts_open",
                "ts_close",
                "family",
                "direction",
                "entry",
                "exit",
                "sl",
                "tp",
                "pips",
                "usd",
                "kl_zone_id",
                "kl_weight",
                "intrinsic_score",
                "session",
                "hour",
                "day_of_week",
                "close_reason",
                "entry_spread_pips",
            ]
        )

    candidates_df = pd.DataFrame(cand_rows)

    # Equity curve — import lazily to keep the module independent.
    from filo_hft.backtest.metrics import compute_metrics, equity_curve

    eq_df = equity_curve(trades_df)
    metrics = compute_metrics(trades_df, cfg)

    return BacktestResult(
        trades=trades_df,
        candidates=candidates_df,
        equity_curve=eq_df,
        metrics=metrics,
    )
