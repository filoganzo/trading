"""TMLab shadow-variant evaluator.

Mirrors the live TMLab stream (`telemetry/tm_lab.csv`): for each accepted
base trade, replay the same entry forward under N alternate trade-management
rules and record the alternative PnL. No resimulation of setup detection —
the signal is the same, only the exit is different.

Each variant applies its own (stop, target, break-even trigger, trailing
rule) to the same entry price and walks forward through the same M1 OHLC
path until the variant-specific SL/TP/trail is hit.

Commissions are applied at close using the same rate as base trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from filo_hft.config import FiloConfig, pips_to_usd


# ======================================================================
# Variant definitions
# ======================================================================


@dataclass
class TMVariant:
    """A single shadow trade-management profile.

    stop_pips/target_pips : absolute pip stop/target. If atr_stop_mult > 0,
        the stop is computed as ATR(window) * mult at entry time, overriding
        stop_pips.
    be_trigger_pips       : once trade is this far in profit, pull SL to
        entry + be_buffer_pips (0 disables).
    trail_start_pips      : once trade is this far in profit, enable trailing
        SL at trail_distance_pips (0 disables).
    adverse_cut_pips      : cut the trade if it moves this far adverse on a
        M1 close (not on intrabar wicks). 0 disables.
    """

    name: str
    stop_pips: float = 20.0
    target_pips: float = 30.0
    be_trigger_pips: float = 6.0
    be_buffer_pips: float = 1.0
    trail_start_pips: float = 8.0
    trail_distance_pips: float = 3.0
    atr_stop_mult: float = 0.0
    atr_window: int = 14
    adverse_cut_pips: float = 0.0
    max_bars_hold: int = 0  # 0 = disabled; >0 = force-close after N M1 bars


def default_variants() -> list[TMVariant]:
    """~70 variants covering the full axes TMLab explores in live."""
    out: list[TMVariant] = []
    out.append(TMVariant(name="base"))

    # ---- fixed-stop sweep -------------------------------------------
    for sp in (10, 12, 14, 16, 18, 22, 25, 28, 32, 40):
        out.append(TMVariant(name=f"stop_{sp}", stop_pips=float(sp)))

    # ---- fixed-target sweep -----------------------------------------
    for tp in (12, 15, 18, 22, 25, 35, 40, 50, 60, 80, 100):
        out.append(TMVariant(name=f"target_{tp}", target_pips=float(tp)))

    # ---- BE trigger sweep -------------------------------------------
    for be in (3, 4, 5, 8, 10, 12, 15):
        out.append(TMVariant(name=f"be_trig_{be}", be_trigger_pips=float(be)))

    # ---- trail distance sweep ---------------------------------------
    for td in (1, 2, 4, 5, 6, 8):
        out.append(TMVariant(
            name=f"tdist_{str(td).replace('.', '_')}",
            trail_distance_pips=float(td)))

    # ---- trail start sweep ------------------------------------------
    for ts in (4, 6, 10, 12, 15, 20):
        out.append(TMVariant(name=f"tstart_{ts}", trail_start_pips=float(ts)))

    # ---- ATR-scaled stops -------------------------------------------
    for mult in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0):
        out.append(TMVariant(
            name=f"atr_stop_{str(mult).replace('.', '_')}",
            atr_stop_mult=mult,
        ))

    # ---- time-exit sweep (max bars held) ----------------------------
    for mb in (2, 3, 5, 8, 15, 30, 60, 120):
        out.append(TMVariant(name=f"time_exit_{mb}", max_bars_hold=mb))

    # ---- asymmetric R profiles (stop = base, target = R * stop) -----
    for r in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        out.append(TMVariant(
            name=f"R_{str(r).replace('.', '_')}",
            stop_pips=20.0,
            target_pips=20.0 * r,
        ))

    # ---- combined "profile" variants — common live combinations -----
    out.append(TMVariant(name="tight_scalp",
                         stop_pips=12.0, target_pips=15.0,
                         be_trigger_pips=4.0, trail_start_pips=6.0,
                         trail_distance_pips=2.0))
    out.append(TMVariant(name="wide_swing",
                         stop_pips=30.0, target_pips=60.0,
                         be_trigger_pips=10.0, trail_start_pips=15.0,
                         trail_distance_pips=5.0))
    out.append(TMVariant(name="runner",
                         stop_pips=20.0, target_pips=100.0,
                         be_trigger_pips=6.0, trail_start_pips=10.0,
                         trail_distance_pips=4.0))
    out.append(TMVariant(name="quick_trail",
                         stop_pips=20.0, target_pips=40.0,
                         be_trigger_pips=4.0, trail_start_pips=5.0,
                         trail_distance_pips=2.0))
    out.append(TMVariant(name="atr_runner",
                         atr_stop_mult=1.5, target_pips=80.0,
                         be_trigger_pips=6.0, trail_start_pips=12.0,
                         trail_distance_pips=4.0))

    # ---- adverse-cut controls ---------------------------------------
    for ac in (5, 8, 12, 15):
        out.append(TMVariant(name=f"adverse_cut_{ac}", adverse_cut_pips=float(ac)))

    # ---- control: no BE / no trail ----------------------------------
    out.append(TMVariant(name="fixed_only",
                         be_trigger_pips=0.0, trail_start_pips=0.0))
    # ---- control: no trail (just BE) --------------------------------
    out.append(TMVariant(name="be_only", trail_start_pips=0.0))

    return out


# ======================================================================
# Intrabar replay helpers
# ======================================================================


def _compute_atr(m1_open: np.ndarray, m1_high: np.ndarray,
                 m1_low: np.ndarray, m1_close: np.ndarray,
                 idx: int, window: int, pip_price: float) -> float:
    """Wilder-style ATR in PIPS at bar `idx`, rolling over the prior `window`."""
    if idx < window + 1:
        return 0.0
    lo = idx - window
    tr = np.zeros(window, dtype=float)
    for k in range(window):
        j = lo + k + 1
        high = m1_high[j]
        low = m1_low[j]
        prev_c = m1_close[j - 1]
        tr[k] = max(high - low, abs(high - prev_c), abs(low - prev_c))
    atr_price = float(np.mean(tr))
    return atr_price / pip_price if pip_price > 0 else 0.0


def _intrabar_path(o: float, h: float, l: float, c: float,
                   direction: int) -> list[float]:
    """Canonical path: for long trades, we visit the bar worst-first (low
    before high) to be conservative; for shorts, high before low.
    Matches engine.py semantics.
    """
    if direction > 0:
        return [o, l, h, c]
    return [o, h, l, c]


# ======================================================================
# Per-trade variant evaluation
# ======================================================================


def replay_variant_for_trade(
    entry_bar_idx: int,
    direction: int,
    entry_price: float,
    variant: TMVariant,
    m1_ts: np.ndarray,
    m1_open: np.ndarray,
    m1_high: np.ndarray,
    m1_low: np.ndarray,
    m1_close: np.ndarray,
    pip_price: float,
    max_bars_forward: int = 240,
) -> dict:
    """Walk the M1 bars forward from entry_bar_idx until the variant's
    SL/TP/trail/adverse-cut hits, or until max_bars_forward bars have passed
    (in which case we mark-to-market at the last close).
    """
    # Stop / target (ATR override if enabled)
    if variant.atr_stop_mult > 0.0:
        atr_pips = _compute_atr(
            m1_open, m1_high, m1_low, m1_close,
            entry_bar_idx, variant.atr_window, pip_price,
        )
        stop_pips = max(1.0, atr_pips * variant.atr_stop_mult)
    else:
        stop_pips = variant.stop_pips
    target_pips = variant.target_pips

    if direction > 0:
        sl = entry_price - stop_pips * pip_price
        tp = entry_price + target_pips * pip_price
    else:
        sl = entry_price + stop_pips * pip_price
        tp = entry_price - target_pips * pip_price

    at_be = False
    trailing = False
    closed = False
    close_price = 0.0
    close_reason = ""
    bars_held = 0

    # Apply time-exit cap from variant if set; otherwise use global walk limit.
    walk_cap = (
        variant.max_bars_hold if variant.max_bars_hold > 0 else max_bars_forward
    )
    last_idx = min(len(m1_ts) - 1, entry_bar_idx + walk_cap)
    for i in range(entry_bar_idx + 1, last_idx + 1):
        bars_held = i - entry_bar_idx
        o = float(m1_open[i])
        h = float(m1_high[i])
        l = float(m1_low[i])
        c = float(m1_close[i])

        for px in _intrabar_path(o, h, l, c, direction):
            if direction > 0:
                gain = px - entry_price
                if px <= sl:
                    close_price = sl
                    close_reason = "sl" if not at_be and not trailing else (
                        "be" if at_be and not trailing else "trail"
                    )
                    closed = True
                    break
                if px >= tp:
                    close_price = tp
                    close_reason = "tp"
                    closed = True
                    break
                if variant.be_trigger_pips > 0 and not at_be \
                        and gain >= variant.be_trigger_pips * pip_price:
                    sl = max(sl, entry_price + variant.be_buffer_pips * pip_price)
                    at_be = True
                if variant.trail_start_pips > 0 \
                        and gain >= variant.trail_start_pips * pip_price:
                    new_sl = px - variant.trail_distance_pips * pip_price
                    if new_sl > sl:
                        sl = new_sl
                        trailing = True
            else:
                gain = entry_price - px
                if px >= sl:
                    close_price = sl
                    close_reason = "sl" if not at_be and not trailing else (
                        "be" if at_be and not trailing else "trail"
                    )
                    closed = True
                    break
                if px <= tp:
                    close_price = tp
                    close_reason = "tp"
                    closed = True
                    break
                if variant.be_trigger_pips > 0 and not at_be \
                        and gain >= variant.be_trigger_pips * pip_price:
                    sl = min(sl, entry_price - variant.be_buffer_pips * pip_price)
                    at_be = True
                if variant.trail_start_pips > 0 \
                        and gain >= variant.trail_start_pips * pip_price:
                    new_sl = px + variant.trail_distance_pips * pip_price
                    if new_sl < sl:
                        sl = new_sl
                        trailing = True

        if closed:
            break

        # bar-close adverse cut
        if variant.adverse_cut_pips > 0.0:
            move_pips = (c - entry_price) / pip_price
            if direction < 0:
                move_pips = -move_pips
            if move_pips <= -variant.adverse_cut_pips:
                close_price = c
                close_reason = "adverse_cut"
                closed = True
                break

    if not closed:
        # mark-to-market at last visited bar
        i = last_idx
        close_price = float(m1_close[i])
        close_reason = "mtm"

    pip_move = (close_price - entry_price) / pip_price
    if direction < 0:
        pip_move = -pip_move
    return {
        "variant": variant.name,
        "pips": pip_move,
        "bars_held": bars_held,
        "close_reason": close_reason,
        "exit": close_price,
    }


# ======================================================================
# Main driver
# ======================================================================


def evaluate_variants_for_trades(
    trades_df: pd.DataFrame,
    m1_bars: pd.DataFrame,
    cfg: FiloConfig,
    variants: Iterable[TMVariant] | None = None,
) -> pd.DataFrame:
    """For every row in trades_df, replay every variant forward and return a
    long-form DataFrame with one row per (base_trade, variant) pair.

    trades_df must contain at least: ts_open, entry (price), direction, family.
    m1_bars is the primary-symbol M1 frame (ts, open, high, low, close).
    """
    if variants is None:
        variants = default_variants()
    variants = list(variants)

    m1 = m1_bars.sort_values("ts").reset_index(drop=True)
    ts_arr = m1["ts"].to_numpy()
    # localize ts to naive UTC for searchsorted parity with engine
    if getattr(m1["ts"].dt, "tz", None) is not None:
        ts_for_search = (
            m1["ts"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
        )
    else:
        ts_for_search = ts_arr.astype("datetime64[ns]")

    open_arr = m1["open"].to_numpy(dtype=float)
    high_arr = m1["high"].to_numpy(dtype=float)
    low_arr = m1["low"].to_numpy(dtype=float)
    close_arr = m1["close"].to_numpy(dtype=float)
    pip_price = cfg.instrument.pip_in_points * 0.01
    commission = cfg.sizing.fixed_lot * cfg.instrument.commission_usd_per_lot_round_turn

    rows: list[dict] = []
    for _, tr in trades_df.iterrows():
        ts_open = pd.Timestamp(tr["ts_open"])
        if ts_open.tzinfo is not None:
            ts_open_naive = ts_open.tz_convert("UTC").tz_localize(None)
        else:
            ts_open_naive = ts_open
        entry_idx = int(np.searchsorted(ts_for_search, np.datetime64(ts_open_naive), side="right")) - 1
        if entry_idx < 0:
            continue
        direction = 1 if str(tr["direction"]) == "long" else -1
        entry_price = float(tr["entry"])
        family = str(tr.get("family", "unknown"))
        for v in variants:
            res = replay_variant_for_trade(
                entry_bar_idx=entry_idx,
                direction=direction,
                entry_price=entry_price,
                variant=v,
                m1_ts=ts_for_search,
                m1_open=open_arr,
                m1_high=high_arr,
                m1_low=low_arr,
                m1_close=close_arr,
                pip_price=pip_price,
            )
            gross = pips_to_usd(res["pips"], cfg.sizing.fixed_lot, cfg)
            ts_op = pd.Timestamp(tr["ts_open"])
            hour = int(ts_op.hour) if ts_op is not pd.NaT else -1
            minute = int(ts_op.minute) if ts_op is not pd.NaT else -1
            bucket_5m = (hour * 60 + minute) // 5 * 5 if hour >= 0 else -1
            dow = int(ts_op.dayofweek) if ts_op is not pd.NaT else -1
            rows.append(
                {
                    "ts_open": tr["ts_open"],
                    "family": family,
                    "direction": tr["direction"],
                    "entry": entry_price,
                    "variant": v.name,
                    "pips": res["pips"],
                    "gross_usd": gross,
                    "commission_usd": commission,
                    "usd": gross - commission,
                    "exit": res["exit"],
                    "bars_held": res["bars_held"],
                    "close_reason": res["close_reason"],
                    "hour": hour,
                    "minute": minute,
                    "bucket_5m": bucket_5m,
                    "day_of_week": dow,
                }
            )
    return pd.DataFrame(rows)
