"""
Trade management: the exit half of the strategy, swept as a first-class alpha
source rather than fixed at some arbitrary ATR multiple.

Supports, in any combination:
  * stop      none | ATR multiple | channel-band | swing structure
  * target    none | ATR multiple | R multiple | channel MIDLINE | opposite band
  * trims     partial exits at R thresholds, remainder runs
  * breakeven promote the stop to entry after +xR
  * trail     ATR trail or chandelier (from running extreme)
  * time      hard bar cap
  * signal    exit when price re-enters the channel, or on an opposing setup

Midline and opposite-band targets matter for a reversion thesis: the trade is
premised on price snapping back INTO the channel, so the channel itself is the
natural objective. Those targets move bar to bar, which is why the simulator
takes full path matrices for the bands rather than a single entry-time level.

Everything is vectorised across trades and looped over bars, so a whole
management grid can be evaluated on one signal set cheaply.

Fill conventions (deliberately conservative)
  - stop and target both touched in one bar -> the STOP fills first, and no trim
    is banked on that bar (the intrabar order is unknowable, so assume the
    worst). If the target is hit WITHOUT the stop, trims below it do fill --
    booking the whole position at the target would overstate the result.
  - a bar that GAPS past a level fills at that bar's OPEN, not at the level.
    Booking a gap-through-stop at the stop price is the classic optimistic
    backtest error: it makes every loss exactly -1R by construction.
  - dynamic targets (channel midline / opposite band) use the band as of the
    PREVIOUS bar. The band on bar j is derived from bar j's own close, so
    resting an order at it is look-ahead -- measured at roughly the size of the
    entire edge on this strategy.
  - a trailing stop is only tightened using bars already closed
  - the time stop closes at that bar's close
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NO_STOP = 1e9        # "no stop" is modelled as an absurdly distant one so that
                     # R stays well-defined and comparable across variants


@dataclass
class Mgmt:
    # --- stop ---------------------------------------------------------------
    stop_mode: str = "atr"          # 'atr' | 'none' | 'band' | 'swing'
    sl_atr: float = 2.0
    swing_lookback: int = 12
    swing_pad_atr: float = 0.25

    # --- target -------------------------------------------------------------
    tp_mode: str = "atr"            # 'atr' | 'none' | 'mid' | 'opposite' | 'r'
    tp_atr: float = 3.0
    tp_r: float = 2.0

    # --- partial exits ------------------------------------------------------
    # [(r_threshold, fraction_of_original)] e.g. [(1.0, .5), (2.0, .25)]
    trims: tuple = ()

    # --- stop management ----------------------------------------------------
    be_at_r: float = 0.0            # 0 = off
    be_offset_r: float = 0.0        # lock in this much when promoting
    trail_mode: str = "none"        # 'none' | 'atr' | 'chandelier'
    trail_atr: float = 1.5
    trail_start_r: float = 0.0      # only start trailing past this MFE

    # --- other exits --------------------------------------------------------
    max_bars: int = 48
    exit_on_reentry: bool = False   # close when price returns inside the band

    def label(self) -> str:
        t = "-".join(f"{r:g}@{f:g}" for r, f in self.trims) or "none"
        return (f"stop={self.stop_mode}:{self.sl_atr:g} tp={self.tp_mode}:"
                f"{self.tp_atr:g}/{self.tp_r:g} trims={t} be={self.be_at_r:g} "
                f"trail={self.trail_mode}:{self.trail_atr:g} bars={self.max_bars}")


def simulate_managed(H, L, C, entry, atr, side, m: Mgmt,
                     MID=None, UPPER=None, LOWER=None, SWING=None,
                     commission_r=None, O=None,
                     MID0=None, UPPER0=None, LOWER0=None):
    """
    Resolve every trade under one management spec.

    H, L, C        (n_trades, horizon) forward path matrices, starting on the
                   FILL bar so a stop can trigger immediately
    entry          fill price per trade (spread already applied)
    atr            ATR at the signal bar, per trade
    side           +1 long, -1 short
    MID/UPPER/LOWER  (n, horizon) channel path, required for band-based exits
    SWING          (n,) structural stop level, required for stop_mode='swing'

    Returns dict of per-trade arrays: r, mfe_r, mae_r, bars, reason, n_partials.
    R is always measured against the INITIAL stop distance, so variants with
    different stops stay comparable.
    """
    n, hz = H.shape
    hz = min(hz, max(1, int(m.max_bars)))
    H, L, C = H[:, :hz], L[:, :hz], C[:, :hz]
    O = O[:, :hz] if O is not None else None
    MIDp = UPp = LOp = None
    if MID is not None:
        MID, UPPER, LOWER = MID[:, :hz], UPPER[:, :hz], LOWER[:, :hz]

        # Lagged copies. The band level usable DURING bar j is the one known at
        # the close of j-1; on the fill bar it is the band at the signal bar.
        def _lag(A, A0):
            out = np.empty_like(A)
            out[:, 0] = A0 if A0 is not None else A[:, 0]
            if A.shape[1] > 1:
                out[:, 1:] = A[:, :-1]
            return out

        MIDp, UPp, LOp = _lag(MID, MID0), _lag(UPPER, UPPER0), _lag(LOWER, LOWER0)

    # ---- initial stop distance -------------------------------------------
    if m.stop_mode == "none":
        sl_dist = np.full(n, NO_STOP, dtype=float)
        r_unit = m.sl_atr * atr          # keep R comparable to the ATR variants
    elif m.stop_mode == "swing":
        if SWING is None:
            raise ValueError("stop_mode='swing' needs SWING")
        sl_dist = np.abs(entry - SWING) + m.swing_pad_atr * atr
        r_unit = sl_dist
    elif m.stop_mode == "band":
        if LOWER is None or LOWER0 is None or UPPER0 is None:
            raise ValueError("stop_mode='band' needs the channel path plus the "
                             "signal-bar levels LOWER0/UPPER0")
        # The SIGNAL bar's band, not the fill bar's. By construction the entry
        # sits right at the band, so |entry - ref| is a near-cancellation of two
        # close numbers: one bar of band drift is a ~16% relative error in the R
        # denominator that every result is quoted in.
        ref = LOWER0 if side > 0 else UPPER0
        sl_dist = np.abs(entry - ref) + m.swing_pad_atr * atr
        r_unit = sl_dist
    else:
        sl_dist = m.sl_atr * atr
        r_unit = sl_dist
    r_unit = np.where(np.isfinite(r_unit) & (r_unit > 0), r_unit, np.nan)

    stop = entry - side * sl_dist
    live = np.isfinite(entry) & np.isfinite(r_unit)

    # ---- static target ----------------------------------------------------
    if m.tp_mode == "none":
        targ = np.full(n, np.nan)
    elif m.tp_mode == "atr":
        targ = entry + side * m.tp_atr * atr
    elif m.tp_mode == "r":
        targ = entry + side * m.tp_r * r_unit
    else:
        targ = np.full(n, np.nan)        # dynamic, resolved per bar below

    remaining = np.where(live, 1.0, 0.0)
    realized = np.zeros(n)               # R already banked from trims
    r_out = np.full(n, np.nan)
    bars = np.full(n, -1, dtype=np.int64)
    reason = np.zeros(n, dtype=np.int8)  # 1 tp 2 sl 3 time 4 trail/be 5 reentry
    mfe = np.zeros(n)
    mae = np.zeros(n)
    n_part = np.zeros(n, dtype=np.int8)
    be_done = np.zeros(n, dtype=bool)
    trims = sorted(m.trims, key=lambda t: t[0])
    trim_hit = np.zeros((n, len(trims)), dtype=bool)

    for j in range(hz):
        act = live & (bars < 0) & np.isfinite(H[:, j])
        if not act.any():
            break

        hi, lo, cl = H[:, j], L[:, j], C[:, j]
        fav = side * ((hi - entry) if side > 0 else (lo - entry))
        adv = side * ((lo - entry) if side > 0 else (hi - entry))
        with np.errstate(invalid="ignore"):
            mfe = np.where(act, np.maximum(mfe, fav / r_unit), mfe)
            mae = np.where(act, np.minimum(mae, adv / r_unit), mae)

        # ---- dynamic targets, lagged one bar so they are observable --------
        if m.tp_mode == "mid":
            targ = MIDp[:, j]
        elif m.tp_mode == "opposite":
            targ = UPp[:, j] if side > 0 else LOp[:, j]

        op = O[:, j] if O is not None else None
        zero = np.zeros(n, dtype=bool)

        # ---- stop and target detection, with gap fills ---------------------
        if side > 0:
            hit_sl = act & (lo <= stop) & (remaining > 1e-9)
            hit_tp = act & np.isfinite(targ) & (hi >= targ) & (remaining > 1e-9)
            gap_sl = (hit_sl & (op <= stop)) if op is not None else zero
            gap_tp = (hit_tp & (op >= targ)) if op is not None else zero
        else:
            hit_sl = act & (hi >= stop) & (remaining > 1e-9)
            hit_tp = act & np.isfinite(targ) & (lo <= targ) & (remaining > 1e-9)
            gap_sl = (hit_sl & (op >= stop)) if op is not None else zero
            gap_tp = (hit_tp & (op <= targ)) if op is not None else zero

        sl_fill = np.where(gap_sl, op, stop) if op is not None else stop
        tp_fill = np.where(gap_tp, op, targ) if op is not None else targ
        tp_only = hit_tp & ~hit_sl             # stop wins a same-bar collision

        # ---- partial exits, skipped on any bar the stop also took ----------
        for ti, (rlvl, frac) in enumerate(trims):
            trigger = entry + side * rlvl * r_unit
            hit = act & ~trim_hit[:, ti] & ~hit_sl & (remaining > 1e-9) & (
                (hi >= trigger) if side > 0 else (lo <= trigger))
            if hit.any():
                take = np.minimum(frac, remaining)
                realized = np.where(hit, realized + take * rlvl, realized)
                remaining = np.where(hit, remaining - take, remaining)
                trim_hit[:, ti] |= hit
                n_part = np.where(hit, n_part + 1, n_part)

        idx = np.where(hit_sl)[0]
        if len(idx):
            rr = side * (sl_fill[idx] - entry[idx]) / r_unit[idx]
            r_out[idx] = realized[idx] + remaining[idx] * rr
            bars[idx] = j
            # "be_trail" only if the stop actually moved off its initial level,
            # not merely because breakeven was once armed
            init_stop = entry[idx] - side * sl_dist[idx]
            reason[idx] = np.where(np.abs(stop[idx] - init_stop) > 1e-12, 4, 2)
            remaining[idx] = 0.0
        idx = np.where(tp_only & (remaining > 1e-9))[0]
        if len(idx):
            rr = side * (tp_fill[idx] - entry[idx]) / r_unit[idx]
            r_out[idx] = realized[idx] + remaining[idx] * rr
            bars[idx] = j
            reason[idx] = 1
            remaining[idx] = 0.0

        still = live & (bars < 0) & np.isfinite(cl)

        # ---- close on re-entry into the channel ---------------------------
        if m.exit_on_reentry and MID is not None:
            back = still & ((cl >= LOWER[:, j]) if side > 0 else (cl <= UPPER[:, j]))
            idx = np.where(back)[0]
            if len(idx):
                rr = side * (cl[idx] - entry[idx]) / r_unit[idx]
                r_out[idx] = realized[idx] + remaining[idx] * rr
                bars[idx] = j
                reason[idx] = 5
                remaining[idx] = 0.0
                still = live & (bars < 0) & np.isfinite(cl)

        # ---- breakeven -----------------------------------------------------
        if m.be_at_r > 0:
            promote = still & ~be_done & (mfe >= m.be_at_r)
            newstop = entry + side * m.be_offset_r * r_unit
            better = (newstop > stop) if side > 0 else (newstop < stop)
            stop = np.where(promote & better, newstop, stop)
            be_done |= promote

        # ---- trailing ------------------------------------------------------
        if m.trail_mode != "none":
            arm = still & (mfe >= m.trail_start_r)
            if m.trail_mode == "chandelier":
                ext = entry + side * mfe * r_unit          # running extreme
            else:
                ext = cl
            cand = ext - side * m.trail_atr * atr
            better = (cand > stop) if side > 0 else (cand < stop)
            stop = np.where(arm & better & np.isfinite(cand), cand, stop)

    # ---- time stop --------------------------------------------------------
    openpos = live & (bars < 0)
    if openpos.any():
        lastc = np.full(n, np.nan)
        for j in range(hz - 1, -1, -1):
            need = openpos & ~np.isfinite(lastc)
            if not need.any():
                break
            lastc = np.where(need, C[:, j], lastc)
        idx = np.where(openpos & np.isfinite(lastc))[0]
        if len(idx):
            rr = side * (lastc[idx] - entry[idx]) / r_unit[idx]
            r_out[idx] = realized[idx] + remaining[idx] * rr
            bars[idx] = hz - 1
            reason[idx] = 3

    if commission_r is not None:
        r_out = r_out - commission_r

    return {"r": r_out, "mfe_r": mfe, "mae_r": mae, "bars": bars + 1,
            "reason": reason, "n_partials": n_part, "r_unit": r_unit,
            "sl_dist": sl_dist}


REASON_NAMES = {1: "tp", 2: "sl", 3: "time", 4: "be_trail", 5: "reentry", 0: "none"}


# ------------------------------------------------------------- preset grid

def management_grid(level: str = "full") -> list[Mgmt]:
    """
    The management search space.

    'core'  ~40 specs, for the first pass over many entry configs
    'full'  ~250 specs, for the finalists only
    """
    out: list[Mgmt] = []
    if level == "core":
        for sl in (1.5, 2.0, 3.0):
            for tp_mode, tp in (("atr", 2.0), ("atr", 3.0), ("atr", 5.0),
                                ("mid", 0.0), ("opposite", 0.0), ("none", 0.0)):
                out.append(Mgmt(stop_mode="atr", sl_atr=sl, tp_mode=tp_mode,
                                tp_atr=tp, max_bars=48))
        for sl in (2.0, 3.0):
            out.append(Mgmt(stop_mode="atr", sl_atr=sl, tp_mode="mid",
                            trims=((1.0, 0.5),), be_at_r=1.0, max_bars=48))
            out.append(Mgmt(stop_mode="atr", sl_atr=sl, tp_mode="none",
                            trims=((1.0, 0.5), (2.0, 0.25)),
                            trail_mode="chandelier", trail_atr=1.5,
                            trail_start_r=1.0, max_bars=96))
        out.append(Mgmt(stop_mode="none", tp_mode="mid", max_bars=96))
        out.append(Mgmt(stop_mode="none", tp_mode="atr", tp_atr=3.0, max_bars=96))
        return out

    for stop_mode, sl in (("atr", 1.0), ("atr", 1.5), ("atr", 2.0), ("atr", 3.0),
                          ("atr", 4.0), ("band", 0.0), ("swing", 0.0), ("none", 0.0)):
        for tp_mode, tp in (("atr", 1.5), ("atr", 2.0), ("atr", 3.0), ("atr", 5.0),
                            ("mid", 0.0), ("opposite", 0.0), ("none", 0.0)):
            for trims in ((), ((1.0, 0.5),), ((1.0, 0.34), (2.0, 0.33))):
                for be in (0.0, 1.0):
                    for trail, tatr in (("none", 0.0), ("chandelier", 1.5)):
                        if trail == "none" and tp_mode == "none" and not trims:
                            continue          # nothing would ever close it early
                        out.append(Mgmt(stop_mode=stop_mode, sl_atr=(sl or 2.0),
                                        tp_mode=tp_mode, tp_atr=tp, trims=trims,
                                        be_at_r=be, trail_mode=trail,
                                        trail_atr=tatr, trail_start_r=1.0,
                                        max_bars=48))
    return out
