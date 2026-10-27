r"""Signal engine: the Python port of ProcessBar() from
"prova finale wowow.mq5".

The EA evaluates on the CLOSE of a bar (g_evalOffset = 1, so shift 0 is the
last closed bar). This module does the same: `evaluate()` drops the bar that
is still forming and decides on the one that just closed. A signal therefore
appears at most once per minute, right after the M1 close.

State that must survive between calls - the armed break, the per-level
cooldowns, the bar counter - lives in the module-level Engine, because a
break on one bar and its reclaim several bars later are one setup.

Correlation is the whole reason the bridge exists: gold and silver come from
OANDA, DXY from IC Markets, and the three are matched on the same UTC minute
by feed.align() before they get here.

    LONG  needs silver GREEN and DXY RED
    SHORT needs silver RED   and DXY GREEN
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import indicators as ind
from levels import LevelSet

PIP = 0.1
POINT = 0.01              # gold quotes to 2dp on all three brokers


@dataclass
class Engine:
    """Cross-call state. One instance per panel session."""
    levels: LevelSet = field(default_factory=lambda: LevelSet(100))
    seeded: bool = False
    bar_index: int = 0
    last_bar_time: int = 0

    armed_long: bool = False
    armed_long_lvl: float = 0.0
    armed_long_bar: int = -1
    armed_short: bool = False
    armed_short_lvl: float = 0.0
    armed_short_bar: int = -1

    cooldowns: list = field(default_factory=list)   # (level_price, until_bar)
    last_signal_id: str = ""

    def prune_cooldowns(self):
        self.cooldowns = [x for x in self.cooldowns if x[1] > self.bar_index]

    def in_cooldown(self, level, tol=POINT):
        return any(abs(p - level) <= tol and u > self.bar_index
                   for p, u in self.cooldowns)


ENGINE = Engine()


def _rome_minutes(ts: int) -> int:
    """Minutes since midnight Rome time for a UTC timestamp.

    Italy is UTC+1, UTC+2 during DST (last Sunday of March to last Sunday of
    October). Computed rather than pulled from a tz database so the bridge has
    no extra dependency.
    """
    import datetime as _dt
    d = _dt.datetime.utcfromtimestamp(ts)

    def last_sunday(year, month):
        day = 31
        while True:
            try:
                c = _dt.date(year, month, day)
            except ValueError:
                day -= 1
                continue
            if c.weekday() == 6:
                return c
            day -= 1

    start = last_sunday(d.year, 3)
    end = last_sunday(d.year, 10)
    # DST switches at 01:00 UTC on both boundaries
    s = _dt.datetime(start.year, start.month, start.day, 1)
    e = _dt.datetime(end.year, end.month, end.day, 1)
    off = 2 if s <= d < e else 1
    loc = d + _dt.timedelta(hours=off)
    return loc.hour * 60 + loc.minute


def _hhmm(s: str) -> int:
    s = s.strip().replace(":", "")
    if len(s) != 4 or not s.isdigit():
        return -1
    return int(s[:2]) * 60 + int(s[2:])


def in_time_window(ts: int, cfg) -> bool:
    if not cfg.get("use_time_window"):
        return True
    a = _hhmm(str(cfg.get("session_start", "")))
    b = _hhmm(str(cfg.get("session_end", "")))
    if a < 0 or b < 0:
        return True
    m = _rome_minutes(ts)
    return (a <= m < b) if a <= b else (m >= a or m < b)   # b<a wraps midnight


def evaluate(bars, h4, d1, cfg, engine: Engine | None = None) -> dict:
    """Decide on the last CLOSED bar of `bars`.

    bars : aligned frame from feed.align() - gold OHLC plus s_open/s_close
           (silver) and d_open/d_close (DXY), one row per shared UTC minute.
    h4/d1: gold HTF frames from feed.get_htf().
    cfg  : the panel settings dict.

    Returns a signal dict for the EA. action is ENTER or FLAT; FLAT carries a
    `reason` naming the gate that blocked, so the panel log explains itself.
    """
    e = engine or ENGINE

    if len(bars) < max(cfg["reg_length"], 120) + 5:
        return {"action": "FLAT", "reason": f"warming up ({len(bars)} bars)"}

    # ---- evaluate on the closed bar -----------------------------------
    closed = bars.iloc[:-1]
    o = closed.open.to_numpy(float)
    h = closed.high.to_numpy(float)
    l = closed.low.to_numpy(float)
    c = closed.close.to_numpy(float)
    t = closed.time.to_numpy("int64")
    bar_time = int(t[-1])

    use_h4 = bool(cfg["enable_h4"])
    use_d1 = bool(cfg["enable_d1"])

    # ---- levels --------------------------------------------------------
    if not e.seeded:
        e.levels = LevelSet(int(cfg["max_levels"]))
        if use_h4:
            e.levels.seed(h4, "H4", int(cfg["seed_htf_bars"]))
        if use_d1:
            e.levels.seed(d1, "D1", int(cfg["seed_htf_bars"]))
        e.seeded = True
    else:
        if use_h4:
            e.levels.update(h4, "H4")
        if use_d1:
            e.levels.update(d1, "D1")

    # Only advance state once per closed bar. A 5-second poll re-reads the
    # same closed bar many times; arming twice off one bar would corrupt the
    # reclaim window.
    new_bar = bar_time != e.last_bar_time
    if new_bar:
        e.bar_index += 1
        e.last_bar_time = bar_time
    e.prune_cooldowns()

    # ---- indicators ----------------------------------------------------
    wt_ok, wt = ind.wavetrend(h, l, c, int(cfg["wt_channel"]), int(cfg["wt_average"]))
    if not wt_ok:
        return {"action": "FLAT", "reason": "wavetrend not ready"}
    thr = float(cfg["wt_threshold"])
    wt_ob, wt_os = wt >= thr, wt <= -thr

    use_lr = bool(cfg["use_linreg_candles"])
    lr_len = int(cfg["linreg_length"]) if use_lr else 1
    if use_lr:
        lr_o = ind.linreg_at(o, h, l, c, 0, lr_len, 0)
        lr_h = ind.linreg_at(o, h, l, c, 1, lr_len, 0)
        lr_l = ind.linreg_at(o, h, l, c, 2, lr_len, 0)
        lr_c = ind.linreg_at(o, h, l, c, 3, lr_len, 0)
    else:
        lr_o, lr_h, lr_l, lr_c = o[-1], h[-1], l[-1], c[-1]

    sig_ok, lr_sig = ind.linreg_signal(o, h, l, c, lr_len,
                                       int(cfg["signal_length"]),
                                       bool(cfg["sma_signal"]), use_lr)

    reg_ok, basis, sd = ind.regression_stats(o, h, l, c, int(cfg["reg_length"]))
    if not reg_ok:
        return {"action": "FLAT", "reason": "regression not ready"}
    dev = float(cfg["reg_dev_mult"])
    up_dev, lo_dev = basis + dev * sd, basis - dev * sd
    # DEV_WICK is the live preset: the LinReg wick must reach the band
    at_upper, at_lower = lr_h >= up_dev, lr_l <= lo_dev

    # ---- the bar itself -------------------------------------------------
    bo, bh, bl, bc = float(o[-1]), float(h[-1]), float(l[-1]), float(c[-1])
    main_green, main_red = bc > bo, bc < bo
    main_doji = bc == bo

    # ---- correlation ----------------------------------------------------
    so, sc = float(closed.s_open.iloc[-1]), float(closed.s_close.iloc[-1])
    do, dc = float(closed.d_open.iloc[-1]), float(closed.d_close.iloc[-1])
    ag_data, op_data = so > 0 and sc > 0, do > 0 and dc > 0
    ag_green, ag_red, ag_doji = sc > so, sc < so, sc == so
    op_green, op_red, op_doji = dc > do, dc < do, dc == do

    use_corr = bool(cfg["use_correlation"])
    # feed.align() inner-joins on the UTC minute, so a row existing at all
    # means both partners are stamped on this same minute: freshness is
    # already guaranteed upstream.
    long_corr_ok = (not use_corr) or (ag_data and op_data and ag_green and op_red)
    short_corr_ok = (not use_corr) or (ag_data and op_data and ag_red and op_green)

    doji_block = bool(cfg["reject_doji"]) and (
        main_doji or (use_corr and (ag_doji or op_doji)))

    # ---- geometry --------------------------------------------------------
    max_dist = float(cfg["max_dist_pips"]) * PIP
    min_room = float(cfg["min_room_pips"]) * PIP
    tol = 0.0

    origin_bars = int(cfg["origin_bars"])
    prior = [float(c[-1 - b]) for b in range(1, origin_bars + 1)
             if len(c) > b]

    any_touch = e.levels.find_touch(bh, bl, bc, tol, use_h4, use_d1)[0]
    long_ctx_full = any_touch and wt_os and at_lower
    short_ctx_full = any_touch and wt_ob and at_upper
    long_ctx_hold = wt_os and at_lower
    short_ctx_hold = wt_ob and at_upper

    break_down, bd_lvl = e.levels.scan_break(True, bc, prior, use_h4, use_d1)
    break_up, bu_lvl = e.levels.scan_break(False, bc, prior, use_h4, use_d1)

    # ---- armed-setup lifecycle (once per bar) ---------------------------
    reclaim_window = int(cfg["reclaim_window"])
    if new_bar:
        if e.armed_long:
            if e.bar_index - e.armed_long_bar > reclaim_window or not long_ctx_hold:
                e.armed_long, e.armed_long_lvl = False, 0.0
        if e.armed_short:
            if e.bar_index - e.armed_short_bar > reclaim_window or not short_ctx_hold:
                e.armed_short, e.armed_short_lvl = False, 0.0

        if break_down and long_ctx_full:
            e.armed_long, e.armed_long_lvl = True, bd_lvl
            e.armed_long_bar = e.bar_index
        if break_up and short_ctx_full:
            e.armed_short, e.armed_short_lvl = True, bu_lvl
            e.armed_short_bar = e.bar_index

    reclaim_long = e.armed_long and e.bar_index > e.armed_long_bar and bc > e.armed_long_lvl
    reclaim_short = e.armed_short and e.bar_index > e.armed_short_bar and bc < e.armed_short_lvl

    # ---- anchor ----------------------------------------------------------
    anchor_long, anchor_short = e.armed_long_lvl, e.armed_short_lvl
    if bool(cfg["use_anchor"]):
        if reclaim_long:
            f, a = e.levels.scan_anchor(True, bc, e.armed_long_lvl, use_h4, use_d1)
            if f:
                anchor_long = a
        if reclaim_short:
            f, a = e.levels.scan_anchor(False, bc, e.armed_short_lvl, use_h4, use_d1)
            if f:
                anchor_short = a

    long_dist = (bc - anchor_long) if reclaim_long else 0.0
    short_dist = (anchor_short - bc) if reclaim_short else 0.0
    long_dist_ok = reclaim_long and long_dist <= max_dist
    short_dist_ok = reclaim_short and short_dist <= max_dist

    has_above, near_above = e.levels.nearest(True, bc, use_h4, use_d1)
    has_below, near_below = e.levels.nearest(False, bc, use_h4, use_d1)
    room_above_ok = (not has_above) or (near_above - bc) >= min_room
    room_below_ok = (not has_below) or (bc - near_below) >= min_room

    long_cd = reclaim_long and e.in_cooldown(anchor_long)
    short_cd = reclaim_short and e.in_cooldown(anchor_short)

    # ---- rejection scan (daily levels only) ------------------------------
    # One candle reaches a daily level, overshoots it by no more than
    # rej_max_dip, and closes back on the origin side. Unlike B&R this is a
    # single-bar event: nothing is armed and nothing persists.
    rej_on = bool(cfg.get("enable_rejection", True)) and use_d1
    rej_max_dip = float(cfg.get("rej_max_dip_pips", 15.0)) * PIP
    approach = int(cfg.get("rej_approach_bars", 1))
    rej_prior = [float(c[-1 - b]) for b in range(1, approach + 1) if len(c) > b]

    if rej_on:
        rej_long_found, rej_long_lvl = e.levels.scan_rejection(
            True, bh, bl, bc, rej_max_dip, tol, rej_prior)
        rej_short_found, rej_short_lvl = e.levels.scan_rejection(
            False, bh, bl, bc, rej_max_dip, tol, rej_prior)
    else:
        rej_long_found = rej_short_found = False
        rej_long_lvl = rej_short_lvl = 0.0

    # candle shape: the wick must point BACK INTO the level
    body = abs(bc - bo)
    up_wick = bh - max(bc, bo)
    dn_wick = min(bc, bo) - bl
    wick_mult = float(cfg.get("rej_wick_body_mult", 2.0))
    rej_min_body = float(cfg.get("rej_min_body_pips", 15.0)) * PIP
    body_ok = body >= rej_min_body
    mode = int(cfg.get("rej_shape_mode", 3))

    def shape_ok(wick_ok):
        if mode == 0:
            return True
        if mode == 1:
            return wick_ok
        if mode == 2:
            return body_ok
        if mode == 3:
            return wick_ok or body_ok
        return wick_ok and body_ok

    rej_shape_long = shape_ok(dn_wick >= wick_mult * body)
    rej_shape_short = shape_ok(up_wick >= wick_mult * body)

    need_color = bool(cfg.get("rej_require_color", True))
    rej_color_long = (not need_color) or main_green
    rej_color_short = (not need_color) or main_red

    rej_long_dist = (bc - rej_long_lvl) if rej_long_found else 0.0
    rej_short_dist = (rej_short_lvl - bc) if rej_short_found else 0.0
    rej_long_dist_ok = rej_long_found and rej_long_dist <= max_dist
    rej_short_dist_ok = rej_short_found and rej_short_dist <= max_dist

    rej_long_cd = rej_long_found and e.in_cooldown(rej_long_lvl)
    rej_short_cd = rej_short_found and e.in_cooldown(rej_short_lvl)

    # ---- signal-side / colour filters (off in the live preset) ----------
    lr_above = sig_ok and lr_c > lr_sig
    lr_below = sig_ok and lr_c < lr_sig
    long_sig_ok = (not cfg.get("require_signal_side", False)) or lr_above
    short_sig_ok = (not cfg.get("require_signal_side", False)) or lr_below
    lr_green, lr_red = lr_o < lr_c, lr_o > lr_c
    long_lr_ok = (not cfg.get("require_lr_color", False)) or lr_green
    short_lr_ok = (not cfg.get("require_lr_color", False)) or lr_red

    in_window = in_time_window(bar_time, cfg)

    # ---- entry decision ---------------------------------------------------
    br_on = bool(cfg.get("enable_break_reclaim", True))

    long_entry = (br_on and reclaim_long and in_window and not doji_block
                  and main_green and long_corr_ok and long_sig_ok and long_lr_ok
                  and long_dist_ok and room_above_ok and not long_cd)
    short_entry = (br_on and reclaim_short and in_window and not doji_block
                   and main_red and short_corr_ok and short_sig_ok and short_lr_ok
                   and short_dist_ok and room_below_ok and not short_cd)

    # Rejection carries its own extreme requirement: the EA tests wtOS/atLower
    # directly rather than going through anyTouch, because the rejected level
    # IS the touch.
    rej_long_entry = (rej_long_found and in_window and not doji_block
                      and rej_color_long and wt_os and at_lower and long_corr_ok
                      and rej_shape_long and long_sig_ok and long_lr_ok
                      and rej_long_dist_ok and room_above_ok and not rej_long_cd)
    rej_short_entry = (rej_short_found and in_window and not doji_block
                       and rej_color_short and wt_ob and at_upper and short_corr_ok
                       and rej_shape_short and short_sig_ok and short_lr_ok
                       and rej_short_dist_ok and room_below_ok and not rej_short_cd)

    # B&R resolves first: it carries an armed setup that would otherwise
    # expire, while a rejection recurs on its own terms.
    take_long = long_entry
    take_short = short_entry and not long_entry
    take_rej_long = rej_long_entry and not long_entry and not short_entry
    take_rej_short = (rej_short_entry and not long_entry and not short_entry
                      and not rej_long_entry)

    diag = {
        "wt": round(wt, 1), "z_close": round((lr_c - basis) / sd, 2) if sd else 0.0,
        "at_upper": at_upper, "at_lower": at_lower, "touch": any_touch,
        "armed_long": e.armed_long, "armed_short": e.armed_short,
        "armed_long_lvl": round(e.armed_long_lvl, 2),
        "armed_short_lvl": round(e.armed_short_lvl, 2),
        "silver": "G" if ag_green else ("R" if ag_red else "D"),
        "dxy": "G" if op_green else ("R" if op_red else "D"),
        "rej_long": rej_long_found, "rej_short": rej_short_found,
        "rej_lvl": round(rej_long_lvl if rej_long_found else rej_short_lvl, 2),
        "bar_time": bar_time, "bar_index": e.bar_index,
        "levels": e.levels.counts(),
    }

    if take_long or take_short or take_rej_long or take_rej_short:
        is_br = take_long or take_short
        is_long = take_long or take_rej_long
        if is_br:
            anchor = anchor_long if is_long else anchor_short
            dist = long_dist if is_long else short_dist
            module = "BR"
        else:
            anchor = rej_long_lvl if is_long else rej_short_lvl
            dist = rej_long_dist if is_long else rej_short_dist
            module = "REJ"

        side = "LONG" if is_long else "SHORT"
        sid = f"{bar_time}-{module}-{side}"
        if sid == e.last_signal_id:
            return {"action": "FLAT", "reason": "signal already sent", "diag": diag}
        e.last_signal_id = sid
        e.cooldowns.append((anchor, e.bar_index + int(cfg["cooldown_bars"])))

        if is_br:
            why = (f"B&R {side} anchor {anchor:.2f} "
                   f"dist {dist / PIP:.1f}p wt {wt:.0f}")
        else:
            dip = (rej_long_lvl - bl) if is_long else (bh - rej_short_lvl)
            why = (f"REJECTION {side} level {anchor:.2f} "
                   f"dip {dip / PIP:.1f}p dist {dist / PIP:.1f}p wt {wt:.0f}")

        return {
            "action": "ENTER",
            "dir": side,
            "module": module,
            "ref_price": bc,
            "anchor": round(anchor, 2),
            "tp_pips": float(cfg["tp_pips"]),
            "sl_pips": float(cfg["sl_pips"]),
            "lots": float(cfg["lots"]),
            "reason": why,
            "diag": diag,
        }

    # ---- why not ----------------------------------------------------------
    if reclaim_long or reclaim_short:
        is_l = reclaim_long
        if not in_window:
            why = "window closed"
        elif doji_block:
            why = "doji"
        elif is_l and not main_green:
            why = "bar colour"
        elif (not is_l) and not main_red:
            why = "bar colour"
        elif is_l and not long_corr_ok:
            why = "corr disagrees"
        elif (not is_l) and not short_corr_ok:
            why = "corr disagrees"
        elif is_l and not long_dist_ok:
            why = f"too extended ({long_dist / PIP:.1f}p)"
        elif (not is_l) and not short_dist_ok:
            why = f"too extended ({short_dist / PIP:.1f}p)"
        elif is_l and not room_above_ok:
            why = "trapped"
        elif (not is_l) and not room_below_ok:
            why = "trapped"
        elif long_cd or short_cd:
            why = "cooldown"
        else:
            why = "blocked"
        return {"action": "FLAT", "reason": f"reclaim {why}", "diag": diag}

    if rej_long_found or rej_short_found:
        is_l = rej_long_found
        if not in_window:
            why = "window closed"
        elif doji_block:
            why = "doji"
        elif is_l and not rej_color_long:
            why = "bar colour"
        elif (not is_l) and not rej_color_short:
            why = "bar colour"
        elif is_l and not (wt_os and at_lower):
            why = "no extreme"
        elif (not is_l) and not (wt_ob and at_upper):
            why = "no extreme"
        elif is_l and not long_corr_ok:
            why = "correlation"
        elif (not is_l) and not short_corr_ok:
            why = "correlation"
        elif is_l and not rej_shape_long:
            why = "shape failed"
        elif (not is_l) and not rej_shape_short:
            why = "shape failed"
        elif is_l and not rej_long_dist_ok:
            why = f"too extended ({rej_long_dist / PIP:.1f}p)"
        elif (not is_l) and not rej_short_dist_ok:
            why = f"too extended ({rej_short_dist / PIP:.1f}p)"
        elif is_l and not room_above_ok:
            why = "trapped"
        elif (not is_l) and not room_below_ok:
            why = "trapped"
        elif rej_long_cd or rej_short_cd:
            why = "cooldown"
        else:
            why = "blocked"
        return {"action": "FLAT", "reason": f"rejection {why}", "diag": diag}

    if e.armed_long or e.armed_short:
        side = "long" if e.armed_long else "short"
        lvl = e.armed_long_lvl if e.armed_long else e.armed_short_lvl
        return {"action": "FLAT",
                "reason": f"armed {side} @ {lvl:.2f}, waiting reclaim",
                "diag": diag}

    return {"action": "FLAT", "reason": "", "diag": diag}
