r"""H4/D1 level engine - a direct port of the level code in
"prova finale wowow.mq5".

A level is the high or the low of a closed HTF bar. It stays "valid" until an
HTF bar CLOSES through it:

    a high dies when a later HTF close is above it
    a low  dies when a later HTF close is below it

Seeding replays HTF history with the same survivor rule the live EA uses
(PruneAbove/PruneBelow then PushCapped), so a panel that has just started
holds the same level set as an EA that has been running for weeks.

Invalidated levels are kept in the list, not deleted: the anchor scan is
allowed to use them (InpAnchorUsesInvalid = true in the live preset).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Level:
    price: float
    valid: bool
    created: int          # unix seconds, UTC
    kind: str             # "H4H" | "H4L" | "D1H" | "D1L"


class LevelSet:
    """The four level lists plus the rules that mutate them."""

    def __init__(self, max_levels: int = 100):
        self.max_levels = max_levels
        self.h4_high: list[Level] = []
        self.h4_low: list[Level] = []
        self.d1_high: list[Level] = []
        self.d1_low: list[Level] = []
        self._last_open = {"H4": 0, "D1": 0}

    # ---- mutation ----------------------------------------------------
    def _add(self, arr: list[Level], price: float, created: int, kind: str):
        arr.append(Level(price=price, valid=True, created=created, kind=kind))
        while len(arr) > self.max_levels:
            arr.pop(0)

    @staticmethod
    def _invalidate(arr: list[Level], htf_close: float, is_high: bool):
        for lv in arr:
            if not lv.valid:
                continue
            if (htf_close > lv.price) if is_high else (htf_close < lv.price):
                lv.valid = False

    def seed(self, htf, tag: str, seed_bars: int):
        """Replay closed HTF bars, keeping only the survivors.

        htf: DataFrame with time/high/low/close, ascending, UTC. The LAST row
        is the bar still forming, so it is excluded - a level may only be
        created by a bar that has closed. Everything up to and INCLUDING the
        most recent closed bar is replayed: in the EA the seed loop runs
        p = idx0+n .. idx0+1, and idx0+1 is that bar. Dropping it would leave
        the level set stale by a full HTF period - on D1 that means today's
        break of yesterday's low is invisible.
        """
        arr_hi = self.h4_high if tag == "H4" else self.d1_high
        arr_lo = self.h4_low if tag == "H4" else self.d1_low
        arr_hi.clear()
        arr_lo.clear()

        closed = htf.iloc[:-1]
        if len(closed) < 2:
            return 0
        if seed_bars > 0:
            closed = closed.iloc[-seed_bars:]

        hi_surv: list[tuple[float, int]] = []
        lo_surv: list[tuple[float, int]] = []

        for h, l, c, t in zip(closed.high.to_numpy(float),
                              closed.low.to_numpy(float),
                              closed.close.to_numpy(float),
                              closed.time.to_numpy("int64")):
            # a close through a survivor kills it, exactly as PruneAbove/Below
            hi_surv = [x for x in hi_surv if not c > x[0]]
            lo_surv = [x for x in lo_surv if not c < x[0]]
            hi_surv.append((h, int(t)))
            lo_surv.append((l, int(t)))
            if len(hi_surv) > self.max_levels:
                hi_surv.pop(0)
            if len(lo_surv) > self.max_levels:
                lo_surv.pop(0)

        for p, t in hi_surv:
            self._add(arr_hi, p, t, tag + "H")
        for p, t in lo_surv:
            self._add(arr_lo, p, t, tag + "L")

        self._last_open[tag] = int(htf.time.iloc[-1])
        return len(hi_surv) + len(lo_surv)

    def update(self, htf, tag: str):
        """Roll levels forward when a new HTF bar has opened."""
        arr_hi = self.h4_high if tag == "H4" else self.d1_high
        arr_lo = self.h4_low if tag == "H4" else self.d1_low
        if len(htf) < 2:
            return
        open_now = int(htf.time.iloc[-1])
        if open_now == self._last_open[tag]:
            return
        self._last_open[tag] = open_now
        prev = htf.iloc[-2]                      # the bar that just closed
        hh, ll, cc = float(prev.high), float(prev.low), float(prev.close)
        ct = int(prev.time)
        self._invalidate(arr_hi, cc, True)
        self._invalidate(arr_lo, cc, False)
        self._add(arr_hi, hh, ct, tag + "H")
        self._add(arr_lo, ll, ct, tag + "L")

    # ---- queries -----------------------------------------------------
    def active(self, use_h4=True, use_d1=True) -> list[list[Level]]:
        out = []
        if use_h4:
            out += [self.h4_high, self.h4_low]
        if use_d1:
            out += [self.d1_high, self.d1_low]
        return out

    def find_touch(self, hi, lo, ref, tol, use_h4=True, use_d1=True):
        """Nearest valid level the bar's range intersects."""
        best = None
        for arr in self.active(use_h4, use_d1):
            for lv in arr:
                if not lv.valid:
                    continue
                if hi >= lv.price - tol and lo <= lv.price + tol:
                    d = abs(ref - lv.price)
                    if best is None or d < best[1]:
                        best = (lv.price, d)
        return (True, best[0]) if best else (False, 0.0)

    def scan_break(self, is_down, close, prior_closes, use_h4=True, use_d1=True):
        """Price now sits through a level that the last `origin_bars` closes
        were all on the other side of. Nearest such level wins."""
        best = None
        for arr in self.active(use_h4, use_d1):
            for lv in arr:
                if not lv.valid:
                    continue
                lp = lv.price
                if not ((close < lp) if is_down else (close > lp)):
                    continue
                ok = True
                for pc in prior_closes:
                    if pc == 0.0 or (pc <= lp if is_down else pc >= lp):
                        ok = False
                        break
                if not ok:
                    continue
                d = abs(close - lp)
                if best is None or d < best[1]:
                    best = (lp, d)
        return (True, best[0]) if best else (False, 0.0)

    def scan_rejection(self, is_long, bh, bl, bc, max_dip, tol, prior_closes):
        """Daily-level rejection: the bar reaches a level, overshoots it by no
        more than max_dip, and closes back on the origin side. The level
        nearest the wick extreme wins. Daily levels only, as in the EA."""
        best = None
        for arr in (self.d1_high, self.d1_low):
            for lv in arr:
                if not lv.valid:
                    continue
                lp = lv.price
                if is_long:
                    reached = bl <= lp + tol
                    depth_ok = bl >= lp - max_dip
                    closed_back = bc > lp
                else:
                    reached = bh >= lp - tol
                    depth_ok = bh <= lp + max_dip
                    closed_back = bc < lp
                if not (reached and depth_ok and closed_back):
                    continue
                ok = True
                for pc in prior_closes:
                    if pc == 0.0 or (pc <= lp if is_long else pc >= lp):
                        ok = False
                        break
                if not ok:
                    continue
                wick = abs((bl if is_long else bh) - lp)
                if best is None or wick < best[1]:
                    best = (lp, wick)
        return (True, best[0]) if best else (False, 0.0)

    def scan_anchor(self, is_long, close, armed, use_h4=True, use_d1=True,
                    use_invalid=True):
        """The furthest level actually crossed between the armed level and the
        current close - the real level being reclaimed."""
        best = None
        for arr in self.active(use_h4, use_d1):
            for lv in arr:
                if not use_invalid and not lv.valid:
                    continue
                lp = lv.price
                if is_long:
                    if armed <= lp <= close and (best is None or lp > best):
                        best = lp
                else:
                    if close <= lp <= armed and (best is None or lp < best):
                        best = lp
        return (best is not None, best if best is not None else 0.0)

    def is_live(self, price, tol, use_h4=True, use_d1=True):
        if price == 0.0:
            return False
        for arr in self.active(use_h4, use_d1):
            for lv in arr:
                if lv.valid and abs(lv.price - price) <= tol:
                    return True
        return False

    def nearest(self, above: bool, ref, use_h4=True, use_d1=True):
        best = None
        for arr in self.active(use_h4, use_d1):
            for lv in arr:
                if not lv.valid:
                    continue
                lp = lv.price
                if above and lp > ref and (best is None or lp < best):
                    best = lp
                elif not above and lp < ref and (best is None or lp > best):
                    best = lp
        return (best is not None, best if best is not None else 0.0)

    def counts(self):
        def c(a):
            return sum(1 for x in a if x.valid), len(a)
        return {"h4_high": c(self.h4_high), "h4_low": c(self.h4_low),
                "d1_high": c(self.d1_high), "d1_low": c(self.d1_low)}
