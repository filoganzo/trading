"""Adverse-bar / adverse-forming veto gates — parity port from MQL5.

Source of truth: mql5/Include/FiloHFT/Setups.mqh (search `adverse_bar` /
`adverse_forming`). This module is BACKTEST-ONLY; the live path runs the
original MQL5 logic. Keep the two in lockstep — if Setups.mqh changes,
update here and bump the parity tests.

Two checks, both universal across setup families, both directions:

(1) adverse_bar  — the LAST CLOSED M1 bar's signed body (close-open)
    moved >= adverse_bar_body_pips AGAINST the intended direction.
    Natural memory window = 1 M1 bar; no extra state needed.

(2) adverse_forming — the CURRENTLY FORMING M1 bar has progressed
    >= adverse_forming_progress_pips from its open against the trade
    AND the bar's full range (high-low) is >= adverse_forming_range_pips.
    Catches "don't short into the giant green bar while it's building".

MQL5 quirk preserved: momentum_cont family is EXEMPT from the forming
check only (shift-1 closed-bar check still applies). Callers signal
this by passing `family="momentum_cont"` to `check_adverse`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AdverseInputs:
    direction: int                 # +1 long, -1 short
    last_closed_bar: dict          # {open, high, low, close}
    forming_bar: dict              # {open, high, low, close, ts_ms_open}
    current_mid: float
    pip_size: float                # e.g. 0.1 for XAUUSD


def _pips(price_delta: float, pip_size: float) -> float:
    if pip_size <= 0.0:
        return 0.0
    return price_delta / pip_size


def _cfg_get(cfg: Any, name: str, default):
    # Works for dataclass-like objects, plain objects, and dicts.
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def check_adverse(
    inp: AdverseInputs,
    cfg: Any,
    family: Optional[str] = None,
) -> tuple[bool, str]:
    """Return (ok, reason). ok=False means VETOED; reason mirrors MQL5 strings.

    cfg fields consulted (all optional — missing => gate disabled / 0):
        adverse_bar_enable            : bool
        adverse_bar_body_pips         : float
        adverse_forming_enable        : bool
        adverse_forming_progress_pips : float
        adverse_forming_range_pips    : float
    """
    direction = inp.direction
    if direction not in (+1, -1):
        return True, ""

    pip = inp.pip_size

    # (1) shift-1 closed-bar adverse body.
    if _cfg_get(cfg, "adverse_bar_enable", False):
        lcb = inp.last_closed_bar or {}
        o1 = float(lcb.get("open", 0.0) or 0.0)
        c1 = float(lcb.get("close", 0.0) or 0.0)
        if o1 > 0.0 and c1 > 0.0:
            body_pips = _pips(c1 - o1, pip)      # signed: +up / -down
            thr = float(_cfg_get(cfg, "adverse_bar_body_pips", 0.0) or 0.0)
            if direction == +1 and body_pips <= -thr:
                return False, (
                    f"adverse_bar_body {(-body_pips):.1f}p (shift1 down) >= {thr:.1f}"
                )
            if direction == -1 and body_pips >= thr:
                return False, (
                    f"adverse_bar_body {body_pips:.1f}p (shift1 up) >= {thr:.1f}"
                )

    # (2) shift-0 forming-bar adverse progress+range.
    # momentum_cont family is intentionally exempt — its thesis buys the
    # pullback OF a big trending bar, so the forming bar is SUPPOSED to
    # have moved against us briefly. shift-1 above still applies.
    if _cfg_get(cfg, "adverse_forming_enable", False) and family != "momentum_cont":
        fb = inp.forming_bar or {}
        o0 = float(fb.get("open", 0.0) or 0.0)
        h0 = float(fb.get("high", 0.0) or 0.0)
        l0 = float(fb.get("low",  0.0) or 0.0)
        if o0 > 0.0 and h0 > 0.0 and l0 > 0.0:
            prog_thr  = float(_cfg_get(cfg, "adverse_forming_progress_pips", 0.0) or 0.0)
            range_thr = float(_cfg_get(cfg, "adverse_forming_range_pips",    0.0) or 0.0)
            range_p = _pips(h0 - l0, pip)
            if direction == -1:
                adv = _pips(h0 - o0, pip)         # how far up from open
                if adv >= prog_thr and range_p >= range_thr:
                    return False, f"adverse_forming up {adv:.1f}p range {range_p:.1f}p"
            else:  # direction == +1
                adv = _pips(o0 - l0, pip)         # how far down from open
                if adv >= prog_thr and range_p >= range_thr:
                    return False, f"adverse_forming down {adv:.1f}p range {range_p:.1f}p"

    return True, ""
