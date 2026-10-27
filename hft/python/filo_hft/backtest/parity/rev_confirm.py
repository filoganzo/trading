"""Reversal-confirmation 4-layer gate — Python parity port of the MQL5
stack in `Setups.mqh` (rev_confirm block, lines ~322-381).

NOT live code. Used by the backtest engine to reproduce MQL5 gating
without pulling in the live feature/correlation/keylevels objects.

The 4 layers are additive — a setup passes when
`layers_passed >= cfg.rev_confirm_min_layers`. Fading families
(rejection, fakeout, failed_auction, mean_reversion) are expected to
call this; trend families are not gated here. Layer ordering matches
MQL5 so the `layers_passed` list reads the same: struct, partner,
burst_decay, wick.

Notes on MQL5 quirks preserved:
  * The struct layer can ADVERSE-VETO (short-circuit hard fail) when
    the nearest M30/M15 zone sits against the trade direction.
  * burst_decay requires the last adverse burst to be in the window
    [min_ms .. max_ms] ago — silence alone is NOT decay.
  * wick uses the *forming* M1 bar (shift 0) in MQL5; here we use the
    most-recent bar in `m1_bars`, which is the closest parity target a
    bar-based backtest can offer.
  * partner uses an OR over XAG and DXY divergence, and short-window
    direction of XAU must still be "wrong way" — see PartnerDivergenceFor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --- pip helper -------------------------------------------------------------

def _pips_from_price(px: float, pip_size: float) -> float:
    if pip_size <= 0:
        return 0.0
    return px / pip_size


# --- inputs / result --------------------------------------------------------

@dataclass
class RevConfirmInputs:
    direction: int                          # +1 long, -1 short
    mid: float                              # current mid price
    ts_ms: int                              # "now" for burst_decay gap calc

    # M1 OHLC of last ~N bars, most-recent last.
    # Each dict: {ts_ms, open, high, low, close, volume}
    m1_bars: list[dict] = field(default_factory=list)

    pip_size: float = 0.01                  # XAUUSD default (1 pip = 0.01)

    # ---- burst_decay layer (Features.BurstDecayAgainst parity) ----
    last_burst_ts_up: int = 0               # ms timestamp of last UP burst
    last_burst_ts_dn: int = 0               # ms timestamp of last DN burst

    # ---- partner layer (Correlation.PartnerDivergenceFor parity) ----
    r_xau_short: float = 0.0                # short-window XAU return (fractional)
    xagusd_ok: bool = False
    dxy_ok: bool = False
    corr_enable_xagusd: bool = True
    corr_enable_dxy: bool = True
    agreement_xagusd: float = 0.0
    short_agreement_xagusd: float = 0.0
    agreement_dxy: float = 0.0
    short_agreement_dxy: float = 0.0

    # ---- struct layer (KeyLevels.MidTfBias parity) ----
    # Pre-computed bias: +1 supports long, -1 supports short, 0 neutral.
    # Caller computes this from M30/M15 zones within struct_range_pips.
    mid_tf_bias: int = 0


@dataclass
class RevConfirmResult:
    ok: bool
    layers_passed: list[str]                # ordered: struct,partner,burst_decay,wick
    reason: str


# --- layer evaluators -------------------------------------------------------

def _layer_struct(inp: RevConfirmInputs, cfg: Any) -> tuple[int, bool]:
    """Returns (pass_delta, adverse_veto). adverse_veto short-circuits."""
    if not getattr(cfg, "rev_confirm_struct_enable", True):
        return 0, False
    bias = inp.mid_tf_bias
    supportive = (inp.direction == +1 and bias == +1) or \
                 (inp.direction == -1 and bias == -1)
    adverse    = (inp.direction == +1 and bias == -1) or \
                 (inp.direction == -1 and bias == +1)
    if adverse and getattr(cfg, "rev_confirm_struct_adverse_veto", True):
        return 0, True
    return (1 if supportive else 0), False


def _layer_partner(inp: RevConfirmInputs, cfg: Any) -> int:
    if not getattr(cfg, "rev_confirm_partner_enable", True):
        return 0
    if not (inp.xagusd_ok or inp.dxy_ok):
        return 0
    # XAU must still be moving the "wrong" way in the short window.
    min_r = 1.0e-5
    if inp.direction == +1 and inp.r_xau_short >= -min_r:
        return 0
    if inp.direction == -1 and inp.r_xau_short <= min_r:
        return 0
    long_min  = getattr(cfg, "rev_confirm_partner_long_ag_min", 0.50)
    short_max = getattr(cfg, "rev_confirm_partner_short_ag_max", 0.15)
    xag_div = (inp.corr_enable_xagusd and inp.xagusd_ok and
               inp.agreement_xagusd > long_min and
               inp.short_agreement_xagusd < short_max)
    dxy_div = (inp.corr_enable_dxy and inp.dxy_ok and
               inp.agreement_dxy > long_min and
               inp.short_agreement_dxy < short_max)
    return 1 if (xag_div or dxy_div) else 0


def _layer_burst_decay(inp: RevConfirmInputs, cfg: Any) -> int:
    if not getattr(cfg, "rev_confirm_burst_decay_enable", True):
        return 0
    min_ms = int(getattr(cfg, "rev_confirm_burst_decay_min_ms", 500))
    max_ms = int(getattr(cfg, "rev_confirm_burst_decay_max_ms", 3000))
    # For LONG reversal we want the last DOWN-burst to have gone quiet.
    last = inp.last_burst_ts_dn if inp.direction == +1 else inp.last_burst_ts_up
    if last <= 0:
        return 0
    gap = inp.ts_ms - last
    return 1 if (min_ms <= gap <= max_ms) else 0


def _layer_wick(inp: RevConfirmInputs, cfg: Any) -> int:
    if not getattr(cfg, "rev_confirm_wick_enable", True):
        return 0
    if not inp.m1_bars:
        return 0
    last = inp.m1_bars[-1]
    h0 = float(last.get("high", 0.0))
    l0 = float(last.get("low",  0.0))
    if h0 <= 0 or l0 <= 0:
        return 0
    min_w = float(getattr(cfg, "rev_confirm_wick_min_pips", 3.0))
    if inp.direction == +1:
        return 1 if _pips_from_price(inp.mid - l0, inp.pip_size) >= min_w else 0
    if inp.direction == -1:
        return 1 if _pips_from_price(h0 - inp.mid, inp.pip_size) >= min_w else 0
    return 0


# --- entry point ------------------------------------------------------------

def check_rev_confirm(inp: RevConfirmInputs, cfg: Any) -> RevConfirmResult:
    """Evaluate the 4 layers. Pass if `layers_passed >= cfg.rev_confirm_min_layers`.

    Layer order (matches MQL5): struct, partner, burst_decay, wick.
    The struct layer can also adverse-veto (hard fail) regardless of
    the other layers.
    """
    if not getattr(cfg, "rev_confirm_enable", True):
        return RevConfirmResult(ok=True, layers_passed=[], reason="rev_confirm disabled")

    if inp.direction not in (+1, -1):
        return RevConfirmResult(ok=False, layers_passed=[], reason=f"bad direction {inp.direction}")

    layers: list[str] = []

    struct_delta, adverse_veto = _layer_struct(inp, cfg)
    if adverse_veto:
        return RevConfirmResult(ok=False, layers_passed=[],
                                reason="rev_confirm struct adverse (m30/m15 against)")
    if struct_delta:
        layers.append("struct")

    if _layer_partner(inp, cfg):
        layers.append("partner")
    if _layer_burst_decay(inp, cfg):
        layers.append("burst_decay")
    if _layer_wick(inp, cfg):
        layers.append("wick")

    min_layers = int(getattr(cfg, "rev_confirm_min_layers", 2))
    if len(layers) < min_layers:
        return RevConfirmResult(
            ok=False,
            layers_passed=layers,
            reason=f"rev_confirm {len(layers)}/{min_layers} [{','.join(layers)}]",
        )
    return RevConfirmResult(
        ok=True,
        layers_passed=layers,
        reason=f"rev_confirm {len(layers)}/{min_layers} [{','.join(layers)}]",
    )
