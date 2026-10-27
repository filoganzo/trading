"""Config loader — mirrors mql5/Include/FiloHFT/Config.mqh.

The MQL5 side and the Python side read the same `config/default.ini`.
This module provides a typed dataclass view and a flat-key accessor so the
sweep runner can override any parameter by name.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class SymbolsCfg:
    primary: str = "XAUUSD"
    silver_proxy: str = "XAGUSD"
    dxy_proxy: str = "dxy_m6"


@dataclass
class InstrumentCfg:
    pip_in_points: float = 10.0
    usd_per_point_per_lot: float = 1.0
    # IC Markets raw-spread XAUUSD round-turn commission.
    commission_usd_per_lot_round_turn: float = 7.0


@dataclass
class SizingCfg:
    fixed_lot: float = 0.10
    max_position_lots: float = 0.30
    max_concurrent_trades: int = 3


@dataclass
class KeylevelsCfg:
    lookback_days: int = 30
    zone_half_width_pips: float = 1.5
    weight_d1: float = 1.00
    weight_h4: float = 0.75
    weight_h1: float = 0.55
    weight_m30: float = 0.40
    weight_m15: float = 0.28
    weight_m5: float = 0.18
    merge_overlap_pips: float = 2.0
    max_setup_distance_pips: float = 30.0
    min_kl_weight: float = 0.22
    slice_commitment_mult: float = 2.0
    slice_cooldown_ms: int = 90000
    slice_floor_factor: float = 0.10


@dataclass
class ChannelCfg:
    length: int = 100
    stdev_mult: float = 1.2
    extreme_block_mult: float = 1.1


@dataclass
class AccelerationCfg:
    window_ms: int = 1500
    velocity_ema_ms: int = 5000
    burst_sigma: float = 2.2
    min_bursts: int = 2
    liquidity_grab_reject_ms: int = 400


@dataclass
class TrendCfg:
    lookback_bars_m1: int = 30
    persistence_min_ratio: float = 0.60


@dataclass
class CorrelationCfg:
    lookback_seconds: int = 20
    min_agreement: float = 0.55
    enable_xagusd: bool = True
    enable_dxy: bool = True
    dxy_inverse: bool = True


@dataclass
class SetupsCfg:
    enable_breakout: bool = True
    enable_retest: bool = False
    enable_rejection: bool = True
    enable_fakeout: bool = True
    enable_failed_auction: bool = True
    enable_engulfing_accel: bool = True
    enable_mean_reversion: bool = True
    enable_momentum_cont: bool = True
    min_setup_score: float = 0.45
    exhaustion_bars: int = 10
    exhaustion_pips: float = 40.0
    # counter-trend vetoes (fading families)
    counter_trend_vel_veto_pips_per_sec: float = 3.0
    counter_trend_trend_veto: bool = True
    # HTF clearance (D1/H4 obstacle distance)
    htf_min_clearance_pips: float = 6.0
    htf_compression_max_width_pips: float = 25.0
    # continuation extreme multiplier override
    continuation_extreme_block_mult: float = 1.5
    # reversal confirmation stack
    rev_confirm_enable: bool = True
    rev_confirm_min_layers: int = 2
    rev_confirm_partner_enable: bool = True
    rev_confirm_partner_long_ag_min: float = 0.50
    rev_confirm_partner_short_ag_max: float = 0.15
    rev_confirm_burst_decay_enable: bool = True
    rev_confirm_burst_decay_min_ms: int = 500
    rev_confirm_burst_decay_max_ms: int = 3000
    rev_confirm_wick_enable: bool = True
    rev_confirm_wick_min_pips: float = 3.0
    rev_confirm_struct_enable: bool = True
    rev_confirm_struct_range_pips: float = 12.0
    rev_confirm_struct_adverse_veto: bool = True
    # adverse gates
    adverse_bar_enable: bool = True
    adverse_bar_body_pips: float = 30.0
    adverse_forming_enable: bool = True
    adverse_forming_progress_pips: float = 10.0
    adverse_forming_range_pips: float = 15.0
    # echo observability
    echo_enable: bool = True
    echo_window_ms: int = 120000
    # momentum_cont family
    momentum_cont_min_move_pips: float = 50.0
    momentum_cont_max_move_pips: float = 120.0
    momentum_cont_persistence_ms: int = 75000
    momentum_cont_persist_min_delta_pips: float = 30.0
    momentum_cont_window_bars: int = 8
    momentum_cont_min_pullback_pips: float = 3.0
    momentum_cont_max_pullback_pips: float = 8.0
    momentum_cont_partner_min_ag: float = 0.40
    momentum_cont_kl_veto_pips: float = 4.0
    momentum_cont_lot_mult: float = 0.2
    momentum_cont_stop_pips: float = 50.0
    momentum_cont_target_pips: float = 50.0


@dataclass
class DailyBiasCfg:
    enable: bool = True
    half_life_ms: int = 360000
    max_age_ms: int = 900000
    break_tolerance_pips: float = 1.5
    bounce_min_wick_pips: float = 4.0
    score_weight: float = 0.20
    nearby_d1_compound_pips: float = 10.0


@dataclass
class ExecutionCfg:
    intent_ttl_ms: int = 4000
    trigger_tolerance_pips: float = 0.3
    max_slippage_pips: float = 0.8
    max_spread_breakout: float = 0.9
    max_spread_reversion: float = 0.7
    max_spread_default: float = 1.0
    reversal_anchor: str = "bid"  # "bid" | "zone"
    reversal_zone_max_dist_pips: float = 5.0


@dataclass
class TradeManagementCfg:
    be_trigger_pips: float = 6.0
    be_buffer_pips: float = 1.0
    trail_start_pips: float = 8.0
    trail_distance_pips: float = 3.0
    management_poll_ms: int = 10
    target_breakout_pips: float = 30.0
    target_reversion_pips: float = 30.0
    target_rejection_pips: float = 30.0
    stop_breakout_pips: float = 20.0
    stop_reversion_pips: float = 20.0
    stop_rejection_pips: float = 20.0
    stack_min_gap_ms: int = 30000
    stack_require_diverse_edge: bool = True
    post_fill_cooldown_ms: int = 10000
    post_stop_opposite_cooldown_ms: int = 45000


@dataclass
class RiskCfg:
    max_loss_per_trade_usd: float = 5.0
    max_loss_per_minute_usd: float = 8.0
    max_loss_per_hour_usd: float = 20.0
    max_loss_per_day_usd: float = 40.0
    max_order_rejects: int = 3


@dataclass
class SafetyCfg:
    abnormal_spread_pips: float = 2.5
    stall_timeout_ms: int = 3000
    latency_alert_ms: int = 25
    max_slippage_pips: float = 1.5
    halt_seconds: int = 30


@dataclass
class SessionsCfg:
    trade_asia: bool = True
    trade_london: bool = True
    trade_ny: bool = True
    trade_overlap_ln_ny: bool = True
    trade_offhours: bool = False


@dataclass
class MlCfg:
    enable: bool = False
    min_probability: float = 0.55
    model_path: str = "python/models/baseline.joblib"
    blend_weight: float = 0.40


@dataclass
class TelemetryCfg:
    enable: bool = True
    output_dir: str = "telemetry"
    flush_interval_ms: int = 500
    log_ticks: bool = False
    log_features: bool = True
    log_keylevels: bool = True
    log_setups: bool = True
    log_intents: bool = True
    log_fills: bool = True
    log_trades: bool = True
    log_risk: bool = True
    log_safety: bool = True
    log_latency: bool = True
    log_errors: bool = True


@dataclass
class FiloConfig:
    """Flat, typed config struct mirroring the MQL5 FiloConfig."""

    symbols: SymbolsCfg = field(default_factory=SymbolsCfg)
    instrument: InstrumentCfg = field(default_factory=InstrumentCfg)
    sizing: SizingCfg = field(default_factory=SizingCfg)
    keylevels: KeylevelsCfg = field(default_factory=KeylevelsCfg)
    channel: ChannelCfg = field(default_factory=ChannelCfg)
    acceleration: AccelerationCfg = field(default_factory=AccelerationCfg)
    trend: TrendCfg = field(default_factory=TrendCfg)
    correlation: CorrelationCfg = field(default_factory=CorrelationCfg)
    setups: SetupsCfg = field(default_factory=SetupsCfg)
    daily_bias: DailyBiasCfg = field(default_factory=DailyBiasCfg)
    execution: ExecutionCfg = field(default_factory=ExecutionCfg)
    trade_management: TradeManagementCfg = field(default_factory=TradeManagementCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    sessions: SessionsCfg = field(default_factory=SessionsCfg)
    ml: MlCfg = field(default_factory=MlCfg)
    telemetry: TelemetryCfg = field(default_factory=TelemetryCfg)

    # ------------------------------------------------------------------
    # dotted-key access: cfg.get("channel.length") / cfg.set("channel.length", 120)

    def get(self, dotted: str):
        section, key = dotted.split(".", 1)
        return getattr(getattr(self, section), key)

    def set(self, dotted: str, value) -> None:
        section, key = dotted.split(".", 1)
        sub = getattr(self, section)
        # cast based on the dataclass field type
        dc_fields = {f.name: f.type for f in fields(sub)}
        if key not in dc_fields:
            raise KeyError(f"unknown key: {dotted}")
        t = dc_fields[key]
        casted = _cast_value(value, t)
        setattr(sub, key, casted)

    def as_flat_dict(self) -> dict:
        out = {}
        for f in fields(self):
            sub = getattr(self, f.name)
            for sf in fields(sub):
                out[f"{f.name}.{sf.name}"] = getattr(sub, sf.name)
        return out


def _cast_value(raw, type_annotation):
    """Best-effort string → typed cast based on a dataclass annotation string."""
    if isinstance(raw, (int, float, bool)) and type_annotation != "str":
        return raw
    s = str(raw).strip()
    if type_annotation == "bool":
        return s.lower() in ("true", "1", "yes", "on")
    if type_annotation == "int":
        return int(float(s))
    if type_annotation == "float":
        return float(s)
    return s


# ----------------------------------------------------------------------
# ini loader — matches the MQL5 parser's behavior (section + key=value)

_INI_SECTION_MAP = {
    "symbols": ("symbols", {}),
    "instrument": ("instrument", {}),
    "sizing": ("sizing", {}),
    "keylevels": ("keylevels", {}),
    "channel": ("channel", {}),
    "acceleration": ("acceleration", {}),
    "trend": ("trend", {}),
    "correlation": ("correlation", {}),
    "setups": ("setups", {}),
    "daily_bias": ("daily_bias", {}),
    "execution": ("execution", {}),
    "trade_management": ("trade_management", {}),
    "risk": ("risk", {}),
    "safety": ("safety", {}),
    "sessions": ("sessions", {}),
    "ml": ("ml", {}),
    "telemetry": ("telemetry", {}),
}


def load_config(path: str | Path) -> FiloConfig:
    """Load config/default.ini into a FiloConfig.

    Values not present in the file keep their dataclass defaults. Unknown keys
    are ignored with a warning print so typos don't crash startup.
    """
    cfg = FiloConfig()
    parser = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    # preserve case for symbol names like "dxy_m6"
    parser.optionxform = str
    parser.read(path)

    for section in parser.sections():
        if section.lower() not in _INI_SECTION_MAP:
            print(f"[config] unknown section [{section}] — ignored")
            continue
        dc_name, _remap = _INI_SECTION_MAP[section.lower()]
        sub = getattr(cfg, dc_name)
        dc_fields = {f.name: f.type for f in fields(sub)}

        for key, raw in parser.items(section):
            norm_key = key.lower()
            if norm_key not in dc_fields:
                print(f"[config] unknown key [{section}].{key} — ignored")
                continue
            try:
                setattr(sub, norm_key, _cast_value(raw, dc_fields[norm_key]))
            except Exception as exc:  # noqa: BLE001
                print(f"[config] failed to set [{section}].{key}={raw!r}: {exc}")
    return cfg


# ----------------------------------------------------------------------
# pip / point helpers — XAUUSD-specific math centralized here


def pips_to_points(pips: float, cfg: FiloConfig) -> float:
    return pips * cfg.instrument.pip_in_points


def points_to_pips(points: float, cfg: FiloConfig) -> float:
    return points / cfg.instrument.pip_in_points


def pips_to_usd(pips: float, lots: float, cfg: FiloConfig) -> float:
    """Convert a pip move to USD PnL for the given lot size."""
    points = pips_to_points(pips, cfg)
    return points * lots * cfg.instrument.usd_per_point_per_lot
