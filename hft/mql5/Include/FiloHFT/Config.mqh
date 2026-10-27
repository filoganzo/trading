//+------------------------------------------------------------------+
//|  FiloHFT/Config.mqh                                              |
//|  Loads config/default.ini from the MT5 files folder and exposes |
//|  a flat struct of every tunable. Also supports live overrides   |
//|  via config/live_overrides.json polled on a 5s timer.           |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_CONFIG_MQH__
#define __FILOHFT_CONFIG_MQH__

#include <FiloHFT/Common.mqh>

//+------------------------------------------------------------------+
//|  Flat config struct — keep field names identical to ini keys     |
//|  so the loader can be mechanical (`section.key` → `section_key`).|
//+------------------------------------------------------------------+
struct FiloConfig {
   // symbols
   string   sym_primary;
   string   sym_silver;
   string   sym_dxy;

   // instrument
   double   pip_in_points;
   double   usd_per_point_per_lot;

   // sizing
   double   fixed_lot;
   double   max_position_lots;
   int      max_concurrent_trades;

   // keylevels
   int      kl_lookback_days;
   double   kl_zone_half_width_pips;
   double   kl_weight_d1;
   double   kl_weight_h4;
   double   kl_weight_h1;
   double   kl_weight_m30;
   double   kl_weight_m15;
   double   kl_weight_m5;
   double   kl_merge_overlap_pips;
   double   kl_max_setup_distance_pips;
   double   kl_min_weight;
   double   kl_slice_commitment_mult;   // body-commit beyond far edge, in half-widths
   int      kl_slice_cooldown_ms;       // linear recovery window after a slice
   double   kl_slice_floor_factor;      // weight scale at slice moment (0..1)

   // channel
   int      chan_length;
   double   chan_stdev_mult;
   double   chan_extreme_block_mult;

   // acceleration
   int      accel_window_ms;
   int      accel_velocity_ema_ms;
   double   accel_burst_sigma;
   int      accel_min_bursts;
   int      accel_liquidity_grab_reject_ms;

   // trend
   int      trend_lookback_bars_m1;
   double   trend_persistence_min_ratio;

   // correlation
   int      corr_lookback_seconds;
   double   corr_min_agreement;
   bool     corr_enable_xagusd;
   bool     corr_enable_dxy;
   bool     corr_dxy_inverse;

   // setups
   bool     setup_breakout;
   bool     setup_retest;
   bool     setup_rejection;
   bool     setup_fakeout;
   bool     setup_failed_auction;
   bool     setup_engulfing_accel;
   bool     setup_mean_reversion;
   double   setup_min_score;
   int      setup_exhaustion_bars;
   double   setup_exhaustion_pips;
   // counter-trend family vetoes (B + C)
   double   setup_counter_trend_vel_veto_pips_per_sec;
   bool     setup_counter_trend_trend_veto;
   // Reversal-confirmation gate: N-of-4 positive confirmations required
   // before a reversal-family setup (mean_reversion/rejection/fakeout/
   // failed_auction) is allowed to fire. Each layer is individually
   // toggleable so they can be A/B'd live.
   //   layer 1 — partner divergence (Correlation delta)
   //   layer 2 — burst decay against the move (Features last-burst ts)
   //   layer 3 — intra-bar wick into the setup direction (iHigh/iLow 0)
   //   layer 4 — M30/M15 level-side bias (KeyLevels.MidTfBias)
   bool     setup_rev_confirm_enable;
   int      setup_rev_confirm_min_layers;
   bool     setup_rev_confirm_partner_enable;
   double   setup_rev_confirm_partner_long_ag_min;
   double   setup_rev_confirm_partner_short_ag_max;
   bool     setup_rev_confirm_burst_decay_enable;
   int      setup_rev_confirm_burst_decay_min_ms;
   int      setup_rev_confirm_burst_decay_max_ms;
   bool     setup_rev_confirm_wick_enable;
   double   setup_rev_confirm_wick_min_pips;
   bool     setup_rev_confirm_struct_enable;
   double   setup_rev_confirm_struct_range_pips;
   bool     setup_rev_confirm_struct_adverse_veto;
   // Adverse-bar gate. Two mechanics that universally (all families,
   // both directions) refuse entries that would be trading INTO a
   // violently-contrary move:
   //   (1) shift-1: last closed M1 bar body >= N pips in the opposite
   //       direction of the intended trade → refuse until that bar rolls
   //       off (natural ~60s memory via iOpen/iClose(1)).
   //   (2) shift-0: the currently-forming M1 bar has already travelled
   //       >= N pips from its open against the intended direction AND
   //       its range is >= M pips → refuse until the bar closes.
   bool     setup_adverse_bar_enable;
   double   setup_adverse_bar_body_pips;
   bool     setup_adverse_forming_enable;
   double   setup_adverse_forming_progress_pips;
   double   setup_adverse_forming_range_pips;

   // Echo-tag overlay — records same-direction same-zone reconfirmations
   // within a rolling window. Purely passive right now (logged, never
   // scored/vetoed); validated downstream by splitting shadow-trade
   // outcomes on echo_count. See Setups.mqh::_ComputeEchoCount.
   bool     setup_echo_enable;
   int      setup_echo_window_ms;

   // Momentum-continuation family. Fires only in the exhaustion dead zone
   // (move_pips >= min_move_pips) in the SAME direction as the dominant
   // move, on a shallow pullback. Thesis: a discretionary "missed the
   // breakout, buy the first dip in a strong trend" play.
   //   - lot_mult: multiplier applied to fixed_lot (e.g. 0.2 -> 0.02 lots)
   //   - stop/target: overrides the normal family profile; BE + trail still
   //     apply from tm_be_trigger_pips / tm_trail_start_pips
   // Other gates (adverse_forming, KL weight, at-zone) are bypassed
   // because momentum_cont's thesis is structural, not zone-based.
   bool     setup_momentum_cont;
   double   setup_momentum_cont_min_move_pips;
   double   setup_momentum_cont_max_move_pips;
   // Time-based persistence: the move must still be running NOW. We sample
   // the mid price `persistence_ms` ago and require the net delta to be
   // >= persist_min_delta_pips in the same direction as the overall move.
   // Much looser than counting N-of-M closed bars — catches short bursts
   // that only needed 75s to qualify, not 6+ minutes.
   int      setup_momentum_cont_persistence_ms;
   double   setup_momentum_cont_persist_min_delta_pips;
   int      setup_momentum_cont_window_bars;
   double   setup_momentum_cont_min_pullback_pips;
   double   setup_momentum_cont_max_pullback_pips;
   double   setup_momentum_cont_partner_min_ag;
   double   setup_momentum_cont_kl_veto_pips;
   double   setup_momentum_cont_lot_mult;
   double   setup_momentum_cont_stop_pips;
   double   setup_momentum_cont_target_pips;

   // Daily Bias — soft score adjustment driven by recent D1 break/bounce
   // events with strong exponential time decay. See DailyBias.mqh.
   bool     daily_bias_enable;
   int      daily_bias_half_life_ms;
   int      daily_bias_max_age_ms;
   double   daily_bias_break_tolerance_pips;
   double   daily_bias_bounce_min_wick_pips;
   double   daily_bias_score_weight;
   double   daily_bias_nearby_d1_compound_pips;
   // minimum clearance (pips) to the nearest D1- or H4-origin key level
   // in the direction of the trade. If price sits closer than this, the
   // setup is refused because there is no runway for hidden BE to lock
   // in before the HTF level acts as a ceiling/floor. 0 disables the gate.
   double   setup_htf_min_clearance_pips;
   // HTF compression-pocket width threshold (pips). Plumbing for a
   // future breakout-class gate that would refuse entries when price is
   // sandwiched between a D1/H4 ceiling and floor whose near-edges are
   // within this many pips. NOT currently wired in Setups.mqh — kept as
   // tuneable config so the gate can be enabled without another recompile
   // once we decide whether the breakout family is flawed globally or
   // only in compression pockets.
   double   setup_htf_compression_max_width_pips;
   // Channel-extreme block multiplier for CONTINUATION families only
   // (breakout/retest/engulfing_accel). Looser than chan_extreme_block_mult
   // because continuation setups are INHERENTLY near the extreme — the
   // whole thesis is "price has already broken out of the envelope."
   // Hard-block only when price is *very* extended (e.g. 1.5 stdev).
   // Must be >= chan_extreme_block_mult or it becomes a no-op.
   double   setup_continuation_extreme_block_mult;

   // execution
   int      exec_intent_ttl_ms;
   double   exec_trigger_tolerance_pips;
   double   exec_max_slippage_pips;
   double   exec_max_spread_breakout;
   double   exec_max_spread_reversion;
   double   exec_max_spread_default;
   // Knife size-down: if the last completed M1 bar moved >= threshold pips
   // AGAINST the intended entry direction, multiply lot by exec_knife_size_mult.
   // Purpose: the trade is still taken (backtest says it's profitable) but
   // sized down to cap tail-slippage exposure during fast moves.
   double   exec_knife_size_threshold_pips;
   double   exec_knife_size_mult;

   // trade management
   double   tm_be_trigger_pips;
   double   tm_be_buffer_pips;
   double   tm_trail_start_pips;
   double   tm_trail_distance_pips;
   int      tm_management_poll_ms;
   double   tm_target_breakout_pips;
   double   tm_target_reversion_pips;
   double   tm_target_rejection_pips;
   double   tm_stop_breakout_pips;
   double   tm_stop_reversion_pips;
   double   tm_stop_rejection_pips;
   // same-direction anti-pyramid gate — a second concurrent trade in
   // the same direction requires (a) the first to be at BE (unchanged),
   // (b) a minimum time gap, and (c) a diverse source of edge.
   int      tm_stack_min_gap_ms;
   bool     tm_stack_require_diverse_edge;
   // post-fill cooldown — after ANY close, refuse new entries for
   // this many ms. Prevents the setup engine from firing on the
   // stop-fill spike itself and ping-ponging direction. 0 disables.
   int      tm_post_fill_cooldown_ms;
   // post-stop OPPOSITE-direction cooldown — after a stop-out
   // specifically, refuse the inverse direction for this many ms.
   // Stricter than the generic cooldown because SAR chop is the
   // dominant failure mode in low-volatility windows. 0 disables.
   int      tm_post_stop_opposite_cooldown_ms;

   // risk
   double   risk_max_loss_per_trade_usd;
   double   risk_max_loss_per_minute_usd;
   double   risk_max_loss_per_hour_usd;
   double   risk_max_loss_per_day_usd;
   int      risk_max_order_rejects;
   // direction-aware post-loss cooldown (A)
   int      risk_post_loss_cooldown_ms;

   // safety
   double   safety_abnormal_spread_pips;
   int      safety_stall_timeout_ms;
   int      safety_latency_alert_ms;
   double   safety_max_slippage_pips;
   int      safety_halt_seconds;

   // sessions
   bool     sess_trade_asia;
   bool     sess_trade_london;
   bool     sess_trade_ny;
   bool     sess_trade_overlap_ln_ny;
   bool     sess_trade_offhours;

   // ml
   bool     ml_enable;
   double   ml_min_probability;
   string   ml_model_path;
   double   ml_blend_weight;

   // telemetry
   bool     tel_enable;
   string   tel_output_dir;
   int      tel_flush_interval_ms;
   bool     tel_log_ticks;
   bool     tel_log_features;
   bool     tel_log_keylevels;
   bool     tel_log_setups;
   bool     tel_log_intents;
   bool     tel_log_fills;
   bool     tel_log_trades;
   bool     tel_log_risk;
   bool     tel_log_safety;
   bool     tel_log_latency;
   bool     tel_log_errors;
   bool     tel_log_shadow;
   bool     tel_log_tmlab;
};

//+------------------------------------------------------------------+
//|  Global config instance.                                         |
//+------------------------------------------------------------------+
FiloConfig g_cfg;

//--- content cache for Config_ReloadOverrides so the caller can tell
//    whether the override file actually changed since the last poll.
string g_last_overrides_content = "";

//--- forward decls
bool Config_Load(const string ini_path, FiloConfig &cfg);
void Config_SetDefaults(FiloConfig &cfg);
bool Config_ParseLine(const string line, string &section, string &key, string &value);
void Config_Apply(const string section, const string key, const string value, FiloConfig &cfg);
bool Config_ReloadOverrides(const string overrides_path, FiloConfig &cfg);

//+------------------------------------------------------------------+
//|  Hard-coded defaults — used if the ini is missing or as a        |
//|  fallback when a key is absent. Keep in sync with default.ini.   |
//+------------------------------------------------------------------+
void Config_SetDefaults(FiloConfig &cfg) {
   // IMPORTANT: these binary defaults must stay in sync with config/default.ini.
   // They are used when the ini fails to load AND during the first few seconds
   // of boot before Config_ReloadOverrides fires. If they are tighter than the
   // live ini, the EA will self-halt at boot and trap itself behind a stale
   // threshold. Keep values WIDE here — the ini narrows them if appropriate.
   cfg.sym_primary                 = "XAUUSD";
   cfg.sym_silver                  = "XAGUSD";
   cfg.sym_dxy                     = "DXY_M6";

   cfg.pip_in_points               = 10.0;
   cfg.usd_per_point_per_lot       = 1.0;

   cfg.fixed_lot                   = 0.10;
   cfg.max_position_lots           = 0.30;
   cfg.max_concurrent_trades       = 3;

   cfg.kl_lookback_days            = 30;
   cfg.kl_zone_half_width_pips     = 1.5;
   cfg.kl_weight_d1                = 1.00;
   cfg.kl_weight_h4                = 0.75;
   cfg.kl_weight_h1                = 0.55;
   cfg.kl_weight_m30               = 0.40;
   cfg.kl_weight_m15               = 0.28;
   cfg.kl_weight_m5                = 0.18;
   cfg.kl_merge_overlap_pips       = 2.0;
   cfg.kl_max_setup_distance_pips  = 30.0;
   cfg.kl_min_weight               = 0.22;
   cfg.kl_slice_commitment_mult    = 2.0;
   cfg.kl_slice_cooldown_ms        = 90000;
   cfg.kl_slice_floor_factor       = 0.10;

   cfg.chan_length                 = 100;
   cfg.chan_stdev_mult             = 1.2;
   cfg.chan_extreme_block_mult     = 1.1;

   cfg.accel_window_ms             = 2000;
   cfg.accel_velocity_ema_ms       = 5000;
   cfg.accel_burst_sigma           = 1.0;
   cfg.accel_min_bursts            = 1;
   cfg.accel_liquidity_grab_reject_ms = 400;

   cfg.trend_lookback_bars_m1      = 30;
   cfg.trend_persistence_min_ratio = 0.60;

   cfg.corr_lookback_seconds       = 20;
   cfg.corr_min_agreement          = 0.55;
   cfg.corr_enable_xagusd          = true;
   cfg.corr_enable_dxy             = true;
   cfg.corr_dxy_inverse            = true;

   cfg.setup_breakout              = true;
   cfg.setup_retest                = false;
   cfg.setup_rejection             = true;
   cfg.setup_fakeout               = true;
   cfg.setup_failed_auction        = true;
   cfg.setup_engulfing_accel       = true;
   cfg.setup_mean_reversion        = true;
   cfg.setup_min_score             = 0.45;
   cfg.setup_exhaustion_bars       = 10;
   cfg.setup_exhaustion_pips       = 40;
   cfg.setup_counter_trend_vel_veto_pips_per_sec = 3.0;
   cfg.setup_counter_trend_trend_veto            = true;
   cfg.setup_htf_min_clearance_pips              = 6.0;
   cfg.setup_htf_compression_max_width_pips      = 25.0;
   cfg.setup_continuation_extreme_block_mult     = 1.5;
   cfg.setup_rev_confirm_enable                  = true;
   cfg.setup_rev_confirm_min_layers              = 2;
   cfg.setup_rev_confirm_partner_enable          = true;
   cfg.setup_rev_confirm_partner_long_ag_min     = 0.50;
   cfg.setup_rev_confirm_partner_short_ag_max    = 0.15;
   cfg.setup_rev_confirm_burst_decay_enable      = true;
   cfg.setup_rev_confirm_burst_decay_min_ms      = 500;
   cfg.setup_rev_confirm_burst_decay_max_ms      = 3000;
   cfg.setup_rev_confirm_wick_enable             = true;
   cfg.setup_rev_confirm_wick_min_pips           = 3.0;
   cfg.setup_rev_confirm_struct_enable           = true;
   cfg.setup_rev_confirm_struct_range_pips       = 12.0;
   cfg.setup_rev_confirm_struct_adverse_veto     = true;

   cfg.setup_adverse_bar_enable                 = true;
   cfg.setup_adverse_bar_body_pips              = 30.0;
   cfg.setup_adverse_forming_enable             = true;
   cfg.setup_adverse_forming_progress_pips      = 10.0;
   cfg.setup_adverse_forming_range_pips         = 15.0;

   cfg.setup_echo_enable                        = true;
   cfg.setup_echo_window_ms                     = 120000;

   cfg.setup_momentum_cont                         = true;
   cfg.setup_momentum_cont_min_move_pips           = 50.0;
   cfg.setup_momentum_cont_max_move_pips           = 120.0;
   cfg.setup_momentum_cont_persistence_ms          = 75000;
   cfg.setup_momentum_cont_persist_min_delta_pips  = 30.0;
   cfg.setup_momentum_cont_window_bars             = 8;
   cfg.setup_momentum_cont_min_pullback_pips       = 3.0;
   cfg.setup_momentum_cont_max_pullback_pips       = 8.0;
   cfg.setup_momentum_cont_partner_min_ag          = 0.40;
   cfg.setup_momentum_cont_kl_veto_pips            = 4.0;
   cfg.setup_momentum_cont_lot_mult                = 0.2;
   cfg.setup_momentum_cont_stop_pips               = 50.0;
   cfg.setup_momentum_cont_target_pips             = 50.0;

   cfg.daily_bias_enable                  = true;
   cfg.daily_bias_half_life_ms            = 360000;
   cfg.daily_bias_max_age_ms              = 900000;
   cfg.daily_bias_break_tolerance_pips    = 1.5;
   cfg.daily_bias_bounce_min_wick_pips    = 4.0;
   cfg.daily_bias_score_weight            = 0.20;
   cfg.daily_bias_nearby_d1_compound_pips = 10.0;

   cfg.exec_intent_ttl_ms          = 4000;
   cfg.exec_trigger_tolerance_pips = 2.0;
   cfg.exec_max_slippage_pips      = 0.8;
   cfg.exec_max_spread_breakout    = 3.0;
   cfg.exec_max_spread_reversion   = 3.0;
   cfg.exec_max_spread_default     = 3.0;
   cfg.exec_knife_size_threshold_pips = 40.0;
   cfg.exec_knife_size_mult           = 0.5;

   cfg.tm_be_trigger_pips          = 6.0;
   cfg.tm_be_buffer_pips           = 1.0;
   cfg.tm_trail_start_pips         = 8.0;
   cfg.tm_trail_distance_pips      = 3.0;
   cfg.tm_management_poll_ms       = 10;
   cfg.tm_target_breakout_pips     = 30.0;
   cfg.tm_target_reversion_pips    = 30.0;
   cfg.tm_target_rejection_pips    = 30.0;
   cfg.tm_stop_breakout_pips       = 20.0;
   cfg.tm_stop_reversion_pips      = 20.0;
   cfg.tm_stop_rejection_pips      = 20.0;
   cfg.tm_stack_min_gap_ms         = 30000;
   cfg.tm_stack_require_diverse_edge = true;
   cfg.tm_post_fill_cooldown_ms            = 10000;   //  10s
   cfg.tm_post_stop_opposite_cooldown_ms   = 45000;   //  45s

   cfg.risk_max_loss_per_trade_usd  = 10000.00;
   cfg.risk_max_loss_per_minute_usd = 10000.00;
   cfg.risk_max_loss_per_hour_usd   = 10000.00;
   cfg.risk_max_loss_per_day_usd    = 10000.00;
   cfg.risk_max_order_rejects       = 20;
   cfg.risk_post_loss_cooldown_ms   = 45000;

   cfg.safety_abnormal_spread_pips = 15.0;
   cfg.safety_stall_timeout_ms     = 30000;
   cfg.safety_latency_alert_ms     = 50;
   cfg.safety_max_slippage_pips    = 3.0;
   cfg.safety_halt_seconds         = 1;

   cfg.sess_trade_asia             = true;
   cfg.sess_trade_london           = true;
   cfg.sess_trade_ny               = true;
   cfg.sess_trade_overlap_ln_ny    = true;
   cfg.sess_trade_offhours         = false;

   cfg.ml_enable                   = false;
   cfg.ml_min_probability          = 0.55;
   cfg.ml_model_path               = "python/models/baseline.joblib";
   cfg.ml_blend_weight             = 0.40;

   cfg.tel_enable                  = true;
   cfg.tel_output_dir              = "telemetry";
   cfg.tel_flush_interval_ms       = 500;
   cfg.tel_log_ticks               = false;
   cfg.tel_log_features            = true;
   cfg.tel_log_keylevels           = true;
   cfg.tel_log_setups              = true;
   cfg.tel_log_intents             = true;
   cfg.tel_log_fills               = true;
   cfg.tel_log_trades              = true;
   cfg.tel_log_risk                = true;
   cfg.tel_log_safety              = true;
   cfg.tel_log_latency             = true;
   cfg.tel_log_errors              = true;
   cfg.tel_log_shadow              = true;
   cfg.tel_log_tmlab               = true;
}

//+------------------------------------------------------------------+
//|  Very small ini parser. Not fancy — handles comments (; or #),    |
//|  [sections], key=value, whitespace. Skips anything weird.         |
//+------------------------------------------------------------------+
bool Config_ParseLine(const string raw, string &section, string &key, string &value) {
   string line = raw;
   StringTrimLeft(line); StringTrimRight(line);
   if(StringLen(line) == 0) return false;
   ushort c0 = StringGetCharacter(line, 0);
   if(c0 == ';' || c0 == '#') return false;
   if(c0 == '[') {
      int end = StringFind(line, "]");
      if(end <= 1) return false;
      section = StringSubstr(line, 1, end - 1);
      StringTrimLeft(section); StringTrimRight(section);
      key = ""; value = "";
      return true;
   }
   int eq = StringFind(line, "=");
   if(eq <= 0) return false;
   key   = StringSubstr(line, 0, eq);
   value = StringSubstr(line, eq + 1);
   StringTrimLeft(key);   StringTrimRight(key);
   StringTrimLeft(value); StringTrimRight(value);
   // strip inline comment
   int semi = StringFind(value, ";");
   if(semi >= 0) { value = StringSubstr(value, 0, semi); StringTrimRight(value); }
   return true;
}

//--- string → bool
bool ParseBool(const string s) {
   string v = s; StringToLower(v);
   return (v == "true" || v == "1" || v == "yes" || v == "on");
}

//+------------------------------------------------------------------+
//|  Dispatch: apply one (section, key, value) to the struct.        |
//+------------------------------------------------------------------+
void Config_Apply(const string section, const string key, const string value, FiloConfig &cfg) {
   string s = section; StringToLower(s);
   string k = key;     StringToLower(k);

   if(s == "symbols") {
      if(k == "primary")       cfg.sym_primary = value;
      else if(k == "silver_proxy") cfg.sym_silver = value;
      else if(k == "dxy_proxy")    cfg.sym_dxy = value;
   }
   else if(s == "instrument") {
      if(k == "pip_in_points")         cfg.pip_in_points = StringToDouble(value);
      else if(k == "usd_per_point_per_lot") cfg.usd_per_point_per_lot = StringToDouble(value);
   }
   else if(s == "sizing") {
      if(k == "fixed_lot")             cfg.fixed_lot = StringToDouble(value);
      else if(k == "max_position_lots") cfg.max_position_lots = StringToDouble(value);
      else if(k == "max_concurrent_trades") cfg.max_concurrent_trades = (int)StringToInteger(value);
   }
   else if(s == "keylevels") {
      if(k == "lookback_days")               cfg.kl_lookback_days = (int)StringToInteger(value);
      else if(k == "zone_half_width_pips")   cfg.kl_zone_half_width_pips = StringToDouble(value);
      else if(k == "weight_d1")              cfg.kl_weight_d1 = StringToDouble(value);
      else if(k == "weight_h4")              cfg.kl_weight_h4 = StringToDouble(value);
      else if(k == "weight_h1")              cfg.kl_weight_h1 = StringToDouble(value);
      else if(k == "weight_m30")             cfg.kl_weight_m30 = StringToDouble(value);
      else if(k == "weight_m15")             cfg.kl_weight_m15 = StringToDouble(value);
      else if(k == "weight_m5")              cfg.kl_weight_m5 = StringToDouble(value);
      else if(k == "merge_overlap_pips")     cfg.kl_merge_overlap_pips = StringToDouble(value);
      else if(k == "max_setup_distance_pips") cfg.kl_max_setup_distance_pips = StringToDouble(value);
      else if(k == "min_kl_weight")          cfg.kl_min_weight = StringToDouble(value);
      else if(k == "slice_commitment_mult")  cfg.kl_slice_commitment_mult = StringToDouble(value);
      else if(k == "slice_cooldown_ms")      cfg.kl_slice_cooldown_ms = (int)StringToInteger(value);
      else if(k == "slice_floor_factor")     cfg.kl_slice_floor_factor = StringToDouble(value);
   }
   else if(s == "channel") {
      if(k == "length")                 cfg.chan_length = (int)StringToInteger(value);
      else if(k == "stdev_mult")        cfg.chan_stdev_mult = StringToDouble(value);
      else if(k == "extreme_block_mult") cfg.chan_extreme_block_mult = StringToDouble(value);
   }
   else if(s == "acceleration") {
      if(k == "window_ms")              cfg.accel_window_ms = (int)StringToInteger(value);
      else if(k == "velocity_ema_ms")   cfg.accel_velocity_ema_ms = (int)StringToInteger(value);
      else if(k == "burst_sigma")       cfg.accel_burst_sigma = StringToDouble(value);
      else if(k == "min_bursts")        cfg.accel_min_bursts = (int)StringToInteger(value);
      else if(k == "liquidity_grab_reject_ms") cfg.accel_liquidity_grab_reject_ms = (int)StringToInteger(value);
   }
   else if(s == "trend") {
      if(k == "lookback_bars_m1")       cfg.trend_lookback_bars_m1 = (int)StringToInteger(value);
      else if(k == "persistence_min_ratio") cfg.trend_persistence_min_ratio = StringToDouble(value);
   }
   else if(s == "correlation") {
      if(k == "lookback_seconds")       cfg.corr_lookback_seconds = (int)StringToInteger(value);
      else if(k == "min_agreement")     cfg.corr_min_agreement = StringToDouble(value);
      else if(k == "enable_xagusd")     cfg.corr_enable_xagusd = ParseBool(value);
      else if(k == "enable_dxy")        cfg.corr_enable_dxy = ParseBool(value);
      else if(k == "dxy_inverse")       cfg.corr_dxy_inverse = ParseBool(value);
   }
   else if(s == "setups") {
      if(k == "enable_breakout")        cfg.setup_breakout = ParseBool(value);
      else if(k == "enable_retest")     cfg.setup_retest = ParseBool(value);
      else if(k == "enable_rejection")  cfg.setup_rejection = ParseBool(value);
      else if(k == "enable_fakeout")    cfg.setup_fakeout = ParseBool(value);
      else if(k == "enable_failed_auction") cfg.setup_failed_auction = ParseBool(value);
      else if(k == "enable_engulfing_accel") cfg.setup_engulfing_accel = ParseBool(value);
      else if(k == "enable_mean_reversion")  cfg.setup_mean_reversion = ParseBool(value);
      else if(k == "min_setup_score")   cfg.setup_min_score = StringToDouble(value);
      else if(k == "exhaustion_bars")   cfg.setup_exhaustion_bars = (int)StringToInteger(value);
      else if(k == "exhaustion_pips")   cfg.setup_exhaustion_pips = StringToDouble(value);
      else if(k == "counter_trend_vel_veto_pips_per_sec")
                                        cfg.setup_counter_trend_vel_veto_pips_per_sec = StringToDouble(value);
      else if(k == "counter_trend_trend_veto")
                                        cfg.setup_counter_trend_trend_veto = ParseBool(value);
      else if(k == "htf_min_clearance_pips" || k == "d1_min_clearance_pips")
                                        cfg.setup_htf_min_clearance_pips = StringToDouble(value);
      else if(k == "htf_compression_max_width_pips")
                                        cfg.setup_htf_compression_max_width_pips = StringToDouble(value);
      else if(k == "continuation_extreme_block_mult")
                                        cfg.setup_continuation_extreme_block_mult = StringToDouble(value);
      else if(k == "rev_confirm_enable")
                                        cfg.setup_rev_confirm_enable = ParseBool(value);
      else if(k == "rev_confirm_min_layers")
                                        cfg.setup_rev_confirm_min_layers = (int)StringToInteger(value);
      else if(k == "rev_confirm_partner_enable")
                                        cfg.setup_rev_confirm_partner_enable = ParseBool(value);
      else if(k == "rev_confirm_partner_long_ag_min")
                                        cfg.setup_rev_confirm_partner_long_ag_min = StringToDouble(value);
      else if(k == "rev_confirm_partner_short_ag_max")
                                        cfg.setup_rev_confirm_partner_short_ag_max = StringToDouble(value);
      else if(k == "rev_confirm_burst_decay_enable")
                                        cfg.setup_rev_confirm_burst_decay_enable = ParseBool(value);
      else if(k == "rev_confirm_burst_decay_min_ms")
                                        cfg.setup_rev_confirm_burst_decay_min_ms = (int)StringToInteger(value);
      else if(k == "rev_confirm_burst_decay_max_ms")
                                        cfg.setup_rev_confirm_burst_decay_max_ms = (int)StringToInteger(value);
      else if(k == "rev_confirm_wick_enable")
                                        cfg.setup_rev_confirm_wick_enable = ParseBool(value);
      else if(k == "rev_confirm_wick_min_pips")
                                        cfg.setup_rev_confirm_wick_min_pips = StringToDouble(value);
      else if(k == "rev_confirm_struct_enable")
                                        cfg.setup_rev_confirm_struct_enable = ParseBool(value);
      else if(k == "rev_confirm_struct_range_pips")
                                        cfg.setup_rev_confirm_struct_range_pips = StringToDouble(value);
      else if(k == "rev_confirm_struct_adverse_veto")
                                        cfg.setup_rev_confirm_struct_adverse_veto = ParseBool(value);
      else if(k == "adverse_bar_enable")
                                        cfg.setup_adverse_bar_enable = ParseBool(value);
      else if(k == "adverse_bar_body_pips")
                                        cfg.setup_adverse_bar_body_pips = StringToDouble(value);
      else if(k == "adverse_forming_enable")
                                        cfg.setup_adverse_forming_enable = ParseBool(value);
      else if(k == "adverse_forming_progress_pips")
                                        cfg.setup_adverse_forming_progress_pips = StringToDouble(value);
      else if(k == "adverse_forming_range_pips")
                                        cfg.setup_adverse_forming_range_pips = StringToDouble(value);
      else if(k == "echo_enable")      cfg.setup_echo_enable = ParseBool(value);
      else if(k == "echo_window_ms")   cfg.setup_echo_window_ms = (int)StringToInteger(value);
      else if(k == "enable_momentum_cont")
                                        cfg.setup_momentum_cont = ParseBool(value);
      else if(k == "momentum_cont_min_move_pips")
                                        cfg.setup_momentum_cont_min_move_pips = StringToDouble(value);
      else if(k == "momentum_cont_max_move_pips")
                                        cfg.setup_momentum_cont_max_move_pips = StringToDouble(value);
      else if(k == "momentum_cont_persistence_ms")
                                        cfg.setup_momentum_cont_persistence_ms = (int)StringToInteger(value);
      else if(k == "momentum_cont_persist_min_delta_pips")
                                        cfg.setup_momentum_cont_persist_min_delta_pips = StringToDouble(value);
      else if(k == "momentum_cont_window_bars")
                                        cfg.setup_momentum_cont_window_bars = (int)StringToInteger(value);
      else if(k == "momentum_cont_min_pullback_pips")
                                        cfg.setup_momentum_cont_min_pullback_pips = StringToDouble(value);
      else if(k == "momentum_cont_max_pullback_pips")
                                        cfg.setup_momentum_cont_max_pullback_pips = StringToDouble(value);
      else if(k == "momentum_cont_partner_min_ag")
                                        cfg.setup_momentum_cont_partner_min_ag = StringToDouble(value);
      else if(k == "momentum_cont_kl_veto_pips")
                                        cfg.setup_momentum_cont_kl_veto_pips = StringToDouble(value);
      else if(k == "momentum_cont_lot_mult")
                                        cfg.setup_momentum_cont_lot_mult = StringToDouble(value);
      else if(k == "momentum_cont_stop_pips")
                                        cfg.setup_momentum_cont_stop_pips = StringToDouble(value);
      else if(k == "momentum_cont_target_pips")
                                        cfg.setup_momentum_cont_target_pips = StringToDouble(value);
   }
   else if(s == "daily_bias") {
      if(k == "enable")                      cfg.daily_bias_enable = ParseBool(value);
      else if(k == "half_life_ms")           cfg.daily_bias_half_life_ms = (int)StringToInteger(value);
      else if(k == "max_age_ms")             cfg.daily_bias_max_age_ms = (int)StringToInteger(value);
      else if(k == "break_tolerance_pips")   cfg.daily_bias_break_tolerance_pips = StringToDouble(value);
      else if(k == "bounce_min_wick_pips")   cfg.daily_bias_bounce_min_wick_pips = StringToDouble(value);
      else if(k == "score_weight")           cfg.daily_bias_score_weight = StringToDouble(value);
      else if(k == "nearby_d1_compound_pips") cfg.daily_bias_nearby_d1_compound_pips = StringToDouble(value);
   }
   else if(s == "execution") {
      if(k == "intent_ttl_ms")          cfg.exec_intent_ttl_ms = (int)StringToInteger(value);
      else if(k == "trigger_tolerance_pips") cfg.exec_trigger_tolerance_pips = StringToDouble(value);
      else if(k == "max_slippage_pips") cfg.exec_max_slippage_pips = StringToDouble(value);
      else if(k == "max_spread_breakout") cfg.exec_max_spread_breakout = StringToDouble(value);
      else if(k == "max_spread_reversion") cfg.exec_max_spread_reversion = StringToDouble(value);
      else if(k == "max_spread_default")   cfg.exec_max_spread_default = StringToDouble(value);
      else if(k == "knife_size_threshold_pips") cfg.exec_knife_size_threshold_pips = StringToDouble(value);
      else if(k == "knife_size_mult")    cfg.exec_knife_size_mult = StringToDouble(value);
   }
   else if(s == "trade_management") {
      if(k == "be_trigger_pips")        cfg.tm_be_trigger_pips = StringToDouble(value);
      else if(k == "be_buffer_pips")    cfg.tm_be_buffer_pips = StringToDouble(value);
      else if(k == "trail_start_pips")  cfg.tm_trail_start_pips = StringToDouble(value);
      else if(k == "trail_distance_pips") cfg.tm_trail_distance_pips = StringToDouble(value);
      else if(k == "management_poll_ms") cfg.tm_management_poll_ms = (int)StringToInteger(value);
      else if(k == "target_breakout_pips") cfg.tm_target_breakout_pips = StringToDouble(value);
      else if(k == "target_reversion_pips") cfg.tm_target_reversion_pips = StringToDouble(value);
      else if(k == "target_rejection_pips") cfg.tm_target_rejection_pips = StringToDouble(value);
      else if(k == "stop_breakout_pips") cfg.tm_stop_breakout_pips = StringToDouble(value);
      else if(k == "stop_reversion_pips") cfg.tm_stop_reversion_pips = StringToDouble(value);
      else if(k == "stop_rejection_pips") cfg.tm_stop_rejection_pips = StringToDouble(value);
      else if(k == "stack_min_gap_ms")   cfg.tm_stack_min_gap_ms = (int)StringToInteger(value);
      else if(k == "stack_require_diverse_edge")
                                         cfg.tm_stack_require_diverse_edge = ParseBool(value);
      else if(k == "post_fill_cooldown_ms")
                                         cfg.tm_post_fill_cooldown_ms = (int)StringToInteger(value);
      else if(k == "post_stop_opposite_cooldown_ms")
                                         cfg.tm_post_stop_opposite_cooldown_ms = (int)StringToInteger(value);
   }
   else if(s == "risk") {
      if(k == "max_loss_per_trade_usd")   cfg.risk_max_loss_per_trade_usd = StringToDouble(value);
      else if(k == "max_loss_per_minute_usd") cfg.risk_max_loss_per_minute_usd = StringToDouble(value);
      else if(k == "max_loss_per_hour_usd")   cfg.risk_max_loss_per_hour_usd = StringToDouble(value);
      else if(k == "max_loss_per_day_usd")    cfg.risk_max_loss_per_day_usd = StringToDouble(value);
      else if(k == "max_order_rejects")       cfg.risk_max_order_rejects = (int)StringToInteger(value);
      else if(k == "post_loss_cooldown_ms")    cfg.risk_post_loss_cooldown_ms = (int)StringToInteger(value);
   }
   else if(s == "safety") {
      if(k == "abnormal_spread_pips")   cfg.safety_abnormal_spread_pips = StringToDouble(value);
      else if(k == "stall_timeout_ms")  cfg.safety_stall_timeout_ms = (int)StringToInteger(value);
      else if(k == "latency_alert_ms")  cfg.safety_latency_alert_ms = (int)StringToInteger(value);
      else if(k == "max_slippage_pips") cfg.safety_max_slippage_pips = StringToDouble(value);
      else if(k == "halt_seconds")      cfg.safety_halt_seconds = (int)StringToInteger(value);
   }
   else if(s == "sessions") {
      if(k == "trade_asia")             cfg.sess_trade_asia = ParseBool(value);
      else if(k == "trade_london")      cfg.sess_trade_london = ParseBool(value);
      else if(k == "trade_ny")          cfg.sess_trade_ny = ParseBool(value);
      else if(k == "trade_overlap_ln_ny") cfg.sess_trade_overlap_ln_ny = ParseBool(value);
      else if(k == "trade_offhours")    cfg.sess_trade_offhours = ParseBool(value);
   }
   else if(s == "ml") {
      if(k == "enable")                 cfg.ml_enable = ParseBool(value);
      else if(k == "min_probability")   cfg.ml_min_probability = StringToDouble(value);
      else if(k == "model_path")        cfg.ml_model_path = value;
      else if(k == "blend_weight")      cfg.ml_blend_weight = StringToDouble(value);
   }
   else if(s == "telemetry") {
      if(k == "enable")                 cfg.tel_enable = ParseBool(value);
      else if(k == "output_dir")        cfg.tel_output_dir = value;
      else if(k == "flush_interval_ms") cfg.tel_flush_interval_ms = (int)StringToInteger(value);
      else if(k == "log_ticks")         cfg.tel_log_ticks = ParseBool(value);
      else if(k == "log_features")      cfg.tel_log_features = ParseBool(value);
      else if(k == "log_keylevels")     cfg.tel_log_keylevels = ParseBool(value);
      else if(k == "log_setups")        cfg.tel_log_setups = ParseBool(value);
      else if(k == "log_intents")       cfg.tel_log_intents = ParseBool(value);
      else if(k == "log_fills")         cfg.tel_log_fills = ParseBool(value);
      else if(k == "log_trades")        cfg.tel_log_trades = ParseBool(value);
      else if(k == "log_risk")          cfg.tel_log_risk = ParseBool(value);
      else if(k == "log_safety")        cfg.tel_log_safety = ParseBool(value);
      else if(k == "log_latency")       cfg.tel_log_latency = ParseBool(value);
      else if(k == "log_errors")        cfg.tel_log_errors = ParseBool(value);
      else if(k == "log_shadow")        cfg.tel_log_shadow = ParseBool(value);
      else if(k == "log_tmlab")         cfg.tel_log_tmlab  = ParseBool(value);
   }
}

//+------------------------------------------------------------------+
//|  Load the ini file. Path is relative to MT5 common Files folder  |
//|  (FILE_COMMON) so Python tooling can share it. If loading fails  |
//|  we log and fall back to defaults — startup still succeeds.      |
//+------------------------------------------------------------------+
bool Config_Load(const string ini_path, FiloConfig &cfg) {
   Config_SetDefaults(cfg);

   int h = FileOpen(ini_path, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE) {
      // try sandbox (per-terminal Files folder) as fallback
      h = FileOpen(ini_path, FILE_READ | FILE_TXT | FILE_ANSI);
   }
   if(h == INVALID_HANDLE) {
      PrintFormat("[Config] could not open %s — using defaults", ini_path);
      return false;
   }

   string section = "";
   while(!FileIsEnding(h)) {
      string line = FileReadString(h);
      string k, v;
      if(Config_ParseLine(line, section, k, v)) {
         if(StringLen(k) > 0) Config_Apply(section, k, v, cfg);
      }
   }
   FileClose(h);
   PrintFormat("[Config] loaded %s", ini_path);
   return true;
}

//+------------------------------------------------------------------+
//|  Live overrides — a flat JSON with dotted keys (`section.key`).  |
//|  Parsing is minimal (JSONObjectKey/Value discovery via string    |
//|  scan) because we only support number/bool/string leaves.        |
//+------------------------------------------------------------------+
bool Config_ReloadOverrides(const string overrides_path, FiloConfig &cfg) {
   int h = FileOpen(overrides_path, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE)
      h = FileOpen(overrides_path, FILE_READ | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE) return false;

   string content = "";
   while(!FileIsEnding(h)) content += FileReadString(h) + "\n";
   FileClose(h);

   // nothing to do if content hasn't changed — saves a reparse and lets the
   // caller distinguish "no-op poll" from "user edited the file"
   if(content == g_last_overrides_content) return false;
   g_last_overrides_content = content;

   // crude "k":v scanner — good enough for a flat overrides file
   int pos = 0;
   while(pos < StringLen(content)) {
      int q1 = StringFind(content, "\"", pos);
      if(q1 < 0) break;
      int q2 = StringFind(content, "\"", q1 + 1);
      if(q2 < 0) break;
      string dotted = StringSubstr(content, q1 + 1, q2 - q1 - 1);
      int colon = StringFind(content, ":", q2);
      if(colon < 0) break;
      // find end of value: comma, newline, or closing brace
      int endp = colon + 1;
      while(endp < StringLen(content)) {
         ushort c = StringGetCharacter(content, endp);
         if(c == ',' || c == '\n' || c == '}' || c == '\r') break;
         endp++;
      }
      string raw = StringSubstr(content, colon + 1, endp - colon - 1);
      StringTrimLeft(raw); StringTrimRight(raw);
      // strip quotes around string values
      if(StringLen(raw) > 0 && StringGetCharacter(raw, 0) == '"') {
         raw = StringSubstr(raw, 1, StringLen(raw) - 2);
      }
      // dotted "section.key"
      int dot = StringFind(dotted, ".");
      if(dot > 0 && StringFind(dotted, "_") != 0) {
         string sec = StringSubstr(dotted, 0, dot);
         string key = StringSubstr(dotted, dot + 1);
         Config_Apply(sec, key, raw, cfg);
      }
      pos = endp + 1;
   }
   return true;
}

#endif // __FILOHFT_CONFIG_MQH__
