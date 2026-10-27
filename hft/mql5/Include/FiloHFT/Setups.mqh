//+------------------------------------------------------------------+
//|  FiloHFT/Setups.mqh                                              |
//|  Setup detection engine. Runs on every tick, scans the KL map    |
//|  against the current feature snapshot, classifies setups, and   |
//|  scores them. Each accepted setup produces an ExecutionIntent   |
//|  by calling ExecutionIntent::Submit(...).                       |
//|                                                                 |
//|  Every candidate — accepted or rejected — is logged to          |
//|  telemetry/setups.csv with a block_reason if rejected, so       |
//|  "why didn't it trade?" is always answerable after the fact.   |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_SETUPS_MQH__
#define __FILOHFT_SETUPS_MQH__

#include <FiloHFT/Common.mqh>
#include <FiloHFT/Config.mqh>
#include <FiloHFT/Telemetry.mqh>
#include <FiloHFT/KeyLevels.mqh>
#include <FiloHFT/Features.mqh>
#include <FiloHFT/Correlation.mqh>
#include <FiloHFT/DailyBias.mqh>
#include <FiloHFT/ExecutionIntent.mqh>
#include <FiloHFT/Risk.mqh>
#include <FiloHFT/Safety.mqh>
#include <FiloHFT/TradeManagement.mqh>
#include <FiloHFT/Execution.mqh>
#include <FiloHFT/Shadow.mqh>

struct SetupCandidate {
   ENUM_SETUP_FAMILY family;
   ENUM_TRADE_DIR    direction;
   KLZone            zone;
   double            kl_distance_pips;
   double            intrinsic_score;
   double            ml_score;
   string            block_reason;
   bool              accepted;
   // Post-accept routing decision. Possible values:
   //   ""                         — candidate was blocked before accept (see block_reason)
   //   "live"                     — submitted to the real intent queue
   //   "shadow_only:<reason>"     — accepted + shadow-tracked, but not routed
   //                                to live execution (InpEnableTrading off,
   //                                risk/safety halt, post-fill cooldown,
   //                                anti-pyramid stack gate, etc.)
   // Dashboards should filter `accepted="accepted" && route_status="live"`
   // to see only the rows that are actually going to become live orders.
   string            route_status;
   // Echo tag — how many prior ACCEPTED candidates with the same direction
   // and same KL zone_id fired within the echo window. 0 = lone signal;
   // 1 = one prior reconfirmation; 2+ = multiple. Set at _Emit time from
   // the ring buffer. The tag is logged but does NOT affect scoring yet —
   // validation happens downstream by splitting shadow-trade outcomes on
   // echo_count. See feedback memory "Always mention echo-tag data...".
   int               echo_count;
};

//--- Circular buffer of recently-accepted candidates, used to compute
//    echo_count. Fixed compile-time size — cheap and tidy.
#define FILO_ECHO_RING 16
struct EchoEntry {
   long            ts_ms;
   string          zone_id;
   ENUM_TRADE_DIR  direction;
};

class Setups {
private:
   //--- short-term cache of last-candidate timestamp per family, to
   //    avoid spamming the same setup on every tick within a window.
   //    Sized to 16 (comfortably > max ENUM_SETUP_FAMILY value) so new
   //    families can slot in without resizing.
   long m_last_emit[16];    // indexed by ENUM_SETUP_FAMILY
   int  m_cooldown_ms;

   //--- echo ring: last N accepted candidates, circular.
   EchoEntry m_echo[FILO_ECHO_RING];
   int       m_echo_head;

   bool _CoolingDown(const ENUM_SETUP_FAMILY f, const long now) const {
      return (now - m_last_emit[(int)f]) < m_cooldown_ms;
   }

   //--- Count prior accepted candidates in the ring that match this
   //    (direction, zone_id) within echo_window_ms. Called at _Emit time.
   int _ComputeEchoCount(const ENUM_TRADE_DIR dir, const string zone_id,
                         const long now_ms) const {
      if(!g_cfg.setup_echo_enable) return 0;
      const long cutoff = now_ms - (long)g_cfg.setup_echo_window_ms;
      int n = 0;
      for(int i = 0; i < FILO_ECHO_RING; i++) {
         if(m_echo[i].ts_ms <= 0)         continue;
         if(m_echo[i].ts_ms < cutoff)     continue;
         if(m_echo[i].direction != dir)   continue;
         if(m_echo[i].zone_id  != zone_id) continue;
         n++;
      }
      return n;
   }

   //--- Record an accepted candidate into the ring so future candidates
   //    can detect it as a prior echo. Only called for accepted rows.
   void _PushEcho(const SetupCandidate &c, const FeatureSnapshot &fs) {
      m_echo[m_echo_head].ts_ms     = fs.ts_ms;
      m_echo[m_echo_head].zone_id   = c.zone.id;
      m_echo[m_echo_head].direction = c.direction;
      m_echo_head = (m_echo_head + 1) % FILO_ECHO_RING;
   }

   void _Emit(const SetupCandidate &c, const FeatureSnapshot &fs) {
      if(!g_cfg.tel_log_setups) return;
      CorrState cs = g_corr.State();
      g_tel.WriteSetup(
         fs.ts_ms,
         SetupFamilyName(c.family),
         DirName(c.direction),
         c.zone.id,
         c.zone.weight,
         c.kl_distance_pips,
         c.intrinsic_score,
         c.ml_score,
         c.accepted ? "accepted" : "blocked",
         c.block_reason,
         fs.chan_pos_norm,
         fs.spread_pips,
         fs.velocity_pips_per_sec,
         cs.agreement_xagusd,
         cs.agreement_dxy,
         c.route_status,
         c.echo_count);
   }

   //--- common gates applied to every family before the family-specific
   //    logic runs. Returns true if the candidate survives gating.
   //
   //    NOTE: the execution spread gate is intentionally NOT applied here.
   //    Setup detection and shadow-trade bookkeeping should run regardless
   //    of current spread — ML needs every signal, not just the subset
   //    where spread happened to be inside our execution threshold.
   //    Execution.mqh re-applies the spread gate at trigger time so real
   //    orders stay protected. Likewise Safety/Risk only gate the real
   //    submission path (allow_entry in FiloHFT.mq5), not this scan.
   bool _CommonGates(SetupCandidate &c, const FeatureSnapshot &fs) {
      // min KL weight — momentum_cont is exempt because its thesis is
      // structural (pullback in strong move), not zone-based. A nearby
      // KL is informational but never required.
      //
      // Uses EFFECTIVE weight: a zone recently sliced against the
      // trade direction (e.g. price committed above a ceiling we want
      // to short off) is penalised on a linear curve for the
      // kl_slice_cooldown_ms window. See KeyLevels::EffectiveWeight.
      if(c.family != SETUP_MOMENTUM_CONT) {
         int dir_sign = (c.direction == DIR_LONG) ? +1 : -1;
         double eff_w = g_kl.EffectiveWeight(c.zone, dir_sign, fs.ts_ms);
         if(eff_w < g_cfg.kl_min_weight) {
            c.block_reason = StringFormat("kl eff_weight %.2f < %.2f (raw %.2f)",
                                          eff_w, g_cfg.kl_min_weight, c.zone.weight);
            return false;
         }
      }

      // Adverse-bar gates — universal (all families, both directions).
      // These refuse any entry that would be trading INTO a violently
      // contrary M1 move. Two complementary checks:
      //   (1) shift-1 adverse body: the last CLOSED M1 bar travelled
      //       >= adverse_bar_body_pips in the OPPOSITE direction of
      //       the intended trade. Natural memory window is one M1 bar
      //       (~60s) — no extra state, just iOpen/iClose(shift=1).
      //   (2) shift-0 adverse forming: the currently-forming M1 bar
      //       has already progressed >= adverse_forming_progress_pips
      //       from its open AGAINST the intended trade AND the bar
      //       range is >= adverse_forming_range_pips. This catches the
      //       "don't short into the giant green bar WHILE it's still
      //       building" case before waiting for the close.
      if(g_cfg.setup_adverse_bar_enable) {
         double o1 = iOpen (_Symbol, PERIOD_M1, 1);
         double c1 = iClose(_Symbol, PERIOD_M1, 1);
         if(o1 > 0 && c1 > 0) {
            double body_pips = PipsFromPrice(c1 - o1); // signed: +up, -down
            double thr = g_cfg.setup_adverse_bar_body_pips;
            if(c.direction == DIR_LONG && body_pips <= -thr) {
               c.block_reason = StringFormat("adverse_bar_body %.1fp (shift1 down) >= %.1f",
                                             -body_pips, thr);
               return false;
            }
            if(c.direction == DIR_SHORT && body_pips >= thr) {
               c.block_reason = StringFormat("adverse_bar_body %.1fp (shift1 up) >= %.1f",
                                             body_pips, thr);
               return false;
            }
         }
      }
      // momentum_cont entries are LITERALLY buying the pullback of a big
      // trending bar — the forming bar is SUPPOSED to have travelled
      // against us briefly before we enter. Exempt it from adverse_forming,
      // otherwise the family can never fire. (shift-1 adverse_bar above
      // still applies — a large closed bar against us is always a veto.)
      if(g_cfg.setup_adverse_forming_enable && c.family != SETUP_MOMENTUM_CONT) {
         double o0 = iOpen(_Symbol, PERIOD_M1, 0);
         double h0 = iHigh(_Symbol, PERIOD_M1, 0);
         double l0 = iLow (_Symbol, PERIOD_M1, 0);
         if(o0 > 0 && h0 > 0 && l0 > 0) {
            double prog_thr  = g_cfg.setup_adverse_forming_progress_pips;
            double range_thr = g_cfg.setup_adverse_forming_range_pips;
            double range_p   = PipsFromPrice(h0 - l0);
            if(c.direction == DIR_SHORT) {
               double adv = PipsFromPrice(h0 - o0); // how far up from open
               if(adv >= prog_thr && range_p >= range_thr) {
                  c.block_reason = StringFormat("adverse_forming up %.1fp range %.1fp",
                                                adv, range_p);
                  return false;
               }
            } else if(c.direction == DIR_LONG) {
               double adv = PipsFromPrice(o0 - l0); // how far down from open
               if(adv >= prog_thr && range_p >= range_thr) {
                  c.block_reason = StringFormat("adverse_forming down %.1fp range %.1fp",
                                                adv, range_p);
                  return false;
               }
            }
         }
      }

      // Higher-TF clearance gate — "no buys right under a D1/H4 ceiling,
      // no sells right above a D1/H4 floor". The trade must have at least
      // enough runway to cover the BE trigger distance; otherwise the
      // hidden SL trips before BE can lock in. Extended from D1-only to
      // D1+H4 because H4 levels on XAUUSD react just as strongly during
      // normal sessions — "less than 6 pips from a D1" is the same
      // situation as "less than 6 pips from an H4" in terms of the
      // bounce risk that ate our break-even.
      if(g_cfg.setup_htf_min_clearance_pips > 0.0) {
         int dir_sign = (c.direction == DIR_LONG) ? +1 : -1;
         double clr = g_kl.NearestHigherTfObstaclePips(fs.mid, dir_sign);
         if(clr < g_cfg.setup_htf_min_clearance_pips) {
            c.block_reason = StringFormat("htf clearance %.1f < %.1f pips",
                                          clr, g_cfg.setup_htf_min_clearance_pips);
            return false;
         }
      }

      // at-zone proximity gate — rejection/retest/fakeout/mean_reversion
      // have a "price is AT the zone" thesis and must not fire from the
      // broader search radius. Breakout and failed_auction skip this:
      // breakout is inherently beyond the boundary, failed_auction is
      // channel-based with KL attached only for telemetry.
      if(c.family == SETUP_REJECTION || c.family == SETUP_RETEST ||
         c.family == SETUP_FAKEOUT    || c.family == SETUP_MEAN_REVERSION) {
         double zone_pips = PipsFromPrice(c.zone.half_width);
         double at_zone   = MathMax(zone_pips * 1.5, 3.0);
         if(c.kl_distance_pips > at_zone) {
            c.block_reason = StringFormat("not at zone (%.1f > %.1f pips)",
                                          c.kl_distance_pips, at_zone);
            return false;
         }
      }

      // channel extreme block for CONTINUATION setups (breakout/retest/engulfing).
      // These families are INHERENTLY near the extreme — the thesis is "price
      // has broken out of the envelope" — so we use a looser threshold
      // (setup_continuation_extreme_block_mult) than the generic reversal
      // extreme gate. Still hard-blocks when price is genuinely extended
      // (e.g. >= 1.5 stdev from channel mid).
      if(c.family == SETUP_BREAKOUT || c.family == SETUP_RETEST ||
         c.family == SETUP_ENGULFING_ACCEL) {
         double cp = fs.chan_pos_norm;
         double thr = g_cfg.setup_continuation_extreme_block_mult;
         if(thr <= 0.0) thr = g_cfg.chan_extreme_block_mult;  // fallback
         bool beyond = (c.direction == DIR_LONG  && cp >=  thr) ||
                       (c.direction == DIR_SHORT && cp <= -thr);
         if(beyond) {
            c.block_reason = StringFormat("at channel extreme %.2f (continuation blocked, thr=%.2f)",
                                          cp, thr);
            return false;
         }
      }

      // Counter-trend vetoes (B + C).
      //
      // mean_reversion / rejection / fakeout / failed_auction are the
      // families that try to fade a move. Each of them can happily fire
      // LONG into a strong down-leg (or SHORT into a rally) — that's the
      // exact 4-consecutive-losing-buys pattern at 2026-04-13 20:05-20:08.
      // We add two narrow blocks that only apply to these families:
      //   (B) velocity veto — price is accelerating hard against the
      //       setup direction. Tuneable via
      //       setup_counter_trend_vel_veto_pips_per_sec (default 3.0).
      //   (C) trend-state veto — the M1 regression trend is actively
      //       pointing the other way. Toggled via
      //       setup_counter_trend_trend_veto (default on).
      // Continuation families (breakout/retest/engulfing) are unaffected;
      // they are already gated by AtChannelExtreme above.
      bool is_counter_trend =
         (c.family == SETUP_MEAN_REVERSION ||
          c.family == SETUP_REJECTION      ||
          c.family == SETUP_FAKEOUT        ||
          c.family == SETUP_FAILED_AUCTION);
      if(is_counter_trend) {
         double v_thr = g_cfg.setup_counter_trend_vel_veto_pips_per_sec;
         if(v_thr > 0.0) {
            if(c.direction == DIR_LONG && fs.velocity_pips_per_sec < -v_thr) {
               c.block_reason = StringFormat("counter-trend veto vel=%.2f < -%.2f",
                                             fs.velocity_pips_per_sec, v_thr);
               return false;
            }
            if(c.direction == DIR_SHORT && fs.velocity_pips_per_sec > v_thr) {
               c.block_reason = StringFormat("counter-trend veto vel=%.2f > %.2f",
                                             fs.velocity_pips_per_sec, v_thr);
               return false;
            }
         }
         if(g_cfg.setup_counter_trend_trend_veto) {
            if(c.direction == DIR_LONG  && fs.trend_state == TREND_DOWN) {
               c.block_reason = "counter-trend veto trend=down";
               return false;
            }
            if(c.direction == DIR_SHORT && fs.trend_state == TREND_UP) {
               c.block_reason = "counter-trend veto trend=up";
               return false;
            }
         }

         // Reversal confirmation stack — N-of-4 positive confirmations
         // required before fading a move. Each layer is independently
         // toggleable for A/B. Layer 4 (M30/M15 structural bias) can
         // also hard-veto on ADVERSE structure regardless of the other
         // layers, so it runs first and short-circuits.
         if(g_cfg.setup_rev_confirm_enable) {
            int    conf  = 0;
            string layers = "";

            if(g_cfg.setup_rev_confirm_struct_enable) {
               int bias = g_kl.MidTfBias(fs.mid,
                            g_cfg.setup_rev_confirm_struct_range_pips);
               bool supportive = (c.direction == DIR_LONG  && bias == +1) ||
                                 (c.direction == DIR_SHORT && bias == -1);
               bool adverse    = (c.direction == DIR_LONG  && bias == -1) ||
                                 (c.direction == DIR_SHORT && bias == +1);
               if(adverse && g_cfg.setup_rev_confirm_struct_adverse_veto) {
                  c.block_reason = "rev_confirm struct adverse (m30/m15 against)";
                  return false;
               }
               if(supportive) { conf++; layers += "struct,"; }
            }

            if(g_cfg.setup_rev_confirm_partner_enable &&
               g_corr.PartnerDivergenceFor(c.direction)) {
               conf++; layers += "partner,";
            }

            if(g_cfg.setup_rev_confirm_burst_decay_enable &&
               g_features.BurstDecayAgainst(c.direction, fs.ts_ms,
                  g_cfg.setup_rev_confirm_burst_decay_min_ms,
                  g_cfg.setup_rev_confirm_burst_decay_max_ms)) {
               conf++; layers += "burst_decay,";
            }

            if(g_cfg.setup_rev_confirm_wick_enable) {
               // Forming M1 bar (shift 0) — the sub-second intra-bar read.
               // For a LONG, current mid should be at least min_pips above
               // the forming bar's low (bid rebuild after a sweep down).
               double h0 = iHigh(_Symbol, PERIOD_M1, 0);
               double l0 = iLow (_Symbol, PERIOD_M1, 0);
               double min_w = g_cfg.setup_rev_confirm_wick_min_pips;
               bool wick_ok = false;
               if(h0 > 0 && l0 > 0) {
                  if(c.direction == DIR_LONG)
                     wick_ok = PipsFromPrice(fs.mid - l0) >= min_w;
                  else if(c.direction == DIR_SHORT)
                     wick_ok = PipsFromPrice(h0 - fs.mid) >= min_w;
               }
               if(wick_ok) { conf++; layers += "wick,"; }
            }

            if(conf < g_cfg.setup_rev_confirm_min_layers) {
               c.block_reason = StringFormat("rev_confirm %d/%d [%s]",
                                             conf,
                                             g_cfg.setup_rev_confirm_min_layers,
                                             layers);
               return false;
            }
         }
      }

      // correlation agreement
      string reason = "";
      if(!g_corr.DirectionalOk(c.direction, reason)) {
         c.block_reason = reason;
         return false;
      }

      // exhaustion — stop buying into an up-move after N bars of
      // consecutive same-direction M1 candles or > N pips of movement.
      int bars = 0;
      double pips_moved = 0.0;
      double prev_close = fs.mid;
      for(int i = 1; i <= g_cfg.setup_exhaustion_bars && i <= iBars(_Symbol, PERIOD_M1) - 1; i++) {
         double o = iOpen (_Symbol, PERIOD_M1, i);
         double cl= iClose(_Symbol, PERIOD_M1, i);
         if(c.direction == DIR_LONG  && cl > o) bars++;
         else if(c.direction == DIR_SHORT && cl < o) bars++;
         else break;
         pips_moved += PipsFromPrice(MathAbs(cl - o));
         prev_close = cl;
      }
      if(bars >= g_cfg.setup_exhaustion_bars ||
         pips_moved >= g_cfg.setup_exhaustion_pips) {
         if(c.family == SETUP_BREAKOUT || c.family == SETUP_RETEST ||
            c.family == SETUP_ENGULFING_ACCEL) {
            c.block_reason = StringFormat("exhaustion %d bars / %.1f pips", bars, pips_moved);
            return false;
         }
      }

      return true;
   }

   //------------------------------------------------------------------
   // Family-specific detectors — each returns true and fills out a
   // candidate (with direction, intrinsic_score) if its pattern fires.
   //------------------------------------------------------------------

   //--- Breakout: price broke through a KL zone by > tolerance with
   //    min_bursts same-direction acceleration in the last window.
   bool _DetectBreakout(const FeatureSnapshot &fs, KLZone &out_zone,
                        ENUM_TRADE_DIR &dir, double &score) {
      if(!g_cfg.setup_breakout) return false;

      // Find the nearest KL in the max-distance band.
      if(!g_kl.FindNearest(fs.mid, g_cfg.kl_max_setup_distance_pips,
                           g_cfg.kl_min_weight, out_zone)) return false;

      double boundary_upper = out_zone.center + out_zone.half_width;
      double boundary_lower = out_zone.center - out_zone.half_width;
      double break_tol = PriceFromPips(g_cfg.exec_trigger_tolerance_pips);

      // High-weight (D1/H4) KL breakouts need a closed M1 bar beyond the
      // boundary — a single-tick poke is a "betting on breakout" trade
      // against a strong level, historically the lower-EV side. On lower
      // weights (M5/M15/M30) we keep the tick-level responsiveness.
      // Note: "high_weight" uses RAW weight because breakout thesis is
      // "break a real level" — the penalty semantics apply at the gate.
      bool high_weight = (out_zone.weight >= 0.60);
      double c1 = iClose(_Symbol, PERIOD_M1, 1);
      double h1 = iHigh (_Symbol, PERIOD_M1, 1);
      double h2 = iHigh (_Symbol, PERIOD_M1, 2);
      double l1 = iLow  (_Symbol, PERIOD_M1, 1);
      double l2 = iLow  (_Symbol, PERIOD_M1, 2);

      if(fs.mid > boundary_upper + break_tol && g_features.HasMinBursts(DIR_LONG)) {
         // approach-direction: recent bars must have probed BELOW zone center,
         // i.e. price actually crossed up through the zone rather than
         // approaching the zone from above during a sell-off.
         if(MathMin(l1, l2) > out_zone.center) return false;
         if(high_weight && c1 <= boundary_upper) return false;
         dir = DIR_LONG;
         double eff_w = g_kl.EffectiveWeight(out_zone, +1, fs.ts_ms);
         double vscore = Clamp(fs.velocity_pips_per_sec / 2.0, 0.0, 1.0);
         double penalty = Clamp(fs.spread_pips / 1.0, 0.0, 0.5);
         score = Clamp(0.4 + 0.5 * eff_w + 0.3 * vscore - penalty, 0.0, 1.0);
         return true;
      }
      if(fs.mid < boundary_lower - break_tol && g_features.HasMinBursts(DIR_SHORT)) {
         // approach-direction: recent bars must have probed ABOVE zone center,
         // i.e. price actually crossed down through the zone rather than
         // approaching the zone from below during a rally.
         if(MathMax(h1, h2) < out_zone.center) return false;
         if(high_weight && c1 >= boundary_lower) return false;
         dir = DIR_SHORT;
         double eff_w = g_kl.EffectiveWeight(out_zone, -1, fs.ts_ms);
         double vscore = Clamp(-fs.velocity_pips_per_sec / 2.0, 0.0, 1.0);
         double penalty = Clamp(fs.spread_pips / 1.0, 0.0, 0.5);
         score = Clamp(0.4 + 0.5 * eff_w + 0.3 * vscore - penalty, 0.0, 1.0);
         return true;
      }
      return false;
   }

   //--- Rejection: price touched a zone and quickly reversed with a
   //    visible wick; requires min_bursts in the reverse direction.
   bool _DetectRejection(const FeatureSnapshot &fs, KLZone &out_zone,
                         ENUM_TRADE_DIR &dir, double &score) {
      if(!g_cfg.setup_rejection) return false;
      if(!g_kl.FindNearest(fs.mid, g_cfg.kl_max_setup_distance_pips,
                           g_cfg.kl_min_weight, out_zone)) return false;

      // look at the latest closed M1 bar for a wick that pierced the zone
      double o = iOpen(_Symbol, PERIOD_M1, 1);
      double c = iClose(_Symbol, PERIOD_M1, 1);
      double h = iHigh (_Symbol, PERIOD_M1, 1);
      double l = iLow  (_Symbol, PERIOD_M1, 1);
      if(o == 0 || c == 0) return false;

      bool wick_top = (h >= out_zone.center - out_zone.half_width) &&
                      (MathMax(o, c) < out_zone.center) &&
                      ((h - MathMax(o, c)) > (MathAbs(c - o) * 1.2));
      bool wick_bot = (l <= out_zone.center + out_zone.half_width) &&
                      (MathMin(o, c) > out_zone.center) &&
                      ((MathMin(o, c) - l) > (MathAbs(c - o) * 1.2));

      if(wick_top && g_features.HasMinBursts(DIR_SHORT)) {
         dir = DIR_SHORT;
         double eff_w = g_kl.EffectiveWeight(out_zone, -1, fs.ts_ms);
         score = Clamp(0.35 + 0.55 * eff_w, 0.0, 1.0);
         return true;
      }
      if(wick_bot && g_features.HasMinBursts(DIR_LONG)) {
         dir = DIR_LONG;
         double eff_w = g_kl.EffectiveWeight(out_zone, +1, fs.ts_ms);
         score = Clamp(0.35 + 0.55 * eff_w, 0.0, 1.0);
         return true;
      }
      return false;
   }

   //--- Fakeout: price broke, then immediately reversed back through
   //    the zone within the last 3 M1 bars.
   bool _DetectFakeout(const FeatureSnapshot &fs, KLZone &out_zone,
                       ENUM_TRADE_DIR &dir, double &score) {
      if(!g_cfg.setup_fakeout) return false;
      if(!g_kl.FindNearest(fs.mid, g_cfg.kl_max_setup_distance_pips,
                           g_cfg.kl_min_weight, out_zone)) return false;

      // need a bar whose high crossed out_zone+half and close <= center
      for(int i = 1; i <= 3 && i < iBars(_Symbol, PERIOD_M1) - 1; i++) {
         double h = iHigh (_Symbol, PERIOD_M1, i);
         double l = iLow  (_Symbol, PERIOD_M1, i);
         double c = iClose(_Symbol, PERIOD_M1, i);
         double top = out_zone.center + out_zone.half_width;
         double bot = out_zone.center - out_zone.half_width;
         if(h > top + PriceFromPips(g_cfg.exec_trigger_tolerance_pips) && c < out_zone.center) {
            if(g_features.HasMinBursts(DIR_SHORT)) {
               dir = DIR_SHORT;
               double eff_w = g_kl.EffectiveWeight(out_zone, -1, fs.ts_ms);
               score = Clamp(0.30 + 0.60 * eff_w, 0.0, 1.0);
               return true;
            }
         }
         if(l < bot - PriceFromPips(g_cfg.exec_trigger_tolerance_pips) && c > out_zone.center) {
            if(g_features.HasMinBursts(DIR_LONG)) {
               dir = DIR_LONG;
               double eff_w = g_kl.EffectiveWeight(out_zone, +1, fs.ts_ms);
               score = Clamp(0.30 + 0.60 * eff_w, 0.0, 1.0);
               return true;
            }
         }
      }
      return false;
   }

   //--- Retest: we already broke recently, and price is now touching
   //    the zone from the opposite side. (Simplified: requires price
   //    within 1.5x half-width and channel slope in setup direction.)
   bool _DetectRetest(const FeatureSnapshot &fs, KLZone &out_zone,
                      ENUM_TRADE_DIR &dir, double &score) {
      if(!g_cfg.setup_retest) return false;
      if(!g_kl.PriceInAnyZone(fs.mid, g_cfg.kl_min_weight, out_zone)) return false;
      if(fs.chan_slope_pips_per_bar > 0.05 &&
         g_features.HasMinBursts(DIR_LONG)) {
         dir = DIR_LONG;
         double eff_w = g_kl.EffectiveWeight(out_zone, +1, fs.ts_ms);
         score = Clamp(0.35 + 0.55 * eff_w, 0.0, 1.0);
         return true;
      }
      if(fs.chan_slope_pips_per_bar < -0.05 &&
         g_features.HasMinBursts(DIR_SHORT)) {
         dir = DIR_SHORT;
         double eff_w = g_kl.EffectiveWeight(out_zone, -1, fs.ts_ms);
         score = Clamp(0.35 + 0.55 * eff_w, 0.0, 1.0);
         return true;
      }
      return false;
   }

   //--- Failed auction: price broke the regression-channel upper/lower
   //    and within N seconds retraced back inside. Use channel extreme
   //    as the "value area edge" proxy.
   bool _DetectFailedAuction(const FeatureSnapshot &fs, KLZone &out_zone,
                             ENUM_TRADE_DIR &dir, double &score) {
      if(!g_cfg.setup_failed_auction) return false;

      // look at M1 bar 1: did high exceed chan_upper but close return inside?
      double h = iHigh (_Symbol, PERIOD_M1, 1);
      double l = iLow  (_Symbol, PERIOD_M1, 1);
      double c = iClose(_Symbol, PERIOD_M1, 1);
      if(c == 0) return false;

      if(h > fs.chan_upper && c < fs.chan_upper - 0.3 * fs.chan_sigma) {
         // short: failed breakout of upper
         if(g_features.HasMinBursts(DIR_SHORT)) {
            dir = DIR_SHORT;
            // nearest KL is informational; failed auction relies on channel
            g_kl.FindNearest(fs.mid, g_cfg.kl_max_setup_distance_pips,
                             g_cfg.kl_min_weight, out_zone);
            double eff_w = g_kl.EffectiveWeight(out_zone, -1, fs.ts_ms);
            score = Clamp(0.45 + 0.35 * eff_w, 0.0, 1.0);
            return true;
         }
      }
      if(l < fs.chan_lower && c > fs.chan_lower + 0.3 * fs.chan_sigma) {
         if(g_features.HasMinBursts(DIR_LONG)) {
            dir = DIR_LONG;
            g_kl.FindNearest(fs.mid, g_cfg.kl_max_setup_distance_pips,
                             g_cfg.kl_min_weight, out_zone);
            double eff_w = g_kl.EffectiveWeight(out_zone, +1, fs.ts_ms);
            score = Clamp(0.45 + 0.35 * eff_w, 0.0, 1.0);
            return true;
         }
      }
      return false;
   }

   //--- Engulfing with acceleration: latest closed M1 bar engulfs the
   //    previous bar AND we have min_bursts in the engulfing direction.
   bool _DetectEngulfingAccel(const FeatureSnapshot &fs, KLZone &out_zone,
                              ENUM_TRADE_DIR &dir, double &score) {
      if(!g_cfg.setup_engulfing_accel) return false;
      if(iBars(_Symbol, PERIOD_M1) < 4) return false;

      double o1 = iOpen (_Symbol, PERIOD_M1, 1);
      double c1 = iClose(_Symbol, PERIOD_M1, 1);
      double o2 = iOpen (_Symbol, PERIOD_M1, 2);
      double c2 = iClose(_Symbol, PERIOD_M1, 2);

      bool bull_engulf = (c2 < o2) && (c1 > o1) && (c1 >= o2) && (o1 <= c2);
      bool bear_engulf = (c2 > o2) && (c1 < o1) && (c1 <= o2) && (o1 >= c2);

      if(!g_kl.FindNearest(fs.mid, g_cfg.kl_max_setup_distance_pips,
                           g_cfg.kl_min_weight, out_zone)) return false;

      if(bull_engulf && g_features.HasMinBursts(DIR_LONG)) {
         dir = DIR_LONG;
         double eff_w = g_kl.EffectiveWeight(out_zone, +1, fs.ts_ms);
         score = Clamp(0.40 + 0.50 * eff_w, 0.0, 1.0);
         return true;
      }
      if(bear_engulf && g_features.HasMinBursts(DIR_SHORT)) {
         dir = DIR_SHORT;
         double eff_w = g_kl.EffectiveWeight(out_zone, -1, fs.ts_ms);
         score = Clamp(0.40 + 0.50 * eff_w, 0.0, 1.0);
         return true;
      }
      return false;
   }

   //--- Mean reversion: price at chan extreme with opposite bursts and
   //    correlation agrees.
   bool _DetectMeanReversion(const FeatureSnapshot &fs, KLZone &out_zone,
                             ENUM_TRADE_DIR &dir, double &score) {
      if(!g_cfg.setup_mean_reversion) return false;

      bool at_upper = fs.chan_pos_norm >=  0.95;
      bool at_lower = fs.chan_pos_norm <= -0.95;

      if(at_upper && g_features.HasMinBursts(DIR_SHORT)) {
         dir = DIR_SHORT;
         g_kl.FindNearest(fs.mid, g_cfg.kl_max_setup_distance_pips,
                          g_cfg.kl_min_weight, out_zone);
         double eff_w = g_kl.EffectiveWeight(out_zone, -1, fs.ts_ms);
         score = Clamp(0.40 + 0.35 * eff_w, 0.0, 1.0);
         return true;
      }
      if(at_lower && g_features.HasMinBursts(DIR_LONG)) {
         dir = DIR_LONG;
         g_kl.FindNearest(fs.mid, g_cfg.kl_max_setup_distance_pips,
                          g_cfg.kl_min_weight, out_zone);
         double eff_w = g_kl.EffectiveWeight(out_zone, +1, fs.ts_ms);
         score = Clamp(0.40 + 0.35 * eff_w, 0.0, 1.0);
         return true;
      }
      return false;
   }

   //    Momentum continuation: the move is ALREADY above exhaustion
   //    (so breakout/retest are blocked) and has travelled a minimum
   //    directional delta over the persistence_ms window (default 75s,
   //    >= 5 pips). We enter on the first SHALLOW pullback from the
   //    extreme (3-8 pips), paying for the late entry with reduced
   //    size (handled in Execution.mqh) and targeting a measured
   //    continuation with a wide stop.
   //
   //    Thesis: plugs the coverage hole in news-driven trending sessions
   //    where every other family is either blocked by exhaustion
   //    (continuation) or vetoed as too-risky-to-fade (reversal).
   //    See project memory "momentum_cont family".
   bool _DetectMomentumCont(const FeatureSnapshot &fs, KLZone &out_zone,
                            ENUM_TRADE_DIR &dir, double &score) {
      if(!g_cfg.setup_momentum_cont) return false;

      if(iBars(_Symbol, PERIOD_M1) < 3) return false;

      // --- ROCKET CANDLE detection ---------------------------------------
      // The thesis of this family is "one M1 bar travelled absurdly far in
      // one direction — catch the first shallow pullback". So we require a
      // single-bar body >= min_move_pips. Scanning hi-lo across a wider
      // window was wrong: slow drift could accumulate to the threshold
      // without any single violent candle, which is exactly the case the
      // user flagged after a small-candle entry leaked through.
      //
      // We accept the rocket as either:
      //   (a) shift-1 CLOSED bar with |close-open| >= min_move_pips, or
      //   (b) shift-0 FORMING bar that has already moved min_move_pips in
      //       one direction from its own open (intra-bar rocket).
      //
      // Direction is fixed by the rocket's sign. Pullback is measured from
      // the rocket's leading extreme.
      double rocket_body_pips = 0.0;
      double rocket_hi = 0.0, rocket_lo = 0.0;
      ENUM_TRADE_DIR mc_dir = DIR_NONE;

      {
         double o1 = iOpen (_Symbol, PERIOD_M1, 1);
         double c1 = iClose(_Symbol, PERIOD_M1, 1);
         double h1 = iHigh (_Symbol, PERIOD_M1, 1);
         double l1 = iLow  (_Symbol, PERIOD_M1, 1);
         if(o1 > 0.0 && c1 > 0.0) {
            double body1 = PipsFromPrice(c1 - o1);
            if(MathAbs(body1) >= g_cfg.setup_momentum_cont_min_move_pips) {
               rocket_body_pips = body1;
               rocket_hi = h1;
               rocket_lo = l1;
               mc_dir = (body1 > 0) ? DIR_LONG : DIR_SHORT;
            }
         }
      }

      if(mc_dir == DIR_NONE) {
         double o0 = iOpen(_Symbol, PERIOD_M1, 0);
         double h0 = iHigh(_Symbol, PERIOD_M1, 0);
         double l0 = iLow (_Symbol, PERIOD_M1, 0);
         if(o0 > 0.0) {
            double up_prog   = PipsFromPrice(h0 - o0);
            double down_prog = PipsFromPrice(o0 - l0);
            if(up_prog >= g_cfg.setup_momentum_cont_min_move_pips) {
               rocket_body_pips = up_prog;
               rocket_hi = h0;
               rocket_lo = l0;
               mc_dir = DIR_LONG;
            } else if(down_prog >= g_cfg.setup_momentum_cont_min_move_pips) {
               rocket_body_pips = -down_prog;
               rocket_hi = h0;
               rocket_lo = l0;
               mc_dir = DIR_SHORT;
            }
         }
      }

      if(mc_dir == DIR_NONE) return false;

      double abs_body = MathAbs(rocket_body_pips);
      if(abs_body > g_cfg.setup_momentum_cont_max_move_pips) return false;

      // --- Time-based velocity sanity ------------------------------------
      // Price persistence_ms ago must still be meaningfully on the pre-
      // rocket side of current mid. This guards against a bar that spiked
      // and fully reverted within its own window (stale rocket), which the
      // body check alone can't see.
      int pers_ms = g_cfg.setup_momentum_cont_persistence_ms;
      if(pers_ms < 1000) pers_ms = 1000;
      long   target_ts  = (long)fs.ts_ms - (long)pers_ms;
      double past_price = 0.0;
      int    max_shift  = (int)MathMin(iBars(_Symbol, PERIOD_M1) - 1, 30);
      for(int i = 0; i <= max_shift; i++) {
         datetime bt = iTime(_Symbol, PERIOD_M1, i);
         if(bt <= 0) break;
         if((long)bt * 1000 <= target_ts) {
            past_price = iClose(_Symbol, PERIOD_M1, i);
            break;
         }
      }
      if(past_price <= 0.0) return false;

      double delta_pips = PipsFromPrice(fs.mid - past_price);
      if(mc_dir == DIR_LONG  &&
         delta_pips <  g_cfg.setup_momentum_cont_persist_min_delta_pips) return false;
      if(mc_dir == DIR_SHORT &&
         delta_pips > -g_cfg.setup_momentum_cont_persist_min_delta_pips) return false;

      // --- Shallow pullback from rocket's leading extreme ----------------
      double pull_pips = (mc_dir == DIR_LONG)
         ? PipsFromPrice(rocket_hi - fs.bid)
         : PipsFromPrice(fs.ask - rocket_lo);
      if(pull_pips < g_cfg.setup_momentum_cont_min_pullback_pips) return false;
      if(pull_pips > g_cfg.setup_momentum_cont_max_pullback_pips) return false;

      // partner correlation agreement — XAG should agree with the move
      CorrState cs = g_corr.State();
      double agr = (mc_dir == DIR_LONG) ? cs.agreement_xagusd
                                        : -cs.agreement_xagusd;
      if(agr < g_cfg.setup_momentum_cont_partner_min_ag) return false;

      // opposing D1 obstacle veto — don't chase INTO a D1 ceiling/floor
      int dir_sign = (mc_dir == DIR_LONG) ? +1 : -1;
      double d1_clr = g_kl.NearestDailyObstaclePips(fs.mid, dir_sign);
      if(d1_clr < g_cfg.setup_momentum_cont_kl_veto_pips) return false;

      // informational: capture nearest KL at any weight. Not required.
      KLZone nearest; ZeroMemory(nearest);
      g_kl.FindNearest(fs.mid, g_cfg.kl_max_setup_distance_pips, 0.0, nearest);
      out_zone = nearest;
      // Use a synthetic zone id when no real KL was found, so the echo
      // ring can still detect same-direction reconfirmations within the
      // same running move (which is what we want for this family).
      if(StringLen(out_zone.id) == 0) {
         out_zone.id     = (mc_dir == DIR_LONG) ? "momcont_long" : "momcont_short";
         out_zone.weight = 1.0;    // bypasses min-weight gate if it were applied
      }

      // score: base 0.55, up to +0.20 for bigger rockets, small penalty
      // for pullback distance deviating from the sweet spot at ~5 pips.
      double base    = 0.55;
      double bonus   = 0.005 * MathMax(0.0, MathMin(abs_body - 50.0, 40.0));
      double penalty = 0.02  * MathAbs(pull_pips - 5.0);
      score = Clamp(base + bonus - penalty, 0.0, 1.0);
      dir   = mc_dir;
      return true;
   }

   //--- convenience: run a detector, gate, and emit. If submit_intents
   //    is false the real execution path is skipped but shadow trades
   //    are still opened — this lets calibration/ML collect outcomes
   //    even when live trading is disabled or blocked by safety/risk.
   void _TryFamily(const ENUM_SETUP_FAMILY fam, const FeatureSnapshot &fs,
                   const bool submit_intents) {
      if(_CoolingDown(fam, fs.ts_ms)) return;

      KLZone zone; ZeroMemory(zone);
      ENUM_TRADE_DIR dir = DIR_NONE;
      double score = 0.0;
      bool hit = false;

      switch(fam) {
         case SETUP_BREAKOUT:        hit = _DetectBreakout       (fs, zone, dir, score); break;
         case SETUP_RETEST:          hit = _DetectRetest         (fs, zone, dir, score); break;
         case SETUP_REJECTION:       hit = _DetectRejection      (fs, zone, dir, score); break;
         case SETUP_FAKEOUT:         hit = _DetectFakeout        (fs, zone, dir, score); break;
         case SETUP_FAILED_AUCTION:  hit = _DetectFailedAuction  (fs, zone, dir, score); break;
         case SETUP_ENGULFING_ACCEL: hit = _DetectEngulfingAccel (fs, zone, dir, score); break;
         case SETUP_MEAN_REVERSION:  hit = _DetectMeanReversion  (fs, zone, dir, score); break;
         case SETUP_MOMENTUM_CONT:   hit = _DetectMomentumCont   (fs, zone, dir, score); break;
         default: return;
      }
      if(!hit) return;

      SetupCandidate cand;
      cand.family          = fam;
      cand.direction       = dir;
      cand.zone            = zone;
      cand.kl_distance_pips= PipsFromPrice(MathAbs(fs.mid - zone.center));
      cand.intrinsic_score = score;
      // Daily Bias soft score adjustment. Long trades get lifted when bias
      // is positive, shorts when bias is negative; contra-direction trades
      // are pushed down by the same magnitude. No hard veto — this just
      // shifts candidates across the min-score gate.
      if(g_cfg.daily_bias_enable) {
         double bias = g_daily_bias.Bias(fs.ts_ms);
         double dir_sign = (dir == DIR_LONG) ? +1.0 : -1.0;
         cand.intrinsic_score += bias * dir_sign * g_cfg.daily_bias_score_weight;
         if(cand.intrinsic_score > 1.0) cand.intrinsic_score = 1.0;
         if(cand.intrinsic_score < 0.0) cand.intrinsic_score = 0.0;
      }
      cand.ml_score        = -1.0; // populated by ML layer if enabled
      cand.block_reason    = "";
      cand.accepted        = false;
      cand.route_status    = "";
      cand.echo_count      = 0;

      if(!_CommonGates(cand, fs)) {
         _Emit(cand, fs);
         return;
      }

      if(cand.intrinsic_score < g_cfg.setup_min_score) {
         cand.block_reason = StringFormat("score %.2f < %.2f",
                                          cand.intrinsic_score, g_cfg.setup_min_score);
         _Emit(cand, fs);
         return;
      }

      cand.accepted = true;

      // Compute the post-accept routing decision so the setups.csv row
      // reflects whether this candidate is actually going to become a
      // live order or only a shadow-tracked paper trade. The dashboard
      // filters on `accepted && route_status == "live"` to show only
      // the rows that are really in line for execution.
      string route = "live";
      bool will_submit = false;
      if(!submit_intents) {
         if(!g_risk.AllowNewEntry())         route = "shadow_only:risk_halt";
         else if(!g_safety.AllowNewEntry())  route = "shadow_only:safety_halt";
         else                                route = "shadow_only:disabled";
      } else if(!g_risk.AllowDirection(cand.direction)) {
         route = "shadow_only:risk_dir";
      } else if(g_exec.PendingCount() > 0) {
         // Strict concurrency: a fill is already in flight. Refuse the
         // live submission so only one intent can be committed at a
         // time. Pairs with the FiloHFT.mq5 trigger gate — without
         // this mirror, Setups would still submit intents that would
         // then be held by the trigger gate until TTL. BE stacking
         // still works because the gate only blocks the ~10s pending
         // window, not the post-fill at-BE state.
         route = "shadow_only:pending_fill";
      } else {
         string cool_reason = "";
         if(!g_tm.PassesFillCooldown(cand.direction, fs.ts_ms, cool_reason)) {
            route = StringFormat("shadow_only:%s", cool_reason);
         } else {
            string stack_reason = "";
            if(!g_tm.CanStack(cand.direction, cand.family, cand.zone.id,
                              fs.ts_ms, stack_reason)) {
               route = StringFormat("shadow_only:%s", stack_reason);
            } else {
               will_submit = true;
            }
         }
      }
      cand.route_status = route;
      // Stamp the echo tag BEFORE emitting and BEFORE pushing into the ring,
      // so self-matches are impossible and each accepted candidate sees only
      // strictly earlier same-zone same-direction accepts.
      cand.echo_count = _ComputeEchoCount(cand.direction, cand.zone.id, fs.ts_ms);
      _Emit(cand, fs);
      _PushEcho(cand, fs);
      m_last_emit[(int)fam] = fs.ts_ms;

      // Shadow ("paper") trade — always opened on accepted candidates so
      // calibration/ML sees the full signal stream, not just live fills.
      // echo_count is passed through so shadow_trades.csv carries the tag
      // directly and can be grouped by it without a setups.csv join.
      CorrState cs_snap = g_corr.State();
      g_shadow.Open(cand.family, cand.direction, cand.zone.id, cand.zone.weight,
                    cand.kl_distance_pips, cand.intrinsic_score,
                    cand.echo_count, fs, cs_snap);

      if(will_submit) {
         g_intents.Submit(cand.family, cand.direction, cand.zone.id,
                          cand.intrinsic_score,
                          cand.zone.weight, cand.kl_distance_pips,
                          cand.echo_count, fs);
      }
   }

public:
   void Init() {
      ArrayInitialize(m_last_emit, 0);
      m_cooldown_ms = 800; // per-family; prevents re-firing on every tick
      for(int i = 0; i < FILO_ECHO_RING; i++) {
         m_echo[i].ts_ms     = 0;
         m_echo[i].zone_id   = "";
         m_echo[i].direction = DIR_LONG;
      }
      m_echo_head = 0;
   }

   void Scan(const FeatureSnapshot &fs, const bool submit_intents) {
      // momentum_cont is the only family that works without any KLs at all
      // (its thesis is structural, not zone-based). Every other family is
      // gated by the existing g_kl.Count() check above.
      _TryFamily(SETUP_MOMENTUM_CONT, fs, submit_intents);
      if(g_kl.Count() == 0) return;
      _TryFamily(SETUP_BREAKOUT,        fs, submit_intents);
      _TryFamily(SETUP_RETEST,          fs, submit_intents);
      _TryFamily(SETUP_REJECTION,       fs, submit_intents);
      _TryFamily(SETUP_FAKEOUT,         fs, submit_intents);
      _TryFamily(SETUP_FAILED_AUCTION,  fs, submit_intents);
      _TryFamily(SETUP_ENGULFING_ACCEL, fs, submit_intents);
      _TryFamily(SETUP_MEAN_REVERSION,  fs, submit_intents);
   }
};

Setups g_setups;

#endif // __FILOHFT_SETUPS_MQH__
