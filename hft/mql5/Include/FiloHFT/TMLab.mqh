//+------------------------------------------------------------------+
//|  FiloHFT/TMLab.mqh                                                |
//|  Parallel trade-management "variant lab".                         |
//|                                                                   |
//|  On every REAL fill (Execution.ExecuteIntent success), spawn N    |
//|  virtual positions in parallel — one per registered TM profile.   |
//|  Each virtual position runs its own (stop, BE, trail, target)     |
//|  state machine against the live tick stream and closes on SL,    |
//|  TP, TTL, or a rule-specific early-exit. Outcomes land in         |
//|  tm_lab.csv with the variant name so end-of-week analysis can    |
//|  rank profiles apples-to-apples on the exact same fills the      |
//|  live engine took.                                                |
//|                                                                   |
//|  Compute per variant is trivial (~30 FLOPs/tick). The cap is     |
//|  statistical power per variant, not CPU.                          |
//|                                                                   |
//|  Assumptions:                                                     |
//|   - immediate fill at the real fill price (slippage unmodelled)  |
//|   - close at bid (long) / ask (short) on touch                    |
//|   - TTL 10 minutes                                                |
//|   - SL/TP/BE/trail logic mirrors TradeManagement._Step for the   |
//|     STATIC rule, and dispatches to rule-specific branches for   |
//|     the dynamic rules (ATR-scaled, peak-ratio, velocity-aware,  |
//|     score-scaled, KL-weight-scaled, parabolic, sigmoid, etc.)    |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_TMLAB_MQH__
#define __FILOHFT_TMLAB_MQH__

#include <FiloHFT/Common.mqh>
#include <FiloHFT/Config.mqh>
#include <FiloHFT/Telemetry.mqh>
#include <FiloHFT/Features.mqh>
#include <FiloHFT/ExecutionIntent.mqh>

#define TMLAB_MAX_PROFILES  64
#define TMLAB_MAX_OPEN      512
#define TMLAB_TTL_MS        600000    // 10 minutes

enum ENUM_TMLAB_RULE {
   TML_STATIC              = 0,   // fixed SL / BE / trail / TP (the 34 static variants)
   TML_ATR_STOP            = 1,   // stop_pips = p1 * ATR
   TML_ATR_TRAIL           = 2,   // trail_dist = p1 * ATR
   TML_PEAK_RATIO_TRAIL    = 3,   // trail_dist = max(2, peak * p1)
   TML_VELOCITY_TRAIL      = 4,   // trail_dist = p1 + p2 * favourable_velocity
   TML_ACCEL_EXTEND_TP     = 5,   // if peak>=p1 AND velocity favourable, extend tp by p2
   TML_ADVERSE_CUT         = 6,   // if adverse velocity >= p1 while trade underwater, cut
   TML_SCORE_TARGET_LIN    = 7,   // tp = p1 + (score - 0.45) * p2, clamped to [p1, p3]
   TML_SCORE_RUNNER        = 8,   // first tp at target_pips; if score>p1, runner to p2
   TML_PARABOLIC_TRAIL     = 9,   // trail_dist = max(1.5, p1 - p2 * peak^p3)
   TML_SIGMOID_TRAIL       = 10,  // trail_dist = p1 - p2 / (1 + exp(-(peak - p3)))
   TML_KL_WEIGHT_TARGET    = 11,  // tp = p1 + kl_weight * p2
   TML_RATCHET_BE          = 12   // step-function BE: +0 at be_trig, +p1 at 12p, +p2 at 20p
};

struct TMProfile {
   string           name;
   ENUM_TMLAB_RULE  rule;
   double           stop_pips;
   double           be_trigger_pips;
   double           be_buffer_pips;
   double           trail_start_pips;
   double           trail_distance_pips;
   double           target_pips;
   double           p1;     // rule-specific
   double           p2;
   double           p3;
};

struct TMLabPosition {
   bool              active;
   int               profile_idx;
   long              parent_ticket;
   ENUM_SETUP_FAMILY family;
   ENUM_TRADE_DIR    direction;
   double            entry_price;
   long              open_ts_ms;
   // entry-time snapshots (for dynamic rules + output)
   double            entry_intrinsic_score;
   double            entry_kl_weight;
   double            entry_atr_pips;
   double            entry_velocity;
   // evolving state
   double            hidden_sl;
   double            hidden_tp;
   bool              be_promoted;
   bool              trail_armed;
   double            peak_pips;
   double            trough_pips;
   // rule-specific state
   int               ratchet_step;   // ratchet_be: 0/1/2/3
   bool              runner_active;  // score_runner: has the runner promotion not yet fired
   // echo tag snapshotted from the originating setup, carried through
   // to tm_lab.csv so variant outcomes can be grouped by echo_count.
   int               echo_count;
};

class TMLab {
private:
   TMProfile       m_profiles[TMLAB_MAX_PROFILES];
   int             m_n_profiles;
   TMLabPosition   m_pos[TMLAB_MAX_OPEN];
   int             m_open_count;
   long            m_total_opened;
   long            m_total_closed;
   int             m_atr_handle;      // M1, period 14

   void _AddStatic(const string name,
                   const double stop, const double be_t, const double be_b,
                   const double tstart, const double tdist, const double tgt) {
      if(m_n_profiles >= TMLAB_MAX_PROFILES) return;
      TMProfile p;
      p.name                = name;
      p.rule                = TML_STATIC;
      p.stop_pips           = stop;
      p.be_trigger_pips     = be_t;
      p.be_buffer_pips      = be_b;
      p.trail_start_pips    = tstart;
      p.trail_distance_pips = tdist;
      p.target_pips         = tgt;
      p.p1 = 0; p.p2 = 0; p.p3 = 0;
      m_profiles[m_n_profiles++] = p;
   }

   void _AddDynamic(const string name, const ENUM_TMLAB_RULE rule,
                    const double stop, const double be_t, const double be_b,
                    const double tstart, const double tdist, const double tgt,
                    const double p1, const double p2, const double p3) {
      if(m_n_profiles >= TMLAB_MAX_PROFILES) return;
      TMProfile p;
      p.name                = name;
      p.rule                = rule;
      p.stop_pips           = stop;
      p.be_trigger_pips     = be_t;
      p.be_buffer_pips      = be_b;
      p.trail_start_pips    = tstart;
      p.trail_distance_pips = tdist;
      p.target_pips         = tgt;
      p.p1 = p1; p.p2 = p2; p.p3 = p3;
      m_profiles[m_n_profiles++] = p;
   }

   //--- 51 variants. Change any row to re-calibrate the grid;
   //    everything is one function so the layout is auditable.
   void _LoadProfiles() {
      m_n_profiles = 0;

      // ---- Family 1: stop sweep 16..25 (10 profiles) -----------------
      _AddStatic("base",    20, 6, 1.0,  8, 3, 30);
      _AddStatic("stop_16", 16, 6, 1.0,  8, 3, 30);
      _AddStatic("stop_17", 17, 6, 1.0,  8, 3, 30);
      _AddStatic("stop_18", 18, 6, 1.0,  8, 3, 30);
      _AddStatic("stop_19", 19, 6, 1.0,  8, 3, 30);
      _AddStatic("stop_21", 21, 6, 1.0,  8, 3, 30);
      _AddStatic("stop_22", 22, 6, 1.0,  8, 3, 30);
      _AddStatic("stop_23", 23, 6, 1.0,  8, 3, 30);
      _AddStatic("stop_24", 24, 6, 1.0,  8, 3, 30);
      _AddStatic("stop_25", 25, 6, 1.0,  8, 3, 30);

      // ---- Family 2: BE trigger sweep 4..9 (5, skip 6=base) ----------
      _AddStatic("be_trig_4", 20, 4, 1.0, 8, 3, 30);
      _AddStatic("be_trig_5", 20, 5, 1.0, 8, 3, 30);
      _AddStatic("be_trig_7", 20, 7, 1.0, 8, 3, 30);
      _AddStatic("be_trig_8", 20, 8, 1.0, 8, 3, 30);
      _AddStatic("be_trig_9", 20, 9, 1.0, 8, 3, 30);

      // ---- Family 3: BE buffer 0 / 0.5 / 1.5 / 2 (4, skip 1=base) ----
      _AddStatic("be_buf_0",   20, 6, 0.0, 8, 3, 30);
      _AddStatic("be_buf_0_5", 20, 6, 0.5, 8, 3, 30);
      _AddStatic("be_buf_1_5", 20, 6, 1.5, 8, 3, 30);
      _AddStatic("be_buf_2",   20, 6, 2.0, 8, 3, 30);

      // ---- Family 4: trail_start sweep 6/10/11/12/14 (5, skip 8) -----
      _AddStatic("tstart_6",  20, 6, 1.0,  6, 3, 30);
      _AddStatic("tstart_10", 20, 6, 1.0, 10, 3, 30);
      _AddStatic("tstart_11", 20, 6, 1.0, 11, 3, 30);
      _AddStatic("tstart_12", 20, 6, 1.0, 12, 3, 30);
      _AddStatic("tstart_14", 20, 6, 1.0, 14, 3, 30);

      // ---- Family 5: trail_distance 2/4/5/6 (4, skip 3=base) ---------
      _AddStatic("tdist_2", 20, 6, 1.0, 8, 2, 30);
      _AddStatic("tdist_4", 20, 6, 1.0, 8, 4, 30);
      _AddStatic("tdist_5", 20, 6, 1.0, 8, 5, 30);
      _AddStatic("tdist_6", 20, 6, 1.0, 8, 6, 30);

      // ---- Family 6: TP sweep 12/15/18/20/25 (5, skip 30=base) -------
      _AddStatic("tp_12", 20, 6, 1.0, 8, 3, 12);
      _AddStatic("tp_15", 20, 6, 1.0, 8, 3, 15);
      _AddStatic("tp_18", 20, 6, 1.0, 8, 3, 18);
      _AddStatic("tp_20", 20, 6, 1.0, 8, 3, 20);
      _AddStatic("tp_25", 20, 6, 1.0, 8, 3, 25);

      // ==== Dynamic families =========================================
      // ---- Family 7: ATR-scaled stop (3) -- p1 = k ATR multiplier ----
      _AddDynamic("atr_stop_1_0", TML_ATR_STOP, 20, 6, 1.0, 8, 3, 30, 1.0, 0, 0);
      _AddDynamic("atr_stop_1_5", TML_ATR_STOP, 20, 6, 1.0, 8, 3, 30, 1.5, 0, 0);
      _AddDynamic("atr_stop_2_0", TML_ATR_STOP, 20, 6, 1.0, 8, 3, 30, 2.0, 0, 0);

      // ---- Family 8: ATR-scaled trail (2) -- p1 = k ------------------
      _AddDynamic("atr_trail_1_0", TML_ATR_TRAIL, 20, 6, 1.0, 8, 3, 30, 1.0, 0, 0);
      _AddDynamic("atr_trail_1_5", TML_ATR_TRAIL, 20, 6, 1.0, 8, 3, 30, 1.5, 0, 0);

      // ---- Family 9: peak-ratio trail (3) -- p1 = ratio --------------
      _AddDynamic("peak_ratio_20", TML_PEAK_RATIO_TRAIL, 20, 6, 1.0, 8, 3, 30, 0.20, 0, 0);
      _AddDynamic("peak_ratio_25", TML_PEAK_RATIO_TRAIL, 20, 6, 1.0, 8, 3, 30, 0.25, 0, 0);
      _AddDynamic("peak_ratio_33", TML_PEAK_RATIO_TRAIL, 20, 6, 1.0, 8, 3, 30, 0.33, 0, 0);

      // ---- Family 10: velocity-trail (2) -- p1=base, p2=k pips-per-(pips/s)
      _AddDynamic("vel_trail_slow", TML_VELOCITY_TRAIL, 20, 6, 1.0, 8, 3, 30, 3.0, 0.5, 0);
      _AddDynamic("vel_trail_fast", TML_VELOCITY_TRAIL, 20, 6, 1.0, 8, 3, 30, 3.0, 1.0, 0);

      // ---- Family 11: accel-extend TP (1) -- p1=peak_thr, p2=extend --
      _AddDynamic("accel_extend", TML_ACCEL_EXTEND_TP, 20, 6, 1.0, 8, 3, 20, 15, 10, 0);

      // ---- Family 12: adverse-velocity cut (1) -- p1=adv pips/s ------
      _AddDynamic("adverse_cut", TML_ADVERSE_CUT, 20, 6, 1.0, 8, 3, 30, 3.0, 0, 0);

      // ==== New dynamic variants (the six extras) ====================
      // 46 score_target_lin: tp = p1 + (score-0.45)*p2 clamped [p1, p3]
      _AddDynamic("score_target_lin", TML_SCORE_TARGET_LIN,
                  20, 6, 1.0, 8, 3, 30, 15, 50, 40);

      // 47 score_runner: initial tp at target_pips (10);
      //    if score>p1 (0.6), relocate tp to p2 (25) on first tp touch
      _AddDynamic("score_runner", TML_SCORE_RUNNER,
                  20, 6, 1.0, 8, 3, 10, 0.60, 25, 0);

      // 48 parabolic_trail: trail_dist = max(1.5, p1 - p2 * peak^p3)
      _AddDynamic("parabolic_trail", TML_PARABOLIC_TRAIL,
                  20, 6, 1.0, 8, 3, 30, 5.0, 0.08, 1.3);

      // 49 sigmoid_trail: trail_dist = p1 - p2 / (1+exp(-(peak - p3)))
      _AddDynamic("sigmoid_trail", TML_SIGMOID_TRAIL,
                  20, 6, 1.0, 8, 3, 30, 5.0, 3.0, 12);

      // 50 kl_weight_target: tp = p1 + kl_weight * p2
      _AddDynamic("kl_weight_target", TML_KL_WEIGHT_TARGET,
                  20, 6, 1.0, 8, 3, 30, 15, 20, 0);

      // 51 ratchet_be: step BE — +0 @ be_trig, +p1 @ 12p, +p2 @ 20p
      _AddDynamic("ratchet_be", TML_RATCHET_BE,
                  20, 6, 1.0, 8, 3, 30, 4, 8, 0);
   }

   double _CurrentATRPips() {
      if(m_atr_handle == INVALID_HANDLE) return 5.0;
      double buf[1];
      if(CopyBuffer(m_atr_handle, 0, 0, 1, buf) != 1) return 5.0;
      if(buf[0] <= 0) return 5.0;
      return PipsFromPrice(buf[0]);
   }

   //--- Compute the effective stop_pips / target_pips for this profile
   //    given the entry-time context. Used by OnFill when seeding the
   //    virtual position.
   void _ResolveInitial(const TMProfile &prof,
                        const double score, const double klw, const double atrp,
                        double &stop_pips, double &tgt_pips) {
      stop_pips = prof.stop_pips;
      tgt_pips  = prof.target_pips;
      switch(prof.rule) {
         case TML_ATR_STOP:
            stop_pips = MathMax(5.0, prof.p1 * atrp);
            break;
         case TML_SCORE_TARGET_LIN: {
            double t = prof.p1 + (score - 0.45) * prof.p2;
            if(t < prof.p1) t = prof.p1;
            if(t > prof.p3) t = prof.p3;
            tgt_pips = t;
            break;
         }
         case TML_KL_WEIGHT_TARGET:
            tgt_pips = prof.p1 + klw * prof.p2;
            break;
         default:
            break;
      }
   }

   //--- Compute the trail-distance (in pips) for this profile given the
   //    current position state and tick context. Used once gain crosses
   //    trail_start_pips.
   double _TrailDistPips(const TMProfile &prof, const TMLabPosition &p,
                         const FeatureSnapshot &fs) {
      switch(prof.rule) {
         case TML_ATR_TRAIL:
            return MathMax(1.5, prof.p1 * p.entry_atr_pips);
         case TML_PEAK_RATIO_TRAIL:
            return MathMax(2.0, p.peak_pips * prof.p1);
         case TML_VELOCITY_TRAIL: {
            double fav_vel = (p.direction == DIR_LONG) ?  fs.velocity_pips_per_sec
                                                       : -fs.velocity_pips_per_sec;
            double d = prof.p1 + prof.p2 * MathMax(0.0, fav_vel);
            return MathMax(1.5, d);
         }
         case TML_PARABOLIC_TRAIL: {
            double peak = MathMax(0.0, p.peak_pips);
            double d = prof.p1 - prof.p2 * MathPow(peak, prof.p3);
            return MathMax(1.5, d);
         }
         case TML_SIGMOID_TRAIL: {
            double z = p.peak_pips - prof.p3;
            double d = prof.p1 - prof.p2 / (1.0 + MathExp(-z));
            return MathMax(1.5, d);
         }
         default:
            return prof.trail_distance_pips;
      }
   }

   void _Close(const int i, const string reason, const double exit_px,
               const long exit_ts) {
      TMLabPosition p = m_pos[i];
      TMProfile     prof = m_profiles[p.profile_idx];
      double realized_pips = (p.direction == DIR_LONG)
         ? PipsFromPrice(exit_px - p.entry_price)
         : PipsFromPrice(p.entry_price - exit_px);
      double realized_usd = realized_pips
                          * g_cfg.pip_in_points
                          * g_cfg.usd_per_point_per_lot
                          * g_cfg.fixed_lot;

      if(g_cfg.tel_log_tmlab) {
         g_tel.WriteTMLab(
            p.open_ts_ms, exit_ts, p.parent_ticket,
            prof.name, SetupFamilyName(p.family), DirName(p.direction),
            p.entry_price, exit_px,
            p.peak_pips, p.trough_pips,
            realized_pips, realized_usd, reason,
            p.be_promoted ? 1 : 0,
            p.trail_armed ? 1 : 0,
            p.entry_intrinsic_score, p.entry_kl_weight,
            p.entry_atr_pips, p.entry_velocity,
            p.echo_count);
      }
      m_pos[i].active = false;
      m_open_count--;
      m_total_closed++;
   }

   void _StepOne(const int i, const FeatureSnapshot &fs) {
      TMLabPosition p    = m_pos[i];
      TMProfile     prof = m_profiles[p.profile_idx];

      double mark = (p.direction == DIR_LONG) ? fs.bid : fs.ask;
      double gain_pips = (p.direction == DIR_LONG)
         ? PipsFromPrice(mark - p.entry_price)
         : PipsFromPrice(p.entry_price - mark);

      if(gain_pips > p.peak_pips)   p.peak_pips   = gain_pips;
      if(gain_pips < p.trough_pips) p.trough_pips = gain_pips;

      // --- dynamic per-tick hooks that can trigger early exits --------

      // Adverse-velocity cut: exit immediately if moving against us fast
      // while underwater. Only fires for that rule.
      if(prof.rule == TML_ADVERSE_CUT) {
         double adv_vel = (p.direction == DIR_LONG) ? -fs.velocity_pips_per_sec
                                                    :  fs.velocity_pips_per_sec;
         if(adv_vel >= prof.p1 && gain_pips < 0.0) {
            m_pos[i] = p;
            _Close(i, "adverse_cut", mark, fs.ts_ms);
            return;
         }
      }

      // Accel-extend TP: if peak has crossed the threshold and velocity
      // is still in our favour, push the hidden TP out by p2 pips from
      // the original target.
      if(prof.rule == TML_ACCEL_EXTEND_TP) {
         double fav_vel = (p.direction == DIR_LONG) ?  fs.velocity_pips_per_sec
                                                    : -fs.velocity_pips_per_sec;
         if(p.peak_pips >= prof.p1 && fav_vel > 0.5) {
            double extended_pips = prof.target_pips + prof.p2;
            double new_tp = (p.direction == DIR_LONG)
               ? p.entry_price + PriceFromPips(extended_pips)
               : p.entry_price - PriceFromPips(extended_pips);
            if((p.direction == DIR_LONG  && new_tp > p.hidden_tp) ||
               (p.direction == DIR_SHORT && new_tp < p.hidden_tp)) {
               p.hidden_tp = new_tp;
            }
         }
      }

      // Score runner: once we reach the first TP and the runner flag
      // is set (score > p1 at entry), relocate the hidden TP out to p2
      // and clear the runner flag so this only fires once.
      if(prof.rule == TML_SCORE_RUNNER) {
         if(p.runner_active && gain_pips >= prof.target_pips) {
            double new_tp = (p.direction == DIR_LONG)
               ? p.entry_price + PriceFromPips(prof.p2)
               : p.entry_price - PriceFromPips(prof.p2);
            p.hidden_tp = new_tp;
            p.runner_active = false;
         }
      }

      // --- Exits: SL / TP --------------------------------------------
      if((p.direction == DIR_LONG  && mark <= p.hidden_sl) ||
         (p.direction == DIR_SHORT && mark >= p.hidden_sl)) {
         m_pos[i] = p;
         _Close(i, p.be_promoted ? (p.trail_armed ? "trail" : "be") : "sl",
                p.hidden_sl, fs.ts_ms);
         return;
      }
      if((p.direction == DIR_LONG  && mark >= p.hidden_tp) ||
         (p.direction == DIR_SHORT && mark <= p.hidden_tp)) {
         m_pos[i] = p;
         _Close(i, "tp", p.hidden_tp, fs.ts_ms);
         return;
      }

      // --- BE / ratchet BE -------------------------------------------
      if(prof.rule == TML_RATCHET_BE) {
         // Three rungs: base rung at be_trigger_pips pinning to entry+0,
         // rung 1 at 12p pinning to entry+p1, rung 2 at 20p pinning to
         // entry+p2. Monotone: each step can only ratchet the SL further
         // in our favour, never reverse it.
         double thresholds[3]; double buffers[3];
         thresholds[0] = prof.be_trigger_pips; buffers[0] = 0.0;
         thresholds[1] = 12.0;                 buffers[1] = prof.p1;
         thresholds[2] = 20.0;                 buffers[2] = prof.p2;
         for(int s = 2; s >= 0; s--) {
            if(p.ratchet_step > s) break;
            if(gain_pips >= thresholds[s]) {
               double new_sl = (p.direction == DIR_LONG)
                  ? p.entry_price + PriceFromPips(buffers[s])
                  : p.entry_price - PriceFromPips(buffers[s]);
               if((p.direction == DIR_LONG  && new_sl > p.hidden_sl) ||
                  (p.direction == DIR_SHORT && new_sl < p.hidden_sl)) {
                  p.hidden_sl = new_sl;
                  p.be_promoted = true;
               }
               p.ratchet_step = s + 1;
               break;
            }
         }
      } else {
         if(!p.be_promoted && gain_pips >= prof.be_trigger_pips) {
            double buf = PriceFromPips(prof.be_buffer_pips);
            double new_sl = (p.direction == DIR_LONG) ? p.entry_price + buf
                                                      : p.entry_price - buf;
            if((p.direction == DIR_LONG  && new_sl > p.hidden_sl) ||
               (p.direction == DIR_SHORT && new_sl < p.hidden_sl)) {
               p.hidden_sl = new_sl;
            }
            p.be_promoted = true;
         }
      }

      // --- Trailing --------------------------------------------------
      if(gain_pips >= prof.trail_start_pips) {
         double dist_pips = _TrailDistPips(prof, p, fs);
         double dist_px   = PriceFromPips(dist_pips);
         double new_sl;
         if(p.direction == DIR_LONG) {
            new_sl = mark - dist_px;
            if(new_sl > p.hidden_sl) { p.hidden_sl = new_sl; p.trail_armed = true; }
         } else {
            new_sl = mark + dist_px;
            if(new_sl < p.hidden_sl) { p.hidden_sl = new_sl; p.trail_armed = true; }
         }
      }

      // --- TTL --------------------------------------------------------
      if(fs.ts_ms - p.open_ts_ms >= TMLAB_TTL_MS) {
         m_pos[i] = p;
         _Close(i, "ttl", mark, fs.ts_ms);
         return;
      }

      m_pos[i] = p;
   }

public:
   void Init() {
      for(int i = 0; i < TMLAB_MAX_OPEN; i++) ZeroMemory(m_pos[i]);
      m_open_count   = 0;
      m_total_opened = 0;
      m_total_closed = 0;
      m_atr_handle   = iATR(_Symbol, PERIOD_M1, 14);
      _LoadProfiles();
   }

   int  ProfileCount() const { return m_n_profiles; }
   int  OpenCount()    const { return m_open_count; }
   long TotalOpened()  const { return m_total_opened; }
   long TotalClosed()  const { return m_total_closed; }

   //--- Spawn one virtual position per profile on a real fill.
   //    Called from Execution.ExecuteIntent right after Buy/Sell success.
   void OnFill(const long              parent_ticket,
               const ENUM_SETUP_FAMILY family,
               const ENUM_TRADE_DIR    dir,
               const double            entry_price,
               const double            intrinsic_score,
               const double            kl_weight,
               const int               echo_count,
               const FeatureSnapshot  &fs) {
      double atr_pips = _CurrentATRPips();
      for(int k = 0; k < m_n_profiles; k++) {
         if(m_open_count >= TMLAB_MAX_OPEN) break;

         int slot = -1;
         for(int i = 0; i < TMLAB_MAX_OPEN; i++) {
            if(!m_pos[i].active) { slot = i; break; }
         }
         if(slot < 0) break;

         TMProfile prof = m_profiles[k];

         double stop_pips = 0, tgt_pips = 0;
         _ResolveInitial(prof, intrinsic_score, kl_weight, atr_pips,
                         stop_pips, tgt_pips);

         TMLabPosition p;
         ZeroMemory(p);
         p.active                = true;
         p.profile_idx           = k;
         p.parent_ticket         = parent_ticket;
         p.family                = family;
         p.direction             = dir;
         p.entry_price           = entry_price;
         p.open_ts_ms            = fs.ts_ms;
         p.entry_intrinsic_score = intrinsic_score;
         p.entry_kl_weight       = kl_weight;
         p.entry_atr_pips        = atr_pips;
         p.entry_velocity        = fs.velocity_pips_per_sec;
         p.hidden_sl = (dir == DIR_LONG) ? entry_price - PriceFromPips(stop_pips)
                                         : entry_price + PriceFromPips(stop_pips);
         p.hidden_tp = (dir == DIR_LONG) ? entry_price + PriceFromPips(tgt_pips)
                                         : entry_price - PriceFromPips(tgt_pips);
         p.be_promoted  = false;
         p.trail_armed  = false;
         p.peak_pips    = 0.0;
         p.trough_pips  = 0.0;
         p.ratchet_step = 0;
         // Runner flag: only set for the score_runner rule, and only if
         // the entry-time score clears the threshold. This is why we
         // snapshot the decision at open — the score is immutable for
         // the life of the virtual position, same as real entries.
         p.runner_active = (prof.rule == TML_SCORE_RUNNER) &&
                           (intrinsic_score > prof.p1);
         p.echo_count    = echo_count;

         m_pos[slot] = p;
         m_open_count++;
         m_total_opened++;
      }
   }

   //--- Per-tick evolution. Cheap: ~30 FLOPs per active variant.
   void Step(const FeatureSnapshot &fs) {
      if(m_open_count <= 0) return;
      for(int i = 0; i < TMLAB_MAX_OPEN; i++) {
         if(!m_pos[i].active) continue;
         _StepOne(i, fs);
      }
   }

   //--- Shutdown / EA unload: close every variant at current mark so
   //    tm_lab.csv never has dangling open rows.
   void CloseAll(const FeatureSnapshot &fs) {
      for(int i = 0; i < TMLAB_MAX_OPEN; i++) {
         if(!m_pos[i].active) continue;
         double mark = (m_pos[i].direction == DIR_LONG) ? fs.bid : fs.ask;
         _Close(i, "shutdown", mark, fs.ts_ms);
      }
   }
};

TMLab g_tmlab;

#endif // __FILOHFT_TMLAB_MQH__
