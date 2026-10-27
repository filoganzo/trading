//+------------------------------------------------------------------+
//|  FiloHFT/ExecutionIntent.mqh                                     |
//|  Internal execution intent state machine.                        |
//|                                                                 |
//|  An intent is the in-memory commitment to trade a given setup    |
//|  IF its trigger zone is hit within TTL. Intents never reach the  |
//|  broker — they convert to market orders at the moment the       |
//|  trigger fires, so the broker only ever sees a market order.    |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_EXECUTION_INTENT_MQH__
#define __FILOHFT_EXECUTION_INTENT_MQH__

#include <FiloHFT/Common.mqh>
#include <FiloHFT/Config.mqh>
#include <FiloHFT/Telemetry.mqh>
#include <FiloHFT/KeyLevels.mqh>
#include <FiloHFT/Features.mqh>

struct ExecutionIntent {
   string            id;
   ENUM_SETUP_FAMILY family;
   ENUM_TRADE_DIR    direction;
   string            zone_id;         // KL zone that generated the setup
   double            trigger_price;   // market price at which to fire
   double            hidden_sl;
   double            hidden_tp;
   long              created_ms;
   int               ttl_ms;
   double            setup_score;
   ENUM_INTENT_STATE state;
   double            tolerance;       // price tolerance (price units)
   // Carried through for TMLab dynamic-rule variants that need to know
   // the quality of the originating setup (score-scaled TPs, KL-weight
   // scaled TPs, etc.). Not used by live execution or TradeManagement.
   double            kl_weight;
   double            kl_distance_pips;
   int               echo_count;      // same-zone same-dir reconfirmations at emit time
};

double IntentStopPipsForFamily(const ENUM_SETUP_FAMILY f) {
   switch(f) {
      case SETUP_BREAKOUT:        return g_cfg.tm_stop_breakout_pips;
      case SETUP_RETEST:          return g_cfg.tm_stop_breakout_pips;
      case SETUP_REJECTION:       return g_cfg.tm_stop_rejection_pips;
      case SETUP_FAKEOUT:         return g_cfg.tm_stop_rejection_pips;
      case SETUP_FAILED_AUCTION:  return g_cfg.tm_stop_reversion_pips;
      case SETUP_ENGULFING_ACCEL: return g_cfg.tm_stop_breakout_pips;
      case SETUP_MEAN_REVERSION:  return g_cfg.tm_stop_reversion_pips;
      case SETUP_MOMENTUM_CONT:   return g_cfg.setup_momentum_cont_stop_pips;
   }
   return g_cfg.tm_stop_reversion_pips;
}

double IntentTargetPipsForFamily(const ENUM_SETUP_FAMILY f) {
   switch(f) {
      case SETUP_BREAKOUT:        return g_cfg.tm_target_breakout_pips;
      case SETUP_RETEST:          return g_cfg.tm_target_breakout_pips;
      case SETUP_REJECTION:       return g_cfg.tm_target_rejection_pips;
      case SETUP_FAKEOUT:         return g_cfg.tm_target_rejection_pips;
      case SETUP_FAILED_AUCTION:  return g_cfg.tm_target_reversion_pips;
      case SETUP_ENGULFING_ACCEL: return g_cfg.tm_target_breakout_pips;
      case SETUP_MEAN_REVERSION:  return g_cfg.tm_target_reversion_pips;
      case SETUP_MOMENTUM_CONT:   return g_cfg.setup_momentum_cont_target_pips;
   }
   return g_cfg.tm_target_reversion_pips;
}

class IntentBook {
private:
   ExecutionIntent  m_intents[32];
   int              m_count;

public:
   void Init() {
      m_count = 0;
      for(int i = 0; i < 32; i++) ZeroMemory(m_intents[i]);
   }

   //--- Submit a new intent. Called by Setups engine with raw fields
   //    (avoids circular include on the SetupCandidate type).
   void Submit(const ENUM_SETUP_FAMILY family,
               const ENUM_TRADE_DIR   dir,
               const string           zone_id,
               const double           setup_score,
               const double           kl_weight,
               const double           kl_distance_pips,
               const int              echo_count,
               const FeatureSnapshot &fs) {
      if(m_count >= 32) {
         // book full — log a visible telemetry row instead of silent drop
         if(g_cfg.tel_log_intents) {
            g_tel.WriteIntent(fs.ts_ms, "dropped_full",
                              SetupFamilyName(family), DirName(dir), "dropped",
                              0.0, 0.0, 0.0, 0, setup_score);
         }
         return;
      }

      double stop_pips   = IntentStopPipsForFamily(family);
      double target_pips = IntentTargetPipsForFamily(family);
      double entry       = (dir == DIR_LONG) ? fs.ask : fs.bid;
      double sl          = (dir == DIR_LONG) ? entry - PriceFromPips(stop_pips)
                                             : entry + PriceFromPips(stop_pips);
      double tp          = (dir == DIR_LONG) ? entry + PriceFromPips(target_pips)
                                             : entry - PriceFromPips(target_pips);

      ExecutionIntent e;
      e.id              = MakeId("int");
      e.family          = family;
      e.direction       = dir;
      e.zone_id         = zone_id;
      e.trigger_price   = entry;
      e.hidden_sl       = sl;
      e.hidden_tp       = tp;
      e.created_ms      = fs.ts_ms;
      e.ttl_ms          = g_cfg.exec_intent_ttl_ms;
      e.setup_score     = setup_score;
      e.state           = INTENT_CREATED;
      e.tolerance       = PriceFromPips(g_cfg.exec_trigger_tolerance_pips);
      e.kl_weight       = kl_weight;
      e.kl_distance_pips= kl_distance_pips;
      e.echo_count      = echo_count;
      m_intents[m_count++] = e;

      if(g_cfg.tel_log_intents) {
         g_tel.WriteIntent(fs.ts_ms, e.id, SetupFamilyName(family),
                           DirName(dir), "created", e.trigger_price,
                           e.hidden_sl, e.hidden_tp, e.ttl_ms, setup_score);
      }
   }

   //--- per-tick driver. Finds the first intent whose trigger is hit,
   //    returns its index and marks it TRIGGERED. Also expires old
   //    intents. Returns -1 if nothing triggered this tick.
   int ProcessTick(const FeatureSnapshot &fs) {
      int trigger_idx = -1;
      for(int i = 0; i < m_count; i++) {
         if(m_intents[i].state != INTENT_CREATED) continue;
         if(fs.ts_ms - m_intents[i].created_ms > m_intents[i].ttl_ms) {
            m_intents[i].state = INTENT_EXPIRED;
            if(g_cfg.tel_log_intents) {
               g_tel.WriteIntent(fs.ts_ms, m_intents[i].id,
                                 SetupFamilyName(m_intents[i].family),
                                 DirName(m_intents[i].direction), "expired",
                                 m_intents[i].trigger_price, m_intents[i].hidden_sl,
                                 m_intents[i].hidden_tp, m_intents[i].ttl_ms,
                                 m_intents[i].setup_score);
            }
            continue;
         }
         double ref = (m_intents[i].direction == DIR_LONG) ? fs.ask : fs.bid;
         if(MathAbs(ref - m_intents[i].trigger_price) <= m_intents[i].tolerance) {
            if(trigger_idx < 0) trigger_idx = i;
         }
      }
      // Compact expired intents unconditionally so the book never leaks
      // dead entries between triggers. The triggered slot (still in
      // INTENT_CREATED state here) is preserved and its index is remapped
      // so the caller's Get(idx) / MarkTriggered(idx) contract still holds.
      int w = 0;
      for(int i = 0; i < m_count; i++) {
         if(m_intents[i].state == INTENT_EXPIRED) continue;
         if(i == trigger_idx) trigger_idx = w;
         if(w != i) m_intents[w] = m_intents[i];
         w++;
      }
      m_count = w;
      return trigger_idx;
   }

   int Count() const { return m_count; }
   bool Get(const int idx, ExecutionIntent &out) const {
      if(idx < 0 || idx >= m_count) return false;
      out = m_intents[idx];
      return true;
   }
   void MarkTriggered(const int idx, const FeatureSnapshot &fs) {
      if(idx < 0 || idx >= m_count) return;
      m_intents[idx].state = INTENT_TRIGGERED;
      if(g_cfg.tel_log_intents) {
         g_tel.WriteIntent(fs.ts_ms, m_intents[idx].id,
                           SetupFamilyName(m_intents[idx].family),
                           DirName(m_intents[idx].direction), "triggered",
                           m_intents[idx].trigger_price, m_intents[idx].hidden_sl,
                           m_intents[idx].hidden_tp, m_intents[idx].ttl_ms,
                           m_intents[idx].setup_score);
      }
   }

   void Compact() {
      int w = 0;
      for(int i = 0; i < m_count; i++) {
         if(m_intents[i].state == INTENT_CREATED) {
            if(w != i) m_intents[w] = m_intents[i];
            w++;
         }
      }
      m_count = w;
   }

   //--- Strict concurrency: once one intent commits to execution, drop
   //    every other CREATED intent in the book. Under async fills the
   //    broker ack for the committed intent can take ~10s; during that
   //    window a second intent would otherwise trigger freely because
   //    TradeManagement::m_trades is still empty. Clearing here kills
   //    that race at the source. Each cancelled intent gets an EXPIRED
   //    telemetry row so the audit trail shows WHY it didn't fire.
   void CancelAllCreated(const string reason, const FeatureSnapshot &fs) {
      for(int i = 0; i < m_count; i++) {
         if(m_intents[i].state != INTENT_CREATED) continue;
         m_intents[i].state = INTENT_EXPIRED;
         if(g_cfg.tel_log_intents) {
            g_tel.WriteIntent(fs.ts_ms, m_intents[i].id,
                              SetupFamilyName(m_intents[i].family),
                              DirName(m_intents[i].direction),
                              StringFormat("cancelled:%s", reason),
                              m_intents[i].trigger_price,
                              m_intents[i].hidden_sl,
                              m_intents[i].hidden_tp,
                              m_intents[i].ttl_ms,
                              m_intents[i].setup_score);
         }
      }
      Compact();
   }
};

IntentBook g_intents;

#endif // __FILOHFT_EXECUTION_INTENT_MQH__
