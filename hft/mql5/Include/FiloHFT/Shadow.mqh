//+------------------------------------------------------------------+
//|  FiloHFT/Shadow.mqh                                              |
//|  "Fake trade" bookkeeping. Every accepted setup opens a virtual |
//|  position that runs alongside real execution, with the same     |
//|  per-family target/stop distances. TP/SL hits are simulated     |
//|  tick-by-tick from bid/ask without ever touching the broker.   |
//|                                                                 |
//|  The point is to capture hypothetical outcomes of every signal |
//|  the engine detects — even when InpEnableTrading=false, when    |
//|  safety halts, or when execution blocks — so ML training and    |
//|  entry-parameter calibration can use the full signal stream     |
//|  instead of only the (possibly tiny) live-fill subset.          |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_SHADOW_MQH__
#define __FILOHFT_SHADOW_MQH__

#include <FiloHFT/Common.mqh>
#include <FiloHFT/Config.mqh>
#include <FiloHFT/Telemetry.mqh>
#include <FiloHFT/Features.mqh>
#include <FiloHFT/Correlation.mqh>

#define SHADOW_MAX_OPEN 64
#define SHADOW_TTL_MS   180000   // 3 minutes — cap on virtual hold time

struct ShadowPosition {
   bool              active;
   ENUM_SETUP_FAMILY family;
   ENUM_TRADE_DIR    direction;
   double            entry_price;
   double            hidden_sl;
   double            hidden_tp;
   long              open_ts_ms;
   // per-entry feature snapshot (captured at open time for ML)
   string            kl_zone_id;
   double            kl_weight;
   double            kl_distance_pips;
   double            intrinsic_score;
   double            entry_spread_pips;
   double            entry_velocity;
   double            entry_chan_pos_norm;
   double            entry_corr_xagusd;
   double            entry_corr_dxy;
   // Echo tag snapshot at open time. Propagated to shadow_trades.csv so
   // downstream analysis can split outcomes on echo_count==0 vs >=1.
   int               echo_count;
};

class ShadowTrades {
private:
   ShadowPosition m_pos[SHADOW_MAX_OPEN];
   int            m_open_count;
   long           m_total_opened;
   long           m_total_closed;

   //--- per-family target/stop lookup, mirrors TradeManagement.
   void _TargetStopPips(const ENUM_SETUP_FAMILY f, double &tgt, double &stp) {
      switch(f) {
         case SETUP_BREAKOUT:
         case SETUP_RETEST:
         case SETUP_ENGULFING_ACCEL:
            tgt = g_cfg.tm_target_breakout_pips;
            stp = g_cfg.tm_stop_breakout_pips;
            return;
         case SETUP_REJECTION:
            tgt = g_cfg.tm_target_rejection_pips;
            stp = g_cfg.tm_stop_rejection_pips;
            return;
         case SETUP_MOMENTUM_CONT:
            tgt = g_cfg.setup_momentum_cont_target_pips;
            stp = g_cfg.setup_momentum_cont_stop_pips;
            return;
         case SETUP_FAKEOUT:
         case SETUP_FAILED_AUCTION:
         case SETUP_MEAN_REVERSION:
         default:
            tgt = g_cfg.tm_target_reversion_pips;
            stp = g_cfg.tm_stop_reversion_pips;
            return;
      }
   }

   void _Close(const int i, const string reason, const double exit_px,
               const long exit_ts) {
      ShadowPosition p = m_pos[i];
      double realized_pips = (p.direction == DIR_LONG)
         ? PipsFromPrice(exit_px - p.entry_price)
         : PipsFromPrice(p.entry_price - exit_px);
      double realized_usd = realized_pips
                          * g_cfg.pip_in_points
                          * g_cfg.usd_per_point_per_lot
                          * g_cfg.fixed_lot;

      g_tel.WriteShadow(
         p.open_ts_ms, exit_ts,
         SetupFamilyName(p.family), DirName(p.direction),
         p.entry_price, exit_px,
         realized_pips, realized_usd,
         reason,
         p.kl_zone_id, p.kl_weight, p.kl_distance_pips,
         p.intrinsic_score,
         p.entry_spread_pips, p.entry_velocity,
         p.entry_chan_pos_norm,
         p.entry_corr_xagusd, p.entry_corr_dxy,
         p.echo_count);

      m_pos[i].active = false;
      m_open_count--;
      m_total_closed++;
   }

public:
   void Init() {
      for(int i = 0; i < SHADOW_MAX_OPEN; i++) ZeroMemory(m_pos[i]);
      m_open_count   = 0;
      m_total_opened = 0;
      m_total_closed = 0;
   }

   int OpenCount()   const { return m_open_count; }
   long TotalOpened() const { return m_total_opened; }
   long TotalClosed() const { return m_total_closed; }

   //--- Open a virtual position for an accepted setup.
   //    Entry price uses the side we would pay: ask for longs, bid for shorts.
   //    Returns true if inserted (false if the slot array is full).
   bool Open(const ENUM_SETUP_FAMILY family,
             const ENUM_TRADE_DIR    dir,
             const string            kl_zone_id,
             const double            kl_weight,
             const double            kl_distance_pips,
             const double            intrinsic_score,
             const int               echo_count,
             const FeatureSnapshot  &fs,
             const CorrState        &cs) {
      if(m_open_count >= SHADOW_MAX_OPEN) return false;

      int slot = -1;
      for(int i = 0; i < SHADOW_MAX_OPEN; i++) {
         if(!m_pos[i].active) { slot = i; break; }
      }
      if(slot < 0) return false;

      double tgt_pips = 0, stp_pips = 0;
      _TargetStopPips(family, tgt_pips, stp_pips);
      double tgt_px = PriceFromPips(tgt_pips);
      double stp_px = PriceFromPips(stp_pips);

      double entry = (dir == DIR_LONG) ? fs.ask : fs.bid;

      ShadowPosition p;
      ZeroMemory(p);
      p.active              = true;
      p.family              = family;
      p.direction           = dir;
      p.entry_price         = entry;
      p.hidden_sl           = (dir == DIR_LONG) ? entry - stp_px : entry + stp_px;
      p.hidden_tp           = (dir == DIR_LONG) ? entry + tgt_px : entry - tgt_px;
      p.open_ts_ms          = fs.ts_ms;
      p.kl_zone_id          = kl_zone_id;
      p.kl_weight           = kl_weight;
      p.kl_distance_pips    = kl_distance_pips;
      p.intrinsic_score     = intrinsic_score;
      p.entry_spread_pips   = fs.spread_pips;
      p.entry_velocity      = fs.velocity_pips_per_sec;
      p.entry_chan_pos_norm = fs.chan_pos_norm;
      p.entry_corr_xagusd   = cs.agreement_xagusd;
      p.entry_corr_dxy      = cs.agreement_dxy;
      p.echo_count          = echo_count;

      m_pos[slot] = p;
      m_open_count++;
      m_total_opened++;
      return true;
   }

   //--- Tick step: check every open virtual for TP/SL/TTL. Exit sides
   //    mirror the open side (close longs at bid, shorts at ask).
   void Step(const FeatureSnapshot &fs) {
      if(m_open_count <= 0) return;
      for(int i = 0; i < SHADOW_MAX_OPEN; i++) {
         if(!m_pos[i].active) continue;
         ShadowPosition p = m_pos[i];
         double mark = (p.direction == DIR_LONG) ? fs.bid : fs.ask;

         // TP
         if((p.direction == DIR_LONG  && mark >= p.hidden_tp) ||
            (p.direction == DIR_SHORT && mark <= p.hidden_tp)) {
            _Close(i, "tp", p.hidden_tp, fs.ts_ms);
            continue;
         }
         // SL
         if((p.direction == DIR_LONG  && mark <= p.hidden_sl) ||
            (p.direction == DIR_SHORT && mark >= p.hidden_sl)) {
            _Close(i, "sl", p.hidden_sl, fs.ts_ms);
            continue;
         }
         // TTL
         if(fs.ts_ms - p.open_ts_ms >= SHADOW_TTL_MS) {
            _Close(i, "ttl", mark, fs.ts_ms);
            continue;
         }
      }
   }

   //--- Force-close everything at current mark (called on shutdown).
   void CloseAll(const FeatureSnapshot &fs) {
      for(int i = 0; i < SHADOW_MAX_OPEN; i++) {
         if(!m_pos[i].active) continue;
         double mark = (m_pos[i].direction == DIR_LONG) ? fs.bid : fs.ask;
         _Close(i, "shutdown", mark, fs.ts_ms);
      }
   }
};

ShadowTrades g_shadow;

#endif // __FILOHFT_SHADOW_MQH__
