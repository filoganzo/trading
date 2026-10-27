//+------------------------------------------------------------------+
//|  FiloHFT/TradeManagement.mqh                                     |
//|  Owns the lifecycle of open positions. Implements HIDDEN SL/TP  |
//|  by tracking the desired exit prices in-memory and closing      |
//|  positions via broker calls only when price reaches them.      |
//|                                                                 |
//|  Features:                                                      |
//|    - break-even transition with buffer                         |
//|    - volatility-aware trailing (simple for first iteration)    |
//|    - concurrency: only allow same-direction stacking if        |
//|      prior positions are >= BE                                 |
//|    - per-family target/stop templates from config              |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_TRADE_MANAGEMENT_MQH__
#define __FILOHFT_TRADE_MANAGEMENT_MQH__

#include <FiloHFT/Common.mqh>
#include <FiloHFT/Config.mqh>
#include <FiloHFT/Telemetry.mqh>
#include <FiloHFT/Features.mqh>
#include <FiloHFT/ExecutionIntent.mqh>
#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>

struct ManagedTrade {
   long              ticket;
   ENUM_TRADE_DIR    direction;
   ENUM_SETUP_FAMILY family;
   string            zone_id;        // KL zone that generated the setup
   double            entry_price;
   double            lots;           // real position volume, used to compute realized_usd
   double            hidden_sl;      // current virtual stop
   double            hidden_tp;      // current virtual target
   double            initial_sl;     // at entry
   double            initial_tp;
   bool              at_breakeven;
   bool              trailing;
   long              opened_ms;
   long              last_mod_ms;
   long              be_candidate_ms; // first ts where gain crossed BE trigger
   string            intent_id;      // linkback to telemetry
};

class TradeManagement {
private:
   ManagedTrade      m_trades[16];
   int               m_count;
   CTrade            m_trade;
   CPositionInfo     m_pos;

   // last-open timestamp per direction, used by CanStack() to enforce
   // the hard time-gap floor on same-direction stacking. Stamped in
   // AttachIntent, so it reflects real fills (not intents that never
   // triggered).
   long              m_last_open_ms_long;
   long              m_last_open_ms_short;

   // last-close state, used by PassesFillCooldown() to kill
   // stop-and-reverse ping-ponging. Stamped every time _Step closes
   // a position (hidden SL/TP) or _Reconcile drops an externally
   // closed position. m_last_close_was_stop distinguishes SL-path
   // (true) from TP-path / broker-close (false); only SL-path
   // triggers the stricter opposite-direction cooldown.
   long              m_last_close_ms;
   bool              m_last_close_was_stop;
   ENUM_TRADE_DIR    m_last_close_dir;

   int _FindByTicket(const long ticket) const {
      for(int i = 0; i < m_count; i++)
         if(m_trades[i].ticket == ticket) return i;
      return -1;
   }

   void _LogTradeEvent(const ManagedTrade &t, const string ev,
                       const FeatureSnapshot &fs,
                       const double price) {
      if(!g_cfg.tel_log_trades) return;
      double unreal_pips = (t.direction == DIR_LONG)
         ? PipsFromPrice(price - t.entry_price)
         : PipsFromPrice(t.entry_price - price);
      // realized_usd derived from the live pip P&L and the actual
      // position volume. For non-"close" events this is the running
      // mark-to-market (still useful for the dashboard).
      double lots = (t.lots > 0.0) ? t.lots : g_cfg.fixed_lot;
      double realized_usd = unreal_pips
                          * g_cfg.pip_in_points
                          * g_cfg.usd_per_point_per_lot
                          * lots;
      g_tel.WriteTrade(fs.ts_ms, t.ticket, ev, DirName(t.direction),
                       price, t.hidden_sl, t.hidden_tp, unreal_pips, realized_usd);
   }

   //--- reconcile our m_trades list with actual broker positions
   void _Reconcile(const FeatureSnapshot &fs) {
      // mark all as "unseen" then re-scan positions
      bool seen[16];
      for(int i = 0; i < 16; i++) seen[i] = false;

      int total = PositionsTotal();
      for(int i = 0; i < total; i++) {
         if(!m_pos.SelectByIndex(i)) continue;
         if(m_pos.Symbol() != _Symbol) continue;
         if(m_pos.Magic()  != 20260413) continue;

         long tk = (long)m_pos.Ticket();
         int idx = _FindByTicket(tk);
         if(idx < 0) {
            // new position we don't know about — adopt it
            if(m_count >= 16) continue;
            ManagedTrade t;
            ZeroMemory(t);
            t.ticket      = tk;
            t.direction   = (m_pos.PositionType() == POSITION_TYPE_BUY) ? DIR_LONG : DIR_SHORT;
            t.family      = SETUP_NONE;
            t.entry_price = m_pos.PriceOpen();
            t.lots        = m_pos.Volume();
            // use default stops if we didn't set them (rare — manual position?)
            double stp = PriceFromPips(g_cfg.tm_stop_breakout_pips);
            double tgp = PriceFromPips(g_cfg.tm_target_breakout_pips);
            t.hidden_sl = (t.direction == DIR_LONG) ? t.entry_price - stp : t.entry_price + stp;
            t.hidden_tp = (t.direction == DIR_LONG) ? t.entry_price + tgp : t.entry_price - tgp;
            t.initial_sl = t.hidden_sl;
            t.initial_tp = t.hidden_tp;
            t.opened_ms  = fs.ts_ms;
            m_trades[m_count++] = t;
            _LogTradeEvent(m_trades[m_count - 1], "open", fs, t.entry_price);
            idx = m_count - 1;
         }
         seen[idx] = true;
      }

      // any entry not seen = closed by broker (hit real SL/TP or our close)
      int w = 0;
      for(int i = 0; i < m_count; i++) {
         if(seen[i]) {
            if(w != i) m_trades[w] = m_trades[i];
            w++;
         } else {
            _LogTradeEvent(m_trades[i], "close", fs,
                           (m_trades[i].direction == DIR_LONG) ? fs.bid : fs.ask);
            // external / reconciled close — treat as non-stop for the
            // stricter opposite cooldown (we can't know why the broker
            // side removed it), but still stamp the generic cooldown
            // so we don't instantly re-enter.
            m_last_close_ms       = fs.ts_ms;
            m_last_close_was_stop = false;
            m_last_close_dir      = m_trades[i].direction;
         }
      }
      m_count = w;
   }

   //--- step a single managed trade: evaluate hidden SL/TP, BE, trailing.
   //    Returns true if this trade was closed this step.
   bool _Step(const int i, const FeatureSnapshot &fs) {
      ManagedTrade t = m_trades[i];

      double mark = (t.direction == DIR_LONG) ? fs.bid : fs.ask;
      double gain_pips = (t.direction == DIR_LONG)
         ? PipsFromPrice(mark - t.entry_price)
         : PipsFromPrice(t.entry_price - mark);

      // exit: hidden SL hit
      if((t.direction == DIR_LONG  && mark <= t.hidden_sl) ||
         (t.direction == DIR_SHORT && mark >= t.hidden_sl)) {
         m_trade.PositionClose(t.ticket);
         _LogTradeEvent(t, "close", fs, mark);
         m_last_close_ms       = fs.ts_ms;
         m_last_close_was_stop = true;
         m_last_close_dir      = t.direction;
         return true;
      }
      // exit: hidden TP hit
      if((t.direction == DIR_LONG  && mark >= t.hidden_tp) ||
         (t.direction == DIR_SHORT && mark <= t.hidden_tp)) {
         m_trade.PositionClose(t.ticket);
         _LogTradeEvent(t, "close", fs, mark);
         m_last_close_ms       = fs.ts_ms;
         m_last_close_was_stop = false;
         m_last_close_dir      = t.direction;
         return true;
      }

      // break-even transition — requires gain to STAY above trigger for
      // BE_SUSTAIN_MS before pinning. A single-tick spike that instantly
      // reverses must not flip at_breakeven=true, because CanOpen uses
      // at_breakeven to allow same-direction stacking.
      const long BE_SUSTAIN_MS = 500;
      if(!t.at_breakeven) {
         if(gain_pips >= g_cfg.tm_be_trigger_pips) {
            if(t.be_candidate_ms == 0) {
               t.be_candidate_ms = fs.ts_ms;
            } else if(fs.ts_ms - t.be_candidate_ms >= BE_SUSTAIN_MS) {
               double buf = PriceFromPips(g_cfg.tm_be_buffer_pips);
               t.hidden_sl = (t.direction == DIR_LONG) ? t.entry_price + buf
                                                       : t.entry_price - buf;
               t.at_breakeven = true;
               t.be_candidate_ms = 0;
               _LogTradeEvent(t, "be", fs, mark);
            }
         } else {
            t.be_candidate_ms = 0;
         }
      }

      // trailing
      if(gain_pips >= g_cfg.tm_trail_start_pips) {
         double dist = PriceFromPips(g_cfg.tm_trail_distance_pips);
         double new_sl;
         if(t.direction == DIR_LONG) {
            new_sl = mark - dist;
            if(new_sl > t.hidden_sl) { t.hidden_sl = new_sl; t.trailing = true; }
         } else {
            new_sl = mark + dist;
            if(new_sl < t.hidden_sl) { t.hidden_sl = new_sl; t.trailing = true; }
         }
         if(t.trailing && fs.ts_ms - t.last_mod_ms >= g_cfg.tm_management_poll_ms) {
            _LogTradeEvent(t, "trail", fs, mark);
            t.last_mod_ms = fs.ts_ms;
         }
      }

      m_trades[i] = t;
      return false;
   }

public:
   void Init() {
      m_count = 0;
      for(int i = 0; i < 16; i++) ZeroMemory(m_trades[i]);
      m_trade.SetExpertMagicNumber(20260413);
      m_trade.SetAsyncMode(false);
      m_last_open_ms_long  = 0;
      m_last_open_ms_short = 0;
      m_last_close_ms       = 0;
      m_last_close_was_stop = false;
      m_last_close_dir      = DIR_NONE;
   }

   int OpenCount() const { return m_count; }

   //--- Called right after Execution confirms the entry fill. Under the
   //    async execution model AttachIntent may run AFTER _Reconcile has
   //    already adopted the broker position with default metadata, so
   //    this method both (a) adopts a fresh position and (b) corrects a
   //    previously-adopted one by overwriting family/zone_id/SL/TP with
   //    the values from the originating intent.
   void AttachIntent(const ExecutionIntent &e, const FeatureSnapshot &fs) {
      int total = PositionsTotal();
      for(int i = total - 1; i >= 0; i--) {
         if(!m_pos.SelectByIndex(i)) continue;
         if(m_pos.Symbol() != _Symbol) continue;
         if(m_pos.Magic()  != 20260413) continue;

         long tk = (long)m_pos.Ticket();
         int existing = _FindByTicket(tk);

         if(existing >= 0) {
            // Reconcile-first path: position was adopted with defaults.
            // Correct it in-place so hidden SL/TP + family metadata
            // reflect the actual setup that fired.
            ManagedTrade t = m_trades[existing];
            t.family     = e.family;
            t.zone_id    = e.zone_id;
            t.hidden_sl  = e.hidden_sl;
            t.hidden_tp  = e.hidden_tp;
            t.initial_sl = e.hidden_sl;
            t.initial_tp = e.hidden_tp;
            t.intent_id  = e.id;
            if(t.lots <= 0.0) t.lots = m_pos.Volume();
            m_trades[existing] = t;
            if(e.direction == DIR_LONG)  m_last_open_ms_long  = fs.ts_ms;
            if(e.direction == DIR_SHORT) m_last_open_ms_short = fs.ts_ms;
            _LogTradeEvent(m_trades[existing], "attach", fs, t.entry_price);
            return;
         }

         if(m_count >= 16) break;
         ManagedTrade t;
         ZeroMemory(t);
         t.ticket      = tk;
         t.direction   = e.direction;
         t.family      = e.family;
         t.zone_id     = e.zone_id;
         t.entry_price = m_pos.PriceOpen();
         t.lots        = m_pos.Volume();
         t.hidden_sl   = e.hidden_sl;
         t.hidden_tp   = e.hidden_tp;
         t.initial_sl  = e.hidden_sl;
         t.initial_tp  = e.hidden_tp;
         t.opened_ms   = fs.ts_ms;
         t.last_mod_ms = fs.ts_ms;
         t.intent_id   = e.id;
         m_trades[m_count++] = t;
         if(e.direction == DIR_LONG)  m_last_open_ms_long  = fs.ts_ms;
         if(e.direction == DIR_SHORT) m_last_open_ms_short = fs.ts_ms;
         _LogTradeEvent(m_trades[m_count - 1], "open", fs, t.entry_price);
         return;
      }
   }

   //--- concurrency check — can we open a new trade in this direction?
   bool CanOpen(const ENUM_TRADE_DIR dir) const {
      if(m_count >= g_cfg.max_concurrent_trades) return false;
      for(int i = 0; i < m_count; i++) {
         // same direction: wait for the prior position to reach BE so its
         // SL is locked above entry before stacking another unit.
         if(m_trades[i].direction == dir && !m_trades[i].at_breakeven) {
            return false;
         }
         // opposite direction: blocked outright. In high-vol conditions
         // the hedged outcome is one-TPs-one-SLs-with-slippage, which
         // cancels any edge the winning side produced. Shadow trades are
         // unconstrained (no CanOpen call) so ML still sees the signals.
         if(m_trades[i].direction != dir) {
            return false;
         }
      }
      return true;
   }

   //--- same-direction anti-pyramid gate. Returns true if a new entry
   //    in `dir` from (family, zone_id) is permitted given what is
   //    already open. A concrete reason string is returned via
   //    `out_reason` so Setups can log it on blocked candidates.
   //
   //    Rule (all must hold, or there are no same-direction opens):
   //      1. Hard time floor: at least tm_stack_min_gap_ms must have
   //         elapsed since the most recent fill in this direction.
   //      2. Diverse edge: if tm_stack_require_diverse_edge is on,
   //         the new (family, zone_id) must differ from EVERY open
   //         same-direction position in family OR zone_id. Same
   //         family at a different zone is OK; same zone with a
   //         different family is OK; same family AND same zone is
   //         refused (it's the same edge re-firing).
   //
   //    Note: this does NOT re-check the at_breakeven prerequisite,
   //    CanOpen() already handles that.
   bool CanStack(const ENUM_TRADE_DIR    dir,
                 const ENUM_SETUP_FAMILY family,
                 const string            zone_id,
                 const long              now_ms,
                 string                 &out_reason) const {
      out_reason = "";
      bool has_same_dir = false;
      for(int i = 0; i < m_count; i++) {
         if(m_trades[i].direction == dir) { has_same_dir = true; break; }
      }
      if(!has_same_dir) return true;  // no stack happening, nothing to gate

      long last = (dir == DIR_LONG) ? m_last_open_ms_long : m_last_open_ms_short;
      if(g_cfg.tm_stack_min_gap_ms > 0 && last > 0 &&
         (now_ms - last) < (long)g_cfg.tm_stack_min_gap_ms) {
         out_reason = StringFormat("stack gap %ldms < %dms",
                                   now_ms - last, g_cfg.tm_stack_min_gap_ms);
         return false;
      }

      if(g_cfg.tm_stack_require_diverse_edge) {
         for(int i = 0; i < m_count; i++) {
            if(m_trades[i].direction != dir) continue;
            if(m_trades[i].family == family && m_trades[i].zone_id == zone_id) {
               out_reason = StringFormat("stack same edge family=%s zone=%s",
                                         SetupFamilyName(family), zone_id);
               return false;
            }
         }
      }
      return true;
   }

   //--- post-fill cooldown gate. Returns true if the new entry in
   //    `dir` is allowed given the most recent close. Fails the
   //    candidate if we are still inside:
   //      (a) tm_post_fill_cooldown_ms after ANY close (generic), OR
   //      (b) tm_post_stop_opposite_cooldown_ms after a STOP close
   //          when `dir` is the inverse of the closed trade (SAR
   //          ping-pong guard).
   //    out_reason carries a concrete block reason for telemetry.
   bool PassesFillCooldown(const ENUM_TRADE_DIR dir,
                           const long           now_ms,
                           string              &out_reason) const {
      out_reason = "";
      if(m_last_close_ms <= 0) return true;
      long dt = now_ms - m_last_close_ms;
      if(dt < 0) return true;

      if(g_cfg.tm_post_fill_cooldown_ms > 0 &&
         dt < (long)g_cfg.tm_post_fill_cooldown_ms) {
         out_reason = StringFormat("post-fill cooldown %ldms < %dms",
                                   dt, g_cfg.tm_post_fill_cooldown_ms);
         return false;
      }
      if(m_last_close_was_stop &&
         g_cfg.tm_post_stop_opposite_cooldown_ms > 0 &&
         dir != m_last_close_dir &&
         dt < (long)g_cfg.tm_post_stop_opposite_cooldown_ms) {
         out_reason = StringFormat("post-stop SAR guard %ldms < %dms",
                                   dt, g_cfg.tm_post_stop_opposite_cooldown_ms);
         return false;
      }
      return true;
   }

   //--- called on every tick (cheap) + on the management poll timer
   void Step(const FeatureSnapshot &fs) {
      _Reconcile(fs);
      int w = 0;
      for(int i = 0; i < m_count; i++) {
         bool closed = _Step(i, fs);
         if(!closed) {
            if(w != i) m_trades[w] = m_trades[i];
            w++;
         }
      }
      m_count = w;
   }
};

TradeManagement g_tm;

#endif // __FILOHFT_TRADE_MANAGEMENT_MQH__
