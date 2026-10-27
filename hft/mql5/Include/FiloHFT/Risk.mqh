//+------------------------------------------------------------------+
//|  FiloHFT/Risk.mqh                                                |
//|  Hard risk limits with TIME-BASED cooldowns.                     |
//|                                                                  |
//|  Unlike a latching kill-switch, every breach sets a deadline     |
//|  (m_halted_until_ms). Once the deadline passes, AllowNewEntry    |
//|  returns true again — the halt auto-clears and a "cleared" row   |
//|  is logged. Cooldown durations match the semantics of the        |
//|  offending bucket:                                               |
//|    - per-trade breach  → 60s cooldown                            |
//|    - per-minute breach → until the minute boundary rolls         |
//|    - per-hour breach   → until the hour boundary rolls           |
//|    - per-day breach    → until the day boundary rolls            |
//|    - reject count      → 60s cooldown; count decays 1/60s        |
//|                                                                  |
//|  A manual Reset() is still available (called when the override   |
//|  file is edited) so the user can unilaterally unstick state.     |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_RISK_MQH__
#define __FILOHFT_RISK_MQH__

#include <FiloHFT/Common.mqh>
#include <FiloHFT/Config.mqh>
#include <FiloHFT/Telemetry.mqh>

class Risk {
private:
   double   m_loss_minute;
   double   m_loss_hour;
   double   m_loss_day;
   datetime m_bucket_minute;
   datetime m_bucket_hour;
   datetime m_bucket_day;

   int      m_reject_count;
   long     m_last_reject_decay_ms;

   long     m_halted_until_ms;
   string   m_halt_reason;

   // direction-aware post-loss cooldown (A). A stop-out in direction X
   // pauses only that direction for cfg.risk_post_loss_cooldown_ms.
   // The opposite direction stays tradeable — we are NOT halting the
   // whole book, only refusing to double-down on the same losing side.
   long     m_long_cooldown_until_ms;
   long     m_short_cooldown_until_ms;

   void _RollBuckets(const datetime now) {
      datetime minute = (datetime)((long)now - (long)now % 60);
      datetime hour   = (datetime)((long)now - (long)now % 3600);
      datetime day    = (datetime)((long)now - (long)now % 86400);
      if(minute != m_bucket_minute) { m_bucket_minute = minute; m_loss_minute = 0.0; }
      if(hour   != m_bucket_hour)   { m_bucket_hour   = hour;   m_loss_hour   = 0.0; }
      if(day    != m_bucket_day)    { m_bucket_day    = day;    m_loss_day    = 0.0; }
   }

   void _HaltUntil(const long until_ms, const string reason,
                   const string limit, const double val, const double thr) {
      if(until_ms > m_halted_until_ms) {
         m_halted_until_ms = until_ms;
         m_halt_reason     = reason;
      }
      g_tel.WriteRisk(NowMs(), limit, val, thr, "halt");
      PrintFormat("[Risk] HALT %s=%.2f exceeded %.2f — cooling down (%s)",
                  limit, val, thr, reason);
   }

public:
   void Init() {
      m_loss_minute = 0.0;
      m_loss_hour   = 0.0;
      m_loss_day    = 0.0;
      m_bucket_minute = 0;
      m_bucket_hour   = 0;
      m_bucket_day    = 0;
      m_reject_count  = 0;
      m_last_reject_decay_ms = 0;
      m_halted_until_ms = 0;
      m_halt_reason = "";
      m_long_cooldown_until_ms  = 0;
      m_short_cooldown_until_ms = 0;
   }

   //--- periodic upkeep — call from OnTimer. Rolls loss buckets,
   //    decays reject count, and clears expired halts.
   void Step() {
      long now = NowMs();
      _RollBuckets(TimeCurrent());

      if(m_reject_count > 0) {
         if(m_last_reject_decay_ms == 0) m_last_reject_decay_ms = now;
         if(now - m_last_reject_decay_ms >= 60000) {
            m_reject_count--;
            m_last_reject_decay_ms = now;
         }
      }

      if(m_halted_until_ms > 0 && now >= m_halted_until_ms) {
         string reason = m_halt_reason;
         m_halted_until_ms = 0;
         m_halt_reason = "";
         g_tel.WriteRisk(now, reason, 0.0, 0.0, "cleared");
         PrintFormat("[Risk] halt cleared (%s)", reason);
      }

      // direction-aware cooldowns expire silently-with-telemetry. A
      // per-direction "cleared" row is useful on the dashboard to
      // confirm the side has been re-enabled after a losing trade.
      if(m_long_cooldown_until_ms > 0 && now >= m_long_cooldown_until_ms) {
         m_long_cooldown_until_ms = 0;
         g_tel.WriteRisk(now, "dir_cooldown_long", 0.0, 0.0, "cleared");
      }
      if(m_short_cooldown_until_ms > 0 && now >= m_short_cooldown_until_ms) {
         m_short_cooldown_until_ms = 0;
         g_tel.WriteRisk(now, "dir_cooldown_short", 0.0, 0.0, "cleared");
      }
   }

   //--- called by Execution when a reject arrives
   void NotifyReject() {
      m_reject_count++;
      m_last_reject_decay_ms = NowMs();
      if(m_reject_count >= g_cfg.risk_max_order_rejects) {
         _HaltUntil(NowMs() + 60000, "rejects",
                    "max_order_rejects",
                    (double)m_reject_count,
                    (double)g_cfg.risk_max_order_rejects);
      }
   }

   //--- called on every trade close with realized PnL in USD
   //    `dir` is the direction of the position that just closed
   //    (DIR_LONG or DIR_SHORT). A losing close arms a per-direction
   //    cooldown so we stop doubling-down on the same losing side —
   //    the exact failure mode from the 4-consecutive-losing-buys
   //    pattern at 20:05–20:08. DIR_NONE preserves legacy behavior
   //    (no direction cooldown) for callers that don't know the side.
   void NotifyClose(const double realized_usd, const ENUM_TRADE_DIR dir = DIR_NONE) {
      datetime now_t = TimeCurrent();
      _RollBuckets(now_t);
      if(realized_usd < 0) {
         double loss = -realized_usd;
         m_loss_minute += loss;
         m_loss_hour   += loss;
         m_loss_day    += loss;

         long now_ms = NowMs();

         if(dir != DIR_NONE && g_cfg.risk_post_loss_cooldown_ms > 0) {
            long until = now_ms + (long)g_cfg.risk_post_loss_cooldown_ms;
            if(dir == DIR_LONG && until > m_long_cooldown_until_ms) {
               m_long_cooldown_until_ms = until;
               g_tel.WriteRisk(now_ms, "dir_cooldown_long", loss,
                               (double)g_cfg.risk_post_loss_cooldown_ms, "halt");
               PrintFormat("[Risk] long cooldown armed (loss=%.2f) for %dms",
                           loss, g_cfg.risk_post_loss_cooldown_ms);
            }
            if(dir == DIR_SHORT && until > m_short_cooldown_until_ms) {
               m_short_cooldown_until_ms = until;
               g_tel.WriteRisk(now_ms, "dir_cooldown_short", loss,
                               (double)g_cfg.risk_post_loss_cooldown_ms, "halt");
               PrintFormat("[Risk] short cooldown armed (loss=%.2f) for %dms",
                           loss, g_cfg.risk_post_loss_cooldown_ms);
            }
         }

         if(loss > g_cfg.risk_max_loss_per_trade_usd) {
            _HaltUntil(now_ms + 60000, "per_trade",
                       "max_loss_per_trade", loss,
                       g_cfg.risk_max_loss_per_trade_usd);
         }
         if(m_loss_minute > g_cfg.risk_max_loss_per_minute_usd) {
            long until = ((long)m_bucket_minute + 60) * 1000;
            _HaltUntil(until, "per_minute",
                       "max_loss_per_minute", m_loss_minute,
                       g_cfg.risk_max_loss_per_minute_usd);
         }
         if(m_loss_hour > g_cfg.risk_max_loss_per_hour_usd) {
            long until = ((long)m_bucket_hour + 3600) * 1000;
            _HaltUntil(until, "per_hour",
                       "max_loss_per_hour", m_loss_hour,
                       g_cfg.risk_max_loss_per_hour_usd);
         }
         if(m_loss_day > g_cfg.risk_max_loss_per_day_usd) {
            long until = ((long)m_bucket_day + 86400) * 1000;
            _HaltUntil(until, "per_day",
                       "max_loss_per_day", m_loss_day,
                       g_cfg.risk_max_loss_per_day_usd);
         }
      }
   }

   //--- gate called before every new entry
   bool AllowNewEntry() const { return NowMs() >= m_halted_until_ms; }

   //--- direction-aware gate. Global halt dominates; then per-direction
   //    cooldown blocks only the specific side. Callers that already
   //    passed AllowNewEntry() can skip the global check if they want —
   //    we re-check it here so a single call is safe at the submit site.
   bool AllowDirection(const ENUM_TRADE_DIR dir) const {
      long now = NowMs();
      if(now < m_halted_until_ms) return false;
      if(dir == DIR_LONG  && now < m_long_cooldown_until_ms)  return false;
      if(dir == DIR_SHORT && now < m_short_cooldown_until_ms) return false;
      return true;
   }

   bool KillEngaged() const { return NowMs() < m_halted_until_ms; }
   long HaltedUntilMs() const { return m_halted_until_ms; }
   string HaltReason() const { return m_halt_reason; }
   int    RejectCount() const { return m_reject_count; }
   double LossMinute() const { return m_loss_minute; }
   double LossHour()   const { return m_loss_hour; }
   double LossDay()    const { return m_loss_day; }
   long   LongCooldownUntilMs()  const { return m_long_cooldown_until_ms; }
   long   ShortCooldownUntilMs() const { return m_short_cooldown_until_ms; }

   //--- manual reset, called when live_overrides.json is edited
   void Reset() {
      long now = NowMs();
      bool was_halted = (m_halted_until_ms > now) ||
                        (m_long_cooldown_until_ms > now) ||
                        (m_short_cooldown_until_ms > now);
      m_halted_until_ms = 0;
      m_halt_reason = "";
      m_long_cooldown_until_ms  = 0;
      m_short_cooldown_until_ms = 0;
      m_reject_count = 0;
      m_last_reject_decay_ms = 0;
      if(was_halted) {
         g_tel.WriteRisk(now, "manual_reset", 0.0, 0.0, "cleared");
         Print("[Risk] manual reset — halt + direction cooldowns cleared");
      }
   }
};

Risk g_risk;

#endif // __FILOHFT_RISK_MQH__
