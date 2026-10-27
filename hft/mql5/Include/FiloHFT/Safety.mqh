//+------------------------------------------------------------------+
//|  FiloHFT/Safety.mqh                                              |
//|  Anomaly detection: spread widening, quote stalls, latency      |
//|  spikes, slippage outliers. Emits safety alerts and can halt    |
//|  new entries for a configurable cool-down.                      |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_SAFETY_MQH__
#define __FILOHFT_SAFETY_MQH__

#include <FiloHFT/Common.mqh>
#include <FiloHFT/Config.mqh>
#include <FiloHFT/Telemetry.mqh>

class Safety {
private:
   long     m_last_tick_ms;
   long     m_halted_until_ms;
   double   m_ema_latency_ms;

   void _HaltFor(const int seconds, const string category, double value, double threshold) {
      long now = NowMs();
      // Direct assign (not MathMax): an earlier halt computed against stale
      // binary defaults (30s) must not outlive the current config. This keeps
      // the cooldown aligned with the currently-loaded halt_seconds.
      m_halted_until_ms = now + (long)seconds * 1000;
      g_tel.WriteSafety(now, category, value, threshold, "halt");
   }

public:
   void Init() {
      m_last_tick_ms    = 0;
      m_halted_until_ms = 0;
      m_ema_latency_ms  = 0.0;
   }

   //--- called on every tick
   void OnTick(const long ts_ms, const double spread_pips) {
      if(m_last_tick_ms > 0) {
         long gap = ts_ms - m_last_tick_ms;
         if(gap > g_cfg.safety_stall_timeout_ms) {
            _HaltFor(g_cfg.safety_halt_seconds, "stall",
                     (double)gap, (double)g_cfg.safety_stall_timeout_ms);
         }
      }
      m_last_tick_ms = ts_ms;

      if(spread_pips > g_cfg.safety_abnormal_spread_pips) {
         g_tel.WriteSafety(ts_ms, "spread", spread_pips,
                           g_cfg.safety_abnormal_spread_pips, "warn");
         _HaltFor(g_cfg.safety_halt_seconds, "spread",
                  spread_pips, g_cfg.safety_abnormal_spread_pips);
      }
   }

   //--- called by Execution with the wall-clock latency of OrderSend
   void OnOrderLatency(const double latency_ms) {
      double a = 0.1;
      m_ema_latency_ms = (1 - a) * m_ema_latency_ms + a * latency_ms;
      if(latency_ms > g_cfg.safety_latency_alert_ms) {
         g_tel.WriteSafety(NowMs(), "latency", latency_ms,
                           (double)g_cfg.safety_latency_alert_ms, "warn");
      }
   }

   //--- called by Execution with observed slippage (pips, absolute value)
   void OnSlippage(const double slip_pips) {
      if(slip_pips > g_cfg.safety_max_slippage_pips) {
         g_tel.WriteSafety(NowMs(), "slippage", slip_pips,
                           g_cfg.safety_max_slippage_pips, "warn");
      }
   }

   //--- new-entry gate
   bool AllowNewEntry() const { return NowMs() >= m_halted_until_ms; }

   long HaltedUntilMs() const { return m_halted_until_ms; }
};

Safety g_safety;

#endif // __FILOHFT_SAFETY_MQH__
