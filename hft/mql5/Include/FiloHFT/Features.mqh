//+------------------------------------------------------------------+
//|  FiloHFT/Features.mqh                                            |
//|  Hot-path feature engine. Updated on every tick.                 |
//|                                                                  |
//|  Produces (per tick) a snapshot containing:                     |
//|    - regression channel over M1 bars (length from config)      |
//|    - standard deviation envelope                                |
//|    - normalized position inside channel                        |
//|    - rolling velocity (pips/second) + EWMA                     |
//|    - acceleration burst counts (up/down within window_ms)     |
//|    - trend state                                               |
//|                                                                  |
//|  Implementation notes:                                           |
//|    - The regression channel is recomputed on M1 bar close only  |
//|      (not every tick) — O(length). Between closes we use the    |
//|      cached coefficients and the live price.                    |
//|    - Velocity uses a ring buffer of (ts_ms, price) pairs.        |
//|    - Bursts are flagged when instantaneous velocity exceeds     |
//|      EMA(velocity) + sigma * stddev(velocity).                  |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_FEATURES_MQH__
#define __FILOHFT_FEATURES_MQH__

#include <FiloHFT/Common.mqh>
#include <FiloHFT/Config.mqh>
#include <FiloHFT/Telemetry.mqh>

struct FeatureSnapshot {
   long             ts_ms;
   double           bid;
   double           ask;
   double           mid;
   double           spread_pips;

   // channel
   double           chan_mid;
   double           chan_upper;
   double           chan_lower;
   double           chan_pos_norm;   // -1 at lower, +1 at upper, signed beyond
   double           chan_slope_pips_per_bar;
   double           chan_sigma;

   // velocity / bursts
   double           velocity_pips_per_sec;
   double           velocity_ema;
   double           velocity_stddev;
   int              burst_count_up;
   int              burst_count_dn;
   long             last_burst_ts_up;
   long             last_burst_ts_dn;

   // trend
   ENUM_TREND_STATE trend_state;
   double           trend_persistence;
};

class Features {
private:
   // regression cache (recomputed on M1 bar close)
   double           m_chan_intercept;    // price at x=0 (newest bar)
   double           m_chan_slope;        // price per bar
   double           m_chan_sigma;        // residual stddev
   datetime         m_last_m1;

   // velocity ring buffer
   long             m_vel_ts [1024];
   double           m_vel_px [1024];
   int              m_vel_head;
   int              m_vel_count;

   double           m_velocity_ema;
   double           m_velocity_var_ema;  // for stddev estimate
   double           m_ema_alpha;         // derived from velocity_ema_ms

   // burst ring buffer — stores timestamps of recent bursts
   long             m_burst_up_ts[64];
   long             m_burst_dn_ts[64];
   int              m_burst_up_count;
   int              m_burst_dn_count;

   // trend cache
   ENUM_TREND_STATE m_trend_state;
   double           m_trend_persistence;

   FeatureSnapshot  m_snap;

   //--- linear regression over recent M1 closes (length = g_cfg.chan_length)
   void _RecomputeChannel() {
      int N = g_cfg.chan_length;
      if(N < 4) N = 4;
      if(iBars(_Symbol, PERIOD_M1) < N + 2) return;

      // Use shift 1..N so we work with closed bars.
      double sx = 0.0, sy = 0.0, sxy = 0.0, sxx = 0.0;
      for(int i = 0; i < N; i++) {
         double x = (double)i;                       // 0 = most recent closed bar
         double y = iClose(_Symbol, PERIOD_M1, i + 1);
         sx  += x;
         sy  += y;
         sxy += x * y;
         sxx += x * x;
      }
      double mean_x = sx / N;
      double mean_y = sy / N;
      double denom  = sxx - sx * mean_x;
      if(MathAbs(denom) < 1e-12) return;
      double slope = (sxy - sx * mean_y) / denom;
      // With x=0 at "most recent closed bar", intercept gives the model
      // value at that bar. We extrapolate forward by 0 bars for the live
      // tick below (good enough for 100-bar channels; extending by 1 bar
      // would be marginal).
      double intercept = mean_y - slope * mean_x;

      // residual stddev
      double ss = 0.0;
      for(int i = 0; i < N; i++) {
         double x = (double)i;
         double y = iClose(_Symbol, PERIOD_M1, i + 1);
         double pred = intercept + slope * x;
         double d = y - pred;
         ss += d * d;
      }
      double sigma = MathSqrt(ss / N);

      m_chan_intercept = intercept;
      // Note: slope is "price per bar going backwards in time" because
      // x increases with shift. Flip the sign so positive = bullish.
      m_chan_slope     = -slope;
      m_chan_sigma     = sigma;
   }

   //--- push a new (ts_ms, price) into the velocity buffer, compute vel
   void _PushVelocity(const long ts_ms, const double price) {
      int cap = 1024;
      m_vel_ts [m_vel_head] = ts_ms;
      m_vel_px [m_vel_head] = price;
      m_vel_head = (m_vel_head + 1) % cap;
      if(m_vel_count < cap) m_vel_count++;

      // compare against the buffer entry ~ (window_ms) behind the current
      int lookback_target = g_cfg.accel_window_ms;
      int idx = m_vel_head;
      double past_px = price;
      long past_ts = ts_ms;
      for(int i = 0; i < m_vel_count; i++) {
         idx = (idx - 1 + cap) % cap;
         if(ts_ms - m_vel_ts[idx] >= lookback_target) {
            past_px = m_vel_px[idx];
            past_ts = m_vel_ts[idx];
            break;
         }
      }
      long dt_ms = ts_ms - past_ts;
      if(dt_ms < 1) dt_ms = 1;
      double dpx_pips = PipsFromPrice(price - past_px);
      double vel = dpx_pips * 1000.0 / dt_ms;     // pips/sec

      // EWMA update
      double a = m_ema_alpha;
      double old_ema = m_velocity_ema;
      m_velocity_ema = (1 - a) * m_velocity_ema + a * vel;
      double diff = vel - old_ema;
      m_velocity_var_ema = (1 - a) * m_velocity_var_ema + a * diff * diff;

      m_snap.velocity_pips_per_sec = vel;
      m_snap.velocity_ema          = m_velocity_ema;
      m_snap.velocity_stddev       = MathSqrt(m_velocity_var_ema);
   }

   //--- register a burst if current |velocity| exceeds mean + sigma*stddev
   void _DetectBurst(const long ts_ms, const double vel) {
      double stddev = MathSqrt(m_velocity_var_ema);
      double thresh = m_velocity_ema + g_cfg.accel_burst_sigma * stddev;
      double nthres = m_velocity_ema - g_cfg.accel_burst_sigma * stddev;

      // drop stale burst entries
      long cutoff = ts_ms - g_cfg.accel_window_ms;
      int w = 0;
      for(int i = 0; i < m_burst_up_count; i++)
         if(m_burst_up_ts[i] >= cutoff) { m_burst_up_ts[w++] = m_burst_up_ts[i]; }
      m_burst_up_count = w;
      w = 0;
      for(int i = 0; i < m_burst_dn_count; i++)
         if(m_burst_dn_ts[i] >= cutoff) { m_burst_dn_ts[w++] = m_burst_dn_ts[i]; }
      m_burst_dn_count = w;

      if(vel > thresh && stddev > 0.05) {
         if(m_burst_up_count < 64) m_burst_up_ts[m_burst_up_count++] = ts_ms;
         m_snap.last_burst_ts_up = ts_ms;
      }
      if(vel < nthres && stddev > 0.05) {
         if(m_burst_dn_count < 64) m_burst_dn_ts[m_burst_dn_count++] = ts_ms;
         m_snap.last_burst_ts_dn = ts_ms;
      }
      m_snap.burst_count_up = m_burst_up_count;
      m_snap.burst_count_dn = m_burst_dn_count;
   }

   //--- trend derived from channel slope + persistence of M1 direction
   void _UpdateTrend() {
      int N = g_cfg.trend_lookback_bars_m1;
      if(N < 4) N = 4;
      if(iBars(_Symbol, PERIOD_M1) < N + 2) return;

      int up_bars = 0, down_bars = 0;
      for(int i = 1; i <= N; i++) {
         double o = iOpen (_Symbol, PERIOD_M1, i);
         double c = iClose(_Symbol, PERIOD_M1, i);
         if(c > o) up_bars++;
         else if(c < o) down_bars++;
      }
      double ratio_up = (double)up_bars / N;
      double ratio_dn = (double)down_bars / N;

      double slope_pips = PipsFromPrice(m_chan_slope);
      double pmin = g_cfg.trend_persistence_min_ratio;

      ENUM_TREND_STATE t = TREND_FLAT;
      double persistence = 0.0;
      if(slope_pips > 0.05 && ratio_up >= pmin) {
         t = TREND_UP;
         persistence = ratio_up;
      } else if(slope_pips < -0.05 && ratio_dn >= pmin) {
         t = TREND_DOWN;
         persistence = ratio_dn;
      } else {
         persistence = MathMax(ratio_up, ratio_dn);
      }

      m_trend_state       = t;
      m_trend_persistence = persistence;
   }

public:
   void Init() {
      m_chan_intercept = 0.0;
      m_chan_slope     = 0.0;
      m_chan_sigma     = 0.0;
      m_last_m1        = 0;

      ArrayInitialize(m_vel_ts, 0);
      ArrayInitialize(m_vel_px, 0.0);
      m_vel_head = 0;
      m_vel_count = 0;

      m_velocity_ema = 0.0;
      m_velocity_var_ema = 0.0;
      // EMA alpha derived so that the "half-life" matches velocity_ema_ms.
      // alpha = 1 - exp(-dt/tau). For tick-driven updates we assume ~50ms dt.
      m_ema_alpha = 1.0 - MathExp(-50.0 / MathMax(1.0, (double)g_cfg.accel_velocity_ema_ms));

      ArrayInitialize(m_burst_up_ts, 0);
      ArrayInitialize(m_burst_dn_ts, 0);
      m_burst_up_count = 0;
      m_burst_dn_count = 0;

      m_trend_state = TREND_FLAT;
      m_trend_persistence = 0.0;

      ZeroMemory(m_snap);
   }

   //--- main entry point — called on every OnTick
   void Update(const MqlTick &tick) {
      long ts_ms = (long)tick.time * 1000 + (long)tick.time_msc % 1000;
      double bid = tick.bid;
      double ask = tick.ask;
      double mid = 0.5 * (bid + ask);

      m_snap.ts_ms       = ts_ms;
      m_snap.bid         = bid;
      m_snap.ask         = ask;
      m_snap.mid         = mid;
      m_snap.spread_pips = PipsFromPrice(ask - bid);

      // recompute channel model on M1 bar flip
      datetime cur_m1 = iTime(_Symbol, PERIOD_M1, 0);
      if(cur_m1 != m_last_m1) {
         _RecomputeChannel();
         _UpdateTrend();
         m_last_m1 = cur_m1;
      }

      m_snap.chan_mid    = m_chan_intercept;       // at x=0 (latest closed bar)
      m_snap.chan_sigma  = m_chan_sigma;
      m_snap.chan_upper  = m_chan_intercept + g_cfg.chan_stdev_mult * m_chan_sigma;
      m_snap.chan_lower  = m_chan_intercept - g_cfg.chan_stdev_mult * m_chan_sigma;
      m_snap.chan_slope_pips_per_bar = PipsFromPrice(m_chan_slope);

      double denom = MathMax(1e-9, m_chan_sigma);
      m_snap.chan_pos_norm = (mid - m_chan_intercept) / (g_cfg.chan_stdev_mult * denom);

      m_snap.trend_state       = m_trend_state;
      m_snap.trend_persistence = m_trend_persistence;

      _PushVelocity(ts_ms, mid);
      _DetectBurst(ts_ms, m_snap.velocity_pips_per_sec);
   }

   FeatureSnapshot Snapshot() const { return m_snap; }

   bool HasMinBursts(const ENUM_TRADE_DIR dir) const {
      if(dir == DIR_LONG)  return m_burst_up_count >= g_cfg.accel_min_bursts;
      if(dir == DIR_SHORT) return m_burst_dn_count >= g_cfg.accel_min_bursts;
      return false;
   }

   bool AtChannelExtreme(const ENUM_TRADE_DIR dir) const {
      double p = m_snap.chan_pos_norm;
      if(dir == DIR_LONG)  return p >=  g_cfg.chan_extreme_block_mult;
      if(dir == DIR_SHORT) return p <= -g_cfg.chan_extreme_block_mult;
      return false;
   }

   //--- Positive-confirmation signal for reversal families. Returns true
   //    when an adverse burst happened recently (within max_ms) but has
   //    been quiet for at least min_ms — "sellers were bursting, now
   //    they've stopped." Uses the existing last-burst timestamps; no
   //    extra state needed.
   //
   //    For a LONG reversal (fading a down-move): we look at the last
   //    DOWN-burst timestamp. If it's within [min_ms..max_ms] ago, the
   //    move decelerated. If it's inside min_ms, sellers are still
   //    active — fail. If it's older than max_ms, we have no proof a
   //    burst ever happened — fail (silence ≠ decay).
   bool BurstDecayAgainst(const ENUM_TRADE_DIR rev_dir, const long now_ms,
                          const long min_ms, const long max_ms) const {
      long last = (rev_dir == DIR_LONG) ? m_snap.last_burst_ts_dn
                                        : m_snap.last_burst_ts_up;
      if(last <= 0) return false;
      long gap = now_ms - last;
      return (gap >= min_ms && gap <= max_ms);
   }
};

Features g_features;

#endif // __FILOHFT_FEATURES_MQH__
