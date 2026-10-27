//+------------------------------------------------------------------+
//|  FiloHFT/DailyBias.mqh                                           |
//|  Directional bias from recent D1 break/bounce events, with      |
//|  strong exponential time decay.                                  |
//|                                                                 |
//|  Discretionary intuition the module encodes:                    |
//|    - If price BREAKS a D1 zone, it tends to reach the next D1   |
//|      level in the direction of the break → bias same side.      |
//|    - If price BOUNCES off a D1 zone, it tends to travel toward  |
//|      the next D1 level on the opposite side → bias away from   |
//|      the level.                                                  |
//|    - Both effects decay fast (3-min half-life by default).      |
//|    - When two D1 levels close together contribute events in the |
//|      same direction, the signal is doubled (compounding), e.g.  |
//|      a bounce off a cluster of two D1 floors is much stronger  |
//|      than a single bounce.                                      |
//|                                                                 |
//|  Consumers: Setups._TryFamily() adds `Bias() * score_weight` to |
//|  the intrinsic score of matching-direction candidates before   |
//|  the min-score gate. No hard veto — this is a soft nudge.      |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_DAILYBIAS_MQH__
#define __FILOHFT_DAILYBIAS_MQH__

#include <FiloHFT/Common.mqh>
#include <FiloHFT/Config.mqh>
#include <FiloHFT/KeyLevels.mqh>

#define FILO_DAILY_BIAS_CAP 16

enum ENUM_DAILY_EVENT_TYPE {
   DB_BREAK  = 0,
   DB_BOUNCE = 1
};

struct DailyBiasEvent {
   string                 level_id;
   double                 level_price;
   ENUM_DAILY_EVENT_TYPE  type;
   int                    direction;   // +1 up, -1 down
   long                   ts_ms;
   datetime               bar_time;    // M5 bar shift-1 time (dedup key)
   bool                   alive;
};

class DailyBias {
private:
   DailyBiasEvent  m_events[FILO_DAILY_BIAS_CAP];
   int             m_count;
   datetime        m_last_m5;

   void _Add(const DailyBiasEvent &e) {
      for(int i = 0; i < m_count; i++) {
         if(m_events[i].alive &&
            m_events[i].level_id == e.level_id &&
            m_events[i].bar_time == e.bar_time &&
            m_events[i].type     == e.type) return;
      }
      if(m_count < FILO_DAILY_BIAS_CAP) {
         m_events[m_count++] = e;
         return;
      }
      int oldest = 0;
      for(int i = 1; i < m_count; i++)
         if(m_events[i].ts_ms < m_events[oldest].ts_ms) oldest = i;
      m_events[oldest] = e;
   }

   void _Prune(const long now_ms) {
      long max_age = (long)g_cfg.daily_bias_max_age_ms;
      for(int i = 0; i < m_count; i++) {
         if(!m_events[i].alive) continue;
         if(now_ms - m_events[i].ts_ms > max_age) m_events[i].alive = false;
      }
   }

   //--- Scan last closed M5 bar for break/bounce events against every
   //    currently-alive D1 zone.
   void _ScanLastM5(const long now_ms) {
      double o1 = iOpen (_Symbol, PERIOD_M5, 1);
      double h1 = iHigh (_Symbol, PERIOD_M5, 1);
      double l1 = iLow  (_Symbol, PERIOD_M5, 1);
      double c1 = iClose(_Symbol, PERIOD_M5, 1);
      datetime t1 = iTime(_Symbol, PERIOD_M5, 1);
      if(o1 == 0 || c1 == 0) return;

      double tol    = PriceFromPips(g_cfg.daily_bias_break_tolerance_pips);
      double min_wk = g_cfg.daily_bias_bounce_min_wick_pips;

      for(int i = 0; i < g_kl.Count(); i++) {
         KLZone z; if(!g_kl.Get(i, z)) continue;
         if(!z.alive) continue;
         if(z.origin_tf != TF_TAG_D1) continue;

         double lo = z.center - z.half_width;
         double hi = z.center + z.half_width;

         // BREAK up: opened clearly below the zone, closed clearly above.
         if(o1 < (lo - tol) && c1 > (hi + tol)) {
            DailyBiasEvent e;
            e.level_id    = z.id;
            e.level_price = z.center;
            e.type        = DB_BREAK;
            e.direction   = +1;
            e.ts_ms       = now_ms;
            e.bar_time    = t1;
            e.alive       = true;
            _Add(e);
            continue;
         }
         // BREAK down
         if(o1 > (hi + tol) && c1 < (lo - tol)) {
            DailyBiasEvent e;
            e.level_id    = z.id;
            e.level_price = z.center;
            e.type        = DB_BREAK;
            e.direction   = -1;
            e.ts_ms       = now_ms;
            e.bar_time    = t1;
            e.alive       = true;
            _Add(e);
            continue;
         }
         // BOUNCE up: wick below zone, close above zone, sufficient wick depth.
         if(l1 < lo && c1 > z.center) {
            double wick_pips = PipsFromPrice(lo - l1);
            if(wick_pips >= min_wk) {
               DailyBiasEvent e;
               e.level_id    = z.id;
               e.level_price = z.center;
               e.type        = DB_BOUNCE;
               e.direction   = +1;
               e.ts_ms       = now_ms;
               e.bar_time    = t1;
               e.alive       = true;
               _Add(e);
            }
            continue;
         }
         // BOUNCE down
         if(h1 > hi && c1 < z.center) {
            double wick_pips = PipsFromPrice(h1 - hi);
            if(wick_pips >= min_wk) {
               DailyBiasEvent e;
               e.level_id    = z.id;
               e.level_price = z.center;
               e.type        = DB_BOUNCE;
               e.direction   = -1;
               e.ts_ms       = now_ms;
               e.bar_time    = t1;
               e.alive       = true;
               _Add(e);
            }
         }
      }
   }

public:
   void Init() {
      m_count   = 0;
      m_last_m5 = 0;
      for(int i = 0; i < FILO_DAILY_BIAS_CAP; i++) m_events[i].alive = false;
   }

   //--- Called every tick. Cheap fast-path: nothing happens unless a new
   //    M5 bar just closed.
   void Step(const FeatureSnapshot &fs) {
      if(!g_cfg.daily_bias_enable) return;
      datetime t0 = iTime(_Symbol, PERIOD_M5, 0);
      if(t0 == 0) return;
      if(m_last_m5 == 0) { m_last_m5 = t0; return; }
      if(t0 == m_last_m5) return;
      m_last_m5 = t0;
      _ScanLastM5(fs.ts_ms);
      _Prune(fs.ts_ms);
   }

   //--- Decayed signed bias in [-1, +1]. Positive = upward bias.
   //    Compounding: if two alive events with the same sign have level
   //    prices within `nearby_d1_compound_pips` of each other, both of
   //    their contributions are doubled.
   double Bias(const long now_ms) const {
      if(!g_cfg.daily_bias_enable) return 0.0;
      if(m_count <= 0) return 0.0;

      double hl  = (double)g_cfg.daily_bias_half_life_ms;
      if(hl <= 0.0) return 0.0;
      long   max_age = (long)g_cfg.daily_bias_max_age_ms;
      double near_px = PriceFromPips(g_cfg.daily_bias_nearby_d1_compound_pips);
      double ln2 = 0.6931471805599453;

      double w[FILO_DAILY_BIAS_CAP];
      int    s[FILO_DAILY_BIAS_CAP];
      double p[FILO_DAILY_BIAS_CAP];
      bool   compound[FILO_DAILY_BIAS_CAP];
      int n = 0;

      for(int i = 0; i < m_count; i++) {
         if(!m_events[i].alive) continue;
         long age = now_ms - m_events[i].ts_ms;
         if(age < 0) age = 0;
         if(age > max_age) continue;
         w[n] = MathExp(-ln2 * (double)age / hl);
         s[n] = m_events[i].direction;
         p[n] = m_events[i].level_price;
         compound[n] = false;
         n++;
      }
      if(n == 0) return 0.0;

      for(int i = 0; i < n; i++) {
         for(int j = i + 1; j < n; j++) {
            if(s[i] == s[j] && MathAbs(p[i] - p[j]) <= near_px) {
               compound[i] = true;
               compound[j] = true;
            }
         }
      }

      double sum = 0.0;
      for(int i = 0; i < n; i++) {
         double c = compound[i] ? 2.0 : 1.0;
         sum += (double)s[i] * w[i] * c;
      }
      if(sum >  1.0) sum =  1.0;
      if(sum < -1.0) sum = -1.0;
      return sum;
   }

   int ActiveCount() const { return m_count; }
};

DailyBias g_daily_bias;

#endif // __FILOHFT_DAILYBIAS_MQH__
