//+------------------------------------------------------------------+
//|  FiloHFT/KeyLevels.mqh                                           |
//|  Multi-timeframe key level engine.                               |
//|                                                                 |
//|  Responsibilities:                                               |
//|    - compute raw highs/lows across D1/H4/H1/M30/M15/M5          |
//|    - body-cross invalidation per origin timeframe               |
//|    - merge overlapping levels into zones with the highest       |
//|      origin weight winning                                      |
//|    - publish a sorted zone array that downstream engines can    |
//|      binary-search by price                                     |
//|                                                                 |
//|  Contract with the setup engine: when a setup needs "nearest    |
//|  KL to price P within D pips", it calls                          |
//|  KeyLevels::FindNearest(P, D, weight_min, out_zone).             |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_KEYLEVELS_MQH__
#define __FILOHFT_KEYLEVELS_MQH__

#include <FiloHFT/Common.mqh>
#include <FiloHFT/Config.mqh>
#include <FiloHFT/Telemetry.mqh>
#include <FiloHFT/Logger.mqh>

//+------------------------------------------------------------------+
//|  A zone is a price microzone with metadata.                      |
//|  center ± half_width is the price range.                         |
//+------------------------------------------------------------------+
struct KLZone {
   string            id;
   double            center;
   double            half_width;        // in price units, not points
   string            origin_tf;         // "D1".."M5"
   double            weight;
   string            role;              // "high" | "low" | "merged"
   datetime          formed_at;
   int               age_bars;
   int               touch_count;
   bool              alive;

   // Slicing-penalty state. The zone is asymmetric: price committing
   // FAR above (body close past C + k*H) disqualifies the zone as a
   // ceiling for some window; committing FAR below disqualifies it as
   // a floor. Each side decays independently on a linear curve from
   // floor_factor back to 1.0 over kl_slice_cooldown_ms.
   //   last_committed_side:  +1 if last body commit was above the zone,
   //                         -1 if below, 0 if we haven't seen one yet.
   //   dead_as_ceiling_until_ms: ts_ms at which ceiling weight returns
   //                             to 100%. 0 means no active penalty.
   //   dead_as_floor_until_ms:   symmetric for floor side.
   int               last_committed_side;
   long              dead_as_ceiling_until_ms;
   long              dead_as_floor_until_ms;
};

class KeyLevels {
private:
   KLZone            m_zones[];         // sorted by center ascending
   int               m_count;

   //--- caches of last seen bar per TF so we only refresh on new bar
   datetime          m_last_bar_d1;
   datetime          m_last_bar_h4;
   datetime          m_last_bar_h1;
   datetime          m_last_bar_m30;
   datetime          m_last_bar_m15;
   datetime          m_last_bar_m5;

   //--- helpers
   double _ZoneHalfWidth() const { return PriceFromPips(g_cfg.kl_zone_half_width_pips); }

   double _WeightFor(const string tf) const {
      if(tf == TF_TAG_D1)  return g_cfg.kl_weight_d1;
      if(tf == TF_TAG_H4)  return g_cfg.kl_weight_h4;
      if(tf == TF_TAG_H1)  return g_cfg.kl_weight_h1;
      if(tf == TF_TAG_M30) return g_cfg.kl_weight_m30;
      if(tf == TF_TAG_M15) return g_cfg.kl_weight_m15;
      if(tf == TF_TAG_M5)  return g_cfg.kl_weight_m5;
      return 0.0;
   }

   int _TfBarsForLookback(const string tf) const {
      // Convert "30 days lookback" to a per-TF bar count.
      int d = g_cfg.kl_lookback_days;
      if(tf == TF_TAG_D1)  return d;
      if(tf == TF_TAG_H4)  return d * 6;
      if(tf == TF_TAG_H1)  return d * 24;
      if(tf == TF_TAG_M30) return d * 48;
      if(tf == TF_TAG_M15) return d * 96;
      if(tf == TF_TAG_M5)  return d * 288;
      return 0;
   }

   ENUM_TIMEFRAMES _Period(const string tf) const {
      if(tf == TF_TAG_D1)  return PERIOD_D1;
      if(tf == TF_TAG_H4)  return PERIOD_H4;
      if(tf == TF_TAG_H1)  return PERIOD_H1;
      if(tf == TF_TAG_M30) return PERIOD_M30;
      if(tf == TF_TAG_M15) return PERIOD_M15;
      if(tf == TF_TAG_M5)  return PERIOD_M5;
      return PERIOD_CURRENT;
   }

   //--- adapted from the user's original code: body-cross invalidation.
   //    A level at `price`, formed at bar index `start_shift` on `tf`,
   //    is invalidated if any later closed bar's body crosses it.
   bool _IsInvalidated(const double price, const ENUM_TIMEFRAMES tf, const int start_shift) const {
      for(int i = start_shift - 1; i >= 1; i--) {
         double o = iOpen (_Symbol, tf, i);
         double c = iClose(_Symbol, tf, i);
         if(o == 0 || c == 0) continue;
         if(MathMax(o, c) > price && MathMin(o, c) < price) return true;
      }
      return false;
   }

   //--- linear lookup by center +/- merge tolerance
   int _FindMergeCandidate(const double center, const double merge_tol) const {
      for(int i = 0; i < m_count; i++) {
         if(!m_zones[i].alive) continue;
         if(MathAbs(m_zones[i].center - center) <= merge_tol) return i;
      }
      return -1;
   }

   //--- insert sorted by center
   void _InsertSorted(const KLZone &z) {
      int n = ArraySize(m_zones);
      if(m_count >= n) ArrayResize(m_zones, n + 64);
      int ins = m_count;
      for(int i = 0; i < m_count; i++) {
         if(m_zones[i].center > z.center) { ins = i; break; }
      }
      for(int j = m_count; j > ins; j--) m_zones[j] = m_zones[j - 1];
      m_zones[ins] = z;
      m_count++;
   }

   void _EmitEvent(const string ev, const KLZone &z) {
      if(g_cfg.tel_log_keylevels) {
         g_tel.WriteKeyLevel(NowMs(), ev, z.id, z.center, z.half_width,
                             z.origin_tf, z.weight, z.role, z.age_bars, z.touch_count);
      }
   }

   //--- add-or-merge: if an existing zone sits within merge tolerance,
   //    keep the one with higher weight (higher-timeframe origin wins).
   //    Otherwise insert as new.
   void _AddOrMerge(const double price, const string tf, const string role,
                    const datetime formed, const int age) {
      double half    = _ZoneHalfWidth();
      double merge_t = PriceFromPips(g_cfg.kl_merge_overlap_pips);
      double w       = _WeightFor(tf);

      int existing = _FindMergeCandidate(price, merge_t);
      if(existing >= 0) {
         if(w <= m_zones[existing].weight) {
            // existing dominates; nothing to do
            return;
         }
         // upgrade the zone to new TF origin
         m_zones[existing].center    = price;
         m_zones[existing].half_width= half;
         m_zones[existing].origin_tf = tf;
         m_zones[existing].weight    = w;
         m_zones[existing].role      = "merged";
         m_zones[existing].formed_at = formed;
         m_zones[existing].age_bars  = age;
         _EmitEvent("merge", m_zones[existing]);
         return;
      }

      KLZone z;
      z.id          = MakeId("kl");
      z.center      = price;
      z.half_width  = half;
      z.origin_tf   = tf;
      z.weight      = w;
      z.role        = role;
      z.formed_at   = formed;
      z.age_bars    = age;
      z.touch_count = 0;
      z.alive       = true;
      z.last_committed_side      = 0;
      z.dead_as_ceiling_until_ms = 0;
      z.dead_as_floor_until_ms   = 0;
      _InsertSorted(z);
      _EmitEvent("add", z);
   }

   //--- refresh one TF: scan back lookback bars, body-cross invalidate,
   //    then add-or-merge the survivors.
   void _RefreshTf(const string tf) {
      ENUM_TIMEFRAMES p = _Period(tf);
      int lookback = MathMin(_TfBarsForLookback(tf), iBars(_Symbol, p) - 1);
      if(lookback <= 1) return;

      for(int i = 1; i <= lookback; i++) {
         double h = iHigh(_Symbol, p, i);
         double l = iLow (_Symbol, p, i);
         datetime t = iTime(_Symbol, p, i);
         if(h == 0 || l == 0) continue;
         if(!_IsInvalidated(h, p, i)) _AddOrMerge(h, tf, "high", t, i);
         if(!_IsInvalidated(l, p, i)) _AddOrMerge(l, tf, "low",  t, i);
      }
   }

   //--- drop dead zones (slow path, only on full rebuild).
   void _Compact() {
      int w = 0;
      for(int i = 0; i < m_count; i++) {
         if(m_zones[i].alive) {
            if(w != i) m_zones[w] = m_zones[i];
            w++;
         }
      }
      m_count = w;
   }

public:
   void Init() {
      ArrayResize(m_zones, 256);
      m_count = 0;
      m_last_bar_d1 = 0; m_last_bar_h4 = 0; m_last_bar_h1 = 0;
      m_last_bar_m30 = 0; m_last_bar_m15 = 0; m_last_bar_m5 = 0;
   }

   //--- full rebuild. Expensive — fires once on OnInit and then only
   //    when a daily bar flips. Per-TF refresh below handles incremental.
   void Rebuild() {
      // mark everything dead, then refresh; _Compact drops the stale.
      for(int i = 0; i < m_count; i++) m_zones[i].alive = false;
      _RefreshTf(TF_TAG_D1);
      _RefreshTf(TF_TAG_H4);
      _RefreshTf(TF_TAG_H1);
      _RefreshTf(TF_TAG_M30);
      _RefreshTf(TF_TAG_M15);
      _RefreshTf(TF_TAG_M5);
      _Compact();
      g_log.Info("KL", StringFormat("rebuild complete — %d zones", m_count));
   }

   //--- per-tick check: has any TF's current bar closed? Refresh if so.
   //    Cheap — just compares cached iTime(1).
   void UpdateIfDue() {
      datetime t_d1  = iTime(_Symbol, PERIOD_D1,  0);
      datetime t_h4  = iTime(_Symbol, PERIOD_H4,  0);
      datetime t_h1  = iTime(_Symbol, PERIOD_H1,  0);
      datetime t_m30 = iTime(_Symbol, PERIOD_M30, 0);
      datetime t_m15 = iTime(_Symbol, PERIOD_M15, 0);
      datetime t_m5  = iTime(_Symbol, PERIOD_M5,  0);

      // on any M5 flip, refresh M5 only; on M15 flip, refresh M5+M15; etc.
      // D1 flip triggers full rebuild.
      if(t_d1 != m_last_bar_d1 && m_last_bar_d1 != 0) { Rebuild(); }
      else if(t_h4  != m_last_bar_h4  && m_last_bar_h4  != 0) { _RefreshTf(TF_TAG_H4);  _Compact(); }
      else if(t_h1  != m_last_bar_h1  && m_last_bar_h1  != 0) { _RefreshTf(TF_TAG_H1);  _Compact(); }
      else if(t_m30 != m_last_bar_m30 && m_last_bar_m30 != 0) { _RefreshTf(TF_TAG_M30); _Compact(); }
      else if(t_m15 != m_last_bar_m15 && m_last_bar_m15 != 0) { _RefreshTf(TF_TAG_M15); _Compact(); }
      else if(t_m5  != m_last_bar_m5  && m_last_bar_m5  != 0) { _RefreshTf(TF_TAG_M5);  _Compact(); }

      m_last_bar_d1  = t_d1;
      m_last_bar_h4  = t_h4;
      m_last_bar_h1  = t_h1;
      m_last_bar_m30 = t_m30;
      m_last_bar_m15 = t_m15;
      m_last_bar_m5  = t_m5;
   }

   //--- Slice tracking. Called every tick from OnTick after UpdateIfDue.
   //    A "slice" is when price commits with body-commitment distance
   //    beyond a zone's far edge: close past C + slice_commitment_mult*H
   //    above (ceiling slice) or below (floor slice). Commitment resets
   //    the relevant directional cooldown timer — the zone is penalized
   //    as a ceiling (resp. floor) for kl_slice_cooldown_ms and recovers
   //    linearly during that window via EffectiveWeight.
   //
   //    Non-cumulative: a fresh commit on the same side re-stamps the
   //    deadline rather than stacking. Only counts a NEW transition
   //    (last_committed_side != +1 and now it is, etc.) — dwelling on
   //    the far side does not keep re-arming the penalty.
   void ObservePrice(const double mid, const long ts_ms) {
      double k = g_cfg.kl_slice_commitment_mult;
      long   cd = (long)g_cfg.kl_slice_cooldown_ms;
      if(k <= 0.0 || cd <= 0) return;

      for(int i = 0; i < m_count; i++) {
         if(!m_zones[i].alive) continue;
         double top_commit = m_zones[i].center + k * m_zones[i].half_width;
         double bot_commit = m_zones[i].center - k * m_zones[i].half_width;

         int new_side = m_zones[i].last_committed_side;
         if(mid >= top_commit)       new_side = +1;
         else if(mid <= bot_commit)  new_side = -1;

         if(new_side != m_zones[i].last_committed_side) {
            if(new_side == +1) {
               // price sliced UP through the zone — it can no longer
               // be trusted as a ceiling for a short entry
               m_zones[i].dead_as_ceiling_until_ms = ts_ms + cd;
            } else if(new_side == -1) {
               m_zones[i].dead_as_floor_until_ms = ts_ms + cd;
            }
            m_zones[i].last_committed_side = new_side;
         }
      }
   }

   //--- Effective weight for a given trade direction at a given time.
   //    dir = DIR_LONG means the trader wants to buy off this zone as a
   //    FLOOR (support) → we consult the floor timer. dir = DIR_SHORT
   //    means the zone must act as a CEILING → consult the ceiling timer.
   //    Outside of an active cooldown, returns raw weight. Inside, weight
   //    is scaled by a linear curve: floor_factor at ts_ms = slice_time
   //    (timer just armed) recovering linearly to 1.0 at deadline.
   double EffectiveWeight(const KLZone &z, const int dir, const long now_ms) const {
      long   cd  = (long)g_cfg.kl_slice_cooldown_ms;
      double ff  = g_cfg.kl_slice_floor_factor;
      if(cd <= 0) return z.weight;

      long deadline = (dir > 0) ? z.dead_as_floor_until_ms
                                : z.dead_as_ceiling_until_ms;
      if(deadline <= 0 || now_ms >= deadline) return z.weight;

      // remaining/cd in [0,1]; 0 = just sliced, 1 = fully recovered
      double remaining = (double)(deadline - now_ms);
      double progress  = 1.0 - (remaining / (double)cd);
      if(progress < 0.0) progress = 0.0;
      if(progress > 1.0) progress = 1.0;
      double scale = ff + (1.0 - ff) * progress;
      return z.weight * scale;
   }

   //--- find the nearest zone to price whose weight >= min_weight and
   //    whose distance (in pips) <= max_dist_pips.
   //    Returns true + fills out_zone if found. Does a linear scan in
   //    the m_count window — cheap for typical counts (a few dozen
   //    zones) and avoids binary-search corner cases with half-widths.
   bool FindNearest(const double price, const double max_dist_pips,
                    const double min_weight, KLZone &out_zone) const {
      int best = -1;
      double best_d = 1e18;
      for(int i = 0; i < m_count; i++) {
         if(!m_zones[i].alive) continue;
         if(m_zones[i].weight < min_weight) continue;
         double d = MathAbs(m_zones[i].center - price);
         if(d < best_d) { best_d = d; best = i; }
      }
      if(best < 0) return false;
      if(PipsFromPrice(best_d) > max_dist_pips) return false;
      out_zone = m_zones[best];
      return true;
   }

   //--- quick price-inside-zone check for the setup engine
   bool PriceInAnyZone(const double price, const double min_weight, KLZone &out_zone) const {
      for(int i = 0; i < m_count; i++) {
         if(!m_zones[i].alive) continue;
         if(m_zones[i].weight < min_weight) continue;
         if(price >= m_zones[i].center - m_zones[i].half_width &&
            price <= m_zones[i].center + m_zones[i].half_width) {
            out_zone = m_zones[i];
            return true;
         }
      }
      return false;
   }

   int Count() const { return m_count; }
   bool Get(const int idx, KLZone &out) const {
      if(idx < 0 || idx >= m_count) return false;
      out = m_zones[idx];
      return true;
   }

   //--- Distance in pips from `price` to the NEAREST D1-origin zone in the
   //    direction of `dir` (>0 = above price, <0 = below price). Used by
   //    the "no buys right under a D1 ceiling" setup gate.
   //
   //    Semantics of the obstacle:
   //      - if price is strictly outside the zone on the wrong side,
   //        distance = (near edge) - price  (or symmetric for shorts)
   //      - if price sits INSIDE the zone, distance = 0 (the zone is
   //        already acting against us)
   //      - if price is already past the zone in the direction of travel,
   //        the zone is not an obstacle and is skipped
   //
   //--- M30/M15 structural bias. Returns +1 if the nearest M30-or-M15 zone
   //    within `range_pips` of `price` sits BELOW price (acting as support
   //    → supportive of a LONG entry), -1 if it sits ABOVE (resistance →
   //    supportive of a SHORT), 0 if no such zone exists inside the range.
   //
   //    Used by the reversal-confirmation gate as a "structural backing"
   //    check. Intentionally only looks at M30+M15 origins — we want the
   //    intermediate-timeframe read, not D1/H4 (too slow to shift) and not
   //    M5 (noisy). If price sits INSIDE the nearest zone (|dist| < zone
   //    half-width), we still use the sign of (center - price); a tiny
   //    numerical tie returns 0.
   int MidTfBias(const double price, const double range_pips) const {
      double best_d = 1e18;
      int    best_side = 0;
      for(int i = 0; i < m_count; i++) {
         if(!m_zones[i].alive) continue;
         if(m_zones[i].origin_tf != TF_TAG_M30 &&
            m_zones[i].origin_tf != TF_TAG_M15) continue;
         double dist_px = MathAbs(m_zones[i].center - price);
         if(PipsFromPrice(dist_px) > range_pips) continue;
         if(dist_px < best_d) {
            best_d = dist_px;
            double delta = m_zones[i].center - price;
            if(delta < -1e-9)      best_side = +1;   // zone below → supports long
            else if(delta > 1e-9)  best_side = -1;   // zone above → supports short
            else                   best_side =  0;
         }
      }
      return best_side;
   }

   //    Returns 1e18 if no D1 obstacle is present in the given direction.
   double NearestDailyObstaclePips(const double price, const int dir) const {
      return _NearestObstaclePipsTf(price, dir, TF_TAG_D1, TF_TAG_D1);
   }

   //    Same shape as NearestDailyObstaclePips but considers both D1 and H4
   //    zones. Used by the "don't trade within BE distance of a higher-TF
   //    level" gate: closer of the two (D1 or H4) in the given direction.
   //    Returns 1e18 if neither TF has an obstacle in front of price.
   double NearestHigherTfObstaclePips(const double price, const int dir) const {
      return _NearestObstaclePipsTf(price, dir, TF_TAG_D1, TF_TAG_H4);
   }

   //    HTF compression pocket detection: is `price` sandwiched between a
   //    D1/H4 ceiling AND a D1/H4 floor that are within `max_width_pips` of
   //    each other? If yes, out_width_pips is set to the pocket width and
   //    we return true. Intended for a future breakout-class gate — NOT
   //    currently called anywhere; kept as latent plumbing so we can flip
   //    it on later once we decide whether the breakout family itself is
   //    the flaw or whether it's a location-specific whipsaw.
   //
   //    Semantics:
   //      - looks at BOTH D1 and H4 zones that are alive
   //      - "ceiling" = nearest such zone with center > price (near-edge)
   //      - "floor"   = nearest such zone with center < price (near-edge)
   //      - if both exist and their near-edges are within max_width_pips,
   //        we're in a pocket
   //      - if price is INSIDE any HTF zone, return zero-width pocket
   bool PriceInHtfCompression(const double price, const double max_width_pips,
                              double &out_width_pips) const {
      double best_above_edge = 1e18;
      double best_below_edge = -1e18;
      bool have_above = false, have_below = false;
      for(int i = 0; i < m_count; i++) {
         if(!m_zones[i].alive) continue;
         if(m_zones[i].origin_tf != TF_TAG_D1 &&
            m_zones[i].origin_tf != TF_TAG_H4) continue;
         double lo = m_zones[i].center - m_zones[i].half_width;
         double hi = m_zones[i].center + m_zones[i].half_width;
         if(lo > price) {
            if(lo < best_above_edge) { best_above_edge = lo; have_above = true; }
         } else if(hi < price) {
            if(hi > best_below_edge) { best_below_edge = hi; have_below = true; }
         } else {
            out_width_pips = 0.0;
            return true;
         }
      }
      if(!have_above || !have_below) return false;
      out_width_pips = PipsFromPrice(best_above_edge - best_below_edge);
      return out_width_pips <= max_width_pips;
   }

private:
   double _NearestObstaclePipsTf(const double price, const int dir,
                                 const int tf_a, const int tf_b) const {
      double best = 1e18;
      for(int i = 0; i < m_count; i++) {
         if(!m_zones[i].alive) continue;
         if(m_zones[i].origin_tf != tf_a && m_zones[i].origin_tf != tf_b) continue;
         double lo = m_zones[i].center - m_zones[i].half_width;
         double hi = m_zones[i].center + m_zones[i].half_width;
         double d;
         if(dir > 0) {
            if(hi <= price) continue;               // zone is below us, not an obstacle
            d = (price < lo) ? (lo - price) : 0.0;  // 0 if inside
         } else {
            if(lo >= price) continue;               // zone is above us, not an obstacle
            d = (price > hi) ? (price - hi) : 0.0;
         }
         if(d < best) best = d;
      }
      return (best >= 1e18) ? 1e18 : PipsFromPrice(best);
   }
};

KeyLevels g_kl;

#endif // __FILOHFT_KEYLEVELS_MQH__
