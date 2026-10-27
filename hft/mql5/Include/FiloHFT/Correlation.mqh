//+------------------------------------------------------------------+
//|  FiloHFT/Correlation.mqh                                         |
//|  Directional agreement with XAGUSD (positive) and dxy_m6         |
//|  (inverse). Used as an external context filter for setups.     |
//|                                                                 |
//|  We do NOT compute a true Pearson correlation on every tick —   |
//|  that's overkill for a 20-second window. Instead we track       |
//|  short-horizon signed returns on all three symbols and compute  |
//|  a simple cosine-similarity-ish score in [-1, 1].               |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_CORRELATION_MQH__
#define __FILOHFT_CORRELATION_MQH__

#include <FiloHFT/Common.mqh>
#include <FiloHFT/Config.mqh>

struct CorrState {
   double agreement_xagusd;         // long-window [-1..1] — positive = XAG with XAU
   double agreement_dxy;            // long-window [-1..1] — positive = DXY inverse-with XAU (desired)
   bool   xagusd_ok;
   bool   dxy_ok;
   // Short-horizon snapshot (used by reversal confirmation).
   // "Long window" stays at corr_lookback_seconds; "short window" is a fixed
   // 4s look-back so we can detect partners detaching from an ongoing XAU
   // move in the last few seconds — the "gold going solo" signal.
   double short_agreement_xagusd;
   double short_agreement_dxy;
   double r_xau_short;              // signed XAU return over the short window
   double r_xau_long;               // signed XAU return over the long window
};

class Correlation {
private:
   //--- Per-symbol mini-ring of (ts_ms, price) over the lookback.
   struct Ring {
      long   ts[256];
      double px[256];
      int    head;
      int    count;
      double last_px;
      long   last_ts;
   };

   Ring m_xau;
   Ring m_xag;
   Ring m_dxy;

   CorrState m_state;

   void _Push(Ring &r, const long ts_ms, const double px) {
      r.ts[r.head] = ts_ms;
      r.px[r.head] = px;
      r.head = (r.head + 1) % 256;
      if(r.count < 256) r.count++;
      r.last_px = px;
      r.last_ts = ts_ms;
   }

   double _Return(const Ring &r, const long lookback_ms) const {
      if(r.count < 2) return 0.0;
      int idx = r.head;
      double past = r.last_px;
      for(int i = 0; i < r.count; i++) {
         idx = (idx - 1 + 256) % 256;
         if(r.last_ts - r.ts[idx] >= lookback_ms) {
            past = r.px[idx];
            break;
         }
      }
      if(MathAbs(past) < 1e-12) return 0.0;
      return (r.last_px - past) / past;
   }

   //--- Sign-agreement score: +1 if both move in the same direction with
   //    similar magnitude, -1 if opposite. Zero if one side is stale or
   //    very small.
   double _Agreement(const double ra, const double rb) const {
      if(MathAbs(ra) < 1e-8 || MathAbs(rb) < 1e-8) return 0.0;
      double sign = (ra * rb >= 0.0) ? 1.0 : -1.0;
      double mag  = MathMin(1.0, MathMin(MathAbs(ra), MathAbs(rb)) /
                                 MathMax(MathAbs(ra), MathAbs(rb)));
      return sign * mag;
   }

public:
   void Init() {
      ZeroMemory(m_xau);
      ZeroMemory(m_xag);
      ZeroMemory(m_dxy);
      ZeroMemory(m_state);
   }

   //--- Call once per tick from OnTick. Pulls latest tick for the proxy
   //    symbols via SymbolInfoTick (cheap — cached by terminal).
   void Update(const long ts_ms, const double xau_mid) {
      _Push(m_xau, ts_ms, xau_mid);

      if(g_cfg.corr_enable_xagusd) {
         MqlTick t;
         if(SymbolInfoTick(g_cfg.sym_silver, t)) {
            double mid = 0.5 * (t.bid + t.ask);
            if(mid > 0) _Push(m_xag, ts_ms, mid);
            m_state.xagusd_ok = true;
         } else {
            m_state.xagusd_ok = false;
         }
      }

      if(g_cfg.corr_enable_dxy) {
         MqlTick t;
         if(SymbolInfoTick(g_cfg.sym_dxy, t)) {
            double mid = 0.5 * (t.bid + t.ask);
            if(mid > 0) _Push(m_dxy, ts_ms, mid);
            m_state.dxy_ok = true;
         } else {
            m_state.dxy_ok = false;
         }
      }

      long window_ms = (long)g_cfg.corr_lookback_seconds * 1000;
      double r_xau = _Return(m_xau, window_ms);
      double r_xag = _Return(m_xag, window_ms);
      double r_dxy = _Return(m_dxy, window_ms);

      m_state.agreement_xagusd = _Agreement(r_xau, r_xag);

      // For DXY, if dxy_inverse is true we invert the sign of r_dxy before
      // agreement, so +1 means DXY is going the *opposite* way (which is
      // the bullish-for-gold scenario).
      double dxy_signed = g_cfg.corr_dxy_inverse ? -r_dxy : r_dxy;
      m_state.agreement_dxy = _Agreement(r_xau, dxy_signed);

      // Short horizon — hardcoded 4s. The discretionary read is "partners
      // were with the move 20s ago but have detached in the last 4s" — a
      // classic divergence. Cheap to compute from the same rings.
      long short_ms = 4000;
      double rs_xau = _Return(m_xau, short_ms);
      double rs_xag = _Return(m_xag, short_ms);
      double rs_dxy = _Return(m_dxy, short_ms);
      double dxy_s_signed = g_cfg.corr_dxy_inverse ? -rs_dxy : rs_dxy;
      m_state.short_agreement_xagusd = _Agreement(rs_xau, rs_xag);
      m_state.short_agreement_dxy    = _Agreement(rs_xau, dxy_s_signed);
      m_state.r_xau_short            = rs_xau;
      m_state.r_xau_long             = r_xau;
   }

   //--- Gate for a directional setup. Returns true if external context
   //    is AT LEAST neutral (not actively disagreeing) or missing.
   bool DirectionalOk(const ENUM_TRADE_DIR dir, string &reason) const {
      double min_ag = g_cfg.corr_min_agreement;
      double sign = (dir == DIR_LONG) ? 1.0 : -1.0;

      if(g_cfg.corr_enable_xagusd && m_state.xagusd_ok) {
         double ag = sign * m_state.agreement_xagusd * sign; // XAG same-direction
         // We want xagusd agreement (symmetric in sign of xau return).
         // m_state.agreement_xagusd already captures same-direction-ness;
         // require it above -min_ag (not strongly disagreeing).
         if(m_state.agreement_xagusd < -min_ag * 0.5) {
            reason = "xagusd disagreeing";
            return false;
         }
      }
      if(g_cfg.corr_enable_dxy && m_state.dxy_ok) {
         // agreement_dxy is already set so that +1 = DXY doing the
         // correct inverse thing. Reject only if strongly disagreeing.
         if(m_state.agreement_dxy < -min_ag * 0.5) {
            reason = "dxy agreeing with xau (inverse expected)";
            return false;
         }
      }
      return true;
   }

   CorrState State() const { return m_state; }

   //--- Positive-confirmation signal for reversal families. Returns true
   //    when XAU is still moving against the proposed reversal direction
   //    in the short window, AND at least one partner that WAS agreeing
   //    in the long window has stopped agreeing in the short window.
   //
   //    For a LONG reversal (fading a down-move): XAU still falling in
   //    the 4s window, but XAG has stopped falling with it OR DXY has
   //    stopped rising with it → gold is making a solo move → fade edge.
   //    Symmetric for SHORT reversals at highs.
   bool PartnerDivergenceFor(const ENUM_TRADE_DIR rev_dir) const {
      if(!(m_state.xagusd_ok || m_state.dxy_ok)) return false;
      // XAU must still be moving the "wrong" way in the short window —
      // we are fading an *ongoing* move, not a stale one. Threshold is
      // ~5 pips relative at 4800 (0.00001 ≈ 0.001%).
      double min_r = 1.0e-5;
      if(rev_dir == DIR_LONG  && m_state.r_xau_short >= -min_r) return false;
      if(rev_dir == DIR_SHORT && m_state.r_xau_short <=  min_r) return false;

      double long_min  = g_cfg.setup_rev_confirm_partner_long_ag_min;
      double short_max = g_cfg.setup_rev_confirm_partner_short_ag_max;

      bool xag_diverged =
         g_cfg.corr_enable_xagusd && m_state.xagusd_ok &&
         m_state.agreement_xagusd       > long_min &&
         m_state.short_agreement_xagusd < short_max;
      bool dxy_diverged =
         g_cfg.corr_enable_dxy && m_state.dxy_ok &&
         m_state.agreement_dxy          > long_min &&
         m_state.short_agreement_dxy    < short_max;
      return xag_diverged || dxy_diverged;
   }
};

Correlation g_corr;

#endif // __FILOHFT_CORRELATION_MQH__
