//+------------------------------------------------------------------+
//|  FiloHFT/Common.mqh                                              |
//|  Shared types, enums, and math helpers used by every engine.     |
//|  Zero state, no globals — pure definitions and inline utilities. |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_COMMON_MQH__
#define __FILOHFT_COMMON_MQH__

#define FILOHFT_VERSION "0.1.0"

//--- timeframe tags used by the KL engine (stringly-typed for telemetry)
#define TF_TAG_D1   "D1"
#define TF_TAG_H4   "H4"
#define TF_TAG_H1   "H1"
#define TF_TAG_M30  "M30"
#define TF_TAG_M15  "M15"
#define TF_TAG_M5   "M5"

//--- setup families
enum ENUM_SETUP_FAMILY {
   SETUP_NONE            = 0,
   SETUP_BREAKOUT        = 1,
   SETUP_RETEST          = 2,
   SETUP_REJECTION       = 3,
   SETUP_FAKEOUT         = 4,
   SETUP_FAILED_AUCTION  = 5,
   SETUP_ENGULFING_ACCEL = 6,
   SETUP_MEAN_REVERSION  = 7,
   SETUP_MOMENTUM_CONT   = 8
};

string SetupFamilyName(const ENUM_SETUP_FAMILY f) {
   switch(f) {
      case SETUP_BREAKOUT:        return "breakout";
      case SETUP_RETEST:          return "retest";
      case SETUP_REJECTION:       return "rejection";
      case SETUP_FAKEOUT:         return "fakeout";
      case SETUP_FAILED_AUCTION:  return "failed_auction";
      case SETUP_ENGULFING_ACCEL: return "engulfing_accel";
      case SETUP_MEAN_REVERSION:  return "mean_reversion";
      case SETUP_MOMENTUM_CONT:   return "momentum_cont";
      default:                    return "none";
   }
}

//--- trade direction
enum ENUM_TRADE_DIR {
   DIR_NONE  =  0,
   DIR_LONG  =  1,
   DIR_SHORT = -1
};

string DirName(const ENUM_TRADE_DIR d) {
   if(d == DIR_LONG)  return "long";
   if(d == DIR_SHORT) return "short";
   return "none";
}

//--- internal intent state
enum ENUM_INTENT_STATE {
   INTENT_CREATED    = 0,
   INTENT_TRIGGERED  = 1,
   INTENT_EXPIRED    = 2,
   INTENT_CANCELLED  = 3
};

string IntentStateName(const ENUM_INTENT_STATE s) {
   switch(s) {
      case INTENT_CREATED:   return "created";
      case INTENT_TRIGGERED: return "triggered";
      case INTENT_EXPIRED:   return "expired";
      case INTENT_CANCELLED: return "cancelled";
   }
   return "unknown";
}

//--- trend state
enum ENUM_TREND_STATE { TREND_FLAT = 0, TREND_UP = 1, TREND_DOWN = -1 };

string TrendName(const ENUM_TREND_STATE t) {
   if(t == TREND_UP)   return "up";
   if(t == TREND_DOWN) return "down";
   return "flat";
}

//--- kill switch categories
enum ENUM_KILL_CATEGORY {
   KILL_NONE     = 0,
   KILL_DATA     = 1,
   KILL_EXECUTION= 2,
   KILL_RISK     = 3,
   KILL_CONF     = 4,
   KILL_BLACKOUT = 5
};

//--- point/pip helpers for XAUUSD on IC Markets
//    1 point = _Point; 1 pip = 10 points (per config.instrument.pip_in_points).
//    We centralize the conversions so the rest of the code stays readable.

double PointsPerPip()        { return (double)10.0; }              // XAUUSD default
double PipsFromPoints(double pts)  { return pts / PointsPerPip(); }
double PointsFromPips(double pips) { return pips * PointsPerPip(); }
double PriceFromPips(double pips)  { return pips * PointsPerPip() * _Point; }
double PipsFromPrice(double price) { return price / (PointsPerPip() * _Point); }

//--- current server epoch milliseconds
long NowMs() {
   // GetTickCount64 is monotonic ms since OS boot — fine for interval logic.
   // For wall-clock stamps use TimeCurrent()*1000 + GetTickCount64 offset only
   // if we need microsecond precision; here we stamp with TimeCurrent()*1000.
   return (long)TimeCurrent() * 1000 + (long)(GetMicrosecondCount() / 1000) % 1000;
}

long NowMicros() { return (long)GetMicrosecondCount(); }

//--- clamp / map helpers
double Clamp(const double v, const double lo, const double hi) {
   return (v < lo) ? lo : (v > hi ? hi : v);
}

double SafeDiv(const double a, const double b, const double fallback = 0.0) {
   return (MathAbs(b) < 1e-12) ? fallback : (a / b);
}

//--- tick-volume-safe reads, used in a couple of engines
double SafeClose(const string sym, const ENUM_TIMEFRAMES tf, const int shift) {
   double v = iClose(sym, tf, shift);
   return (v == 0.0 || v == EMPTY_VALUE) ? 0.0 : v;
}
double SafeHigh (const string sym, const ENUM_TIMEFRAMES tf, const int shift) {
   double v = iHigh(sym, tf, shift);
   return (v == 0.0 || v == EMPTY_VALUE) ? 0.0 : v;
}
double SafeLow  (const string sym, const ENUM_TIMEFRAMES tf, const int shift) {
   double v = iLow(sym, tf, shift);
   return (v == 0.0 || v == EMPTY_VALUE) ? 0.0 : v;
}
double SafeOpen (const string sym, const ENUM_TIMEFRAMES tf, const int shift) {
   double v = iOpen(sym, tf, shift);
   return (v == 0.0 || v == EMPTY_VALUE) ? 0.0 : v;
}

//--- monotonic ids
string MakeId(const string prefix) {
   static long counter = 0;
   counter++;
   return StringFormat("%s_%I64d", prefix, counter);
}

#endif // __FILOHFT_COMMON_MQH__
