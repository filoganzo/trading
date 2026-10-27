//+------------------------------------------------------------------+
//|  FiloHFT/Telemetry.mqh                                           |
//|  Structured CSV event streams. One file per stream per UTC day. |
//|  Writes are buffered in a per-stream string and flushed on a    |
//|  timer or when the buffer exceeds a size threshold.             |
//|                                                                 |
//|  The CSV schemas are documented in docs/telemetry-schema.md.    |
//|  The only way to add a new column is to (a) add an enum tag,    |
//|  (b) add a header line in _WriteHeader, (c) add a writer method.|
//+------------------------------------------------------------------+
#ifndef __FILOHFT_TELEMETRY_MQH__
#define __FILOHFT_TELEMETRY_MQH__

#include <FiloHFT/Common.mqh>
#include <FiloHFT/Config.mqh>
#include <FiloHFT/Logger.mqh>

enum ENUM_TELEMETRY_STREAM {
   TS_FEATURES  = 0,
   TS_KEYLEVELS = 1,
   TS_SETUPS    = 2,
   TS_INTENTS   = 3,
   TS_FILLS     = 4,
   TS_TRADES    = 5,
   TS_RISK      = 6,
   TS_SAFETY    = 7,
   TS_LATENCY   = 8,
   TS_ERRORS    = 9,
   TS_SHADOW    = 10,
   TS_TMLAB     = 11,
   TS_COUNT     = 12
};

class Telemetry {
private:
   int       m_files[TS_COUNT];
   string    m_buf  [TS_COUNT];
   bool      m_open [TS_COUNT];
   bool      m_enabled[TS_COUNT];
   string    m_day_dir;            // telemetry/YYYYMMDD
   string    m_base_dir;           // from config (default "telemetry")
   datetime  m_day_stamp;
   long      m_last_flush_ms;
   int       m_flush_interval_ms;
   int       m_buf_limit;

   //--- helpers
   string _FileName(const ENUM_TELEMETRY_STREAM s) {
      switch(s) {
         case TS_FEATURES:  return "features.csv";
         case TS_KEYLEVELS: return "keylevels.csv";
         case TS_SETUPS:    return "setups.csv";
         case TS_INTENTS:   return "intents.csv";
         case TS_FILLS:     return "fills.csv";
         case TS_TRADES:    return "trades.csv";
         case TS_RISK:      return "risk.csv";
         case TS_SAFETY:    return "safety.csv";
         case TS_LATENCY:   return "latency.csv";
         case TS_ERRORS:    return "errors.csv";
         case TS_SHADOW:    return "shadow_trades.csv";
         case TS_TMLAB:     return "tm_lab.csv";
      }
      return "unknown.csv";
   }

   string _Header(const ENUM_TELEMETRY_STREAM s) {
      switch(s) {
         case TS_FEATURES:
            return "ts_ms,bid,ask,spread_pips,mid,chan_mid,chan_upper,chan_lower,chan_pos_norm,chan_slope,trend_state,velocity,burst_count_up,burst_count_dn,corr_xagusd,corr_dxy,session";
         case TS_KEYLEVELS:
            return "ts_ms,event,zone_id,center,half_width,origin_tf,weight,role,age_bars,touch_count";
         case TS_SETUPS:
            return "ts_ms,family,direction,kl_zone_id,kl_weight,kl_distance_pips,intrinsic_score,ml_score,decision,block_reason,chan_pos_norm,spread_pips,velocity,corr_xagusd,corr_dxy,route_status,echo_count";
         case TS_INTENTS:
            return "ts_ms,intent_id,family,direction,state,trigger_price,hidden_sl,hidden_tp,ttl_ms,setup_score";
         case TS_FILLS:
            return "ts_ms,intent_id,ticket,direction,requested_price,actual_price,slippage_pips,lots,latency_ms";
         case TS_TRADES:
            return "ts_ms,ticket,event,direction,price,hidden_sl,hidden_tp,unrealized_pips,realized_usd";
         case TS_RISK:
            return "ts_ms,limit,value,threshold,action";
         case TS_SAFETY:
            return "ts_ms,category,value,threshold,action";
         case TS_LATENCY:
            return "ts_ms,stage,elapsed_ms,p50_ms,p95_ms,p99_ms";
         case TS_ERRORS:
            return "ts_ms,severity,component,code,message";
         case TS_SHADOW:
            return "ts_open,ts_close,family,direction,entry_px,exit_px,realized_pips,realized_usd,exit_reason,kl_zone_id,kl_weight,kl_distance_pips,intrinsic_score,entry_spread_pips,entry_velocity,entry_chan_pos_norm,entry_corr_xagusd,entry_corr_dxy,echo_count";
         case TS_TMLAB:
            return "ts_open,ts_close,parent_ticket,variant,family,direction,entry_px,exit_px,peak_pips,trough_pips,realized_pips,realized_usd,close_reason,be_promoted,trail_armed,entry_score,entry_kl_weight,entry_atr_pips,entry_velocity,echo_count";
      }
      return "";
   }

   bool _IsEnabledInCfg(const ENUM_TELEMETRY_STREAM s, const FiloConfig &cfg) {
      switch(s) {
         case TS_FEATURES:  return cfg.tel_log_features;
         case TS_KEYLEVELS: return cfg.tel_log_keylevels;
         case TS_SETUPS:    return cfg.tel_log_setups;
         case TS_INTENTS:   return cfg.tel_log_intents;
         case TS_FILLS:     return cfg.tel_log_fills;
         case TS_TRADES:    return cfg.tel_log_trades;
         case TS_RISK:      return cfg.tel_log_risk;
         case TS_SAFETY:    return cfg.tel_log_safety;
         case TS_LATENCY:   return cfg.tel_log_latency;
         case TS_ERRORS:    return cfg.tel_log_errors;
         case TS_SHADOW:    return cfg.tel_log_shadow;
         case TS_TMLAB:     return cfg.tel_log_tmlab;
      }
      return false;
   }

   string _DayDir() {
      MqlDateTime dt;
      TimeToStruct(TimeCurrent(), dt);
      return StringFormat("%s/%04d%02d%02d", m_base_dir, dt.year, dt.mon, dt.day);
   }

   void _EnsureDay(const FiloConfig &cfg) {
      datetime today = iTime(_Symbol, PERIOD_D1, 0);
      if(today == m_day_stamp && StringLen(m_day_dir) > 0) return;
      // close all files so the next write reopens into the new day dir
      for(int i = 0; i < TS_COUNT; i++) {
         _Flush((ENUM_TELEMETRY_STREAM)i);
         if(m_files[i] != INVALID_HANDLE) { FileClose(m_files[i]); m_files[i] = INVALID_HANDLE; }
         m_open[i] = false;
      }
      m_day_dir   = _DayDir();
      m_day_stamp = today;
      // directory creation is implicit when FileOpen writes a path under
      // FILE_COMMON — MT5 creates intermediate dirs for us.
   }

   void _Open(const ENUM_TELEMETRY_STREAM s, const FiloConfig &cfg) {
      if(m_open[s]) return;
      if(!m_enabled[s]) return;
      string path = m_day_dir + "/" + _FileName(s);
      // open for read+write append; write header only if new file
      int h = FileOpen(path, FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ);
      if(h == INVALID_HANDLE) {
         // try sandbox
         h = FileOpen(path, FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_SHARE_READ);
      }
      if(h == INVALID_HANDLE) {
         PrintFormat("[Telemetry] cannot open %s (err=%d)", path, GetLastError());
         return;
      }
      bool fresh = (FileSize(h) == 0);
      FileSeek(h, 0, SEEK_END);
      if(fresh) {
         FileWriteString(h, _Header(s) + "\n");
      }
      m_files[s] = h;
      m_open[s]  = true;
   }

   void _Flush(const ENUM_TELEMETRY_STREAM s) {
      if(m_files[s] == INVALID_HANDLE) return;
      if(StringLen(m_buf[s]) == 0) return;
      FileWriteString(m_files[s], m_buf[s]);
      FileFlush(m_files[s]);
      m_buf[s] = "";
   }

   void _Append(const ENUM_TELEMETRY_STREAM s, const string line) {
      if(!m_enabled[s]) return;
      if(!m_open[s]) _Open(s, g_cfg);
      if(m_files[s] == INVALID_HANDLE) return;
      m_buf[s] += line + "\n";
      if(StringLen(m_buf[s]) >= m_buf_limit) _Flush(s);
   }

public:
   void Init(const FiloConfig &cfg) {
      m_base_dir          = cfg.tel_output_dir;
      m_flush_interval_ms = cfg.tel_flush_interval_ms;
      m_buf_limit         = 8192;
      m_day_stamp         = 0;
      m_last_flush_ms     = NowMs();
      for(int i = 0; i < TS_COUNT; i++) {
         m_files[i]   = INVALID_HANDLE;
         m_buf[i]     = "";
         m_open[i]    = false;
         m_enabled[i] = _IsEnabledInCfg((ENUM_TELEMETRY_STREAM)i, cfg);
      }
      _EnsureDay(cfg);
   }

   void Shutdown() {
      for(int i = 0; i < TS_COUNT; i++) {
         _Flush((ENUM_TELEMETRY_STREAM)i);
         if(m_files[i] != INVALID_HANDLE) { FileClose(m_files[i]); m_files[i] = INVALID_HANDLE; }
         m_open[i] = false;
      }
   }

   void MaybeFlush() {
      long now = NowMs();
      if(now - m_last_flush_ms < m_flush_interval_ms) return;
      _EnsureDay(g_cfg);
      for(int i = 0; i < TS_COUNT; i++) _Flush((ENUM_TELEMETRY_STREAM)i);
      m_last_flush_ms = now;
   }

   //--- typed writers — one per stream, match the header order exactly

   void WriteFeatures(long ts_ms, double bid, double ask, double spread_p,
                      double mid, double cmid, double cup, double clo,
                      double cposn, double cslope, string trend, double vel,
                      int bu, int bd, double cx, double cd, string sess) {
      _Append(TS_FEATURES, StringFormat(
         "%I64d,%.5f,%.5f,%.3f,%.5f,%.5f,%.5f,%.5f,%.3f,%.5f,%s,%.3f,%d,%d,%.3f,%.3f,%s",
         ts_ms, bid, ask, spread_p, mid, cmid, cup, clo, cposn, cslope,
         trend, vel, bu, bd, cx, cd, sess));
   }

   void WriteKeyLevel(long ts_ms, string event, string zone_id, double center,
                      double half_width, string origin_tf, double weight,
                      string role, int age_bars, int touch_count) {
      _Append(TS_KEYLEVELS, StringFormat(
         "%I64d,%s,%s,%.5f,%.5f,%s,%.3f,%s,%d,%d",
         ts_ms, event, zone_id, center, half_width, origin_tf, weight,
         role, age_bars, touch_count));
   }

   void WriteSetup(long ts_ms, string family, string dir, string kl_zone_id,
                   double kl_w, double kl_dist, double intrinsic, double ml,
                   string decision, string block_reason, double cposn,
                   double spread_p, double vel, double cx, double cd,
                   string route_status, int echo_count) {
      string br = block_reason; StringReplace(br, ",", ";");
      string rs = route_status; StringReplace(rs, ",", ";");
      _Append(TS_SETUPS, StringFormat(
         "%I64d,%s,%s,%s,%.3f,%.3f,%.3f,%.3f,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%d",
         ts_ms, family, dir, kl_zone_id, kl_w, kl_dist, intrinsic, ml,
         decision, br, cposn, spread_p, vel, cx, cd, rs, echo_count));
   }

   void WriteIntent(long ts_ms, string intent_id, string family, string dir,
                    string state, double trigger_price, double hidden_sl,
                    double hidden_tp, int ttl_ms, double setup_score) {
      _Append(TS_INTENTS, StringFormat(
         "%I64d,%s,%s,%s,%s,%.5f,%.5f,%.5f,%d,%.3f",
         ts_ms, intent_id, family, dir, state, trigger_price, hidden_sl,
         hidden_tp, ttl_ms, setup_score));
   }

   void WriteFill(long ts_ms, string intent_id, long ticket, string dir,
                  double requested, double actual, double slip_pips,
                  double lots, double latency_ms) {
      _Append(TS_FILLS, StringFormat(
         "%I64d,%s,%I64d,%s,%.5f,%.5f,%.3f,%.2f,%.3f",
         ts_ms, intent_id, ticket, dir, requested, actual, slip_pips,
         lots, latency_ms));
   }

   void WriteTrade(long ts_ms, long ticket, string event, string dir,
                   double price, double hidden_sl, double hidden_tp,
                   double unreal_pips, double realized_usd) {
      _Append(TS_TRADES, StringFormat(
         "%I64d,%I64d,%s,%s,%.5f,%.5f,%.5f,%.3f,%.3f",
         ts_ms, ticket, event, dir, price, hidden_sl, hidden_tp,
         unreal_pips, realized_usd));
   }

   void WriteRisk(long ts_ms, string limit, double value, double threshold,
                  string action) {
      _Append(TS_RISK, StringFormat(
         "%I64d,%s,%.3f,%.3f,%s",
         ts_ms, limit, value, threshold, action));
   }

   void WriteSafety(long ts_ms, string category, double value,
                    double threshold, string action) {
      _Append(TS_SAFETY, StringFormat(
         "%I64d,%s,%.3f,%.3f,%s",
         ts_ms, category, value, threshold, action));
   }

   void WriteLatency(long ts_ms, string stage, double elapsed,
                     double p50, double p95, double p99) {
      _Append(TS_LATENCY, StringFormat(
         "%I64d,%s,%.3f,%.3f,%.3f,%.3f",
         ts_ms, stage, elapsed, p50, p95, p99));
   }

   void WriteError(long ts_ms, string severity, string component, int code, string message) {
      string m = message; StringReplace(m, ",", ";");
      _Append(TS_ERRORS, StringFormat(
         "%I64d,%s,%s,%d,%s",
         ts_ms, severity, component, code, m));
   }

   void WriteShadow(long ts_open, long ts_close, string family, string dir,
                    double entry_px, double exit_px,
                    double realized_pips, double realized_usd,
                    string exit_reason, string kl_zone_id, double kl_weight,
                    double kl_distance_pips, double intrinsic_score,
                    double entry_spread_pips, double entry_velocity,
                    double entry_chan_pos_norm,
                    double entry_corr_xagusd, double entry_corr_dxy,
                    int echo_count) {
      _Append(TS_SHADOW, StringFormat(
         "%I64d,%I64d,%s,%s,%.5f,%.5f,%.3f,%.3f,%s,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d",
         ts_open, ts_close, family, dir, entry_px, exit_px,
         realized_pips, realized_usd, exit_reason,
         kl_zone_id, kl_weight, kl_distance_pips, intrinsic_score,
         entry_spread_pips, entry_velocity, entry_chan_pos_norm,
         entry_corr_xagusd, entry_corr_dxy, echo_count));
   }

   //--- TMLab variant row. One row per virtual-position close, written
   //    alongside real trade telemetry. See TMLab.mqh for the profile
   //    list and dispatch logic.
   void WriteTMLab(long ts_open, long ts_close, long parent_ticket,
                   string variant, string family, string dir,
                   double entry_px, double exit_px,
                   double peak_pips, double trough_pips,
                   double realized_pips, double realized_usd,
                   string close_reason, int be_promoted, int trail_armed,
                   double entry_score, double entry_kl_weight,
                   double entry_atr_pips, double entry_velocity,
                   int echo_count) {
      string vr = variant;      StringReplace(vr, ",", ";");
      string rr = close_reason; StringReplace(rr, ",", ";");
      _Append(TS_TMLAB, StringFormat(
         "%I64d,%I64d,%I64d,%s,%s,%s,%.5f,%.5f,%.3f,%.3f,%.3f,%.3f,%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%d",
         ts_open, ts_close, parent_ticket, vr, family, dir,
         entry_px, exit_px, peak_pips, trough_pips,
         realized_pips, realized_usd, rr, be_promoted, trail_armed,
         entry_score, entry_kl_weight, entry_atr_pips, entry_velocity,
         echo_count));
   }
};

Telemetry g_tel;

#endif // __FILOHFT_TELEMETRY_MQH__
