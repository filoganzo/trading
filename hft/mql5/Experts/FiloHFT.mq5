//+------------------------------------------------------------------+
//|                                                      FiloHFT.mq5 |
//|                      Retail low-latency XAUUSD event-driven EA |
//|                                                                  |
//|  Entry point. Loads config, initializes every engine, dispatches |
//|  OnTick to the engine pipeline, and manages the management-loop  |
//|  timer. See ARCHITECTURE.md for the component map.              |
//+------------------------------------------------------------------+
#property copyright   "FiloHFT"
#property version     "1.00"
#property description "XAUUSD low-latency KL-centric event-driven EA"
#property strict

#include <FiloHFT/Common.mqh>
#include <FiloHFT/Config.mqh>
#include <FiloHFT/Logger.mqh>
#include <FiloHFT/Telemetry.mqh>
#include <FiloHFT/EventBus.mqh>
#include <FiloHFT/KeyLevels.mqh>
#include <FiloHFT/Features.mqh>
#include <FiloHFT/Correlation.mqh>
#include <FiloHFT/DailyBias.mqh>
#include <FiloHFT/ExecutionIntent.mqh>
#include <FiloHFT/Shadow.mqh>
#include <FiloHFT/TMLab.mqh>
#include <FiloHFT/Setups.mqh>
#include <FiloHFT/TradeManagement.mqh>
#include <FiloHFT/Risk.mqh>
#include <FiloHFT/Safety.mqh>
#include <FiloHFT/Execution.mqh>

//--- user-visible inputs. These become overrides over default.ini on load.
input string  InpConfigPath       = "FiloHFT/default.ini"; // path inside MT5 common Files folder
input string  InpLogPath          = "FiloHFT/filo_hft.log";
input string  InpOverridesPath    = "FiloHFT/live_overrides.json";
input bool    InpEnableTrading    = false;                 // master safety — default OFF
input double  InpFixedLotOverride = 0.0;                   // 0 = use config
input int     InpManagementTimerMs= 100;                   // OnTimer cadence

//--- globals that track timing
long    g_last_overrides_reload_ms = 0;
long    g_last_features_log_ms     = 0;
long    g_last_kl_check_ms         = 0;
long    g_last_heartbeat_ms        = 0;

//+------------------------------------------------------------------+
int OnInit() {
   // 1. Load config
   if(!Config_Load(InpConfigPath, g_cfg)) {
      Print("[FiloHFT] WARNING: using compiled-in defaults — ini not loaded.");
   }
   if(InpFixedLotOverride > 0) g_cfg.fixed_lot = InpFixedLotOverride;

   // 2. Init logger + telemetry
   g_log.Init(InpLogPath, LOG_INFO, true);
   g_log.Info("Boot", StringFormat("FiloHFT v%s starting — lot=%.2f trading_enabled=%s",
                                   FILOHFT_VERSION, g_cfg.fixed_lot,
                                   InpEnableTrading ? "yes" : "no"));
   g_tel.Init(g_cfg);
   g_event_log.Init();

   // 3. Init engines in dependency order
   g_kl.Init();
   g_kl.Rebuild();
   g_features.Init();
   g_corr.Init();
   g_daily_bias.Init();
   g_intents.Init();
   g_setups.Init();
   g_exec.Init();
   g_tm.Init();
   g_risk.Init();
   g_safety.Init();
   g_shadow.Init();
   g_tmlab.Init();

   // 4. Timer for management/risk cycles
   EventSetMillisecondTimer(InpManagementTimerMs);

   g_log.Info("Boot", StringFormat("engines initialized — %d KL zones", g_kl.Count()));
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
   EventKillTimer();
   g_log.Info("Shutdown", StringFormat("reason=%d", reason));
   g_tel.Shutdown();
   g_log.Close();
}

//+------------------------------------------------------------------+
void OnTick() {
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   long t0 = NowMicros();

   // -------- 1. Update features from latest tick ----------------------
   g_features.Update(tick);
   FeatureSnapshot fs = g_features.Snapshot();

   // -------- 2. Safety: spread, stall ---------------------------------
   g_safety.OnTick(fs.ts_ms, fs.spread_pips);

   // -------- 3. KL: only check for bar flips periodically -------------
   if(fs.ts_ms - g_last_kl_check_ms > 500) {
      g_kl.UpdateIfDue();
      g_last_kl_check_ms = fs.ts_ms;
   }
   // Slicing-penalty observer runs every tick: it watches for body
   // commitment past a zone's far edge and arms the directional cooldown
   // that EffectiveWeight decays linearly over the next N seconds.
   g_kl.ObservePrice(fs.mid, fs.ts_ms);

   // -------- 4. Correlation update -----------------------------------
   g_corr.Update(fs.ts_ms, fs.mid);
   g_daily_bias.Step(fs);

   // -------- 5. Trade management + shadow step (exits get priority) --
   g_tm.Step(fs);
   g_shadow.Step(fs);
   g_tmlab.Step(fs);

   // -------- 6. Setup scan — always runs so shadow trades and the    -
   //            setups.csv stream capture every signal. Real intents  -
   //            are only submitted when the live-trade gate is open.  -
   bool allow_entry = InpEnableTrading &&
                      g_risk.AllowNewEntry() &&
                      g_safety.AllowNewEntry();
   g_setups.Scan(fs, allow_entry);

   if(allow_entry) {
      // -------- 7. Process intent triggers -----------------------------
      // Strict concurrency gate: refuse to fire ANY new intent while
      // another fill is already in flight (PendingCount > 0). Under
      // the async execution model the broker ack for a committed
      // intent lands ~10s later, and only then does TradeManagement
      // register the position into m_trades. CanOpen/CanStack consult
      // m_trades, so without the PendingCount check two intents on
      // consecutive ticks would both pass and produce a double fill.
      int trig = g_intents.ProcessTick(fs);
      if(trig >= 0) {
         ExecutionIntent e;
         string stack_reason = "";
         bool pending_block = (g_exec.PendingCount() > 0);
         if(pending_block) {
            // leave the intent in the book; let TTL decide. Telemetry
            // via error log so we can audit how often this fires.
            if(g_cfg.tel_log_intents) {
               g_tel.WriteError(fs.ts_ms, "info", "intent_pending_block", 0,
                  StringFormat("trig held — %d fills in flight",
                               g_exec.PendingCount()));
            }
         } else if(g_intents.Get(trig, e) &&
                   g_risk.AllowDirection(e.direction) &&
                   g_tm.CanOpen(e.direction) &&
                   g_tm.CanStack(e.direction, e.family, e.zone_id, fs.ts_ms, stack_reason)) {
            g_intents.MarkTriggered(trig, fs);
            if(g_exec.ExecuteIntent(e, fs)) {
               // AttachIntent runs inside Execution::OnTradeEvent when the
               // DEAL_ADD for the async fill lands — position must exist first.
               // Kill every other CREATED intent so none can leak through the
               // ~10s broker-ack window.
               g_intents.CancelAllCreated("pending_fill", fs);
            } else {
               g_risk.NotifyReject();
               g_safety.OnOrderLatency(0.0); // let safety log the failure
            }
            g_intents.Compact();
         }
      }
   }

   // -------- 8. Feature telemetry (gated to ~250ms) -------------------
   if(g_cfg.tel_log_features && fs.ts_ms - g_last_features_log_ms > 250) {
      CorrState cs = g_corr.State();
      g_tel.WriteFeatures(fs.ts_ms, fs.bid, fs.ask, fs.spread_pips, fs.mid,
                          fs.chan_mid, fs.chan_upper, fs.chan_lower,
                          fs.chan_pos_norm, fs.chan_slope_pips_per_bar,
                          TrendName(fs.trend_state), fs.velocity_pips_per_sec,
                          fs.burst_count_up, fs.burst_count_dn,
                          cs.agreement_xagusd, cs.agreement_dxy, "live");
      g_last_features_log_ms = fs.ts_ms;
   }

   // -------- 9. Batched telemetry flush -------------------------------
   g_tel.MaybeFlush();

   // -------- 10. Tick-handler latency log ------------------------------
   long t1 = NowMicros();
   double elapsed = (t1 - t0) / 1000.0;
   if(g_cfg.tel_log_latency && elapsed > (double)g_cfg.safety_latency_alert_ms) {
      g_tel.WriteLatency(fs.ts_ms, "tick_handler", elapsed, 0, 0, 0);
   }
}

//+------------------------------------------------------------------+
//|  Timer: reload overrides, reconcile trades every InpManagementTimerMs |
//+------------------------------------------------------------------+
void OnTimer() {
   long now = NowMs();

   // 5s hot-reload of live_overrides.json. If the file content changed
   // since the last successful read, treat that as an operator action and
   // clear any stuck risk halt so the new thresholds take effect instantly.
   if(now - g_last_overrides_reload_ms > 5000) {
      bool changed = Config_ReloadOverrides(InpOverridesPath, g_cfg);
      if(changed) {
         g_risk.Reset();
         Print("[FiloHFT] live_overrides.json changed — risk halt auto-cleared");
      }
      g_last_overrides_reload_ms = now;
   }

   // periodic risk upkeep: rolls loss buckets, decays reject count, and
   // auto-clears expired halts. Without this the time-based halt would
   // only observe its deadline inside AllowNewEntry() reads, and the
   // "cleared" telemetry row would never get written.
   g_risk.Step();

   // 10s gate heartbeat — always logs the current state of allow_entry
   // so you can see on the dashboard exactly which gate is shut if
   // trading has gone quiet.
   if(now - g_last_heartbeat_ms > 10000) {
      bool risk_ok   = g_risk.AllowNewEntry();
      bool safety_ok = g_safety.AllowNewEntry();
      bool allow     = InpEnableTrading && risk_ok && safety_ok;
      string state = allow ? "ok" : "halted";
      string reason = "";
      if(!InpEnableTrading) reason = "trading_disabled";
      else if(!risk_ok)     reason = "risk:" + g_risk.HaltReason();
      else if(!safety_ok)   reason = "safety";
      else                  reason = "ok";
      g_tel.WriteRisk(now, "heartbeat:" + reason,
                      (double)(allow ? 1 : 0),
                      (double)g_risk.HaltedUntilMs(), state);
      g_last_heartbeat_ms = now;
   }

   // release pending async submissions that never got a fill event
   g_exec.SweepStale(now);

   // slow management pass in case ticks are sparse
   MqlTick tick;
   if(SymbolInfoTick(_Symbol, tick)) {
      g_features.Update(tick);
      FeatureSnapshot fs = g_features.Snapshot();
      g_tm.Step(fs);
   }
   g_tel.MaybeFlush();
}

//+------------------------------------------------------------------+
//|  Trade transactions → risk notifications                         |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result) {
   // Async execution pipeline: bind req_id→order_id on REQUEST events and
   // reconcile entry fills on DEAL_ADD(ENTRY_IN). Must run before the close
   // handling below so the two paths don't fight over the same DEAL_ADD.
   g_exec.OnTradeEvent(trans, request, result);

   if(trans.type == TRADE_TRANSACTION_DEAL_ADD) {
      // Position closure deals carry profit in HistoryDealGet
      if(HistoryDealSelect(trans.deal)) {
         double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT);
         long   entry  = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
         if(entry == DEAL_ENTRY_OUT || entry == DEAL_ENTRY_INOUT) {
            // DEAL_TYPE on an OUT deal is the broker-side action that
            // closed the position — SELL closed a long, BUY closed a
            // short. Map it back to the direction of the closed
            // position so Risk can arm the correct cooldown.
            long dt = HistoryDealGetInteger(trans.deal, DEAL_TYPE);
            ENUM_TRADE_DIR closed_dir = DIR_NONE;
            if(dt == DEAL_TYPE_SELL) closed_dir = DIR_LONG;
            else if(dt == DEAL_TYPE_BUY) closed_dir = DIR_SHORT;
            g_risk.NotifyClose(profit, closed_dir);
         }
      }
   }
}
