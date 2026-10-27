//+------------------------------------------------------------------+
//|  FiloHFT/Execution.mqh                                           |
//|  Converts triggered intents into broker-side market orders.     |
//|                                                                 |
//|  EXECUTION MODEL — asynchronous since 2026-04-14.                |
//|  Every OrderSend is async (CTrade.SetAsyncMode(true)). The       |
//|  ExecuteIntent call returns immediately after queueing, so the  |
//|  OnTick loop is never blocked by broker roundtrip latency.      |
//|  Fills are reconciled through OnTradeEvent, invoked from        |
//|  OnTradeTransaction in the main EA. Downstream work             |
//|  (fill telemetry, TMLab spawn, TradeManagement attach) happens  |
//|  inside OnTradeEvent once the DEAL_ADD for the entry arrives.   |
//|                                                                 |
//|  Why this matters: IC Markets demo ack latency hit p95 ~10.4s / |
//|  p99 ~21s / max 54s on 2026-04-14. Under the old synchronous    |
//|  model, every slow ack froze the entire tick handler, causing  |
//|  189 tick_handler gaps > 1s in a single session and ~125 pips   |
//|  of adverse total slippage from stale-price fills.              |
//|                                                                 |
//|  Orders are submitted with stoploss/takeprofit omitted — the    |
//|  trade management engine holds those internally (hidden SL/TP). |
//+------------------------------------------------------------------+
#ifndef __FILOHFT_EXECUTION_MQH__
#define __FILOHFT_EXECUTION_MQH__

#include <FiloHFT/Common.mqh>
#include <FiloHFT/Config.mqh>
#include <FiloHFT/Telemetry.mqh>
#include <FiloHFT/Features.mqh>
#include <FiloHFT/ExecutionIntent.mqh>
#include <FiloHFT/TMLab.mqh>
#include <FiloHFT/TradeManagement.mqh>
#include <FiloHFT/Risk.mqh>
#include <FiloHFT/Safety.mqh>
#include <Trade/Trade.mqh>

#define FILOHFT_EXEC_MAGIC            20260413
#define FILOHFT_EXEC_PENDING_MAX      8
#define FILOHFT_EXEC_PENDING_STALE_MS 30000

//--- A single in-flight entry submission. Lives from ExecuteIntent
//    until the matching DEAL_ADD arrives (or until SweepStale expires).
struct PendingExec {
   bool             alive;
   bool             bound;          // order_id has been captured via REQUEST event
   ulong            req_id;         // async request id from ResultRequestId()
   ulong            order_id;       // order ticket allocated by the broker
   ExecutionIntent  intent;
   long             send_ts_ms;     // wall-clock ms at the moment of submission
   double           send_mid;       // mid price captured at submission time
   double           lots_sent;      // lots after all resizing (momentum/knife)
};

class Execution {
private:
   CTrade       m_trade;
   int          m_reject_count;
   PendingExec  m_pending[FILOHFT_EXEC_PENDING_MAX];

   int _AllocPendingSlot() {
      for(int i = 0; i < FILOHFT_EXEC_PENDING_MAX; i++)
         if(!m_pending[i].alive) return i;
      return -1;
   }
   int _FindByReqId(const ulong req_id) const {
      if(req_id == 0) return -1;
      for(int i = 0; i < FILOHFT_EXEC_PENDING_MAX; i++)
         if(m_pending[i].alive && m_pending[i].req_id == req_id) return i;
      return -1;
   }
   int _FindByOrderId(const ulong order_id) const {
      if(order_id == 0) return -1;
      for(int i = 0; i < FILOHFT_EXEC_PENDING_MAX; i++)
         if(m_pending[i].alive && m_pending[i].order_id == order_id) return i;
      return -1;
   }
   int _FindOldestInDir(const ENUM_TRADE_DIR dir) const {
      long oldest = LONG_MAX;
      int idx = -1;
      for(int i = 0; i < FILOHFT_EXEC_PENDING_MAX; i++) {
         if(!m_pending[i].alive) continue;
         if(m_pending[i].intent.direction != dir) continue;
         if(m_pending[i].send_ts_ms < oldest) {
            oldest = m_pending[i].send_ts_ms;
            idx = i;
         }
      }
      return idx;
   }
   void _ReleasePending(const int i) {
      if(i < 0 || i >= FILOHFT_EXEC_PENDING_MAX) return;
      ZeroMemory(m_pending[i]);
   }

public:
   void Init() {
      m_reject_count = 0;
      for(int i = 0; i < FILOHFT_EXEC_PENDING_MAX; i++) ZeroMemory(m_pending[i]);
      m_trade.SetExpertMagicNumber(FILOHFT_EXEC_MAGIC);
      m_trade.SetTypeFilling(ORDER_FILLING_IOC);
      m_trade.SetDeviationInPoints((ulong)MathMax(1, (int)PointsFromPips(g_cfg.exec_max_slippage_pips)));
      m_trade.SetAsyncMode(true);
   }

   int RejectCount() const { return m_reject_count; }

   int PendingCount() const {
      int n = 0;
      for(int i = 0; i < FILOHFT_EXEC_PENDING_MAX; i++)
         if(m_pending[i].alive) n++;
      return n;
   }

   bool HasPendingInDirection(const ENUM_TRADE_DIR dir) const {
      for(int i = 0; i < FILOHFT_EXEC_PENDING_MAX; i++)
         if(m_pending[i].alive && m_pending[i].intent.direction == dir) return true;
      return false;
   }

   //--- Called by main loop when IntentBook reports a trigger.
   //    Non-blocking: queues the async send, stores the pending entry,
   //    returns true if the submission was accepted locally. Fill
   //    reconciliation happens later through OnTradeEvent.
   bool ExecuteIntent(const ExecutionIntent &e, const FeatureSnapshot &fs) {
      // final spread gate — re-check at trigger time
      double max_spread = g_cfg.exec_max_spread_default;
      if(e.family == SETUP_BREAKOUT || e.family == SETUP_RETEST)
         max_spread = g_cfg.exec_max_spread_breakout;
      else if(e.family == SETUP_MEAN_REVERSION || e.family == SETUP_FAILED_AUCTION ||
              e.family == SETUP_REJECTION)
         max_spread = g_cfg.exec_max_spread_reversion;
      if(fs.spread_pips > max_spread) {
         g_tel.WriteError(fs.ts_ms, "warn", "exec", 1,
                          StringFormat("spread %.2f > %.2f at trigger", fs.spread_pips, max_spread));
         return false;
      }

      double lots = g_cfg.fixed_lot;
      if(lots <= 0) return false;

      // Momentum-continuation size-down. This family explicitly enters LATE
      // into a running move, so it accepts smaller R:R for lower entry-price
      // quality by paying with reduced size. Applied BEFORE knife-size so
      // the two compound correctly if both fire.
      if(e.family == SETUP_MOMENTUM_CONT &&
         g_cfg.setup_momentum_cont_lot_mult > 0.0 &&
         g_cfg.setup_momentum_cont_lot_mult < 1.0) {
         double vmin  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
         double vstep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
         if(vstep <= 0.0) vstep = 0.01;
         double reduced = lots * g_cfg.setup_momentum_cont_lot_mult;
         reduced = MathRound(reduced / vstep) * vstep;
         if(reduced < vmin) reduced = vmin;
         g_tel.WriteError(fs.ts_ms, "info", "momentum_cont_size", 0,
            StringFormat("lots %.2f->%.2f mult=%.2f dir=%s",
                         lots, reduced, g_cfg.setup_momentum_cont_lot_mult,
                         DirName(e.direction)));
         lots = reduced;
      }

      // Knife size-down. If the last completed M1 bar moved >= threshold pips
      // AGAINST our entry direction, halve the size. Taking the trade is fine
      // on average (~77% wr in backtest), but tail slippage on the eventual
      // stop-exit is 1.4-2x baseline because we land inside a fat minute.
      // Halving lot caps that tail cost while keeping the expectancy positive.
      double adv_pips = 0.0;
      if(g_cfg.exec_knife_size_threshold_pips > 0.0 &&
         g_cfg.exec_knife_size_mult > 0.0 &&
         g_cfg.exec_knife_size_mult < 1.0) {
         double o1 = iOpen (_Symbol, PERIOD_M1, 1);
         double c1 = iClose(_Symbol, PERIOD_M1, 1);
         if(o1 > 0.0 && c1 > 0.0) {
            double body_pips = PipsFromPrice(c1 - o1);
            adv_pips = (e.direction == DIR_LONG) ? -body_pips : body_pips;
            if(adv_pips >= g_cfg.exec_knife_size_threshold_pips) {
               double vmin  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
               double vstep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
               if(vstep <= 0.0) vstep = 0.01;
               double reduced = lots * g_cfg.exec_knife_size_mult;
               reduced = MathRound(reduced / vstep) * vstep;
               if(reduced < vmin) reduced = vmin;
               g_tel.WriteError(fs.ts_ms, "info", "knife_resize", 0,
                  StringFormat("adv=%.1f pips thr=%.1f lots %.2f->%.2f dir=%s",
                               adv_pips, g_cfg.exec_knife_size_threshold_pips,
                               lots, reduced, DirName(e.direction)));
               lots = reduced;
            }
         }
      }

      int slot = _AllocPendingSlot();
      if(slot < 0) {
         g_tel.WriteError(fs.ts_ms, "warn", "exec", 2,
                          "pending ring full - refusing submission");
         return false;
      }

      // Async submit. Buy/Sell returns true if the request was QUEUED to
      // the broker gateway, not if it was filled. Actual fill arrives
      // later via OnTradeTransaction / OnTradeEvent.
      bool queued = false;
      if(e.direction == DIR_LONG) {
         queued = m_trade.Buy(lots, _Symbol, 0.0 /*price=market*/, 0.0 /*sl*/, 0.0 /*tp*/,
                              "FiloHFT:" + SetupFamilyName(e.family));
      } else {
         queued = m_trade.Sell(lots, _Symbol, 0.0, 0.0, 0.0,
                               "FiloHFT:" + SetupFamilyName(e.family));
      }

      if(!queued) {
         m_reject_count++;
         g_tel.WriteError(fs.ts_ms, "error", "exec",
                          (int)m_trade.ResultRetcode(),
                          StringFormat("async queue failed: %s",
                                       m_trade.ResultRetcodeDescription()));
         return false;
      }

      // Store the pending entry. The req_id binds us to the REQUEST
      // transaction that will land shortly with the broker-allocated
      // order_id; the order_id then binds us to the DEAL_ADD that
      // confirms the actual fill. CTrade doesn't expose request_id via
      // a getter — copy the whole result struct instead.
      MqlTradeResult send_res;
      ZeroMemory(send_res);
      m_trade.Result(send_res);

      PendingExec p;
      ZeroMemory(p);
      p.alive      = true;
      p.bound      = false;
      p.req_id     = send_res.request_id;
      p.order_id   = send_res.order;          // may be 0 at this point in async
      p.bound      = (p.order_id != 0);
      p.intent     = e;
      p.send_ts_ms = (long)fs.ts_ms;
      p.send_mid   = fs.mid;
      p.lots_sent  = lots;
      m_pending[slot] = p;

      if(g_cfg.tel_log_latency) {
         g_tel.WriteLatency(fs.ts_ms, "order_queue", 0.0, 0, 0, 0);
      }
      return true;
   }

   //--- Release pending entries that have been waiting too long without
   //    a matching REQUEST or DEAL transaction. Called from OnTimer.
   void SweepStale(const long now_ms) {
      for(int i = 0; i < FILOHFT_EXEC_PENDING_MAX; i++) {
         if(!m_pending[i].alive) continue;
         long age = now_ms - m_pending[i].send_ts_ms;
         if(age > FILOHFT_EXEC_PENDING_STALE_MS) {
            g_tel.WriteError(now_ms, "warn", "exec_stale_pending", 0,
               StringFormat("releasing pending req_id=%I64u order=%I64u age=%dms dir=%s",
                            m_pending[i].req_id,
                            m_pending[i].order_id,
                            (int)age,
                            DirName(m_pending[i].intent.direction)));
            _ReleasePending(i);
         }
      }
   }

   //--- Routed from OnTradeTransaction in FiloHFT.mq5. Handles both
   //    the REQUEST event (req_id → order_id binding + broker rejects)
   //    and the DEAL_ADD event (actual fill confirmation).
   void OnTradeEvent(const MqlTradeTransaction &trans,
                     const MqlTradeRequest &request,
                     const MqlTradeResult &result) {
      // ----- Path A: REQUEST transaction binds req_id to order_id -----
      if(trans.type == TRADE_TRANSACTION_REQUEST) {
         int idx = _FindByReqId(result.request_id);
         if(idx < 0) return;
         if(result.retcode != TRADE_RETCODE_DONE &&
            result.retcode != TRADE_RETCODE_DONE_PARTIAL &&
            result.retcode != TRADE_RETCODE_PLACED) {
            m_reject_count++;
            g_tel.WriteError(NowMs(), "error", "exec", (int)result.retcode,
               StringFormat("async broker reject req_id=%I64u retcode=%u",
                            result.request_id, result.retcode));
            // Risk/Safety counted these under the synchronous model —
            // keep them notified now that the reject arrives out-of-band.
            g_risk.NotifyReject();
            g_safety.OnOrderLatency((double)(NowMs() - m_pending[idx].send_ts_ms));
            _ReleasePending(idx);
            return;
         }
         if(result.order != 0) {
            m_pending[idx].order_id = result.order;
            m_pending[idx].bound    = true;
         }
         return;
      }

      // ----- Path B: DEAL_ADD with DEAL_ENTRY_IN is our fill -----
      if(trans.type == TRADE_TRANSACTION_DEAL_ADD) {
         if(trans.deal == 0) return;
         if(!HistoryDealSelect(trans.deal)) return;

         long magic = HistoryDealGetInteger(trans.deal, DEAL_MAGIC);
         if(magic != FILOHFT_EXEC_MAGIC) return;

         long entry = HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
         if(entry != DEAL_ENTRY_IN) return;  // closes handled by OnTradeTransaction directly

         ulong deal_order = (ulong)HistoryDealGetInteger(trans.deal, DEAL_ORDER);
         int idx = _FindByOrderId(deal_order);
         if(idx < 0) {
            // Fallback: match by direction + oldest. Happens when REQUEST
            // event carried no order_id and we couldn't bind upfront.
            long deal_type = HistoryDealGetInteger(trans.deal, DEAL_TYPE);
            ENUM_TRADE_DIR filled_dir = DIR_NONE;
            if(deal_type == DEAL_TYPE_BUY)  filled_dir = DIR_LONG;
            if(deal_type == DEAL_TYPE_SELL) filled_dir = DIR_SHORT;
            idx = _FindOldestInDir(filled_dir);
            if(idx < 0) return;  // not ours, or already reconciled
         }
         _ProcessFill(idx, trans.deal);
         return;
      }
   }

private:
   void _ProcessFill(const int idx, const ulong deal_id) {
      if(idx < 0 || idx >= FILOHFT_EXEC_PENDING_MAX) return;
      if(!HistoryDealSelect(deal_id)) { _ReleasePending(idx); return; }

      PendingExec p = m_pending[idx];
      ExecutionIntent e = p.intent;

      double act_px       = HistoryDealGetDouble (deal_id, DEAL_PRICE);
      double filled_vol   = HistoryDealGetDouble (deal_id, DEAL_VOLUME);
      long   deal_time_ms = (long)HistoryDealGetInteger(deal_id, DEAL_TIME_MSC);
      long   position_id  = (long)HistoryDealGetInteger(deal_id, DEAL_POSITION_ID);

      double req_px = e.trigger_price;
      double slip_raw = (e.direction == DIR_LONG) ? (act_px - req_px) : (req_px - act_px);
      double slip_p   = PipsFromPrice(slip_raw);

      double submit_slip_raw = (e.direction == DIR_LONG)
                               ? (act_px - p.send_mid)
                               : (p.send_mid - act_px);
      double submit_slip_p = PipsFromPrice(submit_slip_raw);

      long latency_ms = deal_time_ms - p.send_ts_ms;
      if(latency_ms < 0) latency_ms = 0;

      long ts_now = NowMs();

      if(g_cfg.tel_log_fills) {
         g_tel.WriteFill(ts_now, e.id, position_id, DirName(e.direction),
                         req_px, act_px, slip_p, filled_vol, (double)latency_ms);
      }
      if(g_cfg.tel_log_latency) {
         g_tel.WriteLatency(ts_now, "order_send_async", (double)latency_ms, 0, 0, 0);
         g_tel.WriteError(ts_now, "info", "exec_fill", 0,
            StringFormat("req=%I64u order=%I64u pos=%I64d dir=%s lat=%dms trig_slip=%.1fp submit_slip=%.1fp lots=%.2f",
                         p.req_id, p.order_id, position_id,
                         DirName(e.direction), (int)latency_ms,
                         slip_p, submit_slip_p, filled_vol));
      }

      // Feed the real broker-ack latency into Safety so its EMA and
      // halt thresholds track the async pipeline, not a dead code path.
      g_safety.OnOrderLatency((double)latency_ms);

      // Downstream pipeline now runs — once the position actually exists.
      FeatureSnapshot fs_now = g_features.Snapshot();

      g_tmlab.OnFill(position_id, e.family, e.direction, act_px,
                     e.setup_score, e.kl_weight, e.echo_count, fs_now);

      g_tm.AttachIntent(e, fs_now);

      _ReleasePending(idx);
   }
};

Execution g_exec;

#endif // __FILOHFT_EXECUTION_MQH__
