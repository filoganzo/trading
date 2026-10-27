//+------------------------------------------------------------------+
//|                                              BridgeExecutor.mq5  |
//|  Execution half of the OANDA -> Vantage bridge.                  |
//|                                                                  |
//|  This EA computes NOTHING. The Python panel decides, using OANDA  |
//|  gold/silver and IC Markets DXY, and writes two files into the    |
//|  shared MetaQuotes Common\Files folder:                           |
//|                                                                   |
//|      bridge_settings.json  - lots, TP, SL, breakeven, magic       |
//|      bridge_signal.json    - the current command                  |
//|                                                                   |
//|  The EA reads them, checks that the command is fresh and that it  |
//|  has not already acted on it, and sends a market order.           |
//|                                                                   |
//|  REAL MONEY. Four independent gates must all be open before an    |
//|  order is sent:                                                   |
//|      1. InpAllowLiveTrading (this input, default FALSE)           |
//|      2. "armed" in the signal file (the panel's ON/OFF switch)    |
//|      3. the signal is younger than its ttl                        |
//|      4. its id differs from the last one executed                 |
//|  Any one of them shut means nothing is sent.                      |
//+------------------------------------------------------------------+
#property copyright "bridge"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//=== INPUTS =========================================================
input group "=== Safety ==="
input bool   InpAllowLiveTrading = false;  // MASTER: allow real orders
input bool   InpVerboseLog       = true;   // Log every decision

input group "=== Files ==="
input string InpSettingsFile = "bridge_settings.json";  // Settings (Common\Files)
input string InpSignalFile   = "bridge_signal.json";    // Signal (Common\Files)
input string InpHeartbeatFile= "bridge_heartbeat.json"; // Heartbeat out (Common\Files)

input group "=== Execution ==="
input int    InpPollMs       = 1000;   // Poll interval (ms)
input int    InpMaxSignalAge = 120;    // Hard cap on signal age (s)
input double InpPipSize      = 0.10;   // Pip size (price units)

//=== STATE ==========================================================
CTrade   g_trade;
string   g_lastExecId   = "";
datetime g_lastPoll     = 0;
bool     g_beDone       = false;
ulong    g_beTicket     = 0;
int      g_digits       = 2;
double   g_point        = 0.01;
string   g_lastStatus   = "";

// settings mirrored from the panel
double g_lots = 0.01, g_tpPips = 40.0, g_slPips = 400.0;
double g_beTrigger = 35.0, g_beBuffer = 1.0;
bool   g_useBe = true, g_onePosOnly = true, g_enabled = false;
int    g_magic = 880155, g_slippage = 20, g_maxSpreadPts = 0;

//+------------------------------------------------------------------+
//| Minimal JSON scalar reader.                                       |
//|                                                                   |
//| The panel writes these files, so the shape is known and flat -    |
//| a full parser would be dead weight. Finds "key" then takes the    |
//| text after the colon up to the next comma or brace.               |
//+------------------------------------------------------------------+
string JsonRaw(const string js, const string key, bool &found)
{
   found = false;
   string pat = "\"" + key + "\"";
   int p = StringFind(js, pat);
   if(p < 0) return "";
   int colon = StringFind(js, ":", p + StringLen(pat));
   if(colon < 0) return "";

   int i = colon + 1;
   int n = StringLen(js);
   while(i < n)
   {
      ushort ch = StringGetCharacter(js, i);
      if(ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') break;
      i++;
   }
   if(i >= n) return "";

   string out = "";
   if(StringGetCharacter(js, i) == '"')          // quoted string
   {
      i++;
      while(i < n && StringGetCharacter(js, i) != '"')
      {
         out += ShortToString(StringGetCharacter(js, i));
         i++;
      }
   }
   else                                          // number / bool / null
   {
      while(i < n)
      {
         ushort ch = StringGetCharacter(js, i);
         if(ch == ',' || ch == '}' || ch == '\n' || ch == '\r') break;
         out += ShortToString(ch);
         i++;
      }
   }
   StringTrimLeft(out);
   StringTrimRight(out);
   found = true;
   return out;
}

string JsonStr(const string js, const string key, string def)
{
   bool f; string v = JsonRaw(js, key, f);
   return f ? v : def;
}

double JsonNum(const string js, const string key, double def)
{
   bool f; string v = JsonRaw(js, key, f);
   if(!f || StringLen(v) == 0) return def;
   return StringToDouble(v);
}

bool JsonBool(const string js, const string key, bool def)
{
   bool f; string v = JsonRaw(js, key, f);
   if(!f) return def;
   StringToLower(v);
   if(v == "true"  || v == "1") return true;
   if(v == "false" || v == "0") return false;
   return def;
}

//+------------------------------------------------------------------+
//| Read a whole file from the shared Common\Files folder.            |
//| Returns false if it is missing or locked mid-write.               |
//+------------------------------------------------------------------+
bool ReadCommon(const string name, string &out)
{
   out = "";
   int h = FileOpen(name, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON |
                          FILE_SHARE_READ | FILE_SHARE_WRITE);
   if(h == INVALID_HANDLE) return false;
   while(!FileIsEnding(h)) out += FileReadString(h) + "\n";
   FileClose(h);
   return StringLen(out) > 0;
}

void WriteHeartbeat(const string status)
{
   int h = FileOpen(InpHeartbeatFile, FILE_WRITE | FILE_TXT | FILE_ANSI |
                                      FILE_COMMON | FILE_SHARE_READ);
   if(h == INVALID_HANDLE) return;

   long   type   = -1;
   double vol = 0, openPx = 0, profit = 0;
   if(PositionSelect(_Symbol) && PositionGetInteger(POSITION_MAGIC) == g_magic)
   {
      type   = PositionGetInteger(POSITION_TYPE);
      vol    = PositionGetDouble(POSITION_VOLUME);
      openPx = PositionGetDouble(POSITION_PRICE_OPEN);
      profit = PositionGetDouble(POSITION_PROFIT);
   }

   FileWriteString(h, StringFormat(
      "{\"ts\":%d,\"symbol\":\"%s\",\"status\":\"%s\",\"allow_live\":%s,"
      "\"bid\":%.2f,\"ask\":%.2f,\"spread_pts\":%.0f,"
      "\"pos_type\":%d,\"pos_volume\":%.2f,\"pos_open\":%.2f,\"pos_profit\":%.2f,"
      "\"equity\":%.2f,\"last_exec_id\":\"%s\"}",
      (int)TimeGMT(), _Symbol, status,
      InpAllowLiveTrading ? "true" : "false",
      SymbolInfoDouble(_Symbol, SYMBOL_BID),
      SymbolInfoDouble(_Symbol, SYMBOL_ASK),
      (double)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD),
      (int)type, vol, openPx, profit,
      AccountInfoDouble(ACCOUNT_EQUITY), g_lastExecId));
   FileClose(h);
}

void Status(const string s)
{
   if(InpVerboseLog && s != g_lastStatus) Print(s);
   g_lastStatus = s;
}

//+------------------------------------------------------------------+
//| Settings                                                          |
//+------------------------------------------------------------------+
void LoadSettings()
{
   string js;
   if(!ReadCommon(InpSettingsFile, js)) return;

   g_enabled      = JsonBool(js, "enabled",           g_enabled);
   g_lots         = JsonNum (js, "lots",              g_lots);
   g_magic        = (int)JsonNum(js, "magic",         g_magic);
   g_slippage     = (int)JsonNum(js, "slippage_points", g_slippage);
   g_maxSpreadPts = (int)JsonNum(js, "max_spread_points", g_maxSpreadPts);
   g_tpPips       = JsonNum (js, "tp_pips",           g_tpPips);
   g_slPips       = JsonNum (js, "sl_pips",           g_slPips);
   g_useBe        = JsonBool(js, "use_breakeven",     g_useBe);
   g_beTrigger    = JsonNum (js, "be_trigger_pips",   g_beTrigger);
   g_beBuffer     = JsonNum (js, "be_buffer_pips",    g_beBuffer);
   g_onePosOnly   = JsonBool(js, "one_position_only", g_onePosOnly);

   g_trade.SetExpertMagicNumber(g_magic);
   g_trade.SetDeviationInPoints(g_slippage);
}

//+------------------------------------------------------------------+
//| Position helpers                                                  |
//+------------------------------------------------------------------+
bool HasOurPosition(ulong &ticket)
{
   ticket = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != g_magic) continue;
      ticket = tk;
      return true;
   }
   return false;
}

double NormalizeLots(double lots)
{
   double mn   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double mx   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0) step = 0.01;
   lots = MathRound(lots / step) * step;
   if(lots < mn) lots = mn;
   if(lots > mx) lots = mx;
   return NormalizeDouble(lots, 2);
}

bool SpreadOK()
{
   if(g_maxSpreadPts <= 0) return true;
   return (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) <= g_maxSpreadPts;
}

//+------------------------------------------------------------------+
//| Entry                                                             |
//+------------------------------------------------------------------+
void DoEnter(bool isLong, double lots, double tpPips, double slPips,
             const string sigId, const string reason)
{
   ulong tk;
   if(g_onePosOnly && HasOurPosition(tk))
   {
      Status("skip: already in a position");
      return;
   }
   if(!SpreadOK())
   {
      Status(StringFormat("skip: spread %d > max %d",
             (int)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD), g_maxSpreadPts));
      return;
   }

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ref = isLong ? ask : bid;
   if(ref <= 0)
   {
      Status("skip: no price");
      return;
   }

   double tpDist = tpPips * InpPipSize;
   double slDist = slPips * InpPipSize;
   double sl = isLong ? ref - slDist : ref + slDist;
   double tp = isLong ? ref + tpDist : ref - tpDist;

   // respect the broker's minimum stop distance
   int stopsLvl = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minDist = stopsLvl * g_point;
   if(minDist > 0)
   {
      if(isLong)
      {
         if(ref - sl < minDist) sl = ref - minDist;
         if(tp - ref < minDist) tp = ref + minDist;
      }
      else
      {
         if(sl - ref < minDist) sl = ref + minDist;
         if(ref - tp < minDist) tp = ref - minDist;
      }
   }

   sl = NormalizeDouble(sl, g_digits);
   tp = NormalizeDouble(tp, g_digits);
   lots = NormalizeLots(lots);

   bool ok = isLong ? g_trade.Buy (lots, _Symbol, 0.0, sl, tp, "bridge")
                    : g_trade.Sell(lots, _Symbol, 0.0, sl, tp, "bridge");

   if(ok)
   {
      // Mark the signal consumed only on success, so a rejected order can be
      // retried while the signal is still fresh.
      g_lastExecId = sigId;
      g_beDone = false;
      g_beTicket = 0;
      PrintFormat("ENTER %s %.2f lots @ %.2f  sl %.2f  tp %.2f  [%s] %s",
                  isLong ? "LONG" : "SHORT", lots, ref, sl, tp, sigId, reason);
   }
   else
   {
      PrintFormat("ORDER FAILED %s retcode=%d %s",
                  isLong ? "BUY" : "SELL",
                  g_trade.ResultRetcode(), g_trade.ResultRetcodeDescription());
   }
}

void CloseAll(const string why)
{
   ulong tk;
   while(HasOurPosition(tk))
   {
      if(!g_trade.PositionClose(tk))
      {
         PrintFormat("CLOSE FAILED ticket %I64u retcode=%d", tk,
                     g_trade.ResultRetcode());
         return;
      }
      PrintFormat("CLOSED ticket %I64u (%s)", tk, why);
   }
}

//+------------------------------------------------------------------+
//| Breakeven, managed locally so it keeps working even if the panel  |
//| dies mid-trade.                                                   |
//+------------------------------------------------------------------+
void ManageBreakeven()
{
   if(!g_useBe) return;

   ulong tk;
   if(!HasOurPosition(tk)) { g_beDone = false; g_beTicket = 0; return; }
   if(!PositionSelectByTicket(tk)) return;

   if(tk != g_beTicket) { g_beTicket = tk; g_beDone = false; }
   if(g_beDone) return;

   long   type   = PositionGetInteger(POSITION_TYPE);
   double openPx = PositionGetDouble(POSITION_PRICE_OPEN);
   double curSl  = PositionGetDouble(POSITION_SL);
   double tp     = PositionGetDouble(POSITION_TP);
   bool   isLong = (type == POSITION_TYPE_BUY);

   double px   = isLong ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                        : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double gain = isLong ? (px - openPx) : (openPx - px);
   if(gain < g_beTrigger * InpPipSize) return;

   double buf    = g_beBuffer * InpPipSize;
   double target = isLong ? openPx + buf : openPx - buf;
   target = NormalizeDouble(target, g_digits);

   // never move a stop backwards
   if(isLong  && curSl >= target) { g_beDone = true; return; }
   if(!isLong && curSl > 0 && curSl <= target) { g_beDone = true; return; }

   if(g_trade.PositionModify(tk, target, tp))
   {
      g_beDone = true;
      PrintFormat("BREAKEVEN ticket %I64u sl -> %.2f (gain %.1f pips)",
                  tk, target, gain / InpPipSize);
   }
}

//+------------------------------------------------------------------+
int OnInit()
{
   g_digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   g_point  = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   g_trade.SetExpertMagicNumber(g_magic);
   g_trade.SetDeviationInPoints(g_slippage);
   g_trade.SetTypeFillingBySymbol(_Symbol);

   LoadSettings();
   EventSetMillisecondTimer(InpPollMs);

   PrintFormat("BridgeExecutor on %s | allow_live=%s | magic=%d | reading %s",
               _Symbol, InpAllowLiveTrading ? "TRUE" : "FALSE",
               g_magic, InpSignalFile);

   if(!InpAllowLiveTrading)
      Print("NOTE: InpAllowLiveTrading is FALSE - signals will be logged, "
            "not executed. Set it true when you want the bridge live.");

   WriteHeartbeat("init");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   WriteHeartbeat("deinit");
}

//+------------------------------------------------------------------+
void OnTimer()
{
   LoadSettings();
   ManageBreakeven();

   string js;
   if(!ReadCommon(InpSignalFile, js))
   {
      Status("no signal file - panel not running");
      WriteHeartbeat("no_signal_file");
      return;
   }

   string action = JsonStr (js, "action", "FLAT");
   string sigId  = JsonStr (js, "id",     "");
   string dir    = JsonStr (js, "dir",    "");
   string reason = JsonStr (js, "reason", "");
   bool   armed  = JsonBool(js, "armed",  false);
   double ts     = JsonNum (js, "ts",     0);
   double ttl    = JsonNum (js, "ttl",    90);
   string sym    = JsonStr (js, "symbol", _Symbol);

   if(sym != _Symbol)
   {
      Status("signal is for " + sym + ", not " + _Symbol);
      WriteHeartbeat("wrong_symbol");
      return;
   }

   // The panel stamps ts in real UTC; TimeGMT() is the terminal's UTC. A stale
   // file means the panel died, so we must not act on its last instruction.
   double age = (double)TimeGMT() - ts;
   double maxAge = MathMin(ttl, (double)InpMaxSignalAge);
   if(age > maxAge)
   {
      Status(StringFormat("signal stale (%.0fs old, max %.0fs)", age, maxAge));
      WriteHeartbeat("stale");
      return;
   }

   if(action == "CLOSE_ALL")
   {
      if(sigId != g_lastExecId)
      {
         g_lastExecId = sigId;
         CloseAll("panel CLOSE_ALL");
      }
      WriteHeartbeat("close_all");
      return;
   }

   if(!armed || !g_enabled)
   {
      Status("panel DISARMED");
      WriteHeartbeat("disarmed");
      return;
   }

   if(action != "ENTER")
   {
      Status("armed, waiting: " + reason);
      WriteHeartbeat("armed_waiting");
      return;
   }

   if(sigId == g_lastExecId)
   {
      WriteHeartbeat("already_executed");
      return;
   }

   if(!InpAllowLiveTrading)
   {
      PrintFormat("SIGNAL (not executed, allow_live=false): %s %s | %s",
                  action, dir, reason);
      g_lastExecId = sigId;     // log once, not every poll
      WriteHeartbeat("signal_blocked_by_input");
      return;
   }

   if(!MQLInfoInteger(MQL_TRADE_ALLOWED) || !TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
   {
      Status("algo trading disabled in the terminal");
      WriteHeartbeat("algo_disabled");
      return;
   }

   double lots = JsonNum(js, "lots",    g_lots);
   double tpP  = JsonNum(js, "tp_pips", g_tpPips);
   double slP  = JsonNum(js, "sl_pips", g_slPips);

   DoEnter(dir == "LONG", lots, tpP, slP, sigId, reason);
   WriteHeartbeat("executed");
}

void OnTick()
{
   ManageBreakeven();
}
//+------------------------------------------------------------------+
