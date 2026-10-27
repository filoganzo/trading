//+------------------------------------------------------------------+
//|                                              LinRegRevGold.mq5   |
//|  LinReg-Candle / LinReg-Channel / WaveTrend setup trader (M5)     |
//|                                                                   |
//|  Ports the validated Python engine at                             |
//|    strategies/linreg_rev/{indicators,engine}.py                   |
//|  1:1, so that a tester pass can be reconciled against the         |
//|  Python backtest bar for bar.                                     |
//|                                                                   |
//|  SETUP DETECTION (unchanged from the original TradingView spec)    |
//|    lower setup : green LinReg candle, after >= MinRun red ones,    |
//|                  with a low-band touch inside that run, and a      |
//|                  WaveTrend bull cross at or below -WtLevel.        |
//|    upper setup : the exact mirror.                                 |
//|                                                                   |
//|  DIRECTION                                                        |
//|    Reversal mode : lower setup buys, upper setup sells.            |
//|                    This is the strategy as originally specified.   |
//|                    It did NOT show an edge in testing -- kept only |
//|                    so the two modes can be compared directly.      |
//|    Momentum mode : the setups traded the other way round. This is  |
//|                    the direction the data supports (upper-band     |
//|                    "exhaustion" precedes continuation, not         |
//|                    reversal). Default.                            |
//|                                                                   |
//|  NOTE ON THE CHANNEL: the TradingView channel indicator only draws |
//|  on the last bar (barstate.islast). Here it is evaluated on every   |
//|  bar over the trailing ChanLen window, which is the per-bar        |
//|  generalisation of the same formula.                              |
//+------------------------------------------------------------------+
#property copyright "FINAL HFT"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>

//--- modes ----------------------------------------------------------
enum ENUM_TRADE_MODE
  {
   MODE_MOMENTUM = 0,   // Momentum: upper setup buys, lower setup sells
   MODE_REVERSAL = 1    // Reversal: lower setup buys, upper setup sells
  };

enum ENUM_TOUCH_SRC
  {
   TOUCH_RAW    = 0,    // raw bar high/low
   TOUCH_LINREG = 1     // LinReg-candle bhigh/blow
  };

enum ENUM_SILVER_MODE
  {
   SILVER_OFF    = 0,
   SILVER_ZONE   = 1,   // silver merely in the same band zone, same colour
   SILVER_SIGNAL = 2    // silver must print the full mirror setup
  };

//--- inputs ---------------------------------------------------------
input group "=== Mode ==="
input ENUM_TRADE_MODE InpMode           = MODE_MOMENTUM;
input bool            InpAllowLong      = true;
input bool            InpAllowShort     = false;   // short side had no edge

input group "=== Indicator geometry ==="
input int             InpCandLen        = 11;      // LinReg candle length
input int             InpChanLen        = 200;     // LinReg channel length
input double          InpChanMult       = 1.3;     // band multiplier
input int             InpWtN1           = 10;      // WaveTrend channel length
input int             InpWtN2           = 21;      // WaveTrend average length
input int             InpAtrPeriod      = 14;

input group "=== Entry conditions ==="
input int             InpMinRun         = 6;       // min opposite-colour run
input ENUM_TOUCH_SRC  InpTouchSrc       = TOUCH_RAW;
input int             InpTouchLookback  = 3;       // 0 => touch inside the run
input double          InpWtLevel        = 50.0;    // cross at/beyond +-this
input int             InpWtCrossLookback= 3;       // bars the cross may lag
input bool            InpCounterSlope   = true;    // channel leans against setup

input group "=== Silver confirmation (XAGUSD) ==="
input ENUM_SILVER_MODE InpSilverMode    = SILVER_ZONE;
input string          InpSilverSymbol   = "XAGUSD";
input int             InpSilverLookback = 6;

input group "=== Exits ==="
input double          InpSlAtr          = 2.0;
input double          InpTpAtr          = 3.0;
input int             InpMaxBars        = 24;      // time stop, M5 bars
input double          InpBeAtR          = 0.0;     // 0 = off

input group "=== Sessions (UTC hours, server clock is UTC+3) ==="
input bool            InpSessAsian      = true;    // 00-07 UTC
input bool            InpSessLondon     = true;    // 07-12 UTC
input bool            InpSessOverlap    = true;    // 12-16 UTC
input bool            InpSessNyPm       = true;    // 16-21 UTC
input bool            InpSessOff        = true;    // 21-24 UTC
input int             InpServerUtcOffset= 3;

input group "=== Risk / execution ==="
input double          InpRiskPct        = 0.5;     // % equity risked per trade
input double          InpFixedLots      = 0.0;     // >0 overrides RiskPct
input int             InpMaxConcurrent  = 2;
input int             InpSlippagePoints = 20;
input long            InpMagic          = 51073101;

//--- globals --------------------------------------------------------
CTrade   trade;
datetime g_lastBar = 0;
int      g_warmup  = 0;
int      g_scan    = 0;   // bars of setup history examined per evaluation

//+------------------------------------------------------------------+
//| Rolling OLS over the trailing L values of y ending at index end. |
//| x = 0 .. L-1 with x = L-1 the most recent bar, matching Pine's    |
//| ta.linreg(src, L, 0) and the Python _reg_parts().                 |
//| Returns the fitted value at the most recent bar; optionally the   |
//| slope and the residual standard deviation (ddof = 1, as Pine).    |
//+------------------------------------------------------------------+
double RegEndpoint(const double &y[], const int end, const int L,
                   double &slope, double &stdev)
  {
   slope = 0.0;
   stdev = 0.0;
   if(L < 2 || end < L - 1)
      return(0.0);

   const int    start = end - L + 1;
   const double n     = (double)L;
   double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0, syy = 0.0;
   for(int j = 0; j < L; j++)
     {
      const double xv = (double)j;
      const double yv = y[start + j];
      sx  += xv;
      sy  += yv;
      sxx += xv * xv;
      sxy += xv * yv;
      syy += yv * yv;
     }
   const double sxx_c = sxx - sx * sx / n;
   if(MathAbs(sxx_c) < 1e-12)
      return(sy / n);

   slope = (sxy - sx * sy / n) / sxx_c;
   const double ybar      = sy / n;
   const double intercept = ybar - slope * (sx / n);
   const double endpoint  = intercept + slope * (n - 1.0);

   const double syy_c = syy - sy * sy / n;
   double sse = syy_c - slope * slope * sxx_c;
   if(sse < 0.0)
      sse = 0.0;
   stdev = (L > 1) ? MathSqrt(sse / (n - 1.0)) : 0.0;
   return(endpoint);
  }

double RegEndpointOnly(const double &y[], const int end, const int L)
  {
   double s, d;
   return(RegEndpoint(y, end, L, s, d));
  }

//+------------------------------------------------------------------+
//| Pine ta.ema: alpha = 2/(L+1), recursion seeded with the SMA.      |
//| Writes into out[] over the whole array; out is EMPTY_VALUE until  |
//| the seed bar.                                                     |
//+------------------------------------------------------------------+
void EmaSeries(const double &src[], const int n, const int L, double &out[],
               const int startAt = 0)
  {
   ArrayResize(out, n);
   ArrayInitialize(out, EMPTY_VALUE);
   if(L < 1 || n < startAt + L)
      return;
   const double alpha = 2.0 / (L + 1.0);
   // Seed with the SMA of the first L valid samples at or after startAt, so a
   // chained EMA (WaveTrend feeds one into the next) never seeds on padding.
   double sum = 0.0;
   for(int i = startAt; i < startAt + L; i++)
      sum += src[i];
   const int seed = startAt + L - 1;
   out[seed] = sum / L;
   for(int i = seed + 1; i < n; i++)
      out[i] = alpha * src[i] + (1.0 - alpha) * out[i - 1];
  }

void SmaSeries(const double &src[], const int n, const int L, double &out[])
  {
   ArrayResize(out, n);
   ArrayInitialize(out, EMPTY_VALUE);
   if(L < 1)
      return;
   // start at the first non-EMPTY sample so wt2 = sma(wt1, 4) is not polluted
   int startAt = 0;
   while(startAt < n && src[startAt] == EMPTY_VALUE)
      startAt++;
   if(n < startAt + L)
      return;
   double sum = 0.0;
   for(int i = startAt; i < startAt + L; i++)
      sum += src[i];
   out[startAt + L - 1] = sum / L;
   for(int i = startAt + L; i < n; i++)
     {
      sum += src[i] - src[i - L];
      out[i] = sum / L;
     }
  }

//+------------------------------------------------------------------+
//| Everything the signal test needs, for one symbol, as of the last  |
//| CLOSED bar in rates[].                                            |
//+------------------------------------------------------------------+
struct SetupState
  {
   bool   valid;
   bool   lowerSetup;      // green candle off the lower band + WT bull cross
   bool   upperSetup;      // mirror
   bool   inZoneLower;     // loose test: green and below the lower band
   bool   inZoneUpper;
   double atr;
   double mid, upper, lower, slope;
  };

//+------------------------------------------------------------------+
//| Compute the setup state for a symbol. ``barsBack`` = 0 evaluates   |
//| the most recent closed bar, 1 the one before it, and so on.        |
//+------------------------------------------------------------------+
bool ComputeSetup(const string sym, const int barsBack, SetupState &st)
  {
   st.valid = false;
   st.lowerSetup = false;
   st.upperSetup = false;
   st.inZoneLower = false;
   st.inZoneUpper = false;

   const int need = g_warmup + barsBack + 2;
   MqlRates r[];
   ArraySetAsSeries(r, false);                       // chronological
   const int got = CopyRates(sym, PERIOD_M5, 0, need, r);
   if(got < need)
      return(false);

   // index of the bar under test: r[got-1] is the forming bar, so the last
   // closed bar is got-2.
   const int cur = got - 2 - barsBack;
   if(cur < g_warmup)
      return(false);
   // oldest bar we will touch must have a complete channel window behind it
   if(cur - g_scan + 1 < MathMax(InpChanLen, InpCandLen) - 1)
      return(false);

   const int n = cur + 1;
   double o[], h[], l[], c[];
   ArrayResize(o, n); ArrayResize(h, n); ArrayResize(l, n); ArrayResize(c, n);
   for(int i = 0; i < n; i++)
     {
      o[i] = r[i].open;
      h[i] = r[i].high;
      l[i] = r[i].low;
      c[i] = r[i].close;
     }

   // ---- LinReg candles over the last (MinRun + lookback + 4) bars --------
   const int scan = g_scan;
   bool   green[];
   double blow[], bhigh[];
   ArrayResize(green, scan);
   ArrayResize(blow,  scan);
   ArrayResize(bhigh, scan);
   for(int k = 0; k < scan; k++)
     {
      const int idx = cur - k;                        // k bars back from cur
      if(idx < InpCandLen - 1)
         return(false);
      const double bo = RegEndpointOnly(o, idx, InpCandLen);
      const double bc = RegEndpointOnly(c, idx, InpCandLen);
      green[k] = (bo < bc);                           // Pine: r = bopen < bclose
      blow[k]  = RegEndpointOnly(l, idx, InpCandLen);
      bhigh[k] = RegEndpointOnly(h, idx, InpCandLen);
     }

   // ---- channel + bands on each scanned bar ------------------------------
   double lowBand[], upBand[], slopeArr[];
   ArrayResize(lowBand, scan);
   ArrayResize(upBand,  scan);
   ArrayResize(slopeArr, scan);
   for(int k = 0; k < scan; k++)
     {
      const int idx = cur - k;
      double sl, sd;
      const double mid = RegEndpoint(c, idx, InpChanLen, sl, sd);
      lowBand[k]  = mid - InpChanMult * sd;
      upBand[k]   = mid + InpChanMult * sd;
      slopeArr[k] = sl;
      if(k == 0)
        {
         st.mid = mid;
         st.upper = upBand[0];
         st.lower = lowBand[0];
         st.slope = sl;
        }
     }

   // ---- red / green run immediately before the current bar ---------------
   int redRun = 0, greenRun = 0;
   for(int k = 1; k < scan; k++)
     {
      if(green[k])
         break;
      redRun++;
     }
   for(int k = 1; k < scan; k++)
     {
      if(!green[k])
         break;
      greenRun++;
     }

   // ---- band touch (wick / max-excursion counts) -------------------------
   int sinceLo = -1, sinceHi = -1;
   for(int k = 0; k < scan; k++)
     {
      const int idx = cur - k;
      const double loSrc = (InpTouchSrc == TOUCH_RAW) ? l[idx] : blow[k];
      const double hiSrc = (InpTouchSrc == TOUCH_RAW) ? h[idx] : bhigh[k];
      if(sinceLo < 0 && loSrc <= lowBand[k])
         sinceLo = k;
      if(sinceHi < 0 && hiSrc >= upBand[k])
         sinceHi = k;
     }
   const int loLimit = (InpTouchLookback > 0) ? InpTouchLookback : redRun;
   const int hiLimit = (InpTouchLookback > 0) ? InpTouchLookback : greenRun;
   const bool loOk = (sinceLo >= 0 && sinceLo <= loLimit);
   const bool hiOk = (sinceHi >= 0 && sinceHi <= hiLimit);

   // ---- WaveTrend --------------------------------------------------------
   double ap[];
   ArrayResize(ap, n);
   for(int i = 0; i < n; i++)
      ap[i] = (h[i] + l[i] + c[i]) / 3.0;
   double esa[], dev[], absd[], ci[], wt1[], wt2[];
   EmaSeries(ap, n, InpWtN1, esa);
   const int esaFirst = InpWtN1 - 1;              // first valid esa index
   ArrayResize(absd, n);
   for(int i = 0; i < n; i++)
      absd[i] = (esa[i] == EMPTY_VALUE) ? 0.0 : MathAbs(ap[i] - esa[i]);
   EmaSeries(absd, n, InpWtN1, dev, esaFirst);    // skip the padded prefix
   const int devFirst = esaFirst + InpWtN1 - 1;
   ArrayResize(ci, n);
   for(int i = 0; i < n; i++)
     {
      if(esa[i] == EMPTY_VALUE || dev[i] == EMPTY_VALUE || dev[i] == 0.0)
         ci[i] = 0.0;
      else
         ci[i] = (ap[i] - esa[i]) / (0.015 * dev[i]);
     }
   EmaSeries(ci, n, InpWtN2, wt1, devFirst);
   SmaSeries(wt1, n, 4, wt2);

   const int lb = MathMax(1, InpWtCrossLookback);
   bool bullX = false, bearX = false;
   for(int k = 0; k < lb; k++)
     {
      const int i = cur - k;
      if(i < 1)
         break;
      if(wt1[i] == EMPTY_VALUE || wt2[i] == EMPTY_VALUE ||
         wt1[i - 1] == EMPTY_VALUE || wt2[i - 1] == EMPTY_VALUE)
         continue;
      if(wt1[i] > wt2[i] && wt1[i - 1] <= wt2[i - 1] && wt1[i] <= -InpWtLevel)
         bullX = true;
      if(wt1[i] < wt2[i] && wt1[i - 1] >= wt2[i - 1] && wt1[i] >= InpWtLevel)
         bearX = true;
     }

   // ---- ATR --------------------------------------------------------------
   double atrBuf[];
   const int atrHandle = iATR(sym, PERIOD_M5, InpAtrPeriod);
   if(atrHandle == INVALID_HANDLE)
      return(false);
   ArraySetAsSeries(atrBuf, true);
   if(CopyBuffer(atrHandle, 0, 1 + barsBack, 1, atrBuf) < 1)
      return(false);
   st.atr = atrBuf[0];
   if(st.atr <= 0.0)
      return(false);

   // ---- assemble ---------------------------------------------------------
   bool lowerOk = green[0] && (redRun   >= InpMinRun) && loOk && bullX;
   bool upperOk = !green[0] && (greenRun >= InpMinRun) && hiOk && bearX;
   if(InpCounterSlope)
     {
      if(st.slope >= 0.0)
         lowerOk = false;
      if(st.slope <= 0.0)
         upperOk = false;
     }
   st.lowerSetup  = lowerOk;
   st.upperSetup  = upperOk;
   st.inZoneLower = green[0]  && loOk;
   st.inZoneUpper = !green[0] && hiOk;
   st.valid = true;
   return(true);
  }

//+------------------------------------------------------------------+
//| Silver confirmation: does XAGUSD show the same setup within the   |
//| lookback window?                                                  |
//+------------------------------------------------------------------+
bool SilverConfirms(const bool wantUpper)
  {
   if(InpSilverMode == SILVER_OFF)
      return(true);
   for(int k = 0; k < MathMax(1, InpSilverLookback); k++)
     {
      SetupState s;
      if(!ComputeSetup(InpSilverSymbol, k, s))
         continue;
      if(InpSilverMode == SILVER_SIGNAL)
        {
         if(wantUpper ? s.upperSetup : s.lowerSetup)
            return(true);
        }
      else
        {
         if(wantUpper ? s.inZoneUpper : s.inZoneLower)
            return(true);
        }
     }
   return(false);
  }

//+------------------------------------------------------------------+
bool SessionAllowed(const datetime serverTime)
  {
   MqlDateTime t;
   TimeToStruct(serverTime, t);
   int utcHour = t.hour - InpServerUtcOffset;
   if(utcHour < 0)
      utcHour += 24;
   if(utcHour >= 0  && utcHour < 7)  return(InpSessAsian);
   if(utcHour >= 7  && utcHour < 12) return(InpSessLondon);
   if(utcHour >= 12 && utcHour < 16) return(InpSessOverlap);
   if(utcHour >= 16 && utcHour < 21) return(InpSessNyPm);
   return(InpSessOff);
  }

int CountOwnPositions()
  {
   int c = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      const ulong tk = PositionGetTicket(i);
      if(tk == 0 || !PositionSelectByTicket(tk))
         continue;
      if(PositionGetInteger(POSITION_MAGIC) == InpMagic &&
         PositionGetString(POSITION_SYMBOL) == _Symbol)
         c++;
     }
   return(c);
  }

double LotsFor(const double slDistance)
  {
   if(InpFixedLots > 0.0)
      return(InpFixedLots);
   const double tickVal = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   const double tickSz  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if(tickVal <= 0.0 || tickSz <= 0.0 || slDistance <= 0.0)
      return(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN));
   const double riskCash    = AccountInfoDouble(ACCOUNT_EQUITY) * InpRiskPct / 100.0;
   const double lossPerLot  = (slDistance / tickSz) * tickVal;
   if(lossPerLot <= 0.0)
      return(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN));

   double lots = riskCash / lossPerLot;
   const double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   const double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   const double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step > 0.0)
      lots = MathFloor(lots / step) * step;
   lots = MathMax(vmin, MathMin(vmax, lots));
   return(NormalizeDouble(lots, 2));
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippagePoints);
   trade.SetTypeFillingBySymbol(_Symbol);

   // Wide enough that a long opposite-colour run is never truncated: when
   // InpTouchLookback == 0 the touch limit IS the run length, so an under-counted
   // run would silently tighten the entry test.
   g_scan = InpMinRun + MathMax(InpTouchLookback, InpWtCrossLookback) + 80;

   // The OLDEST bar in the scan window still needs a full channel behind it,
   // so the run-up must cover the channel length PLUS the scan window -- not
   // just the channel length, or the earliest scanned bars get a zero channel
   // and the band tests silently compare against 0.
   g_warmup = MathMax(InpChanLen, InpWtN1 * 3 + InpWtN2 * 6)
              + g_scan + InpCandLen + 60;
   if(InpSilverMode != SILVER_OFF)
     {
      if(!SymbolSelect(InpSilverSymbol, true))
         Print("WARNING: could not select ", InpSilverSymbol,
               " -- silver confirmation will always fail.");
     }
   PrintFormat("LinRegRevGold init: mode=%s long=%s short=%s chanLen=%d mult=%.2f "
               "minRun=%d wtLevel=%.0f silver=%d warmup=%d",
               (InpMode == MODE_MOMENTUM ? "MOMENTUM" : "REVERSAL"),
               (string)InpAllowLong, (string)InpAllowShort, InpChanLen,
               InpChanMult, InpMinRun, InpWtLevel, (int)InpSilverMode, g_warmup);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void ManageOpen()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      const ulong tk = PositionGetTicket(i);
      if(tk == 0 || !PositionSelectByTicket(tk))
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagic ||
         PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;

      // time stop
      const datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
      const int barsHeld = (int)((TimeCurrent() - opened) / (5 * 60));
      if(InpMaxBars > 0 && barsHeld >= InpMaxBars)
        {
         trade.PositionClose(tk);
         continue;
        }

      // optional breakeven
      if(InpBeAtR > 0.0)
        {
         const double entry = PositionGetDouble(POSITION_PRICE_OPEN);
         const double sl    = PositionGetDouble(POSITION_SL);
         const double tp    = PositionGetDouble(POSITION_TP);
         const long   type  = PositionGetInteger(POSITION_TYPE);
         const double risk  = MathAbs(entry - sl);
         if(risk > 0.0)
           {
            const double px = (type == POSITION_TYPE_BUY)
                              ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                              : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
            const double moved = (type == POSITION_TYPE_BUY)
                                 ? (px - entry) : (entry - px);
            const bool beDone = (type == POSITION_TYPE_BUY)
                                ? (sl >= entry) : (sl <= entry);
            if(!beDone && moved >= InpBeAtR * risk)
               trade.PositionModify(tk, entry, tp);
           }
        }
     }
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   ManageOpen();

   // act once per closed M5 bar
   const datetime barTime = iTime(_Symbol, PERIOD_M5, 0);
   if(barTime == g_lastBar)
      return;
   g_lastBar = barTime;

   if(!SessionAllowed(barTime))
      return;
   if(CountOwnPositions() >= InpMaxConcurrent)
      return;

   SetupState st;
   if(!ComputeSetup(_Symbol, 0, st) || !st.valid)
      return;
   if(!st.lowerSetup && !st.upperSetup)
      return;

   // map setup -> direction
   bool goLong = false, goShort = false;
   bool usedUpper = false;
   if(InpMode == MODE_MOMENTUM)
     {
      goLong  = st.upperSetup;   usedUpper = st.upperSetup;
      goShort = st.lowerSetup;
     }
   else
     {
      goLong  = st.lowerSetup;
      goShort = st.upperSetup;   usedUpper = st.upperSetup;
     }
   if(goLong && !InpAllowLong)
      goLong = false;
   if(goShort && !InpAllowShort)
      goShort = false;
   if(!goLong && !goShort)
      return;

   // the silver gate is keyed to the SETUP, not to the trade direction
   const bool wantUpper = usedUpper ? true : false;
   if(!SilverConfirms(wantUpper))
      return;

   const double slDist = InpSlAtr * st.atr;
   const double tpDist = InpTpAtr * st.atr;
   const int    dg     = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   const double lots   = LotsFor(slDist);
   if(lots <= 0.0)
      return;

   if(goLong)
     {
      const double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      const double sl  = NormalizeDouble(ask - slDist, dg);
      const double tp  = NormalizeDouble(ask + tpDist, dg);
      if(!trade.Buy(lots, _Symbol, 0.0, sl, tp, "LinRegRev"))
         PrintFormat("Buy failed: %d %s", trade.ResultRetcode(),
                     trade.ResultRetcodeDescription());
     }
   else
     {
      const double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      const double sl  = NormalizeDouble(bid + slDist, dg);
      const double tp  = NormalizeDouble(bid - tpDist, dg);
      if(!trade.Sell(lots, _Symbol, 0.0, sl, tp, "LinRegRev"))
         PrintFormat("Sell failed: %d %s", trade.ResultRetcode(),
                     trade.ResultRetcodeDescription());
     }
  }
//+------------------------------------------------------------------+
