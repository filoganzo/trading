//+------------------------------------------------------------------+
//|                                             KeyLevelTracker.mq5 |
//|  Multi-timeframe KL visualizer (D1, H4, H1, M30, M15, M5).       |
//|  This is the visual-only companion to the EA's internal KL      |
//|  engine. It does NOT share state with the EA — it recomputes    |
//|  independently. If you want truth, trust telemetry/keylevels.csv|
//|  from the EA. This indicator exists so the chart shows roughly  |
//|  what the engine sees.                                          |
//+------------------------------------------------------------------+
#property copyright "FiloHFT"
#property version   "1.00"
#property indicator_chart_window
#property indicator_plots 0

input int    InpLookbackDays       = 30;
input double InpZoneHalfWidthPips  = 1.5;
input double InpMergeOverlapPips   = 2.0;

input color  InpColorD1  = clrRed;       input int InpWidthD1  = 3;
input color  InpColorH4  = clrOrange;    input int InpWidthH4  = 2;
input color  InpColorH1  = clrGold;      input int InpWidthH1  = 2;
input color  InpColorM30 = clrDeepSkyBlue; input int InpWidthM30 = 1;
input color  InpColorM15 = clrMediumSeaGreen; input int InpWidthM15 = 1;
input color  InpColorM5  = clrSlateGray; input int InpWidthM5  = 1;

string PREFIX = "FLHKL_";

double ZoneHalfWidthPrice() { return InpZoneHalfWidthPips * 10.0 * _Point; }
double MergeOverlapPrice () { return InpMergeOverlapPips  * 10.0 * _Point; }

// -----------------------------------------------------------------
// Invalidation: a level is dead if any later closed bar's body
// crosses it. Same rule as the EA's KL engine.
bool IsInvalidated(const double levelPrice, const ENUM_TIMEFRAMES tf, const int startShift) {
   for(int i = startShift - 1; i >= 1; i--) {
      double o = iOpen(_Symbol, tf, i);
      double c = iClose(_Symbol, tf, i);
      if(o == 0 || c == 0) continue;
      if(MathMax(o, c) > levelPrice && MathMin(o, c) < levelPrice) return true;
   }
   return false;
}

// -----------------------------------------------------------------
bool IsOverlapping(const double price, const double &existing[], const int count) {
   for(int i = 0; i < count; i++)
      if(MathAbs(price - existing[i]) < MergeOverlapPrice()) return true;
   return false;
}

// -----------------------------------------------------------------
void PlotLevel(const double price, const string tag, color clr, const int width) {
   string name = PREFIX + tag + "_" + DoubleToString(price, _Digits);
   if(ObjectFind(0, name) >= 0) return;
   ObjectCreate(0, name, OBJ_HLINE, 0, 0, price);
   ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, name, OBJPROP_STYLE, STYLE_SOLID);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, width);
   ObjectSetInteger(0, name, OBJPROP_BACK, true);
}

// -----------------------------------------------------------------
int BarsForTf(const ENUM_TIMEFRAMES tf) {
   int d = InpLookbackDays;
   switch(tf) {
      case PERIOD_D1:  return d;
      case PERIOD_H4:  return d * 6;
      case PERIOD_H1:  return d * 24;
      case PERIOD_M30: return d * 48;
      case PERIOD_M15: return d * 96;
      case PERIOD_M5:  return d * 288;
   }
   return 0;
}

// -----------------------------------------------------------------
void ScanTf(const ENUM_TIMEFRAMES tf, const string tag, color clr, const int width,
            double &existing[], int &existing_count) {
   int lookback = MathMin(BarsForTf(tf), iBars(_Symbol, tf) - 1);
   for(int i = 1; i <= lookback; i++) {
      double h = iHigh(_Symbol, tf, i);
      double l = iLow (_Symbol, tf, i);
      if(h == 0 || l == 0) continue;

      if(!IsOverlapping(h, existing, existing_count) && !IsInvalidated(h, tf, i)) {
         PlotLevel(h, tag, clr, width);
         if(existing_count < ArraySize(existing)) existing[existing_count++] = h;
      }
      if(!IsOverlapping(l, existing, existing_count) && !IsInvalidated(l, tf, i)) {
         PlotLevel(l, tag, clr, width);
         if(existing_count < ArraySize(existing)) existing[existing_count++] = l;
      }
   }
}

// -----------------------------------------------------------------
void RefreshLevels() {
   ObjectsDeleteAll(0, PREFIX);

   double existing[2048];
   int    count = 0;

   // Higher TF first so lower TF levels are suppressed when overlapping
   ScanTf(PERIOD_D1,  "D1",  InpColorD1,  InpWidthD1,  existing, count);
   ScanTf(PERIOD_H4,  "H4",  InpColorH4,  InpWidthH4,  existing, count);
   ScanTf(PERIOD_H1,  "H1",  InpColorH1,  InpWidthH1,  existing, count);
   ScanTf(PERIOD_M30, "M30", InpColorM30, InpWidthM30, existing, count);
   ScanTf(PERIOD_M15, "M15", InpColorM15, InpWidthM15, existing, count);
   ScanTf(PERIOD_M5,  "M5",  InpColorM5,  InpWidthM5,  existing, count);

   ChartRedraw();
}

// -----------------------------------------------------------------
int OnInit() { RefreshLevels(); return(INIT_SUCCEEDED); }
void OnDeinit(const int reason) { ObjectsDeleteAll(0, PREFIX); }

int OnCalculate(const int rates_total, const int prev_calculated, const datetime &time[],
                const double &open[], const double &high[], const double &low[],
                const double &close[], const long &tick_volume[], const long &volume[],
                const int &spread[]) {
   static datetime last_bar;
   if(rates_total <= 0) return 0;
   if(time[0] != last_bar) {
      RefreshLevels();
      last_bar = time[0];
   }
   return rates_total;
}
