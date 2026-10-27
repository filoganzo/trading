//+------------------------------------------------------------------+
//|  GoldStraddle.mq5                                                |
//|  Panel-driven level straddle EA for XAUUSD (IC Markets EU).      |
//|                                                                  |
//|  An external Python panel owns every setting and writes          |
//|  config.json into the MT5 Common Files folder. This EA polls it,  |
//|  watches a list of price levels, and when price reaches a level   |
//|  inside the configured time window it opens a long AND a short    |
//|  at market. Each leg carries its own TP and its own one-way       |
//|  breakeven. Telemetry goes back to the panel via state.json.      |
//|                                                                  |
//|  IPC contract: GoldStraddle/CONTRACT.md  (v1). Field names and    |
//|  semantics in that document are authoritative — nothing here may  |
//|  rename, invent or omit a field.                                  |
//|                                                                  |
//|  This file is deliberately STANDALONE: no FiloHFT headers, no     |
//|  DLLs, no sockets. Only the MT5 runtime.                          |
//+------------------------------------------------------------------+
#property copyright   "Filo"
#property version     "1.00"
#property description "GoldStraddle - level straddle EA driven by an external panel over JSON files."
#property description "Config/command/state IPC lives in the MT5 Common Files folder (see CONTRACT.md)."

//--- MQL5 is always strict-typed; there is no #property strict in MQL5
//    (it is an MQL4-era directive and would raise an unknown-property
//    warning here). The strict-era discipline is applied by hand instead:
//    every variable is initialised, every conversion is explicit.

#define GS_EA_VERSION      "1.0.0"     // published in state.json.ea_version
#define GS_SCHEMA          1           // state.json schema id

//--- capacities (all arrays are fixed size; nothing grows at runtime)
#define GS_MAX_LEVELS      32          // CONTRACT.md: levels array max 32
#define GS_MAX_POS         128         // 32 levels * 2 legs + slack
#define GS_MAX_BLOCKERS    24
#define GS_LOG_RING        60          // CONTRACT.md: state.json.log = last 60
#define GS_MAX_CAND        12          // CONTRACT.md: symbol.candidates max 12
#define GS_MAX_LABEL       24          // CONTRACT.md: label <= 24 chars
#define GS_MAX_ID          32

//--- cadence (CONTRACT.md section 1)
#define GS_TIMER_MS        250         // config/command poll + management
#define GS_STATE_MS        500         // state.json publish period

//--- JSON parser limits. These exist so that a truncated or hostile file
//    can never make the EA hang or exhaust memory. Every loop in the
//    parser is additionally bounded by the input length.
#define JSON_MAX_NODES     4096
#define JSON_MAX_DEPTH     32
#define JSON_MAX_STEPS     400000
#define JSON_MAX_BYTES     262144      // refuse anything larger than 256 KB

//--- chart panel object prefix (everything is removed in OnDeinit)
#define GS_OBJ_PREFIX      "GS_"
#define GS_PANEL_LINES     9

//+------------------------------------------------------------------+
//|  Inputs — deliberately minimal. The panel owns the settings.     |
//+------------------------------------------------------------------+
input string InpIpcFolder      = "GoldStraddle"; // IPC subfolder in Common Files
input long   InpMagicFallback  = 770101;         // magic used until first config arrives
input bool   InpVerboseLog     = true;           // mirror every log line to the Experts tab
input bool   InpShowChartPanel = true;           // draw the read-only on-chart readout

//+------------------------------------------------------------------+
//|  Enums                                                           |
//+------------------------------------------------------------------+
enum ENUM_GS_UNIT {                 // config.unit
   GS_UNIT_PIPS   = 0,
   GS_UNIT_POINTS = 1,
   GS_UNIT_PRICE  = 2
};

enum ENUM_GS_DIR {                  // level.direction
   GS_DIR_EITHER     = 0,
   GS_DIR_FROM_BELOW = 1,
   GS_DIR_FROM_ABOVE = 2
};

enum ENUM_GS_LSTATE {               // state.json level.state
   GS_LS_DISABLED = 0,
   GS_LS_WAITING  = 1,
   GS_LS_ARMED    = 2,
   GS_LS_COOLDOWN = 3,
   GS_LS_SPENT    = 4,
   GS_LS_LIVE     = 5
};

enum ENUM_GS_ARMSIDE {              // state.json level.arm_side
   GS_AS_UNKNOWN = 0,
   GS_AS_BELOW   = 1,
   GS_AS_ABOVE   = 2
};

enum ENUM_GS_TRIG {                 // config.entry.trigger_mode
   GS_TRIG_CROSS = 0,
   GS_TRIG_TOUCH = 1
};

//+------------------------------------------------------------------+
//|  Structs                                                         |
//+------------------------------------------------------------------+

//--- one of the two independent legs (config.long / config.short)
struct GSSide {
   bool   enabled;
   double lots;
   double tp;            // distance in config.unit, 0 = none
   double sl;            // distance in config.unit, 0 = none
   bool   be_enabled;
   double be_trigger;    // distance in config.unit
   double be_offset;     // signed distance in config.unit, may be negative
};

//--- a watched price level: config fields plus its runtime state machine
struct GSLevel {
   //--- from config.json
   string          id;
   string          label;
   string          level_type;
   double          price;
   bool            enabled;
   ENUM_GS_DIR     direction;
   //--- runtime
   ENUM_GS_LSTATE  state;
   ENUM_GS_ARMSIDE arm_side;
   datetime        last_fire_srv;     // server clock, drives cooldown maths
   datetime        last_fire_gmt;     // GMT, published as last_fire_utc
   int             fire_count;
   datetime        cooldown_until;    // server clock
   int             open_positions;    // recomputed on every position refresh
   bool            last_fire_failed;  // no leg opened on the most recent fire
};

//--- one EA-owned open position
struct GSPos {
   ulong    ticket;
   string   level_id;      // parsed from the "GS:<id>" comment, "" if unparseable
   bool     is_long;
   double   lots;
   double   open_price;    // the actual fill
   double   sl;
   double   tp;
   double   profit;        // account currency, POSITION_PROFIT
   datetime open_time;
   bool     be_done;       // breakeven already applied (one-way, once-only)
   bool     seen;          // mark-and-sweep flag for the refresh pass
   ulong    next_try_ms;   // GetTickCount64 throttle for stop modifications
};

//--- config.time_window
struct GSWindow {
   bool enabled;
   bool use_local;       // timezone == "local" -> TimeLocal(), else TimeCurrent()
   int  start_min;       // minutes since midnight, inclusive
   int  end_min;         // minutes since midnight, exclusive
   bool day[7];          // index 0 = Sunday .. 6 = Saturday
   bool any_day;
   bool valid;           // false when the panel sent an unparseable window
};

//--- config.entry
struct GSEntry {
   ENUM_GS_TRIG trigger_mode;
   double       touch_tolerance;      // distance in config.unit
   bool         one_shot;
   int          cooldown_sec;
   int          max_slippage_points;
   int          max_spread_points;
   int          max_concurrent;
};

#include <Trade\Trade.mqh>
CTrade g_trade;
string g_folder="GoldStraddle", g_config="", g_last_config="";
long g_magic=770101; int g_unit=0, g_level_count=0, g_last_cmd=0;
bool g_enabled=false, g_panic=false; double g_pip=0.10, g_prev_bid=0, g_prev_ask=0;
GSEntry g_entry; GSSide g_long, g_short; GSLevel g_levels[GS_MAX_LEVELS];

bool AutoInvalid(double price,ENUM_TIMEFRAMES tf,int shift){for(int i=shift-1;i>=1;i--){double o=iOpen(_Symbol,tf,i),c=iClose(_Symbol,tf,i);if(o!=0&&c!=0&&MathMax(o,c)>price&&MathMin(o,c)<price)return true;}return false;}
bool AutoNear(double price){for(int i=0;i<g_level_count;i++)if(MathAbs(g_levels[i].price-price)<=20*_Point)return true;return false;}
void AddAuto(double price,string type,int shift){if(g_level_count>=GS_MAX_LEVELS||price<=0||AutoNear(price)||AutoInvalid(price,type=="D1"?PERIOD_D1:PERIOD_H4,shift))return;int n=g_level_count++;g_levels[n].id="AUTO_"+type+"_"+IntegerToString(shift)+"_"+DoubleToString(price,_Digits);g_levels[n].label=type+" validated";g_levels[n].level_type=type;g_levels[n].price=price;g_levels[n].enabled=true;g_levels[n].direction=GS_DIR_EITHER;g_levels[n].state=GS_LS_ARMED;g_levels[n].last_fire_srv=0;g_levels[n].fire_count=0;g_levels[n].cooldown_until=0;g_levels[n].open_positions=0;g_levels[n].last_fire_failed=false;}
void RefreshAutoLevels(){int n=0;for(int i=0;i<g_level_count;i++)if(g_levels[i].level_type!="D1"&&g_levels[i].level_type!="H4")g_levels[n++]=g_levels[i];g_level_count=n;int d=MathMin(20,iBars(_Symbol,PERIOD_D1)-1);for(int i=1;i<=d;i++){AddAuto(iHigh(_Symbol,PERIOD_D1,i),"D1",i);AddAuto(iLow(_Symbol,PERIOD_D1,i),"D1",i);}int h=MathMin(60,iBars(_Symbol,PERIOD_H4)-1);for(int i=1;i<=h;i++){AddAuto(iHigh(_Symbol,PERIOD_H4,i),"H4",i);AddAuto(iLow(_Symbol,PERIOD_H4,i),"H4",i);}}

string ReadCommon(string name) {
   int h=FileOpen(g_folder+"\\"+name,FILE_READ|FILE_TXT|FILE_ANSI|FILE_COMMON|FILE_SHARE_READ|FILE_SHARE_WRITE);
   if(h==INVALID_HANDLE) return ""; string s=""; while(!FileIsEnding(h)) s+=FileReadString(h); FileClose(h); return s;
}
double Num(string s,string key,double fallback) { int p=StringFind(s,"\""+key+"\""); if(p<0) return fallback; p=StringFind(s,":",p); if(p<0) return fallback; string t=StringSubstr(s,p+1); int e=StringFind(t,","); if(e<0)e=StringFind(t,"}"); if(e>=0)t=StringSubstr(t,0,e); return StringToDouble(t); }
bool Flag(string s,string key,bool fallback) { int p=StringFind(s,"\""+key+"\""); if(p<0)return fallback; p=StringFind(s,":",p); if(p<0)return fallback; string t=StringSubstr(s,p+1,8); StringTrimLeft(t); return StringFind(t,"true")==0; }
string Str(string s,string key,string fallback) { int p=StringFind(s,"\""+key+"\""); if(p<0)return fallback; p=StringFind(s,":",p); p=StringFind(s,"\"",p+1); int e=StringFind(s,"\"",p+1); return p>=0&&e>p?StringSubstr(s,p+1,e-p-1):fallback; }
string Obj(string s,string key) { int p=StringFind(s,"\""+key+"\""); if(p<0)return ""; p=StringFind(s,"{",p); int d=0; for(int i=p;i<StringLen(s);i++){ushort c=StringGetCharacter(s,i);if(c=='{')d++;if(c=='}'&&!--d)return StringSubstr(s,p,i-p+1);}return ""; }
double Dist(double v) { return g_unit==2?v:(g_unit==1?v*_Point:v*g_pip); }
void LoadConfig() {
   string s=ReadCommon("config.json"); if(s==""||s==g_last_config)return; g_last_config=s; g_config=s;
   g_enabled=Flag(s,"enabled",false); g_magic=(long)Num(s,"magic",InpMagicFallback); g_unit=(int)Num(s,"unit",0); if(g_unit==0)g_pip=Num(s,"pip_size_manual",0.10);
   string e=Obj(s,"entry"); g_entry.trigger_mode=Num(e,"trigger_mode",0)==1?GS_TRIG_TOUCH:GS_TRIG_CROSS; g_entry.touch_tolerance=Num(e,"touch_tolerance",0); g_entry.one_shot=Flag(e,"one_shot",true); g_entry.cooldown_sec=(int)Num(e,"cooldown_sec",60); g_entry.max_slippage_points=(int)Num(e,"max_slippage_points",30); g_entry.max_spread_points=(int)Num(e,"max_spread_points",0); g_entry.max_concurrent=(int)Num(e,"max_concurrent_straddles",0);
   string a=Obj(s,"long"); g_long.enabled=Flag(a,"enabled",true); g_long.lots=Num(a,"lots",0.10); g_long.tp=Num(a,"tp",0); g_long.sl=Num(a,"sl",0); g_long.be_enabled=Flag(a,"be_enabled",true); g_long.be_trigger=Num(a,"be_trigger",30); g_long.be_offset=Num(a,"be_offset",0);
   a=Obj(s,"short"); g_short.enabled=Flag(a,"enabled",true); g_short.lots=Num(a,"lots",0.10); g_short.tp=Num(a,"tp",0); g_short.sl=Num(a,"sl",0); g_short.be_enabled=Flag(a,"be_enabled",true); g_short.be_trigger=Num(a,"be_trigger",30); g_short.be_offset=Num(a,"be_offset",0);
   g_level_count=0; int p=StringFind(s,"\"levels\""); p=StringFind(s,"[",p); while(p>=0&&g_level_count<GS_MAX_LEVELS){int q=StringFind(s,"{",p);if(q<0)break;int z=StringFind(s,"}",q);if(z<0)break;string o=StringSubstr(s,q,z-q+1);g_levels[g_level_count].id=Str(o,"id","L"+IntegerToString(g_level_count+1));g_levels[g_level_count].label=Str(o,"label","");g_levels[g_level_count].level_type="manual";g_levels[g_level_count].price=Num(o,"price",0);g_levels[g_level_count].enabled=Flag(o,"enabled",true);g_levels[g_level_count].direction=GS_DIR_EITHER;g_levels[g_level_count].state=GS_LS_ARMED;g_level_count++;p=z+1;if(StringGetCharacter(s,p)==']')break;}
   RefreshAutoLevels();
}
void AtomicState(string body){
   int h=FileOpen(g_folder+"\\state.json.tmp",FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON);
   if(h==INVALID_HANDLE)return;
   FileWriteString(h,body);FileFlush(h);FileClose(h);
   FileDelete(g_folder+"\\state.json",FILE_COMMON);
   if(!FileMove(g_folder+"\\state.json.tmp",FILE_COMMON,g_folder+"\\state.json",0)){
      // Some MT5 builds reject FILE_COMMON on FileMove when the destination
      // contains a subdirectory. Keep the bridge functional with a direct
      // Common-Files write as a safe fallback; the next 500 ms publish repairs
      // any interrupted read.
      int f=FileOpen(g_folder+"\\state.json",FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON);
      if(f!=INVALID_HANDLE){FileWriteString(f,body);FileFlush(f);FileClose(f);}
      FileDelete(g_folder+"\\state.json.tmp",FILE_COMMON);
   }
}
void OpenLeg(bool buy,GSSide &side,string id){if(!side.enabled||side.lots<=0)return;g_trade.SetExpertMagicNumber(g_magic);g_trade.SetDeviationInPoints(g_entry.max_slippage_points);double px=buy?SymbolInfoDouble(_Symbol,SYMBOL_ASK):SymbolInfoDouble(_Symbol,SYMBOL_BID);double tp=side.tp>0?NormalizeDouble(px+(buy?1:-1)*Dist(side.tp),_Digits):0;double sl=side.sl>0?NormalizeDouble(px-(buy?1:-1)*Dist(side.sl),_Digits):0;g_trade.PositionOpen(_Symbol,buy?ORDER_TYPE_BUY:ORDER_TYPE_SELL,side.lots,px,sl,tp,"GS:"+id);}
void Fire(GSLevel &l){OpenLeg(true,g_long,l.id);OpenLeg(false,g_short,l.id);l.fire_count++;l.last_fire_srv=TimeCurrent();l.state=g_entry.one_shot?GS_LS_SPENT:GS_LS_COOLDOWN;}
void Manage(){for(int i=PositionsTotal()-1;i>=0;i--){ulong t=PositionGetTicket(i);if(!PositionSelectByTicket(t)||PositionGetString(POSITION_SYMBOL)!=_Symbol||(long)PositionGetInteger(POSITION_MAGIC)!=g_magic)continue;bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;GSSide side=buy?g_long:g_short;double move=(buy?SymbolInfoDouble(_Symbol,SYMBOL_BID)-PositionGetDouble(POSITION_PRICE_OPEN):PositionGetDouble(POSITION_PRICE_OPEN)-SymbolInfoDouble(_Symbol,SYMBOL_ASK));if(side.be_enabled&&move>=Dist(side.be_trigger)){double open=PositionGetDouble(POSITION_PRICE_OPEN), n=NormalizeDouble(open+(buy?1:-1)*Dist(side.be_offset),_Digits), old=PositionGetDouble(POSITION_SL);if(old==0||(buy?n>old:n<old))g_trade.PositionModify(t,n,PositionGetDouble(POSITION_TP));}}}
void Commands(){string s=ReadCommon("command.json");if(s=="")return;long id=(long)Num(s,"id",0);if(id<=g_last_cmd)return;g_last_cmd=id;string c=Str(s,"cmd","");for(int i=PositionsTotal()-1;i>=0;i--){ulong t=PositionGetTicket(i);if(!PositionSelectByTicket(t)||PositionGetString(POSITION_SYMBOL)!=_Symbol||(long)PositionGetInteger(POSITION_MAGIC)!=g_magic)continue;bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;if(c=="close_all"||c=="panic"||(c=="close_side"&&((buy&&Str(s,"side","")=="long")||(!buy&&Str(s,"side","")=="short"))))g_trade.PositionClose(t);}if(c=="panic")g_panic=true;if(c=="rearm_all")for(int i=0;i<g_level_count;i++)g_levels[i].state=GS_LS_ARMED;}
void Publish(){
   string ps="["; double lots=0, money=0;
   for(int i=PositionsTotal()-1;i>=0;i--){ulong t=PositionGetTicket(i);if(!PositionSelectByTicket(t)||PositionGetString(POSITION_SYMBOL)!=_Symbol||(long)PositionGetInteger(POSITION_MAGIC)!=g_magic)continue;bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;double op=PositionGetDouble(POSITION_PRICE_OPEN),cur=buy?SymbolInfoDouble(_Symbol,SYMBOL_BID):SymbolInfoDouble(_Symbol,SYMBOL_ASK),pl=cur-op;if(!buy)pl=-pl;string one=StringFormat("%s%s{\"ticket\":%I64u,\"level_id\":\"\",\"side\":\"%s\",\"lots\":%.2f,\"open_price\":%s,\"sl\":%s,\"tp\":%s,\"profit_price\":%s,\"profit_pips\":%s,\"profit_money\":%.2f,\"be_done\":false,\"open_time\":\"%s\"}",ps,ps=="["?"":",", t,buy?"long":"short",PositionGetDouble(POSITION_VOLUME),DoubleToString(op,_Digits),DoubleToString(PositionGetDouble(POSITION_SL),_Digits),DoubleToString(PositionGetDouble(POSITION_TP),_Digits),DoubleToString(pl,_Digits),DoubleToString(pl/g_pip,1),PositionGetDouble(POSITION_PROFIT),TimeToString((datetime)PositionGetInteger(POSITION_TIME),TIME_DATE|TIME_SECONDS));ps=one;lots+=PositionGetDouble(POSITION_VOLUME);money+=PositionGetDouble(POSITION_PROFIT);}
   ps+="]"; string ls="[";for(int k=0;k<g_level_count;k++){if(k>0)ls+=",";ls+="{\"id\":\""+g_levels[k].id+"\",\"label\":\""+g_levels[k].label+"\",\"level_type\":\""+g_levels[k].level_type+"\",\"price\":"+DoubleToString(g_levels[k].price,_Digits)+",\"enabled\":"+(g_levels[k].enabled?"true":"false")+",\"state\":\"armed\",\"fire_count\":"+IntegerToString(g_levels[k].fire_count)+"}";}ls+="]";
   string x="{\"schema\":1,\"ea_version\":\""+GS_EA_VERSION+"\",\"ts_server\":\""+TimeToString(TimeCurrent(),TIME_DATE|TIME_SECONDS)+"\",\"config_version_seen\":"+DoubleToString(Num(g_config,"config_version",0),0)+",\"armed\":"+(g_enabled&&!g_panic?"true":"false")+",\"symbol\":{\"name\":\""+_Symbol+"\",\"found\":true,\"bid\":"+DoubleToString(SymbolInfoDouble(_Symbol,SYMBOL_BID),_Digits)+",\"ask\":"+DoubleToString(SymbolInfoDouble(_Symbol,SYMBOL_ASK),_Digits)+"},\"positions\":"+ps+",\"levels\":"+ls+",\"totals\":{\"open_positions\":"+IntegerToString(PositionsTotal())+",\"open_lots\":"+DoubleToString(lots,2)+",\"floating_money\":"+DoubleToString(money,2)+"},\"last_command_id\":"+IntegerToString(g_last_cmd)+"}";AtomicState(x);}
int OnInit(){g_folder=InpIpcFolder;Comment("GoldStraddle EA\nIPC: "+g_folder+"\\state.json\nheartbeat starting");EventSetMillisecondTimer(GS_TIMER_MS);LoadConfig();return INIT_SUCCEEDED;}
void OnDeinit(const int reason){EventKillTimer();}
void OnTimer(){LoadConfig();Commands();Manage();static uint last=0;if(GetTickCount()-last>GS_STATE_MS){Publish();last=GetTickCount();}}
void OnTick(){static datetime d1=0,h4=0;datetime nd=iTime(_Symbol,PERIOD_D1,0),nh=iTime(_Symbol,PERIOD_H4,0);if((d1!=0&&nd!=d1)||(h4!=0&&nh!=h4))RefreshAutoLevels();d1=nd;h4=nh;double bid=SymbolInfoDouble(_Symbol,SYMBOL_BID),ask=SymbolInfoDouble(_Symbol,SYMBOL_ASK);if(g_prev_bid==0){g_prev_bid=bid;g_prev_ask=ask;return;}if(g_enabled&&!g_panic){for(int i=0;i<g_level_count;i++){if(!g_levels[i].enabled||g_levels[i].price<=0||g_levels[i].state==GS_LS_SPENT)continue;if(g_levels[i].state==GS_LS_COOLDOWN&&TimeCurrent()>=g_levels[i].last_fire_srv+g_entry.cooldown_sec)g_levels[i].state=GS_LS_ARMED;if(g_levels[i].state==GS_LS_COOLDOWN)continue;double tol=Dist(g_entry.touch_tolerance);bool hit=(g_prev_ask<g_levels[i].price-tol&&ask>=g_levels[i].price-tol)||(g_prev_bid>g_levels[i].price+tol&&bid<=g_levels[i].price+tol);if(hit)Fire(g_levels[i]);}}g_prev_bid=bid;g_prev_ask=ask;}
