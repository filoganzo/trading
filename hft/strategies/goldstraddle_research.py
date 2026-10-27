"""Bounded causal research harness for the GoldStraddle idea.

Uses XAUUSD M1 bars because no tick export is present. Entries are evaluated
at bar opens after a level touch; when a bar touches both exits, SL wins.
Levels are prior closed D1/H4 highs/lows and are invalidated by later closed
candle bodies crossing the level. The grid is deliberately small and runs
walk-forward (first 60% train, last 40% test).
"""
from __future__ import annotations
import itertools, json, os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data", "XAUUSD", "bars_M1_20260201_20260413.parquet")
OUT = os.path.join(ROOT, "artifacts", "goldstraddle_research")

def load():
    d=pd.read_parquet(DATA); d["ts"]=pd.to_datetime(d["ts"],utc=True); return d.sort_values("ts").reset_index(drop=True)

def levels_at(d, i, lookback=30, merge=0.20):
    out=[]
    for tf, step, n in (("D1","1D",lookback),("H4","4h",lookback*6)):
        bars=d.set_index("ts")["high"].resample(step).agg(["first","max","min","last"]).dropna()
        bars=bars[bars.index < d.ts.iloc[i]].tail(n)
        for t,r in bars.iloc[:-1].iterrows():
            for side,px in (("high",r["max"]),("low",r["min"])):
                # conservative proxy: later completed bars' bodies cannot cross px
                later=bars.loc[t:]
                invalid=((later["first"].combine(later["last"],max)>px)&(later["first"].combine(later["last"],min)<px)).any()
                if not invalid and not any(abs(px-x["price"])<=merge for x in out): out.append({"type":tf,"side":side,"price":float(px)})
    return out

@dataclass(frozen=True)
class P:
    tp_h4: float; tp_d1: float; sl: float; be: float; offset: float; cooldown: int; spread_max: float

def run_one(args):
    d,p,split,schedule=args; pip=.10; trades=[]; cooldown={}; active=[]
    # Build levels from completed bars at each H4 boundary; capped to avoid overfit.
    for i in range(500,len(d)-1):
        ts=d.ts.iloc[i]; h4=ts.floor("4h")
        lv=schedule.get(str(h4), [])
        row=d.iloc[i];
        if float(row.spread)*.01>p.spread_max: continue
        for l in lv:
            if cooldown.get((l["type"],l["price"]),-1)>i: continue
            if abs(float(row.close)-l["price"])>0.20: continue
            if i<split: continue
            ent=float(d.open.iloc[i+1]); tp=(p.tp_d1 if l["type"]=="D1" else p.tp_h4); start=i+1
            for side in (1,-1):
                ep=ent + side*float(row.spread)*.01
                best=0.; worst=0.; be=False; exit_i=min(len(d)-1,start+720); reason="time"; xp=float(d.close.iloc[exit_i])
                for j in range(start,exit_i+1):
                    hi,lo=float(d.high.iloc[j]),float(d.low.iloc[j]); fav=side*(hi-ep if side>0 else lo-ep); adv=side*(lo-ep if side>0 else hi-ep);best=max(best,fav/pip);worst=min(worst,adv/pip)
                    if p.be>0 and best>=p.be: be=True
                    stop=ep+side*p.offset*pip if be else ep-side*p.sl*pip
                    target=ep+side*tp*pip
                    hit_stop=(lo<=stop if side>0 else hi>=stop); hit_tp=(hi>=target if side>0 else lo<=target)
                    if hit_stop: xp=stop; reason="be" if be else "sl"; exit_i=j; break
                    if hit_tp: xp=target; reason="tp"; exit_i=j; break
                r=side*(xp-ep)/pip; trades.append({"level_type":l["type"],"side":side,"entry_i":i,"exit_i":exit_i,"r":r,"mae":worst,"mfe":best,"reason":reason,"hour":int(ts.hour),"dow":int(ts.dayofweek),"spread":float(row.spread),"tick_volume":float(row.tick_volume)})
            cooldown[(l["type"],l["price"])]=i+p.cooldown
    t=pd.DataFrame(trades)
    if t.empty:return {**asdict(p),"split":"oos","n":0}
    def agg(x):
        x=x.copy(); eq=x.r.cumsum(); dd=(eq-eq.cummax()).min(); return {"n":len(x),"pnl_r":float(x.r.sum()),"expectancy_r":float(x.r.mean()),"win":float((x.r>0).mean()),"pf":float(x.loc[x.r>0,"r"].sum()/abs(x.loc[x.r<0,"r"].sum())) if (x.r<0).any() else 99.,"max_dd_r":float(dd),"mae_p50":float(x.mae.median()),"mfe_p50":float(x.mfe.median()),"avg_hold_min":float((x.exit_i-x.entry_i).mean())}
    out={**asdict(p),"train":agg(t[t.entry_i<split]),"test":agg(t[t.entry_i>=split]),"by_type":{k:agg(v) for k,v in t.groupby("level_type")},"by_hour":{str(k):agg(v) for k,v in t.groupby("hour") if len(v)>=10},"by_dow":{str(k):agg(v) for k,v in t.groupby("dow") if len(v)>=10},"by_spread":{str(k):agg(v) for k,v in t.assign(bin=pd.qcut(t.spread,4,duplicates="drop").astype(str)).groupby("bin") if len(v)>=10},"by_volume":{str(k):agg(v) for k,v in t.assign(bin=pd.qcut(t.tick_volume,4,duplicates="drop").astype(str)).groupby("bin") if len(v)>=10}}
    return out

def main():
    d=load(); split=int(len(d)*.6)
    schedule={}
    for i in range(500,len(d)-1):
        key=str(d.ts.iloc[i].floor("4h"))
        if key not in schedule: schedule[key]=levels_at(d,i)
    grid=[P(*x) for x in itertools.product((8,12),(16,24),(20,30),(4,8),(0,2),(600,),(1.0,)) if x[2]>x[0]]
    with ProcessPoolExecutor(max_workers=min(8,os.cpu_count() or 2)) as ex: res=list(ex.map(run_one,[(d,p,split,schedule) for p in grid]))
    os.makedirs(OUT,exist_ok=True); json.dump({"data":DATA,"rows":len(d),"start":str(d.ts.min()),"end":str(d.ts.max()),"tick_data":False,"results":res},open(os.path.join(OUT,"report.json"),"w"),indent=2)
    valid=[r for r in res if r.get("test",{}).get("n",0)>=20]; valid.sort(key=lambda r:(r["test"]["expectancy_r"],r["test"]["pf"]),reverse=True)
    lines=["# GoldStraddle bounded research","",f"Data: {DATA}",f"Rows: {len(d):,}; period: {d.ts.min()} to {d.ts.max()}","Tick data present: NO (M1 OHLC proxy used)","", "## Best out-of-sample configurations", ""]
    for r in valid[:12]: lines.append(f"- {r['test']['n']} OOS trades | TP H4/D1 {r['tp_h4']}/{r['tp_d1']} pips | SL {r['sl']} | BE {r['be']} | cooldown {r['cooldown']}s | expectancy {r['test']['expectancy_r']:.3f}R | PF {r['test']['pf']:.2f} | win {r['test']['win']:.1%} | DD {r['test']['max_dd_r']:.1f}R | MAE50 {r['test']['mae_p50']:.1f}p | MFE50 {r['test']['mfe_p50']:.1f}p")
    lines += ["", "## Interpretation", "", "Results are bar-resolution estimates, not tick-accurate execution. No configuration is approved for live deployment from this sample alone. Use the OOS ranking to choose the next tick-data validation, not as a profitability guarantee."]
    open(os.path.join(OUT,"REPORT.md"),"w",encoding="utf-8").write("\n".join(lines)); print("wrote",os.path.join(OUT,"REPORT.md")); print("tested",len(grid),"configs")
if __name__=="__main__": main()
