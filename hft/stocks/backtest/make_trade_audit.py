"""
Full trade audit: re-runs best v8 params on US-Ene, captures every
price level and produces PNG trade-journal tables + corrected stats.

Columns per trade:
  #  date_entry  sym  dir  entry_px  level_px  sl_px  tp_px  exit_px
  exit_type  r  n_shares  risk$  pnl$  cum_pnl$
"""
import sys, warnings
from datetime import datetime
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
sys.path.insert(0, r'path/to/repo/hft')

from stocks.backtest.config import (RESULTS_DIR, BT_CACHE_DIR,
                                    BT_CACHE_M15_DIR, BT_CACHE_W1_DIR)
from stocks.backtest.data.extender import load_bt_bars, load_bt_bars_m15, load_bt_bars_w1
from stocks.backtest.data.universe import build_universe
from stocks.backtest.signals.wt_lrc_signal import (
    _lrc_vectorized, _lrc_candle_close_vectorized, _atr_vectorized)
from stocks.backtest.signals.htf_signal import (
    precompute_weekly_levels, find_htf_level, find_m15_entry)

DESKTOP  = Path(r"path/to/output")
US_EXCH  = {"NYSE", "NAS"}
EU_EXCH  = {"LSE","ETR","PAR","IT","MAD","AMS","IE","SWX"}
ALL_EXCH = US_EXCH | EU_EXCH
MIN_BARS = 200
LRC_CL=15; LRC_CS=9; ATR_P=14

LEVERAGE  = 20
CAPITAL   = 1_000.0
EFFECTIVE = CAPITAL * LEVERAGE   # $20 000

LRC_LEN=80; SIGMA_MULT=1.0
HTF_PCT=0.5; M15_PCT=0.5
MIN_SCORE=60; SWING_WING=2
SL_ATR=1.0; TP_RR=1.0

BG="#0f1117"; CARD="#1a1d27"; EDGE="#2a2d3a"
GREEN="#00d09c"; RED="#ff4d4d"; BLUE="#4a9eff"
GOLD="#ffd700"; GRAY="#8b95a1"; WHITE="#e8eaf0"; PURPLE="#b48eff"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": EDGE, "text.color": WHITE,
    "axes.labelcolor": GRAY, "xtick.color": GRAY, "ytick.color": GRAY,
    "grid.color": EDGE, "grid.linewidth": 0.5,
    "font.family": "monospace", "figure.dpi": 150,
    "text.parse_math": False,   # treat $ as literal dollar sign
})


# ── Precompute ────────────────────────────────────────────────────────────────
def precompute(df, m15_df, w1_df):
    c=df["close"].values; h=df["high"].values
    l=df["low"].values;   o=df["open"].values
    htf_tf = HTF_PCT/100.0
    mid,upper,lower,_=_lrc_vectorized(c,LRC_LEN,SIGMA_MULT)
    w_levels={}
    if w1_df is not None and not w1_df.empty and len(w1_df)>=4:
        wc={col.lower():col for col in w1_df.columns}
        wh=w1_df[wc.get("high","High")].values
        wl=w1_df[wc.get("low","Low")].values
        lvls=precompute_weekly_levels(wh,wl,[htf_tf],swing_wing=SWING_WING)
        for tf,lv in lvls.items(): w_levels[(SWING_WING,tf)]=lv
    else:
        w_levels[(SWING_WING,htf_tf)]=[]
    m15_by_date={}
    if m15_df is not None and not m15_df.empty:
        m=m15_df.copy(); m.columns=[cc.lower() for cc in m.columns]
        m["_d"]=m.index.normalize()
        for d,g in m.groupby("_d"):
            m15_by_date[d]=list(zip(g["high"].values,g["low"].values,
                                    g["close"].values,g["open"].values))
    return {"c":c,"h":h,"l":l,"o":o,"n":len(c),"idx":df.index,
            "lrc_c":_lrc_candle_close_vectorized(c,LRC_CL,LRC_CS),
            "atr":_atr_vectorized(h,l,c,ATR_P),
            "mid":mid,"upper":upper,"lower":lower,
            "w_levels":w_levels,"m15":m15_by_date}


def find_exit_full(h,l,c,eb,ep,sl_px,tp_px,direction,n):
    for i in range(eb,n):
        sl_hit=(l[i]<=sl_px) if direction=="LONG" else (h[i]>=sl_px)
        tp_hit=(h[i]>=tp_px) if direction=="LONG" else (l[i]<=tp_px)
        if sl_hit: return i, float(sl_px), "SL"
        if tp_hit: return i, float(tp_px), "TP"
    return n-1, float(c[n-1]), "END"


def collect_trades(precomp, sym):
    pc=precomp; htf_tf=HTF_PCT/100.0; m15_tf=M15_PCT/100.0
    n=pc["n"]; lrc_c=pc["lrc_c"]; atr=pc["atr"]
    h=pc["h"]; l=pc["l"]; c=pc["c"]; idx=pc["idx"]
    upper=pc["upper"]; lower=pc["lower"]
    mid=pc["mid"]
    w_levels=pc["w_levels"].get((SWING_WING,htf_tf),[])
    if not w_levels: return []
    min_i=max(LRC_LEN,LRC_CL+LRC_CS+1,ATR_P+1)
    if min_i>=n-1: return []

    trades=[]
    for i in range(min_i,n-1):
        lci=lrc_c[i]
        if np.isnan(lci) or np.isnan(upper[i]) or np.isnan(atr[i]): continue
        if   lci>upper[i]: direction="SHORT"
        elif lci<lower[i]: direction="LONG"
        else: continue

        price=float(c[i])
        lv,score=find_htf_level(w_levels,price,htf_tf,MIN_SCORE)
        if np.isnan(lv): continue

        eb=i+1
        day_key=pd.Timestamp(idx[eb]).normalize()
        m15_bars=pc["m15"].get(day_key)
        if not m15_bars: continue
        _,entry_px=find_m15_entry(m15_bars,direction,lv,m15_tf)
        if entry_px is None: continue

        ep=float(entry_px)
        pos=+1 if direction=="SHORT" else -1
        sl_px=float(lv+pos*SL_ATR*float(atr[i]))
        if pos*(sl_px-ep)<=0: continue
        risk=abs(ep-sl_px)
        if risk<1e-6: continue
        tp_px=float(ep-pos*TP_RR*risk)

        xb,exit_px,exit_type=find_exit_full(h,l,c,eb,ep,sl_px,tp_px,direction,n)

        r=(exit_px-ep)/risk if direction=="LONG" else (ep-exit_px)/risk
        n_shares=math.floor(EFFECTIVE/ep)
        dollar_risk=round(n_shares*risk,2)
        dollar_pnl =round(r*dollar_risk,2)

        trades.append({
            "sym":        sym,
            "direction":  direction,
            "entry_date": str(pd.Timestamp(idx[eb]).date()),
            "exit_date":  str(pd.Timestamp(idx[xb]).date()),
            "entry_px":   round(ep,3),
            "level_px":   round(float(lv),3),
            "sl_px":      round(sl_px,3),
            "tp_px":      round(tp_px,3),
            "exit_px":    round(exit_px,3),
            "exit_type":  exit_type,
            "r":          round(r,3),
            "n_shares":   n_shares,
            "risk_usd":   dollar_risk,
            "pnl_usd":    dollar_pnl,
        })
    return trades


# ── Main data run ─────────────────────────────────────────────────────────────
print("Loading universe...")
univ=build_universe()
bars={}; dvols={}; sym_meta={}
for _,row in univ.iterrows():
    sym=row["symbol"]; exch=row["exchange"]
    if exch not in ALL_EXCH: continue
    df=load_bt_bars(sym)
    if df.empty or len(df)<MIN_BARS: continue
    df=df[["Open","High","Low","Close","Volume"]].copy()
    df.columns=["open","high","low","close","volume"]
    dvol=(df["close"]*df["volume"]).replace(0,np.nan).dropna()
    if len(dvol)<MIN_BARS: continue
    bars[sym]=df; dvols[sym]=float(dvol.median())
    sym_meta[sym]={"exchange":exch,"sector":row.get("sector","Unknown")}

us_all={s for s,m in sym_meta.items() if m["exchange"] in US_EXCH}
us_top25=frozenset(pd.Series({s:dvols[s] for s in us_all}).nlargest(int(len(us_all)*0.25)).index)
ene_syms=frozenset(s for s in us_top25 if sym_meta.get(s,{}).get("sector")=="Energy")
print(f"US-Ene: {sorted(ene_syms)}")

w1a=BT_CACHE_W1_DIR.exists()  and any(BT_CACHE_W1_DIR.glob("*.parquet"))
m15a=BT_CACHE_M15_DIR.exists() and any(BT_CACHE_M15_DIR.glob("*.parquet"))

print("Precomputing + collecting trades...")
all_trades=[]
for sym in sorted(ene_syms):
    if sym not in bars: continue
    try:
        w1d =load_bt_bars_w1(sym)  if w1a  else pd.DataFrame()
        m15d=load_bt_bars_m15(sym) if m15a else pd.DataFrame()
        pc=precompute(bars[sym],m15d,w1d)
        t=collect_trades(pc,sym)
        all_trades.extend(t)
        print(f"  {sym}: {len(t)} trades")
    except Exception as e:
        print(f"  {sym} ERROR: {e}")

df=pd.DataFrame(all_trades).sort_values("entry_date").reset_index(drop=True)
df.index=range(1,len(df)+1)

# cumulative pnl
df["cum_pnl"]=df["pnl_usd"].cumsum().round(2)

print(f"\nTotal: {len(df)} trades across {df['sym'].nunique()} symbols")
print(f"PnL: ${df['pnl_usd'].sum():,.2f}")
print(f"WR:  {(df['r']>0).mean()*100:.1f}%")
print(f"Avg R: {df['r'].mean():.3f}")
print(f"Avg risk$/trade: ${df['risk_usd'].mean():.0f}")
print(f"Avg pnl$/trade:  ${df['pnl_usd'].mean():.0f}")

# Save full audit CSV
RESULTS_DIR.mkdir(parents=True,exist_ok=True)
ts=datetime.now().strftime("%Y%m%d_%H%M")
csv_path=RESULTS_DIR/f"v8_full_audit_{ts}.csv"
df.to_csv(csv_path)
print(f"Full audit CSV -> {csv_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PNG A — Corrected Statistics (leverage-explicit)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nGenerating stats PNG...")

wins  = df[df["r"]>0]; losses=df[df["r"]<0]
longs = df[df["direction"]=="LONG"]; shorts=df[df["direction"]=="SHORT"]
cum   = df["pnl_usd"].cumsum()
roll_max=cum.cummax(); dd=cum-roll_max
max_dd=dd.min()
total_pnl=df["pnl_usd"].sum()
sharpe_r = (df["r"].mean()/df["r"].std())*np.sqrt(len(df)/2) if df["r"].std()>0 else 0

fig=plt.figure(figsize=(18,11))
fig.patch.set_facecolor(BG)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,18); ax.set_ylim(0,11); ax.axis("off")

ax.text(9,10.6,"FULL TRADE AUDIT — Leverage-Explicit PnL",
        ha="center",va="center",fontsize=18,fontweight="bold",color=WHITE)
ax.text(9,10.15,
        f"US-Energy   LRC-80  sigma=1.0  HTF=0.5% score>=60  wing=2  "
        f"SL=1.0xATR  TP=1.0R   Capital=${CAPITAL:,.0f} x {LEVERAGE}x = ${EFFECTIVE:,.0f} effective",
        ha="center",va="center",fontsize=9.5,color=GRAY)

from matplotlib.patches import FancyBboxPatch
def card(ax_,x,y,w,h,val,lbl,sub,col):
    r=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.1",
                     linewidth=1.5,edgecolor=col+"66",facecolor=CARD)
    ax_.add_patch(r)
    ax_.text(x+w/2,y+h*0.68,val,ha="center",va="center",
             fontsize=14,fontweight="bold",color=col)
    ax_.text(x+w/2,y+h*0.35,lbl,ha="center",va="center",fontsize=8,color=GRAY)
    ax_.text(x+w/2,y+h*0.12,sub,ha="center",va="center",fontsize=7,color=GRAY,style="italic")

metrics=[
    (f"${total_pnl:,.0f}", "TOTAL PnL",     f"{total_pnl/CAPITAL*100:.0f}% on ${CAPITAL:.0f} capital",GREEN),
    (f"{(df['r']>0).mean()*100:.1f}%","WIN RATE",f"{len(df)} total trades",GREEN),
    (f"{sharpe_r:.2f}",  "SHARPE (R)",    "annualised, R-based",BLUE),
    (f"${max_dd:,.0f}", "MAX DRAWDOWN",   f"{max_dd/CAPITAL*100:.0f}% of capital",RED),
    (f"${df['risk_usd'].mean():.0f}","AVG RISK/TRADE",f"= {df['risk_usd'].mean()/EFFECTIVE*100:.1f}% of ${EFFECTIVE:.0f}",GOLD),
    (f"${df['pnl_usd'].mean():.0f}","AVG PnL/TRADE", f"{df['r'].mean():.3f}R avg",GREEN),
]
for i,(val,lbl,sub,col) in enumerate(metrics):
    card(ax,0.3+i*2.95,8.55,2.7,1.45,val,lbl,sub,col)

# Direction breakdown
ax.axhline(8.3,color=EDGE,linewidth=0.8,xmin=0.02,xmax=0.98)
ax.text(0.4,8.05,"DIRECTION",fontsize=9.5,fontweight="bold",color=WHITE)
cols_h=["","N","WR%","Avg R","Avg Risk$","Avg PnL$","Total PnL$"]
cx=[0.4,2.2,4.0,5.6,7.2,9.2,11.2]
for j,h_ in enumerate(cols_h):
    ax.text(cx[j],7.78,h_,fontsize=8.5,color=GRAY,fontweight="bold")
for row_i,(dname,sub_df,col) in enumerate([
        ("LONG",  longs, GREEN),
        ("SHORT", shorts,RED),
        ("ALL",   df,    WHITE)]):
    y_=7.43-row_i*0.43
    vals=[dname,
          str(len(sub_df)),
          f"{(sub_df['r']>0).mean()*100:.1f}%",
          f"{sub_df['r'].mean():.3f}R",
          f"${sub_df['risk_usd'].mean():.0f}",
          f"${sub_df['pnl_usd'].mean():.0f}",
          f"${sub_df['pnl_usd'].sum():,.0f}"]
    for j,v in enumerate(vals):
        ax.text(cx[j],y_,v,fontsize=9,color=col if j==0 else WHITE)

# Leverage explanation box
ax.axhline(6.3,color=EDGE,linewidth=0.8,xmin=0.02,xmax=0.98)
ax.text(0.4,6.05,"HOW 20x LEVERAGE AFFECTS EACH TRADE",fontsize=9.5,fontweight="bold",color=GOLD)
lines_=[
    f"Position size  =  ${EFFECTIVE:,.0f} effective  /  entry_px  =  n_shares (full leverage, every trade)",
    f"Dollar risk    =  n_shares  x  |entry_px - sl_px|   (sl = weekly_level +/- 1.0 x ATR)",
    f"Dollar PnL     =  R  x  dollar_risk   (R = how many SL-widths we gained or lost)",
    f"Example LONG:  entry=$90  sl=$87  ->  risk=$3/share  |  200 shares  ->  risk=$600  |  1R win = +$600",
    f"Avg risk/trade = USD{df['risk_usd'].mean():.0f}  ({df['risk_usd'].mean()/EFFECTIVE*100:.1f}% of USD{EFFECTIVE:.0f} effective).  "
    f"Range: USD{df['risk_usd'].min():.0f} - USD{df['risk_usd'].max():.0f}",
]
for i,ln in enumerate(lines_):
    ax.text(0.4,5.6-i*0.44,ln,fontsize=8.8,color=GRAY if i<4 else WHITE)

# By symbol
ax.axhline(3.5,color=EDGE,linewidth=0.8,xmin=0.02,xmax=0.98)
ax.text(0.4,3.25,"PER-SYMBOL BREAKDOWN",fontsize=9.5,fontweight="bold",color=WHITE)
sym_grp=df.groupby("sym").agg(
    n=("r","count"),
    wr=("r",lambda x:round((x>0).mean()*100,1)),
    avg_r=("r","mean"),
    total_pnl=("pnl_usd","sum"),
    avg_risk=("risk_usd","mean"),
).sort_values("total_pnl",ascending=False).reset_index()

sh=["Symbol","N","WR%","Avg R","Avg Risk$","Total PnL$"]
sx=[0.4,2.8,4.3,5.7,7.2,9.0]
for j,h_ in enumerate(sh):
    ax.text(sx[j],3.0,h_,fontsize=8.5,color=GRAY,fontweight="bold")
for ri,row in sym_grp.iterrows():
    y_=2.6-ri*0.4
    if y_<0.2: break
    col=GREEN if row["total_pnl"]>=0 else RED
    vals=[row["sym"].replace(".NYSE","").replace(".NAS",""),
          str(int(row["n"])),
          f"{row['wr']:.1f}%",
          f"{row['avg_r']:.3f}R",
          f"${row['avg_risk']:.0f}",
          f"${row['total_pnl']:,.0f}"]
    for j,v in enumerate(vals):
        ax.text(sx[j],y_,v,fontsize=8.8,color=col if j in [0,5] else WHITE)

plt.savefig(DESKTOP/"strategy_A_corrected_stats.png",
            bbox_inches="tight",facecolor=BG,dpi=150)
plt.close()
print("  -> strategy_A_corrected_stats.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PNG B — Trade Journal Pages (25 trades per page)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating trade journal PNGs...")

PAGE_SIZE=25
n_pages=math.ceil(len(df)/PAGE_SIZE)

for page in range(n_pages):
    chunk=df.iloc[page*PAGE_SIZE:(page+1)*PAGE_SIZE].copy()
    fig=plt.figure(figsize=(22,13))
    fig.patch.set_facecolor(BG)
    ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,22); ax.set_ylim(0,13); ax.axis("off")

    ax.text(11,12.6,f"TRADE JOURNAL  —  US-Energy  (page {page+1}/{n_pages})",
            ha="center",va="center",fontsize=14,fontweight="bold",color=WHITE)
    ax.text(11,12.18,
            "LRC-80 | sigma=1.0 | HTF 0.5% score>=60 | wing=2 | SL=1.0xATR | TP=1.0R  "
            f"|  ${CAPITAL:.0f} x {LEVERAGE}x = ${EFFECTIVE:,.0f} effective",
            ha="center",va="center",fontsize=8.5,color=GRAY)

    # Header
    COLS=["#","Date","Sym","Dir","Entry $","Level $","SL $","TP $","Exit $","Type",
          "R","Shares","Risk $","PnL $","CumPnL $"]
    XS  =[0.15,0.85,2.3,3.5,4.5,5.9,7.1,8.3,9.5,10.6,
          11.5,12.4,13.5,14.7,16.1]
    WID=0.85
    ax.axhline(11.85,color=EDGE,linewidth=1.2,xmin=0.005,xmax=0.99)
    for j,col in enumerate(COLS):
        ax.text(XS[j],11.65,col,fontsize=7.8,color=GRAY,fontweight="bold")
    ax.axhline(11.5,color=EDGE,linewidth=0.7,xmin=0.005,xmax=0.99)

    # Rows
    for row_i,(idx_,row) in enumerate(chunk.iterrows()):
        y_=11.15-row_i*0.43
        win=row["r"]>0
        row_col=GREEN if win else RED
        bg_alpha=0.06 if row_i%2==0 else 0.0
        if bg_alpha>0:
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((0.05,y_-0.16),21.9,0.38,
                                   facecolor=WHITE,alpha=0.04,zorder=1))

        d_col=GREEN if row["direction"]=="LONG" else RED
        vals=[
            str(idx_),
            row["entry_date"],
            row["sym"].replace(".NYSE","").replace(".NAS",""),
            row["direction"],
            f"{row['entry_px']:.2f}",
            f"{row['level_px']:.2f}",
            f"{row['sl_px']:.2f}",
            f"{row['tp_px']:.2f}",
            f"{row['exit_px']:.2f}",
            row["exit_type"],
            f"{row['r']:+.3f}",
            str(int(row["n_shares"])),
            f"${row['risk_usd']:.0f}",
            f"${row['pnl_usd']:+,.0f}",
            f"${row['cum_pnl']:+,.0f}",
        ]
        row_colors=[WHITE,WHITE,WHITE,d_col,WHITE,GOLD,RED,GREEN,
                    row_col,row_col,row_col,WHITE,GOLD,row_col,row_col]
        for j,(v,c_) in enumerate(zip(vals,row_colors)):
            ax.text(XS[j],y_,v,fontsize=8,color=c_,va="center",zorder=2)

        ax.axhline(y_-0.2,color=EDGE,linewidth=0.3,xmin=0.005,xmax=0.99,alpha=0.5)

    # Page footer
    pnl_chunk=chunk["pnl_usd"].sum()
    wr_chunk=(chunk["r"]>0).mean()*100
    ax.text(0.4,0.35,
            f"Page {page+1} totals:  {len(chunk)} trades  |  WR {wr_chunk:.1f}%  |  PnL ${pnl_chunk:+,.0f}  "
            f"|  Running total ${chunk['cum_pnl'].iloc[-1]:+,.0f}",
            fontsize=9,color=GRAY)

    fname=f"strategy_B{page+1:02d}_trade_journal_p{page+1}.png"
    plt.savefig(DESKTOP/fname,bbox_inches="tight",facecolor=BG,dpi=150)
    plt.close()
    print(f"  -> {fname}")


# ═══════════════════════════════════════════════════════════════════════════════
# PNG C — Equity Curve with leverage context
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating equity curve PNG...")

fig=plt.figure(figsize=(18,9))
fig.patch.set_facecolor(BG)
gs=GridSpec(2,2,figure=fig,hspace=0.45,wspace=0.32,
            top=0.92,bottom=0.08,left=0.06,right=0.97)
fig.suptitle(f"Equity Curve & Risk Profile  —  ${CAPITAL:.0f} x {LEVERAGE}x Leverage  (US-Energy)",
             fontsize=13,fontweight="bold",color=WHITE)

# Top-left: cumulative PnL
ax1=fig.add_subplot(gs[0,0:])
dates=pd.to_datetime(df["entry_date"])
cum_pnl=df["pnl_usd"].cumsum().values
roll_mx=np.maximum.accumulate(cum_pnl)
dd_arr =cum_pnl-roll_mx

ax1.plot(dates,cum_pnl,color=GREEN,linewidth=2,zorder=3)
ax1.fill_between(dates,0,cum_pnl,where=cum_pnl>=0,alpha=0.15,color=GREEN)
ax1.fill_between(dates,cum_pnl,roll_mx,where=dd_arr<0,alpha=0.35,color=RED,label="Drawdown")
ax1.axhline(0,color=GRAY,linewidth=0.7,linestyle="--")
ax1.axhline(CAPITAL,color=BLUE,linewidth=0.8,linestyle=":",label=f"Starting capital ${CAPITAL:.0f}")

ax1.set_title("Cumulative PnL (USD) — full 20x leveraged sizing",fontsize=11,color=WHITE)
from matplotlib.ticker import FuncFormatter
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"${x:,.0f}"))
ax1.legend(fontsize=8,framealpha=0.3)
ax1.grid(True,alpha=0.3)

# Annotate milestones
final=cum_pnl[-1]
ax1.annotate(f"  Final: ${final:+,.0f}\n  ({final/CAPITAL*100:+.0f}% on capital)",
             xy=(dates.iloc[-1],final),color=GREEN,fontsize=9,fontweight="bold")
ax1.annotate(f"  Max DD: ${dd_arr.min():,.0f}",
             xy=(dates[dd_arr.argmin()],cum_pnl[dd_arr.argmin()]),
             color=RED,fontsize=8.5)

# Bottom-left: Risk per trade bar
ax2=fig.add_subplot(gs[1,0])
ax2.bar(range(len(df)),df["risk_usd"],
        color=[GREEN if d=="LONG" else RED for d in df["direction"]],
        width=0.8,alpha=0.7)
ax2.axhline(df["risk_usd"].mean(),color=GOLD,linewidth=1.2,
            linestyle="--",label=f"Avg ${df['risk_usd'].mean():.0f}")
ax2.set_title("Dollar Risk per Trade",fontsize=10,color=WHITE)
ax2.set_xlabel("Trade #",color=GRAY)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f"${x:,.0f}"))
ax2.legend(fontsize=8,framealpha=0.3)
ax2.grid(True,alpha=0.3,axis="y")

# Bottom-right: R distribution histogram
ax3=fig.add_subplot(gs[1,1])
r_vals=df["r"].values
bins=np.linspace(-1.5,3.5,30)
ax3.hist(r_vals[r_vals>0],bins=bins,color=GREEN,alpha=0.7,label="Wins")
ax3.hist(r_vals[r_vals<0],bins=bins,color=RED,  alpha=0.7,label="Losses")
ax3.axvline(r_vals.mean(),color=GOLD,linewidth=1.5,linestyle="--",
            label=f"Avg {r_vals.mean():.3f}R")
ax3.axvline(0,color=GRAY,linewidth=0.8)
ax3.set_title("R Distribution",fontsize=10,color=WHITE)
ax3.set_xlabel("R",color=GRAY)
ax3.legend(fontsize=8,framealpha=0.3)
ax3.grid(True,alpha=0.3,axis="y")

plt.savefig(DESKTOP/"strategy_C_equity_risk.png",
            bbox_inches="tight",facecolor=BG,dpi=150)
plt.close()
print("  -> strategy_C_equity_risk.png")

print("\nDone. All PNGs on Desktop.")
print(f"\nFull trade list also at: {csv_path}")
