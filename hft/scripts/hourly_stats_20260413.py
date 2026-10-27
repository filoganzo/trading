"""Hour-by-hour breakdown of today's live FiloHFT book."""
import sys, io
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_csv("artifacts/trades_20260413.csv")
df["entry_ts"] = pd.to_datetime(df["entry_ts"])
df["exit_ts"]  = pd.to_datetime(df["exit_ts"])
df = df.sort_values("entry_ts").reset_index(drop=True)
df["hour"] = df.entry_ts.dt.hour
df["minute"] = df.entry_ts.dt.minute
df["hm"] = df.entry_ts.dt.strftime("%H:%M")
df["win"] = df.pnl_usd > 0
df["r_unit"] = df.pnl_usd / (df.lots * 10 * 20)  # normalized to 20-pip stop units

# Config change timeline (approx, from log boot times + session recall)
TIMELINE = [
    ("16:29", "live enabled, 0.1 lot, counter-trend B+C gates shipped"),
    ("17:05", "re-enabled live after break"),
    ("17:14", "knife size-down @ 40p threshold shipped"),
    ("18:14", "D1 clearance gate shipped"),
    ("21:04", "1-lot bump"),
]

def block(title):
    print(f"\n{'='*62}\n{title}\n{'='*62}")

block("OVERALL BOOK (live trades ≥0.1 lot)")
live = df[df.lots >= 0.05].copy()
n  = len(live); w = live.win.sum(); l = n - w
print(f"trades: {n}    wins: {w}    losses: {l}    winrate: {w/n:.1%}")
print(f"net pnl: ${live.pnl_usd.sum():+.2f}")
print(f"avg win: ${live[live.win].pnl_usd.mean():+.2f}   avg loss: ${live[~live.win].pnl_usd.mean():+.2f}")
print(f"best: ${live.pnl_usd.max():+.2f}    worst: ${live.pnl_usd.min():+.2f}")
print(f"avg pips: wins {live[live.win].pnl_pips.mean():+.1f}    losses {live[~live.win].pnl_pips.mean():+.1f}")
print(f"median duration: {live.dur_s.median():.0f}s    p90: {live.dur_s.quantile(.9):.0f}s")

block("BY HOUR")
by_h = live.groupby("hour").agg(
    n=("pnl_usd","size"),
    wins=("win","sum"),
    wr=("win","mean"),
    net_usd=("pnl_usd","sum"),
    avg_win_pips=("pnl_pips", lambda s: s[s>0].mean() if (s>0).any() else np.nan),
    avg_loss_pips=("pnl_pips", lambda s: s[s<=0].mean() if (s<=0).any() else np.nan),
    median_dur_s=("dur_s","median"),
).round(2)
print(by_h.to_string())

block("BY 30-MIN BUCKET")
live["bucket"] = live.entry_ts.dt.floor("30min").dt.strftime("%H:%M")
by_b = live.groupby("bucket").agg(
    n=("pnl_usd","size"),
    wr=("win","mean"),
    net=("pnl_usd","sum"),
    avg_win_pips=("pnl_pips", lambda s: s[s>0].mean() if (s>0).any() else np.nan),
    avg_loss_pips=("pnl_pips", lambda s: s[s<=0].mean() if (s<=0).any() else np.nan),
).round(2)
print(by_b.to_string())

block("BY LOT SIZE (config regime proxy)")
by_l = live.groupby("lots").agg(
    n=("pnl_usd","size"),
    wr=("win","mean"),
    net_usd=("pnl_usd","sum"),
    avg_win_pips=("pnl_pips", lambda s: s[s>0].mean() if (s>0).any() else np.nan),
    avg_loss_pips=("pnl_pips", lambda s: s[s<=0].mean() if (s<=0).any() else np.nan),
    n_wins_ge_20p=("pnl_pips", lambda s: int((s>=20).sum())),
    n_loss_through_20p=("pnl_pips", lambda s: int((s<=-20).sum())),
).round(2)
print(by_l.to_string())

block("BY SIDE")
by_s = live.groupby("side").agg(
    n=("pnl_usd","size"),
    wr=("win","mean"),
    net=("pnl_usd","sum"),
    avg_pips=("pnl_pips","mean"),
).round(2)
print(by_s.to_string())

block("CONFIG REGIME — BEFORE vs AFTER 21:04 (1-lot era)")
cutoff = pd.Timestamp("2026-04-13 21:04:00")
pre  = live[live.entry_ts <  cutoff]
post = live[live.entry_ts >= cutoff]
for label, sub in [("pre 21:04 (0.1 lot era, partial features)", pre),
                   ("post 21:04 (1-lot + all gates)", post)]:
    if len(sub) == 0: print(f"{label}: none"); continue
    w = sub.win.sum(); n = len(sub)
    print(f"{label}:")
    print(f"  n={n}  wr={w/n:.1%}  net=${sub.pnl_usd.sum():+.2f}")
    print(f"  avg_win={sub[sub.win].pnl_pips.mean():+.1f}p  avg_loss={sub[~sub.win].pnl_pips.mean():+.1f}p")
    print(f"  wins ≥20p: {(sub.pnl_pips>=20).sum()}    losses through -20p: {(sub.pnl_pips<=-20).sum()}")

block("WIN-SIZE DISTRIBUTION (pips)")
for label, sub in [("all live", live), ("post 21:04", post)]:
    if len(sub)==0: continue
    w = sub[sub.win]
    bins = [(0,5),(5,8),(8,12),(12,16),(16,20),(20,30),(30,100)]
    print(f"{label}  (n={len(sub)}, wins={len(w)}):")
    for lo,hi in bins:
        c = ((w.pnl_pips>=lo)&(w.pnl_pips<hi)).sum()
        bar = '█'*c
        print(f"  {lo:>2}-{hi:<3}p: {c:>3}  {bar}")

block("LOSS-SIZE DISTRIBUTION (pips)")
for label, sub in [("all live", live), ("post 21:04", post)]:
    if len(sub)==0: continue
    l = sub[~sub.win]
    bins = [(-5,0),(-10,-5),(-15,-10),(-20,-15),(-25,-20),(-100,-25)]
    print(f"{label}  (n={len(sub)}, losses={len(l)}):")
    for lo,hi in bins:
        c = ((l.pnl_pips>=lo)&(l.pnl_pips<hi)).sum()
        bar = '█'*c
        print(f"  {lo:>4} to {hi:<4}p: {c:>3}  {bar}")

block("SAR DETECTOR (prev loser, gap<10s, side flipped)")
df2 = live.copy()
df2["prev_exit"] = df2.exit_ts.shift(1)
df2["prev_side"] = df2.side.shift(1)
df2["prev_pnl_p"] = df2.pnl_pips.shift(1)
df2["gap_s"]    = (df2.entry_ts - df2.prev_exit).dt.total_seconds()
df2["flipped"]  = df2.side != df2.prev_side
sar = df2[(df2.prev_pnl_p <= 0) & (df2.gap_s < 10) & (df2.flipped == True)]
print(f"SAR events: {len(sar)} of {len(live)} ({100*len(sar)/max(1,len(live)):.1f}%)")
print(f"  SAR net: ${sar.pnl_usd.sum():+.2f}")
print(f"  SAR wr:  {sar.win.mean():.1%}")
print(f"  SAR sub-1s gaps: {(sar.gap_s<1).sum()}")
# post-21:04 SAR
sar_post = sar[sar.entry_ts >= cutoff]
print(f"  SAR post 21:04: {len(sar_post)}    net: ${sar_post.pnl_usd.sum():+.2f}")

block("WINRATE TREND (consecutive 15-trade rolling)")
live2 = live.copy().reset_index(drop=True)
live2["roll_wr"]  = live2.win.rolling(15, min_periods=5).mean()
live2["roll_pnl"] = live2.pnl_usd.rolling(15, min_periods=5).sum()
print(f"{'idx':>4} {'time':>8} {'side':>4} {'pnl':>8} {'roll_wr':>8} {'roll_pnl':>10}")
for i in range(0, len(live2), 20):
    r = live2.iloc[i]
    wr_s = f"{r.roll_wr:.1%}" if pd.notna(r.roll_wr) else "   -"
    rp = f"${r.roll_pnl:+.0f}" if pd.notna(r.roll_pnl) else "   -"
    print(f"{i:>4} {r.entry_ts.strftime('%H:%M:%S'):>8} {r.side:>4} {r.pnl_usd:>+8.1f} {wr_s:>8} {rp:>10}")
# also last
r = live2.iloc[-1]
wr_s = f"{r.roll_wr:.1%}" if pd.notna(r.roll_wr) else "   -"
rp = f"${r.roll_pnl:+.0f}" if pd.notna(r.roll_pnl) else "   -"
print(f"{len(live2)-1:>4} {r.entry_ts.strftime('%H:%M:%S'):>8} {r.side:>4} {r.pnl_usd:>+8.1f} {wr_s:>8} {rp:>10}")

block("TIMELINE OF CONFIG CHANGES (reference)")
for t, what in TIMELINE:
    n_before = (live.entry_ts < pd.Timestamp(f"2026-04-13 {t}:00")).sum()
    print(f"  {t}  trades-before={n_before:>3}   {what}")

block("BLOCKED SETUPS (telemetry CSVs)")
print("no setups.csv / intents.csv found in Common/Files/FiloHFT/")
print("cannot quantify blocked vs accepted counts — telemetry CSVs are not being written")
print("(known open issue, same root cause as earlier 'only filo_hft.log exists')")
