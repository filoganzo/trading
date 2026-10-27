"""Parse today's MT5 terminal log → reconstruct trades → match vs M1 bars.

Strategy:
  - deals carry authoritative (time, side, lots, price) and arrive in order
  - walk them chronologically; track running net position
  - when net goes 0 -> nonzero = new OPEN
  - when net returns to 0 = CLOSE of current position
  - pnl = sum(signed_lot * price) reversed (entry outflow − exit inflow)
"""
from __future__ import annotations
import re, sys, io, json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

LOG  = Path("C:/Users/<user>/AppData/Roaming/MetaQuotes/Terminal/<TERMINAL_ID>/logs/20260413.log")
BARS = Path("path/to/repo/hft/data/XAUUSD/bars_M1_20260201_20260413.parquet")
DATE = datetime(2026, 4, 13)
PIP  = 0.10     # XAUUSD: 1 pip = $0.10

raw = LOG.read_text(encoding="utf-16-le", errors="replace")
lines = raw.splitlines()

# deal #ID {buy|sell} LOTS XAUUSD at PRICE done (based on order #ORD)
deal_re = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})\.(\d{3}).*?Trades\s+'(\d+)':\s+deal\s+#(\d+)\s+"
    r"(buy|sell)\s+([\d.]+)\s+XAUUSD\s+at\s+([\d.]+)\s+done\s+\(based on order\s+#(\d+)\)"
)

deals = []   # list of dicts
for L in lines:
    m = deal_re.search(L)
    if not m:
        continue
    h, mi, s, ms, acc, did, side, lots, px, ord_id = m.groups()
    ts = DATE.replace(hour=int(h), minute=int(mi), second=int(s), microsecond=int(ms) * 1000)
    deals.append(dict(
        ts=ts, deal_id=int(did), side=side, lots=float(lots),
        price=float(px), order_id=int(ord_id),
    ))

deals.sort(key=lambda d: (d["ts"], d["deal_id"]))
print(f"parsed {len(deals)} deals")

# Walk deals; pair flat→position→flat round trips
trades = []
pos_lots = 0.0         # signed lots (+long, -short)
avg_entry = 0.0
entry_ts = None
entry_side = None

for d in deals:
    signed = d["lots"] if d["side"] == "buy" else -d["lots"]
    new_pos = round(pos_lots + signed, 4)
    if abs(pos_lots) < 1e-9:
        # flat → open
        pos_lots  = new_pos
        avg_entry = d["price"]
        entry_ts  = d["ts"]
        entry_side = d["side"]
    elif abs(new_pos) < 1e-9:
        # closing back to flat
        exit_px = d["price"]
        dir_sign = 1 if entry_side == "buy" else -1
        pnl_pips = (exit_px - avg_entry) / PIP * dir_sign
        pnl_usd  = (exit_px - avg_entry) * dir_sign * abs(pos_lots) * 100.0  # XAUUSD: 1 lot * $1 move = $100
        dur_s = (d["ts"] - entry_ts).total_seconds()
        trades.append(dict(
            entry_ts=entry_ts, exit_ts=d["ts"], dur_s=dur_s,
            side=entry_side, lots=abs(pos_lots),
            entry=avg_entry, exit=exit_px,
            pnl_pips=pnl_pips, pnl_usd=pnl_usd,
        ))
        pos_lots = 0.0
        avg_entry = 0.0
        entry_ts = None
        entry_side = None
    else:
        # partial scale (unusual for this EA)
        pos_lots = new_pos

df = pd.DataFrame(trades)
print(f"\nreconstructed {len(df)} round-trip trades")
if len(df) == 0:
    sys.exit(0)

df["win"] = df.pnl_usd > 0
print(f"winners: {df.win.sum()}  losers: {(~df.win).sum()}  winrate: {df.win.mean():.1%}")
print(f"net pnl: ${df.pnl_usd.sum():.2f}")
print(f"avg win: ${df[df.win].pnl_usd.mean():.2f}    avg loss: ${df[~df.win].pnl_usd.mean():.2f}")
print(f"median dur: {df.dur_s.median():.0f}s    p90 dur: {df.dur_s.quantile(.9):.0f}s")
print(f"biggest winner: ${df.pnl_usd.max():.2f}    biggest loser: ${df.pnl_usd.min():.2f}")

# filter to the LIVE bot trades — they began when trading_enabled=yes and lot was 0.1 or 1.0
# drop the 0.01 manual testers at the top
live = df[df.lots >= 0.05].copy()
print(f"\nLIVE bot trades (lots>=0.05): {len(live)}")
print(f"  winrate: {live.win.mean():.1%}   net: ${live.pnl_usd.sum():.2f}")
print(f"  by hour:")
live["hour"] = live.entry_ts.dt.hour
print(live.groupby("hour").agg(n=("pnl_usd","size"), wr=("win","mean"), net=("pnl_usd","sum")).round(2))

print("\nby side:")
print(live.groupby("side").agg(n=("pnl_usd","size"), wr=("win","mean"), net=("pnl_usd","sum"), avg=("pnl_usd","mean")).round(2))

print("\nby lot size:")
print(live.groupby("lots").agg(n=("pnl_usd","size"), wr=("win","mean"), net=("pnl_usd","sum")).round(2))

# Pull today's M1 bars directly from MT5 (parquet cache stops at Apr 10)
import MetaTrader5 as mt5
if not mt5.initialize():
    print("mt5 init failed:", mt5.last_error()); sys.exit(1)
from datetime import timedelta
bars_raw = mt5.copy_rates_range(
    "XAUUSD", mt5.TIMEFRAME_M1,
    DATE, DATE + timedelta(days=1),
)
mt5.shutdown()
bars = pd.DataFrame(bars_raw)
bars["ts"] = pd.to_datetime(bars["time"], unit="s", utc=True).dt.tz_localize(None)
bars = bars.set_index("ts").sort_index()
print(f"\nM1 bars today: {len(bars)}  range: {bars.index.min()} → {bars.index.max()}")

def bar_at(ts, offset_min=0):
    target = ts.replace(second=0, microsecond=0) + pd.Timedelta(minutes=offset_min)
    if target in bars.index: return bars.loc[target]
    return None

# Context features per trade: prior-M1 adverse move, entry-bar range, next-bar range
def enrich(row):
    t_open = pd.Timestamp(row.entry_ts).replace(second=0, microsecond=0)
    prior = bar_at(row.entry_ts, -1)
    entry_bar = bar_at(row.entry_ts, 0)
    next_bar  = bar_at(row.entry_ts, 1)
    out = {}
    if prior is not None:
        body_pips = (prior.close - prior.open) / PIP
        out["adv_prev_pips"] = -body_pips if row.side == "buy" else body_pips
        out["prior_range_pips"] = (prior.high - prior.low) / PIP
    if entry_bar is not None:
        out["entry_bar_range_pips"] = (entry_bar.high - entry_bar.low) / PIP
    if next_bar is not None:
        out["next_bar_range_pips"] = (next_bar.high - next_bar.low) / PIP
    return pd.Series(out)

live = pd.concat([live.reset_index(drop=True), live.apply(enrich, axis=1).reset_index(drop=True)], axis=1)

print("\nAdverse move on prior M1 (pips AGAINST entry dir):")
print(live.groupby("win")["adv_prev_pips"].describe().round(1))

print("\nEntry-bar range pips:")
print(live.groupby("win")["entry_bar_range_pips"].describe().round(1))

print("\nTop 10 losers (biggest $ loss):")
cols = ["entry_ts","side","lots","entry","exit","dur_s","pnl_usd","adv_prev_pips","entry_bar_range_pips","next_bar_range_pips"]
print(live.sort_values("pnl_usd").head(10)[cols].to_string(index=False))

print("\nTop 10 winners:")
print(live.sort_values("pnl_usd", ascending=False).head(10)[cols].to_string(index=False))

# save
out = Path("path/to/repo/hft/artifacts/trades_20260413.csv")
out.parent.mkdir(exist_ok=True, parents=True)
live.to_csv(out, index=False)
print(f"\nsaved: {out}")
