"""
Knife-veto, take 2: slippage-exposure framing.

The backtest can't model slippage directly (it assumes perfect fills at SL),
so instead we measure a proxy: how volatile is the minute the trade ENTERS
and the minute AFTER? A trade living inside a 40-pip M1 range is exposed
to much worse stop-exit slippage than one living in a 5-pip range.

For each trade bucket (blocked/kept at the knife threshold), we report:
  - distribution of entry-minute range (pips)
  - distribution of entry+1-minute range (pips)
  - share of trades where either range > 20 pips (the nominal SL distance)
  - share where either range > 30 pips (catastrophic)
"""
import pandas as pd
import numpy as np
from pathlib import Path

PIP = 0.10
ROOT = Path(r"path/to/repo/hft")
WINDOWS = ["feb_h1", "feb_h2", "mar_h1", "mar_h2", "apr"]

bars = pd.read_parquet(ROOT / "data/XAUUSD/bars_M1_20260201_20260413.parquet")
bars = bars.sort_values("ts").reset_index(drop=True)
bars["bar_min"] = bars["ts"].dt.floor("min")
bars["range_pips"] = (bars["high"] - bars["low"]) / PIP
bars["body_signed_pips"] = (bars["close"] - bars["open"]) / PIP
bars_idx = bars.set_index("bar_min")

def bar_feat(ts, offset, col):
    try:
        bar_ts = ts.floor("min") + pd.Timedelta(minutes=offset)
        return bars_idx.loc[bar_ts, col]
    except KeyError:
        return np.nan

rows = []
for win in WINDOWS:
    tpath = ROOT / f"artifacts/bt_combo/{win}/trades.parquet"
    if not tpath.exists():
        continue
    t = pd.read_parquet(tpath)
    t["window"] = win
    # signed move over the last completed M1 bar (body of k-1)
    t["prev_body"] = t["ts_open"].apply(lambda x: bar_feat(x, -1, "body_signed_pips"))
    # adverse = how much the body moved AGAINST our entry direction
    t["adv_prev"]  = np.where(t["direction"] == "long", -t["prev_body"], t["prev_body"])
    # ranges at and right after entry (proxy for slippage exposure during SL exit)
    t["range_k0"]  = t["ts_open"].apply(lambda x: bar_feat(x,  0, "range_pips"))
    t["range_k1"]  = t["ts_open"].apply(lambda x: bar_feat(x, +1, "range_pips"))
    t["range_k2"]  = t["ts_open"].apply(lambda x: bar_feat(x, +2, "range_pips"))
    rows.append(t)

all_t = pd.concat(rows, ignore_index=True)
all_t = all_t.dropna(subset=["adv_prev", "range_k0", "range_k1", "range_k2"])
print(f"loaded {len(all_t)} trades with full bar context")
print()

def describe(x, label):
    return (f"{label:>12}  n={len(x):>5}  "
            f"mean={x.mean():>5.1f}  p50={np.percentile(x,50):>5.1f}  "
            f"p90={np.percentile(x,90):>5.1f}  p99={np.percentile(x,99):>5.1f}  "
            f"max={x.max():>6.1f}")

print("=== full-book baseline ranges (pips) ===")
for c in ("range_k0", "range_k1", "range_k2"):
    print(describe(all_t[c], c))
print()

MAX_RANGE_NEXT = all_t[["range_k0","range_k1","range_k2"]].max(axis=1)

for thr in (8, 10, 12, 15, 20):
    blocked = all_t[all_t["adv_prev"] >= thr]
    kept    = all_t[all_t["adv_prev"] <  thr]
    if len(blocked) < 30: continue
    print(f"=== adv_prev >= {thr} pips  (blocked={len(blocked)}  kept={len(kept)}) ===")
    for c in ("range_k0", "range_k1", "range_k2"):
        print("  " + describe(blocked[c], f"BLK_{c}"))
        print("  " + describe(kept[c],    f"KEP_{c}"))
    # tail exposure: fraction of trades with ANY of k0..k2 > X pips
    blk_max = blocked[["range_k0","range_k1","range_k2"]].max(axis=1)
    kep_max = kept   [["range_k0","range_k1","range_k2"]].max(axis=1)
    for cut in (20, 30, 40):
        b = 100 * (blk_max > cut).mean()
        k = 100 * (kep_max > cut).mean()
        lift = b / k if k > 0 else float("inf")
        print(f"  P(max range in first 3 bars > {cut} pips):  "
              f"blocked={b:5.1f}%   kept={k:5.1f}%   lift x{lift:.2f}")
    # loser-only tail
    losers_blk = blocked[blocked["usd"] < 0]
    losers_kep = kept   [kept   ["usd"] < 0]
    if len(losers_blk) > 10 and len(losers_kep) > 10:
        print(f"  losers-only avg usd: blocked={losers_blk['usd'].mean():.2f}  "
              f"kept={losers_kep['usd'].mean():.2f}  "
              f"worst: blocked={losers_blk['usd'].min():.2f}  "
              f"kept={losers_kep['usd'].min():.2f}")
    print()
