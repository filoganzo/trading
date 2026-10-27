"""
Knife-veto backtest analysis.

For each trade in artifacts/bt_combo/<window>/trades.parquet, compute the
signed move (in pips) over the prior 1/2/3 M1 bars ending at the bar
BEFORE entry. Sweep thresholds and report whether a knife-veto filter
would kill losers harder than winners.

Pip convention: XAUUSD with pip_in_points=10, point=0.01 -> 1 pip = 0.10.
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
bars_idx = bars.set_index("bar_min")

def move_pips_prior(entry_ts, n_bars):
    """Signed move over the n_bars M1 bars just before entry, in pips."""
    bar_k = entry_ts.floor("min")
    # close of the bar that ended just before entry's minute:
    end_bar  = bar_k - pd.Timedelta(minutes=1)
    start_bar = end_bar - pd.Timedelta(minutes=n_bars - 1)
    try:
        c_end   = bars_idx.loc[end_bar,   "close"]
        o_start = bars_idx.loc[start_bar, "open"]
    except KeyError:
        return np.nan
    return (c_end - o_start) / PIP

rows = []
for win in WINDOWS:
    tpath = ROOT / f"artifacts/bt_combo/{win}/trades.parquet"
    if not tpath.exists():
        continue
    t = pd.read_parquet(tpath)
    t["window"] = win
    for n in (1, 2, 3):
        t[f"move_{n}m"] = t["ts_open"].apply(lambda x: move_pips_prior(x, n))
    rows.append(t)

all_t = pd.concat(rows, ignore_index=True)
print(f"loaded {len(all_t)} trades across {len(rows)} windows")
print(f"baseline: winrate={100*(all_t['usd']>0).mean():.1f}%  "
      f"avg_usd={all_t['usd'].mean():.3f}  "
      f"total_usd={all_t['usd'].sum():.0f}")
print()

# for each trade, compute adverse_move = +move for SHORT, -move for LONG.
# positive adverse_move means "the thing just ran AGAINST our entry direction"
def adverse(row, col):
    m = row[col]
    if pd.isna(m): return np.nan
    return -m if row["direction"] == "long" else m

for n_bars in (1, 2, 3):
    col = f"move_{n_bars}m"
    all_t[f"adv_{n_bars}m"] = all_t.apply(lambda r: adverse(r, col), axis=1)

print(f"{'window':>6} {'thr':>4} {'blocked':>8} {'block%':>7}  "
      f"{'blk_wr':>7} {'blk_avg':>8} {'blk_sum':>9}  "
      f"{'kept_wr':>7} {'kept_avg':>8} {'kept_sum':>9}")

for n_bars in (1, 2, 3):
    adv_col = f"adv_{n_bars}m"
    valid = all_t.dropna(subset=[adv_col])
    print(f"\n--- window = {n_bars} M1 bar(s) before entry  (n={len(valid)}) ---")
    for thr in (6, 8, 10, 12, 15, 20):
        blocked = valid[valid[adv_col] >= thr]
        kept    = valid[valid[adv_col] <  thr]
        if len(blocked) == 0:
            continue
        b_wr  = 100 * (blocked["usd"] > 0).mean()
        b_avg = blocked["usd"].mean()
        b_sum = blocked["usd"].sum()
        k_wr  = 100 * (kept["usd"] > 0).mean()
        k_avg = kept["usd"].mean()
        k_sum = kept["usd"].sum()
        print(f"{n_bars:>6} {thr:>4} {len(blocked):>8} {100*len(blocked)/len(valid):>6.1f}%  "
              f"{b_wr:>6.1f}% {b_avg:>8.3f} {b_sum:>9.1f}  "
              f"{k_wr:>6.1f}% {k_avg:>8.3f} {k_sum:>9.1f}")

# diagnostic: how does blocked PnL compare to kept PnL at the candidate thresholds?
print("\n=== summary: does blocked set underperform kept set? ===")
for n_bars in (1, 2, 3):
    adv_col = f"adv_{n_bars}m"
    valid = all_t.dropna(subset=[adv_col])
    for thr in (8, 10, 12, 15):
        blocked = valid[valid[adv_col] >= thr]
        kept    = valid[valid[adv_col] <  thr]
        if len(blocked) < 20: continue
        delta_avg = blocked["usd"].mean() - kept["usd"].mean()
        print(f"  {n_bars}m / thr={thr:>2}:  "
              f"blk_avg-kept_avg = {delta_avg:+.3f} usd/trade  "
              f"(block {len(blocked)}/{len(valid)}, "
              f"keep_total_delta={-blocked['usd'].sum():+.0f})")
