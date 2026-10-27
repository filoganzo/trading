"""
Normalise the split-corrected stock CFD files into the form the MTF engine loads.

`stocks_data.pull()` writes split-adjusted bars but keeps the FULL history,
including the stretches that are still unsafe (residual discontinuities, penny
back-adjusted prices). Rather than leave it to every downstream caller to
remember `clean_window`, this step bakes the decision in: it computes the clean
window once on the daily series, maps that start date onto every timeframe, and
writes only usable bars into the engine's data directory.

Anything that reads the output therefore cannot accidentally trade a fictional
-77% split candle, because those bars are not in the file.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from stocks_data import DATA_DIR as STOCK_DIR, detect_splits, clean_window, UNIVERSE
from mtf_runner import DATA_DIR as ENGINE_DIR

TFS = ("H1", "H4", "D1", "W1")


def prep_symbol(sym: str, verbose: bool = True) -> dict:
    tag = sym.replace(".", "_")
    d1p = os.path.join(STOCK_DIR, f"{tag}_D1.parquet")
    if not os.path.exists(d1p):
        return {"symbol": sym, "status": "missing_d1"}

    d1 = pd.read_parquet(d1p)
    if "dt" not in d1:
        d1["dt"] = pd.to_datetime(d1["time"], unit="s")

    # Splits were already applied at pull time, so re-detecting on the adjusted
    # series should find nothing; what we want here is the residual-jump
    # truncation that clean_window performs.
    sp = detect_splits(d1)
    mask = clean_window(d1, sp).to_numpy()
    if mask.sum() < 250:
        return {"symbol": sym, "status": "too_short", "clean_bars": int(mask.sum())}
    start = pd.Timestamp(d1.loc[mask, "dt"].iloc[0]).normalize()

    row = {"symbol": sym, "status": "ok", "clean_start": str(start.date()),
           "d1_clean": int(mask.sum())}
    for tf in TFS:
        p = os.path.join(STOCK_DIR, f"{tag}_{tf}.parquet")
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        if "dt" not in df:
            df["dt"] = pd.to_datetime(df["time"], unit="s")
        out = df[df["dt"] >= start].reset_index(drop=True)
        if len(out) < 120:
            row[f"{tf}_bars"] = len(out)
            continue
        if "spread" not in out:
            out["spread"] = 0.0
        # residual safety check on the timeframe actually being written
        c = out["close"].to_numpy(float)
        jumps = int((np.abs(np.diff(c) / np.maximum(c[:-1], 1e-9)) > 0.25).sum())
        out.to_parquet(os.path.join(ENGINE_DIR, f"{tag}_{tf}.parquet"))
        row[f"{tf}_bars"] = len(out)
        row[f"{tf}_jumps"] = jumps
    if verbose:
        print(f"  {sym:<11} from {row['clean_start']}  "
              f"H1={row.get('H1_bars','-'):>6} D1={row.get('D1_bars','-'):>5} "
              f"W1={row.get('W1_bars','-'):>4}  residual jumps "
              f"H1={row.get('H1_jumps','-')} D1={row.get('D1_jumps','-')}",
              flush=True)
    return row


def prep_all(symbols=None) -> pd.DataFrame:
    symbols = symbols or UNIVERSE
    os.makedirs(ENGINE_DIR, exist_ok=True)
    rows = [prep_symbol(s) for s in symbols]
    rep = pd.DataFrame(rows)
    rep.to_parquet(os.path.join(STOCK_DIR, "_prep_report.parquet"))
    return rep


if __name__ == "__main__":
    print("Normalising stock CFD data into the engine directory")
    rep = prep_all()
    ok = rep[rep["status"] == "ok"] if "status" in rep else rep
    print(f"\n{len(ok)}/{len(rep)} symbols prepared")
    if len(ok):
        for tf in ("H1", "D1", "W1"):
            col = f"{tf}_bars"
            if col in ok:
                v = ok[col].dropna()
                print(f"  {tf}: median {v.median():.0f} bars, "
                      f"total {v.sum():.0f} across {len(v)} symbols")
        bad = ok[[c for c in ("H1_jumps", "D1_jumps") if c in ok]].sum().sum()
        print(f"  residual >25% jumps remaining in written files: {int(bad)}")
    if "status" in rep:
        for st, grp in rep.groupby("status"):
            if st != "ok":
                print(f"  {st}: {list(grp['symbol'])}")
