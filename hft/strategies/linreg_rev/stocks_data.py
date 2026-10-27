"""
Stock CFD data acquisition with split-artifact correction.

Why this file exists
--------------------
IC Markets' stock CFD history is only partially split-adjusted. AAPL.NAS shows
-74.1% on 2020-08-31 (the 4:1 split) and TSLA.NAS -77.5% on the same date (5:1),
while their pre-2020 history IS back-adjusted. An uncorrected split is not a
price move: it is a step discontinuity that pins every band and oscillator to an
extreme and manufactures signals that never existed. Running the strategy on raw
data would produce impressive, entirely fictional results.

Separating a split from a real crash
------------------------------------
Both look like a large negative return. Two things tell them apart:

  ratio     a split moves price by an almost exact factor (2, 3, 4, 5, 7, 10,
            or 3/2, 5/4 ...). A crash lands on an arbitrary number.
  range     on a split date the stock trades NORMALLY -- the day's high/low
            range relative to its own close is ordinary. On a crash date the
            intraday range explodes.

AAPL on 2000-09-29 fell 51.9%, which is close enough to a 2:1 ratio to fool the
ratio test on its own; its intraday range was enormous, so the range test keeps
it. That case is the reason both tests are required.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "stocks")

# Auto-correctable split factors. Deliberately restricted to >= 2:1.
#
# Small factors (5/4, 4/3, 3/2) are indistinguishable from ordinary 20-33% gap
# days. Including them made the detector "correct" NVDA's +24.6% AI-guidance
# session of 2023-05-25 and three TSLA earnings gaps -- destroying real price
# action. Genuine sub-2:1 splits are rare; missing one costs a truncated
# history, whereas falsely correcting a real move corrupts the series silently.
SPLIT_FACTORS = [2, 3, 4, 5, 6, 7, 8, 10, 15, 20]

# Sector-diversified US large caps. Chosen for depth of history and liquidity,
# not for performance -- picking winners after the fact would be survivorship
# bias baked straight into the universe.
UNIVERSE = [
    # mega-cap tech
    "AAPL.NAS", "MSFT.NAS", "GOOG.NAS", "AMZN.NAS", "META.NAS", "NVDA.NAS",
    "INTC.NAS", "CSCO.NAS", "ADBE.NAS", "ORCL.NYSE", "IBM.NYSE", "TXN.NAS",
    # financials
    "JPM.NYSE", "BAC.NYSE", "WFC.NYSE", "GS.NYSE", "MS.NYSE", "AXP.NYSE",
    # healthcare
    "JNJ.NYSE", "PFE.NYSE", "MRK.NYSE", "ABT.NYSE", "UNH.NYSE", "AMGN.NAS",
    # consumer
    "KO.NYSE", "PEP.NAS", "PG.NYSE", "WMT.NYSE", "MCD.NYSE", "NKE.NYSE",
    "COST.NAS", "SBUX.NAS", "HD.NYSE", "TGT.NYSE",
    # industrial / energy / materials
    "CAT.NYSE", "BA.NYSE", "GE.NYSE", "MMM.NYSE", "HON.NAS",
    "XOM.NYSE", "CVX.NYSE", "COP.NYSE",
    # telecom / utilities / other
    "T.NYSE", "VZ.NYSE", "DIS.NYSE", "TSLA.NAS", "F.NYSE", "GM.NYSE",
]


MIN_PRICE = 5.0        # below this, 0.01 tick quantisation swamps the signal


def clean_window(df: pd.DataFrame, splits: pd.DataFrame,
                 jump_thresh: float = 0.25, min_price: float = 5.0,
                 start: str = "2005-01-01") -> pd.Series:
    """
    Usable window AFTER split correction, truncated past any residual
    discontinuity we could not confidently explain.

    Detection cannot be made reliable on this feed: TSLA's genuine 5:1 split
    prints a ratio of 4.435 because the stock also rose 12.6% that session, so
    it evades ratio matching entirely. Rather than guess, anything still showing
    a >25% single-bar move after correction is treated as unexplained, and the
    symbol's history is started AFTER the last such bar. Losing history is
    cheap; silently trading a fictional -77% candle is not.
    """
    adj = back_adjust(df, splits)
    ok = usable_window(adj, min_price=min_price, start=start).to_numpy().copy()
    c = adj["close"].to_numpy(float)
    ret = np.zeros(len(c))
    ret[1:] = np.diff(c) / np.maximum(c[:-1], 1e-9)
    bad = np.where(np.abs(ret) > jump_thresh)[0]
    if len(bad):
        ok[:int(bad.max()) + 1] = False
    return pd.Series(ok, index=df.index)


def usable_window(df: pd.DataFrame, min_price: float = MIN_PRICE,
                  start: str = "2005-01-01") -> pd.Series:
    """
    Boolean mask of bars fit to compute indicators on.

    Two exclusions, both load-bearing:

    * **Penny back-adjusted history.** Heavily-split growth names have adjusted
      prices of a few cents deep in their history -- NVDA trades at $0.07 in
      2002 on this feed, where the $0.01 tick is a 14% move and high == low on
      most bars. ATR, bands and candle colour computed there are quantisation
      noise, not price action.
    * **Pre-2005.** Decimalisation, feed quality and corporate-action coverage
      are all materially worse further back.
    """
    c = df["close"].to_numpy(float)
    ok = np.isfinite(c) & (c >= min_price)
    ok &= (df["dt"] >= pd.Timestamp(start)).to_numpy()
    return pd.Series(ok, index=df.index)


def detect_splits(df: pd.DataFrame, ratio_tol: float = 0.08,
                  range_mult: float = 2.0, min_jump: float = 0.30,
                  min_price: float = MIN_PRICE) -> pd.DataFrame:
    """
    Find split discontinuities in a daily OHLC frame.

    A bar is flagged when the close-to-close ratio sits within ``ratio_tol`` of a
    known split factor AND that day's own range is not unusually wide (a genuine
    crash of similar magnitude has an enormous range).

    ``ratio_tol`` is deliberately loose (8%). A split day also carries that day's
    real price move -- AAPL's 4:1 on 2020-08-31 prints a ratio of 3.869, not
    4.000, because the stock genuinely rose 3.4% that session. Tightening the
    ratio to catch only exact factors misses real splits; the RANGE test is what
    actually separates a split from a crash, so the ratio only has to get close.

    Bars below ``min_price`` are skipped entirely: at a few cents, tick rounding
    produces exact 4/3 and 3/4 ratios constantly, and a degenerate high == low
    makes the range test read "calm" for every one of them. Without this floor
    the detector flags dozens of phantom splits per symbol.
    """
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    n = len(c)
    cols = ["idx", "dt", "ratio", "factor", "day_range", "median_range", "kind"]
    if n < 30:
        return pd.DataFrame(columns=cols)

    rng = (h - l) / np.maximum(c, 1e-9)
    med_rng = pd.Series(rng).rolling(120, min_periods=20).median().to_numpy()

    rows = []
    for i in range(1, n):
        if not (np.isfinite(c[i]) and np.isfinite(c[i - 1])) or c[i] <= 0:
            continue
        step = c[i] / c[i - 1] - 1.0
        if abs(step) < min_jump:
            continue
        if c[i] < min_price or c[i - 1] < min_price:
            continue          # quantisation regime -- ratios there are noise
        ratio = c[i - 1] / c[i]                      # >1 for a forward split
        best, bestf = None, None
        for f in SPLIT_FACTORS:
            for cand in (f, 1.0 / f):
                if abs(ratio / cand - 1.0) <= ratio_tol:
                    err = abs(ratio / cand - 1.0)
                    if best is None or err < best:
                        best, bestf = err, cand
        if best is None:
            continue
        mr = med_rng[i] if np.isfinite(med_rng[i]) else np.nanmedian(rng)
        # a degenerate median range (high == low, i.e. a dead/quantised series)
        # must not be allowed to make everything look "calm"
        if not np.isfinite(mr) or mr <= 1e-6:
            continue
        calm = rng[i] <= range_mult * mr
        rows.append({"idx": i, "dt": df["dt"].iloc[i], "ratio": ratio,
                     "factor": bestf, "day_range": rng[i], "median_range": mr,
                     "kind": "split" if calm else "crash?"})
    return pd.DataFrame(rows, columns=cols)


def back_adjust(df: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """
    Back-adjust prices for confirmed splits.

    Everything BEFORE a split is divided by the factor, so the series becomes
    continuous and the most recent prices stay on today's scale.
    """
    out = df.copy()
    if splits is None or len(splits) == 0 or "kind" not in splits.columns:
        out["adj_factor"] = 1.0
        return out
    conf = splits[splits["kind"] == "split"]
    if conf.empty:
        out["adj_factor"] = 1.0
        return out
    factor = np.ones(len(out))
    for _, r in conf.iterrows():
        factor[:int(r["idx"])] /= float(r["factor"])
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].to_numpy(float) * factor
    out["adj_factor"] = factor
    return out


def verify(df: pd.DataFrame, label: str = "", thresh: float = 0.25) -> dict:
    """Post-adjustment sanity: no remaining split-shaped discontinuities."""
    c = df["close"].to_numpy(float)
    ret = np.diff(c) / np.maximum(c[:-1], 1e-9)
    big = np.abs(ret) > thresh
    return {"symbol": label, "bars": len(df),
            "start": str(df["dt"].iloc[0].date()), "end": str(df["dt"].iloc[-1].date()),
            "remaining_jumps": int(big.sum()),
            "worst_move": float(np.nanmax(np.abs(ret))) if len(ret) else np.nan,
            "min_price": float(np.nanmin(c)), "max_price": float(np.nanmax(c))}


def pull(symbols=None, timeframes=("H1", "H4", "D1", "W1"), max_bars: int = 99000,
         verbose: bool = True) -> pd.DataFrame:
    """
    Download, split-correct and store. Daily bars drive split detection; the
    resulting adjustment is applied to every timeframe so they stay consistent.
    """
    import MetaTrader5 as mt5
    tfmap = {"H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
             "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1}
    symbols = symbols or UNIVERSE
    os.makedirs(DATA_DIR, exist_ok=True)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    report = []
    for sym in symbols:
        if not mt5.symbol_select(sym, True):
            report.append({"symbol": sym, "status": "not_available"})
            continue
        r = mt5.copy_rates_from_pos(sym, tfmap["D1"], 0, max_bars)
        if r is None or len(r) < 300:
            report.append({"symbol": sym, "status": "insufficient_d1"})
            continue
        d1 = pd.DataFrame(r)
        d1["dt"] = pd.to_datetime(d1["time"], unit="s")
        sp = detect_splits(d1)
        confirmed = sp[sp["kind"] == "split"] if len(sp) else sp

        row = {"symbol": sym, "status": "ok",
               "splits_found": int(len(confirmed)),
               "crash_like_kept": int((sp["kind"] == "crash?").sum()) if len(sp) else 0}

        for tfname in timeframes:
            rr = mt5.copy_rates_from_pos(sym, tfmap[tfname], 0, max_bars)
            if rr is None or len(rr) == 0:
                continue
            df = pd.DataFrame(rr)
            df["dt"] = pd.to_datetime(df["time"], unit="s")
            if len(confirmed):
                # map each split's DATE onto this timeframe's bar index
                factor = np.ones(len(df))
                for _, s in confirmed.iterrows():
                    cutoff = pd.Timestamp(s["dt"]).normalize()
                    factor[df["dt"].to_numpy() < np.datetime64(cutoff)] /= float(s["factor"])
                for col in ("open", "high", "low", "close"):
                    df[col] = df[col].to_numpy(float) * factor
                df["adj_factor"] = factor
            else:
                df["adj_factor"] = 1.0
            df.to_parquet(os.path.join(DATA_DIR, f"{sym.replace('.', '_')}_{tfname}.parquet"))
            v = verify(df, sym)
            row[f"{tfname}_bars"] = v["bars"]
            row[f"{tfname}_start"] = v["start"]
            row[f"{tfname}_jumps"] = v["remaining_jumps"]
        if verbose:
            print(f"  {sym:<12} splits={row['splits_found']} "
                  f"crash-like-kept={row['crash_like_kept']} "
                  f"D1={row.get('D1_bars','-')} from {row.get('D1_start','-')} "
                  f"residual-jumps={row.get('D1_jumps','-')}", flush=True)
        report.append(row)
    mt5.shutdown()
    rep = pd.DataFrame(report)
    rep.to_parquet(os.path.join(DATA_DIR, "_universe_report.parquet"))
    return rep


if __name__ == "__main__":
    print("Stock CFD pull with split correction")
    rep = pull()
    ok = rep[rep.status == "ok"] if "status" in rep else rep
    print(f"\n{len(ok)}/{len(rep)} symbols usable")
    if "D1_jumps" in ok:
        print(f"symbols with residual >25% daily jumps after adjustment: "
              f"{int((ok['D1_jumps'] > 0).sum())}")
        bad = ok[ok["D1_jumps"] > 0]
        if len(bad):
            print(bad[["symbol", "splits_found", "D1_jumps"]].to_string(index=False))
