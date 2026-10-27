"""
Stage A M1 analysis: read the four XAUUSD M1 sweep parquets and report
family-level (htf_tf x dir) statistics. Read-only -- no engine code here.

EXPLORATORY ONLY: the M1 base has ~3.4 months of history (2026-05-21 ..
2026-08-31), capped by the MT5 terminal at 99,000 bars. One quarter of one
regime cannot validate anything.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

pd.set_option("display.width", 240)
HZ = (6, 12, 24, 48)
NMIN = 60
TCUT = 1.5
# one-sided P(t > 1.5) for large-n t ~ N(0,1) is ~6.68%; both halves clearing
# it independently by luck alone is ~0.45%
CHANCE_BOTH_T = 0.0668 ** 2

FILES = ["sweeps/XAUUSD_M1_M5.parquet", "sweeps/XAUUSD_M1_M15.parquet",
         "sweeps/XAUUSD_M1_M30.parquet", "sweeps/XAUUSD_M1_none.parquet"]


def load() -> pd.DataFrame:
    parts = []
    for f in FILES:
        if not os.path.exists(f):
            print("MISSING " + f)
            continue
        d = pd.read_parquet(f)
        parts.append(d)
        print("%-34s %6d rows" % (f, len(d)))
    return pd.concat(parts, ignore_index=True)


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".4f")


def sec(t: str) -> None:
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


def main() -> None:
    raw = load()
    df = raw[raw["n"] >= NMIN].copy()

    sec("0. CONFIG COUNTS  (raw vs n>=%d)" % NMIN)
    tot = raw.groupby(["htf_tf", "dir"]).size().rename("configs_raw")
    kept = df.groupby(["htf_tf", "dir"]).size().rename("configs_kept")
    cnt = pd.concat([tot, kept], axis=1).fillna(0).astype(int)
    cnt["kept_share"] = cnt["configs_kept"] / cnt["configs_raw"]
    print(md(cnt.reset_index()))
    print("\nTOTAL raw rows %d -> n>=%d: %d  (%.1f%% kept)"
          % (len(raw), NMIN, len(df), 100 * len(df) / len(raw)))

    sec("1. SIGNAL COUNT SITUATION  (all raw rows, before the n filter)")
    g = raw.groupby(["htf_tf", "dir"])["n"]
    sig = pd.DataFrame({
        "configs": g.size(), "n_min": g.min(), "n_p10": g.quantile(.10),
        "n_median": g.median(), "n_p90": g.quantile(.90), "n_max": g.max(),
        "sh_n<100": raw.assign(x=raw["n"] < 100).groupby(["htf_tf", "dir"])["x"].mean(),
        "sh_n<200": raw.assign(x=raw["n"] < 200).groupby(["htf_tf", "dir"])["x"].mean(),
    })
    print(md(sig.reset_index()))
    print("\nNOTE: sweep_mtf.py drops any config with <30 signals before writing a")
    print("row, so the raw table already excludes the very thinnest configs.")
    g2 = df.groupby(["htf_tf", "dir"])
    halves = pd.DataFrame({"n_is_median": g2["n_is"].median(),
                           "n_oos_median": g2["n_oos"].median(),
                           "n_median": g2["n"].median()})
    print("\nmedian per-half signal counts among n>=%d configs:" % NMIN)
    print(md(halves.reset_index()))

    sec("2. EXCESS DRIFT exc6/12/24/48  (M1 base -> 6/12/24/48 MINUTES ahead)")
    rows = []
    for (h, d0), gg in df.groupby(["htf_tf", "dir"]):
        r = {"htf_tf": h, "dir": d0, "configs": len(gg)}
        for k in HZ:
            c = gg["exc%d" % k]
            r["exc%d_mean" % k] = c.mean()
            r["exc%d_med" % k] = c.median()
            r["exc%d_pos" % k] = (c > 0).mean()
        rows.append(r)
    e = pd.DataFrame(rows)
    print("MEAN excess drift (ATR units):")
    print(md(e[["htf_tf", "dir", "configs"] + ["exc%d_mean" % k for k in HZ]]))
    print("\nMEDIAN excess drift:")
    print(md(e[["htf_tf", "dir"] + ["exc%d_med" % k for k in HZ]]))
    print("\nSHARE of configs with exc > 0:")
    print(md(e[["htf_tf", "dir"] + ["exc%d_pos" % k for k in HZ]]))

    sec("3. IS/OOS CONSISTENCY  (chronological split at 60%% of the sample)")
    rows = []
    for (h, d0), gg in df.groupby(["htf_tf", "dir"]):
        r = {"htf_tf": h, "dir": d0, "configs": len(gg),
             "exp_by_chance": CHANCE_BOTH_T * len(gg)}
        for k in HZ:
            bp = (gg["exc%d_is" % k] > 0) & (gg["exc%d_oos" % k] > 0)
            bt = (gg["t%d_is" % k] > TCUT) & (gg["t%d_oos" % k] > TCUT)
            r["h%d_bothpos" % k] = bp.mean()
            r["h%d_botht_n" % k] = int(bt.sum())
            r["h%d_botht_sh" % k] = bt.mean()
        rows.append(r)
    io = pd.DataFrame(rows)
    print("SHARE with BOTH halves exc>0  (chance baseline = 0.25):")
    print(md(io[["htf_tf", "dir", "configs"] + ["h%d_bothpos" % k for k in HZ]]))
    print("\nCOUNT with BOTH t>%.1f  (exp_by_chance = %.4f%% x configs):"
          % (TCUT, 100 * CHANCE_BOTH_T))
    print(md(io[["htf_tf", "dir", "configs", "exp_by_chance"]
                + ["h%d_botht_n" % k for k in HZ]]))
    print("\nsame, as a SHARE (chance baseline %.4f):" % CHANCE_BOTH_T)
    print(md(io[["htf_tf", "dir"] + ["h%d_botht_sh" % k for k in HZ]]))

    sec("4. pmid24  (P reach channel MIDLINE before 1 ATR adverse; >0.5 = reversion)")
    rows = []
    for (h, d0), gg in df.groupby(["htf_tf", "dir"]):
        c = gg["pmid24"].dropna()
        rows.append({"htf_tf": h, "dir": d0, "configs": len(gg), "n_valid": len(c),
                     "mean": c.mean(), "median": c.median(),
                     "p10": c.quantile(.10), "p90": c.quantile(.90),
                     "share>0.5": (c > 0.5).mean()})
    print(md(pd.DataFrame(rows)))

    sec("5. edge  (mean MFE / mean |MAE|)")
    rows = []
    for (h, d0), gg in df.groupby(["htf_tf", "dir"]):
        rows.append({"htf_tf": h, "dir": d0, "configs": len(gg),
                     "edge6": gg["edge6"].mean(),
                     "edge12_mean": gg["edge12"].mean(),
                     "edge12_med": gg["edge12"].median(),
                     "edge12_sh>1": (gg["edge12"] > 1).mean(),
                     "edge24": gg["edge24"].mean(),
                     "edge48_mean": gg["edge48"].mean(),
                     "edge48_med": gg["edge48"].median(),
                     "edge48_sh>1": (gg["edge48"] > 1).mean()})
    print(md(pd.DataFrame(rows)))

    sec("6. HTF ALIGNMENT: forming vs closed")
    h = df[df["htf_tf"] != "none"]
    rows = []
    for (a, d0), gg in h.groupby(["htf_align", "dir"]):
        r = {"align": a, "dir": d0, "configs": len(gg), "n_med": gg["n"].median()}
        for k in HZ:
            r["exc%d" % k] = gg["exc%d" % k].mean()
        r["pmid24"] = gg["pmid24"].mean()
        r["edge12"] = gg["edge12"].mean()
        r["edge48"] = gg["edge48"].mean()
        r["bothpos12"] = ((gg["exc12_is"] > 0) & (gg["exc12_oos"] > 0)).mean()
        r["botht12_n"] = int(((gg["t12_is"] > TCUT) & (gg["t12_oos"] > TCUT)).sum())
        rows.append(r)
    print(md(pd.DataFrame(rows)))
    print("\nper htf_tf x align x dir (means):")
    piv = h.groupby(["htf_tf", "htf_align", "dir"])[
        ["n", "exc12", "exc24", "pmid24", "edge12"]].mean()
    print(md(piv.reset_index()))

    sec("7. HTF CONFIRMATION vs NONE")
    df2 = df.assign(gate=np.where(df["htf_tf"] == "none", "none", "htf"))
    rows = []
    for (gt, d0), gg in df2.groupby(["gate", "dir"]):
        r = {"gate": gt, "dir": d0, "configs": len(gg), "n_med": gg["n"].median()}
        for k in HZ:
            r["exc%d" % k] = gg["exc%d" % k].mean()
            r["exc%d_pos" % k] = (gg["exc%d" % k] > 0).mean()
        r["pmid24"] = gg["pmid24"].mean()
        r["edge12"] = gg["edge12"].mean()
        r["edge48"] = gg["edge48"].mean()
        rows.append(r)
    hv = pd.DataFrame(rows)
    print(md(hv[["gate", "dir", "configs", "n_med"] + ["exc%d" % k for k in HZ]
               + ["pmid24", "edge12", "edge48"]]))
    print("\nshare exc>0:")
    print(md(hv[["gate", "dir"] + ["exc%d_pos" % k for k in HZ]]))

    print("\nMATCHED: same base config, HTF-gated (averaged over HTF grid) minus ungated")
    bkeys = ["chan_len", "min_red_run", "touch_src", "wt_level",
             "wt_cross_lookback", "dir"]
    cols = ["exc12", "exc24", "exc48", "pmid24", "edge12", "n"]
    # use the UNFILTERED frame for the ungated side so pairs are not lost
    none = raw[raw["htf_tf"] == "none"].set_index(bkeys)
    for htf in ["M5", "M15", "M30"]:
        sub = df[df["htf_tf"] == htf]
        if not len(sub):
            continue
        agg = sub.groupby(bkeys)[cols].mean()
        j = agg.join(none[cols], rsuffix="_none", how="inner")
        if not len(j):
            print("  %s: no matched pairs" % htf)
            continue
        d_ = {c: (j[c] - j[c + "_none"]).mean() for c in cols}
        print("  %-4s vs none (pairs=%3d): " % (htf, len(j))
              + "  ".join("d%s=%+.4f" % (c, v) for c, v in d_.items()))

    sec("8. DISTRIBUTION of exc12 across n>=%d configs" % NMIN)
    for (hh, dd), gg in df.groupby(["htf_tf", "dir"]):
        q = gg["exc12"].quantile([.05, .25, .5, .75, .95]).to_dict()
        print("  %5s %5s  cfg=%5d  p05=%+.4f p25=%+.4f p50=%+.4f p75=%+.4f p95=%+.4f"
              % (hh, dd, len(gg), q[.05], q[.25], q[.5], q[.75], q[.95]))

    regime_correction(df)


def regime_correction(df: pd.DataFrame) -> None:
    """
    The sweep detrends every config against the FULL-sample unconditional drift.
    The M1 window is two opposite regimes (June 2026 crash, August 2026 rally),
    so the full-sample baseline is ~0 while each half is strongly signed. Redo
    the IS/OOS read against HALF-SPECIFIC baselines.

        exc      = mean(drift) - side*base_full
        corrected= mean(drift) - side*base_half
                 = exc + side*(base_full - base_half)
    """
    from mtf_runner import base_indicators, BaseParams
    sec("9. REGIME CORRECTION: half-specific unconditional drift")
    d = base_indicators("XAUUSD", "M1", BaseParams())
    utc = d["utc"].to_numpy()
    split = np.datetime64(pd.Timestamp("2026-07-21"))
    c = d["close"].to_numpy(float); a = d["atr"].to_numpy(float); n = len(c)
    ism = utc < split
    full, isb, oosb = {}, {}, {}
    for k in HZ:
        f = np.full(n, np.nan); f[:n - k] = c[k:] - c[:n - k]
        z = f / a
        ok = np.isfinite(z)
        full[k] = z[ok].mean()
        isb[k] = z[ism & ok].mean()
        oosb[k] = z[(~ism) & ok].mean()
    bt = pd.DataFrame({"h": list(HZ),
                       "uncond_full": [full[k] for k in HZ],
                       "uncond_IS": [isb[k] for k in HZ],
                       "uncond_OOS": [oosb[k] for k in HZ]})
    print("unconditional drift of XAUUSD M1 by half (ATR units):")
    print(md(bt))

    rows = []
    for (h, d0), gg in df.groupby(["htf_tf", "dir"]):
        sd = 1.0 if d0 == "long" else -1.0
        r = {"htf_tf": h, "dir": d0, "cfg": len(gg)}
        for k in HZ:
            cis = gg["exc%d_is" % k] + sd * (full[k] - isb[k])
            coos = gg["exc%d_oos" % k] + sd * (full[k] - oosb[k])
            r["h%d_IS" % k] = cis.mean()
            r["h%d_OOS" % k] = coos.mean()
            r["h%d_bothpos" % k] = ((cis > 0) & (coos > 0)).mean()
        rows.append(r)
    o = pd.DataFrame(rows)
    print("\nregime-corrected MEAN excess by half:")
    print(md(o[["htf_tf", "dir", "cfg"] + sum(
        [["h%d_IS" % k, "h%d_OOS" % k] for k in HZ], [])]))
    print("\nregime-corrected SHARE both halves positive (chance 0.25):")
    print(md(o[["htf_tf", "dir"] + ["h%d_bothpos" % k for k in HZ]]))

    if "spread" in d:
        from mtf_runner import POINT
        s = d["spread"].to_numpy(float) * POINT["XAUUSD"]
        print("\ncost scale: mean spread %.3f USD = %.4f ATR one-way, "
              "%.4f ATR round-trip (mean ATR %.3f USD)"
              % (s.mean(), s.mean() / np.nanmean(a), 2 * s.mean() / np.nanmean(a), np.nanmean(a)))
        print("the sweep charges ONE spread (at entry) and no exit cost, so "
              "subtract ~%.4f ATR more for a round trip." % (s.mean() / np.nanmean(a)))


if __name__ == "__main__":
    main()
