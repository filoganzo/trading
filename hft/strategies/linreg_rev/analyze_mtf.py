"""Stage A cross-pair analysis. Reads existing sweep parquets, reports only."""
from __future__ import annotations
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

FILES = ["sweeps/XAUUSD_M5_M15.parquet", "sweeps/XAUUSD_M5_M30.parquet",
         "sweeps/XAUUSD_M5_M45.parquet", "sweeps/XAUUSD_M5_H1.parquet",
         "sweeps/XAUUSD_M5_none.parquet"]
HZ = (6, 12, 24, 48)
ORDER = ["none", "M15", "M30", "M45", "H1"]
NMIN = 60

raw = pd.concat([pd.read_parquet(f) for f in FILES], ignore_index=True)
df = raw[raw["n"] >= NMIN].copy()
df["htf_tf"] = pd.Categorical(df["htf_tf"], ORDER, ordered=True)


def md(dfx, cols, floats=3):
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in dfx.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                vals.append("nan" if not np.isfinite(v) else f"{v:.{floats}f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


print("### 0. Config counts (n>=60 filter)\n")
cnt = (raw.assign(keep=raw["n"] >= NMIN)
          .groupby(["htf_tf", "dir"], observed=True)
          .agg(total=("n", "size"), kept=("keep", "sum"),
               med_n=("n", "median")).reset_index())
cnt["htf_tf"] = pd.Categorical(cnt["htf_tf"], ORDER, ordered=True)
print(md(cnt.sort_values(["htf_tf", "dir"]),
         ["htf_tf", "dir", "total", "kept", "med_n"], 1))

g = df.groupby(["htf_tf", "dir"], observed=True)

# ---------- 1. excess drift ----------
print("\n\n### 1. Excess drift (ATR units)\n")
for h in HZ:
    t = g.agg(cfgs=("n", "size"), mean=(f"exc{h}", "mean"),
              median=(f"exc{h}", "median"), sd=(f"exc{h}", "std")).reset_index()
    t["share_gt0"] = g[f"exc{h}"].apply(lambda s: (s > 0).mean()).values
    t = t.sort_values(["htf_tf", "dir"])
    print(f"\n**exc{h}**\n")
    print(md(t, ["htf_tf", "dir", "cfgs", "mean", "median", "sd", "share_gt0"], 4))

# ---------- 2. IS/OOS consistency ----------
print("\n\n### 2. IS/OOS consistency\n")
for h in HZ:
    rows = []
    for (k, dd), sub in g:
        N = len(sub)
        rows.append(dict(
            htf_tf=k, dir=dd, cfgs=N,
            share_both_pos=((sub[f"exc{h}_is"] > 0) & (sub[f"exc{h}_oos"] > 0)).mean(),
            cnt_both_t15=int(((sub[f"t{h}_is"] > 1.5) & (sub[f"t{h}_oos"] > 1.5)).sum()),
            exp_chance=0.0045 * N,
            cnt_both_t15_neg=int(((sub[f"t{h}_is"] < -1.5) & (sub[f"t{h}_oos"] < -1.5)).sum())))
    t = pd.DataFrame(rows)
    t["ratio"] = t["cnt_both_t15"] / t["exp_chance"]
    t["htf_tf"] = pd.Categorical(t["htf_tf"], ORDER, ordered=True)
    t = t.sort_values(["htf_tf", "dir"])
    print(f"\n**h={h}**\n")
    print(md(t, ["htf_tf", "dir", "cfgs", "share_both_pos", "cnt_both_t15",
                 "exp_chance", "ratio", "cnt_both_t15_neg"], 3))

# ---------- 3. pmid24 ----------
print("\n\n### 3. pmid24 -- P(reach midline before 1 ATR adverse)\n")
t = g.agg(cfgs=("pmid24", "size"), mean=("pmid24", "mean"),
          median=("pmid24", "median"),
          p10=("pmid24", lambda s: s.quantile(.10)),
          p90=("pmid24", lambda s: s.quantile(.90)),
          share_gt_half=("pmid24", lambda s: (s > 0.5).mean())).reset_index()
print(md(t.sort_values(["htf_tf", "dir"]),
         ["htf_tf", "dir", "cfgs", "mean", "median", "p10", "p90", "share_gt_half"], 4))

# ---------- 4. edge ----------
print("\n\n### 4. Path asymmetry edge = mean MFE / mean |MAE|\n")
t = g.agg(cfgs=("edge12", "size"), edge6=("edge6", "mean"),
          edge12=("edge12", "mean"), edge24=("edge24", "mean"),
          edge48=("edge48", "mean"),
          e12_gt1=("edge12", lambda s: (s > 1).mean()),
          e48_gt1=("edge48", lambda s: (s > 1).mean())).reset_index()
print(md(t.sort_values(["htf_tf", "dir"]),
         ["htf_tf", "dir", "cfgs", "edge6", "edge12", "edge24", "edge48",
          "e12_gt1", "e48_gt1"], 4))

# ---------- 5. forming vs closed ----------
print("\n\n### 5. htf_align: forming vs closed\n")
hd = df[df["htf_tf"] != "none"].copy()
t = (hd.groupby(["htf_tf", "dir", "htf_align"], observed=True)
       .agg(cfgs=("n", "size"), med_n=("n", "median"), exc12=("exc12", "mean"),
            exc24=("exc24", "mean"), exc48=("exc48", "mean"),
            pmid24=("pmid24", "mean"), edge12=("edge12", "mean"),
            edge48=("edge48", "mean")).reset_index())
t["htf_tf"] = pd.Categorical(t["htf_tf"], ORDER, ordered=True)
print(md(t.sort_values(["htf_tf", "dir", "htf_align"]),
         ["htf_tf", "dir", "htf_align", "cfgs", "med_n", "exc12", "exc24",
          "exc48", "pmid24", "edge12", "edge48"], 4))

print("\n**forming minus closed (paired on identical base+htf params, pre-n-filter)**\n")
keys = ["htf_tf", "dir", "chan_len", "min_red_run", "touch_src", "wt_level",
        "wt_cross_lookback", "htf_chan_len", "htf_wt_level", "htf_band_src"]
hd_all = raw[raw["htf_tf"] != "none"].copy()
piv = hd_all.pivot_table(index=keys, columns="htf_align",
                         values=["exc12", "exc24", "exc48", "pmid24", "edge12",
                                 "edge48", "n"], observed=True)
rows = []
for (k, dd), sub in piv.groupby(level=["htf_tf", "dir"], observed=True):
    r = dict(htf_tf=k, dir=dd, pairs=len(sub))
    for m in ["n", "exc12", "exc24", "exc48", "pmid24", "edge12", "edge48"]:
        d_ = (sub[(m, "forming")] - sub[(m, "closed")]).dropna()
        r[f"d_{m}"] = d_.mean()
        if m == "pmid24":
            r["win_pmid"] = (d_ > 0).mean()
    rows.append(r)
t = pd.DataFrame(rows)
t["htf_tf"] = pd.Categorical(t["htf_tf"], ORDER, ordered=True)
print(md(t.sort_values(["htf_tf", "dir"]),
         ["htf_tf", "dir", "pairs", "d_n", "d_exc12", "d_exc24", "d_exc48",
          "d_pmid24", "win_pmid", "d_edge12", "d_edge48"], 4))

# ---------- 6. HTF vs none ----------
print("\n\n### 6. HTF confirmation vs NO confirmation (paired on base config)\n")
bkeys = ["dir", "chan_len", "min_red_run", "touch_src", "wt_level",
         "wt_cross_lookback"]
none = raw[raw["htf_tf"] == "none"].set_index(bkeys)
mets = ["n", "exc12", "exc24", "exc48", "pmid24", "edge12", "edge48"]
rows = []
for htf in ["M15", "M30", "M45", "H1"]:
    sub = df[df["htf_tf"] == htf]
    agg = sub.groupby(bkeys, observed=True)[mets].mean()
    j = agg.join(none[mets], rsuffix="_none", how="inner")
    for dd in ["long", "short"]:
        s = j.xs(dd, level="dir")
        r = dict(htf_tf=htf, dir=dd, base_cfgs=len(s))
        for m in mets:
            r[f"htf_{m}"] = s[m].mean()
            r[f"none_{m}"] = s[f"{m}_none"].mean()
            r[f"d_{m}"] = (s[m] - s[f"{m}_none"]).mean()
            if m in ("pmid24", "exc24"):
                r[f"win_{m}"] = (s[m] > s[f"{m}_none"]).mean()
        rows.append(r)
t = pd.DataFrame(rows)
t["htf_tf"] = pd.Categorical(t["htf_tf"], ORDER, ordered=True)
t = t.sort_values(["dir", "htf_tf"])
print("\n**signal count & pmid24**\n")
print(md(t, ["htf_tf", "dir", "base_cfgs", "htf_n", "none_n", "d_n",
             "htf_pmid24", "none_pmid24", "d_pmid24", "win_pmid24"], 4))
print("\n**excess drift**\n")
print(md(t, ["htf_tf", "dir", "htf_exc12", "none_exc12", "d_exc12",
             "htf_exc24", "none_exc24", "d_exc24", "win_exc24",
             "htf_exc48", "none_exc48", "d_exc48"], 4))
print("\n**edge**\n")
print(md(t, ["htf_tf", "dir", "htf_edge12", "none_edge12", "d_edge12",
             "htf_edge48", "none_edge48", "d_edge48"], 4))

# ---------- 7. top configs, context only ----------
print("\n\n### 7. Top 5 configs by exc24 (context only, NOT a recommendation)\n")
for dd in ["long", "short"]:
    s = df[df["dir"] == dd].nlargest(5, "exc24")
    print(f"\n**{dd}**\n")
    print(md(s, ["htf_tf", "chan_len", "min_red_run", "touch_src", "wt_level",
                 "htf_chan_len", "htf_wt_level", "htf_band_src", "htf_align",
                 "n", "exc24", "t24", "exc24_is", "t24_is", "exc24_oos",
                 "t24_oos", "pmid24"], 3))

# ---------- 8. globals ----------
print("\n\n### 8. Global summary across all n>=60 configs\n")
print(f"total rows loaded: {len(raw)},  kept n>=60: {len(df)}")
print(f"overall mean pmid24: {df['pmid24'].mean():.4f}, "
      f"share>0.5: {(df['pmid24']>0.5).mean():.4f}")
for h in HZ:
    print(f"exc{h}: mean={df[f'exc{h}'].mean():+.4f} "
          f"share>0={(df[f'exc{h}']>0).mean():.4f} "
          f"both_t>1.5={int(((df[f't{h}_is']>1.5)&(df[f't{h}_oos']>1.5)).sum())} "
          f"both_t<-1.5={int(((df[f't{h}_is']<-1.5)&(df[f't{h}_oos']<-1.5)).sum())} "
          f"(chance~{0.0045*len(df):.0f})")

# ---------- 9. long/short symmetry decomposition ----------
print("\n\n### 9. Symmetry decomposition of excess drift\n")
print("sym  = (exc_long + exc_short)/2  -> direction-neutral component "
      "(a real reversion edge must be > 0)")
print("anti = (exc_long - exc_short)/2  -> directional/trend-beta component\n")
rows = []
for h in HZ:
    m = df.groupby(["htf_tf", "dir"], observed=True)[f"exc{h}"].mean().unstack()
    for k in ORDER:
        if k in m.index:
            L, S = m.loc[k, "long"], m.loc[k, "short"]
            rows.append(dict(h=h, htf_tf=k, exc_long=L, exc_short=S,
                             sym=(L + S) / 2, anti=(L - S) / 2))
t = pd.DataFrame(rows)
print(md(t, ["h", "htf_tf", "exc_long", "exc_short", "sym", "anti"], 4))

print("\n**pmid24 symmetry (mean of long and short)**\n")
p = df.groupby(["htf_tf", "dir"], observed=True)["pmid24"].mean().unstack()
p["avg_both_sides"] = p.mean(axis=1)
print(md(p.reset_index(), ["htf_tf", "long", "short", "avg_both_sides"], 4))
