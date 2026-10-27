"""
Read the four difference-in-differences sweeps (M5 base vs M15/M30/M45/H1 HTF)
and answer one question: does the base trigger carry information BEYOND the
regime the HTF gate selects, and is whatever it carries a plateau or a spike?

Everything is filtered to nA >= 50 so a cell's mean is not one lucky week.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (12, 24, 48)
HTFS = ["M15", "M30", "M45", "H1"]
MINA = 50
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)


def load() -> pd.DataFrame:
    fr = []
    for h in HTFS:
        p = f"sweeps/did_XAUUSD_M5_{h}.parquet"
        d = pd.read_parquet(p)
        fr.append(d)
        print(f"  {p}: {len(d)} rows, {int((d.nA >= MINA).sum())} with nA>={MINA}")
    df = pd.concat(fr, ignore_index=True)
    return df[df.nA >= MINA].copy()


def md(df: pd.DataFrame, floatfmt="%.3f") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt % v)
    return d.to_markdown(index=False)


def _cat(df):
    """Order rows HTF-ascending, long first."""
    df = df.copy()
    df["htf_tf"] = pd.Categorical(df["htf_tf"], HTFS, ordered=True)
    df["dir"] = pd.Categorical(df["dir"], ["long", "short"], ordered=True)
    return df.sort_values(["htf_tf", "dir"])


# --- 1 + 2: A vs B, trigger value, DiD ------------------------------------
def block_ab(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (htf, dn), g in df.groupby(["htf_tf", "dir"], observed=True):
        for h in HORIZONS:
            tg, t = g[f"trig_gated_{h}"], g[f"t_gated_{h}"]
            did = g[f"did_{h}"]
            rows.append({
                "htf_tf": htf, "dir": dn, "h": h, "n_cfg": len(g),
                "meanA": g[f"meanA_{h}"].mean(),
                "meanB": g[f"meanB_{h}"].mean(),
                "trig_gated_mean": tg.mean(),
                "t_median": t.median(),
                "pct_pos": 100 * (tg > 0).mean(),
                "pct_t>1.96": 100 * (t > 1.96).mean(),
                "pct_t<-1.96": 100 * (t < -1.96).mean(),
                "trig_ungated_mean": g[f"trig_ungated_{h}"].mean(),
                "DiD_mean": did.mean(),
                "DiD_pct_pos": 100 * (did > 0).mean(),
            })
    return _cat(pd.DataFrame(rows)).sort_values(["htf_tf", "dir", "h"])


# --- 3: is the reversion bet priced to win? -------------------------------
def block_pricing(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (htf, dn), g in df.groupby(["htf_tf", "dir"], observed=True):
        rows.append({
            "htf_tf": htf, "dir": dn, "n_cfg": len(g),
            "mid_dist_atr": g["mid_dist_atr"].mean(),
            "p_mid_first": g["p_mid_first"].mean(),
            "avg_target_atr": g["avg_target_atr"].mean(),
            "breakeven_p": g["breakeven_p"].mean(),
            "edge_vs_breakeven": g["edge_vs_breakeven"].mean(),
            "expectancy_atr": g["expectancy_atr"].mean(),
            "exp_atr_median": g["expectancy_atr"].median(),
            "pct_exp>0": 100 * (g["expectancy_atr"] > 0).mean(),
        })
    return _cat(pd.DataFrame(rows))


# --- 4: HTF ranking ------------------------------------------------------
def block_rank(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dn in ["long", "short"]:
        for h in HORIZONS:
            g = df[df["dir"] == dn]
            s = g.groupby("htf_tf", observed=True)[f"trig_gated_{h}"].mean()
            s = s.reindex(HTFS)
            r = {"dir": dn, "h": h}
            r.update({k: v for k, v in s.items()})
            r["best"] = s.idxmax()
            rows.append(r)
    return pd.DataFrame(rows)


# --- 5: plateau or spike -------------------------------------------------
def block_grid(df: pd.DataFrame, dn: str, htf: str | None, h: int = 48):
    g = df[df["dir"] == dn]
    if htf:
        g = g[g.htf_tf == htf]
    col = f"trig_gated_{h}"
    out = {}
    out["align x wt_level"] = g.pivot_table(index="htf_align", columns="htf_wt_level",
                                            values=col, aggfunc="mean")
    out["align x chan_len"] = g.pivot_table(index="htf_align", columns="htf_chan_len",
                                            values=col, aggfunc="mean")
    out["wt_level x chan_len"] = g.pivot_table(index="htf_wt_level", columns="htf_chan_len",
                                               values=col, aggfunc="mean")
    out["htf_tf x align"] = g.pivot_table(index="htf_tf", columns="htf_align",
                                          values=col, aggfunc="mean").reindex(HTFS)
    return out


def full_grid(df, dn, h=48):
    """Every (align, wt_level, chan_len) cell x htf_tf -- the plateau check."""
    g = df[df["dir"] == dn]
    p = g.pivot_table(index=["htf_align", "htf_wt_level", "htf_chan_len"],
                      columns="htf_tf", values=f"trig_gated_{h}", aggfunc="mean")
    return p.reindex(columns=HTFS)


def main():
    print("Loading DiD sweeps (M5 base):")
    df = load()
    print(f"  combined, nA>={MINA}: {len(df)} configs\n")

    print("## 1+2. A vs B, trigger value with regime held constant, and DiD\n")
    print(md(block_ab(df)))

    print("\n\n## 3. Reversion bet, priced\n")
    print(md(block_pricing(df)))

    print("\n\n## 4. HTF ranking by mean trig_gated\n")
    print(md(block_rank(df)))

    for dn in ["long", "short"]:
        print(f"\n\n## 5. Parameter grids, mean trig_gated_48, {dn.upper()}, all HTFs pooled\n")
        for name, tab in block_grid(df, dn, None).items():
            print(f"\n### {name}\n")
            print(tab.round(3).to_markdown())
        print(f"\n### full (align, wt_level, chan_len) x htf_tf -- {dn.upper()}\n")
        print(full_grid(df, dn).round(3).to_markdown())

    # dispersion: plateau vs spike
    print("\n\n## 5b. Plateau diagnostic -- spread of cell means vs within-cell noise\n")
    rows = []
    for dn in ["long", "short"]:
        for htf in HTFS:
            g = df[(df["dir"] == dn) & (df.htf_tf == htf)]
            cells = g.groupby(["htf_align", "htf_wt_level", "htf_chan_len"],
                              observed=True)["trig_gated_48"].mean()
            rows.append({"dir": dn, "htf_tf": htf, "n_cells": len(cells),
                         "cell_min": cells.min(), "cell_med": cells.median(),
                         "cell_max": cells.max(),
                         "cell_std": cells.std(),
                         "pct_cells_pos": 100 * (cells > 0).mean(),
                         "max_over_med": (cells.max() / cells.median())
                         if cells.median() != 0 else np.nan})
    print(md(pd.DataFrame(rows)))


if __name__ == "__main__":
    main()
