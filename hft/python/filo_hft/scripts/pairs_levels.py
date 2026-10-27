"""
FTSE MIB Ratio Key-Level Screener
===================================
Adds value to an experienced pairs trader by:

  1. SCORING key levels by how many times the ratio has respected them
     (not just visual S/R — each level gets a "touch count" and recency score)

  2. MULTI-TIMEFRAME CONFLUENCE — flag levels that appear on BOTH M15
     and H1 at the same ratio value (within 0.3%). These are the strongest
     levels. A human can't track this across a dozen pairs simultaneously.

  3. HALF-LIFE — Ornstein-Uhlenbeck estimate of how long the trade should
     take to revert 50%. Tells the trader which timeframe to hold on.

  4. TREND FILTER — is the ratio currently ranging or trending? Only range
     periods produce reliable reversion. Shows a simple "RANGE / TREND" flag.

  5. MULTI-PAIR CONFIRMATION — if ISP/UCG and ISP/G are BOTH approaching
     key levels at the same time, that's a structural divergence in ISP vs
     its peers, not a single-pair noise event. Much stronger signal.

Run:
    python -m filo_hft.scripts.pairs_levels            # full scan
    python -m filo_hft.scripts.pairs_levels --html     # open HTML in browser

Output:  data/pairs_levels/screener.html
         data/pairs_levels/levels_detail.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────
# Pair universe with full economic rationale
# ─────────────────────────────────────────────────────────────────
PAIRS = {
    # ── Trader's core pairs ──
    "ISP/UCG" : ("ISP",  "UCG",
        "Italy's two largest banks — same ECB rate sensitivity, same BTP duration risk, "
        "same Italian NPA legacy. When one outperforms the other, it's usually a "
        "short-term sentiment divergence (analyst note, block trade, index weight change) "
        "that reverts within days.",
        "Banks", "core"),
    "ISP/G"   : ("ISP",  "G",
        "Intesa Sanpaolo (bank) + Generali (insurer). ISP owns 6.7% of Generali. "
        "Both are large-cap Italian financial institutions sensitive to Italian sovereign "
        "risk and BTP spread. Ratio diverges on sector rotation (banks vs insurance) "
        "and reverts when the sector rotation unwinds.",
        "Financials", "core"),
    "UCG/G"   : ("UCG",  "G",
        "UniCredit + Generali — both pillars of the Italian financial system, both in "
        "the same FTSE MIB basket, both repriced together on BTP spread moves. "
        "Divergence often driven by M&A rumours on one leg.",
        "Financials", "core"),
    # ── High-confidence additions ──
    "ISP/MB"  : ("ISP",  "MB",
        "Intesa + Mediobanca. ISP holds ~13% of MB — the cross-shareholding creates "
        "a structural gravitational pull. When the ratio widens, the ownership linkage "
        "pulls it back. Confirmed positive across M15/M30/H1 in backtests.",
        "Banks", "add"),
    "ISP/BAMI": ("ISP",  "BAMI",
        "Intesa + Banco BPM — both Italian retail banking, near-identical NII sensitivity "
        "to Euribor. BAMI is a frequent M&A target which can temporarily inflate its premium.",
        "Banks", "add"),
    "UCG/MB"  : ("UCG",  "MB",
        "UniCredit + Mediobanca — both sensitive to Italian financial sector flows. "
        "MB acts as a mid-cap banking index; UCG diverges on international news.",
        "Banks", "add"),
    "G/MB"    : ("G",    "MB",
        "Generali + Mediobanca. MB holds ~17% of Generali historically — "
        "a structural link that makes sustained divergences rare.",
        "Financials", "add"),
    # ── Utility bond-proxy cluster ──
    "ENEL/TRN": ("ENEL", "TRN",
        "Enel (electricity generator/distributor) + Terna (grid operator). "
        "Both regulated by ARERA under same tariff methodology. "
        "Trade like Italian infrastructure bonds. Diverge on energy price vs pure-grid news.",
        "Utilities", "utility"),
    "IG/TRN"  : ("IG",   "TRN",
        "Italgas (gas distribution monopoly) + Terna (electricity grid monopoly). "
        "Both ARERA-regulated, both 'bond proxy' stocks. Move together on BTP yield changes.",
        "Utilities", "utility"),
    "SRG/TRN" : ("SRG",  "TRN",
        "Snam (gas pipelines) + Terna (electricity grid). Most direct pair of Italian "
        "regulated network monopolies — same regulatory regime, same ESG investor base.",
        "Utilities", "utility"),
    "HER/A2A" : ("HER",  "A2A",
        "Hera + A2A — both Italian multi-utility operators (gas, electricity, waste). "
        "Highly similar revenue structures. Diverge on local government contract news.",
        "Utilities", "utility"),
    "ENEL/SRG": ("ENEL", "SRG",
        "Enel + Snam — electricity vs gas Italian infrastructure. "
        "Diverge on energy commodity spreads (gas vs power price ratio).",
        "Utilities", "utility"),
}

YF_SUFFIX   = ".MI"
TIMEFRAMES  = {
    "M15": {"interval": "15m", "days": 58,  "bars_per_h": 4},
    "H1":  {"interval": "1h",  "days": 58,  "bars_per_h": 1},
    "D1":  {"interval": "1d",  "days": 365, "bars_per_h": 1/7.5},
}
# Lookback in bars for z-score (all equivalent to ~60 trading hours)
WINDOW = {"M15": 240, "H1": 60, "D1": 8}


# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────
def fetch(sym: str, interval: str, days: int) -> pd.Series | None:
    import yfinance as yf
    end   = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    try:
        raw = yf.download(sym + YF_SUFFIX,
                          start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"),
                          interval=interval, progress=False,
                          auto_adjust=True, actions=False)
    except Exception:
        return None
    if raw is None or len(raw) < 20:
        return None
    raw.index = pd.to_datetime(raw.index, utc=True)
    col = next((c for c in raw.columns if str(c).lower() == "close"), raw.columns[3])
    return raw[col].rename(sym).dropna()


def load_cache(tf: str) -> dict[str, pd.Series]:
    cfg  = TIMEFRAMES[tf]
    syms = list({s for _, (a, b, *_) in PAIRS.items() for s in (a, b)})
    out  = {}
    for sym in syms:
        s = fetch(sym, cfg["interval"], cfg["days"])
        if s is not None:
            out[sym] = s
    return out


# ─────────────────────────────────────────────────────────────────
# Key level detection — SIGNIFICANCE SCORED
# ─────────────────────────────────────────────────────────────────
def find_scored_levels(ratio: np.ndarray, atr: float,
                       order: int = 5, n_top: int = 8) -> pd.DataFrame:
    """
    Find key levels scored by:
      - touch_count  : how many distinct local extrema cluster near this value
      - recency_score: recent touches weighted more (exponential decay)
      - tightness    : how closely clustered the touches are (tighter = stronger)

    Returns DataFrame with columns: level, touch_count, recency_score, score
    """
    n = len(ratio)
    if n < order * 3:
        return pd.DataFrame(columns=["level", "touch_count", "recency_score", "score"])

    hi_idx = argrelextrema(ratio, np.greater_equal, order=order)[0]
    lo_idx = argrelextrema(ratio, np.less_equal,    order=order)[0]
    all_idx = np.concatenate([hi_idx, lo_idx])
    all_val = ratio[all_idx]

    if len(all_val) < 2:
        return pd.DataFrame(columns=["level", "touch_count", "recency_score", "score"])

    # epsilon for clustering = 0.4 × ATR of ratio
    eps = max(atr * 0.4, np.std(ratio) * 0.005)

    rows = []
    used = np.zeros(len(all_val), dtype=bool)

    for i in range(len(all_val)):
        if used[i]:
            continue
        nearby_mask = np.abs(all_val - all_val[i]) <= eps
        cluster_idx = all_idx[nearby_mask]
        cluster_val = all_val[nearby_mask]
        used[nearby_mask] = True

        level      = float(np.median(cluster_val))
        count      = int(np.sum(nearby_mask))
        tightness  = 1.0 / (float(np.std(cluster_val)) / eps + 0.01)

        # Recency: touches near the end of the series get higher weight
        recency    = float(np.sum(np.exp(-3 * (1 - cluster_idx / n))))

        score      = count * tightness * (1 + recency)
        rows.append({"level": level, "touch_count": count,
                     "recency_score": round(recency, 2),
                     "tightness": round(tightness, 2),
                     "score": round(score, 2)})

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    return df.head(n_top).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# Multi-timeframe confluence
# ─────────────────────────────────────────────────────────────────
def find_confluence(levels_m15: pd.DataFrame, levels_h1: pd.DataFrame,
                    ratio_now: float, tol_pct: float = 0.003) -> pd.DataFrame:
    """
    A level is a CONFLUENCE level if it appears on both M15 and H1
    within tol_pct of the current ratio.
    Returns levels with confluence_tf column.
    """
    if levels_m15.empty or levels_h1.empty:
        return pd.DataFrame()

    rows = []
    for _, r_m15 in levels_m15.iterrows():
        lv = r_m15["level"]
        nearby_h1 = levels_h1[np.abs(levels_h1["level"] - lv) / ratio_now <= tol_pct]
        if len(nearby_h1) > 0:
            best_h1 = nearby_h1.iloc[0]
            rows.append({
                "level":        round(lv, 5),
                "dist_pct":     round((lv - ratio_now) / ratio_now * 100, 3),
                "m15_touches":  r_m15["touch_count"],
                "h1_touches":   best_h1["touch_count"],
                "m15_score":    r_m15["score"],
                "h1_score":     best_h1["score"],
                "combined_score": round(r_m15["score"] + best_h1["score"], 1),
                "confluence":   True,
            })
        else:
            rows.append({
                "level":        round(lv, 5),
                "dist_pct":     round((lv - ratio_now) / ratio_now * 100, 3),
                "m15_touches":  r_m15["touch_count"],
                "h1_touches":   0,
                "m15_score":    r_m15["score"],
                "h1_score":     0,
                "combined_score": round(r_m15["score"], 1),
                "confluence":   False,
            })
    df = pd.DataFrame(rows).sort_values("combined_score", ascending=False)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# Helpers: half-life, trend filter, z-score
# ─────────────────────────────────────────────────────────────────
def half_life_hours(ratio: pd.Series, bars_per_h: float) -> float | None:
    """OU half-life in hours."""
    try:
        s   = ratio.dropna().values.astype(float)
        lag = s[:-1]
        delta = np.diff(s)
        beta  = np.polyfit(lag, delta, 1)[0]
        if beta >= 0:
            return None
        hl_bars = -np.log(2) / beta
        return float(hl_bars / bars_per_h)
    except Exception:
        return None


def trend_filter(ratio: pd.Series, window: int = 60) -> tuple[str, float]:
    """
    Fit a linear regression on the last `window` bars of the ratio.
    Normalise slope by ratio std to get a dimensionless trend score.
    |score| < 0.5  -> RANGE  (trade freely)
    0.5 - 1.0      -> WATCH  (proceed with caution)
    |score| > 1.0  -> TREND  (avoid — you'd be fighting trend)
    """
    s = ratio.dropna().values[-window:]
    if len(s) < 20:
        return "?", 0.0
    x = np.arange(len(s), dtype=float)
    slope, _ = np.polyfit(x, s, 1)
    score = slope * len(s) / (np.std(s) + 1e-12)
    if abs(score) < 0.5:
        label = "RANGE"
    elif abs(score) < 1.0:
        label = "WATCH"
    else:
        label = "TREND"
    return label, round(float(score), 2)


def zscore_now(ratio: pd.Series, window: int) -> tuple[float, float, float]:
    """Current z-score, mean, std of the ratio."""
    s   = ratio.dropna()
    mu  = float(s.rolling(window).mean().iloc[-1])
    std = float(s.rolling(window).std().iloc[-1])
    z   = float((s.iloc[-1] - mu) / std) if std > 0 else 0.0
    return round(z, 3), round(mu, 5), round(std, 5)


def coint_pval(sa: pd.Series, sb: pd.Series) -> float:
    try:
        merged = pd.DataFrame({"a": sa, "b": sb}).dropna()
        if len(merged) < 60:
            return 1.0
        _, p, _ = coint(merged["a"], merged["b"])
        return round(float(p), 4)
    except Exception:
        return 1.0


# ─────────────────────────────────────────────────────────────────
# Per-pair analysis
# ─────────────────────────────────────────────────────────────────
def analyse_pair(name: str, sym_a: str, sym_b: str,
                 caches: dict[str, dict]) -> dict | None:
    """Run full analysis for one pair across timeframes."""

    # Need at least M15 and H1
    for tf in ("M15", "H1"):
        if sym_a not in caches[tf] or sym_b not in caches[tf]:
            return None

    result = {"name": name, "sym_a": sym_a, "sym_b": sym_b}

    for tf in ("M15", "H1", "D1"):
        c = caches.get(tf, {})
        if sym_a not in c or sym_b not in c:
            result[tf] = None
            continue
        merged = pd.DataFrame({"a": c[sym_a], "b": c[sym_b]}).dropna()
        if len(merged) < WINDOW[tf] + 10:
            result[tf] = None
            continue

        ratio       = (merged["a"] / merged["b"]).rename("ratio")
        ratio_arr   = ratio.values.astype(float)
        atr         = float(pd.Series(ratio_arr).diff().abs().rolling(WINDOW[tf]).mean().iloc[-1])
        z, mu, std  = zscore_now(ratio, WINDOW[tf])
        trend_lbl, trend_score = trend_filter(ratio, WINDOW[tf])
        hl          = half_life_hours(ratio, TIMEFRAMES[tf]["bars_per_h"])
        levels      = find_scored_levels(ratio_arr, atr)

        result[tf] = {
            "ratio_now":   round(float(ratio.iloc[-1]), 5),
            "z":           z,
            "mu":          mu,
            "std":         std,
            "atr":         round(atr, 6),
            "trend":       trend_lbl,
            "trend_score": trend_score,
            "hl_h":        round(hl, 1) if hl else None,
            "levels":      levels,
            "n_bars":      len(ratio),
        }

    # Cointegration on H1
    result["coint_p"] = coint_pval(
        caches["H1"].get(sym_a, pd.Series()),
        caches["H1"].get(sym_b, pd.Series())
    )

    # Multi-timeframe confluence (M15 vs H1)
    if result["M15"] and result["H1"]:
        result["confluence"] = find_confluence(
            result["M15"]["levels"],
            result["H1"]["levels"],
            result["M15"]["ratio_now"],
        )
    else:
        result["confluence"] = pd.DataFrame()

    return result


# ─────────────────────────────────────────────────────────────────
# Print screener table
# ─────────────────────────────────────────────────────────────────
def print_screener(analyses: list[dict], rationale_map: dict) -> None:
    print(f"\n{'='*90}")
    print(f"  FTSE MIB RATIO KEY-LEVEL SCREENER  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*90}")
    hdr = f"{'Pair':<12} {'Z(M15)':>7} {'Z(H1)':>7} {'Trend':>6} "
    hdr += f"{'HL(h)':>6} {'CointP':>7} {'Nearest level':>14} {'Dist%':>6} "
    hdr += f"{'Touches':>8} {'CONFLUENCE':>11}"
    print(hdr)
    print("-" * 90)

    for a in sorted(analyses, key=_sort_key):
        if a is None:
            continue
        m15 = a.get("M15")
        h1  = a.get("H1")
        if not m15 or not h1:
            continue

        z_m15   = m15["z"]
        z_h1    = h1["z"]
        trend   = m15["trend"]
        hl      = m15["hl_h"] or h1.get("hl_h")
        cp      = a["coint_p"]
        cf      = a["confluence"]

        # Nearest level (M15) above or below
        lv_df   = m15["levels"]
        ratio_n = m15["ratio_now"]
        near    = _nearest_level(lv_df, ratio_n)

        # Confluence flag
        n_conf  = int(cf["confluence"].sum()) if len(cf) > 0 and "confluence" in cf.columns else 0

        trend_color = "!" if trend == "TREND" else ("~" if trend == "WATCH" else " ")
        cf_flag     = f"  {'*' * min(n_conf, 3)} {n_conf} TF-levels" if n_conf > 0 else ""

        line = (f"{a['name']:<12} {z_m15:>+7.2f} {z_h1:>+7.2f} "
                f"{trend_color}{trend:<5} "
                f"{str(round(hl,1)) if hl else '—':>6} "
                f"{cp:>7.4f} "
                f"{near['level']:>14.5f} "
                f"{near['dist_pct']:>+6.2f}% "
                f"{near['touches']:>8} "
                f"{cf_flag}")
        print(line)

    print()
    print("  Trend: RANGE=clean signal  WATCH=caution  TREND=avoid")
    print("  Nearest level: closest key level and distance from current ratio")
    print("  Touches: how many times ratio has respected that level (M15)")
    print("  CONFLUENCE ***: level appears on BOTH M15 and H1 — highest priority")
    print()

    # Multi-pair confirmation
    _print_multipair_confirmation(analyses)


def _sort_key(a: dict) -> tuple:
    if a is None:
        return (99, 99)
    m15 = a.get("M15") or {}
    h1  = a.get("H1")  or {}
    cf  = a.get("confluence", pd.DataFrame())
    n_conf = int(cf["confluence"].sum()) if len(cf) > 0 and "confluence" in cf.columns else 0
    # Priority: confluence count DESC, then abs(z) DESC
    return (-n_conf, -max(abs(m15.get("z", 0)), abs(h1.get("z", 0))))


def _nearest_level(levels_df: pd.DataFrame, ratio_now: float) -> dict:
    if levels_df is None or len(levels_df) == 0:
        return {"level": ratio_now, "dist_pct": 0.0, "touches": 0}
    dists = np.abs(levels_df["level"].values - ratio_now)
    idx   = int(np.argmin(dists))
    row   = levels_df.iloc[idx]
    return {
        "level":    float(row["level"]),
        "dist_pct": float((row["level"] - ratio_now) / ratio_now * 100),
        "touches":  int(row["touch_count"]),
    }


def _print_multipair_confirmation(analyses: list[dict]) -> None:
    """
    If the same stock appears in multiple pairs that are BOTH near a level,
    that stock is the common driver. Flag it.
    """
    near_level: dict[str, list[str]] = {}  # stock -> list of pair names near a level
    for a in analyses:
        if a is None:
            continue
        m15 = a.get("M15")
        if not m15:
            continue
        lv_df = m15["levels"]
        ratio_n = m15["ratio_now"]
        near = _nearest_level(lv_df, ratio_n)
        if abs(near["dist_pct"]) < 1.0:  # within 1% of a key level
            for sym in (a["sym_a"], a["sym_b"]):
                near_level.setdefault(sym, []).append(a["name"])

    multi = {k: v for k, v in near_level.items() if len(v) >= 2}
    if multi:
        print("  MULTI-PAIR CONFIRMATION (same stock near a key level in 2+ pairs):")
        for sym, pairs in sorted(multi.items(), key=lambda x: -len(x[1])):
            print(f"    {sym} is near a key level in: {', '.join(pairs)}")
        print("  -> If these signals align in direction, the shared stock is the driver.")
        print()


# ─────────────────────────────────────────────────────────────────
# Detailed level chart for top pairs
# ─────────────────────────────────────────────────────────────────
def plot_pair_levels(a: dict, out_path: Path) -> None:
    m15 = a.get("M15")
    h1  = a.get("H1")
    cf  = a.get("confluence", pd.DataFrame())
    if not m15 or not h1:
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 2])
    fig.suptitle(f"{a['name']}  —  Key Level Map  ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: M15 ratio + all levels ──
    ax = axes[0]

    # Get ratio from cache using levels
    lv_df_m15 = m15["levels"]
    ratio_now = m15["ratio_now"]
    z_m15     = m15["z"]

    # We only have scalar state here, not the series — show level map as horizontal bars
    mu, std = m15["mu"], m15["std"]
    z_range = np.linspace(-3, 3, 200)
    ratio_range = mu + z_range * std

    ax.set_xlim(-3.2, 3.2)
    ax.axvline(0, color="black", lw=0.8, alpha=0.3)

    # Plot all M15 levels
    if len(lv_df_m15) > 0:
        max_score = lv_df_m15["score"].max()
        for _, row in lv_df_m15.iterrows():
            lv    = row["level"]
            lv_z  = (lv - mu) / std if std > 0 else 0
            alpha = 0.3 + 0.7 * row["score"] / max_score
            width = 0.05 + 0.15 * row["score"] / max_score
            color = "#e74c3c" if lv > ratio_now else "#2ecc71"
            ax.axhline(lv, color=color, lw=width*6, alpha=alpha,
                       label=f"L{int(row['touch_count'])}t" if row["touch_count"] >= 3 else None)
            ax.text(3.0, lv, f" {int(row['touch_count'])}x", va="center",
                    fontsize=7, color=color, alpha=alpha)

    # Confluence levels
    if len(cf) > 0:
        conf_levels = cf[cf["confluence"] == True]["level"]
        for lv in conf_levels:
            ax.axhline(lv, color="#9b59b6", lw=2.5, alpha=0.9, ls="-.")

    # Current ratio marker
    ax.axhline(ratio_now, color="navy", lw=2.0, ls="--", label=f"Now = {ratio_now:.5f}")

    # Z-score bands
    for z_val, label, col in [(1.5, "+1.5z", "#f39c12"), (-1.5, "-1.5z", "#f39c12"),
                               (2.5, "+2.5z", "#e74c3c"), (-2.5, "-2.5z", "#e74c3c")]:
        lv = mu + z_val * std
        ax.axhline(lv, color=col, lw=0.8, ls=":", alpha=0.6, label=label)

    ax.set_ylabel("Ratio (A/B)")
    ax.set_title(f"Key levels on ratio chart  |  M15 Z = {z_m15:+.2f}  |  "
                 f"H1 Z = {h1['z']:+.2f}  |  Trend = {m15['trend']}", fontsize=10)

    # Legend: only unique labels
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=7, loc="upper left")

    # ── Panel 2: confluence table ──
    ax2 = axes[1]
    ax2.axis("off")

    if len(cf) > 0:
        cf_disp = cf.copy()
        cf_disp["MTF"] = cf_disp["confluence"].map({True: "YES ***", False: "—"})
        cf_disp["dist"] = cf_disp["dist_pct"].apply(lambda x: f"{x:+.2f}%")
        cols  = ["level", "dist", "m15_touches", "h1_touches", "combined_score", "MTF"]
        avail = [c for c in cols if c in cf_disp.columns]
        tbl   = cf_disp.head(8)[avail].values.tolist()
        col_labels = [c.replace("_", " ").title() for c in avail]
        t = ax2.table(cellText=tbl, colLabels=col_labels,
                      cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        t.auto_set_font_size(False)
        t.set_fontsize(9)

        # Colour confluence rows
        for row_i, row_data in enumerate(tbl):
            for col_i in range(len(row_data)):
                cell = t[row_i + 1, col_i]
                if "YES" in str(row_data[-1]):
                    cell.set_facecolor("#f0e6ff")
                    cell.set_text_props(fontweight="bold")

        ax2.set_title("Level detail: M15/H1 confluence  (purple = confirmed on both TFs)",
                      fontsize=9, pad=10)
    else:
        ax2.text(0.5, 0.5, "No level data", ha="center", va="center", fontsize=11)

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# HTML screener
# ─────────────────────────────────────────────────────────────────
def build_html(analyses: list[dict], plots_dir: Path, out_path: Path) -> None:
    rows_html = ""
    for a in sorted(analyses, key=_sort_key):
        if a is None:
            continue
        m15 = a.get("M15")
        h1  = a.get("H1")
        if not m15 or not h1:
            continue

        pair_info = PAIRS[a["name"]]; rationale, sector, tier = pair_info[2], pair_info[3], pair_info[4]
        tier_badge = {"core": "#e74c3c", "add": "#3498db", "utility": "#27ae60"}.get(tier, "#95a5a6")
        tier_label = {"core": "Trader's pairs", "add": "Recommended add", "utility": "Utility cluster"}.get(tier, tier)

        z_m15  = m15["z"]
        z_h1   = h1["z"]
        trend  = m15["trend"]
        hl     = m15["hl_h"] or h1.get("hl_h")
        cp     = a["coint_p"]
        cf     = a["confluence"]
        n_conf = int(cf["confluence"].sum()) if len(cf) > 0 and "confluence" in cf.columns else 0

        lv_df    = m15["levels"]
        ratio_n  = m15["ratio_now"]
        near     = _nearest_level(lv_df, ratio_n)

        # Signal logic
        entry_z = 1.5
        if abs(z_m15) >= entry_z and trend == "RANGE":
            sig_color = "#e74c3c" if z_m15 > 0 else "#2ecc71"
            sig_label = ("SHORT A / LONG B" if z_m15 > 0 else "LONG A / SHORT B")
        elif abs(z_m15) >= entry_z and trend != "RANGE":
            sig_color = "#f39c12"
            sig_label = "SIGNAL but TRENDING — wait"
        elif abs(z_m15) >= entry_z - 0.3:
            sig_color = "#f39c12"
            sig_label = "APPROACHING ENTRY"
        else:
            sig_color = "#95a5a6"
            sig_label = "neutral"

        # Z bars
        def z_bar(z: float, max_z: float = 3.0) -> str:
            pct = min(abs(z) / max_z * 100, 100)
            col = "#e74c3c" if z > 0 else "#2ecc71"
            return (f'<div style="display:flex;align-items:center;gap:6px">'
                    f'<div style="background:#eee;border-radius:3px;width:80px;height:12px;overflow:hidden">'
                    f'<div style="background:{col};width:{pct:.0f}%;height:100%"></div></div>'
                    f'<span style="font-weight:bold">{z:+.2f}</span></div>')

        # Trend badge
        trend_col  = {"RANGE": "#27ae60", "WATCH": "#f39c12", "TREND": "#e74c3c"}.get(trend, "#888")
        trend_html = f'<span style="background:{trend_col};color:white;padding:2px 7px;border-radius:4px;font-size:0.85em">{trend}</span>'

        # Confluence stars
        conf_html = ""
        if n_conf > 0:
            stars = "&#9733;" * min(n_conf, 3)
            conf_html = (f'<span style="color:#9b59b6;font-weight:bold;font-size:1.1em" '
                         f'title="{n_conf} levels confirmed on both M15 and H1">'
                         f'{stars} {n_conf} MTF</span>')

        # Best confluence levels (top 3)
        conf_levels_html = ""
        if len(cf) > 0 and "confluence" in cf.columns:
            top_conf = cf[cf["confluence"] == True].head(3)
            for _, row in top_conf.iterrows():
                dist_col = "#e74c3c" if row["dist_pct"] > 0 else "#2ecc71"
                conf_levels_html += (
                    f'<div style="font-size:0.8em;margin:2px 0">'
                    f'<b>{row["level"]:.5f}</b> '
                    f'<span style="color:{dist_col}">{row["dist_pct"]:+.2f}%</span> '
                    f'({int(row["m15_touches"])}M15 + {int(row["h1_touches"])}H1 touches)'
                    f'</div>'
                )

        # Chart link
        chart_name = f"{a['name'].replace('/', '_')}_levels.png"
        chart_path = plots_dir / chart_name
        chart_html = (f'<a href="plots/{chart_name}" target="_blank" '
                      f'style="font-size:0.8em;color:#3498db">view chart</a>'
                      if chart_path.exists() else "")

        rows_html += f"""
        <tr>
          <td>
            <strong style="font-size:1.05em">{a['name']}</strong><br>
            <span style="background:{tier_badge};color:white;padding:1px 6px;
                  border-radius:3px;font-size:0.75em">{tier_label}</span><br>
            <small style="color:#888">{sector}</small>
          </td>
          <td>{z_bar(z_m15)}</td>
          <td>{z_bar(z_h1)}</td>
          <td>{trend_html}</td>
          <td style="text-align:center">
            {f'{hl:.0f}h' if hl else '—'}
            {f'<br><small style="color:#888">({hl/24:.1f}d)</small>' if hl and hl > 24 else ''}
          </td>
          <td style="text-align:center">
            <span style="color:{'#27ae60' if cp < 0.05 else '#f39c12' if cp < 0.15 else '#888'};
                  font-weight:bold">{cp:.4f}</span>
          </td>
          <td>
            <span style="font-size:1.0em;font-weight:bold">{near['level']:.5f}</span>
            <span style="color:{'#e74c3c' if near['dist_pct']>0 else '#2ecc71'}">
              {near['dist_pct']:+.2f}%</span><br>
            <small>{near['touches']} M15 touches</small>
          </td>
          <td>{conf_html}<br>{conf_levels_html}</td>
          <td>
            <span style="background:{sig_color};color:white;padding:4px 8px;
                  border-radius:5px;font-size:0.85em;font-weight:bold">{sig_label}</span>
          </td>
          <td style="font-size:0.78em;color:#555;max-width:220px">{rationale}</td>
          <td>{chart_html}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="300">
  <title>FTSE MIB Ratio Level Screener</title>
  <style>
    body   {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
              background:#f5f6fa;margin:0;padding:20px }}
    h1     {{ color:#2c3e50;margin-bottom:4px }}
    .sub   {{ color:#7f8c8d;margin-bottom:18px;font-size:0.88em }}
    table  {{ border-collapse:collapse;width:100%;background:white;
              border-radius:10px;overflow:hidden;
              box-shadow:0 2px 12px rgba(0,0,0,0.09) }}
    th     {{ background:#2c3e50;color:white;padding:10px 10px;
              text-align:left;font-size:0.82em;white-space:nowrap }}
    td     {{ padding:9px 10px;border-bottom:1px solid #ecf0f1;vertical-align:middle }}
    tr:hover td {{ background:#fafbfc }}
    .legend {{ margin-top:20px;font-size:0.83em;color:#666;line-height:1.7 }}
    .legend b {{ color:#2c3e50 }}
  </style>
</head>
<body>
  <h1>FTSE MIB Ratio Key-Level Screener</h1>
  <div class="sub">
    Window: <b>240 M15 bars = 60 trading hours</b> &nbsp;|&nbsp;
    Entry threshold: <b>&plusmn;1.5 &sigma;</b> &nbsp;|&nbsp;
    Generated: <b>{datetime.now().strftime('%Y-%m-%d %H:%M')}</b>
    &nbsp;(auto-refreshes every 5 min)
  </div>
  <table>
    <thead>
      <tr>
        <th>Pair</th>
        <th>Z-score M15</th>
        <th>Z-score H1</th>
        <th>Trend filter</th>
        <th>Half-life</th>
        <th>Coint p</th>
        <th>Nearest key level</th>
        <th>MTF confluence &#9733;</th>
        <th>Signal</th>
        <th>Why it works</th>
        <th>Chart</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>

  <div class="legend">
    <b>Z-score:</b> how many standard deviations the ratio is from its 60-hour mean.
    Entry signal at &plusmn;1.5&sigma;.<br>
    <b>Trend filter:</b> RANGE = ratio oscillating (trade freely) &nbsp;|&nbsp;
    WATCH = mild trend, proceed carefully &nbsp;|&nbsp;
    TREND = strong trend, avoid mean-reversion entries.<br>
    <b>Half-life:</b> Ornstein-Uhlenbeck estimate — how long before the spread reverts 50%.
    Use this to choose hold duration and stop timing.<br>
    <b>MTF confluence &#9733;:</b> key levels that appear on BOTH M15 AND H1 charts at the same
    price. These are the strongest levels — they've been validated across two independent
    timeframes. More stars = more validated levels present.<br>
    <b>Nearest key level:</b> closest scored S/R level on M15 (distance from current ratio).
    Touch count = how many times the ratio has respected this level historically.<br>
    <b>Colour codes:</b>
    <span style="background:#e74c3c;color:white;padding:1px 6px;border-radius:3px">SHORT</span>
    = ratio above entry, short A long B &nbsp;
    <span style="background:#2ecc71;color:white;padding:1px 6px;border-radius:3px">LONG</span>
    = ratio below entry, long A short B &nbsp;
    <span style="background:#f39c12;color:white;padding:1px 6px;border-radius:3px">APPROACHING</span>
    = within 0.3&sigma; of entry.<br>
    <b>Coint p:</b> Engle-Granger cointegration on H1 data.
    <span style="color:#27ae60">Green &lt;0.05</span> &nbsp;
    <span style="color:#f39c12">Amber &lt;0.15</span> &nbsp;
    <span style="color:#888">Grey = not statistically cointegrated
    (but may still trade well on economic linkage)</span>.
  </div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main(open_browser: bool = False) -> None:
    out   = Path("data/pairs_levels")
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  FTSE MIB Ratio Key-Level Screener")
    print(f"{'='*62}\n")

    # Load data for all timeframes in parallel conceptually (sequential here)
    print("[1/3] Loading M15 + H1 + D1 data...")
    caches: dict[str, dict] = {}
    for tf in ("M15", "H1", "D1"):
        caches[tf] = load_cache(tf)
        print(f"  {tf}: {len(caches[tf])} symbols loaded")

    # Analyse all pairs
    print("\n[2/3] Analysing pairs...")
    analyses = []
    for name, (sym_a, sym_b, rationale, sector, tier) in PAIRS.items():
        a = analyse_pair(name, sym_a, sym_b, caches)
        analyses.append(a)
        if a is None:
            print(f"  {name}: skip (missing data)")
            continue

        m15 = a.get("M15") or {}
        h1  = a.get("H1")  or {}
        cf  = a.get("confluence", pd.DataFrame())
        n_cf = int(cf["confluence"].sum()) if len(cf) > 0 and "confluence" in cf.columns else 0
        hl   = m15.get("hl_h") or h1.get("hl_h")
        print(f"  {name:<12} Z_M15={m15.get('z', 0):+.2f}  "
              f"Z_H1={h1.get('z', 0):+.2f}  "
              f"trend={m15.get('trend','?'):5}  "
              f"HL={str(round(hl,1))+'h' if hl else '—':6}  "
              f"coint_p={a.get('coint_p', 1):.4f}  "
              f"MTF_conf={n_cf}")

    # Print terminal screener
    print_screener(analyses, {k: (v[0], v[1], v[2]) for k, v in PAIRS.items()
                               if k in [a["name"] for a in analyses if a]})

    # Generate charts for all valid pairs
    print("[3/3] Generating charts + HTML screener...")
    for a in analyses:
        if a is None:
            continue
        plot_pair_levels(a, plots / f"{a['name'].replace('/','_')}_levels.png")
        print(f"  chart: {a['name']}")

    # Save level details CSV
    detail_rows = []
    for a in analyses:
        if a is None:
            continue
        cf = a.get("confluence", pd.DataFrame())
        if len(cf) > 0:
            cf["pair"] = a["name"]
            detail_rows.append(cf)
    if detail_rows:
        pd.concat(detail_rows).to_csv(out / "levels_detail.csv", index=False)

    # HTML screener
    build_html(analyses, plots, out / "screener.html")
    print(f"\n[done] Open: {out / 'screener.html'}")

    if open_browser:
        webbrowser.open(str((out / "screener.html").resolve()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", action="store_true", help="Open HTML in browser after run")
    args = parser.parse_args()
    project_root = Path(__file__).parents[3]
    os.chdir(project_root)
    main(open_browser=args.html)
