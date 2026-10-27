"""
Charts for v2 LRC backtest.

Reads existing sweep CSV, re-runs the best combo with full portfolio simulation
at multiple concurrency levels (3 / 5 / 10 / unlimited), then saves a dashboard
PNG to the Desktop.

Run:
    python -m stocks.backtest.charts_v2
"""
import sys, warnings
from pathlib import Path
from datetime import datetime

sys.path.insert(0, r'path/to/repo/hft')
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from stocks.backtest.config import RESULTS_DIR
from stocks.backtest.data.extender import load_bt_bars
from stocks.backtest.data.universe import build_universe
from stocks.backtest.signals.wt_lrc_signal import (
    _lrc_vectorized, _lrc_candle_close_vectorized,
    _wavetrend_vectorized, _atr_vectorized, find_signals,
)

# ── Catppuccin Mocha palette ──────────────────────────────────────────────────
CRUST   = "#11111b"; BASE    = "#1e1e2e"; MANTLE  = "#181825"
SURFACE0= "#313244"; SURFACE1= "#45475a"; OVERLAY0= "#6c7086"
TEXT    = "#cdd6f4"; SUBTEXT = "#a6adc8"
LAVENDER= "#b4befe"; BLUE    = "#89b4fa"; TEAL    = "#94e2d5"
GREEN   = "#a6e3a1"; YELLOW  = "#f9e2af"; PEACH   = "#fab387"
RED     = "#f38ba8"; MAUVE   = "#cba6f7"; MAROON  = "#eba0ac"

US_EXCHANGES = {"NYSE", "NAS"}
MIN_BARS     = 200
LRC_CANDLE_LEN = 15; LRC_CANDLE_SMOOTH = 9
WT_N1 = 10; WT_N2 = 21; ATR_P = 14


# ── Data loading ──────────────────────────────────────────────────────────────

def load_us_liquid():
    univ = build_universe()
    us   = univ[univ["exchange"].isin(US_EXCHANGES)]
    bars = {}; dvols = {}
    for _, row in us.iterrows():
        sym = row["symbol"]
        df  = load_bt_bars(sym)
        if df.empty or len(df) < MIN_BARS: continue
        df  = df[["Open","High","Low","Close","Volume"]].copy()
        df.columns = ["open","high","low","close","volume"]
        dvol = (df["close"] * df["volume"]).replace(0, np.nan).dropna()
        if len(dvol) >= MIN_BARS:
            dvols[sym] = float(dvol.median()); bars[sym] = df
    if not dvols: return {}
    s = pd.Series(dvols); thresh = float(s.quantile(0.75))
    liquid = set(s[s >= thresh].index)
    return {sym: bars[sym] for sym in liquid if sym in bars}


def precompute_symbol(df, ll, sm):
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    mid, upper, lower, _ = _lrc_vectorized(c, ll, sm)
    return {
        "close": c, "high": h, "low": l, "open": df["open"].values,
        "n": len(c), "index": df.index,
        "lrc_c": _lrc_candle_close_vectorized(c, LRC_CANDLE_LEN, LRC_CANDLE_SMOOTH),
        "wt1":   _wavetrend_vectorized(c, h, l, WT_N1, WT_N2)[0],
        "atr":   _atr_vectorized(h, l, c, ATR_P),
        "lrc_data": {(ll, sm): (mid, upper, lower)},
    }


# ── Portfolio simulation with concurrency cap ─────────────────────────────────

def _find_exit(pc, entry_bar, entry_px, sl, tp, direction):
    h = pc["high"]; l = pc["low"]; n = pc["n"]
    for i in range(entry_bar, n):
        sl_hit = (l[i] <= sl) if direction == "LONG" else (h[i] >= sl)
        tp_hit = (h[i] >= tp) if direction == "LONG" else (l[i] <= tp)
        if sl_hit and tp_hit: return i, sl, "SL"
        if sl_hit:            return i, sl, "SL"
        if tp_hit:            return i, tp, "TP"
    return n - 1, float(pc["close"][n - 1]), "END"


def simulate_portfolio(precomp_all, signals_all, max_concurrent):
    """
    Portfolio-level simulation.
    signals_all: {sym: [(bar_idx, direction, sl, tp_rr, close, readiness), ...]}
    max_concurrent: int or None (unlimited)
    Returns list of trade dicts with dates.
    """
    candidates = []
    for sym, sigs in signals_all.items():
        pc = precomp_all[sym]; n = pc["n"]
        for sig in sigs:
            bi, direction, sl_raw, tp_rr_s, _, readiness = sig
            if bi + 1 >= n: continue
            entry_bar = bi + 1
            entry_px  = float(pc["open"][entry_bar])
            risk      = abs(entry_px - sl_raw)
            if risk < 1e-6: continue
            pos  = +1 if direction == "SHORT" else -1
            tp_px = entry_px - pos * tp_rr_s * risk
            exit_bar, exit_px, result = _find_exit(pc, entry_bar, entry_px,
                                                    sl_raw, tp_px, direction)
            r = ((exit_px - entry_px) / risk if direction == "LONG"
                 else (entry_px - exit_px) / risk)
            candidates.append({
                "sym":        sym,
                "entry_date": pc["index"][entry_bar],
                "exit_date":  pc["index"][exit_bar],
                "direction":  direction,
                "entry":      entry_px,
                "sl":         sl_raw,
                "tp":         tp_px,
                "exit":       exit_px,
                "result":     result,
                "r":          round(r, 3),
                "readiness":  readiness,
            })

    if not candidates:
        return []

    candidates.sort(key=lambda t: (t["entry_date"], -t["readiness"]))

    if max_concurrent is None:
        return candidates

    accepted  = []
    open_pos  = []   # (exit_date, sym)

    for t in candidates:
        ed = t["entry_date"]
        open_pos = [(xd, s) for xd, s in open_pos if xd > ed]
        if len(open_pos) >= max_concurrent: continue
        if t["sym"] in {s for _, s in open_pos}:  continue
        accepted.append(t)
        open_pos.append((t["exit_date"], t["sym"]))

    return accepted


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _equity_curve(trades):
    if not trades: return pd.Series(dtype=float)
    df = pd.DataFrame(trades).sort_values("exit_date")
    df["equity"] = df["r"].cumsum()
    return df.set_index("exit_date")["equity"]


def _monthly_returns(trades):
    if not trades: return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["year"]  = df["exit_date"].dt.year
    df["month"] = df["exit_date"].dt.month
    return df.groupby(["year","month"])["r"].sum().unstack(fill_value=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sweep_csv = sorted(RESULTS_DIR.glob("v2_sweep_*.csv"))[-1]
    print(f"Loading sweep: {sweep_csv.name}")
    df_sweep = pd.read_csv(sweep_csv).sort_values("sharpe_r", ascending=False)
    best = df_sweep.iloc[0]
    print(f"Best combo: {dict(best[['lrc_length','sigma_mult','wt_ob',\
'touch_pct','sl_atr_mult','tp_rr','sharpe_r','win_rate','n_trades']])}")

    ll  = int(best["lrc_length"]); sm  = float(best["sigma_mult"])
    wob = float(best["wt_ob"]);    tp  = float(best["touch_pct"])
    sl  = float(best["sl_atr_mult"]); rr = float(best["tp_rr"])

    print("Loading bars…")
    bars = load_us_liquid()
    print(f"Precomputing {len(bars)} symbols…")
    precomp = {}
    for sym, d in bars.items():
        try: precomp[sym] = precompute_symbol(d, ll, sm)
        except Exception: pass

    print("Finding signals…")
    signals_all = {}
    for sym, pc in precomp.items():
        sigs = find_signals(pc, ll, sm, wob, tp, sl, rr)
        if sigs: signals_all[sym] = sigs

    print("Simulating portfolio at concurrency 3 / 5 / 10 / unlimited…")
    concurrencies = [3, 5, 10, None]
    labels        = ["Max 3", "Max 5", "Max 10", "Unlimited"]
    colors        = [RED, PEACH, YELLOW, GREEN]
    trades_by_conc = {c: simulate_portfolio(precomp, signals_all, c)
                      for c in concurrencies}

    # ── Build figure ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": BASE,  "axes.facecolor":  MANTLE,
        "axes.edgecolor":   SURFACE1, "axes.labelcolor": SUBTEXT,
        "xtick.color": SUBTEXT, "ytick.color": SUBTEXT,
        "text.color":  TEXT,    "grid.color": SURFACE0,
        "grid.linewidth": 0.5,  "font.family": "monospace",
        "font.size": 9,
    })

    fig = plt.figure(figsize=(20, 14), facecolor=BASE)
    fig.suptitle(
        f"LRC v2 Strategy  ·  lrc={ll}  σ={sm}  WT_OB={wob}  "
        f"touch={tp}%  SL=×{sl}ATR  TP={rr}R  ·  US Liquid ({len(bars)} symbols)",
        color=LAVENDER, fontsize=13, fontweight="bold", y=0.98,
    )

    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35,
                  left=0.06, right=0.97, top=0.93, bottom=0.06)

    # ── Row 0: Equity curves ──────────────────────────────────────────────────
    ax_eq = fig.add_subplot(gs[0, :])
    ax_eq.set_title("Cumulative R — portfolio concurrency comparison", color=TEXT)
    ax_eq.axhline(0, color=OVERLAY0, linewidth=0.7, linestyle="--")
    ax_eq.set_xlabel("Date"); ax_eq.set_ylabel("Cumulative R")
    for conc, label, col in zip(concurrencies, labels, colors):
        tr = trades_by_conc[conc]
        if not tr: continue
        eq = _equity_curve(tr)
        n  = len(tr)
        wr = sum(1 for t in tr if t["r"] > 0) / n * 100 if n else 0
        ax_eq.plot(eq.index, eq.values, color=col, linewidth=1.5,
                   label=f"{label}  ({n} trades, {wr:.0f}% WR)")
    ax_eq.legend(loc="upper left", facecolor=SURFACE0, edgecolor=SURFACE1,
                 labelcolor=TEXT, fontsize=8)
    ax_eq.grid(True, axis="y")

    # ── Row 1 left: Monthly returns heatmap (unlimited) ───────────────────────
    ax_heat = fig.add_subplot(gs[1, :2])
    ax_heat.set_title("Monthly R  (unlimited concurrency)", color=TEXT)
    mon = _monthly_returns(trades_by_conc[None])
    if not mon.empty:
        vmax = max(abs(mon.values.max()), abs(mon.values.min()), 1)
        cmap = LinearSegmentedColormap.from_list(
            "rg", [RED, BASE, GREEN], N=256)
        im = ax_heat.imshow(mon.values, cmap=cmap, vmin=-vmax, vmax=vmax,
                            aspect="auto", interpolation="nearest")
        ax_heat.set_xticks(range(12))
        ax_heat.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"],
                                 color=SUBTEXT, fontsize=8)
        ax_heat.set_yticks(range(len(mon.index)))
        ax_heat.set_yticklabels(mon.index.tolist(), color=SUBTEXT, fontsize=8)
        for r in range(len(mon.index)):
            for c in range(12):
                val = mon.values[r, c]
                if val != 0:
                    ax_heat.text(c, r, f"{val:.0f}", ha="center", va="center",
                                 fontsize=6, color=TEXT if abs(val) < vmax * 0.7 else CRUST)
        plt.colorbar(im, ax=ax_heat, label="Monthly R", shrink=0.85)

    # ── Row 1 right: R distribution ───────────────────────────────────────────
    ax_rdist = fig.add_subplot(gs[1, 2])
    ax_rdist.set_title("R distribution  (unlimited)", color=TEXT)
    tr_unlim = trades_by_conc[None]
    if tr_unlim:
        r_vals = np.array([t["r"] for t in tr_unlim])
        ax_rdist.axvline(0, color=OVERLAY0, linewidth=0.8, linestyle="--")
        ax_rdist.axvline(r_vals.mean(), color=YELLOW, linewidth=1.2,
                         linestyle="--", label=f"Mean {r_vals.mean():.2f}R")
        bins = np.linspace(max(-2, r_vals.min()), min(rr + 0.5, r_vals.max()), 40)
        ax_rdist.hist(r_vals[r_vals < 0], bins=bins[bins < 0],
                      color=RED, alpha=0.7, edgecolor=MAROON)
        ax_rdist.hist(r_vals[r_vals >= 0], bins=bins[bins >= 0],
                      color=GREEN, alpha=0.7, edgecolor=TEAL)
        ax_rdist.legend(facecolor=SURFACE0, edgecolor=SURFACE1,
                        labelcolor=TEXT, fontsize=8)
        ax_rdist.set_xlabel("R multiple"); ax_rdist.set_ylabel("Count")

    # ── Row 2 left: Sharpe heatmap (sigma_mult × wt_ob) ──────────────────────
    ax_sh = fig.add_subplot(gs[2, 0])
    ax_sh.set_title("Sharpe: σ_mult × WT_OB\n(best touch/sl/tp per cell)", color=TEXT)
    pivot = (df_sweep.groupby(["sigma_mult","wt_ob"])["sharpe_r"]
                     .max().unstack(fill_value=0))
    im2 = ax_sh.imshow(pivot.values, cmap="plasma", aspect="auto",
                       interpolation="nearest")
    ax_sh.set_xticks(range(len(pivot.columns)))
    ax_sh.set_xticklabels(pivot.columns.tolist(), color=SUBTEXT, fontsize=8)
    ax_sh.set_yticks(range(len(pivot.index)))
    ax_sh.set_yticklabels(pivot.index.tolist(), color=SUBTEXT, fontsize=8)
    ax_sh.set_xlabel("WT_OB"); ax_sh.set_ylabel("sigma_mult")
    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            ax_sh.text(c, r, f"{pivot.values[r,c]:.1f}", ha="center",
                       va="center", fontsize=7, color=TEXT)
    plt.colorbar(im2, ax=ax_sh, shrink=0.85, label="Sharpe")

    # ── Row 2 mid: Sharpe by TP_RR × SL_ATR ─────────────────────────────────
    ax_tp = fig.add_subplot(gs[2, 1])
    ax_tp.set_title("Sharpe: TP_RR × SL_ATR_MULT", color=TEXT)
    pivot2 = (df_sweep.groupby(["tp_rr","sl_atr_mult"])["sharpe_r"]
                      .max().unstack(fill_value=0))
    im3 = ax_tp.imshow(pivot2.values, cmap="plasma", aspect="auto",
                        interpolation="nearest")
    ax_tp.set_xticks(range(len(pivot2.columns)))
    ax_tp.set_xticklabels(pivot2.columns.tolist(), color=SUBTEXT, fontsize=8)
    ax_tp.set_yticks(range(len(pivot2.index)))
    ax_tp.set_yticklabels(pivot2.index.tolist(), color=SUBTEXT, fontsize=8)
    ax_tp.set_xlabel("SL mult"); ax_tp.set_ylabel("TP RR")
    for r in range(len(pivot2.index)):
        for c in range(len(pivot2.columns)):
            ax_tp.text(c, r, f"{pivot2.values[r,c]:.1f}", ha="center",
                       va="center", fontsize=7, color=TEXT)
    plt.colorbar(im3, ax=ax_tp, shrink=0.85, label="Sharpe")

    # ── Row 2 right: trade count + WR by concurrency ─────────────────────────
    ax_bar = fig.add_subplot(gs[2, 2])
    ax_bar.set_title("Trades & Win Rate by concurrency", color=TEXT)
    labs  = [l for l, c in zip(labels, concurrencies)]
    ns    = [len(trades_by_conc[c]) for c in concurrencies]
    wrs   = [(sum(1 for t in trades_by_conc[c] if t["r"] > 0) /
              max(1, len(trades_by_conc[c])) * 100) for c in concurrencies]
    x     = np.arange(len(labs))
    bars1 = ax_bar.bar(x - 0.2, ns,  0.35, color=BLUE,   label="# Trades",   alpha=0.8)
    ax_r  = ax_bar.twinx()
    ax_r.plot(x, wrs, "o--", color=GREEN, linewidth=1.5, label="Win %")
    ax_r.set_ylabel("Win rate %", color=GREEN); ax_r.tick_params(colors=GREEN)
    ax_r.set_ylim(30, 60)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(labs, color=SUBTEXT, fontsize=8)
    ax_bar.set_ylabel("Trade count", color=BLUE); ax_bar.tick_params(colors=BLUE)
    lines1, lbls1 = ax_bar.get_legend_handles_labels()
    lines2, lbls2 = ax_r.get_legend_handles_labels()
    ax_bar.legend(lines1 + lines2, lbls1 + lbls2, facecolor=SURFACE0,
                  edgecolor=SURFACE1, labelcolor=TEXT, fontsize=7, loc="upper left")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = Path.home() / "Desktop" / "LRC_v2_Backtest.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BASE)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
