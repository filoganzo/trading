"""
pm_report.py

Fixed-params portfolio backtest (sigma=1.0, M15, 1R, no filters) across ALL live symbols.
Produces:
  1. Desktop PNG — PM-grade report with equity curve + MC fan, monthly heatmap,
     drawdown, R-distribution, MC final-PnL distribution, key-metrics table
  2. Console summary

Usage:
    python -m stocks.backtest.pm_report
    python -m stocks.backtest.pm_report --mc 2000 --out ~/Desktop/my_report.png
"""
import argparse
import io
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

# UTF-8 stdout fix
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stocks.backtest.runner_v11 import (
    precompute_symbol_v11, _find_signals_v11, _find_exit, load_spx_rets,
)
from stocks.backtest.data.extender import load_bt_bars, load_bt_bars_m15, load_bt_bars_w1

# ── Fixed universal params ────────────────────────────────────────────────────
SIGMA        = 1.0
ENTRY_TF     = "M15"
TP_RR        = 1.0
REL_THRESH   = 0.0
ATR_CONTRACT = 1.0

# ── Best fit-all split params (from fit_all_sweep) ───────────────────────────
LONG_DEFAULTS  = dict(sigma=2.0, entry_tf="M30", tp_rr=1.0, rel_thresh=-0.05)
SHORT_DEFAULTS = dict(sigma=1.5, entry_tf="M30", tp_rr=2.0, rel_thresh=-0.05)

LRC_LENGTH    = 80
HTF_TOUCH_PCT = 0.5
HTF_FRAC      = HTF_TOUCH_PCT / 100.0
MIN_HTF_SCORE = 60
SWING_WING    = 2
ENTRY_TOUCH   = 0.005
SL_MULT       = 1.0
BREAKEVEN_RR  = 0.0
CAPITAL       = 1_000.0
LEVERAGE      = 20
EFFECTIVE     = CAPITAL * LEVERAGE
MAX_CONCURRENT = 8

_RESULTS = Path(__file__).parent.parent / "data" / "backtest_results"
_DESKTOP = Path.home() / "Desktop"


# ── Data helpers ─────────────────────────────────────────────────────────────

def _normalize(df):
    cap = ["Open","High","Low","Close","Volume"]
    low = ["open","high","low","close","volume"]
    if cap[0] in df.columns:
        df = df[cap].copy(); df.columns = low
    return df


def _backtest_symbol(sym, exchange, sector, spx_rets,
                     sigma=None, entry_tf=None, tp_rr_override=None, rel_thresh=None):
    sigma      = sigma      if sigma      is not None else SIGMA
    entry_tf   = entry_tf   if entry_tf   is not None else ENTRY_TF
    tp_rr_use  = tp_rr_override if tp_rr_override is not None else TP_RR
    rel_thresh = rel_thresh if rel_thresh is not None else REL_THRESH

    try:
        daily = _normalize(load_bt_bars(sym))
        m15   = load_bt_bars_m15(sym)
        w1    = load_bt_bars_w1(sym)
        if daily is None or len(daily) < LRC_LENGTH+20: return []
        if m15 is None or m15.empty: return []
        if w1  is None or w1.empty or len(w1) < 20: return []
    except Exception:
        return []
    try:
        pc = precompute_symbol_v11(
            df=daily, m15_df=m15, w1_df=w1,
            lrc_lengths=[LRC_LENGTH], sigma_mults=[sigma],
            htf_touch_pcts=[HTF_TOUCH_PCT], swing_wings=[SWING_WING],
        )
    except Exception:
        return []
    try:
        sigs = _find_signals_v11(
            pc=pc, lrc_length=LRC_LENGTH, sigma_mult=sigma,
            htf_tf=HTF_FRAC, entry_touch_tf=ENTRY_TOUCH,
            min_htf_score=MIN_HTF_SCORE, swing_wing=SWING_WING,
            entry_tf_str=entry_tf, rel_perf_thresh=rel_thresh,
            atr_contract_ratio=ATR_CONTRACT, spx_ret5_map=spx_rets,
        )
    except Exception:
        return []
    if not sigs: return []

    n=pc["n"]; c=pc["close"]; h=pc["high"]; l=pc["low"]; idx=pc["index"]
    _, upper_arr, lower_arr = pc["lrc_data"].get((LRC_LENGTH, sigma),(None,None,None))

    trades = []
    for (si, ei, ep, direction, lv, atr_v, htf_score) in sigs:
        pos   = +1 if direction=="SHORT" else -1
        sl_px = lv + pos * SL_MULT * atr_v
        risk  = abs(ep - sl_px)
        if risk < 1e-6 or pos*(sl_px-ep) <= 0: continue
        tp_px = ep - pos * tp_rr_use * risk
        dr    = (EFFECTIVE/ep)*risk

        xb, xp, xtype, mfe_r, mae_r = _find_exit(
            h, l, c, ei, ep, sl_px, tp_px, direction, BREAKEVEN_RR, risk, n)

        r   = (xp-ep)/risk if direction=="LONG" else (ep-xp)/risk
        pnl = r * dr

        ci = float(c[si])
        ui = float(upper_arr[si]) if upper_arr is not None else float("nan")
        li = float(lower_arr[si]) if lower_arr is not None else float("nan")
        pct_ext = ((ci-ui)/ui*100) if direction=="SHORT" else ((li-ci)/li*100)

        trades.append({
            "sym": sym, "exchange": exchange, "sector": sector,
            "direction": direction,
            "signal_date": pd.Timestamp(idx[si]).date(),
            "entry_date":  pd.Timestamp(idx[ei]).date(),
            "exit_date":   pd.Timestamp(idx[xb]).date(),
            "entry_px": round(float(ep),5), "exit_px": round(float(xp),5),
            "sl_px": round(float(sl_px),5), "tp_px": round(float(tp_px),5),
            "level_px": round(float(lv),5),
            "r": round(r,3), "pnl_usd": round(pnl,2),
            "exit_type": xtype, "mfe_r": mfe_r, "mae_r": mae_r,
            "atr_v": round(float(atr_v),5), "htf_score": htf_score,
            "pct_ext": round(pct_ext,3),
        })
    return trades


# ── Portfolio construction with MC cap ───────────────────────────────────────

def build_portfolio(trades_df):
    df = trades_df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    df.sort_values("entry_date", inplace=True)

    accepted = []; open_pos = []
    for _, row in df.iterrows():
        ed = row["entry_date"]
        open_pos = [(xd,s) for xd,s in open_pos if xd > ed]
        if len(open_pos) >= MAX_CONCURRENT: continue
        accepted.append(row)
        open_pos.append((row["exit_date"], row["sym"]))
    return pd.DataFrame(accepted).reset_index(drop=True)


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _sharpe(r_series):
    s = r_series.std()
    return float(r_series.mean()/s * np.sqrt(252)) if len(r_series)>1 and s>0 else 0.0

def _max_dd(pnl_series):
    eq = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(eq)
    return float((peak - eq).max())

def _cagr(pnl_series, entry_dates):
    total = float(np.sum(pnl_series))
    span  = (entry_dates.max() - entry_dates.min()).days / 365.25
    if span <= 0: return 0.0
    start_eq = EFFECTIVE
    return float((1 + total/start_eq)**(1/span) - 1) * 100


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def run_montecarlo(pnl_series, n_sims=1000):
    """Resample actual dollar PnL per trade — no approximation."""
    n   = len(pnl_series)
    arr = pnl_series.values  # actual dollar PnL per trade

    paths = np.zeros((n_sims, n+1))
    for s in range(n_sims):
        sample = np.random.choice(arr, size=n, replace=True)
        paths[s] = np.concatenate([[0], np.cumsum(sample)])

    pcts = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)

    final_vals = paths[:, -1]
    max_dds    = np.array([_max_dd(paths[s, 1:] - paths[s, :-1]) for s in range(n_sims)])
    # Sharpe on R still meaningful — use R proxy: pnl / mean(|pnl|)
    mean_abs = np.mean(np.abs(arr))
    sharpes  = np.array([
        _sharpe(pd.Series(paths[s,1:]-paths[s,:-1]) / mean_abs)
        for s in range(n_sims)
    ])

    return pcts, final_vals, max_dds, sharpes


# ── PNG report ────────────────────────────────────────────────────────────────

DARK = "#0d1117"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED   = "#f85149"
GOLD  = "#d29922"
MUTED = "#8b949e"
WHITE = "#e6edf3"


def generate_png(port_df, all_df, mc_pcts, mc_final, mc_dds, mc_sharpes, out_path,
                 params_label: str = "sigma=1.0 | M15 | 1R TP | no filters"):
    port_df = port_df.copy()
    port_df["exit_date"] = pd.to_datetime(port_df["exit_date"])
    port_df["entry_date"] = pd.to_datetime(port_df["entry_date"])
    port_df.sort_values("exit_date", inplace=True)

    r_arr  = port_df["r"].values
    pnl_arr = port_df["pnl_usd"].values
    equity  = np.cumsum(pnl_arr)
    peak    = np.maximum.accumulate(equity)
    dd_arr  = peak - equity
    n_trades = len(port_df)

    # Monthly returns
    port_df["month"] = port_df["exit_date"].dt.to_period("M")
    monthly_pnl = port_df.groupby("month")["pnl_usd"].sum()
    monthly_r   = port_df.groupby("month")["r"].mean()

    # Build year×month matrix
    months_idx = monthly_pnl.index
    years  = sorted(set(m.year for m in months_idx))
    grid   = pd.DataFrame(index=years, columns=range(1,13), dtype=float)
    for m, v in monthly_pnl.items():
        grid.loc[m.year, m.month] = v
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 24), facecolor=DARK)
    fig.patch.set_facecolor(DARK)

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97, top=0.93, bottom=0.05,
    )

    ax_eq   = fig.add_subplot(gs[0, :2])   # equity + MC fan (wide)
    ax_heat = fig.add_subplot(gs[0, 2])    # monthly heatmap
    ax_dd   = fig.add_subplot(gs[1, :2])   # drawdown
    ax_rdist= fig.add_subplot(gs[1, 2])    # R distribution
    ax_mc   = fig.add_subplot(gs[2, :2])   # MC final PnL distribution
    ax_tab  = fig.add_subplot(gs[2, 2])    # stats table
    ax_sec  = fig.add_subplot(gs[3, :])    # sector/exchange breakdown

    def _style(ax, title=""):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        if title:
            ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=6)
        ax.title.set_color(WHITE)

    trades_x = np.arange(n_trades)

    # ── 1. Equity curve + MC fan ──────────────────────────────────────────────
    _style(ax_eq, f"Equity Curve  ({params_label})")
    mc_x = np.arange(mc_pcts.shape[1])
    ax_eq.fill_between(mc_x, mc_pcts[0], mc_pcts[4], alpha=0.12, color=ACCENT, label="MC 5-95%")
    ax_eq.fill_between(mc_x, mc_pcts[1], mc_pcts[3], alpha=0.20, color=ACCENT, label="MC 25-75%")
    ax_eq.plot(mc_x, mc_pcts[2], color=ACCENT, lw=1.2, ls="--", alpha=0.6, label="MC median")
    ax_eq.plot(trades_x+1, equity, color=GREEN, lw=2.2, label="Actual equity")
    ax_eq.axhline(0, color=MUTED, lw=0.6, ls=":")
    ax_eq.set_xlabel("Trade #", color=MUTED, fontsize=8)
    ax_eq.set_ylabel("Cumulative PnL ($)", color=MUTED, fontsize=8)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax_eq.legend(fontsize=7, facecolor="#161b22", labelcolor=WHITE, framealpha=0.8)

    # ── 2. Monthly returns heatmap ────────────────────────────────────────────
    _style(ax_heat, "Monthly PnL ($)")
    grid_vals = grid.values.astype(float)
    vmax = np.nanmax(np.abs(grid_vals)) if not np.all(np.isnan(grid_vals)) else 1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax_heat.imshow(grid_vals, cmap="RdYlGn", norm=norm, aspect="auto")
    ax_heat.set_xticks(range(12)); ax_heat.set_xticklabels(month_labels, fontsize=6, color=MUTED)
    ax_heat.set_yticks(range(len(years))); ax_heat.set_yticklabels(years, fontsize=7, color=MUTED)
    for yi, yr in enumerate(years):
        for mi in range(12):
            v = grid_vals[yi, mi]
            if not np.isnan(v):
                ax_heat.text(mi, yi, f"{v:,.0f}", ha="center", va="center",
                             fontsize=5, color="white" if abs(v) > vmax*0.5 else "black")
    plt.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02).ax.tick_params(colors=MUTED, labelsize=6)

    # ── 3. Drawdown ───────────────────────────────────────────────────────────
    _style(ax_dd, "Drawdown (USD)")
    ax_dd.fill_between(trades_x+1, -dd_arr, 0, color=RED, alpha=0.55)
    ax_dd.plot(trades_x+1, -dd_arr, color=RED, lw=1)
    ax_dd.axhline(0, color=MUTED, lw=0.6)
    ax_dd.set_xlabel("Trade #", color=MUTED, fontsize=8)
    ax_dd.set_ylabel("Drawdown ($)", color=MUTED, fontsize=8)
    ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))

    # ── 4. R distribution ────────────────────────────────────────────────────
    _style(ax_rdist, "R Multiple Distribution")
    wins  = r_arr[r_arr > 0]
    losses= r_arr[r_arr <= 0]
    bins = np.linspace(r_arr.min()-0.1, r_arr.max()+0.1, 40)
    ax_rdist.hist(losses, bins=bins, color=RED,   alpha=0.75, label=f"Loss ({len(losses)})")
    ax_rdist.hist(wins,   bins=bins, color=GREEN, alpha=0.75, label=f"Win ({len(wins)})")
    ax_rdist.axvline(0, color=MUTED, lw=0.8)
    ax_rdist.axvline(np.mean(r_arr), color=GOLD, lw=1.5, ls="--",
                     label=f"Mean={np.mean(r_arr):.2f}R")
    ax_rdist.set_xlabel("R multiple", color=MUTED, fontsize=8)
    ax_rdist.legend(fontsize=7, facecolor="#161b22", labelcolor=WHITE, framealpha=0.8)

    # ── 5. MC final PnL distribution ─────────────────────────────────────────
    _style(ax_mc, f"Monte Carlo Final PnL Distribution (n=1,000 sims, {n_trades} trades resampled)")
    p5, p25, p50, p75, p95 = np.percentile(mc_final, [5,25,50,75,95])
    ax_mc.hist(mc_final, bins=60, color=ACCENT, alpha=0.7, edgecolor="none")
    for pv, label, col in [(p5,"5%",RED),(p25,"25%",GOLD),(p50,"50%",GREEN),(p75,"75%",GOLD),(p95,"95%",RED)]:
        ax_mc.axvline(pv, color=col, lw=1.4, ls="--",
                      label=f"{label}: ${pv:,.0f}")
    actual_final = equity[-1]
    ax_mc.axvline(actual_final, color=WHITE, lw=2, label=f"Actual: ${actual_final:,.0f}")
    ax_mc.set_xlabel("Final PnL ($)", color=MUTED, fontsize=8)
    ax_mc.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax_mc.legend(fontsize=7, facecolor="#161b22", labelcolor=WHITE, framealpha=0.8, ncol=2)

    # ── 6. Key metrics table ──────────────────────────────────────────────────
    _style(ax_tab, "Key Metrics")
    ax_tab.axis("off")
    wr      = float((r_arr > 0).mean() * 100)
    sharpe  = _sharpe(pd.Series(r_arr))
    max_dd  = float(dd_arr.max())
    avg_r   = float(np.mean(r_arr))
    tp_rate = float((port_df["exit_type"]=="TP").mean()*100)
    sl_rate = float((port_df["exit_type"]=="SL").mean()*100)
    calmar  = abs(equity[-1] / max_dd) if max_dd > 0 else 0
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses)>0 else 999
    n_long  = int((port_df["direction"]=="LONG").sum())
    n_short = int((port_df["direction"]=="SHORT").sum())
    date_range = f"{port_df['entry_date'].min().date()} to {port_df['entry_date'].max().date()}"
    n_syms  = port_df["sym"].nunique()

    mc_wr50 = float(np.percentile(mc_final, 50))
    mc_dd50 = float(np.percentile(mc_dds, 50))
    mc_sh50 = float(np.percentile(mc_sharpes, 50))

    rows = [
        ("Trades (portfolio)",    f"{n_trades}"),
        ("Symbols",               f"{n_syms}"),
        ("Date range",            date_range),
        ("", ""),
        ("Win Rate",              f"{wr:.1f}%"),
        ("Avg R",                 f"{avg_r:.3f}"),
        ("Profit Factor",         f"{profit_factor:.2f}"),
        ("Sharpe (R-series)",     f"{sharpe:.2f}"),
        ("Calmar Ratio",          f"{calmar:.2f}"),
        ("", ""),
        ("Total PnL",             f"${equity[-1]:,.0f}"),
        ("Max Drawdown",          f"${max_dd:,.0f}"),
        ("TP / SL rate",          f"{tp_rate:.0f}% / {sl_rate:.0f}%"),
        ("LONG / SHORT",          f"{n_long} / {n_short}"),
        ("", ""),
        ("MC median final PnL",   f"${mc_wr50:,.0f}"),
        ("MC median max DD",      f"${mc_dd50:,.0f}"),
        ("MC median Sharpe",      f"{mc_sh50:.2f}"),
        ("MC 5th pct PnL",        f"${p5:,.0f}"),
        ("MC 95th pct PnL",       f"${p95:,.0f}"),
    ]
    y = 0.98
    for label, val in rows:
        if label == "":
            y -= 0.03; continue
        col = MUTED if not val else WHITE
        ax_tab.text(0.02, y, label, transform=ax_tab.transAxes,
                    fontsize=8, color=MUTED, va="top")
        color = WHITE
        if any(kw in label for kw in ["PnL","Total","final"]):
            color = GREEN if (val.startswith("$") and not val.startswith("$-")) else RED
        elif "Drawdown" in label or "DD" in label:
            color = RED
        ax_tab.text(0.98, y, val, transform=ax_tab.transAxes,
                    fontsize=8, color=color, va="top", ha="right", fontweight="bold")
        y -= 0.047

    # ── 7. Sector breakdown ───────────────────────────────────────────────────
    _style(ax_sec, "PnL by Sector  (bar) and Exchange  (right)")
    if "sector" in port_df.columns and port_df["sector"].notna().any():
        sec_pnl = port_df.groupby("sector")["pnl_usd"].sum().sort_values()
        colors_sec = [GREEN if v >= 0 else RED for v in sec_pnl.values]
        bars = ax_sec.barh(sec_pnl.index, sec_pnl.values, color=colors_sec, alpha=0.8)
        ax_sec.set_xlabel("Total PnL ($)", color=MUTED, fontsize=8)
        ax_sec.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        # Overlay exchange breakdown as inset
        if "exchange" in port_df.columns:
            ax_inset = ax_sec.inset_axes([0.72, 0.05, 0.26, 0.9])
            ax_inset.set_facecolor("#161b22")
            exch_pnl = port_df.groupby("exchange")["pnl_usd"].sum().sort_values()
            colors_ex = [GREEN if v >= 0 else RED for v in exch_pnl.values]
            ax_inset.barh(exch_pnl.index, exch_pnl.values, color=colors_ex, alpha=0.8)
            ax_inset.set_title("By Exchange", color=WHITE, fontsize=7, pad=3)
            ax_inset.tick_params(colors=MUTED, labelsize=6)
            for spine in ax_inset.spines.values(): spine.set_edgecolor("#30363d")
            ax_inset.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x/1000:.0f}k"))

    # ── Title ─────────────────────────────────────────────────────────────────
    dir_label = "  |  LONG ONLY" if port_df["direction"].eq("LONG").all() else \
                "  |  SHORT ONLY" if port_df["direction"].eq("SHORT").all() else \
                "  |  SPLIT PARAMS" if params_label.startswith("LONG:") else ""
    fig.suptitle(
        f"FiloHFT  |  LRC Mean-Reversion Strategy  |  Fit-All Portfolio{dir_label}  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        color=WHITE, fontsize=13, fontweight="bold", y=0.97,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"PNG saved -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",             default=None)
    ap.add_argument("--mc",              type=int, default=1000)
    ap.add_argument("--out",             default=None)
    ap.add_argument("--long-only",       action="store_true")
    ap.add_argument("--short-only",      action="store_true")
    ap.add_argument("--split-direction", action="store_true",
                    help="Use best fit-all LONG params for longs, SHORT params for shorts")
    args = ap.parse_args()

    results_dir = Path(__file__).parent.parent / "data" / "backtest_results"
    if args.csv:
        live_csv = Path(args.csv)
    else:
        files = sorted(results_dir.glob("discover_live_*.csv"), reverse=True)
        live_csv = files[0] if files else None
    if live_csv is None:
        print("No discover_live CSV found."); sys.exit(1)

    if args.split_direction:
        direction_tag = "_SPLIT"
    elif args.long_only:
        direction_tag = "_LONG"
    elif args.short_only:
        direction_tag = "_SHORT"
    else:
        direction_tag = ""
    default_name = f"FiloHFT_PM_Report{direction_tag}.png"
    out_path = Path(args.out) if args.out else _DESKTOP / default_name

    print(f"Loading: {live_csv.name}")
    live = pd.read_csv(live_csv)
    spx_rets = load_spx_rets()

    all_trades = []

    if args.split_direction:
        lp = LONG_DEFAULTS; sp = SHORT_DEFAULTS
        params_label = (f"LONG: sigma={lp['sigma']} | {lp['entry_tf']} | {lp['tp_rr']}R | rel={lp['rel_thresh']}  "
                        f"||  SHORT: sigma={sp['sigma']} | {sp['entry_tf']} | {sp['tp_rr']}R | rel={sp['rel_thresh']}")
        print(f"SPLIT-DIRECTION mode")
        print(f"  LONG  params: sigma={lp['sigma']} | {lp['entry_tf']} | {lp['tp_rr']}R | rel={lp['rel_thresh']}")
        print(f"  SHORT params: sigma={sp['sigma']} | {sp['entry_tf']} | {sp['tp_rr']}R | rel={sp['rel_thresh']}")
        print(f"Running LONG pass across {len(live)} symbols...")
        long_trades = []
        for i, row in live.iterrows():
            sym = row["sym"]; exch = str(row.get("exchange","")); sec = str(row.get("sector",""))
            t = _backtest_symbol(sym, exch, sec, spx_rets,
                                 sigma=lp["sigma"], entry_tf=lp["entry_tf"],
                                 tp_rr_override=lp["tp_rr"], rel_thresh=lp["rel_thresh"])
            long_trades.extend([x for x in t if x["direction"] == "LONG"])
            if (i+1) % 100 == 0:
                print(f"  [LONG pass] {i+1}/{len(live)}  |  {len(long_trades)} longs")
        print(f"Running SHORT pass across {len(live)} symbols...")
        short_trades = []
        for i, row in live.iterrows():
            sym = row["sym"]; exch = str(row.get("exchange","")); sec = str(row.get("sector",""))
            t = _backtest_symbol(sym, exch, sec, spx_rets,
                                 sigma=sp["sigma"], entry_tf=sp["entry_tf"],
                                 tp_rr_override=sp["tp_rr"], rel_thresh=sp["rel_thresh"])
            short_trades.extend([x for x in t if x["direction"] == "SHORT"])
            if (i+1) % 100 == 0:
                print(f"  [SHORT pass] {i+1}/{len(live)}  |  {len(short_trades)} shorts")
        all_trades = long_trades + short_trades
        print(f"Combined: {len(long_trades)} longs + {len(short_trades)} shorts = {len(all_trades)} trades")
    else:
        params_label = f"sigma={SIGMA} | {ENTRY_TF} | {TP_RR}R | rel={REL_THRESH}"
        print(f"Fixed params: {params_label}")
        print(f"Running backtest across {len(live)} symbols...")
        for i, row in live.iterrows():
            sym = row["sym"]; exch = str(row.get("exchange","")); sec = str(row.get("sector",""))
            trades = _backtest_symbol(sym, exch, sec, spx_rets)
            all_trades.extend(trades)
            if (i+1) % 100 == 0:
                print(f"  {i+1}/{len(live)}  |  {len(all_trades)} trades")

    if not all_trades:
        print("No trades found."); sys.exit(1)

    all_df = pd.DataFrame(all_trades)
    if args.long_only:
        all_df = all_df[all_df["direction"] == "LONG"].copy()
        print(f"  Long-only filter: {len(all_df)} trades kept")
    elif args.short_only:
        all_df = all_df[all_df["direction"] == "SHORT"].copy()
        print(f"  Short-only filter: {len(all_df)} trades kept")

    if all_df.empty:
        print("No trades after direction filter."); sys.exit(1)

    port_df  = build_portfolio(all_df)
    port_df["exit_date"]  = pd.to_datetime(port_df["exit_date"])
    port_df["entry_date"] = pd.to_datetime(port_df["entry_date"])

    n  = len(port_df)
    wr = float((port_df["r"] > 0).mean() * 100)
    sh = _sharpe(port_df["r"])
    mdd= _max_dd(port_df["pnl_usd"].values)
    pnl= float(port_df["pnl_usd"].sum())

    print(f"\n{'='*60}")
    print(f"PORTFOLIO RESULTS")
    print(f"{'='*60}")
    print(f"  Trades (after MC cap) : {n}")
    print(f"  Symbols               : {port_df['sym'].nunique()}")
    print(f"  Win Rate              : {wr:.1f}%")
    print(f"  Sharpe                : {sh:.2f}")
    print(f"  Total PnL             : ${pnl:,.0f}")
    print(f"  Max Drawdown          : ${mdd:,.0f}")
    print(f"  LONG / SHORT          : {(port_df['direction']=='LONG').sum()} / {(port_df['direction']=='SHORT').sum()}")

    print(f"\nRunning Monte Carlo ({args.mc} sims)...")
    np.random.seed(42)
    mc_pcts, mc_final, mc_dds, mc_sharpes = run_montecarlo(port_df["pnl_usd"], n_sims=args.mc)

    print(f"  MC median PnL   : ${np.percentile(mc_final,50):,.0f}")
    print(f"  MC 5th pct PnL  : ${np.percentile(mc_final,5):,.0f}")
    print(f"  MC 95th pct PnL : ${np.percentile(mc_final,95):,.0f}")
    print(f"  MC median max DD: ${np.percentile(mc_dds,50):,.0f}")
    print(f"  MC median Sharpe: {np.percentile(mc_sharpes,50):.2f}")

    print(f"\nGenerating PNG...")
    generate_png(port_df, all_df, mc_pcts, mc_final, mc_dds, mc_sharpes, out_path, params_label)


if __name__ == "__main__":
    main()
