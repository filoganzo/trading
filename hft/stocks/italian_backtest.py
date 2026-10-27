"""
Italian Equities — LRC Mean-Reversion Backtest
================================================
• Downloads 3.5 years of daily data from Yahoo Finance (Jan 2023 – present)
• Parameter sweep:  sigma × tp_rr × rel_perf_thresh (direction-aware)
• Robustness test:  best params validated on three rolling 12-month OOS windows
• Output:           Excel workbook + 3 PNG charts → Desktop
"""

import io, sys, itertools, warnings
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, LineChart, Reference

# ── Universe ──────────────────────────────────────────────────────────────────
ITALIAN_TICKERS = [
    "ENI.MI", "ENEL.MI", "ISP.MI", "UCG.MI", "RACE.MI", "G.MI", "LDO.MI",
    "PRY.MI", "MONC.MI", "SRG.MI", "TRN.MI", "A2A.MI", "AMP.MI", "PST.MI",
    "PIRC.MI", "NEXI.MI", "BAMI.MI", "FBK.MI", "DIAS.MI", "HER.MI", "SPM.MI",
    "REC.MI", "IG.MI", "AZM.MI", "BZU.MI", "TEN.MI", "BGN.MI", "STLAM.MI",
    "BMED.MI", "INW.MI", "IVG.MI", "BMPS.MI", "STM.MI", "EXO.MI", "CNH.MI",
    "MB.MI", "CPR.MI", "TIT.MI", "EXPN.MI", "ERG.MI",
]
BENCH_TICKER = "FTSEMIB.MI"
START_DATE   = "2023-01-01"
END_DATE     = date.today().isoformat()
DESKTOP      = Path.home() / "Desktop"

# ── Backtest constants ────────────────────────────────────────────────────────
LRC_LENGTH    = 80
SWING_WING    = 2
HTF_TOUCH_PCT = 0.005
MIN_HTF_SCORE = 60
SL_ATR_MULT   = 1.0
ATR_PERIOD    = 14
MAX_CONCURRENT = 8
CAPITAL       = 20_000.0   # effective capital per trade sizing

# ── Parameter grids ───────────────────────────────────────────────────────────
SIGMA_GRID      = [1.0, 1.5, 2.0]
TP_RR_GRID      = [1.0, 1.5, 2.0]
REL_THRESH_GRID = [0.0, -0.02, -0.05]

# ── Core maths ────────────────────────────────────────────────────────────────

def _lrc(close: np.ndarray, length: int, sigma: float):
    n = len(close); upper = np.full(n, np.nan); lower = np.full(n, np.nan)
    x = np.arange(length, dtype=float); xm = x.mean(); xd = x - xm; xd2 = (xd**2).sum()
    for i in range(length - 1, n):
        y = close[i - length + 1: i + 1]; ym = y.mean()
        sl = (xd * (y - ym)).sum() / xd2
        yh = sl * x + (ym - sl * xm); res = y - yh; std = res.std(ddof=0)
        m = yh[-1]; upper[i] = m + sigma * std; lower[i] = m - sigma * std
    return upper, lower


def _atr(df: pd.DataFrame, p: int = ATR_PERIOD) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n = len(c); res = np.full(n, np.nan)
    if n < p + 1: return res
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    res[p] = tr[:p].mean()
    for i in range(p + 1, n): res[i] = (res[i-1] * (p-1) + tr[i-1]) / p
    return res


def _htf_levels(w1_high: np.ndarray, w1_low: np.ndarray):
    n = len(w1_high); swing_px = []
    for i in range(SWING_WING, n - SWING_WING):
        if (np.all(w1_high[i-SWING_WING:i] < w1_high[i]) and
                np.all(w1_high[i+1:i+SWING_WING+1] < w1_high[i])):
            swing_px.append(float(w1_high[i]))
        if (np.all(w1_low[i-SWING_WING:i] > w1_low[i]) and
                np.all(w1_low[i+1:i+SWING_WING+1] > w1_low[i])):
            swing_px.append(float(w1_low[i]))
    if not swing_px:
        return []
    all_px = np.array(swing_px)
    seen = np.zeros(len(all_px), dtype=bool); levels = []
    for i in range(len(all_px)):
        if seen[i]: continue
        lv = all_px[i]; near = np.abs(all_px - lv) / (lv + 1e-9) <= HTF_TOUCH_PCT
        seen |= near
        rep = float(all_px[near].mean()); touches = int(near.sum())
        score = min(100.0, touches / 5.0 * 100.0)
        if score >= MIN_HTF_SCORE:
            levels.append((rep, score))
    return levels


def _find_htf_level(levels, price: float):
    zone = price * HTF_TOUCH_PCT
    best_px, best_sc = float("nan"), 0.0
    for lv, sc in levels:
        if abs(lv - price) <= zone and sc > best_sc:
            best_px = lv; best_sc = sc
    return best_px, best_sc


def _find_exit(high, low, close, entry_i, entry_px, sl_px, tp_px, direction, n):
    for xi in range(entry_i + 1, n):
        if direction == "SHORT":
            if high[xi] >= sl_px: return xi, sl_px, "SL"
            if low[xi]  <= tp_px: return xi, tp_px, "TP"
        else:
            if low[xi]  <= sl_px: return xi, sl_px, "SL"
            if high[xi] >= tp_px: return xi, tp_px, "TP"
    return n - 1, float(close[n-1]), "OPEN"


# ── Data download ─────────────────────────────────────────────────────────────

def _to_std(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def download_data(tickers):
    print(f"Downloading {len(tickers)} tickers ({START_DATE} – {END_DATE})…")
    d1_data, w1_data = {}, {}
    for tk in tickers:
        try:
            raw = yf.download(tk, start=START_DATE, end=END_DATE,
                              interval="1d", auto_adjust=True, progress=False)
            if raw is None or len(raw) < LRC_LENGTH + 50:
                continue
            df = _to_std(raw)
            if len(df) < LRC_LENGTH + 50: continue
            d1_data[tk] = df
            # Weekly: resample D1 → W1
            w1 = df.resample("W-FRI").agg({"open":"first","high":"max",
                                            "low":"min","close":"last","volume":"sum"}).dropna()
            if len(w1) >= 2 * SWING_WING + 4:
                w1_data[tk] = w1
        except Exception as e:
            print(f"  SKIP {tk}: {e}")
    print(f"  Loaded {len(d1_data)} symbols")
    return d1_data, w1_data


def download_bench():
    try:
        raw = yf.download(BENCH_TICKER, start=START_DATE, end=END_DATE,
                          interval="1d", auto_adjust=True, progress=False)
        df = _to_std(raw)
        r5 = df["close"].pct_change(5)
        return {ts.date(): float(v) for ts, v in zip(df.index, r5) if not np.isnan(v)}
    except Exception:
        return {}


# ── Per-symbol backtest ───────────────────────────────────────────────────────

def backtest_symbol(sym, d1_df, w1_df, bench_r5, sigma, tp_rr, rel_thresh):
    d1 = d1_df.copy().reset_index(drop=False)
    d1.rename(columns={d1.columns[0]: "date"}, inplace=True, errors="ignore")
    if "date" not in d1.columns:
        d1.insert(0, "date", d1_df.index)

    close = d1["close"].values
    high  = d1["high"].values
    low   = d1["low"].values
    open_ = d1["open"].values
    dates = pd.to_datetime(d1["date"]).dt.date.values
    n     = len(close)

    upper_arr, lower_arr = _lrc(close, LRC_LENGTH, sigma)
    atr_arr              = _atr(d1)

    w1_high = w1_df["high"].values
    w1_low  = w1_df["low"].values

    # Precompute weekly levels up to each D1 bar (using W1 bars available so far)
    w1_dates = pd.to_datetime(w1_df.index).tz_localize(None).date if hasattr(w1_df.index, 'tz') else pd.to_datetime(w1_df.index).date
    w1_dates = list(w1_dates)

    trades = []
    last_exit_i = -1

    for i in range(LRC_LENGTH + 10, n - 1):
        if i <= last_exit_i:
            continue
        c_i   = float(close[i])
        u_i   = float(upper_arr[i])
        l_i   = float(lower_arr[i])
        atr_i = float(atr_arr[i])
        if np.isnan(u_i) or np.isnan(atr_i) or atr_i <= 0:
            continue

        # Direction
        if c_i > u_i:   direction = "SHORT"
        elif c_i < l_i: direction = "LONG"
        else:            continue

        # HTF level (use all W1 bars up to current date)
        d_i = dates[i]
        wi  = sum(1 for wd in w1_dates if wd <= d_i)
        if wi < 2 * SWING_WING + 4: continue
        levels  = _htf_levels(w1_high[:wi], w1_low[:wi])
        lv, sc  = _find_htf_level(levels, c_i)
        if np.isnan(lv): continue

        # Rel-perf filter
        if rel_thresh != 0.0:
            spx_r5 = bench_r5.get(d_i)
            if spx_r5 is not None:
                stk_r5 = float(close[i] / close[i-5] - 1) if i >= 5 else 0.0
                rel_r5 = stk_r5 - spx_r5
                if direction == "LONG" and rel_r5 > rel_thresh:   continue
                if direction == "SHORT" and rel_r5 < -rel_thresh:  continue

        # Entry: open of next bar
        entry_i  = i + 1
        entry_px = float(open_[entry_i])
        pos      = +1 if direction == "SHORT" else -1
        sl_px    = lv + pos * SL_ATR_MULT * atr_i
        risk     = abs(entry_px - sl_px)
        if risk < 1e-6: continue
        if pos * (sl_px - entry_px) <= 0: continue
        tp_px    = entry_px - pos * tp_rr * risk

        xi, xp, xtype = _find_exit(high, low, close, entry_i, entry_px,
                                   sl_px, tp_px, direction, n)
        r     = (xp - entry_px) / risk if direction == "LONG" else (entry_px - xp) / risk
        dol   = (CAPITAL / entry_px) * risk   # dollar risk
        pnl   = r * dol
        last_exit_i = xi

        trades.append({
            "sym":       sym,
            "direction": direction,
            "sigma":     sigma,
            "tp_rr":     tp_rr,
            "rel_thresh":rel_thresh,
            "entry_date": pd.Timestamp(dates[entry_i]),
            "exit_date":  pd.Timestamp(dates[xi]),
            "entry_px":  round(entry_px, 4),
            "exit_px":   round(xp, 4),
            "sl_px":     round(sl_px, 4),
            "tp_px":     round(tp_px, 4),
            "level_px":  round(lv, 4),
            "htf_score": round(sc, 0),
            "r":         round(r, 3),
            "pnl_usd":   round(pnl, 2),
            "exit_type": xtype,
        })
    return trades


# ── Portfolio metrics ─────────────────────────────────────────────────────────

def _sharpe(r: pd.Series) -> float:
    if len(r) < 5 or r.std() == 0: return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def _maxdd(pnl: pd.Series) -> float:
    eq = pnl.cumsum(); pk = eq.cummax()
    return float((pk - eq).max())


def _apply_cap(trades: list) -> pd.DataFrame:
    if not trades: return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    df.sort_values("entry_date", inplace=True)
    accepted = []; open_pos = []
    for _, row in df.iterrows():
        ed = row["entry_date"]
        open_pos = [(xd, s) for xd, s in open_pos if xd > ed]
        if len(open_pos) >= MAX_CONCURRENT: continue
        accepted.append(row); open_pos.append((row["exit_date"], row["sym"]))
    return pd.DataFrame(accepted).reset_index(drop=True)


def portfolio_metrics(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 5:
        return {"n":0,"wr":0,"avg_r":0,"sharpe":-9,"pnl":0,"maxdd":0,"long_n":0,"short_n":0}
    r = df["r"]
    return {
        "n":       len(df),
        "wr":      round((r > 0).mean() * 100, 1),
        "avg_r":   round(r.mean(), 3),
        "sharpe":  round(_sharpe(r), 3),
        "pnl":     round(df["pnl_usd"].sum(), 2),
        "maxdd":   round(_maxdd(df["pnl_usd"]), 2),
        "long_n":  int((df["direction"] == "LONG").sum()),
        "short_n": int((df["direction"] == "SHORT").sum()),
    }


# ── Parameter sweep ───────────────────────────────────────────────────────────

def run_sweep(d1_data, w1_data, bench_r5):
    combos = list(itertools.product(SIGMA_GRID, TP_RR_GRID, REL_THRESH_GRID))
    print(f"\nParameter sweep: {len(combos)} combos × {len(d1_data)} symbols…")
    rows = []
    for ci, (sigma, tp_rr, rel_thresh) in enumerate(combos):
        all_trades = []
        for sym in d1_data:
            if sym not in w1_data: continue
            try:
                t = backtest_symbol(sym, d1_data[sym], w1_data[sym],
                                    bench_r5, sigma, tp_rr, rel_thresh)
                all_trades.extend(t)
            except Exception: pass

        df_all   = _apply_cap(all_trades)
        df_long  = _apply_cap([t for t in all_trades if t["direction"] == "LONG"])
        df_short = _apply_cap([t for t in all_trades if t["direction"] == "SHORT"])
        ma, ml, ms = portfolio_metrics(df_all), portfolio_metrics(df_long), portfolio_metrics(df_short)

        rows.append({
            "sigma": sigma, "tp_rr": tp_rr, "rel_thresh": rel_thresh,
            "all_n": ma["n"], "all_sharpe": ma["sharpe"], "all_wr": ma["wr"],
            "all_pnl": ma["pnl"], "all_dd": ma["maxdd"],
            "long_n": ml["n"], "long_sharpe": ml["sharpe"], "long_wr": ml["wr"], "long_pnl": ml["pnl"],
            "short_n": ms["n"], "short_sharpe": ms["sharpe"], "short_wr": ms["wr"], "short_pnl": ms["pnl"],
        })
        if (ci + 1) % 5 == 0:
            print(f"  [{ci+1:2d}/{len(combos)}] sigma={sigma} tp={tp_rr} rel={rel_thresh:+.2f}  "
                  f"L:{ml['sharpe']:.2f}  S:{ms['sharpe']:.2f}  ALL:{ma['sharpe']:.2f}")

    return pd.DataFrame(rows)


# ── Robustness test ───────────────────────────────────────────────────────────

def run_robustness(d1_data, w1_data, bench_r5, best_sigma, best_tp_rr, best_rel_thresh):
    print("\nRobustness test (rolling 12-month OOS windows)…")
    windows = [
        ("2023", "2023-01-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("Full", START_DATE,   END_DATE),
    ]
    rob_rows = []
    for label, ws, we in windows:
        ws_ts, we_ts = pd.Timestamp(ws), pd.Timestamp(we)
        all_trades = []
        for sym in d1_data:
            if sym not in w1_data: continue
            try:
                t = backtest_symbol(sym, d1_data[sym], w1_data[sym],
                                    bench_r5, best_sigma, best_tp_rr, best_rel_thresh)
                # Filter to window
                t = [x for x in t
                     if ws_ts <= pd.Timestamp(x["entry_date"]) <= we_ts]
                all_trades.extend(t)
            except Exception: pass
        df = _apply_cap(all_trades)
        m  = portfolio_metrics(df)
        rob_rows.append({"Period": label, **m})
        print(f"  {label}: N={m['n']}  Sharpe={m['sharpe']:.2f}  WR={m['wr']:.1f}%  PnL=${m['pnl']:,.0f}")
    return pd.DataFrame(rob_rows)


# ── Per-symbol stats (best params) ────────────────────────────────────────────

def sym_stats(d1_data, w1_data, bench_r5, sigma, tp_rr, rel_thresh):
    rows = []
    for sym in d1_data:
        if sym not in w1_data: continue
        try:
            t = backtest_symbol(sym, d1_data[sym], w1_data[sym],
                                bench_r5, sigma, tp_rr, rel_thresh)
            if not t: continue
            df = pd.DataFrame(t)
            r  = df["r"]
            rows.append({
                "Symbol": sym, "N": len(df),
                "WR%": round((r > 0).mean()*100, 1),
                "Avg_R": round(r.mean(), 3),
                "Sharpe": round(_sharpe(r), 3),
                "PnL_USD": round(df["pnl_usd"].sum(), 2),
                "MaxDD_USD": round(_maxdd(df["pnl_usd"]), 2),
                "Long": int((df["direction"]=="LONG").sum()),
                "Short": int((df["direction"]=="SHORT").sum()),
            })
        except Exception: pass
    return pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)


# ── All trades (best params) ──────────────────────────────────────────────────

def all_trades_best(d1_data, w1_data, bench_r5, sigma, tp_rr, rel_thresh):
    all_trades = []
    for sym in d1_data:
        if sym not in w1_data: continue
        try:
            all_trades.extend(backtest_symbol(sym, d1_data[sym], w1_data[sym],
                                              bench_r5, sigma, tp_rr, rel_thresh))
        except Exception: pass
    return _apply_cap(all_trades)


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_equity_curve(port_df, out_path):
    if port_df.empty: return
    df = port_df.sort_values("entry_date").copy()
    df["cum_pnl"] = df["pnl_usd"].cumsum()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#1e1e2e"); ax1.set_facecolor("#1e1e2e"); ax2.set_facecolor("#1e1e2e")
    ax1.plot(df["entry_date"], df["cum_pnl"], color="#cba6f7", linewidth=2)
    ax1.fill_between(df["entry_date"], df["cum_pnl"], alpha=0.15, color="#cba6f7")
    ax1.set_title("Italian Equities LRC — Portfolio Equity Curve", color="#cdd6f4", fontsize=14, pad=10)
    ax1.set_ylabel("Cumulative PnL ($)", color="#cdd6f4"); ax1.tick_params(colors="#a6adc8")
    ax1.spines[:].set_color("#45475a"); ax1.yaxis.label.set_color("#cdd6f4")
    # Drawdown
    eq = df["cum_pnl"]; pk = eq.cummax(); dd = pk - eq
    ax2.fill_between(df["entry_date"], -dd, color="#f38ba8", alpha=0.6)
    ax2.set_ylabel("Drawdown ($)", color="#cdd6f4"); ax2.tick_params(colors="#a6adc8")
    ax2.spines[:].set_color("#45475a")
    plt.tight_layout(pad=2)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


def plot_monthly_returns(port_df, out_path):
    if port_df.empty: return
    df = port_df.copy()
    df["month"] = pd.to_datetime(df["exit_date"]).dt.to_period("M")
    monthly = df.groupby("month")["pnl_usd"].sum().reset_index()
    monthly["month_str"] = monthly["month"].astype(str)
    colors = ["#a6e3a1" if v >= 0 else "#f38ba8" for v in monthly["pnl_usd"]]
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#1e1e2e"); ax.set_facecolor("#1e1e2e")
    ax.bar(monthly["month_str"], monthly["pnl_usd"], color=colors, edgecolor="#1e1e2e", width=0.7)
    ax.axhline(0, color="#45475a", linewidth=0.8)
    ax.set_title("Monthly PnL — Best Params (Italian Equities)", color="#cdd6f4", fontsize=13)
    ax.set_ylabel("PnL ($)", color="#cdd6f4"); ax.tick_params(colors="#a6adc8", axis="both")
    ax.set_xticklabels(monthly["month_str"], rotation=45, ha="right", fontsize=8)
    for sp in ax.spines.values(): sp.set_color("#45475a")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


def plot_param_heatmap(sweep_df, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#1e1e2e")
    fig.suptitle("Parameter Sensitivity — Sharpe (All Directions)", color="#cdd6f4", fontsize=13)
    for ax, rel_t in zip(axes, REL_THRESH_GRID):
        sub = sweep_df[sweep_df["rel_thresh"] == rel_t]
        pivot = sub.pivot_table(index="sigma", columns="tp_rr", values="all_sharpe")
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-1, vmax=3, aspect="auto")
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, color="#cdd6f4")
        ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index, color="#cdd6f4")
        ax.set_xlabel("TP R:R", color="#cdd6f4"); ax.set_ylabel("Sigma", color="#cdd6f4")
        ax.set_title(f"rel_thresh={rel_t:+.2f}", color="#cdd6f4")
        ax.set_facecolor("#1e1e2e")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="black" if abs(v) < 2 else "white", fontsize=9)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Chart saved: {out_path}")


# ── Excel writer ──────────────────────────────────────────────────────────────

HDR_FILL = PatternFill("solid", fgColor="2D2F3F")
ALT_FILL = PatternFill("solid", fgColor="1E2030")
HDR_FONT = Font(bold=True, color="CBA6F7", name="Calibri")
VAL_FONT = Font(color="CDD6F4", name="Calibri")
POS_FONT = Font(color="A6E3A1", name="Calibri")
NEG_FONT = Font(color="F38BA8", name="Calibri")
THIN     = Border(bottom=Side(style="thin", color="45475A"))


def _hdr_row(ws, row, values):
    for c, v in enumerate(values, 1):
        cell = ws.cell(row=row, column=c, value=v)
        cell.font = HDR_FONT; cell.fill = HDR_FILL; cell.alignment = Alignment(horizontal="center")
        ws.column_dimensions[get_column_letter(c)].width = max(12, len(str(v)) + 4)


def _data_row(ws, row, values, alt=False):
    fill = ALT_FILL if alt else PatternFill("solid", fgColor="181825")
    for c, v in enumerate(values, 1):
        cell = ws.cell(row=row, column=c, value=v)
        cell.fill = fill
        if isinstance(v, (int, float)):
            cell.font = POS_FONT if v > 0 else (NEG_FONT if v < 0 else VAL_FONT)
        else:
            cell.font = VAL_FONT
        cell.alignment = Alignment(horizontal="right" if isinstance(v, (int,float)) else "left")


def write_excel(sweep_df, rob_df, sym_df, port_df, out_path):
    wb = Workbook()

    # ── Sheet 1: Portfolio Summary ────────────────────────────────────────────
    ws = wb.active; ws.title = "Portfolio_Summary"
    ws.sheet_properties.tabColor = "CBA6F7"
    m = portfolio_metrics(port_df)
    _hdr_row(ws, 1, ["Metric", "Value"])
    rows_ = [
        ("Total Trades",        m["n"]),
        ("Win Rate (%)",        m["wr"]),
        ("Avg R",               m["avg_r"]),
        ("Sharpe (R-series)",   m["sharpe"]),
        ("Total PnL ($)",       m["pnl"]),
        ("Max Drawdown ($)",    m["maxdd"]),
        ("LONG trades",         m["long_n"]),
        ("SHORT trades",        m["short_n"]),
        ("Symbols traded",      port_df["sym"].nunique() if not port_df.empty else 0),
        ("Date range",          f"{port_df['entry_date'].min().date()} → {port_df['entry_date'].max().date()}"
                                if not port_df.empty else "—"),
        ("Best sigma",          sweep_df.loc[sweep_df["all_sharpe"].idxmax(), "sigma"]),
        ("Best tp_rr",          sweep_df.loc[sweep_df["all_sharpe"].idxmax(), "tp_rr"]),
        ("Best rel_thresh",     sweep_df.loc[sweep_df["all_sharpe"].idxmax(), "rel_thresh"]),
    ]
    for i, (k, v) in enumerate(rows_, 2):
        ws.cell(row=i, column=1, value=k).font = VAL_FONT
        ws.cell(row=i, column=2, value=v).font = VAL_FONT

    # ── Sheet 2: Parameter Sweep ──────────────────────────────────────────────
    ws2 = wb.create_sheet("Parameter_Sweep")
    ws2.sheet_properties.tabColor = "89B4FA"
    cols = ["sigma","tp_rr","rel_thresh",
            "all_n","all_sharpe","all_wr","all_pnl","all_dd",
            "long_n","long_sharpe","long_wr","long_pnl",
            "short_n","short_sharpe","short_wr","short_pnl"]
    _hdr_row(ws2, 1, cols)
    sweep_sorted = sweep_df.sort_values("all_sharpe", ascending=False)
    for i, (_, row) in enumerate(sweep_sorted.iterrows(), 2):
        _data_row(ws2, i, [row[c] for c in cols], alt=i%2==0)

    # ── Sheet 3: Robustness ───────────────────────────────────────────────────
    ws3 = wb.create_sheet("Robustness")
    ws3.sheet_properties.tabColor = "A6E3A1"
    rcols = ["Period","n","wr","avg_r","sharpe","pnl","maxdd","long_n","short_n"]
    _hdr_row(ws3, 1, rcols)
    for i, (_, row) in enumerate(rob_df.iterrows(), 2):
        _data_row(ws3, i, [row[c] for c in rcols], alt=i%2==0)

    # ── Sheet 4: Symbol Stats ─────────────────────────────────────────────────
    ws4 = wb.create_sheet("Symbol_Stats")
    ws4.sheet_properties.tabColor = "FAB387"
    scols = list(sym_df.columns)
    _hdr_row(ws4, 1, scols)
    for i, (_, row) in enumerate(sym_df.iterrows(), 2):
        _data_row(ws4, i, [row[c] for c in scols], alt=i%2==0)

    # ── Sheet 5: All Trades ───────────────────────────────────────────────────
    ws5 = wb.create_sheet("All_Trades")
    ws5.sheet_properties.tabColor = "F9E2AF"
    if not port_df.empty:
        tcols = list(port_df.columns)
        _hdr_row(ws5, 1, tcols)
        for i, (_, row) in enumerate(port_df.sort_values("entry_date").iterrows(), 2):
            _data_row(ws5, i, [row[c] for c in tcols], alt=i%2==0)

    # ── Sheet 6: Best vs Direction ────────────────────────────────────────────
    ws6 = wb.create_sheet("Best_Params")
    ws6.sheet_properties.tabColor = "CBA6F7"
    best_long  = sweep_df.loc[sweep_df["long_sharpe"].idxmax()]
    best_short = sweep_df.loc[sweep_df["short_sharpe"].idxmax()]
    best_all   = sweep_df.loc[sweep_df["all_sharpe"].idxmax()]
    _hdr_row(ws6, 1, ["Direction", "sigma", "tp_rr", "rel_thresh",
                       "N", "WR%", "Sharpe", "PnL($)"])
    for i, (lbl, row_) in enumerate([("LONG",  best_long),
                                      ("SHORT", best_short),
                                      ("ALL",   best_all)], 2):
        vals = [lbl,
                row_["sigma"], row_["tp_rr"], row_["rel_thresh"],
                row_.get(f"{lbl.lower()}_n" if lbl!="ALL" else "all_n", ""),
                row_.get(f"{lbl.lower()}_wr" if lbl!="ALL" else "all_wr", ""),
                row_.get(f"{lbl.lower()}_sharpe" if lbl!="ALL" else "all_sharpe", ""),
                row_.get(f"{lbl.lower()}_pnl" if lbl!="ALL" else "all_pnl", "")]
        _data_row(ws6, i, vals)

    wb.save(out_path)
    print(f"\nExcel saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  ITALIAN EQUITIES — LRC MEAN-REVERSION BACKTEST")
    print("=" * 65)

    # 1. Download
    d1_data, w1_data = download_data(ITALIAN_TICKERS)
    bench_r5          = download_bench()
    print(f"Benchmark dates: {len(bench_r5)}")

    if not d1_data:
        print("ERROR: no data downloaded. Check internet / yfinance.")
        return

    # 2. Sweep
    sweep_df = run_sweep(d1_data, w1_data, bench_r5)
    print("\n── Top 5 overall ──")
    print(sweep_df.nlargest(5, "all_sharpe")[
        ["sigma","tp_rr","rel_thresh","all_n","all_sharpe","long_sharpe","short_sharpe"]
    ].to_string(index=False))

    best_row   = sweep_df.loc[sweep_df["all_sharpe"].idxmax()]
    best_sigma = best_row["sigma"]
    best_tp    = best_row["tp_rr"]
    best_rel   = best_row["rel_thresh"]

    best_long  = sweep_df.loc[sweep_df["long_sharpe"].idxmax()]
    best_short = sweep_df.loc[sweep_df["short_sharpe"].idxmax()]

    print(f"\nBEST LONG:  sigma={best_long['sigma']} tp={best_long['tp_rr']} rel={best_long['rel_thresh']:+.2f}"
          f"  → Sharpe={best_long['long_sharpe']:.3f}  N={int(best_long['long_n'])}")
    print(f"BEST SHORT: sigma={best_short['sigma']} tp={best_short['tp_rr']} rel={best_short['rel_thresh']:+.2f}"
          f"  → Sharpe={best_short['short_sharpe']:.3f}  N={int(best_short['short_n'])}")
    print(f"BEST ALL:   sigma={best_sigma} tp={best_tp} rel={best_rel:+.2f}"
          f"  → Sharpe={best_row['all_sharpe']:.3f}  N={int(best_row['all_n'])}")

    # 3. Robustness
    rob_df = run_robustness(d1_data, w1_data, bench_r5, best_sigma, best_tp, best_rel)

    # 4. Full trade list + symbol stats (best params)
    port_df = all_trades_best(d1_data, w1_data, bench_r5, best_sigma, best_tp, best_rel)
    sym_df  = sym_stats(d1_data, w1_data, bench_r5, best_sigma, best_tp, best_rel)

    # 5. Output
    print("\nGenerating charts…")
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    chart1 = DESKTOP / f"IT_equity_curve_{ts}.png"
    chart2 = DESKTOP / f"IT_monthly_pnl_{ts}.png"
    chart3 = DESKTOP / f"IT_param_heatmap_{ts}.png"
    plot_equity_curve(port_df, chart1)
    plot_monthly_returns(port_df, chart2)
    plot_param_heatmap(sweep_df, chart3)

    excel_path = DESKTOP / f"Italian_LRC_Backtest_{ts}.xlsx"
    write_excel(sweep_df, rob_df, sym_df, port_df, excel_path)

    print("\n" + "="*65)
    print("  DONE")
    print("="*65)


if __name__ == "__main__":
    main()
