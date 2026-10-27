"""
portfolio_excel.py

Full portfolio backtest using per-symbol best parameters from discover_live CSV.
Outputs a multi-sheet Excel workbook:

  Trades          — every trade: dates, prices, SL/TP, R, PnL, exit type
  Symbol_Stats    — per-symbol: n_trades, win_rate, sharpe, total_pnl, max_dd
  Portfolio       — aggregated equity curve, monthly returns, overall stats
  Param_Summary   — parameters used per symbol

Usage:
    python -m stocks.backtest.portfolio_excel
    python -m stocks.backtest.portfolio_excel --csv path/to/discover_live.csv
    python -m stocks.backtest.portfolio_excel --out my_results.xlsx
"""
import argparse
import io
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stocks.backtest.runner_v11 import (
    precompute_symbol_v11,
    _find_signals_v11,
    _find_exit,
    load_spx_rets,
)
from stocks.backtest.data.extender import load_bt_bars, load_bt_bars_m15, load_bt_bars_w1

# ── Constants ─────────────────────────────────────────────────────────────────
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
EFFECTIVE     = CAPITAL * LEVERAGE   # $20,000
MAX_CONCURRENT = 8

_RESULTS = Path(__file__).parent.parent / "data" / "backtest_results"
_OUT_DIR  = _RESULTS


def _latest_csv(prefix: str) -> Path | None:
    files = sorted(_RESULTS.glob(f"{prefix}_*.csv"), reverse=True)
    return files[0] if files else None


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    cap = ["Open", "High", "Low", "Close", "Volume"]
    low = ["open", "high", "low", "close", "volume"]
    if cap[0] in df.columns:
        df = df[cap].copy()
        df.columns = low
    return df


def _sharpe(rets: pd.Series) -> float:
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(252))


def _max_dd_pct(pnl_series: pd.Series) -> float:
    equity = pnl_series.cumsum()
    peak = equity.cummax()
    dd = (peak - equity)
    return float(dd.max())


# ── Per-symbol backtest returning full trade list ─────────────────────────────

def _backtest_symbol(sym: str, params: dict, spx_rets: dict) -> list[dict]:
    sigma      = float(params.get("sigma", 1.5))
    rel_thresh = float(params.get("rel_perf_thresh", 0.0))
    entry_tf   = str(params.get("entry_tf", "M30"))
    atr_ratio  = float(params.get("atr_contract", 1.0))
    tp_rr      = float(params.get("tp_rr", 1.0))
    sector     = str(params.get("sector", ""))
    exchange   = str(params.get("exchange", ""))

    try:
        daily_df = load_bt_bars(sym)
        m15_df   = load_bt_bars_m15(sym)
        w1_df    = load_bt_bars_w1(sym)
        daily_df = _normalize_df(daily_df)
        if daily_df is None or len(daily_df) < LRC_LENGTH + 20:
            return []
        if m15_df is None or m15_df.empty:
            return []
        if w1_df is None or w1_df.empty or len(w1_df) < 20:
            return []
    except Exception:
        return []

    try:
        pc = precompute_symbol_v11(
            df=daily_df,
            m15_df=m15_df,
            w1_df=w1_df,
            lrc_lengths=[LRC_LENGTH],
            sigma_mults=[sigma],
            htf_touch_pcts=[HTF_TOUCH_PCT],
            swing_wings=[SWING_WING],
        )
    except Exception:
        return []

    try:
        sigs = _find_signals_v11(
            pc=pc,
            lrc_length=LRC_LENGTH,
            sigma_mult=sigma,
            htf_tf=HTF_FRAC,
            entry_touch_tf=ENTRY_TOUCH,
            min_htf_score=MIN_HTF_SCORE,
            swing_wing=SWING_WING,
            entry_tf_str=entry_tf,
            rel_perf_thresh=rel_thresh,
            atr_contract_ratio=atr_ratio,
            spx_ret5_map=spx_rets,
        )
    except Exception:
        return []

    if not sigs:
        return []

    n   = pc["n"]
    c   = pc["close"]
    h   = pc["high"]
    l_  = pc["low"]
    idx = pc["index"]

    lrc_key = (LRC_LENGTH, sigma)
    _, upper_arr, lower_arr = pc["lrc_data"].get(lrc_key, (None, None, None))

    trades = []
    for (sig_i, entry_i, entry_px, direction, level_px, atr_v, htf_score) in sigs:
        pos    = +1 if direction == "SHORT" else -1
        sl_px  = level_px + pos * SL_MULT * atr_v
        risk   = abs(entry_px - sl_px)
        if risk < 1e-6:
            continue
        if pos * (sl_px - entry_px) <= 0:
            continue
        tp_px  = entry_px - pos * tp_rr * risk
        dollar_risk = (EFFECTIVE / entry_px) * risk

        xb, xp, xtype, mfe_r, mae_r = _find_exit(
            h, l_, c, entry_i, entry_px, sl_px, tp_px,
            direction, BREAKEVEN_RR, risk, n,
        )

        r = (xp - entry_px) / risk if direction == "LONG" else (entry_px - xp) / risk
        pnl = r * dollar_risk

        ci = float(c[sig_i])
        ui = float(upper_arr[sig_i]) if upper_arr is not None else float("nan")
        li = float(lower_arr[sig_i]) if lower_arr is not None else float("nan")
        pct_ext = ((ci - ui) / ui * 100) if direction == "SHORT" else ((li - ci) / li * 100)

        trades.append({
            "sym":          sym,
            "exchange":     exchange,
            "sector":       sector,
            "direction":    direction,
            "signal_date":  pd.Timestamp(idx[sig_i]).date(),
            "entry_date":   pd.Timestamp(idx[entry_i]).date(),
            "exit_date":    pd.Timestamp(idx[xb]).date(),
            "entry_px":     round(float(entry_px), 5),
            "exit_px":      round(float(xp), 5),
            "sl_px":        round(float(sl_px), 5),
            "tp_px":        round(float(tp_px), 5),
            "level_px":     round(float(level_px), 5),
            "r":            round(r, 3),
            "pnl_usd":      round(pnl, 2),
            "exit_type":    xtype,
            "mfe_r":        mfe_r,
            "mae_r":        mae_r,
            "atr_v":        round(float(atr_v), 5),
            "htf_score":    htf_score,
            "pct_ext":      round(pct_ext, 3),
            "sigma":        sigma,
            "entry_tf":     entry_tf,
            "tp_rr":        tp_rr,
            "rel_thresh":   rel_thresh,
        })

    return trades


# ── Symbol-level stats ────────────────────────────────────────────────────────

def _sym_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sym, g in df.groupby("sym"):
        r_series = g["r"]
        n  = len(g)
        wr = (g["r"] > 0).mean() * 100
        total_pnl = g["pnl_usd"].sum()
        avg_r = r_series.mean()
        sh    = _sharpe(r_series)
        mdd   = _max_dd_pct(g["pnl_usd"].reset_index(drop=True))
        first = g["entry_date"].min()
        last  = g["entry_date"].max()
        rows.append({
            "sym":       sym,
            "exchange":  g["exchange"].iloc[0],
            "sector":    g["sector"].iloc[0],
            "direction_mix": f"{(g['direction']=='LONG').sum()}L/{(g['direction']=='SHORT').sum()}S",
            "n_trades":  n,
            "win_rate":  round(wr, 1),
            "avg_r":     round(avg_r, 3),
            "sharpe_r":  round(sh, 3),
            "total_pnl": round(total_pnl, 2),
            "max_dd":    round(mdd, 2),
            "first_trade": first,
            "last_trade":  last,
        })
    return pd.DataFrame(rows).sort_values("sharpe_r", ascending=False).reset_index(drop=True)


# ── Portfolio summary ─────────────────────────────────────────────────────────

def _portfolio_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df2 = df.copy()
    df2["entry_date"] = pd.to_datetime(df2["entry_date"])
    df2["exit_date"]  = pd.to_datetime(df2["exit_date"])

    # Apply max-concurrent cap: sort by entry_date, enforce MAX_CONCURRENT
    df2.sort_values("entry_date", inplace=True)
    accepted = []
    open_pos: list[tuple] = []  # (exit_date, sym)
    for _, row in df2.iterrows():
        ed = row["entry_date"]
        open_pos = [(xd, s) for xd, s in open_pos if xd > ed]
        if len(open_pos) >= MAX_CONCURRENT:
            continue
        accepted.append(row)
        open_pos.append((row["exit_date"], row["sym"]))
    df_port = pd.DataFrame(accepted).reset_index(drop=True)

    # Monthly returns
    df_port["month"] = df_port["exit_date"].dt.to_period("M")
    monthly = df_port.groupby("month")["pnl_usd"].sum().reset_index()
    monthly.columns = ["month", "pnl_usd"]
    monthly["cumulative_pnl"] = monthly["pnl_usd"].cumsum()

    # Overall stats
    n   = len(df_port)
    wr  = (df_port["r"] > 0).mean() * 100
    avg_r = df_port["r"].mean()
    sh  = _sharpe(df_port["r"])
    total_pnl = df_port["pnl_usd"].sum()
    mdd = _max_dd_pct(df_port["pnl_usd"].reset_index(drop=True))
    tp_rate = (df_port["exit_type"] == "TP").mean() * 100
    sl_rate = (df_port["exit_type"] == "SL").mean() * 100

    summary = pd.DataFrame([{
        "metric": "Total Trades",        "value": n},
        {"metric": "Win Rate (%)",        "value": round(wr, 1)},
        {"metric": "Avg R",               "value": round(avg_r, 3)},
        {"metric": "Sharpe (R-series)",   "value": round(sh, 3)},
        {"metric": "Total PnL ($)",       "value": round(total_pnl, 2)},
        {"metric": "Max Drawdown ($)",    "value": round(mdd, 2)},
        {"metric": "TP Rate (%)",         "value": round(tp_rate, 1)},
        {"metric": "SL Rate (%)",         "value": round(sl_rate, 1)},
        {"metric": "LONG count",          "value": (df_port["direction"] == "LONG").sum()},
        {"metric": "SHORT count",         "value": (df_port["direction"] == "SHORT").sum()},
        {"metric": "Symbols traded",      "value": df_port["sym"].nunique()},
        {"metric": "Date range",          "value": f"{df_port['entry_date'].min().date()} -> {df_port['entry_date'].max().date()}"},
    ])

    return monthly, summary, df_port


# ── Excel writer ──────────────────────────────────────────────────────────────

def write_excel(trades_df: pd.DataFrame, sym_stats: pd.DataFrame,
                monthly: pd.DataFrame, summary: pd.DataFrame,
                port_trades: pd.DataFrame, live_csv: Path,
                out_path: Path):
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Sheet 1: All trades (chronological, full detail)
        trades_df.sort_values(["entry_date", "sym"]).to_excel(
            writer, sheet_name="Trades", index=False)

        # Sheet 2: Portfolio trades (after MC cap)
        port_trades.sort_values("entry_date").to_excel(
            writer, sheet_name="Portfolio_Trades", index=False)

        # Sheet 3: Symbol stats
        sym_stats.to_excel(writer, sheet_name="Symbol_Stats", index=False)

        # Sheet 4: Monthly returns
        monthly.to_excel(writer, sheet_name="Monthly_Returns", index=False)

        # Sheet 5: Portfolio summary
        summary.to_excel(writer, sheet_name="Portfolio_Summary", index=False)

        # Sheet 6: Parameters used (from discover_live CSV)
        pd.read_csv(live_csv).to_excel(writer, sheet_name="Params_Used", index=False)

    print(f"\nExcel saved -> {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Path to discover_live CSV")
    ap.add_argument("--out", default=None, help="Output Excel path")
    ap.add_argument("--top", type=int, default=None, help="Limit to top N symbols by Sharpe")
    args = ap.parse_args()

    if args.csv:
        live_csv = Path(args.csv)
    else:
        live_csv = _latest_csv("discover_live")
        if live_csv is None:
            print("No discover_live_*.csv found. Run discover_all.py first.")
            sys.exit(1)

    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(args.out) if args.out else _OUT_DIR / f"portfolio_backtest_{ts}.xlsx"

    print(f"Loading: {live_csv.name}")
    live = pd.read_csv(live_csv)
    if args.top:
        live = live.nlargest(args.top, "sharpe_r")

    print(f"Symbols: {len(live)}")
    spx_rets = load_spx_rets()

    all_trades = []
    for i, row in live.iterrows():
        sym = row["sym"]
        trades = _backtest_symbol(sym, row.to_dict(), spx_rets)
        all_trades.extend(trades)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(live)} symbols  |  {len(all_trades)} trades so far")

    if not all_trades:
        print("No trades found.")
        sys.exit(1)

    trades_df = pd.DataFrame(all_trades)
    sym_stats  = _sym_stats(trades_df)
    monthly, summary, port_trades = _portfolio_stats(trades_df)

    # Print summary to console
    print(f"\n{'='*60}")
    print(f"PORTFOLIO BACKTEST — {ts}")
    print(f"{'='*60}")
    print(summary.to_string(index=False))
    print(f"\nTop 20 symbols by Sharpe:")
    print(sym_stats.head(20)[["sym","n_trades","win_rate","sharpe_r","total_pnl"]].to_string(index=False))

    write_excel(trades_df, sym_stats, monthly, summary, port_trades, live_csv, out)


if __name__ == "__main__":
    main()
