"""
fit_all_sweep.py

Portfolio-level fit-all parameter sweep, evaluated separately for LONG and SHORT.

For each (sigma, entry_tf, tp_rr, rel_thresh) combination:
  - Runs backtest across ALL live symbols
  - Splits trades by direction
  - Computes portfolio-level Sharpe (with MAX_CONCURRENT cap)
  - Outputs results CSV + prints best params per direction

Usage:
    python -m stocks.backtest.fit_all_sweep
    python -m stocks.backtest.fit_all_sweep --csv path/to/discover_live.csv
    python -m stocks.backtest.fit_all_sweep --out results.csv
"""
import argparse
import io
import sys
import itertools
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
EFFECTIVE     = CAPITAL * LEVERAGE
MAX_CONCURRENT = 8

# ── Parameter grid ────────────────────────────────────────────────────────────
SIGMA_GRID      = [1.0, 1.5, 2.0]
ENTRY_TF_GRID   = ["M15", "M30"]
TP_RR_GRID      = [1.0, 1.5, 2.0]
REL_THRESH_GRID = [0.0, -0.01, -0.02, -0.03, -0.05]

ALL_COMBOS = list(itertools.product(SIGMA_GRID, ENTRY_TF_GRID, TP_RR_GRID, REL_THRESH_GRID))

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
    if len(rets) < 5 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(252))


def _max_dd_pct(pnl_series: pd.Series) -> float:
    equity = pnl_series.cumsum()
    peak = equity.cummax()
    return float((peak - equity).max())


def _apply_mc_cap(trades: list[dict]) -> pd.DataFrame:
    """Sort by entry_date, enforce MAX_CONCURRENT open positions."""
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    df.sort_values("entry_date", inplace=True)
    accepted = []
    open_pos = []
    for _, row in df.iterrows():
        ed = row["entry_date"]
        open_pos = [(xd, s) for xd, s in open_pos if xd > ed]
        if len(open_pos) >= MAX_CONCURRENT:
            continue
        accepted.append(row)
        open_pos.append((row["exit_date"], row["sym"]))
    return pd.DataFrame(accepted).reset_index(drop=True)


def _portfolio_metrics(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 5:
        return {"n": 0, "wr": 0.0, "sharpe": -99.0, "total_pnl": 0.0, "max_dd": 0.0, "avg_r": 0.0}
    r = df["r"]
    return {
        "n":         len(df),
        "wr":        round((r > 0).mean() * 100, 1),
        "avg_r":     round(r.mean(), 3),
        "sharpe":    round(_sharpe(r), 3),
        "total_pnl": round(df["pnl_usd"].sum(), 2),
        "max_dd":    round(_max_dd_pct(df["pnl_usd"].reset_index(drop=True)), 2),
    }


# ── Load and precompute all symbols ──────────────────────────────────────────

def load_symbols(live_csv: Path) -> list[str]:
    return pd.read_csv(live_csv)["sym"].tolist()


def precompute_all(symbols: list[str], spx_rets: dict) -> dict:
    """
    Returns: {sym: {"pc": ..., "meta": {exchange, sector}} | None}
    Precomputes with all sigma values at once.
    """
    result = {}
    n = len(symbols)
    for i, sym in enumerate(symbols):
        if (i + 1) % 100 == 0:
            print(f"  Precomputing {i+1}/{n} ...", flush=True)
        try:
            daily_df = load_bt_bars(sym)
            m15_df   = load_bt_bars_m15(sym)
            w1_df    = load_bt_bars_w1(sym)
            daily_df = _normalize_df(daily_df)
            if daily_df is None or len(daily_df) < LRC_LENGTH + 20:
                result[sym] = None; continue
            if m15_df is None or m15_df.empty:
                result[sym] = None; continue
            if w1_df is None or w1_df.empty or len(w1_df) < 20:
                result[sym] = None; continue
        except Exception:
            result[sym] = None; continue

        try:
            pc = precompute_symbol_v11(
                df=daily_df,
                m15_df=m15_df,
                w1_df=w1_df,
                lrc_lengths=[LRC_LENGTH],
                sigma_mults=SIGMA_GRID,
                htf_touch_pcts=[HTF_TOUCH_PCT],
                swing_wings=[SWING_WING],
            )
            result[sym] = {"pc": pc}
        except Exception:
            result[sym] = None

    loaded = sum(1 for v in result.values() if v is not None)
    print(f"Precomputed {loaded}/{n} symbols OK")
    return result


# ── Single param combo backtest ───────────────────────────────────────────────

def _run_combo(precomputed: dict, live_df: pd.DataFrame,
               sigma: float, entry_tf: str, tp_rr: float, rel_thresh: float,
               spx_rets: dict) -> list[dict]:
    all_trades = []
    for _, row in live_df.iterrows():
        sym = row["sym"]
        obj = precomputed.get(sym)
        if obj is None:
            continue
        pc = obj["pc"]

        lrc_key = (LRC_LENGTH, sigma)
        if lrc_key not in pc.get("lrc_data", {}):
            continue

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
                atr_contract_ratio=1.0,
                spx_ret5_map=spx_rets,
            )
        except Exception:
            continue

        if not sigs:
            continue

        n   = pc["n"]
        c   = pc["close"]
        h   = pc["high"]
        l_  = pc["low"]
        idx = pc["index"]
        _, upper_arr, lower_arr = pc["lrc_data"][lrc_key]

        exchange = str(row.get("exchange", ""))
        sector   = str(row.get("sector", ""))

        for (sig_i, entry_i, entry_px, direction, level_px, atr_v, htf_score) in sigs:
            pos   = +1 if direction == "SHORT" else -1
            sl_px = level_px + pos * SL_MULT * atr_v
            risk  = abs(entry_px - sl_px)
            if risk < 1e-6:
                continue
            if pos * (sl_px - entry_px) <= 0:
                continue
            tp_px = entry_px - pos * tp_rr * risk
            dollar_risk = (EFFECTIVE / entry_px) * risk

            xb, xp, xtype, mfe_r, mae_r = _find_exit(
                h, l_, c, entry_i, entry_px, sl_px, tp_px,
                direction, BREAKEVEN_RR, risk, n,
            )
            r   = (xp - entry_px) / risk if direction == "LONG" else (entry_px - xp) / risk
            pnl = r * dollar_risk

            all_trades.append({
                "sym":       sym,
                "exchange":  exchange,
                "sector":    sector,
                "direction": direction,
                "entry_date": pd.Timestamp(idx[entry_i]).date(),
                "exit_date":  pd.Timestamp(idx[xb]).date(),
                "r":         round(r, 3),
                "pnl_usd":   round(pnl, 2),
                "exit_type": xtype,
            })

    return all_trades


# ── Main sweep ────────────────────────────────────────────────────────────────

def sweep(live_csv: Path, out_path: Path):
    live = pd.read_csv(live_csv)
    print(f"Symbols: {len(live)}")
    spx_rets = load_spx_rets()

    print("Precomputing symbols (all sigmas)...")
    precomputed = precompute_all(live["sym"].tolist(), spx_rets)

    n_combos = len(ALL_COMBOS)
    print(f"\nSweeping {n_combos} parameter combinations...")

    rows = []
    for ci, (sigma, entry_tf, tp_rr, rel_thresh) in enumerate(ALL_COMBOS):
        trades = _run_combo(precomputed, live, sigma, entry_tf, tp_rr, rel_thresh, spx_rets)

        # Split by direction, apply MC cap per direction
        long_t  = [t for t in trades if t["direction"] == "LONG"]
        short_t = [t for t in trades if t["direction"] == "SHORT"]

        df_long  = _apply_mc_cap(long_t)
        df_short = _apply_mc_cap(short_t)
        df_all   = _apply_mc_cap(trades)

        m_long  = _portfolio_metrics(df_long)
        m_short = _portfolio_metrics(df_short)
        m_all   = _portfolio_metrics(df_all)

        rows.append({
            "sigma": sigma, "entry_tf": entry_tf, "tp_rr": tp_rr, "rel_thresh": rel_thresh,
            # LONG
            "long_n":      m_long["n"],
            "long_wr":     m_long["wr"],
            "long_avg_r":  m_long["avg_r"],
            "long_sharpe": m_long["sharpe"],
            "long_pnl":    m_long["total_pnl"],
            "long_dd":     m_long["max_dd"],
            # SHORT
            "short_n":      m_short["n"],
            "short_wr":     m_short["wr"],
            "short_avg_r":  m_short["avg_r"],
            "short_sharpe": m_short["sharpe"],
            "short_pnl":    m_short["total_pnl"],
            "short_dd":     m_short["max_dd"],
            # COMBINED
            "all_n":      m_all["n"],
            "all_sharpe": m_all["sharpe"],
            "all_pnl":    m_all["total_pnl"],
        })

        if (ci + 1) % 10 == 0 or (ci + 1) == n_combos:
            print(f"  [{ci+1:3d}/{n_combos}] sigma={sigma} {entry_tf} tp={tp_rr} rel={rel_thresh:+.2f}  "
                  f"L-sh={m_long['sharpe']:.2f}  S-sh={m_short['sharpe']:.2f}  "
                  f"ALL-sh={m_all['sharpe']:.2f}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(out_path, index=False)
    print(f"\nSweep saved -> {out_path}")

    _print_best(results)
    return results


def _print_best(df: pd.DataFrame):
    print(f"\n{'='*72}")
    print("BEST FIT-ALL PARAMS")
    print(f"{'='*72}")

    # Best for LONG
    best_long = df.loc[df["long_sharpe"].idxmax()]
    print(f"\n[LONG]  Sharpe={best_long['long_sharpe']:.3f}  WR={best_long['long_wr']:.1f}%  "
          f"N={int(best_long['long_n'])}  PnL=${best_long['long_pnl']:,.0f}")
    print(f"        sigma={best_long['sigma']}  entry_tf={best_long['entry_tf']}  "
          f"tp_rr={best_long['tp_rr']}  rel_thresh={best_long['rel_thresh']:+.3f}")

    # Best for SHORT
    best_short = df.loc[df["short_sharpe"].idxmax()]
    print(f"\n[SHORT] Sharpe={best_short['short_sharpe']:.3f}  WR={best_short['short_wr']:.1f}%  "
          f"N={int(best_short['short_n'])}  PnL=${best_short['short_pnl']:,.0f}")
    print(f"        sigma={best_short['sigma']}  entry_tf={best_short['entry_tf']}  "
          f"tp_rr={best_short['tp_rr']}  rel_thresh={best_short['rel_thresh']:+.3f}")

    # Best combined
    best_all = df.loc[df["all_sharpe"].idxmax()]
    print(f"\n[ALL]   Sharpe={best_all['all_sharpe']:.3f}  N={int(best_all['all_n'])}  "
          f"PnL=${best_all['all_pnl']:,.0f}")
    print(f"        sigma={best_all['sigma']}  entry_tf={best_all['entry_tf']}  "
          f"tp_rr={best_all['tp_rr']}  rel_thresh={best_all['rel_thresh']:+.3f}")

    # Top 5 per direction
    print(f"\n--- Top 5 LONG combos by Sharpe ---")
    cols_l = ["sigma","entry_tf","tp_rr","rel_thresh","long_n","long_wr","long_sharpe","long_pnl"]
    print(df.nlargest(5, "long_sharpe")[cols_l].to_string(index=False))

    print(f"\n--- Top 5 SHORT combos by Sharpe ---")
    cols_s = ["sigma","entry_tf","tp_rr","rel_thresh","short_n","short_wr","short_sharpe","short_pnl"]
    print(df.nlargest(5, "short_sharpe")[cols_s].to_string(index=False))

    print(f"\n{'='*72}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.csv:
        live_csv = Path(args.csv)
    else:
        live_csv = _latest_csv("discover_live")
        if live_csv is None:
            print("No discover_live_*.csv found. Run discover_all.py first.")
            sys.exit(1)

    ts  = datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(args.out) if args.out else _OUT_DIR / f"fit_all_sweep_{ts}.csv"

    print(f"Sweep against: {live_csv.name}")
    sweep(live_csv, out)


if __name__ == "__main__":
    main()
