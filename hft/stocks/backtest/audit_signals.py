"""
audit_signals.py

Quantitative audit of scanner_live signals.
For every ACTIVE/WATCHING signal, verifies:
  1. Raw close IS outside the 80-bar LRC band on the signal bar
  2. Extension magnitude (% beyond band — should be > 0)
  3. Signal recency (entry bar within last LOOKBACK trading days)
  4. Rel-perf divergence vs SPX  (stock underperformed for LONG, outperformed for SHORT)
  5. HTF weekly level proximity (within 0.5%)

Usage:
    python -m stocks.backtest.audit_signals
    python -m stocks.backtest.audit_signals --csv path/to/discover_live.csv
    python -m stocks.backtest.audit_signals --top 50
"""
import argparse
import io
import sys
from datetime import date, datetime
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
    load_spx_rets,
)
from stocks.backtest.data.extender import load_bt_bars, load_bt_bars_m15, load_bt_bars_w1

# ── Constants matching scanner_live / discover_all ──────────────────────────
LRC_LENGTH    = 80
HTF_TOUCH_PCT = 0.5
HTF_FRAC      = HTF_TOUCH_PCT / 100.0
MIN_HTF_SCORE = 60
SWING_WING    = 2
ENTRY_TOUCH   = 0.005
LOOKBACK_BARS = 30          # audit window: last 30 trading days

_RESULTS = Path(__file__).parent.parent / "data" / "backtest_results"
_DATA    = Path(__file__).parent.parent / "data"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _latest_csv(prefix: str) -> Path | None:
    files = sorted(_RESULTS.glob(f"{prefix}_*.csv"), reverse=True)
    return files[0] if files else None


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    cap  = ["Open", "High", "Low", "Close", "Volume"]
    low  = ["open", "high", "low", "close", "volume"]
    if cap[0] in df.columns:
        df = df[cap].copy()
        df.columns = low
    return df


def _pct_outside(ci: float, upper: float, lower: float, direction: str) -> float:
    """Positive means price IS outside the band."""
    if direction == "SHORT":
        return (ci - upper) / upper * 100.0 if upper > 0 else 0.0
    else:
        return (lower - ci) / lower * 100.0 if lower > 0 else 0.0


# ── Main audit ───────────────────────────────────────────────────────────────

def audit(live_csv: Path, top_n: int | None = None) -> pd.DataFrame:
    live = pd.read_csv(live_csv)
    if top_n:
        live = live.nlargest(top_n, "sharpe_r")

    spx_rets = load_spx_rets()

    rows = []
    n_syms = len(live)
    for idx_row, row in live.iterrows():
        sym   = row["sym"]
        sigma = float(row.get("sigma", 1.5))
        rel_thresh = float(row.get("rel_perf_thresh", 0.0))
        entry_tf   = str(row.get("entry_tf", "M30"))
        atr_ratio  = float(row.get("atr_contract", 1.0))

        try:
            daily_df = load_bt_bars(sym)
            m15_df   = load_bt_bars_m15(sym)
            w1_df    = load_bt_bars_w1(sym)
            daily_df = _normalize_df(daily_df)
            if daily_df is None or len(daily_df) < LRC_LENGTH + 10:
                continue
            if m15_df is None or m15_df.empty:
                continue
            if w1_df is None or w1_df.empty or len(w1_df) < 20:
                continue
        except Exception:
            continue

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
                atr_contract_ratio=atr_ratio,
                spx_ret5_map=spx_rets,
            )
        except Exception:
            continue

        if not sigs:
            continue

        n = pc["n"]
        c = pc["close"]
        idx = pc["index"]
        ret5_map = pc["ret5_map"]
        lrc_key = (LRC_LENGTH, sigma)
        if lrc_key not in pc.get("lrc_data", {}):
            continue
        _, upper_arr, lower_arr = pc["lrc_data"][lrc_key]

        cutoff_i = max(0, n - LOOKBACK_BARS)

        for (sig_i, entry_i, entry_px, direction, level_px, atr_v, htf_score) in sigs:
            if entry_i < cutoff_i:
                continue

            ci      = float(c[sig_i])
            ui      = float(upper_arr[sig_i])
            li      = float(lower_arr[sig_i])
            pct_ext = _pct_outside(ci, ui, li, direction)

            sig_date   = pd.Timestamp(idx[sig_i]).date()
            entry_date = pd.Timestamp(idx[entry_i]).date()
            days_old   = (date.today() - entry_date).days

            stock_ret5 = ret5_map.get(pd.Timestamp(idx[sig_i]).normalize())
            spx_ret5   = spx_rets.get(pd.Timestamp(idx[sig_i]).normalize())
            rel_ret5   = None
            rel_ok     = None
            if stock_ret5 is not None and spx_ret5 is not None:
                rel_ret5 = round(stock_ret5 - spx_ret5, 4)
                if direction == "LONG":
                    rel_ok = rel_ret5 <= rel_thresh if rel_thresh != 0.0 else True
                else:
                    rel_ok = rel_ret5 >= -rel_thresh if rel_thresh != 0.0 else True

            htf_prox_pct = abs(ci - level_px) / level_px * 100.0 if level_px else None

            # ── Pass / fail ─────────────────────────────────────────────
            check_band   = pct_ext > 0        # raw close IS outside band
            check_fresh  = days_old <= 3
            check_rel    = rel_ok if rel_ok is not None else True
            check_htf    = htf_prox_pct is not None and htf_prox_pct <= 0.5

            all_pass = check_band and check_fresh and check_rel and check_htf

            rows.append({
                "sym":          sym,
                "direction":    direction,
                "sig_date":     sig_date,
                "entry_date":   entry_date,
                "days_old":     days_old,
                "close":        round(ci, 4),
                "upper":        round(ui, 4),
                "lower":        round(li, 4),
                "pct_outside":  round(pct_ext, 3),
                "level_px":     round(float(level_px), 4),
                "htf_prox_pct": round(htf_prox_pct, 3) if htf_prox_pct else None,
                "htf_score":    htf_score,
                "rel_ret5":     rel_ret5,
                "rel_thresh":   rel_thresh,
                "sigma":        sigma,
                "entry_tf":     entry_tf,
                "check_band":   check_band,
                "check_fresh":  check_fresh,
                "check_rel":    check_rel,
                "check_htf":    check_htf,
                "PASS":         all_pass,
            })

    return pd.DataFrame(rows)


def _print_report(df: pd.DataFrame):
    if df.empty:
        print("No signals found in the audit window.")
        return

    n_total = len(df)
    n_pass  = df["PASS"].sum()
    print(f"\n{'='*72}")
    print(f"SIGNAL AUDIT   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*72}")
    print(f"Signals in window: {n_total}   PASS: {n_pass}   FAIL: {n_total - n_pass}")
    print()

    # Check breakdown
    for chk in ["check_band", "check_fresh", "check_rel", "check_htf"]:
        pct = df[chk].mean() * 100
        label = {
            "check_band":  "Raw close outside LRC band",
            "check_fresh": "Signal age <=3 days",
            "check_rel":   "Rel-perf diverged from SPX",
            "check_htf":   "Within 0.5% of HTF level",
        }[chk]
        status = "OK" if pct >= 95 else "WARN" if pct >= 80 else "FAIL"
        print(f"  [{status:4s}] {label:35s}  {pct:5.1f}%")

    print()
    print("── Per-signal detail ──────────────────────────────────────────────")
    cols = ["sym", "direction", "sig_date", "close", "pct_outside", "htf_prox_pct",
            "rel_ret5", "days_old", "check_band", "check_fresh", "PASS"]
    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", 200)
    print(df[cols].sort_values(["PASS", "pct_outside"], ascending=[False, False]).to_string(index=False))

    fails = df[~df["PASS"]]
    if not fails.empty:
        print(f"\n── FAILURES ({len(fails)}) ───────────────────────────────────────────")
        print(fails[cols].to_string(index=False))

    print(f"\n{'='*72}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default=None)
    ap.add_argument("--top",  type=int, default=None)
    ap.add_argument("--out",  default=None, help="Save audit CSV to path")
    args = ap.parse_args()

    if args.csv:
        live_csv = Path(args.csv)
    else:
        live_csv = _latest_csv("discover_live")
        if live_csv is None:
            print("No discover_live_*.csv found — run discover_all.py first.")
            sys.exit(1)

    print(f"Auditing against: {live_csv.name}")
    df = audit(live_csv, top_n=args.top)
    _print_report(df)

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Saved audit to {args.out}")


if __name__ == "__main__":
    main()
