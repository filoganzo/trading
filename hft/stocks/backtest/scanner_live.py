"""
stocks/backtest/scanner_live.py

Live signal scanner — identical logic to runner_v11 / discover_all.

Loads the discover_live watchlist (per-symbol best params from the latest
discover_live_*.csv), refreshes data from MT5 if requested, then reports:

  ACTIVE  — LRC extension + HTF level + rel-perf + M30 pin-bar all confirmed
             → set alert, entry valid NOW (entry bar already has bars)
  WATCHING — LRC extension + HTF level + rel-perf confirmed on the signal bar
             but pin-bar entry not yet confirmed (today's bars pending)
             → watch tomorrow's open for the pin-bar

Usage:
    python -m stocks.backtest.scanner_live              # scan once, cached data
    python -m stocks.backtest.scanner_live --refresh    # pull fresh MT5 data first
    python -m stocks.backtest.scanner_live --loop       # repeat every 30 min
    python -m stocks.backtest.scanner_live --symbol XOP.NYSE  # single symbol
    python -m stocks.backtest.scanner_live --top 30     # top 30 by Sharpe only
    python -m stocks.backtest.scanner_live --csv path/to/discover_live.csv
"""

import argparse
import io
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── UTF-8 stdout — prevents UnicodeEncodeError on Windows terminals ───────────
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Ensure project root is on path ───────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent   # …/FINAL HFT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Logging (file + console, both UTF-8) ─────────────────────────────────────
_LOG_PATH = Path(__file__).parent.parent / "data" / "scanner_live.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_LOG_PATH), encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Project imports ───────────────────────────────────────────────────────────
from stocks.backtest.config import (
    BT_CACHE_DIR, BT_CACHE_M15_DIR, BT_CACHE_W1_DIR, RESULTS_DIR,
)
from stocks.backtest.data.extender import (
    load_bt_bars, load_bt_bars_m15, load_bt_bars_w1,
    _cache_path, _cache_path_m15, _cache_path_w1,
    _pull_mt5, _pull_mt5_m15, _pull_mt5_w1,
)
from stocks.backtest.runner_v11 import (
    precompute_symbol_v11,
    _find_signals_v11,
)
from stocks.backtest.signals.htf_signal import find_htf_level, find_m15_entry

try:
    import MetaTrader5 as mt5
    _HAS_MT5 = True
except ImportError:
    _HAS_MT5 = False

# ── Fixed params (same as discover_all.py — do not change) ───────────────────
LRC_LENGTH    = 80
HTF_TOUCH_PCT = 0.5           # percent → precompute converts to fraction internally
HTF_FRAC      = HTF_TOUCH_PCT / 100.0   # 0.005 — used in _find_signals_v11 lookup
MIN_HTF_SCORE = 60
SWING_WING    = 2
ENTRY_TOUCH   = 0.005         # fraction (0.5%) — passed to find_m15_entry as-is
SL_MULT       = 1.0
LRC_CL        = 15
LRC_CS        = 9
ATR_P         = 14

# How many trailing daily bars to include in the signal lookback
# (covers: signal bar = today-1, entry bar = today)
SIGNAL_LOOKBACK_BARS = 1


# ── SPX returns loader (mirrors discover_all.py — tries multiple filenames) ───

def _load_spx_rets() -> dict:
    """Load US500 5-bar returns map: {date: float}. Empty dict on failure."""
    for name in ("US500.parquet", "US500.NAS.parquet", "US500.NYSE.parquet"):
        p = BT_CACHE_DIR / name
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            df.columns = [c.lower() for c in df.columns]
            r5 = df["close"].pct_change(5)
            result = {
                pd.Timestamp(d).normalize(): float(v)
                for d, v in r5.items() if pd.notna(v)
            }
            log.info("SPX rets loaded: %d dates from %s", len(result), name)
            return result
        except Exception as e:
            log.warning("Failed to load SPX rets from %s: %s", name, e)
    log.warning("No US500 parquet found in %s — rel-perf filter will be bypassed", BT_CACHE_DIR)
    return {}


# ── Watchlist ─────────────────────────────────────────────────────────────────

def _find_latest_discover_csv() -> Optional[Path]:
    """Return the most recently modified discover_live_*.csv in RESULTS_DIR."""
    candidates = sorted(
        RESULTS_DIR.glob("discover_live_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_watchlist(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load discover_live CSV → DataFrame with one row per symbol.
    Required columns: sym, sigma, entry_tf, rel_perf_thresh, atr_contract, tp_rr.
    """
    if csv_path is None:
        csv_path = _find_latest_discover_csv()
    if csv_path is None or not csv_path.exists():
        log.error(
            "No discover_live_*.csv found in %s. Run discover_all.py first.", RESULTS_DIR
        )
        return pd.DataFrame()

    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    required = {"sym", "sigma", "entry_tf", "rel_perf_thresh", "atr_contract", "tp_rr"}
    missing = required - set(df.columns)
    if missing:
        log.error("discover_live CSV missing columns: %s", missing)
        return pd.DataFrame()

    df["sym"] = df["sym"].astype(str).str.strip()
    log.info("Watchlist loaded: %d symbols from %s", len(df), csv_path.name)
    return df.reset_index(drop=True)


# ── MT5 data refresh ──────────────────────────────────────────────────────────

def _mt5_connect() -> bool:
    if not _HAS_MT5:
        log.warning("MetaTrader5 package not installed — run: pip install MetaTrader5")
        return False
    if not mt5.initialize():
        log.error("MT5 initialize() failed: %s", mt5.last_error())
        return False
    info = mt5.account_info()
    if info:
        log.info("MT5 connected — account %s | server %s", info.login, info.server)
    return True


def _merge_frames(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Concat, sort by index, drop exact-timestamp duplicates (keep latest)."""
    if fresh.empty:
        return existing
    if existing.empty:
        return fresh
    combined = pd.concat([existing, fresh]).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def refresh_symbol(sym: str) -> None:
    """
    Pull the latest bars for sym from MT5 and merge into the local parquet cache.
    Silently skips if MT5 is unavailable or returns no data.
    """
    if not _HAS_MT5:
        return

    # ── daily ────────────────────────────────────────────────────────────────
    try:
        existing = load_bt_bars(sym)
        fresh    = _pull_mt5(sym)
        merged   = _merge_frames(existing, fresh)
        if not merged.empty and len(merged) >= 50:
            merged.to_parquet(_cache_path(sym))
    except Exception as e:
        log.warning("Daily refresh skipped for %s: %s", sym, e)

    # ── M15 ──────────────────────────────────────────────────────────────────
    try:
        existing = load_bt_bars_m15(sym)
        fresh    = _pull_mt5_m15(sym)
        merged   = _merge_frames(existing, fresh)
        if not merged.empty and len(merged) >= 200:
            merged.to_parquet(_cache_path_m15(sym))
    except Exception as e:
        log.warning("M15 refresh skipped for %s: %s", sym, e)

    # ── weekly ───────────────────────────────────────────────────────────────
    try:
        existing = load_bt_bars_w1(sym)
        fresh    = _pull_mt5_w1(sym)
        merged   = _merge_frames(existing, fresh)
        if not merged.empty and len(merged) >= 20:
            merged.to_parquet(_cache_path_w1(sym))
    except Exception as e:
        log.warning("W1 refresh skipped for %s: %s", sym, e)


# ── Pending-signal detector (no pin-bar requirement) ─────────────────────────

def _check_pending(pc: dict, sigma: float, entry_tf_str: str,
                   rel_perf_thresh: float, atr_contract_ratio: float,
                   spx_rets: dict) -> list:
    """
    Scan the last SIGNAL_LOOKBACK_BARS daily bars for the LRC extension +
    HTF level + rel-perf + ATR-contraction conditions WITHOUT requiring a
    pin-bar entry (entry bar has no M30 data yet).

    Returns list of dicts with: bar_i, direction, level_px, htf_score, date.
    """
    n        = pc["n"]
    lrc_c    = pc["lrc_c"]
    atr      = pc["atr"]
    c        = pc["close"]
    idx      = pc["index"]
    w_levels = pc["w_levels"].get((SWING_WING, HTF_FRAC), [])
    ret5_map = pc["ret5_map"]

    if not w_levels:
        return []

    mid, upper, lower = pc["lrc_data"][(LRC_LENGTH, sigma)]
    min_i   = max(LRC_LENGTH, LRC_CL + LRC_CS + 1, ATR_P + 1)
    start_i = max(min_i, n - SIGNAL_LOOKBACK_BARS)

    # Choose entry-TF bar dict to check if entry bar ALREADY has data
    if entry_tf_str == "M30":
        m15_by_dt = pc["m30_by_date"]
    elif entry_tf_str == "H1":
        m15_by_dt = pc["h1_by_date"]
    else:
        m15_by_dt = pc["m15_by_date"]

    results = []
    for i in range(start_i, n):
        ci = c[i]
        if np.isnan(ci) or np.isnan(upper[i]) or np.isnan(atr[i]):
            continue

        if ci > upper[i]:
            direction = "SHORT"
        elif ci < lower[i]:
            direction = "LONG"
        else:
            continue

        # ATR contraction
        if atr_contract_ratio < 1.0 and i >= 1:
            atr_peak = float(np.max(atr[max(0, i - 10):i]))
            if atr_peak > 0 and atr[i] / atr_peak > atr_contract_ratio:
                continue

        # Relative performance
        if rel_perf_thresh != 0.0 and i >= 5:
            date_key   = pd.Timestamp(idx[i]).normalize()
            stock_ret5 = ret5_map.get(date_key)
            spx_ret5   = spx_rets.get(date_key)
            if stock_ret5 is not None and spx_ret5 is not None:
                rel_ret5 = stock_ret5 - spx_ret5
                if direction == "LONG" and rel_ret5 > rel_perf_thresh:
                    continue
                if direction == "SHORT" and rel_ret5 < -rel_perf_thresh:
                    continue

        # HTF level proximity
        price     = float(c[i])
        lv, score = find_htf_level(w_levels, price, HTF_FRAC, MIN_HTF_SCORE)
        if np.isnan(lv):
            continue

        # Only mark as "pending" if entry bar does NOT yet have M30 data
        if i + 1 < n:
            entry_date = pd.Timestamp(idx[i + 1]).normalize()
        else:
            entry_date = None   # signal bar is the very last bar

        entry_bar_has_data = (
            entry_date is not None
            and m15_by_dt.get(entry_date) is not None
        )
        if entry_bar_has_data:
            continue   # _find_signals_v11 already handles this case

        results.append({
            "bar_i":     i,
            "direction": direction,
            "level_px":  lv,
            "htf_score": score,
            "date":      pd.Timestamp(idx[i]).normalize(),
            "atr":       float(atr[i]),
            "close":     float(c[i]),
        })

    return results


# ── Per-symbol scan ───────────────────────────────────────────────────────────

def scan_symbol(sym: str, params: dict, spx_rets: dict) -> tuple[list, list]:
    """
    Run full precompute + signal detection for one symbol.

    Returns:
        active   — list of confirmed-entry signal dicts (TRADE NOW)
        watching — list of pending signal dicts (wait for pin-bar)
    """
    daily_df = load_bt_bars(sym)
    m15_df   = load_bt_bars_m15(sym)
    w1_df    = load_bt_bars_w1(sym)

    # Data sufficiency checks — skip symbol gracefully on any gap
    if daily_df is None or daily_df.empty or len(daily_df) < 100:
        return [], []
    if m15_df is None or m15_df.empty:
        return [], []
    if w1_df is None or w1_df.empty or len(w1_df) < 20:
        return [], []

    # precompute_symbol_v11 expects lowercase OHLCV columns; load_bt_bars
    # returns Capitalized columns (Open/High/Low/Close/Volume) — normalise here.
    _ohlcv = ["open", "high", "low", "close", "volume"]
    _cap   = ["Open", "High", "Low", "Close", "Volume"]
    if _cap[0] in daily_df.columns:
        daily_df = daily_df[_cap].copy()
        daily_df.columns = _ohlcv

    sigma              = float(params["sigma"])
    entry_tf_str       = str(params["entry_tf"]).strip()
    rel_perf_thresh    = float(params["rel_perf_thresh"])
    atr_contract_ratio = float(params["atr_contract"])
    tp_rr              = float(params["tp_rr"])

    # Precompute — same call as discover_all / runner_v11
    try:
        pc = precompute_symbol_v11(
            df=daily_df,
            m15_df=m15_df,
            w1_df=w1_df,
            lrc_lengths=[LRC_LENGTH],
            sigma_mults=[sigma],
            htf_touch_pcts=[HTF_TOUCH_PCT],   # 0.5 — converted to 0.005 internally
            swing_wings=[SWING_WING],
        )
    except Exception as e:
        log.debug("precompute failed for %s: %s", sym, e)
        return [], []

    if pc is None or pc.get("n", 0) < 100:
        return [], []

    n   = pc["n"]
    idx = pc["index"]
    atr = pc["atr"]

    # ── ACTIVE signals: full pipeline including pin-bar ───────────────────────
    try:
        raw_sigs = _find_signals_v11(
            pc=pc,
            lrc_length=LRC_LENGTH,
            sigma_mult=sigma,
            htf_tf=HTF_FRAC,             # 0.005 — matches w_levels key
            entry_touch_tf=ENTRY_TOUCH,  # 0.005 fraction
            min_htf_score=MIN_HTF_SCORE,
            swing_wing=SWING_WING,
            entry_tf_str=entry_tf_str,
            rel_perf_thresh=rel_perf_thresh,
            atr_contract_ratio=atr_contract_ratio,
            spx_ret5_map=spx_rets,
        )
    except Exception as e:
        log.debug("_find_signals_v11 failed for %s: %s", sym, e)
        raw_sigs = []

    # Filter to signals whose entry bar is within the lookback window
    cutoff_entry = max(0, n - SIGNAL_LOOKBACK_BARS)
    active = []
    for (sig_i, entry_i, entry_px, direction, level_px, atr_v, htf_score) in raw_sigs:
        if entry_i < cutoff_entry:
            continue
        sl_dist = atr_v * SL_MULT
        sl = entry_px - sl_dist if direction == "LONG" else entry_px + sl_dist
        tp = entry_px + sl_dist * tp_rr if direction == "LONG" else entry_px - sl_dist * tp_rr

        active.append({
            "sym":         sym,
            "status":      "ACTIVE",
            "direction":   direction,
            "entry_px":    round(float(entry_px), 5),
            "level_px":    round(float(level_px), 5),
            "sl":          round(float(sl), 5),
            "tp":          round(float(tp), 5),
            "tp_rr":       tp_rr,
            "atr":         round(float(atr_v), 5),
            "htf_score":   round(float(htf_score), 1),
            "signal_date": pd.Timestamp(idx[sig_i]).normalize(),
            "entry_date":  pd.Timestamp(idx[entry_i]).normalize() if entry_i < n else None,
            "sigma":       sigma,
            "entry_tf":    entry_tf_str,
            "rel_perf":    rel_perf_thresh,
            "sharpe":      float(params.get("sharpe_r", 0.0)),
            "exchange":    str(params.get("exchange", "")),
            "sector":      str(params.get("sector", "")),
        })

    # ── WATCHING signals: conditions met, pin-bar not yet seen ────────────────
    try:
        pending = _check_pending(
            pc=pc,
            sigma=sigma,
            entry_tf_str=entry_tf_str,
            rel_perf_thresh=rel_perf_thresh,
            atr_contract_ratio=atr_contract_ratio,
            spx_rets=spx_rets,
        )
    except Exception as e:
        log.debug("_check_pending failed for %s: %s", sym, e)
        pending = []

    watching = []
    for p in pending:
        atr_v   = p["atr"]
        close_v = p["close"]
        watching.append({
            "sym":         sym,
            "status":      "WATCHING",
            "direction":   p["direction"],
            "entry_px":    None,
            "level_px":    round(float(p["level_px"]), 5),
            "sl":          None,
            "tp":          None,
            "tp_rr":       tp_rr,
            "atr":         round(float(atr_v), 5),
            "htf_score":   round(float(p["htf_score"]), 1),
            "signal_date": p["date"],
            "entry_date":  None,
            "sigma":       sigma,
            "entry_tf":    entry_tf_str,
            "rel_perf":    rel_perf_thresh,
            "sharpe":      float(params.get("sharpe_r", 0.0)),
            "exchange":    str(params.get("exchange", "")),
            "sector":      str(params.get("sector", "")),
            "last_close":  round(float(close_v), 5),
        })

    return active, watching


# ── Alert formatting ──────────────────────────────────────────────────────────

def _fmt_date(ts) -> str:
    if ts is None:
        return "?"
    return ts.strftime("%Y-%m-%d")


def format_signal(sig: dict) -> str:
    status = sig["status"]
    sym    = sig["sym"]
    d      = sig["direction"]
    arrow  = "^" if d == "LONG" else "v"
    lvl    = sig["level_px"]
    score  = sig["htf_score"]
    tf     = sig["entry_tf"]
    sigma  = sig["sigma"]
    rp     = sig["rel_perf"]
    sharpe = sig["sharpe"]
    sdate  = _fmt_date(sig["signal_date"])
    exch   = sig.get("exchange", "")
    sec    = sig.get("sector", "")

    if status == "ACTIVE":
        epx = sig["entry_px"]
        sl  = sig["sl"]
        tp  = sig["tp"]
        rr  = sig["tp_rr"]
        ed  = _fmt_date(sig["entry_date"])
        return (
            f"  [{arrow} {d:5s} ACTIVE  ] {sym:<12s} ({exch}/{sec})\n"
            f"    entry={epx:.4f}  sl={sl:.4f}  tp={tp:.4f}  ({rr}R)\n"
            f"    level={lvl:.4f} (score={score:.0f})  "
            f"TF={tf}  sigma={sigma}  rel_perf={rp:+.0%}\n"
            f"    signal={sdate}  entry_bar={ed}  "
            f"Sharpe={sharpe:.2f}"
        )
    else:  # WATCHING
        close = sig.get("last_close", float("nan"))
        atr   = sig["atr"]
        return (
            f"  [{arrow} {d:5s} WATCHING] {sym:<12s} ({exch}/{sec})\n"
            f"    last_close={close:.4f}  level={lvl:.4f} (score={score:.0f})\n"
            f"    ATR={atr:.4f}  TF={tf}  sigma={sigma}  rel_perf={rp:+.0%}\n"
            f"    signal={sdate}  "
            f"Sharpe={sharpe:.2f}  — wait for {tf} pin-bar at level"
        )


# ── Main scan ─────────────────────────────────────────────────────────────────

def run_scan(watchlist: pd.DataFrame, do_refresh: bool = True) -> tuple[list, list]:
    """
    Scan all watchlist symbols.
    Returns (active_signals, watching_signals), each sorted by Sharpe desc.
    """
    if watchlist.empty:
        return [], []

    # MT5 connection (only needed for refresh)
    mt5_ok = False
    if do_refresh:
        mt5_ok = _mt5_connect()
        if not mt5_ok:
            log.warning("MT5 unavailable — scanning from cached data only")

    # Load SPX returns once
    spx_rets = _load_spx_rets()

    # Refresh US500 benchmark first if connected
    if do_refresh and mt5_ok:
        log.info("Refreshing US500 benchmark...")
        refresh_symbol("US500")
        spx_rets = _load_spx_rets()

    all_active   = []
    all_watching = []

    log.info("Scanning %d symbols...", len(watchlist))
    for _, row in watchlist.iterrows():
        sym = str(row["sym"]).strip()

        if do_refresh and mt5_ok:
            refresh_symbol(sym)

        params = {
            "sigma":           row["sigma"],
            "entry_tf":        row["entry_tf"],
            "rel_perf_thresh": row["rel_perf_thresh"],
            "atr_contract":    row["atr_contract"],
            "tp_rr":           row["tp_rr"],
            "sharpe_r":        row.get("sharpe_r", 0.0),
            "exchange":        row.get("exchange", ""),
            "sector":          row.get("sector", ""),
        }

        try:
            active, watching = scan_symbol(sym, params, spx_rets)
        except Exception as e:
            log.warning("scan_symbol(%s) raised unexpectedly: %s", sym, e)
            active, watching = [], []

        all_active.extend(active)
        all_watching.extend(watching)

    # Deduplicate: keep the most recent signal per symbol (highest entry_date)
    def _dedup(signals: list) -> list:
        seen: dict = {}
        for s in signals:
            sym = s["sym"]
            ed  = s.get("entry_date") or s.get("signal_date")
            if sym not in seen or (ed is not None and (seen[sym] is None or ed > seen[sym][0])):
                seen[sym] = (ed, s)
        return [v for _, v in seen.values()]

    all_active   = _dedup(all_active)
    all_watching = _dedup(all_watching)

    all_active.sort(key=lambda s: s["sharpe"], reverse=True)
    all_watching.sort(key=lambda s: s["sharpe"], reverse=True)
    return all_active, all_watching


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FiloHFT live signal scanner (v11 logic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cached", action="store_true",
        help="Skip MT5 connection — use cached data only (for offline testing)",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Run continuously on --interval schedule",
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Minutes between scans in --loop mode (default: 30)",
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Scan a single symbol (e.g. XOP.NYSE)",
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Only scan top N symbols by Sharpe",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to discover_live_*.csv (default: latest in backtest_results/)",
    )
    args = parser.parse_args()

    # Load and filter watchlist
    csv_path  = Path(args.csv) if args.csv else None
    watchlist = load_watchlist(csv_path)
    if watchlist.empty:
        sys.exit(1)

    if args.symbol:
        watchlist = watchlist[watchlist["sym"] == args.symbol.strip()]
        if watchlist.empty:
            log.error("Symbol %s not found in watchlist.", args.symbol)
            sys.exit(1)

    if args.top and "sharpe_r" in watchlist.columns:
        watchlist = watchlist.nlargest(args.top, "sharpe_r").reset_index(drop=True)
        log.info("Restricted to top %d symbols by Sharpe.", args.top)

    def _scan_once() -> None:
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
        mode = "CACHED" if args.cached else "LIVE (MT5)"
        sep  = "=" * 72
        log.info(sep)
        log.info("SCAN  %s  |  %d symbols  |  %s", ts, len(watchlist), mode)
        log.info(sep)

        active, watching = run_scan(watchlist, do_refresh=not args.cached)

        if not active and not watching:
            log.info("No signals.")
            return

        if active:
            log.info("\n--- ACTIVE SIGNALS (%d) ---", len(active))
            for s in active:
                log.info("\n%s", format_signal(s))

        if watching:
            log.info("\n--- WATCHING (%d) ---", len(watching))
            for s in watching:
                log.info("\n%s", format_signal(s))

        log.info("")

    if args.loop:
        try:
            while True:
                _scan_once()
                log.info("Next scan in %d min — Ctrl+C to stop\n", args.interval)
                time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            log.info("Scanner stopped.")
    else:
        _scan_once()


if __name__ == "__main__":
    main()
