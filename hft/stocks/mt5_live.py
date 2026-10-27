"""
mt5_live.py

Intraday execution loop for the LRC mean-reversion strategy.

Architecture
------------
  Every 30 minutes during market hours (configurable):
    1. Pull fresh M30 bars from MT5 for all watchlist symbols
    2. Run scan_symbol() — same two-stage check as scanner_live:
         WATCHING  = daily LRC extension + HTF level + rel-perf confirmed
                     but M30 reversal bar NOT yet formed  → wait
         ACTIVE    = all of the above + M30 pin-bar confirmed right now
                     → place MARKET order with SL + TP immediately
    3. Skip symbols already in the book (open position or traded this session)
    4. Enforce MAX_CONCURRENT = 8 open positions

Position sizing: 1% of account balance risked at the stop level.
    lots = (balance × 0.01) / (|entry - sl| × contract_size)

Usage
-----
    python -m stocks.mt5_live              # run loop all day (market hours)
    python -m stocks.mt5_live --dry-run    # simulate — no orders sent
    python -m stocks.mt5_live --top 50     # restrict to top 50 symbols by Sharpe
    python -m stocks.mt5_live --once       # single scan pass then exit (for testing)

Schedule
--------
    Windows Task Scheduler: run at 09:30 ET (15:30 CET) Mon–Fri.
    The script self-terminates after the last scan before 16:00 ET.
"""

import argparse
import io
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import MetaTrader5 as mt5
    _HAS_MT5 = True
except ImportError:
    _HAS_MT5 = False
    print("WARNING: MetaTrader5 package not installed. Run: pip install MetaTrader5")

from stocks.backtest.scanner_live import scan_symbol, _load_spx_rets
from stocks.backtest.data.extender import _pull_mt5_m15
from stocks.backtest.config import RESULTS_DIR

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_PATH = Path(__file__).parent / "data" / "mt5_live.log"
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

# ── Fit-all optimised params  (from fit_all_sweep) ───────────────────────────
# Keys must match what scan_symbol() reads from the params dict.
LONG_PARAMS = {
    "sigma":          2.0,
    "entry_tf":       "M30",
    "tp_rr":          1.0,
    "rel_perf_thresh": -0.05,
    "atr_contract":   1.0,
}
SHORT_PARAMS = {
    "sigma":          1.5,
    "entry_tf":       "M30",
    "tp_rr":          2.0,
    "rel_perf_thresh": -0.05,
    "atr_contract":   1.0,
}

# ── Execution settings ────────────────────────────────────────────────────────
MAX_CONCURRENT  = 8
RISK_PCT        = 0.01          # 1% of balance risked per trade
MAGIC           = 202600        # identifies our orders / positions in MT5
COMMENT         = "FiloHFT"
SCAN_INTERVAL_S = 30 * 60       # 30 minutes between scans
ET              = ZoneInfo("America/New_York")
MARKET_OPEN     = (9, 30)       # ET hour, minute
MARKET_CLOSE    = (15, 55)      # stop scanning 5 min before close

# Symbol suffix candidates — tried in order until accepted by the broker
_SUFFIXES = ["", ".NAS", ".NYSE", ".US", ".OTC", "m"]
_PREFIXES = ["", "#"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _latest_csv(prefix: str = "discover_live") -> Path | None:
    files = sorted(RESULTS_DIR.glob(f"{prefix}_*.csv"), reverse=True)
    return files[0] if files else None


def _in_market_hours() -> bool:
    now = datetime.now(ET)
    t = (now.hour, now.minute)
    return MARKET_OPEN <= t <= MARKET_CLOSE and now.weekday() < 5


def _next_scan_time() -> datetime:
    return datetime.now(ET) + timedelta(seconds=SCAN_INTERVAL_S)


# ── Symbol resolution ─────────────────────────────────────────────────────────
_sym_cache: dict[str, str | None] = {}

def resolve_symbol(bare: str) -> str | None:
    if bare in _sym_cache:
        return _sym_cache[bare]
    for prefix in _PREFIXES:
        for suffix in _SUFFIXES:
            candidate = f"{prefix}{bare}{suffix}"
            info = mt5.symbol_info(candidate)
            if info is not None:
                mt5.symbol_select(candidate, True)
                _sym_cache[bare] = candidate
                return candidate
    _sym_cache[bare] = None
    return None


# ── Position sizing ───────────────────────────────────────────────────────────

def calc_lots(mt5_sym: str, entry_px: float, sl_px: float, balance: float) -> float | None:
    """
    Risk exactly RISK_PCT of balance at the stop level.
    For US stock CFDs where 1 lot = 1 share:
        lots = (balance × 0.01) / |entry - sl|
    """
    info = mt5.symbol_info(mt5_sym)
    if info is None:
        return None
    risk_dist = abs(entry_px - sl_px)
    if risk_dist < 1e-8:
        return None
    dollar_risk   = balance * RISK_PCT
    contract_size = info.trade_contract_size  # shares per lot
    raw_lots      = dollar_risk / (risk_dist * contract_size)
    step          = info.volume_step if info.volume_step > 0 else 0.01
    lots          = max(info.volume_min, (raw_lots // step) * step)
    lots          = min(lots, info.volume_max)
    return round(lots, 8)


# ── MT5 state helpers ─────────────────────────────────────────────────────────

def _strip_suffix(name: str) -> str:
    for sfx in _SUFFIXES[1:]:
        if name.endswith(sfx):
            return name[:-len(sfx)]
    for pfx in _PREFIXES[1:]:
        if name.startswith(pfx):
            return name[len(pfx):]
    return name


def get_open_syms() -> set[str]:
    """Bare symbol names with an open position belonging to this strategy."""
    positions = mt5.positions_get() or []
    return {_strip_suffix(p.symbol) for p in positions if p.magic == MAGIC}


def count_open() -> int:
    return len([p for p in (mt5.positions_get() or []) if p.magic == MAGIC])


# ── Order placement ───────────────────────────────────────────────────────────

def place_market_order(sig: dict, balance: float, dry_run: bool = False) -> bool:
    """
    Place a market order for an ACTIVE (M30-confirmed) signal.
    SL and TP are attached at order time.
    """
    mt5_sym = resolve_symbol(sig["sym"])
    if mt5_sym is None:
        log.warning("  Symbol not in MT5: %s", sig["sym"])
        return False

    info = mt5.symbol_info(mt5_sym)
    if info is None or not info.visible:
        log.warning("  Not tradeable: %s", mt5_sym)
        return False

    direction = sig["direction"]
    entry_px  = sig["entry_px"]
    sl_px     = sig["sl"]
    tp_px     = sig["tp"]

    lots = calc_lots(mt5_sym, entry_px, sl_px, balance)
    if lots is None or lots <= 0:
        log.warning("  Cannot size %s  (lots=%.4f)", sig["sym"], lots or 0)
        return False

    # Get live bid/ask
    tick = mt5.symbol_info_tick(mt5_sym)
    if tick is None:
        log.warning("  No tick data for %s", mt5_sym)
        return False

    if direction == "LONG":
        order_type = mt5.ORDER_TYPE_BUY
        price      = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price      = tick.bid

    # Normalise SL/TP to broker tick
    pt = info.point
    def _norm(px): return round(round(px / pt) * pt, 8)
    sl_px = _norm(sl_px)
    tp_px = _norm(tp_px)

    log.info(
        "  %-5s %-12s  price=%.4f  sl=%.4f  tp=%.4f  lots=%.2f  "
        "R=%.1f  htf=%.0f  sigma=%.1f %s",
        direction, mt5_sym, price, sl_px, tp_px, lots,
        sig["tp_rr"], sig["htf_score"], sig["sigma"], sig["entry_tf"],
    )

    if dry_run:
        log.info("  [DRY-RUN] Order NOT sent")
        return True

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       mt5_sym,
        "volume":       lots,
        "type":         order_type,
        "price":        price,
        "sl":           sl_px,
        "tp":           tp_px,
        "deviation":    20,           # max slippage in points
        "magic":        MAGIC,
        "comment":      COMMENT,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result is None:
        log.error("  order_send returned None for %s", mt5_sym)
        return False
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info("  -> Deal #%d filled @ %.4f", result.deal, result.price)
        return True
    else:
        log.error("  -> FAILED  retcode=%d  %s", result.retcode, result.comment)
        return False


# ── Data refresh ──────────────────────────────────────────────────────────────

def refresh_m15_data(symbols: list[str], dry_run: bool = False):
    """Pull fresh M30 bars from MT5 into the local cache before scanning."""
    if dry_run or not _HAS_MT5:
        return
    for sym in symbols:
        try:
            _pull_mt5_m15(sym)
        except Exception:
            pass  # non-fatal; scanner will use cached data


# ── Single scan cycle ─────────────────────────────────────────────────────────

def run_scan_cycle(live: pd.DataFrame, spx_rets: dict,
                   session_traded: set, dry_run: bool) -> int:
    """
    Scan all symbols, place market orders for new ACTIVE signals.
    Returns number of orders placed this cycle.
    """
    open_syms   = get_open_syms()
    open_count  = count_open()
    placed      = 0
    n_active    = 0
    n_watching  = 0
    balance     = mt5.account_info().balance if not dry_run else 1_000.0

    for _, row in live.iterrows():
        sym = row["sym"]

        # Run scan twice: once with LONG params, once with SHORT params
        for params in (LONG_PARAMS, SHORT_PARAMS):
            try:
                active, watching = scan_symbol(sym, params, spx_rets)
            except Exception as e:
                log.debug("scan_symbol failed %s: %s", sym, e)
                continue

            n_active  += len(active)
            n_watching += len(watching)

            for sig in active:
                direction = sig["direction"]
                key       = (sym, direction)

                # Already traded this session or open position
                if key in session_traded or sym in open_syms:
                    continue

                # Cap check
                if open_count >= MAX_CONCURRENT:
                    log.info("Max concurrent (%d) reached — holding remaining signals",
                             MAX_CONCURRENT)
                    return placed

                ok = place_market_order(sig, balance, dry_run=dry_run)
                if ok:
                    placed      += 1
                    open_count  += 1
                    open_syms.add(sym)
                    session_traded.add(key)

    log.info("Scan complete — ACTIVE: %d  WATCHING: %d  Placed: %d  Book: %d/%d",
             n_active, n_watching, placed, open_count, MAX_CONCURRENT)
    return placed


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",     default=None)
    ap.add_argument("--top",     type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Simulate — no orders sent")
    ap.add_argument("--once",    action="store_true", help="Single scan pass then exit")
    args = ap.parse_args()

    log.info("=" * 64)
    log.info("FiloHFT MT5 Intraday Executor%s",
             "  [DRY-RUN]" if args.dry_run else "")
    log.info("LONG  params: sigma=%.1f | %s | %.0fR | rel=%.2f",
             LONG_PARAMS["sigma"], LONG_PARAMS["entry_tf"],
             LONG_PARAMS["tp_rr"], LONG_PARAMS["rel_perf_thresh"])
    log.info("SHORT params: sigma=%.1f | %s | %.0fR | rel=%.2f",
             SHORT_PARAMS["sigma"], SHORT_PARAMS["entry_tf"],
             SHORT_PARAMS["tp_rr"], SHORT_PARAMS["rel_perf_thresh"])
    log.info("=" * 64)

    # ── Load watchlist ────────────────────────────────────────────────────────
    live_csv = Path(args.csv) if args.csv else _latest_csv("discover_live")
    if live_csv is None:
        log.error("No discover_live CSV found. Run discover_all.py first.")
        sys.exit(1)
    live = pd.read_csv(live_csv)
    if args.top:
        live = live.nlargest(args.top, "sharpe_r")
    log.info("Watchlist: %s  (%d symbols)", live_csv.name, len(live))

    # ── Connect to MT5 ────────────────────────────────────────────────────────
    if not _HAS_MT5 and not args.dry_run:
        log.error("MetaTrader5 package not installed.")
        sys.exit(1)
    if not args.dry_run:
        if not mt5.initialize():
            log.error("MT5 initialize() failed: %s", mt5.last_error())
            sys.exit(1)
        acct = mt5.account_info()
        log.info("MT5 connected  |  Account #%d  |  Balance=%.2f %s",
                 acct.login, acct.balance, acct.currency)

    spx_rets       = _load_spx_rets()
    session_traded : set = set()   # (sym, direction) pairs traded today
    symbols        = live["sym"].tolist()

    # ── Loop ─────────────────────────────────────────────────────────────────
    while True:
        now_et = datetime.now(ET)

        if not _in_market_hours() and not args.once:
            if now_et.weekday() < 5:
                log.info("Outside market hours (%s ET) — waiting...",
                         now_et.strftime("%H:%M"))
            time.sleep(60)
            continue

        log.info("--- Scan cycle %s ET ---", now_et.strftime("%Y-%m-%d %H:%M"))

        # Pull fresh M30 data from MT5 into local cache
        if not args.dry_run:
            log.info("Refreshing M30 data for %d symbols...", len(symbols))
            refresh_m15_data(symbols)

        run_scan_cycle(live, spx_rets, session_traded, dry_run=args.dry_run)

        if args.once:
            break

        # Sleep until next 30-min mark
        next_scan = _next_scan_time()
        sleep_s   = (next_scan - datetime.now(ET)).total_seconds()
        log.info("Next scan at %s ET  (%.0f min)",
                 next_scan.strftime("%H:%M"), sleep_s / 60)
        time.sleep(max(sleep_s, 1))

    if not args.dry_run and _HAS_MT5:
        mt5.shutdown()
    log.info("MT5 executor stopped.")


if __name__ == "__main__":
    main()
