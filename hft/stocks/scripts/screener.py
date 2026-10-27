"""
stocks/scripts/screener.py

MT5-only stock screener — no external web calls, no options data.
Pulls daily OHLCV from MT5, screens via LRC channel, scores via
key-level confluence + beta + momentum vs country benchmark index.

Usage:
    python stocks/scripts/screener.py
    python stocks/scripts/screener.py --sigma 1.5 --lrc 100 --days 365
    python stocks/scripts/screener.py --symbol AAPL.NAS   # single symbol debug
"""

import argparse
import io
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Force UTF-8 on Windows terminals
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
except ImportError:
    sys.exit("MetaTrader5 package not found. Run: pip install MetaTrader5")

# =============================================================================
# Config
# =============================================================================
LRC_LENGTH      = 100       # linear regression channel lookback (bars)
STD_DEV_MULT    = 1.3       # σ multiplier for band width
HISTORY_DAYS    = 730       # bars pulled per symbol (2 years daily)
PIVOT_SWING_N   = 10        # bars each side for pivot high/low detection
ROUND_LEVELS_N  = 3         # how many round-number levels to generate per symbol
BETA_WIN_SHORT  = 60        # rolling beta window (trading days)
BETA_WIN_LONG   = 252
REL_PERF_WIN    = 20        # relative performance lookback
WINSOR_P        = 0.025     # winsorization percentile for beta calc
VOL_BASELINE    = 60        # bars for volume baseline

# Score weights (must sum to 100)
W_DEVIATION     = 30
W_KEYLEVEL      = 30
W_REL_PERF      = 20
W_BETA          = 10
W_VOL_SPIKE     = 10

# Minimum score to appear in output (set 0 to show all)
MIN_SCORE       = 40

# =============================================================================
# Benchmark mapping: IC Markets suffix → index CFD ticker
# =============================================================================
SUFFIX_BENCHMARK = {
    "NYSE": "US500",
    "NAS":  "US500",
    "LSE":  "UK100",
    "ETR":  "DE40",
    "PAR":  "F40",
    "IT":   "IT40",
    "MAD":  "ES35",
    "AMS":  "NETH25",
    "IE":   "STOXX50",
    "SWX":  "SWI20",
}

ALL_BENCHMARKS = list(dict.fromkeys(SUFFIX_BENCHMARK.values()))

# =============================================================================
# MT5 helpers
# =============================================================================
def mt5_connect() -> bool:
    if not mt5.initialize():
        log.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    log.info(f"MT5 connected — {mt5.terminal_info().name} | account {info.login} | {info.server}")
    return True


def _pull_bars(symbol: str, days: int, tf=None) -> Optional[pd.DataFrame]:
    if tf is None:
        tf = mt5.TIMEFRAME_D1
    end   = datetime.utcnow()
    start = end - timedelta(days=days + 60)   # buffer for non-trading days
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_range(symbol, tf, start, end)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume"
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df.iloc[-days:] if len(df) >= days else df


CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"


def _cache_path(symbol: str) -> Path:
    safe = symbol.replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{safe}.parquet"


def _cache_is_fresh(path: Path) -> bool:
    """True if the cache file contains a bar from the last completed trading day."""
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path)
        if df.empty:
            # Empty stub = insufficient history symbol; treat as fresh so we skip MT5 re-fetch
            # Re-check weekly in case the symbol starts trading (file mtime > 7 days → stale)
            import os
            age_days = (datetime.utcnow().timestamp() - os.path.getmtime(path)) / 86400
            return age_days < 7
        last_bar = pd.to_datetime(df.index[-1], utc=True).date()
        today = datetime.utcnow().date()
        # Walk back to last weekday
        expected = today - timedelta(days=1)
        while expected.weekday() >= 5:   # Sat=5, Sun=6
            expected -= timedelta(days=1)
        return last_bar >= expected
    except Exception:
        return False


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None
        # Ensure standard column names
        df.columns = [c.capitalize() if c.lower() in ("open","high","low","close","volume") else c for c in df.columns]
        return df
    except Exception:
        return None


def _save_cache(path: Path, df: pd.DataFrame) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    except Exception as e:
        log.debug(f"Cache write failed for {path.name}: {e}")


def fetch_all_bars(symbols: List[str], days: int) -> Dict[str, pd.DataFrame]:
    """
    Pull daily bars for every symbol.
    Cached: fresh symbols load from parquet in <1ms each.
    Stale/missing symbols are fetched from MT5 and cached for next run.
    """
    out: Dict[str, pd.DataFrame] = {}
    stale: List[str] = []

    # Pass 1 — load from cache where fresh
    for sym in symbols:
        cp = _cache_path(sym)
        if _cache_is_fresh(cp):
            df = _load_cache(cp)
            if df is not None and len(df) >= LRC_LENGTH + 30:
                out[sym] = df
        else:
            stale.append(sym)

    log.info(f"Cache: {len(out)} fresh, {len(stale)} stale/missing — fetching from MT5...")

    # Pass 2 — fetch stale/missing from MT5
    total = len(stale)
    for i, sym in enumerate(stale, 1):
        if i % 100 == 0 or i == total:
            log.info(f"  MT5 pull {i}/{total}")
        df = _pull_bars(sym, days)
        if df is not None and len(df) >= LRC_LENGTH + 30:
            out[sym] = df
            _save_cache(_cache_path(sym), df)
        else:
            # Cache empty stub so we don't re-hit MT5 for this symbol daily
            _save_cache(_cache_path(sym), pd.DataFrame())

    return out


# =============================================================================
# Ticker utilities
# =============================================================================
def load_universe() -> List[str]:
    path = Path(__file__).parent.parent / "config" / "tickers.json"
    if not path.exists():
        sys.exit(f"tickers.json not found at {path}. Run list_mt5_stocks.py first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    tickers: List[str] = []
    for region_info in data.values():
        tickers.extend(region_info.get("tickers", []))
    # Drop -24 (24/5 variants) — they duplicate the base ticker
    tickers = [t for t in tickers if "-24" not in t]
    return tickers


def suffix_of(symbol: str) -> str:
    """Return IC Markets exchange suffix, e.g. 'AAPL.NAS' -> 'NAS'."""
    if "." in symbol:
        return symbol.rsplit(".", 1)[-1].upper()
    return ""


def benchmark_for(symbol: str) -> str:
    return SUFFIX_BENCHMARK.get(suffix_of(symbol), "US500")


# =============================================================================
# Signal computations
# =============================================================================
def _clean(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def calc_lrc(df: pd.DataFrame, length: int, sigma_mult: float) -> Optional[Dict]:
    close = _clean(df["Close"])
    if len(close) < length:
        return None
    window = close.iloc[-length:].values.astype(float)
    x = np.arange(length).reshape(-1, 1)
    reg = LinearRegression().fit(x, window)
    line = reg.intercept_ + reg.coef_[0] * np.arange(length)
    resid = window - line
    std = float(np.std(resid, ddof=0))
    if std == 0:
        return None
    upper = line + sigma_mult * std
    lower = line - sigma_mult * std
    price = float(close.iloc[-1])
    dev = (price - float(line[-1])) / std

    if price > float(upper[-1]):
        position = "above"
    elif price < float(lower[-1]):
        position = "below"
    else:
        return None   # inside band — skip

    return {
        "price":    price,
        "midline":  float(line[-1]),
        "upper":    float(upper[-1]),
        "lower":    float(lower[-1]),
        "std":      std,
        "dev":      dev,
        "position": position,
    }


def detect_key_levels(df: pd.DataFrame, swing_n: int = PIVOT_SWING_N) -> List[float]:
    """Return list of significant price levels from daily bars."""
    close = _clean(df["Close"])
    high  = _clean(df["High"])
    low   = _clean(df["Low"])
    if close.empty:
        return []

    levels: List[float] = []

    # 1. Pivot highs and lows (swing points)
    h = high.values
    l = low.values
    for i in range(swing_n, len(h) - swing_n):
        if h[i] == max(h[i - swing_n: i + swing_n + 1]):
            levels.append(float(h[i]))
        if l[i] == min(l[i - swing_n: i + swing_n + 1]):
            levels.append(float(l[i]))

    # 2. 52-week high and low
    tail = close.iloc[-252:] if len(close) >= 252 else close
    levels.append(float(tail.max()))
    levels.append(float(tail.min()))

    # 3. Prior 3 monthly highs/lows
    monthly = df["High"].resample("ME").max()
    monthly_l = df["Low"].resample("ME").min()
    levels.extend(monthly.iloc[-4:-1].tolist())
    levels.extend(monthly_l.iloc[-4:-1].tolist())

    # 4. Round number levels near current price
    price = float(close.iloc[-1])
    mag = 10 ** (int(np.log10(price)) - 1) if price > 0 else 1.0
    for k in range(-ROUND_LEVELS_N, ROUND_LEVELS_N + 1):
        levels.append(round(price / mag) * mag + k * mag)

    return sorted(set(round(v, 6) for v in levels if v > 0))


def nearest_level(levels: List[float], price: float) -> Tuple[Optional[float], Optional[float]]:
    """Return (nearest_level, pct_distance) from current price."""
    if not levels:
        return None, None
    dists = [(abs(lv - price) / price * 100, lv) for lv in levels]
    dists.sort()
    pct, lv = dists[0]
    return lv, pct


def _pct_returns(close: pd.Series) -> pd.Series:
    return _clean(close).pct_change().dropna()


def _winsorize(s: pd.Series, p: float) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def calc_beta(asset_close: pd.Series, bench_close: pd.Series, win: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    ar = _pct_returns(asset_close)
    br = _pct_returns(bench_close)
    idx = ar.index.intersection(br.index)
    ar, br = ar.loc[idx], br.loc[idx]
    if len(ar) < win:
        return None, None, None
    aw = _winsorize(ar.iloc[-win:], WINSOR_P)
    bw = _winsorize(br.iloc[-win:], WINSOR_P)
    X = bw.values.reshape(-1, 1)
    y = aw.values
    try:
        reg = LinearRegression().fit(X, y)
        beta = float(reg.coef_[0])
        resid = pd.Series(y - reg.predict(X), index=aw.index)
        ss_res = float(np.sum((y - reg.predict(X)) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
        resid_z = float(resid.iloc[-1] / resid.std(ddof=1)) if resid.std(ddof=1) > 0 else None
        return beta, r2, resid_z
    except Exception:
        return None, None, None


def calc_rel_perf(asset_close: pd.Series, bench_close: pd.Series, win: int = REL_PERF_WIN) -> Optional[float]:
    ac = _clean(asset_close)
    bc = _clean(bench_close)
    idx = ac.index.intersection(bc.index)
    ac, bc = ac.loc[idx], bc.loc[idx]
    if len(ac) < win + 1:
        return None
    a_ret = float(ac.iloc[-1] / ac.iloc[-win - 1] - 1.0) * 100
    b_ret = float(bc.iloc[-1] / bc.iloc[-win - 1] - 1.0) * 100
    return a_ret - b_ret


def calc_vol_spike(df: pd.DataFrame, baseline: int = VOL_BASELINE) -> Optional[float]:
    vol = df["Volume"].dropna()
    if len(vol) < baseline + 1:
        return None
    base = float(vol.iloc[-(baseline + 1):-1].mean())
    if base == 0:
        return None
    return float(vol.iloc[-1] / base)


def avg_days_to_move(close: pd.Series, pct: float) -> Optional[float]:
    c = _clean(close).values
    if len(c) < 30:
        return None
    times = []
    for i in range(len(c) - 1):
        up, dn = c[i] * (1 + pct), c[i] * (1 - pct)
        for j in range(i + 1, len(c)):
            if c[j] >= up or c[j] <= dn:
                times.append(j - i)
                break
    return float(np.mean(times)) if times else None


# =============================================================================
# Composite score (0–100)
# =============================================================================
def score_setup(
    dev: float,
    level_pct: Optional[float],
    rel_perf: Optional[float],
    beta: Optional[float],
    vol_spike: Optional[float],
    position: str,       # "above" or "below"
) -> float:
    # Deviation score: 1σ=0, 2σ=15, 3σ+=30
    dev_abs = abs(dev)
    s_dev = min(W_DEVIATION, W_DEVIATION * (dev_abs - 1.0) / 2.0) if dev_abs >= 1 else 0.0

    # Key level proximity: within 0.5%=30, within 2%=10, >2%=0
    if level_pct is None:
        s_level = 0.0
    elif level_pct <= 0.5:
        s_level = float(W_KEYLEVEL)
    elif level_pct <= 2.0:
        s_level = W_KEYLEVEL * (1 - (level_pct - 0.5) / 1.5)
    else:
        s_level = 0.0

    # Relative perf: for "below" band we want laggard (rel_perf very negative);
    # for "above" band we want outperformer (rel_perf very positive)
    if rel_perf is None:
        s_rel = 0.0
    else:
        rp_abs = abs(rel_perf)
        if position == "below":
            s_rel = min(W_REL_PERF, W_REL_PERF * rp_abs / 10.0) if rel_perf < 0 else 0.0
        else:
            s_rel = min(W_REL_PERF, W_REL_PERF * rp_abs / 10.0) if rel_perf > 0 else 0.0

    # Beta: higher beta = higher swing potential
    if beta is None:
        s_beta = float(W_BETA) * 0.5
    elif beta >= 1.5:
        s_beta = float(W_BETA)
    elif beta >= 0.8:
        s_beta = W_BETA * (beta - 0.8) / 0.7
    else:
        s_beta = 0.0

    # Volume spike
    if vol_spike is None:
        s_vol = float(W_VOL_SPIKE) * 0.5
    elif vol_spike >= 3.0:
        s_vol = float(W_VOL_SPIKE)
    elif vol_spike >= 1.5:
        s_vol = W_VOL_SPIKE * (vol_spike - 1.5) / 1.5
    else:
        s_vol = 0.0

    return round(s_dev + s_level + s_rel + s_beta + s_vol, 1)


# =============================================================================
# Main screening loop
# =============================================================================
def run_screener(
    all_data: Dict[str, pd.DataFrame],
    bench_data: Dict[str, pd.DataFrame],
    lrc_length: int,
    sigma: float,
    min_score: float = MIN_SCORE,
) -> pd.DataFrame:
    rows = []
    symbols = [s for s in all_data if s not in ALL_BENCHMARKS]

    for sym in symbols:
        df = all_data[sym]
        lrc = calc_lrc(df, lrc_length, sigma)
        if lrc is None:
            continue

        bench_name = benchmark_for(sym)
        bench_df   = bench_data.get(bench_name)

        levels     = detect_key_levels(df)
        near_lv, lv_pct = nearest_level(levels, lrc["price"])

        vol_spike  = calc_vol_spike(df)
        rel_perf   = calc_rel_perf(df["Close"], bench_df["Close"]) if bench_df is not None else None
        beta60, r2_60, resid_z60 = calc_beta(df["Close"], bench_df["Close"], BETA_WIN_SHORT) if bench_df is not None else (None, None, None)
        beta252, r2_252, _       = calc_beta(df["Close"], bench_df["Close"], BETA_WIN_LONG)  if bench_df is not None else (None, None, None)

        avg3  = avg_days_to_move(df["Close"], 0.03)
        avg5  = avg_days_to_move(df["Close"], 0.05)
        avg10 = avg_days_to_move(df["Close"], 0.10)

        score = score_setup(lrc["dev"], lv_pct, rel_perf, beta60, vol_spike, lrc["position"])
        if score < min_score:
            continue

        # Bench context
        bench_price = float(bench_df["Close"].iloc[-1]) if bench_df is not None else None
        bench_levels = detect_key_levels(bench_df) if bench_df is not None else []
        _, bench_lv_pct = nearest_level(bench_levels, bench_price) if bench_price else (None, None)

        rows.append({
            "symbol":           sym,
            "position":         lrc["position"],
            "score":            score,
            "deviation_σ":      round(lrc["dev"], 3),
            "price":            round(lrc["price"], 4),
            "lrc_midline":      round(lrc["midline"], 4),
            "lrc_upper":        round(lrc["upper"], 4),
            "lrc_lower":        round(lrc["lower"], 4),
            "nearest_level":    round(near_lv, 4) if near_lv else None,
            "level_dist_%":     round(lv_pct, 3) if lv_pct else None,
            "beta_60d":         round(beta60, 3) if beta60 is not None else None,
            "r2_60d":           round(r2_60, 3) if r2_60 is not None else None,
            "beta_252d":        round(beta252, 3) if beta252 is not None else None,
            "r2_252d":          round(r2_252, 3) if r2_252 is not None else None,
            "resid_z_60d":      round(resid_z60, 3) if resid_z60 is not None else None,
            "rel_perf_20d_%":   round(rel_perf, 3) if rel_perf is not None else None,
            "vol_spike_x":      round(vol_spike, 2) if vol_spike is not None else None,
            "avg_days_±3%":     round(avg3, 1) if avg3 else None,
            "avg_days_±5%":     round(avg5, 1) if avg5 else None,
            "avg_days_±10%":    round(avg10, 1) if avg10 else None,
            "benchmark":        bench_name,
            "bench_price":      round(bench_price, 2) if bench_price else None,
            "bench_level_dist_%": round(bench_lv_pct, 3) if bench_lv_pct else None,
        })

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df_out


# =============================================================================
# Output
# =============================================================================
def _fmt(v, d=2):
    return f"{v:.{d}f}" if (v is not None and pd.notna(v)) else "N/A"


def print_results(df: pd.DataFrame, position: str):
    label = "ABOVE UPPER BAND" if position == "above" else "BELOW LOWER BAND"
    sub = df[df["position"] == position].copy()
    print(f"\n{'='*80}")
    print(f"  {label}  ({len(sub)} setups)")
    print(f"{'='*80}")
    if sub.empty:
        print("  None"); return
    cols = ["symbol","score","deviation_σ","price","level_dist_%","nearest_level",
            "beta_60d","rel_perf_20d_%","vol_spike_x","avg_days_±5%","benchmark"]
    sub = sub.reset_index(drop=True)
    sub.insert(0, "#", sub.index + 1)
    disp = sub[["#"] + cols].copy()
    disp["deviation_σ"] = disp["deviation_σ"].map(lambda x: f"{x:.2f}σ")
    disp["price"]       = disp["price"].map(lambda x: _fmt(x, 4))
    disp["level_dist_%"]= disp["level_dist_%"].map(lambda x: f"{x:.2f}%" if x is not None and pd.notna(x) else "N/A")
    disp["nearest_level"]= disp["nearest_level"].map(lambda x: _fmt(x, 2))
    for c in ["beta_60d","rel_perf_20d_%","vol_spike_x"]:
        disp[c] = disp[c].map(lambda x: _fmt(x, 2))
    disp["avg_days_±5%"] = disp["avg_days_±5%"].map(lambda x: _fmt(x, 1))
    print(disp.to_string(index=False))


def save_csv(df: pd.DataFrame, tag: str) -> str:
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    fname = out_dir / f"screener_{tag}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(fname, index=False, encoding="utf-8-sig")
    return str(fname)


# =============================================================================
# Entry point
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma",  type=float, default=STD_DEV_MULT, help="σ band multiplier")
    ap.add_argument("--lrc",    type=int,   default=LRC_LENGTH,   help="LRC lookback bars")
    ap.add_argument("--days",   type=int,   default=HISTORY_DAYS, help="history days to pull")
    ap.add_argument("--minscore", type=float, default=MIN_SCORE,  help="minimum composite score")
    ap.add_argument("--symbol", type=str,   default=None,         help="debug single symbol")
    args = ap.parse_args()

    if not mt5_connect():
        sys.exit(1)

    # Build fetch list
    if args.symbol:
        universe = [args.symbol]
        benchmarks = [benchmark_for(args.symbol)]
    else:
        universe   = load_universe()
        benchmarks = ALL_BENCHMARKS
        log.info(f"Universe: {len(universe)} symbols + {len(benchmarks)} benchmark indices")

    fetch_list = list(dict.fromkeys(benchmarks + universe))
    log.info(f"Fetching {len(fetch_list)} symbols from MT5 ({args.days} days D1)...")

    all_data = fetch_all_bars(fetch_list, args.days)
    mt5.shutdown()
    log.info(f"Data loaded for {len(all_data)} symbols")

    bench_data = {b: all_data[b] for b in benchmarks if b in all_data}

    results = run_screener(all_data, bench_data, args.lrc, args.sigma, min_score=args.minscore)

    if results.empty:
        print("\nNo setups above minimum score. Try lowering --minscore or --sigma.")
        return

    print_results(results, "above")
    print_results(results, "below")

    csv_path = save_csv(results, f"s{args.sigma}_lrc{args.lrc}")
    print(f"\nSaved → {csv_path}")
    print(f"Total setups: {len(results)}  |  Above: {(results['position']=='above').sum()}  |  Below: {(results['position']=='below').sum()}")


if __name__ == "__main__":
    main()
