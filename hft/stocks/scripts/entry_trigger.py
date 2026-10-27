"""
stocks/scripts/entry_trigger.py

Entry trigger layer — runs on top of screener output.
For each screener setup:
  1. Pulls W1 + D1 + H4 bars → detects key levels per timeframe
  2. Checks if current price is within touch distance of a confluent level
  3. Pulls M15 bars → measures velocity + acceleration toward the mean
  4. Scores the entry signal (0–100)
  5. Outputs trade parameters: entry, SL, TP, holding estimate

All data from MT5 only — no external web calls.

Usage:
    python stocks/scripts/entry_trigger.py
    python stocks/scripts/entry_trigger.py --csv stocks/data/screener_s1.3_lrc100_20260423_0902.csv
    python stocks/scripts/entry_trigger.py --symbol AAPL.NAS
    python stocks/scripts/entry_trigger.py --ltf M5 --touch-pct 0.8
"""

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
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

# Timeframes for key level detection (ordered highest → lowest)
HTF_KEY_LEVELS = ["W1", "D1", "H4"]

# Entry confirmation timeframe
LTF_DEFAULT = "M15"

# Touch distance: price must be within X% of a key level to qualify
TOUCH_PCT = 0.75

# Velocity: LTF bars to look back for momentum measurement
VELOCITY_BARS   = 4      # how many LTF bars to measure velocity over
ACCEL_BARS      = 2      # bars for acceleration confirmation
ATR_PERIOD      = 14     # ATR for normalisation

# Min bars to pull per timeframe
HTF_BARS = {
    "W1": 104,   # 2 years weekly
    "D1": 365,
    "H4": 252,   # ~6 weeks of H4
}
LTF_BARS = {
    "M15": 192,  # 2 trading days
    "M30": 96,
    "H1":  96,
    "M5":  288,
}

# Pivot swing detection (bars each side per HTF)
PIVOT_N = {
    "W1": 4,
    "D1": 8,
    "H4": 6,
}

# Minimum entry score to print
MIN_ENTRY_SCORE = 50

# Score weights
W_LEVEL_CONFLUENCE = 35   # how many TFs confirm the level
W_LEVEL_PROXIMITY  = 25   # how close price is to the level
W_VELOCITY         = 25   # magnitude of LTF reversal velocity
W_ACCELERATION     = 15   # is velocity increasing?

TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
}

# =============================================================================
# MT5 helpers
# =============================================================================

def mt5_connect() -> bool:
    if not mt5.initialize():
        log.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return False
    info = mt5.account_info()
    log.info(f"MT5 connected — {mt5.terminal_info().name} | {info.login} | {info.server}")
    return True


def pull_bars(symbol: str, tf_str: str, n_bars: int) -> Optional[pd.DataFrame]:
    tf = TF_MAP.get(tf_str.upper())
    if tf is None:
        return None
    mt5.symbol_select(symbol, True)
    # Use from_pos: faster than range for recent data
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time").rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume",
    })
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# =============================================================================
# Level importance weights
# =============================================================================

# Higher timeframe = exponentially more significant
TF_WEIGHT: Dict[str, float] = {
    "W1":  4.0,
    "D1":  2.5,
    "H4":  1.5,
    "ALL": 1.0,   # round numbers
}

# Level type weights: prior period levels > clean pivots > round numbers
KIND_WEIGHT: Dict[str, float] = {
    "period_high": 3.0,
    "period_low":  3.0,
    "pivot_high":  2.0,
    "pivot_low":   2.0,
    "round":       1.5,
}

# Confluence multiplier: each additional confirming TF adds more weight
CONFLUENCE_MULT = {1: 1.0, 2: 1.6, 3: 2.5}


# =============================================================================
# Key level detection per HTF
# =============================================================================

class KeyLevel(NamedTuple):
    price:      float
    tf:         str            # "W1", "D1", "H4"
    kind:       str            # "pivot_high", "pivot_low", "period_high", "period_low", "round"
    strength:   int            # count of pivot bars that confirmed it (1–N)


def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    if len(df) < period + 1:
        return float((df["High"] - df["Low"]).mean())
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.iloc[-period:].mean())


def _pivot_levels(df: pd.DataFrame, tf: str) -> List[KeyLevel]:
    """
    Detect pivot highs/lows. Strength = how many consecutive bars on each side
    respect the pivot (deeper swing = higher strength, capped at 5).
    """
    n = PIVOT_N.get(tf, 5)
    h = df["High"].values
    l = df["Low"].values
    out: List[KeyLevel] = []
    for i in range(n, len(h) - n):
        if h[i] >= max(h[i - n: i]) and h[i] >= max(h[i + 1: i + n + 1]):
            # Count how far the bars around the pivot stay below it
            left  = sum(1 for j in range(i - 1, max(0, i - n * 2) - 1, -1) if h[j] < h[i])
            right = sum(1 for j in range(i + 1, min(len(h), i + n * 2 + 1))  if h[j] < h[i])
            strength = min(5, 1 + (left + right) // 3)
            out.append(KeyLevel(float(h[i]), tf, "pivot_high", strength))
        if l[i] <= min(l[i - n: i]) and l[i] <= min(l[i + 1: i + n + 1]):
            left  = sum(1 for j in range(i - 1, max(0, i - n * 2) - 1, -1) if l[j] > l[i])
            right = sum(1 for j in range(i + 1, min(len(l), i + n * 2 + 1))  if l[j] > l[i])
            strength = min(5, 1 + (left + right) // 3)
            out.append(KeyLevel(float(l[i]), tf, "pivot_low", strength))
    return out


def _period_levels(df: pd.DataFrame, tf: str) -> List[KeyLevel]:
    """Prior completed period highs/lows (weekly → monthly for D1, daily for H4, etc.)"""
    out: List[KeyLevel] = []
    if tf == "D1":
        # Prior 4 weekly high/lows
        wh = df["High"].resample("W").max().iloc[-5:-1]
        wl = df["Low"].resample("W").min().iloc[-5:-1]
        for v in wh: out.append(KeyLevel(float(v), tf, "period_high", 2))
        for v in wl: out.append(KeyLevel(float(v), tf, "period_low",  2))
    elif tf == "H4":
        # Prior 5 daily high/lows
        dh = df["High"].resample("D").max().iloc[-6:-1]
        dl = df["Low"].resample("D").min().iloc[-6:-1]
        for v in dh: out.append(KeyLevel(float(v), tf, "period_high", 2))
        for v in dl: out.append(KeyLevel(float(v), tf, "period_low",  2))
    elif tf == "W1":
        # Prior 3 monthly high/lows
        mh = df["High"].resample("ME").max().iloc[-4:-1]
        ml = df["Low"].resample("ME").min().iloc[-4:-1]
        for v in mh: out.append(KeyLevel(float(v), tf, "period_high", 3))
        for v in ml: out.append(KeyLevel(float(v), tf, "period_low",  3))
    return out


def _round_levels(price: float, atr: float) -> List[KeyLevel]:
    """Psychological round-number levels within 3 ATRs of current price."""
    if price <= 0 or atr <= 0:
        return []
    mag = 10 ** max(0, int(np.log10(price)) - 1)
    levels = []
    base = round(price / mag) * mag
    for k in range(-5, 6):
        lv = base + k * mag
        if abs(lv - price) <= 3 * atr and lv > 0:
            levels.append(KeyLevel(round(lv, 6), "ALL", "round", 1))
    return levels


def get_key_levels(
    htf_data: Dict[str, pd.DataFrame],
    current_price: float,
) -> List[KeyLevel]:
    """Collect all key levels from W1 + D1 + H4."""
    all_levels: List[KeyLevel] = []
    for tf in HTF_KEY_LEVELS:
        df = htf_data.get(tf)
        if df is None or df.empty:
            continue
        all_levels.extend(_pivot_levels(df, tf))
        all_levels.extend(_period_levels(df, tf))

    # Round levels (once, not per-TF)
    atr_d1 = _atr(htf_data["D1"]) if "D1" in htf_data and htf_data["D1"] is not None else current_price * 0.01
    all_levels.extend(_round_levels(current_price, atr_d1))

    return all_levels


def cluster_levels(
    levels: List[KeyLevel],
    current_price: float,
    atr: float,
    cluster_tol: float = 0.005,   # 0.5% cluster radius
) -> List[Dict]:
    """
    Merge nearby levels into clusters. A cluster touched by W1 + D1 + H4
    is far stronger than a single-TF level.
    Returns list of cluster dicts sorted by distance to current price.
    """
    if not levels:
        return []

    prices = np.array([lv.price for lv in levels])
    used   = [False] * len(levels)
    clusters = []

    for i, lv in enumerate(levels):
        if used[i]:
            continue
        # Find all levels within cluster_tol of this one
        mask = np.abs(prices - lv.price) / lv.price <= cluster_tol
        members = [levels[j] for j in range(len(levels)) if mask[j]]
        for j in range(len(levels)):
            if mask[j]:
                used[j] = True

        tfs_present   = list({m.tf for m in members} - {"ALL"})
        round_present = any(m.kind == "round" for m in members)
        avg_price     = float(np.mean([m.price for m in members]))

        # --- Level importance score (0–100) ---
        # Raw: sum of TF_weight × kind_weight × pivot_strength per member
        raw = sum(
            TF_WEIGHT.get(m.tf, 1.0) * KIND_WEIGHT.get(m.kind, 1.0) * m.strength
            for m in members
        )
        # Confluence multiplier: more TFs = exponentially more significant
        c_mult      = CONFLUENCE_MULT.get(min(len(tfs_present), 3), 1.0)
        round_bonus = 8.0 if round_present else 0.0
        level_score = min(100.0, raw * c_mult + round_bonus)

        clusters.append({
            "price":          round(avg_price, 6),
            "tfs":            tfs_present,
            "tf_count":       len(tfs_present),
            "has_round":      round_present,
            "level_score":    round(level_score, 1),
            "dist_pct":       abs(avg_price - current_price) / current_price * 100,
            "side":           "below" if avg_price < current_price else "above",
            "kinds":          list({m.kind for m in members}),
        })

    clusters.sort(key=lambda c: c["dist_pct"])
    return clusters


# =============================================================================
# LTF velocity + acceleration
# =============================================================================

def compute_velocity(
    ltf_df: pd.DataFrame,
    direction: str,     # "up" (long) or "down" (short)
    v_bars: int = VELOCITY_BARS,
    atr_period: int = ATR_PERIOD,
) -> Dict:
    """
    Measure how strongly price is moving in `direction` on the LTF.

    Returns dict with:
      velocity      — normalised (ATR units/bar) over v_bars
      acceleration  — velocity of last ACCEL_BARS vs prior ACCEL_BARS
      body_score    — candle body size as fraction of ATR (last 3 bars avg)
      close_pos     — close position within bar range [-1..+1], +1=closed at high
      confirmed     — bool: velocity direction matches trade direction
    """
    if ltf_df is None or len(ltf_df) < v_bars + atr_period + 5:
        return {"velocity": 0.0, "acceleration": 0.0, "body_score": 0.0,
                "close_pos": 0.0, "confirmed": False}

    df = ltf_df.copy()
    atr_val = _atr(df, atr_period)
    if atr_val == 0:
        return {"velocity": 0.0, "acceleration": 0.0, "body_score": 0.0,
                "close_pos": 0.0, "confirmed": False}

    closes = df["Close"].values

    # Velocity over last v_bars
    raw_move    = float(closes[-1] - closes[-v_bars - 1])
    velocity    = raw_move / (atr_val * v_bars)

    # Acceleration: compare velocity of last ACCEL_BARS vs prior ACCEL_BARS
    v_recent    = float(closes[-1] - closes[-ACCEL_BARS - 1]) / (atr_val * ACCEL_BARS)
    v_prior     = float(closes[-ACCEL_BARS - 1] - closes[-2 * ACCEL_BARS - 1]) / (atr_val * ACCEL_BARS)
    acceleration = v_recent - v_prior

    # Candle body quality (last 3 bars)
    bodies = (df["Close"].iloc[-3:] - df["Open"].iloc[-3:]).abs()
    body_score = float(bodies.mean() / atr_val)

    # Close position within bar range (last bar): +1 = closed at high, -1 = at low
    last = df.iloc[-1]
    rng  = last["High"] - last["Low"]
    close_pos = float((last["Close"] - last["Low"]) / rng * 2 - 1) if rng > 0 else 0.0

    # Confirmation: are we actually moving in the right direction?
    if direction == "up":
        confirmed = velocity > 0.1 and close_pos > 0
    else:
        confirmed = velocity < -0.1 and close_pos < 0

    return {
        "velocity":     round(velocity, 4),
        "acceleration": round(acceleration, 4),
        "body_score":   round(body_score, 4),
        "close_pos":    round(close_pos, 4),
        "confirmed":    confirmed,
    }


# =============================================================================
# Entry scoring
# =============================================================================

def score_entry(
    cluster: Dict,
    vel: Dict,
    touch_pct: float,
) -> float:
    # Level quality: directly from the cluster's importance score (0–100 → scaled to weight)
    s_conf = W_LEVEL_CONFLUENCE * (cluster["level_score"] / 100.0)

    # Proximity: within 0.25% = full, within touch_pct = linear decay
    dist = cluster["dist_pct"]
    if dist <= 0.25:
        s_prox = float(W_LEVEL_PROXIMITY)
    elif dist <= touch_pct:
        s_prox = W_LEVEL_PROXIMITY * (1 - (dist - 0.25) / (touch_pct - 0.25))
    else:
        s_prox = 0.0

    # Velocity: normalised abs velocity, capped at 2 ATRs/bar = full marks
    v_abs = min(abs(vel["velocity"]), 2.0)
    s_vel = W_VELOCITY * (v_abs / 2.0) if vel["confirmed"] else 0.0

    # Acceleration: positive acceleration in right direction
    accel_sign = vel["acceleration"] if vel["velocity"] >= 0 else -vel["acceleration"]
    if accel_sign > 0.05:
        s_accel = min(W_ACCELERATION, W_ACCELERATION * accel_sign / 0.5)
    else:
        s_accel = 0.0

    return round(s_conf + s_prox + s_vel + s_accel, 1)


# =============================================================================
# SL / TP calculation
# =============================================================================

def calc_trade_params(
    position:      str,         # "above" or "below"
    entry_price:   float,
    cluster_price: float,
    lrc_midline:   float,
    atr_d1:        float,
) -> Dict:
    """
    position "above" → mean-reversion SHORT
    position "below" → mean-reversion LONG
    """
    sl_buffer = atr_d1 * 0.5   # SL sits 0.5 ATR beyond the level

    if position == "above":
        # SHORT: sell near resistance, SL above level, TP at midline
        sl       = cluster_price + sl_buffer
        tp       = lrc_midline
        risk     = abs(entry_price - sl)
        reward   = abs(entry_price - tp)
        direction = "SHORT"
    else:
        # LONG: buy near support, SL below level, TP at midline
        sl       = cluster_price - sl_buffer
        tp       = lrc_midline
        risk     = abs(entry_price - sl)
        reward   = abs(entry_price - tp)
        direction = "LONG"

    rr = round(reward / risk, 2) if risk > 0 else None
    return {
        "direction": direction,
        "entry":     round(entry_price, 4),
        "sl":        round(sl, 4),
        "tp":        round(tp, 4),
        "risk_pts":  round(risk, 4),
        "reward_pts":round(reward, 4),
        "rr":        rr,
    }


# =============================================================================
# Per-symbol analysis
# =============================================================================

def analyse_symbol(
    symbol:      str,
    position:    str,          # "above" | "below" from screener
    lrc_midline: float,
    screener_score: float,
    ltf_str:     str,
    touch_pct:   float,
) -> Optional[Dict]:
    """Pull HTF + LTF bars, detect levels, measure velocity, score entry."""

    # Pull all required bars
    htf_data: Dict[str, pd.DataFrame] = {}
    for tf, n in HTF_BARS.items():
        df = pull_bars(symbol, tf, n)
        if df is not None and not df.empty:
            htf_data[tf] = df

    if "D1" not in htf_data:
        log.debug(f"{symbol}: no D1 data — skipping")
        return None

    ltf_df = pull_bars(symbol, ltf_str, LTF_BARS.get(ltf_str, 192))

    # Current price from D1 last bar (or LTF if available)
    current_price = float(htf_data["D1"]["Close"].iloc[-1])
    if ltf_df is not None and not ltf_df.empty:
        current_price = float(ltf_df["Close"].iloc[-1])

    atr_d1 = _atr(htf_data["D1"])

    # Key levels
    all_levels = get_key_levels(htf_data, current_price)
    clusters   = cluster_levels(all_levels, current_price, atr_d1)

    if not clusters:
        return None

    # Best level: closest within touch_pct on the correct side
    # For "above" band (short): look for resistance ABOVE or very close above price
    # For "below" band (long):  look for support BELOW or very close below price
    target_side = "above" if position == "above" else "below"

    # Primary: same-side level (resistance for short, support for long)
    same_side = [c for c in clusters if c["side"] == target_side and c["dist_pct"] <= touch_pct]
    # Fallback: price already pierced through (dist_pct very small, other side)
    fallback   = [c for c in clusters if c["dist_pct"] <= touch_pct * 0.3]

    best = None
    if same_side:
        best = same_side[0]
    elif fallback:
        best = fallback[0]

    if best is None:
        return None

    # Velocity on LTF
    direction = "down" if position == "above" else "up"
    vel = compute_velocity(ltf_df, direction) if ltf_df is not None else {
        "velocity": 0.0, "acceleration": 0.0, "body_score": 0.0,
        "close_pos": 0.0, "confirmed": False,
    }

    entry_score = score_entry(best, vel, touch_pct)
    if entry_score < MIN_ENTRY_SCORE:
        return None

    trade = calc_trade_params(position, current_price, best["price"], lrc_midline, atr_d1)

    return {
        "symbol":            symbol,
        "position":          position,
        "screener_score":    screener_score,
        "entry_score":       entry_score,
        "combined_score":    round((screener_score + entry_score) / 2, 1),
        **trade,
        "level_price":       best["price"],
        "level_score":       best["level_score"],
        "level_tfs":         "+".join(sorted(best["tfs"])),
        "level_tf_count":    best["tf_count"],
        "level_has_round":   best["has_round"],
        "level_dist_%":      round(best["dist_pct"], 3),
        "velocity":          vel["velocity"],
        "acceleration":      vel["acceleration"],
        "body_score":        vel["body_score"],
        "vel_confirmed":     vel["confirmed"],
        "ltf_used":          ltf_str,
        "current_price":     round(current_price, 4),
        "lrc_midline":       round(lrc_midline, 4),
        "atr_d1":            round(atr_d1, 4),
    }


# =============================================================================
# Batch run over screener output
# =============================================================================

def find_latest_screener_csv() -> Optional[Path]:
    data_dir = Path(__file__).parent.parent / "data"
    csvs = sorted(data_dir.glob("screener_*.csv"), reverse=True)
    return csvs[0] if csvs else None


def load_screener(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {"symbol", "position", "lrc_midline", "score"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"Screener CSV missing columns: {missing}")
    return df


def run_entry_scan(
    screener_df: pd.DataFrame,
    ltf_str: str,
    touch_pct: float,
    symbol_filter: Optional[str] = None,
    min_score: float = MIN_ENTRY_SCORE,
) -> pd.DataFrame:
    global MIN_ENTRY_SCORE
    MIN_ENTRY_SCORE = min_score
    if symbol_filter:
        screener_df = screener_df[screener_df["symbol"] == symbol_filter]

    signals = []
    total = len(screener_df)
    log.info(f"Running entry trigger on {total} screener setups (LTF={ltf_str}, touch={touch_pct}%)...")

    for i, row in enumerate(screener_df.itertuples(), 1):
        if i % 50 == 0 or i == total:
            log.info(f"  {i}/{total}")
        result = analyse_symbol(
            symbol       = row.symbol,
            position     = row.position,
            lrc_midline  = float(row.lrc_midline),
            screener_score = float(row.score),
            ltf_str      = ltf_str,
            touch_pct    = touch_pct,
        )
        if result:
            signals.append(result)

    if not signals:
        return pd.DataFrame()

    out = pd.DataFrame(signals).sort_values("combined_score", ascending=False).reset_index(drop=True)
    return out


# =============================================================================
# Output
# =============================================================================

SIGNAL_COLS = [
    "symbol", "combined_score", "entry_score", "screener_score",
    "direction", "current_price", "entry", "sl", "tp", "rr",
    "level_price", "level_score", "level_tfs", "level_tf_count", "level_has_round", "level_dist_%",
    "velocity", "acceleration", "body_score", "vel_confirmed",
    "atr_d1", "ltf_used",
]


def print_signals(df: pd.DataFrame):
    for direction in ("LONG", "SHORT"):
        sub = df[df["direction"] == direction].reset_index(drop=True)
        print(f"\n{'='*90}")
        print(f"  {direction} SIGNALS  ({len(sub)})")
        print(f"{'='*90}")
        if sub.empty:
            print("  None"); continue
        sub.insert(0, "#", sub.index + 1)
        disp = sub[["#"] + SIGNAL_COLS].copy()
        for c in ["combined_score","entry_score","screener_score","level_score","velocity","acceleration","body_score"]:
            disp[c] = disp[c].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
        for c in ["current_price","entry","sl","tp","level_price","atr_d1"]:
            disp[c] = disp[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        disp["rr"]            = disp["rr"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        disp["level_dist_%"]  = disp["level_dist_%"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        disp["vel_confirmed"] = disp["vel_confirmed"].map(lambda x: "YES" if x else "NO")
        disp["level_has_round"] = disp["level_has_round"].map(lambda x: "YES" if x else "NO")
        print(disp.to_string(index=False))


def save_signals(df: pd.DataFrame) -> str:
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    fname = out_dir / f"signals_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(fname, index=False, encoding="utf-8-sig")
    return str(fname)


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",       default=None,        help="Screener CSV path (default: latest in stocks/data/)")
    ap.add_argument("--symbol",    default=None,        help="Run on single symbol only")
    ap.add_argument("--ltf",       default=LTF_DEFAULT, help=f"LTF for velocity (M5/M15/M30/H1, default {LTF_DEFAULT})")
    ap.add_argument("--touch-pct", type=float, default=TOUCH_PCT, help="Max %% distance from key level")
    ap.add_argument("--min-score", type=float, default=MIN_ENTRY_SCORE, help="Min entry score to show")
    args = ap.parse_args()

    # Resolve screener CSV
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = find_latest_screener_csv()
        if csv_path is None:
            sys.exit("No screener CSV found. Run screener.py first.")
        log.info(f"Using screener: {csv_path.name}")

    if not mt5_connect():
        sys.exit(1)

    screener_df = load_screener(csv_path)
    log.info(f"Loaded {len(screener_df)} screener setups")

    signals = run_entry_scan(screener_df, args.ltf.upper(), args.touch_pct, args.symbol, min_score=args.min_score)

    mt5.shutdown()

    if signals.empty:
        print(f"\nNo entry signals above score {MIN_ENTRY_SCORE}. Try --touch-pct 1.5 or --min-score 40.")
        return

    print_signals(signals)

    csv_out = save_signals(signals)
    print(f"\nSaved → {csv_out}")
    print(f"Total signals: {len(signals)}  |  LONG: {(signals['direction']=='LONG').sum()}  |  SHORT: {(signals['direction']=='SHORT').sum()}")


if __name__ == "__main__":
    main()
