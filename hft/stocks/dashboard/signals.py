"""
Scan the US liquid universe for live LRC mean-reversion signals.
Returns a DataFrame of actionable setups with entry, SL, TP, and position size.
"""
import pickle
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd

from stocks.backtest.config import BT_CACHE_DIR, RESULTS_DIR
from stocks.backtest.data.extender import load_bt_bars
from stocks.backtest.data.universe import build_universe
from stocks.backtest.data.loader import _atr_series
from stocks.backtest.signals.lrc_signal import compute_lrc_signals
from stocks.backtest.signals.entry_signal import compute_entry_signals

# Best entry config from sweep (entry_id=66)
LRC_LENGTH  = 120
SIGMA_MULT  = 1.0
MIN_SCORE   = 40
TOUCH_PCT   = 1.5
VEL_REQ     = False

# Variant lev+0.3atr|rr5|none
SL_ATR_MULT = 0.3
TP_RR       = 5.0

US_EXCHANGES = {"NYSE", "NAS"}
TMP_US = RESULTS_DIR / "_tmp_us"


def _load_us_liquid_symbols() -> Dict[str, dict]:
    """Return {symbol: {exchange, sector}} for the US liquid universe."""
    univ = build_universe()
    us = univ[univ["exchange"].isin(US_EXCHANGES)].copy()

    # Apply same liquid filter: top-25% by dollar-volume median
    scores = {}
    for _, row in us.iterrows():
        sym = row["symbol"]
        df = load_bt_bars(sym)
        if df.empty or "Volume" not in df.columns or len(df) < 200:
            continue
        dvol = (df["Close"] * df["Volume"]).replace(0, np.nan).dropna()
        if len(dvol) >= 200:
            scores[sym] = float(dvol.median())

    if not scores:
        return {}
    s = pd.Series(scores)
    thresh = float(s.quantile(0.75))
    liquid = set(s[s >= thresh].index)

    meta = {}
    for _, row in us[us["symbol"].isin(liquid)].iterrows():
        meta[row["symbol"]] = {"exchange": row["exchange"], "sector": row.get("sector", "Other")}
    return meta


def _vol_tier(atr_pct: float, low_t: float, high_t: float) -> str:
    if np.isnan(atr_pct):  return "unknown"
    if atr_pct <= low_t:   return "low_vol"
    if atr_pct <= high_t:  return "mid_vol"
    return "high_vol"


def scan(
    account: float = 100_000,
    risk_pct: float = 0.01,
    lrc_length: int = LRC_LENGTH,
    sigma_mult: float = SIGMA_MULT,
    min_score: float = MIN_SCORE,
    touch_pct: float = TOUCH_PCT,
    max_signal_age_bars: int = 3,
) -> pd.DataFrame:
    """
    Scan every US liquid stock for today's LRC signal.
    Returns a DataFrame with one row per setup, sorted by combined_score desc.
    """
    meta = _load_us_liquid_symbols()
    if not meta:
        return pd.DataFrame()

    # Vol tier thresholds from current universe
    atr_pcts = {}
    bars_cache: Dict[str, pd.DataFrame] = {}

    for sym in meta:
        df = load_bt_bars(sym)
        if df.empty or len(df) < 50:
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df["atr"] = _atr_series(df)
        df = df.dropna(subset=["atr"])
        if len(df) < 60:
            continue
        bars_cache[sym] = df
        hl = df["High"] - df["Low"]
        atr14 = hl.rolling(14).mean()
        med = (atr14 / df["Close"]).dropna().median()
        if not np.isnan(med):
            atr_pcts[sym] = float(med)

    if atr_pcts:
        s = pd.Series(atr_pcts)
        low_t  = float(s.quantile(0.33))
        high_t = float(s.quantile(0.67))
    else:
        low_t, high_t = 0.018, 0.025

    rows = []
    for sym, df in bars_cache.items():
        try:
            sf = compute_lrc_signals(df, lrc_length, sigma_mult)
            ef = compute_entry_signals(df, touch_pct, VEL_REQ)
        except Exception:
            continue

        last = len(df) - 1

        # Position must be outside band on last bar
        pos_last = int(sf["position"].iloc[last])
        if pos_last == 0:
            continue

        # Check signal is fresh: position first crossed outside within last max_signal_age_bars
        age = 0
        for k in range(1, max_signal_age_bars + 2):
            if last - k < 0:
                break
            if sf["position"].iloc[last - k] == pos_last:
                age += 1
            else:
                break
        if age > max_signal_age_bars:
            continue   # stale — price has been outside band too long

        # Entry validity
        if not bool(ef["valid"].iloc[last]):
            continue

        lrc_score   = float(sf["lrc_score"].iloc[last])
        level_score = float(ef["level_score"].iloc[last])
        combined    = (lrc_score + level_score) / 2.0
        if combined < min_score:
            continue

        # Trade levels
        close      = float(df["Close"].iloc[last])
        atr_val    = float(df["atr"].iloc[last])
        lrc_mid    = float(sf["lrc_mid"].iloc[last])
        level_px   = float(ef["level_price"].iloc[last])
        direction  = pos_last   # +1=short, -1=long

        if np.isnan(level_px) or level_px <= 0:
            level_px = close * (1 - direction * 0.02)

        # SL: level_price + 0.3*ATR (buffer beyond level)
        sl  = level_px + direction * SL_ATR_MULT * atr_val
        risk_per_share = abs(close - sl)
        if risk_per_share < 0.001:
            continue

        # TP: 5R from entry
        tp = close - direction * TP_RR * risk_per_share

        # Position size
        risk_usd  = account * risk_pct
        shares    = risk_usd / risk_per_share
        notional  = shares * close

        m = meta[sym]
        rows.append({
            "symbol":        sym,
            "exchange":      m["exchange"],
            "sector":        m["sector"],
            "vol_tier":      _vol_tier(atr_pcts.get(sym, float("nan")), low_t, high_t),
            "direction":     "SHORT" if direction == +1 else "LONG",
            "last_close":    round(close, 2),
            "entry_est":     round(close, 2),   # tomorrow's open — use close as estimate
            "sl":            round(sl, 2),
            "tp":            round(tp, 2),
            "risk_per_share":round(risk_per_share, 4),
            "shares":        int(round(shares)),
            "notional":      round(notional, 0),
            "risk_usd":      round(risk_usd, 0),
            "rr":            TP_RR,
            "combined_score":round(combined, 1),
            "lrc_score":     round(lrc_score, 1),
            "level_score":   round(level_score, 1),
            "dev_sigma":     round(float(sf["dev_sigma"].iloc[last]), 2),
            "signal_age_bars": age,
            "data_date":     str(df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1])[:10],
            "atr":           round(atr_val, 4),
            "lrc_mid":       round(lrc_mid, 2),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("combined_score", ascending=False).reset_index(drop=True)
    return out


def scan_all(
    account: float = 100_000,
    risk_pct: float = 0.01,
    lrc_length: int = LRC_LENGTH,
    sigma_mult: float = SIGMA_MULT,
    min_score: float = MIN_SCORE,
    touch_pct: float = TOUCH_PCT,
) -> pd.DataFrame:
    """
    Returns ALL US liquid stocks with a proximity-to-signal score (0-100).
    Stocks with an active signal have status='SIGNAL'; others show how close they are.
    """
    meta = _load_us_liquid_symbols()
    if not meta:
        return pd.DataFrame()

    atr_pcts = {}
    bars_cache: Dict[str, pd.DataFrame] = {}
    for sym in meta:
        df = load_bt_bars(sym)
        if df.empty or len(df) < 60:
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df["atr"] = _atr_series(df)
        df = df.dropna(subset=["atr"])
        if len(df) < 60:
            continue
        bars_cache[sym] = df
        hl = df["High"] - df["Low"]
        atr14 = hl.rolling(14).mean()
        med = (atr14 / df["Close"]).dropna().median()
        if not np.isnan(med):
            atr_pcts[sym] = float(med)

    if atr_pcts:
        s = pd.Series(atr_pcts)
        low_t  = float(s.quantile(0.33))
        high_t = float(s.quantile(0.67))
    else:
        low_t, high_t = 0.018, 0.025

    rows = []
    for sym, df in bars_cache.items():
        try:
            sf = compute_lrc_signals(df, lrc_length, sigma_mult)
            ef = compute_entry_signals(df, touch_pct, VEL_REQ)
        except Exception:
            continue

        last       = len(df) - 1
        dev_sigma  = float(sf["dev_sigma"].iloc[last])
        pos_last   = int(sf["position"].iloc[last])
        close      = float(df["Close"].iloc[last])
        atr_val    = float(df["atr"].iloc[last])
        lrc_mid    = float(sf["lrc_mid"].iloc[last])
        level_px   = float(ef["level_price"].iloc[last])
        level_dist = float(ef["level_dist_%"].iloc[last])
        valid_entry = bool(ef["valid"].iloc[last])

        lrc_score   = float(sf["lrc_score"].iloc[last])
        level_score = float(ef["level_score"].iloc[last])
        combined    = (lrc_score + level_score) / 2.0

        # Proximity: sigma component (how far towards / past the band)
        sigma_prox = min(100.0, abs(dev_sigma) / sigma_mult * 100.0)

        # Proximity: level component
        if level_dist <= touch_pct:
            level_prox = (touch_pct - level_dist) / touch_pct * 100.0
        else:
            level_prox = max(0.0, (3.5 - level_dist) / 3.5 * 40.0)

        readiness = sigma_prox * 0.6 + level_prox * 0.4

        # Explicit condition breakdown
        lrc_met   = (pos_last != 0)
        level_met = (level_dist <= touch_pct)
        score_met = (combined >= min_score)

        is_signal = lrc_met and level_met and score_met
        if is_signal:
            readiness = max(readiness, combined)
            status    = "SIGNAL"
            condition = "LRC ✓  LVL ✓"
        elif lrc_met and not level_met:
            status    = "NEAR"
            condition = "LRC ✓  LVL ✗"   # outside band — wait for level touch
        elif not lrc_met and level_met:
            status    = "WATCH"
            condition = "LRC ✗  LVL ✓"   # near level — wait for LRC extension
        elif readiness >= 45:
            status    = "WATCH"
            condition = "LRC ✗  LVL ✗"
        else:
            status    = "COLD"
            condition = "—"

        direction = "SHORT" if pos_last == +1 else ("LONG" if pos_last == -1 else
                    ("SHORT?" if dev_sigma > 0 else "LONG?"))

        sl, tp, shares, notional = float("nan"), float("nan"), 0, 0.0
        if is_signal and not np.isnan(level_px) and level_px > 0:
            direction_int = pos_last
            sl_price = level_px + direction_int * SL_ATR_MULT * atr_val
            risk_per_share = abs(close - sl_price)
            if risk_per_share > 0.001:
                sl       = round(sl_price, 2)
                tp       = round(close - direction_int * TP_RR * risk_per_share, 2)
                shares   = int(round((account * risk_pct) / risk_per_share))
                notional = round(shares * close, 0)

        m = meta[sym]
        rows.append({
            "symbol":     sym,
            "exchange":   m["exchange"],
            "sector":     m["sector"],
            "vol_tier":   _vol_tier(atr_pcts.get(sym, float("nan")), low_t, high_t),
            "status":     status,
            "condition":  condition,
            "readiness":  round(readiness, 1),
            "direction":  direction,
            "dev_sigma":  round(dev_sigma, 2),
            "level_dist": round(level_dist, 2),
            "close":      round(close, 2),
            "sl":         sl,
            "tp":         tp,
            "shares":     shares,
            "notional":   notional,
            "risk_usd":   round(account * risk_pct, 0),
            "atr":        round(atr_val, 4),
            "lrc_mid":    round(lrc_mid, 2),
            "data_date":  str(df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1])[:10],
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["readiness", "symbol"], ascending=[False, True])
        .reset_index(drop=True)
    )
