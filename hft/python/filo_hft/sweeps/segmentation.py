"""Segment backtest trade logs by session / hour / dow / setup_family / KL tf.

The main consumer is the sweep-results dashboard: it calls
`segment_trades` with a subset of dimensions to produce conditional
P&L / win-rate / expectancy tables, and `session_by_hour_heatmap` for a
24x7 heatmap.

Session boundaries are read from `config/sessions.json`. KL weight
buckets are derived from the MQL5 weight constants stored on
`KeylevelsCfg` (D1=1.00, H4=0.75, H1=0.55, M30=0.40, M15=0.28, M5=0.18).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# config constants


# MQL5-aligned key-level weights by timeframe. The values here are the
# canonical defaults from `KeylevelsCfg`; the buckets are derived from
# them so that a trade's `kl_weight` can be binned qualitatively without
# having to know the timeframe.
KL_WEIGHTS_BY_TF: dict[str, float] = {
    "D1": 1.00,
    "H4": 0.75,
    "H1": 0.55,
    "M30": 0.40,
    "M15": 0.28,
    "M5": 0.18,
}

# Qualitative buckets:
#   low  : M5 / M15 zones  (weight < 0.33, halfway between M15 and M30)
#   mid  : M30 / H1 zones  (0.33 <= weight < 0.65, halfway between H1 and H4)
#   high : H4 / D1 zones   (weight >= 0.65)
# Boundaries picked so each bucket straddles two native weights and the
# frontiers don't fall on a native value.
KL_WEIGHT_BUCKETS: dict[str, tuple[float, float]] = {
    "low": (-np.inf, 0.33),
    "mid": (0.33, 0.65),
    "high": (0.65, np.inf),
}


# ---------------------------------------------------------------------------
# session lookup


def _default_sessions_path() -> Path:
    """Locate `config/sessions.json` relative to the repo root."""
    # This file lives at:
    #   <root>/python/filo_hft/sweeps/segmentation.py
    # Config lives at:
    #   <root>/config/sessions.json
    here = Path(__file__).resolve()
    root = here.parents[3]
    return root / "config" / "sessions.json"


@lru_cache(maxsize=4)
def _load_session_table(path: str | None = None) -> tuple[tuple[int, int, str], ...]:
    """Read sessions.json once and memoize.

    Returns a tuple of (start_minute, end_minute, name) in priority
    order — highest priority first — so the first hit wins.
    """
    p = Path(path) if path else _default_sessions_path()
    if not p.exists():
        # Sensible fallback matching the shipped default.
        raw = {
            "sessions": [
                {"name": "asia",          "start": "00:00", "end": "07:00", "priority": 1},
                {"name": "london_open",   "start": "07:00", "end": "09:00", "priority": 3},
                {"name": "london",        "start": "09:00", "end": "12:00", "priority": 2},
                {"name": "overlap_ln_ny", "start": "12:00", "end": "16:00", "priority": 4},
                {"name": "ny",            "start": "16:00", "end": "20:00", "priority": 3},
                {"name": "offhours",      "start": "20:00", "end": "24:00", "priority": 0},
            ]
        }
    else:
        raw = json.loads(p.read_text(encoding="utf-8"))

    sessions = list(raw.get("sessions", []))
    # sort by priority desc so the first match wins when windows overlap
    sessions.sort(key=lambda s: s.get("priority", 0), reverse=True)

    table: list[tuple[int, int, str]] = []
    for entry in sessions:
        start_min = _hhmm_to_minutes(entry["start"])
        end_min = _hhmm_to_minutes(entry["end"])
        table.append((start_min, end_min, entry["name"]))
    return tuple(table)


def _hhmm_to_minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def session_of(ts: pd.Timestamp, *, sessions_path: str | None = None) -> str:
    """Classify a UTC timestamp into one of the configured session names.

    If the timestamp is naive it is treated as UTC. If no window
    matches, returns "offhours".
    """
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        utc_ts = ts
    else:
        utc_ts = ts.tz_convert("UTC")
    minute = utc_ts.hour * 60 + utc_ts.minute

    table = _load_session_table(sessions_path)
    for start_min, end_min, name in table:
        if start_min <= minute < end_min:
            return name
        # handle wrap-around windows (end < start) e.g. 22:00 -> 02:00
        if end_min < start_min and (minute >= start_min or minute < end_min):
            return name
    return "offhours"


def _session_series(ts_series: pd.Series) -> pd.Series:
    """Vectorised-ish session classification for a Series of timestamps."""
    if ts_series.empty:
        return pd.Series([], dtype=object)

    ts = pd.to_datetime(ts_series, utc=True, errors="coerce")
    minutes = ts.dt.hour * 60 + ts.dt.minute

    table = _load_session_table()
    # Priority order is already baked in. For each row, walk the table.
    out = np.full(len(minutes), "offhours", dtype=object)
    covered = np.zeros(len(minutes), dtype=bool)
    mins_arr = minutes.to_numpy()
    for start_min, end_min, name in table:
        if end_min >= start_min:
            mask = (mins_arr >= start_min) & (mins_arr < end_min)
        else:
            mask = (mins_arr >= start_min) | (mins_arr < end_min)
        assign = mask & (~covered)
        out[assign] = name
        covered |= assign
        if covered.all():
            break
    return pd.Series(out, index=ts_series.index, dtype=object)


# ---------------------------------------------------------------------------
# metrics over a group of trades


def _sharpe(pnl: pd.Series) -> float:
    if len(pnl) < 2:
        return float("nan")
    std = float(pnl.std(ddof=1))
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(pnl.mean() / std * np.sqrt(len(pnl)))


def _max_drawdown_pct(pnl: pd.Series) -> float:
    if pnl.empty:
        return float("nan")
    equity = pnl.cumsum()
    # treat start-of-run as zero, ignore negative peaks by using a
    # rolling max over the cumulative curve
    peak = equity.cummax()
    drawdown = equity - peak
    # express as pct of running peak when peak > 0; otherwise use absolute dd
    # against a nominal 1.0 bankroll so we always return a finite number
    denom = peak.replace(0.0, np.nan)
    dd_pct = (drawdown / denom).fillna(drawdown)
    return float(dd_pct.min() * 100.0)


def _group_metrics(group: pd.DataFrame) -> pd.Series:
    pnl_usd = group.get("usd")
    if pnl_usd is None:
        pnl_usd = pd.Series(dtype=float)
    pips = group.get("pips")
    if pips is None:
        pips = pd.Series(dtype=float)

    pnl_usd = pd.to_numeric(pnl_usd, errors="coerce").fillna(0.0)
    pips = pd.to_numeric(pips, errors="coerce").fillna(0.0)

    n = int(len(group))
    wins = int((pnl_usd > 0).sum())
    win_rate = float(wins / n) if n else float("nan")

    return pd.Series(
        {
            "trade_count": n,
            "pnl_usd": float(pnl_usd.sum()),
            "win_rate": win_rate,
            "expectancy_pips": float(pips.mean()) if n else float("nan"),
            "sharpe": _sharpe(pnl_usd),
            "max_dd_pct": _max_drawdown_pct(pnl_usd),
        }
    )


# ---------------------------------------------------------------------------
# feature enrichment


_ALLOWED_DIMENSIONS: frozenset[str] = frozenset(
    {
        "session",
        "hour",
        "day_of_week",
        "setup_family",
        "kl_timeframe",
        "kl_weight_bucket",
    }
)


def _kl_weight_bucket(weight: float) -> str:
    if pd.isna(weight):
        return "unknown"
    for name, (lo, hi) in KL_WEIGHT_BUCKETS.items():
        if lo <= float(weight) < hi:
            return name
    return "unknown"


def _kl_timeframe_for_weight(weight: float) -> str:
    """Best-effort reverse-lookup: which MQL5 TF is this weight closest to?"""
    if pd.isna(weight):
        return "unknown"
    w = float(weight)
    closest_tf = min(
        KL_WEIGHTS_BY_TF.items(),
        key=lambda kv: abs(kv[1] - w),
    )[0]
    return closest_tf


def _ensure_dimensions(trades: pd.DataFrame, dimensions: Sequence[str]) -> pd.DataFrame:
    """Add any dimension columns the caller asked for but that aren't present."""
    df = trades.copy()

    if df.empty:
        for d in dimensions:
            if d not in df.columns:
                df[d] = pd.Series(dtype=object)
        return df

    # canonical timestamp for deriving session/hour/dow
    ts_col = None
    for candidate in ("ts_open", "ts", "open_time", "time"):
        if candidate in df.columns:
            ts_col = candidate
            break

    if "session" in dimensions and "session" not in df.columns:
        if ts_col is not None:
            df["session"] = _session_series(df[ts_col])
        else:
            df["session"] = "offhours"

    if "hour" in dimensions and "hour" not in df.columns:
        if ts_col is not None:
            df["hour"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.hour
        else:
            df["hour"] = np.nan

    if "day_of_week" in dimensions and "day_of_week" not in df.columns:
        if ts_col is not None:
            df["day_of_week"] = pd.to_datetime(
                df[ts_col], utc=True, errors="coerce"
            ).dt.dayofweek
        else:
            df["day_of_week"] = np.nan

    if "setup_family" in dimensions and "setup_family" not in df.columns:
        if "family" in df.columns:
            df["setup_family"] = df["family"]
        else:
            df["setup_family"] = "unknown"

    if "kl_timeframe" in dimensions and "kl_timeframe" not in df.columns:
        if "kl_weight" in df.columns:
            df["kl_timeframe"] = df["kl_weight"].map(_kl_timeframe_for_weight)
        else:
            df["kl_timeframe"] = "unknown"

    if "kl_weight_bucket" in dimensions and "kl_weight_bucket" not in df.columns:
        if "kl_weight" in df.columns:
            df["kl_weight_bucket"] = df["kl_weight"].map(_kl_weight_bucket)
        else:
            df["kl_weight_bucket"] = "unknown"

    return df


# ---------------------------------------------------------------------------
# public API


def segment_trades(
    trades: pd.DataFrame,
    dimensions: Sequence[str],
) -> pd.DataFrame:
    """Group `trades` by the requested dimensions and summarise each group.

    Parameters
    ----------
    trades
        A trade log (as emitted by the backtest engine). Must contain
        `usd` and `pips` columns for PnL / expectancy; any other
        dimension columns that aren't already present will be derived
        from `ts_open`, `family` and `kl_weight`.
    dimensions
        Any subset of {"session", "hour", "day_of_week",
        "setup_family", "kl_timeframe", "kl_weight_bucket"}.

    Returns
    -------
    pd.DataFrame
        One row per group with `trade_count, pnl_usd, win_rate,
        expectancy_pips, sharpe, max_dd_pct`. Sorted by pnl_usd desc.
    """
    dims = list(dimensions)
    for d in dims:
        if d not in _ALLOWED_DIMENSIONS:
            raise ValueError(
                f"unknown dimension {d!r}; allowed: {sorted(_ALLOWED_DIMENSIONS)}"
            )
    if not dims:
        raise ValueError("segment_trades requires at least one dimension")

    if trades is None or trades.empty:
        # return a frame with the right schema so downstream code
        # doesn't need special-case handling
        cols = dims + [
            "trade_count",
            "pnl_usd",
            "win_rate",
            "expectancy_pips",
            "sharpe",
            "max_dd_pct",
        ]
        return pd.DataFrame(columns=cols)

    enriched = _ensure_dimensions(trades, dims)

    grouped = enriched.groupby(dims, dropna=False, sort=False, as_index=False)
    summary = grouped.apply(_group_metrics).reset_index(drop=True)

    # `groupby.apply` sometimes loses the dimension columns depending on
    # pandas version; re-derive them from the original grouper object
    # to keep behaviour stable.
    if not all(d in summary.columns for d in dims):
        summary = (
            enriched.groupby(dims, dropna=False, sort=False)
            .apply(_group_metrics)
            .reset_index()
        )

    summary = summary.sort_values("pnl_usd", ascending=False).reset_index(drop=True)
    return summary


def session_by_hour_heatmap(trades: pd.DataFrame) -> pd.DataFrame:
    """Return a 24-hour x 7-day-of-week pivot of expectancy (pips).

    Rows are day-of-week (Mon=0..Sun=6), columns are UTC hour 0..23.
    Empty cells are NaN. Suitable for direct plotting by the dashboard.
    """
    if trades is None or trades.empty:
        return pd.DataFrame(
            data=np.nan,
            index=pd.RangeIndex(7, name="day_of_week"),
            columns=pd.RangeIndex(24, name="hour"),
        )

    df = _ensure_dimensions(trades, ["hour", "day_of_week"])
    if "pips" not in df.columns:
        df = df.assign(pips=np.nan)

    # coerce types so pivot_table doesn't explode on weird dtypes
    df = df.assign(
        hour=pd.to_numeric(df["hour"], errors="coerce").astype("Int64"),
        day_of_week=pd.to_numeric(df["day_of_week"], errors="coerce").astype("Int64"),
        pips=pd.to_numeric(df["pips"], errors="coerce"),
    )
    df = df.dropna(subset=["hour", "day_of_week"])

    pivot = df.pivot_table(
        index="day_of_week",
        columns="hour",
        values="pips",
        aggfunc="mean",
    )

    # reindex so every hour/day is present even if unobserved
    pivot = pivot.reindex(index=range(7), columns=range(24))
    pivot.index.name = "day_of_week"
    pivot.columns.name = "hour"
    return pivot


# ---------------------------------------------------------------------------
# tiny utility used by the CLI / tests


def available_dimensions() -> frozenset[str]:
    return _ALLOWED_DIMENSIONS


__all__ = [
    "KL_WEIGHT_BUCKETS",
    "KL_WEIGHTS_BY_TF",
    "available_dimensions",
    "segment_trades",
    "session_by_hour_heatmap",
    "session_of",
]
