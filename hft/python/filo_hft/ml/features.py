"""Feature extraction for the ML scoring layer.

This module bridges the two possible data sources (live telemetry CSVs and
backtest parquet output) into a single ``(X, y, meta)`` dataset that the
training driver can consume.

Both sources are designed, by construction, to share column names — so the
same :func:`build_dataset` builder works on either. The only difference is
how you load the raw frames:

* telemetry → :func:`load_from_telemetry` walks ``telemetry/YYYYMMDD/`` dirs,
  reads ``setups.csv`` / ``intents.csv`` / ``fills.csv`` / ``trades.csv`` and
  joins them on ``intent_id`` / ``ticket``. Only setups whose ``decision`` is
  ``"accepted"`` AND whose trade has a ``close`` event with ``realized_usd``
  end up as training rows.
* backtest → :func:`load_from_backtest` is just a thin
  :func:`pandas.read_parquet` wrapper.

The join logic is deliberately tolerant of missing streams (a dev day may
have setups but no fills yet), returning an empty DataFrame rather than
crashing so the CLI can skip the day cleanly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from filo_hft.config import FiloConfig
from filo_hft.data.loaders import load_telemetry


# ----------------------------------------------------------------------
# Schema constants

#: The 7 setup families, in a fixed order that pins one-hot column layout.
FAMILIES: tuple[str, ...] = (
    "breakout",
    "retest",
    "rejection",
    "fakeout",
    "failed_auction",
    "engulfing_accel",
    "mean_reversion",
)

#: The authoritative feature column list used by the trained model.
#:
#: Order matters — the trained model expects columns in exactly this order.
FEATURE_COLUMNS: list[str] = [
    "intrinsic_score",
    "kl_weight",
    "kl_distance_pips",
    "chan_pos_norm",
    "spread_pips",
    "velocity",
    "corr_xagusd",
    "corr_dxy",
    *[f"family_{f}" for f in FAMILIES],
    "direction_long",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


# ----------------------------------------------------------------------
# Data loading — Source A: live telemetry CSVs


def _discover_telemetry_days(telemetry_root: Path) -> list[str]:
    """Return sorted YYYYMMDD dir names under ``telemetry_root``."""
    if not telemetry_root.exists():
        return []
    return sorted(
        p.name
        for p in telemetry_root.iterdir()
        if p.is_dir() and p.name.isdigit() and len(p.name) == 8
    )


def load_from_telemetry(
    telemetry_root: str | Path,
    days: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the (trades, candidates) pair from live telemetry CSVs.

    The returned frames have column names matched to the backtest parquet
    output so the downstream :func:`build_dataset` works uniformly.

    Parameters
    ----------
    telemetry_root:
        Directory that contains ``YYYYMMDD/`` subdirs.
    days:
        Iterable of ``YYYYMMDD`` strings. If ``None``, all days discovered
        under ``telemetry_root`` are used.

    Returns
    -------
    (trades_df, candidates_df):
        ``candidates_df`` is the full setup table (accepted + blocked).
        ``trades_df`` is the per-setup realized-PnL view, one row per
        closed trade, annotated with the originating setup features.
    """
    root = Path(telemetry_root)
    if days is None:
        day_list = _discover_telemetry_days(root)
    else:
        day_list = list(days)

    all_candidates: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []

    for day in day_list:
        setups = load_telemetry(root, day, "setups")
        intents = load_telemetry(root, day, "intents")
        fills = load_telemetry(root, day, "fills")
        trades_raw = load_telemetry(root, day, "trades")

        if setups.empty:
            continue

        # --- candidates: every setup the engine considered ----------------
        cand = setups.copy()
        cand["day"] = day
        # ts_ms → ts_open for uniformity with the backtest schema; the
        # candidate's setup time IS its "open" time for training purposes.
        if "ts_ms" in cand.columns:
            cand["ts_open"] = pd.to_datetime(cand["ts_ms"], unit="ms", utc=True)
        all_candidates.append(cand)

        # --- trades: join accepted setups → intents → fills → trades -------
        accepted = setups[setups.get("decision", "") == "accepted"].copy()
        if accepted.empty or intents.empty or fills.empty or trades_raw.empty:
            continue

        # Setups don't carry intent_id directly; the EA writes an intent row
        # with state='created' at the same ts_ms for the setup it was built
        # from. We join setup → created-intent by (ts_ms, family, direction).
        created = intents[intents.get("state", "") == "created"].copy()
        if created.empty:
            continue
        created = created[["ts_ms", "intent_id", "family", "direction"]]
        accepted_with_intent = accepted.merge(
            created,
            on=["ts_ms", "family", "direction"],
            how="inner",
        )
        if accepted_with_intent.empty:
            continue

        # intent_id → ticket via fills
        fills_slim = fills[["intent_id", "ticket"]].drop_duplicates("intent_id")
        with_ticket = accepted_with_intent.merge(fills_slim, on="intent_id", how="inner")
        if with_ticket.empty:
            continue

        # ticket → realized_usd via trades close event
        closes = trades_raw[trades_raw.get("event", "") == "close"].copy()
        if closes.empty:
            continue
        closes_slim = closes[["ticket", "ts_ms", "realized_usd"]].rename(
            columns={"ts_ms": "ts_close_ms"}
        )

        merged = with_ticket.merge(closes_slim, on="ticket", how="inner")
        if merged.empty:
            continue

        merged["ts_open"] = pd.to_datetime(merged["ts_ms"], unit="ms", utc=True)
        merged["ts_close"] = pd.to_datetime(merged["ts_close_ms"], unit="ms", utc=True)
        merged["day"] = day
        all_trades.append(merged)

    trades_df = (
        pd.concat(all_trades, ignore_index=True)
        if all_trades
        else pd.DataFrame()
    )
    candidates_df = (
        pd.concat(all_candidates, ignore_index=True)
        if all_candidates
        else pd.DataFrame()
    )
    return trades_df, candidates_df


# ----------------------------------------------------------------------
# Data loading — Source C: shadow_trades.csv (virtual positions)


def load_from_shadow(
    telemetry_root: str | Path,
    days: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (trades, candidates) from ``shadow_trades.csv``.

    The EA opens a virtual position for every accepted setup (see
    Shadow.mqh) and closes it on the same hidden TP/SL distances the real
    execution would have used. Each row is already a closed trade with
    entry features inlined, so no join is needed — this makes shadow the
    fastest path to a labeled training set when live trading is gated off.

    The returned ``candidates_df`` is empty: shadow_trades.csv only records
    accepted setups. The training loop tolerates an empty candidates frame.
    """
    root = Path(telemetry_root)
    if days is None:
        day_list = _discover_telemetry_days(root)
    else:
        day_list = list(days)

    frames: list[pd.DataFrame] = []
    for day in day_list:
        df = load_telemetry(root, day, "shadow_trades")
        if df.empty:
            continue
        df = df.copy()
        df["day"] = day
        df["ts_open"] = pd.to_datetime(df["ts_open"], unit="ms", utc=True)
        df["ts_close"] = pd.to_datetime(df["ts_close"], unit="ms", utc=True)
        # Strip the entry_ prefix so build_dataset's canonical column
        # lookups (velocity, spread_pips, chan_pos_norm, corr_*) hit.
        df = df.rename(
            columns={
                "entry_velocity": "velocity",
                "entry_spread_pips": "spread_pips",
                "entry_chan_pos_norm": "chan_pos_norm",
                "entry_corr_xagusd": "corr_xagusd",
                "entry_corr_dxy": "corr_dxy",
            }
        )
        frames.append(df)

    trades_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return trades_df, pd.DataFrame()


# ----------------------------------------------------------------------
# Data loading — Source B: backtest parquet


def load_from_backtest(
    trades_path: str | Path,
    candidates_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the (trades, candidates) pair from backtest parquet files."""
    trades_df = pd.read_parquet(trades_path)
    candidates_df = pd.read_parquet(candidates_path)

    # Normalize ts_open to pandas datetime for downstream sorting.
    for df in (trades_df, candidates_df):
        if "ts_open" in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df["ts_open"].dtype
        ):
            df["ts_open"] = pd.to_datetime(df["ts_open"], utc=True, errors="coerce")

    return trades_df, candidates_df


# ----------------------------------------------------------------------
# Feature engineering


def _one_hot_family(series: pd.Series) -> pd.DataFrame:
    """Produce a fixed-layout one-hot for ``family`` (7 columns)."""
    s = series.fillna("unknown").astype(str)
    out = pd.DataFrame(
        {f"family_{f}": (s == f).astype(np.int8) for f in FAMILIES},
        index=series.index,
    )
    return out


def _cyclic(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Return (sin, cos) cyclic encoding of ``values`` with given period."""
    theta = 2.0 * np.pi * (values.astype(float) / period)
    return np.sin(theta), np.cos(theta)


def _is_datetime_dtype(dtype) -> bool:
    """True for naive or tz-aware datetime64 dtypes.

    ``np.issubdtype`` raises on tz-aware pandas dtypes, so we use
    :func:`pandas.api.types.is_datetime64_any_dtype` which handles both.
    """
    return bool(pd.api.types.is_datetime64_any_dtype(dtype))


def _coerce_ts(df: pd.DataFrame) -> pd.Series:
    """Return a UTC-datetime ``ts_open`` column, inferring if needed."""
    if "ts_open" in df.columns:
        ts = df["ts_open"]
        if not _is_datetime_dtype(ts.dtype):
            ts = pd.to_datetime(ts, utc=True, errors="coerce")
        return ts
    if "ts_ms" in df.columns:
        return pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    raise KeyError("trades frame lacks ts_open / ts_ms — cannot derive time features")


def build_dataset(
    trades: pd.DataFrame,
    candidates: pd.DataFrame,  # noqa: ARG001 — kept in signature for future use
    cfg: FiloConfig,  # noqa: ARG001 — kept for symmetry; no config knobs today
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build the ``(X, y, meta)`` dataset from joined trades.

    Only the ``trades`` frame is consumed today — it is already the inner
    join of accepted-setup → filled → closed, which is precisely the labeled
    training set. ``candidates`` is accepted as a parameter so future work
    (e.g. learning to predict block reasons, or treating blocked setups as
    negative examples with a different label) can use it without changing
    the call site.

    The label is simple and honest:

    ``y = (realized_usd > 0).astype(int)``

    i.e. any net-positive trade is a positive outcome. Breakeven closes
    (realized_usd == 0) are labeled negative on purpose — we want the model
    to bias toward setups that actually move.
    """
    if trades.empty:
        empty_X = pd.DataFrame(columns=FEATURE_COLUMNS)
        empty_y = pd.Series(dtype=np.int8, name="y")
        empty_meta = pd.DataFrame(
            columns=["ts_open", "family", "intent_id", "realized_usd"]
        )
        return empty_X, empty_y, empty_meta

    df = trades.copy()

    # --- time features -------------------------------------------------
    ts_open = _coerce_ts(df)
    df["ts_open"] = ts_open
    df = df.dropna(subset=["ts_open"])
    if df.empty:
        return build_dataset(df.iloc[0:0], candidates, cfg)  # pragma: no cover

    hour = df["ts_open"].dt.hour.astype(float)
    dow = df["ts_open"].dt.dayofweek.astype(float)
    hour_sin, hour_cos = _cyclic(hour, 24.0)
    dow_sin, dow_cos = _cyclic(dow, 7.0)

    # --- numeric features with safe defaults --------------------------
    def _num(col: str, default: float = 0.0) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    def _num_any(cols: list[str], default: float = 0.0) -> pd.Series:
        for col in cols:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    X = pd.DataFrame(index=df.index)
    X["intrinsic_score"] = _num("intrinsic_score")
    X["kl_weight"] = _num("kl_weight")
    X["kl_distance_pips"] = _num("kl_distance_pips")
    X["chan_pos_norm"] = _num("chan_pos_norm")
    X["spread_pips"] = _num_any(["spread_pips", "entry_spread_pips"])
    X["velocity"] = _num("velocity")
    X["corr_xagusd"] = _num("corr_xagusd")
    X["corr_dxy"] = _num("corr_dxy")

    family_col = df["family"] if "family" in df.columns else pd.Series(
        "unknown", index=df.index
    )
    X = pd.concat([X, _one_hot_family(family_col)], axis=1)

    if "direction" in df.columns:
        X["direction_long"] = (
            df["direction"].astype(str).str.lower().eq("long").astype(np.int8)
        )
    else:
        X["direction_long"] = np.int8(0)

    X["hour_sin"] = hour_sin
    X["hour_cos"] = hour_cos
    X["dow_sin"] = dow_sin
    X["dow_cos"] = dow_cos

    # reorder to the canonical column list (in case concat reordered)
    X = X[FEATURE_COLUMNS].astype(np.float32)

    # --- label ---------------------------------------------------------
    if "realized_usd" in df.columns:
        realized_src = df["realized_usd"]
    elif "usd" in df.columns:
        realized_src = df["usd"]
    else:
        realized_src = pd.Series(0.0, index=df.index)
    realized = pd.to_numeric(realized_src, errors="coerce").fillna(0.0)
    y = (realized > 0.0).astype(np.int8).rename("y")

    # --- meta ---------------------------------------------------------
    meta = pd.DataFrame(
        {
            "ts_open": df["ts_open"].values,
            "family": family_col.astype(str).values,
            "intent_id": (
                df["intent_id"].astype(str).values
                if "intent_id" in df.columns
                else np.arange(len(df)).astype(str)
            ),
            "realized_usd": realized.values,
        },
        index=df.index,
    )

    # Stable, time-ordered index (important for the train/val split later).
    order = np.argsort(meta["ts_open"].values, kind="stable")
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    meta = meta.iloc[order].reset_index(drop=True)

    return X, y, meta
