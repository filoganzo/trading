"""Read-side helpers for the Streamlit observability dashboard.

Thin wrappers around :mod:`filo_hft.data.loaders` that add the bits the UI
needs: locating the latest ``YYYYMMDD`` directory, listing available days,
best-effort timestamp normalisation, and batched tail-reads for every
stream in one call.

The module deliberately knows nothing about Streamlit — caching is applied
at the call site in :mod:`filo_hft.gui.dashboard` so these helpers stay
test-friendly and reusable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from filo_hft.data.loaders import TELEMETRY_STREAMS, tail_telemetry

_DAY_RE = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class TelemetryBundle:
    """A snapshot of every telemetry stream for a single day.

    Streams that are missing or empty on disk are represented by an empty
    ``DataFrame`` (never ``None``) so panels can check ``df.empty`` and
    render a fallback without special-casing ``KeyError``.
    """

    telemetry_root: Path
    day: str
    frames: Dict[str, pd.DataFrame]

    def get(self, stream: str) -> pd.DataFrame:
        return self.frames.get(stream, pd.DataFrame())


def list_available_days(telemetry_root: Path) -> List[str]:
    """Return every ``YYYYMMDD`` subdirectory of ``telemetry_root``, newest first.

    Non-matching directories and regular files are ignored. Missing roots
    return an empty list so the caller can show a friendly message.
    """

    if not telemetry_root.exists() or not telemetry_root.is_dir():
        return []
    days: List[str] = []
    for child in telemetry_root.iterdir():
        if child.is_dir() and _DAY_RE.match(child.name):
            days.append(child.name)
    days.sort(reverse=True)
    return days


def latest_day(telemetry_root: Path) -> Optional[str]:
    """Return today's ``YYYYMMDD`` if present, else the most recent folder."""

    days = list_available_days(telemetry_root)
    if not days:
        return None
    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    if today in days:
        return today
    return days[0]


def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a UTC ``ts`` datetime column derived from ``ts_ms``.

    Rows without a numeric ``ts_ms`` are dropped since every panel keys off
    time. Returns a defensive copy so the caller can mutate freely.
    """

    if df.empty or "ts_ms" not in df.columns:
        return df
    out = df.copy()
    out["ts_ms"] = pd.to_numeric(out["ts_ms"], errors="coerce")
    out = out.dropna(subset=["ts_ms"])
    out["ts"] = pd.to_datetime(out["ts_ms"].astype("int64"), unit="ms", utc=True)
    return out


def load_bundle(
    telemetry_root: Path,
    day: str,
    tail_rows: int = 2000,
    streams: Optional[Iterable[str]] = None,
) -> TelemetryBundle:
    """Tail-read every requested telemetry stream for ``day``.

    :param telemetry_root: directory that contains ``YYYYMMDD`` subfolders.
    :param day: the target ``YYYYMMDD`` directory name.
    :param tail_rows: last N rows to read from each CSV; keeps the dashboard
        light when files grow to millions of rows.
    :param streams: optional subset of :data:`TELEMETRY_STREAMS`; defaults
        to all of them.
    """

    wanted = tuple(streams) if streams is not None else TELEMETRY_STREAMS
    frames: Dict[str, pd.DataFrame] = {}
    for stream in wanted:
        if stream not in TELEMETRY_STREAMS:
            frames[stream] = pd.DataFrame()
            continue
        try:
            raw = tail_telemetry(telemetry_root, day, stream, n=tail_rows)
        except Exception:  # noqa: BLE001  — dashboard must never crash on a bad row
            raw = pd.DataFrame()
        frames[stream] = _ensure_ts(raw)
    return TelemetryBundle(telemetry_root=telemetry_root, day=day, frames=frames)


def window_by_minutes(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Return only rows inside the last ``minutes`` of the frame's own clock.

    The "own clock" matters: during replay / backtest the wall clock is
    ahead of the telemetry by days, so anchoring on ``df['ts'].max()``
    keeps the window meaningful.
    """

    if df.empty or "ts" not in df.columns or minutes <= 0:
        return df
    end = df["ts"].max()
    start = end - pd.Timedelta(minutes=minutes)
    return df[df["ts"] >= start].reset_index(drop=True)


def active_keylevels(kl_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the keylevels event log down to currently-active zones.

    The stream is an append-only event log (``add``/``remove``/``merge``/
    ``invalidate``). For rendering we only want the last surviving state
    per ``zone_id``: keep it iff the latest event is an ``add`` or
    ``merge``.
    """

    if kl_df.empty or "zone_id" not in kl_df.columns:
        return kl_df
    ordered = kl_df.sort_values("ts_ms")
    latest = ordered.groupby("zone_id", as_index=False).tail(1)
    if "event" in latest.columns:
        latest = latest[latest["event"].isin(["add", "merge"])]
    return latest.reset_index(drop=True)


def open_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per ticket whose latest event is not ``close``.

    Columns returned: ``ticket``, ``direction``, ``entry_price``,
    ``hidden_sl``, ``hidden_tp``, ``unrealized_pips``, ``last_event``,
    ``ts``. Approximate — the EA emits no explicit "still open" flag, so
    we infer from the absence of a terminal ``close`` event.
    """

    if trades_df.empty or "ticket" not in trades_df.columns:
        return pd.DataFrame(
            columns=[
                "ticket",
                "direction",
                "entry_price",
                "hidden_sl",
                "hidden_tp",
                "unrealized_pips",
                "last_event",
                "ts",
            ]
        )
    ordered = trades_df.sort_values("ts_ms")
    last_per_ticket = ordered.groupby("ticket", as_index=False).tail(1)
    still_open = last_per_ticket[last_per_ticket.get("event") != "close"].copy()

    opens = ordered[ordered.get("event") == "open"].groupby("ticket", as_index=False).head(1)
    entry_by_ticket = dict(zip(opens["ticket"], opens.get("price", pd.Series(dtype=float))))
    still_open["entry_price"] = still_open["ticket"].map(entry_by_ticket)

    cols = {
        "ticket": "ticket",
        "direction": "direction",
        "entry_price": "entry_price",
        "hidden_sl": "hidden_sl",
        "hidden_tp": "hidden_tp",
        "unrealized_pips": "unrealized_pips",
        "event": "last_event",
        "ts": "ts",
    }
    present = {src: dst for src, dst in cols.items() if src in still_open.columns}
    return still_open[list(present.keys())].rename(columns=present).reset_index(drop=True)
