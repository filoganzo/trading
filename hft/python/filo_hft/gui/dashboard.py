"""FiloHFT live observability dashboard (Streamlit).

Launch via :mod:`filo_hft.scripts.run_dashboard` (recommended) or directly
with::

    streamlit run python/filo_hft/gui/dashboard.py

Environment overrides (set by ``run_dashboard.py``):

* ``FILOHFT_TELEMETRY_ROOT`` — path to the ``telemetry`` directory.
* ``FILOHFT_CONFIG_PATH`` — optional ``config/default.ini`` for pip math.

Every panel reads real data from the CSVs that the EA writes. If a stream
is missing or empty the panel shows a clear "no data" placeholder instead
of a blank box.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from filo_hft.config import FiloConfig, load_config
from filo_hft.gui.telemetry_reader import (
    TelemetryBundle,
    active_keylevels,
    latest_day,
    list_available_days,
    load_bundle,
    open_trades,
    window_by_minutes,
)

# ---------------------------------------------------------------------------
# Constants

_TF_COLORS = {
    "D1": "#e53935",   # red
    "H4": "#fb8c00",   # orange
    "H1": "#fdd835",   # gold
    "M30": "#1e88e5",  # blue
    "M15": "#43a047",  # green
    "M5": "#9e9e9e",   # gray
}

_TREND_COLORS = {
    "up": "#2e7d32",
    "down": "#c62828",
    "flat": "#616161",
}


# ---------------------------------------------------------------------------
# Cached IO


@st.cache_data(ttl=2, show_spinner=False)
def _cached_bundle(
    telemetry_root_str: str,
    day: str,
    tail_rows: int,
) -> TelemetryBundle:
    """Tail-read every stream for one day. Cached so autorefresh is cheap."""

    return load_bundle(Path(telemetry_root_str), day, tail_rows=tail_rows)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_days(telemetry_root_str: str) -> list[str]:
    return list_available_days(Path(telemetry_root_str))


@st.cache_data(ttl=300, show_spinner=False)
def _cached_config(config_path_str: Optional[str]) -> Optional[FiloConfig]:
    if not config_path_str:
        return None
    try:
        return load_config(config_path_str)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Small utilities


def _fmt_float(v: object, digits: int = 2) -> str:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if pd.isna(f):
        return "—"
    return f"{f:.{digits}f}"


def _recent_halt(df: pd.DataFrame, within_seconds: int = 300) -> bool:
    """True if the frame has an ``action == 'halt'`` row inside the window."""

    if df.empty or "action" not in df.columns or "ts" not in df.columns:
        return False
    end = df["ts"].max()
    cutoff = end - pd.Timedelta(seconds=within_seconds)
    recent = df[df["ts"] >= cutoff]
    return bool((recent["action"] == "halt").any())


def _latest_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None
    return df.iloc[-1]


def _latency_p95(latency_df: pd.DataFrame) -> Optional[float]:
    """Most recent ``p95_ms`` for the ``tick_handler`` stage."""

    if latency_df.empty or "stage" not in latency_df.columns:
        return None
    tick = latency_df[latency_df["stage"] == "tick_handler"]
    if tick.empty:
        if "p95_ms" in latency_df.columns and not latency_df["p95_ms"].dropna().empty:
            return float(latency_df["p95_ms"].dropna().iloc[-1])
        return None
    if "p95_ms" not in tick.columns:
        return None
    series = tick["p95_ms"].dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


# ---------------------------------------------------------------------------
# Panels


def _render_header(bundle: TelemetryBundle) -> None:
    """Panel 1: live status strip with halt banners."""

    features = bundle.get("features")
    risk = bundle.get("risk")
    safety = bundle.get("safety")
    latency = bundle.get("latency")

    if _recent_halt(risk):
        st.error("RISK KILL-SWITCH ACTIVE — a recent risk.csv row has action=halt.", icon="🛑")
    if _recent_halt(safety):
        st.warning("SAFETY HALT ACTIVE — a recent safety.csv row has action=halt.", icon="⚠️")

    latest = _latest_row(features)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spread (pips)", _fmt_float(latest["spread_pips"], 2) if latest is not None else "—")
    c2.metric("Session", str(latest["session"]) if latest is not None and "session" in latest else "—")
    c3.metric(
        "Kill switch",
        "HALT" if _recent_halt(risk) else "ok",
        delta=None,
        delta_color="inverse",
    )
    c4.metric(
        "Safety",
        "HALT" if _recent_halt(safety) else "ok",
        delta=None,
        delta_color="inverse",
    )
    p95 = _latency_p95(latency)
    c5.metric("Tick p95 (ms)", _fmt_float(p95, 2) if p95 is not None else "—")


def _render_price_channel(bundle: TelemetryBundle, minutes: int) -> None:
    """Panels 2 + 3: mid / channel line chart with KL overlay and setup markers."""

    st.subheader("Price + channel + key levels")
    features = bundle.get("features")
    if features.empty or "mid" not in features.columns:
        st.info("No features.csv data for this day yet.")
        return
    windowed = window_by_minutes(features, minutes)
    if windowed.empty:
        st.info("No feature rows inside the selected window.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=windowed["ts"], y=windowed["mid"], mode="lines",
            name="mid", line=dict(color="#212121", width=1.5),
        )
    )
    for col, color, dash in (
        ("chan_mid", "#1976d2", "solid"),
        ("chan_upper", "#1976d2", "dot"),
        ("chan_lower", "#1976d2", "dot"),
    ):
        if col in windowed.columns:
            fig.add_trace(
                go.Scatter(
                    x=windowed["ts"], y=windowed[col], mode="lines",
                    name=col, line=dict(color=color, width=1, dash=dash),
                )
            )

    # KL horizontal lines (panel 3) — clamp to the visible price band.
    kl = active_keylevels(bundle.get("keylevels"))
    if not kl.empty and "center" in kl.columns:
        y_min = float(windowed["mid"].min())
        y_max = float(windowed["mid"].max())
        band = (y_max - y_min) * 0.25 if y_max > y_min else 1.0
        for _, zone in kl.iterrows():
            center = zone.get("center")
            if center is None or pd.isna(center):
                continue
            if not (y_min - band <= float(center) <= y_max + band):
                continue
            tf = str(zone.get("origin_tf", "")).upper()
            color = _TF_COLORS.get(tf, "#9e9e9e")
            weight = float(zone.get("weight", 0.3) or 0.3)
            width = max(1.0, min(6.0, 1.0 + weight * 4.0))
            fig.add_hline(
                y=float(center),
                line=dict(color=color, width=width, dash="dash"),
                opacity=0.55,
                annotation_text=f"{tf} w={weight:.2f}",
                annotation_position="right",
                annotation_font=dict(color=color, size=10),
            )

    # Accepted setup markers (panel 2 overlay).
    setups = bundle.get("setups")
    if not setups.empty and "decision" in setups.columns:
        accepted = setups[setups["decision"] == "accepted"]
        if "ts" in accepted.columns and not accepted.empty:
            t_min = windowed["ts"].min()
            t_max = windowed["ts"].max()
            accepted = accepted[(accepted["ts"] >= t_min) & (accepted["ts"] <= t_max)]
        if not accepted.empty:
            y_marker = float(windowed["mid"].iloc[-1])
            fig.add_trace(
                go.Scatter(
                    x=accepted["ts"],
                    y=[y_marker] * len(accepted),
                    mode="markers",
                    name="accepted setups",
                    marker=dict(symbol="triangle-up", size=11, color="#00c853"),
                    hovertext=[
                        f"{row.get('family', '?')} / {row.get('direction', '?')}"
                        for _, row in accepted.iterrows()
                    ],
                )
            )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.05),
        xaxis_title=None,
        yaxis_title="price",
    )
    st.plotly_chart(fig, use_container_width=True)

    if active_keylevels(bundle.get("keylevels")).empty:
        st.caption("keylevels.csv missing or empty — KL overlay skipped.")


def _render_burst_panel(bundle: TelemetryBundle, minutes: int) -> None:
    """Panel 4: velocity and burst counts."""

    st.subheader("Acceleration / bursts")
    features = bundle.get("features")
    if features.empty:
        st.info("No features.csv data.")
        return
    windowed = window_by_minutes(features, minutes)
    if windowed.empty:
        st.info("No rows in window.")
        return
    fig = go.Figure()
    if "velocity" in windowed.columns:
        fig.add_trace(
            go.Scatter(
                x=windowed["ts"], y=windowed["velocity"], mode="lines",
                name="velocity (pips/s)", line=dict(color="#1e88e5", width=1.5),
            )
        )
    if "burst_count_up" in windowed.columns:
        fig.add_trace(
            go.Bar(
                x=windowed["ts"], y=windowed["burst_count_up"],
                name="burst up", marker_color="#2e7d32", opacity=0.55,
            )
        )
    if "burst_count_dn" in windowed.columns:
        fig.add_trace(
            go.Bar(
                x=windowed["ts"], y=-windowed["burst_count_dn"],
                name="burst dn", marker_color="#c62828", opacity=0.55,
            )
        )
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.12),
        barmode="relative",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_trend_corr_tiles(bundle: TelemetryBundle) -> None:
    """Panel 5: trend state and correlation tiles."""

    features = bundle.get("features")
    latest = _latest_row(features)
    c1, c2, c3 = st.columns(3)
    if latest is None:
        for c, label in zip((c1, c2, c3), ("trend", "corr XAGUSD", "corr DXY")):
            c.metric(label, "—")
        return
    trend = str(latest.get("trend_state", "—"))
    color = _TREND_COLORS.get(trend, "#616161")
    c1.markdown(
        f"<div style='padding:12px;border-radius:8px;"
        f"background:{color};color:white;text-align:center;font-size:20px;'>"
        f"trend: <b>{trend}</b></div>",
        unsafe_allow_html=True,
    )
    for col, label, target in (
        ("corr_xagusd", "corr XAGUSD", c2),
        ("corr_dxy", "corr DXY", c3),
    ):
        try:
            v = float(latest.get(col))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            v = float("nan")
        if pd.isna(v):
            target.metric(label, "—")
        else:
            indicator = "↑" if v > 0.2 else ("↓" if v < -0.2 else "·")
            target.metric(label, f"{indicator} {v:+.2f}")


def _render_setup_queue(
    bundle: TelemetryBundle,
    families: list[str],
    decisions: list[str],
) -> None:
    """Panel 6: last-N setups table, decision-coloured, with block reasons."""

    st.subheader("Candidate setup queue (last 20)")
    setups = bundle.get("setups")
    if setups.empty:
        st.info("No setups.csv data.")
        return
    df = setups.copy()
    if families and "family" in df.columns:
        df = df[df["family"].isin(families)]
    if decisions and "decision" in df.columns:
        df = df[df["decision"].isin(decisions)]
    if df.empty:
        st.info("No setups match the current filters.")
        return
    df = df.sort_values("ts_ms").tail(20)
    cols = [
        c
        for c in (
            "ts",
            "family",
            "direction",
            "decision",
            "block_reason",
            "intrinsic_score",
            "ml_score",
            "kl_weight",
            "kl_distance_pips",
            "chan_pos_norm",
            "spread_pips",
            "velocity",
        )
        if c in df.columns
    ]
    view = df[cols].reset_index(drop=True)

    def _color_row(row: pd.Series) -> list[str]:
        decision = row.get("decision", "")
        if decision == "accepted":
            return ["background-color: #c8e6c9"] * len(row)
        if decision == "blocked":
            return ["background-color: #ffcdd2"] * len(row)
        return [""] * len(row)

    styled = view.style.apply(_color_row, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_live_trades(bundle: TelemetryBundle) -> None:
    """Panel 7: trades whose latest event isn't ``close``."""

    st.subheader("Live trades")
    trades = bundle.get("trades")
    if trades.empty:
        st.info("No trades.csv data.")
        return
    live = open_trades(trades)
    if live.empty:
        st.success("No open trades right now.")
        return
    # Cross-reference current mark from features.csv so unrealized shows something
    # even if the trades stream hasn't emitted a recent management event.
    features = bundle.get("features")
    mark: Optional[float] = None
    if not features.empty and "mid" in features.columns:
        try:
            mark = float(features["mid"].iloc[-1])
        except (TypeError, ValueError):
            mark = None
    if mark is not None:
        live = live.copy()
        live["current_mark"] = mark
    st.dataframe(live, use_container_width=True, hide_index=True)


def _render_execution_metrics(bundle: TelemetryBundle) -> None:
    """Panel 8: slippage + latency histograms with percentile readouts."""

    st.subheader("Execution metrics")
    fills = bundle.get("fills")
    latency = bundle.get("latency")

    col_slip, col_lat = st.columns(2)

    with col_slip:
        st.markdown("**Slippage (pips)**")
        if fills.empty or "slippage_pips" not in fills.columns:
            st.info("No fills.csv data.")
        else:
            slip = pd.to_numeric(fills["slippage_pips"], errors="coerce").dropna()
            if slip.empty:
                st.info("No slippage values.")
            else:
                fig = go.Figure(
                    data=[go.Histogram(x=slip, nbinsx=25, marker_color="#1e88e5")]
                )
                fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f"p50 {slip.quantile(0.50):+.2f}  ·  "
                    f"p95 {slip.quantile(0.95):+.2f}  ·  "
                    f"p99 {slip.quantile(0.99):+.2f}"
                )

    with col_lat:
        st.markdown("**Fill latency (ms)**")
        latencies_sources: list[pd.Series] = []
        if not fills.empty and "latency_ms" in fills.columns:
            latencies_sources.append(pd.to_numeric(fills["latency_ms"], errors="coerce"))
        if not latency.empty and "elapsed_ms" in latency.columns:
            lat_df = latency[latency.get("stage", "") == "order_send"]
            if not lat_df.empty:
                latencies_sources.append(pd.to_numeric(lat_df["elapsed_ms"], errors="coerce"))
        if not latencies_sources:
            st.info("No fill-latency data.")
        else:
            combined = pd.concat(latencies_sources).dropna()
            if combined.empty:
                st.info("No numeric latency values.")
            else:
                fig = go.Figure(
                    data=[go.Histogram(x=combined, nbinsx=25, marker_color="#6a1b9a")]
                )
                fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f"p50 {combined.quantile(0.50):.2f}  ·  "
                    f"p95 {combined.quantile(0.95):.2f}  ·  "
                    f"p99 {combined.quantile(0.99):.2f}"
                )


def _render_safety_risk(bundle: TelemetryBundle) -> None:
    """Panel 9: raw safety.csv and risk.csv tails side-by-side."""

    st.subheader("Safety + kill-switch log")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**risk.csv (last 20)**")
        risk = bundle.get("risk")
        if risk.empty:
            st.info("No risk events.")
        else:
            st.dataframe(
                risk.sort_values("ts_ms").tail(20).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
    with c2:
        st.markdown("**safety.csv (last 20)**")
        safety = bundle.get("safety")
        if safety.empty:
            st.info("No safety events.")
        else:
            st.dataframe(
                safety.sort_values("ts_ms").tail(20).reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )


def _render_session_badge(bundle: TelemetryBundle) -> None:
    """Panel 10: single-glance current session badge."""

    features = bundle.get("features")
    latest = _latest_row(features)
    session = "—"
    if latest is not None and "session" in latest:
        session = str(latest["session"])
    st.markdown(
        "<div style='padding:8px 14px;border-radius:6px;"
        "background:#263238;color:#fff;display:inline-block;"
        f"font-size:16px;'>session: <b>{session}</b></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main


def main() -> None:
    st.set_page_config(
        page_title="FiloHFT — Live Observability",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("FiloHFT — Live Observability")

    default_root = os.environ.get("FILOHFT_TELEMETRY_ROOT", "../telemetry")
    config_path = os.environ.get("FILOHFT_CONFIG_PATH")
    _ = _cached_config(config_path)  # loaded for side-effect / future use

    with st.sidebar:
        st.header("Source")
        telemetry_root = st.text_input("Telemetry root", value=default_root)
        root_path = Path(telemetry_root).expanduser()
        days = _cached_days(str(root_path))
        if not days:
            st.warning(f"No YYYYMMDD folders under {root_path}.")
            selected_day: Optional[str] = None
        else:
            default_day = latest_day(root_path) or days[0]
            selected_day = st.selectbox(
                "Day",
                options=days,
                index=days.index(default_day) if default_day in days else 0,
            )

        st.header("View")
        window_minutes = st.slider("Window (minutes)", 5, 240, 30, step=5)
        tail_rows = st.number_input(
            "Tail rows per stream", min_value=200, max_value=50_000,
            value=2000, step=200,
        )

        st.header("Refresh")
        auto_refresh = st.checkbox("Auto-refresh every 2s", value=False)
        if st.button("Refresh now"):
            st.cache_data.clear()
            _rerun()

        st.header("Setup filters")
        families_all = [
            "breakout", "retest", "rejection", "fakeout",
            "failed_auction", "engulfing_accel", "mean_reversion",
        ]
        sel_families = st.multiselect("Family", families_all, default=families_all)
        sel_decisions = st.multiselect(
            "Decision", ["accepted", "blocked"], default=["accepted", "blocked"],
        )

    if selected_day is None:
        st.stop()

    bundle = _cached_bundle(str(root_path), selected_day, int(tail_rows))

    _render_header(bundle)
    st.divider()
    _render_session_badge(bundle)
    _render_trend_corr_tiles(bundle)
    st.divider()
    _render_price_channel(bundle, window_minutes)
    _render_burst_panel(bundle, window_minutes)
    st.divider()
    _render_setup_queue(bundle, sel_families, sel_decisions)
    _render_live_trades(bundle)
    st.divider()
    _render_execution_metrics(bundle)
    _render_safety_risk(bundle)

    if auto_refresh:
        _autorefresh_or_rerun(interval_ms=2000)


def _rerun() -> None:
    """Call whichever rerun API is available on this Streamlit version."""

    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if fn is not None:
        fn()


def _autorefresh_or_rerun(interval_ms: int) -> None:
    """Prefer ``streamlit_autorefresh`` if installed, else soft fall-back.

    We deliberately do not add ``streamlit_autorefresh`` as a hard
    dependency; the soft fall-back lets the user click "Refresh now" or
    toggle the checkbox off to regain control.
    """

    try:
        from streamlit_autorefresh import st_autorefresh  # type: ignore

        st_autorefresh(interval=interval_ms, key="filo_autorefresh")
        return
    except ImportError:
        pass
    st.caption(
        "streamlit_autorefresh not installed — use the 'Refresh now' "
        "button for a manual pull."
    )


if __name__ == "__main__":
    main()
