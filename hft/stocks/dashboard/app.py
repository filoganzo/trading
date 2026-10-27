"""
LRC Signal Dashboard — run with:
    streamlit run stocks/dashboard/app.py
"""
import sys
sys.path.insert(0, 'path/to/repo/hft')

import streamlit as st
import pandas as pd
from datetime import datetime

from stocks.dashboard.signals import scan

st.set_page_config(
    page_title="LRC Signals",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    account   = st.number_input("Account size ($)", value=100_000, step=5_000, min_value=1_000)
    risk_pct  = st.slider("Risk per trade (%)", min_value=0.25, max_value=3.0, value=1.0, step=0.25) / 100
    st.divider()
    st.caption("**Signal parameters** (from backtest sweep)")
    lrc_length = st.selectbox("LRC length", [80, 100, 120], index=2)
    sigma_mult = st.selectbox("Sigma threshold", [1.0, 1.3, 1.5, 2.0], index=0)
    min_score  = st.slider("Min combined score", 20, 80, 40, step=5)
    touch_pct  = st.selectbox("Max level distance (%)", [0.75, 1.5], index=1)
    max_age    = st.slider("Max signal age (bars)", 1, 7, 3)
    st.divider()
    refresh = st.button("🔄 Refresh signals", use_container_width=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 LRC Mean-Reversion Signal Dashboard")
st.caption(
    f"Strategy: **lev+0.3atr | rr5 | none**  ·  "
    f"Universe: US liquid (NYSE + NAS, top 25% by dollar volume)  ·  "
    f"Account: **${account:,.0f}**  ·  Risk: **{risk_pct*100:.2f}% / trade**  ·  "
    f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
st.divider()

# ── Scan ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _scan(account, risk_pct, lrc_length, sigma_mult, min_score, touch_pct, max_age):
    return scan(account, risk_pct, lrc_length, sigma_mult, min_score, touch_pct, max_age)

if refresh:
    st.cache_data.clear()

with st.spinner("Scanning US liquid universe..."):
    signals = _scan(account, risk_pct, lrc_length, sigma_mult, min_score, touch_pct, max_age)

# ── Summary metrics ───────────────────────────────────────────────────────────
n_signals = len(signals)
n_long    = int((signals["direction"] == "LONG").sum())  if n_signals else 0
n_short   = int((signals["direction"] == "SHORT").sum()) if n_signals else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Signals today", n_signals)
col2.metric("Long setups",   n_long,   delta="▲" if n_long  else None)
col3.metric("Short setups",  n_short,  delta="▼" if n_short else None)
total_risk = n_signals * account * risk_pct
col4.metric("Total risk if all taken", f"${total_risk:,.0f}", delta=f"{total_risk/account*100:.1f}% of acct")

# ── No signals ────────────────────────────────────────────────────────────────
if signals.empty:
    st.info("✅ No fresh signals right now. Markets may be closed or no setups meet the criteria.")
    st.stop()

# ── Signals table ─────────────────────────────────────────────────────────────
st.subheader(f"⚡ {n_signals} Setup{'s' if n_signals != 1 else ''} Detected")

DIR_ICON   = {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT"}
TIER_ICON  = {"low_vol": "🔵 Low", "mid_vol": "🟡 Mid", "high_vol": "🔴 High", "unknown": "⚪ ?"}

display = signals.copy()
display["direction"]  = display["direction"].map(DIR_ICON)
display["vol_tier"]   = display["vol_tier"].map(TIER_ICON)
display["entry_est"]  = display["entry_est"].map(lambda x: f"~${x:,.2f}")
display["sl"]         = display["sl"].map(lambda x: f"${x:,.2f}")
display["tp"]         = display["tp"].map(lambda x: f"${x:,.2f}")
display["risk_usd"]   = display["risk_usd"].map(lambda x: f"${x:,.0f}")
display["notional"]   = display["notional"].map(lambda x: f"${x:,.0f}")
display["combined_score"] = display["combined_score"].map(lambda x: f"{x:.0f}/100")
display["dev_sigma"]  = display["dev_sigma"].map(lambda x: f"{x:+.2f}σ")

table_cols = [
    "symbol", "exchange", "sector", "vol_tier", "direction",
    "entry_est", "sl", "tp", "shares", "risk_usd", "notional",
    "combined_score", "dev_sigma", "signal_age_bars", "data_date",
]
col_labels = {
    "symbol": "Symbol", "exchange": "Exch", "sector": "Sector",
    "vol_tier": "Vol Tier", "direction": "Direction",
    "entry_est": "Entry (est.)", "sl": "Stop Loss", "tp": "Take Profit",
    "shares": "Shares", "risk_usd": "Risk $", "notional": "Notional",
    "combined_score": "Score", "dev_sigma": "Deviation",
    "signal_age_bars": "Age (bars)", "data_date": "Data Date",
}

st.dataframe(
    display[table_cols].rename(columns=col_labels),
    use_container_width=True,
    height=min(400, 50 + 36 * n_signals),
    hide_index=True,
)

# ── Detail cards ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("📋 Trade Sheets")

for _, row in signals.iterrows():
    dir_color = "🟢" if row["direction"] == "LONG" else "🔴"
    sl_dist_pct  = abs(row["entry_est"] - row["sl"]) / row["entry_est"] * 100
    tp_dist_pct  = abs(row["entry_est"] - row["tp"]) / row["entry_est"] * 100

    with st.expander(
        f"{dir_color} **{row['symbol']}** — {row['direction']}  |  "
        f"Entry ~${row['entry_est']:,.2f}  |  SL ${row['sl']:,.2f}  |  TP ${row['tp']:,.2f}  |  "
        f"Score {row['combined_score']:.0f}/100"
    ):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry (est.)",  f"~${row['entry_est']:,.2f}", help="Based on last close. Actual fill = next market open.")
        c1.metric("Direction",     row["direction"])
        c2.metric("Stop Loss",     f"${row['sl']:,.2f}", delta=f"-{sl_dist_pct:.2f}% from entry", delta_color="inverse")
        c2.metric("Take Profit",   f"${row['tp']:,.2f}", delta=f"+{tp_dist_pct:.2f}% from entry")
        c3.metric("Shares",        f"{row['shares']:,}")
        c3.metric("Notional",      f"${row['notional']:,.0f}")
        c4.metric("Risk $",        f"${row['risk_usd']:,.0f}")
        c4.metric("Reward $",      f"${row['risk_usd'] * row['rr']:,.0f}")

        st.caption(
            f"**Sector:** {row['sector']}  ·  **Exchange:** {row['exchange']}  ·  "
            f"**Vol tier:** {row['vol_tier']}  ·  "
            f"**LRC deviation:** {row['dev_sigma']:+.2f}σ  ·  "
            f"**LRC mid:** ${row['lrc_mid']:,.2f}  ·  "
            f"**ATR:** ${row['atr']:.3f}  ·  "
            f"**Signal age:** {row['signal_age_bars']} bar(s)  ·  "
            f"**Data as of:** {row['data_date']}"
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.warning(
    "⚠️ **Entry price is estimated from last close — actual fill is at next market open.** "
    "Place a market order at open or a limit order near the estimated entry. "
    "Set SL and TP immediately after entry. Never risk more than specified per trade."
)
st.caption(
    "Strategy: LRC mean-reversion · lev+0.3atr|rr5|none · "
    "Backtest summary: see local results"
)
