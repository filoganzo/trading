"""Dump authoritative MT5 deal history to parquet.

Pulls directly from the running MT5 terminal via the MetaTrader5 package.
Every deal carries broker-reported profit, commission, swap, fee — summing
these matches exactly what the terminal's Account History tab shows.

This is the single source of truth for "how did the day go" questions.
Family/session attribution is done by joining position_id back to the
engine's telemetry streams (tm_lab.csv parent_ticket, intents fallback).

Usage:
    python mt5_history_dump.py 20260415
    python mt5_history_dump.py 20260413 20260415     # range
    python mt5_history_dump.py                        # today UTC

Outputs:
    path/to/output/filohft_backtest_20260415/mt5_deals_<day>.parquet
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import MetaTrader5 as mt5  # type: ignore

OUT_DIR = Path("path/to/output/filohft_backtest_20260415")


def _parse_args(argv: list[str]) -> tuple[datetime, datetime, str]:
    if len(argv) >= 3:
        day_from = argv[1]
        day_to = argv[2]
    elif len(argv) == 2:
        day_from = argv[1]
        day_to = argv[1]
    else:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        day_from = today
        day_to = today
    start = datetime.strptime(day_from, "%Y%m%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(day_to, "%Y%m%d").replace(tzinfo=timezone.utc) + timedelta(days=1)
    label = day_from if day_from == day_to else f"{day_from}_{day_to}"
    return start, end, label


def fetch_deals(start: datetime, end: datetime) -> pd.DataFrame:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"account_info None: {mt5.last_error()}")
        print(
            f"account: login={info.login}  server={info.server}  "
            f"currency={info.currency}  balance={info.balance:,.2f}  "
            f"equity={info.equity:,.2f}  mode={info.margin_mode}  lev={info.leverage}"
        )
        deals = mt5.history_deals_get(start, end)
        if deals is None:
            raise RuntimeError(f"history_deals_get: {mt5.last_error()}")
        df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys()) if deals else pd.DataFrame()
    finally:
        mt5.shutdown()

    if df.empty:
        return df

    # readable types
    df["ts"] = pd.to_datetime(df["time_msc"], unit="ms", utc=True)
    type_map = {
        mt5.DEAL_TYPE_BUY: "buy",
        mt5.DEAL_TYPE_SELL: "sell",
        mt5.DEAL_TYPE_BALANCE: "balance",
        mt5.DEAL_TYPE_CREDIT: "credit",
    }
    entry_map = {
        mt5.DEAL_ENTRY_IN: "in",
        mt5.DEAL_ENTRY_OUT: "out",
        mt5.DEAL_ENTRY_INOUT: "inout",
        mt5.DEAL_ENTRY_OUT_BY: "out_by",
    }
    df["type_name"] = df["type"].map(type_map).fillna(df["type"].astype(str))
    df["entry_name"] = df["entry"].map(entry_map).fillna(df["entry"].astype(str))
    df["total"] = df["profit"] + df["commission"] + df["swap"] + df["fee"]
    return df


def summarize(df: pd.DataFrame) -> str:
    if df.empty:
        return "no deals"
    syms = df[df["type_name"].isin(["buy", "sell"])]
    total = df["total"].sum()
    profit = df["profit"].sum()
    comm = df["commission"].sum()
    swap = df["swap"].sum()
    fee = df["fee"].sum()

    lines = []
    lines.append(f"deals total rows  : {len(df)}")
    lines.append(f"  trading deals   : {len(syms)}")
    lines.append(f"  balance/credit  : {len(df) - len(syms)}")
    lines.append(f"volume lots       : {syms['volume'].sum():.2f}")
    lines.append("")
    lines.append(f"gross profit      : {profit:+,.2f}")
    lines.append(f"commission        : {comm:+,.2f}")
    lines.append(f"swap              : {swap:+,.2f}")
    lines.append(f"fee               : {fee:+,.2f}")
    lines.append(f"TOTAL (terminal)  : {total:+,.2f}")
    lines.append("")

    # Per symbol
    if "symbol" in df.columns:
        by_sym = (
            df.groupby("symbol")
            .agg(
                deals=("ticket", "size"),
                vol=("volume", "sum"),
                profit=("profit", "sum"),
                comm=("commission", "sum"),
                swap=("swap", "sum"),
                total=("total", "sum"),
            )
            .round(2)
            .sort_values("total")
        )
        lines.append("per symbol:")
        lines.append(by_sym.to_string())
        lines.append("")

    # IN vs OUT deals on trading rows
    if not syms.empty:
        by_entry = (
            syms.groupby("entry_name")
            .agg(
                deals=("ticket", "size"),
                vol=("volume", "sum"),
                profit=("profit", "sum"),
                comm=("commission", "sum"),
                swap=("swap", "sum"),
                total=("total", "sum"),
            )
            .round(2)
        )
        lines.append("per deal entry (in=open, out=close):")
        lines.append(by_entry.to_string())
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    start, end, label = _parse_args(sys.argv)
    print(f"window: {start.isoformat()} -> {end.isoformat()} (label={label})")
    df = fetch_deals(start, end)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"mt5_deals_{label}.parquet"
    if not df.empty:
        df.to_parquet(out, index=False)
        print(f"wrote {out}  ({len(df)} rows)")
    print()
    print(summarize(df))


if __name__ == "__main__":
    main()
