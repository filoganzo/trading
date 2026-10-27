"""Multi-terminal market data with explicit clock alignment.

The MT5 python API holds ONE connection at a time, so we connect, pull, and
shut down for each terminal in turn. At a 1-minute decision cadence that is
comfortably fast.

Every bar timestamp is converted to true UTC before anything is compared or
merged. Broker server clocks differ (OANDA measured 1h behind IC) and both
shift with DST, so the offset is measured live from the current tick rather
than hard-coded.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from config import (OANDA_EXE, IC_EXE, VANTAGE_EXE,
                    OANDA_GOLD, OANDA_SILVER, IC_DXY, VANTAGE_GOLD)


class FeedError(RuntimeError):
    pass


@dataclass
class Quote:
    bid: float
    ask: float
    spread_points: float
    server_utc_offset_h: float


def _connect(path: str, attempts: int = 3):
    for _ in range(attempts):
        if mt5.initialize(path=path):
            return True
        time.sleep(1.0)
    return False


def _server_utc_offset_hours(symbol: str | None = None) -> float:
    """Broker server time minus true UTC, in hours, measured from a live tick.

    MT5 reports tick.time as the broker's wall clock stamped as if UTC, so the
    difference against real UTC is the server offset. Rounded to the nearest
    half hour because all broker offsets are whole or half hours.

    The tick must come from a symbol that is actually streaming: an arbitrary
    symbols_get()[0] may be dormant and return a tick that is days stale, which
    yields a nonsense offset. We therefore require the caller's symbol and
    sanity-bound the result to +/-14h, returning 0.0 if it cannot be trusted.
    """
    if symbol is None:
        return 0.0
    t = mt5.symbol_info_tick(symbol)
    if t is None or t.time == 0:
        return 0.0
    delta = t.time - time.time()
    if abs(delta) > 14 * 3600:          # stale tick, not a real offset
        return 0.0
    return round(delta / 1800.0) * 0.5


def _pull(path: str, symbol: str, timeframe, count: int):
    """Return (DataFrame with UTC 'time', offset_hours). Raises on failure."""
    if not _connect(path):
        raise FeedError(f"cannot initialise terminal: {path} :: {mt5.last_error()}")
    try:
        if not mt5.symbol_select(symbol, True):
            raise FeedError(f"symbol not available: {symbol} on {path}")
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            raise FeedError(f"no rates for {symbol}: {mt5.last_error()}")
        off = _server_utc_offset_hours(symbol)
        df = pd.DataFrame(rates)
        df["time"] = df["time"].astype("int64") - int(off * 3600)   # -> true UTC
        return df, off
    finally:
        mt5.shutdown()


def get_bars(count: int = 1500, timeframe=mt5.TIMEFRAME_M1):
    """Pull gold+silver from OANDA and DXY from IC, all on a true-UTC axis."""
    gold, off_o = _pull(OANDA_EXE, OANDA_GOLD, timeframe, count)
    silver, _ = _pull(OANDA_EXE, OANDA_SILVER, timeframe, count)
    dxy, off_i = _pull(IC_EXE, IC_DXY, timeframe, count)
    return {
        "gold": gold, "silver": silver, "dxy": dxy,
        "offsets": {"oanda_h": off_o, "ic_h": off_i},
    }


def get_htf(count: int = 400):
    """H4 and D1 gold from OANDA, for the level engine."""
    h4, _ = _pull(OANDA_EXE, OANDA_GOLD, mt5.TIMEFRAME_H4, count)
    d1, _ = _pull(OANDA_EXE, OANDA_GOLD, mt5.TIMEFRAME_D1, count)
    return h4, d1


def vantage_quote() -> Quote:
    """Live executable quote from the Vantage terminal."""
    if not _connect(VANTAGE_EXE):
        raise FeedError("cannot initialise Vantage terminal")
    try:
        if not mt5.symbol_select(VANTAGE_GOLD, True):
            raise FeedError(f"{VANTAGE_GOLD} not available on Vantage")
        t = mt5.symbol_info_tick(VANTAGE_GOLD)
        si = mt5.symbol_info(VANTAGE_GOLD)
        if t is None or si is None:
            raise FeedError("no Vantage tick")
        return Quote(bid=t.bid, ask=t.ask,
                     spread_points=(t.ask - t.bid) / si.point,
                     server_utc_offset_h=_server_utc_offset_hours(VANTAGE_GOLD))
    finally:
        mt5.shutdown()


def vantage_account():
    """Account + open-position snapshot, used by the panel's status line."""
    if not _connect(VANTAGE_EXE):
        raise FeedError("cannot initialise Vantage terminal")
    try:
        ai = mt5.account_info()
        ti = mt5.terminal_info()
        pos = mt5.positions_get(symbol=VANTAGE_GOLD) or []
        return {
            "login": None if ai is None else ai.login,
            "server": None if ai is None else ai.server,
            "balance": None if ai is None else ai.balance,
            "equity": None if ai is None else ai.equity,
            "currency": None if ai is None else ai.currency,
            "trade_allowed": None if ti is None else ti.trade_allowed,
            "positions": [
                {"ticket": p.ticket, "type": "BUY" if p.type == 0 else "SELL",
                 "volume": p.volume, "price": p.price_open,
                 "sl": p.sl, "tp": p.tp, "profit": p.profit, "magic": p.magic}
                for p in pos
            ],
        }
    finally:
        mt5.shutdown()


def align(frames: dict) -> pd.DataFrame:
    """Inner-join the three series on the shared UTC minute.

    An inner join is deliberate: if one broker is lagging or a symbol has a gap
    we would rather compute on fewer bars than silently pair mismatched minutes.
    """
    g = frames["gold"][["time", "open", "high", "low", "close", "spread"]]
    s = frames["silver"][["time", "open", "close"]].rename(
        columns={"open": "s_open", "close": "s_close"})
    d = frames["dxy"][["time", "open", "close"]].rename(
        columns={"open": "d_open", "close": "d_close"})
    out = g.merge(s, on="time", how="inner").merge(d, on="time", how="inner")
    return out.sort_values("time").reset_index(drop=True)
