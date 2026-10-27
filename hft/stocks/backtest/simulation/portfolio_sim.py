"""
Portfolio simulation: one pass per (variant, max_concurrent).
Receives a pre-built CandidateFrame and the all_bars dict.
Returns (trades_df, equity_curve_series).
"""
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from stocks.backtest.config import NOTIONAL
from stocks.backtest.simulation.trade_manager import (
    PositionState, check_and_update, compute_sl, compute_tp, is_fully_closed,
)


def _mark_to_market(pos: PositionState, close: float) -> float:
    """Unrealised P&L in dollars for open position."""
    raw   = (close - pos.entry_price) * (-pos.direction)
    risk  = abs(pos.entry_price - pos.sl)
    if risk == 0 or pos.entry_price == 0:
        return 0.0
    pnl_r = raw / risk
    return (pos.notional * (1 - pos.partial_frac)) * (pnl_r * risk / pos.entry_price)


def run_portfolio_sim(
    candidates_df: pd.DataFrame,
    all_bars: Dict[str, pd.DataFrame],
    variant: Dict[str, Any],
    max_concurrent: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    candidates_df columns:
      date, symbol, direction, entry_price, sl_raw (key level), atr_at_entry,
      lrc_mid, combined_score, exchange, sector, variant_id(=0, overridden here)

    all_bars: symbol → OHLCV+ATR DataFrame indexed by datetime

    Returns:
      trades_df   — one row per closed partial/full trade
      equity_curve — pd.Series indexed by date, values = cumulative P&L
    """
    vid = variant["variant_id"]

    # Index candidates by date for O(1) lookup
    cands_by_date: Dict = {}
    for row in candidates_df.itertuples(index=False):
        d = row.date
        if d not in cands_by_date:
            cands_by_date[d] = []
        cands_by_date[d].append(row)

    # Build sorted date range from all bars
    all_dates = sorted({
        d for df in all_bars.values() for d in df.index
    })

    open_positions: List[PositionState] = []
    closed_records: List[Dict]          = []
    equity_curve: Dict                  = {}
    cumulative_pnl = 0.0

    for date in all_dates:
        # 1. Update open positions (exit check)
        still_open = []
        for pos in open_positions:
            sym_bars = all_bars.get(pos.symbol)
            if sym_bars is None or date not in sym_bars.index:
                still_open.append(pos)
                continue
            bar = sym_bars.loc[date]
            exits = check_and_update(
                pos, float(bar["High"]), float(bar["Low"]),
                float(bar["Close"]), date, variant,
            )
            for e in exits:
                cumulative_pnl += e["pnl_usd"]
                closed_records.append(e)
            if not is_fully_closed(pos):
                still_open.append(pos)

        open_positions = still_open

        # 2. Open new positions from today's candidates
        slots = max_concurrent - len(open_positions)
        if slots > 0 and date in cands_by_date:
            day_cands = sorted(cands_by_date[date],
                               key=lambda c: c.combined_score, reverse=True)[:slots]
            total_score = sum(c.combined_score for c in day_cands) or 1.0

            for cand in day_cands:
                # entry_price is next bar's open — already set in CandidateFrame
                ep  = cand.entry_price
                atr = cand.atr_at_entry
                if ep <= 0 or np.isnan(ep) or atr <= 0 or np.isnan(atr):
                    continue

                sl  = compute_sl(ep, cand.direction, cand.sl_raw, atr, variant)
                tp_main, tp1, tp2 = compute_tp(ep, sl, cand.lrc_mid, cand.direction, variant)

                # Check TP is on the correct side of entry
                if tp1 is None:
                    ok = ((cand.direction == -1 and tp_main > ep) or
                          (cand.direction == +1 and tp_main < ep))
                    if not ok:
                        continue
                else:
                    ok1 = ((cand.direction == -1 and tp1 > ep) or
                           (cand.direction == +1 and tp1 < ep))
                    if not ok1:
                        continue

                alloc = NOTIONAL * (cand.combined_score / total_score)

                pos = PositionState(
                    symbol       = cand.symbol,
                    variant_id   = vid,
                    direction    = cand.direction,
                    entry_date   = date,
                    entry_price  = ep,
                    sl           = sl,
                    tp_main      = tp_main if tp1 is None else float("nan"),
                    tp1          = tp1,
                    tp2          = tp2,
                    tp1_frac     = variant.get("tp1_frac", 0.5),
                    tp2_frac     = variant.get("tp2_frac", 0.5),
                    notional     = alloc,
                    atr          = atr,
                )
                open_positions.append(pos)

        # 3. Mark to market
        mtm = sum(
            _mark_to_market(p, float(all_bars[p.symbol].loc[date]["Close"]))
            for p in open_positions
            if p.symbol in all_bars and date in all_bars[p.symbol].index
        )
        equity_curve[date] = cumulative_pnl + mtm

    trades_df   = pd.DataFrame(closed_records) if closed_records else pd.DataFrame()
    equity_s    = pd.Series(equity_curve, name="equity")
    return trades_df, equity_s
