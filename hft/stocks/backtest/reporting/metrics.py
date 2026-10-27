"""Compute per-variant, per-segment performance metrics."""
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _sharpe(pnl_r_series: pd.Series, periods_per_year: float = 252) -> float:
    if len(pnl_r_series) < 5:
        return float("nan")
    mu  = pnl_r_series.mean()
    std = pnl_r_series.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(mu / std * np.sqrt(periods_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    roll_max = equity.cummax()
    dd = (equity - roll_max) / (roll_max.abs() + 1e-9)
    return float(dd.min())


def _profit_factor(pnl_r: pd.Series) -> float:
    wins  = pnl_r[pnl_r > 0].sum()
    losses = pnl_r[pnl_r < 0].abs().sum()
    return float(wins / losses) if losses > 0 else float("nan")


def compute_trade_metrics(
    trades: pd.DataFrame,
    segment_type: str,
    segment_value: str,
    variant_id: int,
) -> Dict:
    if trades.empty or len(trades) < 3:
        return _empty(variant_id, segment_type, segment_value)

    pnl_r = trades["pnl_r"]
    wins  = pnl_r[pnl_r > 0]
    bars  = (pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])).dt.days

    return {
        "variant_id":     variant_id,
        "segment_type":   segment_type,
        "segment_value":  segment_value,
        "total_trades":   len(trades),
        "win_rate":       round(float(len(wins) / len(trades)), 4),
        "avg_R":          round(float(pnl_r.mean()), 4),
        "median_R":       round(float(pnl_r.median()), 4),
        "profit_factor":  round(_profit_factor(pnl_r), 3),
        "sharpe_R":       round(_sharpe(pnl_r), 3),
        "avg_bars_held":  round(float(bars.mean()), 1),
        "max_win_R":      round(float(pnl_r.max()), 3),
        "max_loss_R":     round(float(pnl_r.min()), 3),
        "total_pnl_R":    round(float(pnl_r.sum()), 3),
        "total_pnl_usd":    round(float(trades["pnl_usd"].sum()), 2),
    }


def compute_portfolio_metrics(
    equity: pd.Series,
    variant_id: int,
    max_concurrent: int,
) -> Dict:
    if equity.empty or len(equity) < 10:
        return {}

    # equity is cumulative P&L starting from 0; shift by initial capital for % calcs
    initial_capital = NOTIONAL * max_concurrent
    ev = equity + initial_capital

    daily_ret = ev.pct_change().fillna(0)
    years = (equity.index[-1] - equity.index[0]).days / 365.25 if len(equity) > 1 else 1.0
    final_pnl = float(equity.iloc[-1])

    cagr = ((initial_capital + final_pnl) / initial_capital) ** (1 / years) - 1 \
           if years > 0 and initial_capital > 0 else float("nan")

    roll_max = ev.cummax()
    max_dd   = float(((ev - roll_max) / roll_max).min()) * 100  # %
    calmar   = (cagr * 100) / abs(max_dd) if max_dd < 0 else float("nan")

    return {
        "variant_id":      variant_id,
        "max_concurrent":  max_concurrent,
        "final_pnl_usd":   round(final_pnl, 2),
        "cagr_%":          round(cagr * 100, 2),
        "sharpe":          round(_sharpe(daily_ret), 3),
        "max_drawdown_%":  round(max_dd, 2),
        "calmar":          round(calmar, 3),
        "segment_type":    "portfolio",
        "segment_value":   f"mc{max_concurrent}",
    }


def _empty(vid, st, sv) -> Dict:
    return {
        "variant_id": vid, "segment_type": st, "segment_value": sv,
        "total_trades": 0, "win_rate": float("nan"), "avg_R": float("nan"),
        "median_R": float("nan"), "profit_factor": float("nan"),
        "sharpe_R": float("nan"), "avg_bars_held": float("nan"),
        "max_win_R": float("nan"), "max_loss_R": float("nan"),
        "total_pnl_R": float("nan"), "total_pnl_usd": float("nan"),
    }


NOTIONAL = 1_000.0   # local copy for portfolio metrics
