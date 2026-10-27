"""Trade log → metrics helpers.

Standalone functions so the sweep runner can call them without spinning
up the full engine. `compute_metrics` and `equity_curve` are the two
public entry points.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from filo_hft.config import FiloConfig

__all__ = ["compute_metrics", "equity_curve"]


# ---------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------


def equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    """Cumulative USD equity + drawdown percentage from a closed-trade log.

    Returns a DataFrame with columns `ts`, `equity_usd`, `drawdown_pct`.
    The starting equity is 0 USD (pure PnL-relative curve).
    """
    cols = ["ts", "equity_usd", "drawdown_pct"]
    if trades is None or len(trades) == 0 or "usd" not in trades.columns:
        return pd.DataFrame(columns=cols)

    df = trades.sort_values("ts_close").reset_index(drop=True).copy()
    df["equity_usd"] = df["usd"].cumsum()

    # drawdown relative to the running peak. When the peak is 0 or negative
    # we fall back to an absolute USD drawdown (no divide-by-zero).
    peaks = df["equity_usd"].cummax()
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = np.where(
            peaks > 0,
            (peaks - df["equity_usd"]) / peaks * 100.0,
            0.0,
        )
    df["drawdown_pct"] = dd_pct
    out = df[["ts_close", "equity_usd", "drawdown_pct"]].rename(columns={"ts_close": "ts"})
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------


def _sharpe(returns_usd: pd.Series) -> float:
    if len(returns_usd) < 2:
        return 0.0
    mean = float(returns_usd.mean())
    std = float(returns_usd.std(ddof=1))
    if std <= 1e-12 or math.isnan(std):
        return 0.0
    # Per-trade Sharpe scaled by sqrt(n). This is a bar-to-bar proxy; the
    # sweep runner annualizes separately if it needs to.
    return mean / std * math.sqrt(len(returns_usd))


def _max_drawdown_pct(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    dd = peak - equity
    # Use the peak as denominator; fall back to 1.0 when equity never
    # turns positive so max_dd_pct reads as absolute USD dollars in that
    # degenerate case.
    denom = peak.where(peak > 0, 1.0)
    pct = (dd / denom * 100.0).max()
    return float(pct) if not math.isnan(pct) else 0.0


def compute_metrics(trades: pd.DataFrame, cfg: FiloConfig) -> dict:
    """Produce a metrics dictionary from a closed-trade log.

    Always returns the full schema (with zeros for the empty case) so
    downstream code can rely on the keys being present.
    """
    empty_breakdown: dict[str, dict] = {}
    result: dict = {
        "total_pnl_usd": 0.0,
        "trade_count": 0,
        "win_rate": 0.0,
        "expectancy_pips": 0.0,
        "expectancy_usd": 0.0,
        "sharpe": 0.0,
        "max_dd_pct": 0.0,
        "gross_win_usd": 0.0,
        "gross_loss_usd": 0.0,
        "profit_factor": 0.0,
        "avg_win_pips": 0.0,
        "avg_loss_pips": 0.0,
        "per_family": empty_breakdown,
    }

    if trades is None or len(trades) == 0:
        return result

    df = trades.copy()
    df = df.sort_values("ts_close").reset_index(drop=True)
    wins = df[df["usd"] > 0]
    losses = df[df["usd"] < 0]
    n = len(df)

    total_pnl = float(df["usd"].sum())
    gross_win = float(wins["usd"].sum())
    gross_loss = float(losses["usd"].sum())
    win_rate = float(len(wins)) / n if n > 0 else 0.0
    expectancy_pips = float(df["pips"].mean()) if "pips" in df.columns else 0.0
    expectancy_usd = float(df["usd"].mean())
    sharpe = _sharpe(df["usd"])
    equity = df["usd"].cumsum()
    max_dd = _max_drawdown_pct(equity)
    profit_factor = (gross_win / abs(gross_loss)) if gross_loss < 0 else float("inf") if gross_win > 0 else 0.0

    result.update(
        {
            "total_pnl_usd": total_pnl,
            "trade_count": int(n),
            "win_rate": win_rate,
            "expectancy_pips": expectancy_pips,
            "expectancy_usd": expectancy_usd,
            "sharpe": sharpe,
            "max_dd_pct": max_dd,
            "gross_win_usd": gross_win,
            "gross_loss_usd": gross_loss,
            "profit_factor": profit_factor,
            "avg_win_pips": float(wins["pips"].mean()) if len(wins) > 0 else 0.0,
            "avg_loss_pips": float(losses["pips"].mean()) if len(losses) > 0 else 0.0,
        }
    )

    # Per-family breakdown
    if "family" in df.columns:
        per_family: dict[str, dict] = {}
        for fam, sub in df.groupby("family"):
            sub_wins = sub[sub["usd"] > 0]
            per_family[str(fam)] = {
                "trade_count": int(len(sub)),
                "total_pnl_usd": float(sub["usd"].sum()),
                "win_rate": float(len(sub_wins)) / len(sub) if len(sub) > 0 else 0.0,
                "expectancy_pips": float(sub["pips"].mean()) if "pips" in sub.columns else 0.0,
                "expectancy_usd": float(sub["usd"].mean()),
                "avg_trade_duration_s": float(
                    (sub["ts_close"] - sub["ts_open"]).dt.total_seconds().mean()
                )
                if "ts_open" in sub.columns
                else 0.0,
            }
        result["per_family"] = per_family

    return result
