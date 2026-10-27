"""Offline backtest / replay engine for FiloHFT.

Mirrors the MQL5 hot-path (KeyLevels / Features / Setups / TradeManagement)
closely enough for parameter sweeps and ML label generation. Bar-level (M1)
replay — NOT tick-accurate.
"""

from filo_hft.backtest.engine import BacktestResult, run_backtest
from filo_hft.backtest.metrics import compute_metrics, equity_curve
from filo_hft.backtest.replay import load_replay_bars, run_from_disk

__all__ = [
    "BacktestResult",
    "run_backtest",
    "compute_metrics",
    "equity_curve",
    "load_replay_bars",
    "run_from_disk",
]
