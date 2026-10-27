"""Parameter sweep runner and segmentation analytics for FiloHFT.

This package provides offline research tooling that sits on top of the
backtest engine (filo_hft.backtest.engine). Use `runner.run_sweep` to
evaluate the Cartesian product of parameter overrides in parallel, and
use `segmentation` to slice resulting trade logs by session, hour,
day-of-week, setup family or key-level timeframe.
"""

from filo_hft.sweeps.runner import (
    ParamGrid,
    run_sweep,
    run_single_param_sweep,
    run_cross_sweep,
)
from filo_hft.sweeps.segmentation import (
    segment_trades,
    session_of,
    session_by_hour_heatmap,
    KL_WEIGHT_BUCKETS,
)

__all__ = [
    "ParamGrid",
    "run_sweep",
    "run_single_param_sweep",
    "run_cross_sweep",
    "segment_trades",
    "session_of",
    "session_by_hour_heatmap",
    "KL_WEIGHT_BUCKETS",
]
