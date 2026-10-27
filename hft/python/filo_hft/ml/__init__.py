"""ML scoring layer for FiloHFT.

This package adds a second, probability-of-positive-outcome score on top of
the deterministic setup engine. It is advisory in the first implementation:
the deterministic engine still produces candidates and their intrinsic score,
and the ML layer is used as a ranker / soft filter.

Public surface:

- :func:`filo_hft.ml.features.build_dataset`   — ``(trades, candidates, cfg) -> (X, y, meta)``
- :func:`filo_hft.ml.features.load_from_telemetry`
- :func:`filo_hft.ml.features.load_from_backtest`
- :func:`filo_hft.ml.features.load_from_shadow`
- :data:`filo_hft.ml.features.FEATURE_COLUMNS`
- :func:`filo_hft.ml.train.train_baseline`
- :func:`filo_hft.ml.train.save_model`
- :class:`filo_hft.ml.train.TrainedModel`
- :class:`filo_hft.ml.infer.BaselinePredictor`
"""

from __future__ import annotations

from filo_hft.ml.features import (
    FEATURE_COLUMNS,
    FAMILIES,
    build_dataset,
    load_from_backtest,
    load_from_shadow,
    load_from_telemetry,
)
from filo_hft.ml.infer import BaselinePredictor
from filo_hft.ml.train import TrainedModel, save_model, train_baseline

__all__ = [
    "FEATURE_COLUMNS",
    "FAMILIES",
    "build_dataset",
    "load_from_telemetry",
    "load_from_backtest",
    "load_from_shadow",
    "TrainedModel",
    "train_baseline",
    "save_model",
    "BaselinePredictor",
]
