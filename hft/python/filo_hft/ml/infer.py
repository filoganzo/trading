"""Inference-time predictor for the baseline ML scoring layer.

The predictor is the public interface that the backtest and live layers use
to turn a feature vector into ``P(positive outcome)``. It is deliberately
small and dependency-light:

* Loads a :class:`filo_hft.ml.train.TrainedModel` once at construction time.
* Exposes a single :meth:`predict_proba` method that accepts *either* a
  single feature dict (the hot path for live scoring) *or* a
  :class:`pandas.DataFrame` of many rows (the batch path used by the
  backtest). The return type mirrors the input: a float for a dict, an
  :class:`numpy.ndarray` for a frame.
* Unknown / missing feature keys default to 0.0 — the features were chosen
  so zero is a neutral value (e.g. ``direction_long=0`` simply means
  "short", ``hour_sin=hour_cos=0`` means "unknown hour"). This is a
  deliberate graceful-degradation choice, not a silent failure: the EA is
  expected to fill every column, and the prediction will still be sensible
  if it doesn't.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd

from filo_hft.ml.features import FEATURE_COLUMNS
from filo_hft.ml.train import TrainedModel

FeatureInput = Union[dict, pd.DataFrame]


class BaselinePredictor:
    """Wraps a :class:`TrainedModel` for fast, vectorized inference."""

    def __init__(self, model_path: str | Path) -> None:
        obj = joblib.load(Path(model_path))
        if not isinstance(obj, TrainedModel):
            raise TypeError(
                f"expected TrainedModel at {model_path}, got {type(obj).__name__}"
            )
        self._tm: TrainedModel = obj
        self._model = obj.model
        # Freeze the column order at load time — the model was trained on
        # this exact layout and LightGBM will complain if we feed it
        # something else.
        self._columns: list[str] = list(obj.feature_columns)
        self._n_features: int = len(self._columns)

    # ------------------------------------------------------------------
    # Metadata passthroughs (handy for the EA sidecar bridge)

    @property
    def feature_columns(self) -> list[str]:
        return list(self._columns)

    @property
    def training_date(self) -> str:
        return self._tm.training_date

    @property
    def val_metrics(self) -> dict:
        return dict(self._tm.val_metrics)

    # ------------------------------------------------------------------
    # Prediction

    def predict_proba(self, features: FeatureInput) -> float | np.ndarray:
        """Return ``P(positive outcome)`` for one row or many.

        ``features`` may be:

        * a single ``dict`` — returns a ``float``,
        * a ``pandas.DataFrame`` — returns a 1-D ``numpy.ndarray`` of length
          ``len(features)``.
        """
        if isinstance(features, dict):
            row = self._row_from_dict(features)
            # Wrap in a single-row DataFrame so LightGBM sees matching
            # feature names and doesn't emit the sklearn "no feature names"
            # warning on the hot path.
            row_df = pd.DataFrame(row, columns=self._columns)
            prob = self._model.predict_proba(row_df)[0, 1]
            return float(prob)

        if isinstance(features, pd.DataFrame):
            frame = self._frame_from_df(features)
            probs = self._model.predict_proba(frame)[:, 1]
            return np.asarray(probs, dtype=np.float64)

        raise TypeError(
            f"features must be dict or DataFrame, got {type(features).__name__}"
        )

    # ------------------------------------------------------------------
    # internals

    def _row_from_dict(self, features: dict) -> np.ndarray:
        """Build a 1×N float32 array from a loose feature dict."""
        row = np.zeros((1, self._n_features), dtype=np.float32)
        for i, col in enumerate(self._columns):
            val = features.get(col, 0.0)
            try:
                row[0, i] = float(val)
            except (TypeError, ValueError):
                row[0, i] = 0.0
        return row

    def _frame_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex the frame to the trained column order, filling gaps."""
        missing = [c for c in self._columns if c not in df.columns]
        if missing:
            # Build the missing columns in one go to avoid pandas'
            # per-insert copy cost.
            filler = pd.DataFrame(
                0.0, index=df.index, columns=missing, dtype=np.float32
            )
            df = pd.concat([df, filler], axis=1)
        out = df[self._columns].astype(np.float32)
        # Fill NaNs to 0.0 (same neutral-default convention as the dict path).
        return out.fillna(0.0)


# ----------------------------------------------------------------------
# Small convenience constructor kept out of __init__ to stay importable
# without triggering a model load at package import time.


def load_predictor(model_path: str | Path) -> BaselinePredictor:
    """Factory: load and return a :class:`BaselinePredictor`."""
    return BaselinePredictor(model_path)


__all__ = ["BaselinePredictor", "load_predictor", "FEATURE_COLUMNS"]
