"""Training driver for the baseline ML scoring layer.

The baseline is a LightGBM binary classifier that predicts
``P(realized_usd > 0)`` from the setup-time feature vector built in
:mod:`filo_hft.ml.features`. The model's output is a second score that the
execution / backtest layer uses as a ranker or soft filter on top of the
deterministic ``intrinsic_score``.

Key design decisions (deliberate, documented here so the next engineer
doesn't wonder):

* **Time-ordered 80/20 split**, *not* random. Financial data has regime
  structure; a random split leaks future-into-past and produces dishonest
  validation metrics. The split is done on the meta frame's ``ts_open``.
* **Early stopping on the validation fold** with
  ``early_stopping_rounds=30`` to avoid overfit given the typically small
  training sets we'll have early on.
* **Class weighting**: ``class_weight="balanced"`` via ``is_unbalance=True``
  so that a 40/60 win/loss base rate doesn't pin predictions to the majority
  class.
* **Per-family metrics** with a minimum-support cutoff (10 samples) — any
  family below the cutoff is reported as ``None`` and noted separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from filo_hft.config import FiloConfig
from filo_hft.ml.features import FAMILIES, FEATURE_COLUMNS


# ----------------------------------------------------------------------
# Defaults

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "verbosity": -1,
    "is_unbalance": True,
}

EARLY_STOPPING_ROUNDS: int = 30
MIN_FAMILY_SUPPORT: int = 10
VAL_FRACTION: float = 0.20


# ----------------------------------------------------------------------
# Dataclasses


@dataclass
class TrainedModel:
    """Serializable bundle holding the trained booster plus metadata.

    The booster itself is the LightGBM ``LGBMClassifier`` instance; we keep
    the sklearn wrapper (not the raw ``Booster``) so prediction stays a
    one-liner in :class:`filo_hft.ml.infer.BaselinePredictor`.
    """

    model: lgb.LGBMClassifier
    feature_columns: list[str]
    training_date: str
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    per_family_metrics: dict[str, dict[str, float] | None]
    params: dict[str, Any] = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0

    # ---- convenience ------------------------------------------------
    def to_report_dict(self) -> dict[str, Any]:
        """Plain dict view (sans the big booster) for logging."""
        d = asdict(self)
        d.pop("model", None)
        return d


# ----------------------------------------------------------------------
# Metric helpers


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC AUC that returns NaN if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _precision_at_k(
    y_true: np.ndarray, y_prob: np.ndarray, k_frac: float
) -> float:
    """Precision over the top ``k_frac`` predictions by probability."""
    if len(y_true) == 0:
        return float("nan")
    k = max(1, int(np.ceil(len(y_true) * k_frac)))
    top_idx = np.argsort(-y_prob)[:k]
    return float(y_true[top_idx].mean())


def _basic_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute the standard dict of metrics at threshold 0.5."""
    if len(y_true) == 0:
        return {
            "auc": float("nan"),
            "avg_precision": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "precision_top_quartile": float("nan"),
            "n": 0,
            "pos_rate": float("nan"),
        }
    y_hat = (y_prob >= 0.5).astype(int)
    return {
        "auc": _safe_auc(y_true, y_prob),
        "avg_precision": (
            float(average_precision_score(y_true, y_prob))
            if len(np.unique(y_true)) > 1
            else float("nan")
        ),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "precision_top_quartile": _precision_at_k(y_true, y_prob, 0.25),
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
    }


def _per_family_metrics(
    meta: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray
) -> dict[str, dict[str, float] | None]:
    """Metric dict keyed by family, ``None`` if support < MIN_FAMILY_SUPPORT."""
    out: dict[str, dict[str, float] | None] = {}
    fam_series = meta["family"].astype(str).values
    for fam in FAMILIES:
        mask = fam_series == fam
        support = int(mask.sum())
        if support < MIN_FAMILY_SUPPORT:
            out[fam] = None
            continue
        out[fam] = _basic_metrics(y_true[mask], y_prob[mask])
    return out


# ----------------------------------------------------------------------
# Training


def _time_split(
    X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, val_fraction: float
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Split ``X / y / meta`` chronologically: earlier → train, later → val.

    Assumes the frames are already sorted by ``meta['ts_open']`` (they are,
    by construction inside ``build_dataset``), but we re-sort defensively
    here because callers may filter rows.
    """
    order = np.argsort(meta["ts_open"].values, kind="stable")
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    meta = meta.iloc[order].reset_index(drop=True)

    n = len(X)
    split = max(1, int(n * (1.0 - val_fraction)))
    X_tr, X_va = X.iloc[:split], X.iloc[split:]
    y_tr, y_va = y.iloc[:split], y.iloc[split:]
    m_tr, m_va = meta.iloc[:split], meta.iloc[split:]
    return X_tr, X_va, y_tr, y_va, m_tr, m_va


def train_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    cfg: FiloConfig,  # noqa: ARG001 — reserved for future tuning overrides
    params: dict[str, Any] | None = None,
) -> TrainedModel:
    """Fit a LightGBM classifier and return a :class:`TrainedModel` bundle.

    The training set is the first 80% of rows by ``ts_open``; the remaining
    20% is the validation fold used for both early stopping and the reported
    validation metrics.
    """
    if len(X) == 0:
        raise ValueError("train_baseline called with an empty feature frame")

    # Defensive column alignment: reindex to the canonical order so a
    # hand-constructed frame with extra / reordered columns still works.
    X = X.reindex(columns=FEATURE_COLUMNS, fill_value=0.0).astype(np.float32)

    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    X_tr, X_va, y_tr, y_va, m_tr, m_va = _time_split(X, y, meta, VAL_FRACTION)

    model = lgb.LGBMClassifier(**p)
    fit_kwargs: dict[str, Any] = {}
    # Only pass an eval set (and therefore early stopping) if the val fold
    # has both classes — otherwise LightGBM raises on AUC / logloss eval.
    if len(X_va) > 0 and len(np.unique(y_va)) > 1:
        fit_kwargs["eval_set"] = [(X_va, y_va)]
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ]

    model.fit(X_tr, y_tr, **fit_kwargs)

    # --- metrics ------------------------------------------------------
    train_prob = model.predict_proba(X_tr)[:, 1]
    train_metrics = _basic_metrics(y_tr.values, train_prob)

    if len(X_va) > 0:
        val_prob = model.predict_proba(X_va)[:, 1]
        val_metrics = _basic_metrics(y_va.values, val_prob)
        per_family = _per_family_metrics(m_va, y_va.values, val_prob)
    else:
        val_metrics = _basic_metrics(
            np.array([], dtype=np.int8), np.array([], dtype=float)
        )
        per_family = {fam: None for fam in FAMILIES}

    return TrainedModel(
        model=model,
        feature_columns=list(FEATURE_COLUMNS),
        training_date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        per_family_metrics=per_family,
        params=p,
        n_train=int(len(X_tr)),
        n_val=int(len(X_va)),
    )


# ----------------------------------------------------------------------
# Persistence


def save_model(tm: TrainedModel, path: str | Path) -> Path:
    """Persist a :class:`TrainedModel` to disk via joblib.

    Creates the parent directory if it doesn't exist, and returns the
    resolved absolute path to the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tm, out)
    return out.resolve()
