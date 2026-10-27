"""Parameter sweep runner.

Runs the FiloHFT backtest engine across a Cartesian product of parameter
overrides, in parallel across cores. Each worker receives a deep-copied
`FiloConfig` with its overrides applied; bars are loaded once up front
and shared via the joblib payload so they are not reloaded per point.

Outputs (when `out_dir` is given):
    sweep_results.parquet          — one row per config point, metrics flat
    sweep_trades/<sig>.parquet     — full trade log per point
    sweep_summary.md               — top-N configs by the ranking metric

The Cartesian product is capped at 200 combinations by default; set
`cap=None` (or pass `--no-cap` on the CLI) to disable the cap.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from filo_hft.config import FiloConfig

# The engine + metrics are imported lazily inside the worker so that the
# parent process can still import this module even if the backtest agent
# hasn't landed its files yet. When the worker runs, the import must
# succeed or the whole sweep fails loudly.

logger = logging.getLogger(__name__)

DEFAULT_COMBINATION_CAP: int = 200
DEFAULT_RANK_METRIC: str = "expectancy_pips"


# ---------------------------------------------------------------------------
# ParamGrid


@dataclass
class ParamGrid:
    """A Cartesian grid of dotted-key parameter overrides.

    Examples
    --------
    >>> grid = ParamGrid({"channel.length": [80, 100, 120]})
    >>> list(grid.iter_points())
    [{'channel.length': 80}, {'channel.length': 100}, {'channel.length': 120}]

    >>> grid = ParamGrid({
    ...     "channel.length": [80, 100],
    ...     "setups.min_setup_score": [0.50, 0.60],
    ... })
    >>> len(list(grid.iter_points()))
    4
    """

    axes: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # normalise: force list-of-values, reject empty axes
        normalised: dict[str, list[Any]] = {}
        for key, values in self.axes.items():
            if not isinstance(key, str) or "." not in key:
                raise ValueError(
                    f"ParamGrid keys must be dotted (e.g. 'channel.length'); got {key!r}"
                )
            values_list = list(values)
            if not values_list:
                raise ValueError(f"ParamGrid axis {key!r} has no values")
            normalised[key] = values_list
        self.axes = normalised

    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        if not self.axes:
            return 0
        total = 1
        for vals in self.axes.values():
            total *= len(vals)
        return total

    def iter_points(self) -> Iterable[dict[str, Any]]:
        if not self.axes:
            return iter(())
        keys = list(self.axes.keys())
        value_lists = [self.axes[k] for k in keys]
        return (dict(zip(keys, combo)) for combo in itertools.product(*value_lists))

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# helpers


def _signature(overrides: Mapping[str, Any]) -> str:
    """Short, deterministic filename-safe hash of an overrides dict."""
    payload = json.dumps(overrides, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return digest


def _apply_overrides(cfg: FiloConfig, overrides: Mapping[str, Any]) -> FiloConfig:
    new_cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        new_cfg.set(key, value)
    return new_cfg


def _flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a possibly-nested metrics dict into scalar columns."""
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, Mapping):
            for subkey, subval in value.items():
                out[f"{key}.{subkey}"] = subval
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# worker


def _run_one_point(
    overrides: dict[str, Any],
    bars_by_tf: dict[str, pd.DataFrame],
    corr_bars: dict[str, pd.DataFrame],
    base_cfg: FiloConfig,
) -> dict[str, Any]:
    """Execute a single backtest for one override point.

    Returns a dict with the flat metric row and the trade log as a
    plain DataFrame (picklable). Importing the engine here keeps this
    module importable even when the engine module hasn't landed yet.
    """
    # Imports are local so joblib workers don't choke if the engine
    # module is rebuilt between sweeps.
    from filo_hft.backtest.engine import run_backtest  # noqa: WPS433
    from filo_hft.backtest.metrics import compute_metrics  # noqa: WPS433

    cfg = _apply_overrides(base_cfg, overrides)
    result = run_backtest(bars_by_tf, cfg, corr_bars)

    trades: pd.DataFrame
    if hasattr(result, "trades") and result.trades is not None:
        trades = pd.DataFrame(result.trades).copy()
    else:
        trades = pd.DataFrame()

    # Prefer engine-provided metrics, fall back to compute_metrics.
    metrics_dict: dict[str, Any]
    if hasattr(result, "metrics") and isinstance(result.metrics, Mapping):
        metrics_dict = dict(result.metrics)
    else:
        metrics_dict = dict(compute_metrics(trades, cfg))

    row: dict[str, Any] = {"param_signature": _signature(overrides)}
    for key, value in overrides.items():
        row[f"param.{key}"] = value
    row.update(_flatten_metrics(metrics_dict))
    row["trade_count"] = int(len(trades))

    return {"row": row, "trades": trades, "overrides": overrides}


# ---------------------------------------------------------------------------
# public API


def run_single_param_sweep(
    bars_by_tf: dict[str, pd.DataFrame],
    corr_bars: dict[str, pd.DataFrame],
    cfg: FiloConfig,
    key: str,
    values: Sequence[Any],
    *,
    n_jobs: int = -1,
    out_dir: str | Path | None = None,
    rank_metric: str = DEFAULT_RANK_METRIC,
    cap: int | None = DEFAULT_COMBINATION_CAP,
) -> pd.DataFrame:
    """Sweep a single dotted parameter across `values`.

    Convenience wrapper around `run_sweep` that builds a one-axis grid.
    """
    grid = ParamGrid({key: list(values)})
    return run_sweep(
        bars_by_tf=bars_by_tf,
        corr_bars=corr_bars,
        cfg=cfg,
        param_grid=grid,
        n_jobs=n_jobs,
        out_dir=out_dir,
        rank_metric=rank_metric,
        cap=cap,
    )


def run_cross_sweep(
    bars_by_tf: dict[str, pd.DataFrame],
    corr_bars: dict[str, pd.DataFrame],
    cfg: FiloConfig,
    axes: Mapping[str, Sequence[Any]],
    *,
    n_jobs: int = -1,
    out_dir: str | Path | None = None,
    rank_metric: str = DEFAULT_RANK_METRIC,
    cap: int | None = DEFAULT_COMBINATION_CAP,
) -> pd.DataFrame:
    """Sweep the Cartesian product of 2+ dotted parameters."""
    grid = ParamGrid({k: list(v) for k, v in axes.items()})
    return run_sweep(
        bars_by_tf=bars_by_tf,
        corr_bars=corr_bars,
        cfg=cfg,
        param_grid=grid,
        n_jobs=n_jobs,
        out_dir=out_dir,
        rank_metric=rank_metric,
        cap=cap,
    )


def run_sweep(
    bars_by_tf: dict[str, pd.DataFrame],
    corr_bars: dict[str, pd.DataFrame],
    cfg: FiloConfig,
    param_grid: ParamGrid | Mapping[str, Sequence[Any]],
    *,
    n_jobs: int = -1,
    out_dir: str | Path | None = None,
    rank_metric: str = DEFAULT_RANK_METRIC,
    cap: int | None = DEFAULT_COMBINATION_CAP,
    top_n: int = 10,
) -> pd.DataFrame:
    """Run a parameter sweep in parallel and optionally persist results.

    Parameters
    ----------
    bars_by_tf
        Dict keyed by "M1"/"M5"/"M15"/"M30"/"H1"/"H4"/"D1" mapping to
        OHLCV DataFrames. Loaded once and passed by reference to each
        worker; do NOT reload in the worker.
    corr_bars
        Correlation symbols keyed by symbol name.
    cfg
        Base FiloConfig — deep-copied for every worker before the
        overrides are applied.
    param_grid
        Either a `ParamGrid` or a plain mapping of dotted keys to value
        lists.
    n_jobs
        Number of parallel workers (joblib loky backend).
    out_dir
        If provided, writes sweep_results.parquet, sweep_trades/*.parquet
        and sweep_summary.md under this directory.
    rank_metric
        Metric column used to rank the top-N summary. Silently falls
        back to total PnL if the metric is missing.
    cap
        Hard limit on the number of combinations. Pass `None` to
        disable. When exceeded a warning is emitted and the sweep
        aborts unless the cap is lifted.
    top_n
        Rows shown in the top-N markdown summary.
    """
    grid = (
        param_grid
        if isinstance(param_grid, ParamGrid)
        else ParamGrid({k: list(v) for k, v in param_grid.items()})
    )

    n_points = grid.size
    if n_points == 0:
        raise ValueError("param_grid is empty")

    if cap is not None and n_points > cap:
        msg = (
            f"sweep has {n_points} combinations which exceeds the default cap "
            f"of {cap}; pass cap=None (or --no-cap on the CLI) to override"
        )
        warnings.warn(msg, stacklevel=2)
        raise ValueError(msg)

    points = list(grid.iter_points())

    logger.info("running sweep: %d points, n_jobs=%s", n_points, n_jobs)

    worker_outputs: list[dict[str, Any]] = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_one_point)(pt, bars_by_tf, corr_bars, cfg) for pt in points
    )

    rows = [wo["row"] for wo in worker_outputs]
    results_df = pd.DataFrame(rows)

    # Ensure a stable column order: param.* first, then metrics.
    param_cols = sorted(c for c in results_df.columns if c.startswith("param."))
    other_cols = [c for c in results_df.columns if c not in param_cols]
    # Put param_signature at the very front
    if "param_signature" in other_cols:
        other_cols.remove("param_signature")
        ordered = ["param_signature"] + param_cols + other_cols
    else:
        ordered = param_cols + other_cols
    results_df = results_df[ordered]

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        trades_dir = out_path / "sweep_trades"
        trades_dir.mkdir(parents=True, exist_ok=True)

        results_df.to_parquet(out_path / "sweep_results.parquet", index=False)

        for wo in worker_outputs:
            sig = wo["row"]["param_signature"]
            trades = wo["trades"]
            if trades is None or len(trades) == 0:
                # still write an empty frame so the dashboard finds it
                pd.DataFrame().to_parquet(trades_dir / f"{sig}.parquet")
            else:
                trades.to_parquet(trades_dir / f"{sig}.parquet", index=False)

        summary_md = _build_summary_md(results_df, rank_metric=rank_metric, top_n=top_n)
        (out_path / "sweep_summary.md").write_text(summary_md, encoding="utf-8")

    return results_df


def _build_summary_md(
    results_df: pd.DataFrame,
    *,
    rank_metric: str,
    top_n: int,
) -> str:
    """Render a tiny markdown table with the top-N configs."""
    if results_df.empty:
        return "# Sweep summary\n\n(no results)\n"

    rank_col = rank_metric
    if rank_col not in results_df.columns:
        # sensible fallback chain
        for candidate in ("expectancy_pips", "total_pnl_usd", "sharpe"):
            if candidate in results_df.columns:
                rank_col = candidate
                break
        else:
            rank_col = results_df.columns[-1]

    df = results_df.copy()
    df = df.sort_values(rank_col, ascending=False).head(top_n)

    show_cols: list[str] = [c for c in df.columns if c.startswith("param.")]
    for extra in (
        "total_pnl_usd",
        "trade_count",
        "win_rate",
        "expectancy_pips",
        "sharpe",
        "max_dd_pct",
    ):
        if extra in df.columns and extra not in show_cols:
            show_cols.append(extra)
    if rank_col not in show_cols:
        show_cols.append(rank_col)

    view = df[show_cols]

    lines: list[str] = []
    lines.append("# Sweep summary")
    lines.append("")
    lines.append(f"Total configurations evaluated: **{len(results_df)}**")
    lines.append("")
    lines.append(f"Top {len(view)} by `{rank_col}`:")
    lines.append("")
    lines.append("| " + " | ".join(show_cols) + " |")
    lines.append("|" + "|".join("---" for _ in show_cols) + "|")
    for _, row in view.iterrows():
        cells = []
        for col in show_cols:
            val = row[col]
            if isinstance(val, float) and not np.isnan(val):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)
