"""
Vectorized rolling LRC (linear regression channel) using sliding_window_view.
Computes midline, std, deviation_sigma for every bar in one pass.
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def rolling_lrc(closes: np.ndarray, length: int):
    """
    Returns (midline, std_dev) arrays of same length as closes.
    First (length-1) values are NaN.
    midline = fitted LRC value at the LAST bar of each window.
    """
    n = len(closes)
    if n < length:
        nan = np.full(n, np.nan)
        return nan, nan

    windows = sliding_window_view(closes.astype(np.float64), length)   # (n-L+1, L)
    x       = np.arange(length, dtype=np.float64)
    x_mean  = x.mean()
    x_c     = x - x_mean
    x_var   = float((x_c ** 2).sum())

    y_mean  = windows.mean(axis=1)                   # (n-L+1,)
    # slope via dot product (vectorised over all windows simultaneously)
    y_c     = windows - y_mean[:, None]              # (n-L+1, L)
    slope   = (y_c * x_c).sum(axis=1) / x_var       # (n-L+1,)
    intercept = y_mean - slope * x_mean              # (n-L+1,)

    fitted  = intercept[:, None] + slope[:, None] * x   # (n-L+1, L)
    resid   = windows - fitted
    std     = resid.std(axis=1, ddof=0)              # (n-L+1,)

    midline_end = intercept + slope * (length - 1)   # fitted value at last bar

    pad = np.full(length - 1, np.nan)
    return np.concatenate([pad, midline_end]), np.concatenate([pad, std])


def compute_lrc_signals(df: pd.DataFrame, lrc_length: int, sigma_mult: float) -> pd.DataFrame:
    """
    Build a per-bar SignalFrame.
    Returns DataFrame aligned to df.index with columns:
      lrc_mid, lrc_upper, lrc_lower, lrc_std, dev_sigma, position, lrc_score
    """
    closes = df["Close"].values
    mid, std = rolling_lrc(closes, lrc_length)

    with np.errstate(invalid="ignore", divide="ignore"):
        dev = np.where(std > 0, (closes - mid) / std, np.nan)

    upper = mid + sigma_mult * std
    lower = mid - sigma_mult * std

    # position: +1 above upper, -1 below lower, 0 inside
    pos = np.where(closes > upper, 1,
          np.where(closes < lower, -1, 0)).astype(np.int8)
    pos[np.isnan(dev)] = 0

    # simple score: deviation magnitude clipped to [0, 30] (30 = 3 σ)
    lrc_score = np.clip(np.abs(dev) * 10, 0, 30)

    return pd.DataFrame({
        "lrc_mid":   mid,
        "lrc_upper": upper,
        "lrc_lower": lower,
        "lrc_std":   std,
        "dev_sigma": dev,
        "position":  pos,
        "lrc_score": lrc_score,
    }, index=df.index)
