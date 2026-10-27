"""
Validation harness: the tests that decide whether an edge is real.

1. random_null    -- Monte Carlo of random entries with identical exits, count
                     and session mix. A long-only strategy in a market that rose
                     42% will look profitable; the only question that matters is
                     whether the signal beats coin-flip entries in the SAME
                     market. This is the single most important test here.
2. walk_forward   -- rolling optimise-then-trade, so every trade evaluated is
                     out of sample with respect to its own parameters.
3. by_period      -- performance per quarter, to expose regime dependence.
4. mc_bootstrap   -- block bootstrap of the trade sequence for a confidence
                     interval on expectancy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import (Params, load, add_indicators, make_signals, build_paths,
                    simulate_side, silver_gate, stats, backtest, POINT, CONTRACT)


# ---------------------------------------------------------------- random null

def random_null(symbol: str, p: Params, n_trades: int, n_iter: int = 500,
                side: int = +1, session_match: pd.Series | None = None,
                seed: int = 7) -> pd.DataFrame:
    """
    Draw ``n_trades`` random entry bars and run them through the same exit rule.

    ``session_match`` : an optional session->count mapping. When supplied the
    random entries reproduce the signal's session distribution, so the null is
    not accidentally beaten just by trading at different hours.
    """
    rng = np.random.default_rng(seed)
    d = add_indicators(load(symbol, "M5"), p, symbol)
    n = len(d)
    hz = int(p.max_bars)
    atr_arr = d["atr"].to_numpy(float)
    spread = d["spread"].to_numpy(float)
    point, contract = POINT[symbol], CONTRACT[symbol]

    eligible = np.where(np.isfinite(atr_arr)
                        & np.isfinite(d["lrc_lower"].to_numpy(float))
                        & (np.arange(n) < n - hz - 2))[0]
    if session_match is not None:
        sess = d["session"].to_numpy()
        pools = {s: eligible[sess[eligible] == s] for s in session_match.index}

    rows = []
    for it in range(n_iter):
        if session_match is not None:
            picks = np.concatenate([
                rng.choice(pools[s], size=min(int(c), len(pools[s])), replace=False)
                for s, c in session_match.items() if len(pools[s])])
        else:
            picks = rng.choice(eligible, size=min(n_trades, len(eligible)),
                               replace=False)
        picks = np.sort(picks)
        H, L, C, eo = build_paths(d, picks, hz)
        cost = spread[np.minimum(picks + 1, n - 1)] + p.extra_slippage_points
        entry = eo + side * cost * point
        a = atr_arr[picks]
        r, ebar, reason, mfe, mae = simulate_side(
            H, L, C, entry, p.sl_atr * a, p.tp_atr * a, side,
            be_at_r=p.be_at_r,
            trail_dist=(p.trail_atr * a if p.trail_atr > 0 else None))
        comm = p.commission_per_lot_rt / (p.sl_atr * a * contract)
        rnet = r - comm
        rnet = rnet[np.isfinite(rnet)]
        if len(rnet) == 0:
            continue
        rows.append({"iter": it, "n": len(rnet), "total_r": rnet.sum(),
                     "expectancy_r": rnet.mean(),
                     "win_rate": (rnet > 0).mean()})
    return pd.DataFrame(rows)


def null_report(symbol: str, p: Params, trades: pd.DataFrame, side: int = +1,
                n_iter: int = 500, match_sessions: bool = True) -> dict:
    """Percentile rank of the real strategy inside the random-entry null."""
    sm = trades["session"].value_counts() if match_sessions else None
    null = random_null(symbol, p, len(trades), n_iter=n_iter, side=side,
                       session_match=sm)
    real = stats(trades)
    out = {"real_expectancy_r": real["expectancy_r"],
           "real_total_r": real["total_r"], "real_n": real["n"],
           "null_mean_expectancy_r": float(null["expectancy_r"].mean()),
           "null_sd_expectancy_r": float(null["expectancy_r"].std(ddof=1)),
           "null_iters": len(null)}
    if len(null):
        out["pct_rank"] = float((null["expectancy_r"] < real["expectancy_r"]).mean())
        out["p_value_one_sided"] = float(
            (null["expectancy_r"] >= real["expectancy_r"]).mean())
        out["z_vs_null"] = float((real["expectancy_r"] - null["expectancy_r"].mean())
                                 / null["expectancy_r"].std(ddof=1))
    return out


# --------------------------------------------------------------- by period

def by_period(trades: pd.DataFrame, freq: str = "QE") -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame()
    t = trades.copy()
    t["period"] = pd.to_datetime(t["utc"]).dt.to_period(
        {"QE": "Q", "ME": "M"}.get(freq, "Q"))
    g = t.groupby("period")["r"]
    out = pd.DataFrame({"n": g.size(), "total_r": g.sum(),
                        "expectancy_r": g.mean(),
                        "win_rate": g.apply(lambda x: (x > 0).mean())})
    return out.reset_index()


# ------------------------------------------------------------- bootstrap CI

def mc_bootstrap(trades: pd.DataFrame, n_iter: int = 5000, block: int = 10,
                 seed: int = 11) -> dict:
    """Moving-block bootstrap CI for expectancy (blocks keep autocorrelation)."""
    r = trades["r"].to_numpy(float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < block * 2:
        return {}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_iter, nb))
    means = np.empty(n_iter)
    for i in range(n_iter):
        s = np.concatenate([r[a:a + block] for a in starts[i]])[:n]
        means[i] = s.mean()
    return {"expectancy_r": float(r.mean()),
            "ci05": float(np.percentile(means, 5)),
            "ci50": float(np.percentile(means, 50)),
            "ci95": float(np.percentile(means, 95)),
            "p_le_zero": float((means <= 0).mean())}


# ------------------------------------------------------------ walk forward

def walk_forward(symbol: str, param_grid: list[Params], n_folds: int = 6,
                 train_frac: float = 0.6, metric: str = "expectancy_r",
                 min_trades: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling walk-forward. The sample is cut into ``n_folds`` test windows; for
    each, the best parameter set on the preceding train window is selected and
    then traded, untouched, on the test window.

    Returns (per-fold summary, concatenated out-of-sample trades).
    """
    d0 = load(symbol, "M5")
    utc = pd.to_datetime(d0["utc"])
    t0, t1 = utc.iloc[0], utc.iloc[-1]
    span = (t1 - t0) / (n_folds + train_frac / (1 - train_frac))
    train_span = span * (train_frac / (1 - train_frac))

    # cache one full-sample trade list per candidate, then slice by date
    cache = {}
    for i, p in enumerate(param_grid):
        try:
            cache[i] = backtest(symbol, p)
        except Exception:
            cache[i] = pd.DataFrame()

    folds, oos_all = [], []
    for f in range(n_folds):
        te_a = t0 + train_span + span * f
        te_b = te_a + span
        tr_a, tr_b = te_a - train_span, te_a
        best, best_score = None, -np.inf
        for i, p in enumerate(param_grid):
            t = cache[i]
            if len(t) == 0:
                continue
            u = pd.to_datetime(t["utc"])
            tr = t[(u >= tr_a) & (u < tr_b)]
            if len(tr) < min_trades:
                continue
            sc = stats(tr)[metric]
            if np.isfinite(sc) and sc > best_score:
                best, best_score = i, sc
        if best is None:
            folds.append({"fold": f, "test_start": te_a, "test_end": te_b,
                          "chosen": None, "train_score": np.nan, "n": 0,
                          "expectancy_r": np.nan, "total_r": np.nan})
            continue
        t = cache[best]
        u = pd.to_datetime(t["utc"])
        te = t[(u >= te_a) & (u < te_b)].copy()
        te["fold"] = f
        oos_all.append(te)
        st = stats(te)
        folds.append({"fold": f, "test_start": te_a, "test_end": te_b,
                      "chosen": best, "train_score": best_score,
                      "n": st["n"], "expectancy_r": st["expectancy_r"],
                      "total_r": st["total_r"], "win_rate": st["win_rate"]})
    oos = pd.concat(oos_all).reset_index(drop=True) if oos_all else pd.DataFrame()
    return pd.DataFrame(folds), oos
