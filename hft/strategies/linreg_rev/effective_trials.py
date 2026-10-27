"""
Effective number of independent trials in a correlated parameter sweep.

Why this matters
----------------
The Deflated Sharpe Ratio asks how impressive the best result is GIVEN how many
things were tried. Feed it the raw grid size and it over-deflates and kills real
edges; feed it 1 and it under-deflates and blesses noise. Our grids are 10k-40k
configurations built from a handful of parameters over one instrument, so the
configs are massively correlated: neighbouring cells trade nearly the same bars.
They are nowhere near that many independent bets.

Three estimators, deliberately reported together
------------------------------------------------
  participation ratio   N_eff = (sum L)^2 / sum L^2 over correlation eigenvalues.
                        Smooth, no threshold to pick, but sensitive to a long
                        tail of tiny eigenvalues.
  variance-explained    number of principal components needed to reach some
                        fraction of total variance. Interpretable, threshold-y.
  clustering            count of clusters after hierarchical clustering on
                        1 - |corr|. Closest to "how many distinct strategies is
                        this really", and the one to prefer when they disagree.

They will not agree exactly. Report the range and run the DSR at the most
conservative (largest) value as the headline; a result that only survives at the
smallest estimate is not a result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _clean_matrix(X: np.ndarray) -> np.ndarray:
    """Drop all-NaN / zero-variance columns and mean-fill the rest."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("expected a 2-D (observations x strategies) matrix")
    col_mu = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if len(inds[0]):
        X = X.copy()
        X[inds] = np.take(col_mu, inds[1])
    sd = X.std(axis=0, ddof=1)
    keep = np.isfinite(sd) & (sd > 1e-12)
    return X[:, keep]


def correlation_eigenvalues(X: np.ndarray) -> np.ndarray:
    X = _clean_matrix(X)
    if X.shape[1] < 2:
        return np.array([1.0])
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    ev = np.linalg.eigvalsh(C)
    return np.clip(ev, 0.0, None)[::-1]


def participation_ratio(X: np.ndarray) -> float:
    ev = correlation_eigenvalues(X)
    s1, s2 = ev.sum(), (ev ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 1.0


def n_components_for_variance(X: np.ndarray, frac: float = 0.95) -> int:
    ev = correlation_eigenvalues(X)
    tot = ev.sum()
    if tot <= 0:
        return 1
    return int(np.searchsorted(np.cumsum(ev) / tot, frac) + 1)


def cluster_count(X: np.ndarray, corr_threshold: float = 0.5) -> int:
    """
    Hierarchical (average-linkage) clustering on 1 - |corr|, cut so that members
    of a cluster correlate above ``corr_threshold``.
    """
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
    except ImportError:
        return -1
    X = _clean_matrix(X)
    if X.shape[1] < 2:
        return X.shape[1]
    C = np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)
    D = 1.0 - np.abs(C)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    D[D < 0] = 0.0
    Z = linkage(squareform(D, checks=False), method="average")
    return int(fcluster(Z, t=1.0 - corr_threshold, criterion="distance").max())


def effective_trials(X: np.ndarray, corr_threshold: float = 0.5,
                     var_frac: float = 0.95, verbose: bool = True) -> dict:
    """
    ``X`` : (observations x strategies). Per-period returns are ideal; a matrix
    of per-bar signal indicators works as a proxy when returns are unavailable.
    """
    Xc = _clean_matrix(X)
    n_obs, n_raw = Xc.shape
    ev = correlation_eigenvalues(Xc)
    C = np.nan_to_num(np.corrcoef(Xc, rowvar=False), nan=0.0)
    iu = np.triu_indices_from(C, k=1)
    out = {
        "n_strategies": int(n_raw),
        "n_obs": int(n_obs),
        "mean_abs_corr": float(np.mean(np.abs(C[iu]))) if len(iu[0]) else 0.0,
        "median_corr": float(np.median(C[iu])) if len(iu[0]) else 0.0,
        "participation_ratio": participation_ratio(Xc),
        "pcs_for_variance": n_components_for_variance(Xc, var_frac),
        "clusters": cluster_count(Xc, corr_threshold),
        "top_eigenvalue_share": float(ev[0] / ev.sum()) if ev.sum() > 0 else 1.0,
    }
    cands = [out["participation_ratio"], out["pcs_for_variance"]]
    if out["clusters"] > 0:
        cands.append(out["clusters"])
    out["n_eff_min"] = float(min(cands))
    out["n_eff_max"] = float(max(cands))
    out["n_eff_recommended"] = float(out["clusters"]) if out["clusters"] > 0 \
        else float(out["participation_ratio"])
    if verbose:
        print(f"  strategies={out['n_strategies']}  obs={out['n_obs']}  "
              f"mean|corr|={out['mean_abs_corr']:.3f}")
        print(f"  participation ratio : {out['participation_ratio']:.1f}")
        print(f"  PCs for {var_frac:.0%} var    : {out['pcs_for_variance']}")
        print(f"  clusters (r>{corr_threshold})    : {out['clusters']}")
        print(f"  -> n_eff range {out['n_eff_min']:.0f} .. {out['n_eff_max']:.0f}"
              f"  (recommended {out['n_eff_recommended']:.0f})")
    return out


def returns_matrix_from_trades(trade_frames: dict, freq: str = "D") -> pd.DataFrame:
    """
    Turn {config_label: trades_df} into an aligned (period x config) return
    matrix, which is what the correlation estimates need.

    Trades are bucketed by exit-free calendar period and summed, so configs that
    trade the same days on the same side show up as correlated.
    """
    cols = {}
    for k, t in trade_frames.items():
        if t is None or len(t) == 0:
            continue
        s = pd.Series(t["r"].to_numpy(float),
                      index=pd.to_datetime(t["utc"])).resample(freq).sum()
        cols[k] = s
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).fillna(0.0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("VALIDATION")

    print("\n1. 200 independent noise strategies -> n_eff should be ~200")
    X = rng.normal(size=(2000, 200))
    r = effective_trials(X)
    ok1 = r["participation_ratio"] > 150
    print("   PASS" if ok1 else "   FAIL")

    print("\n2. 200 columns that are 5 distinct signals + noise -> n_eff ~5-ish")
    base = rng.normal(size=(2000, 5))
    X = np.hstack([base[:, [i % 5]] + 0.15 * rng.normal(size=(2000, 1))
                   for i in range(200)])
    r = effective_trials(X)
    ok2 = r["clusters"] <= 12 and r["participation_ratio"] < 30
    print("   PASS" if ok2 else "   FAIL")

    print("\n3. 200 identical columns -> n_eff should collapse to ~1")
    col = rng.normal(size=(2000, 1))
    X = np.repeat(col, 200, axis=1) + 1e-9 * rng.normal(size=(2000, 200))
    r = effective_trials(X)
    ok3 = r["participation_ratio"] < 3 and r["clusters"] <= 2
    print("   PASS" if ok3 else "   FAIL")

    print("\n4. deflation sensitivity: what n_eff does to the DSR verdict")
    try:
        from stats_battery import deflated_sharpe_ratio
        sr, nobs = 0.075, 1500
        for nt in (1, 5, 50, 500, 20000):
            d = deflated_sharpe_ratio(sr, nobs, nt, skew=-0.2, kurt=5.0)
            print(f"   n_trials={nt:>6}  DSR={d:.4f}"
                  f"   {'survives' if d > 0.95 else 'killed'}")
        print("   -> this is why the trial count has to be estimated, not guessed")
    except Exception as e:
        print(f"   (stats_battery unavailable: {e})")

    print(f"\nOVERALL: {'PASS' if (ok1 and ok2 and ok3) else 'FAIL'}")
