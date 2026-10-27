"""
stats_battery.py
================

Self-contained statistical validation battery for quantitative strategy
research.  Dependencies: numpy, pandas, scipy only.

References
----------
* Bailey, D. & Lopez de Prado, M. (2012) "The Sharpe Ratio Efficient Frontier",
  Journal of Risk 15(2).                                     -> PSR, MinTRL
* Bailey, D. & Lopez de Prado, M. (2014) "The Deflated Sharpe Ratio",
  Journal of Portfolio Management 40(5).                     -> DSR, E[max SR]
* Bailey, Borwein, Lopez de Prado & Zhu (2017) "The Probability of Backtest
  Overfitting", Journal of Computational Finance 20(4).      -> CSCV / PBO
* Lopez de Prado, M. (2018) "Advances in Financial Machine Learning", ch. 7 & 12.
                                                             -> purging, embargo, CPCV
* Benjamini, Y. & Hochberg, Y. (1995) JRSS-B 57(1).          -> FDR
* Harvey, C. & Liu, Y. (2015) "Backtesting", Journal of Portfolio Management.
                                                             -> Sharpe haircuts
* Kunsch, H. (1989) Annals of Statistics 17(3).              -> moving block bootstrap

CONVENTIONS (read these, they are the usual source of silent errors)
--------------------------------------------------------------------
* `sharpe` is ALWAYS the *non-annualised, per-observation* Sharpe ratio, i.e.
  mean(r) / std(r) over the sampling frequency of the data.  If you have an
  annualised Sharpe from D observations per year, pass sharpe / sqrt(D).
* `skew` is the third standardised moment (0 for a normal).
* `kurt` is the *non-excess* (Pearson) kurtosis: 3.0 for a normal.
  scipy: stats.kurtosis(x, fisher=False).  Passing excess kurtosis will make
  every deflation look too generous.
* `n_trials` is the number of *effectively independent* configurations searched.
  A grid of 40,000 highly-correlated variants is nowhere near 40,000
  independent trials; if you can, cluster them first and pass the cluster count.
  This is the single largest source of approximation in the DSR.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "cpcv_splits",
    "benjamini_hochberg",
    "min_track_record_length",
    "block_bootstrap_ci",
    "haircut_sharpe",
]

EULER_MASCHERONI = 0.5772156649015329
_EPS = 1e-12


# ---------------------------------------------------------------------------
# small internal helpers
# ---------------------------------------------------------------------------
def _sr_denominator(sharpe: float, skew: float, kurt: float) -> float:
    """sqrt(1 - g3*SR + (g4-1)/4 * SR^2) -- the non-normality correction of the
    standard error of the Sharpe estimator (Mertens / Bailey-LdP)."""
    v = 1.0 - skew * sharpe + 0.25 * (kurt - 1.0) * sharpe * sharpe
    if v <= 0.0:
        # can only happen with extreme moment inputs; clip rather than blow up.
        v = _EPS
    return math.sqrt(v)


def _sharpe(x: np.ndarray) -> float:
    s = x.std(ddof=0)
    return float(x.mean() / s) if s > 0 else 0.0


# ---------------------------------------------------------------------------
# 1. Probabilistic / Deflated Sharpe Ratio
# ---------------------------------------------------------------------------
def probabilistic_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sr_benchmark: float = 0.0,
) -> float:
    """P(true SR > sr_benchmark) given an observed SR from `n_obs` observations,
    correcting the estimator's standard error for skew and (non-excess) kurtosis.

    PSR = Phi[ (SR - SR*) * sqrt(n - 1) / sqrt(1 - g3*SR + (g4-1)/4 * SR^2) ]
    """
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    z = (sharpe - sr_benchmark) * math.sqrt(n_obs - 1.0) / _sr_denominator(sharpe, skew, kurt)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max SR] over `n_trials` independent trials whose SR estimates are
    ~N(0, var_sr).  First-order Gumbel (extreme value) approximation from
    Bailey & Lopez de Prado (2014):

        E[max] = sqrt(var) * [ (1-g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N*e)) ]

    with g the Euler-Mascheroni constant.  Returns 0.0 for n_trials <= 1.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    if var_sr < 0:
        raise ValueError("var_sr must be >= 0")
    n = float(n_trials)
    q1 = stats.norm.ppf(1.0 - 1.0 / n)
    q2 = stats.norm.ppf(1.0 - 1.0 / (n * math.e))
    return float(math.sqrt(var_sr) * ((1.0 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2))


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
    sr_benchmark: float = 0.0,
    var_trials: Optional[float] = None,
) -> float:
    """Deflated Sharpe Ratio -- P(true SR > benchmark) after accounting for the
    selection bias induced by searching `n_trials` configurations, and for
    non-normal returns.

    Parameters
    ----------
    sharpe : observed, NON-ANNUALISED Sharpe of the selected strategy.
    n_obs : number of return observations behind `sharpe`.
    n_trials : number of *effectively independent* trials searched.
    skew, kurt : sample skew and NON-EXCESS kurtosis of the selected strategy.
    sr_benchmark : the SR the true strategy must beat (usually 0).
    var_trials : variance of the SR estimates *across trials*.  If None
        (default) it is set to the null-hypothesis variance 1/(n_obs-1),
        i.e. the variance of the SR estimator when the true SR equals
        `sr_benchmark` and returns are normal.  If you actually have the SRs of
        all your trials, pass np.var(trial_sharpes) -- it is strictly better,
        because a wide spread of trial SRs means a higher bar to clear.

    Returns
    -------
    float in [0, 1] : probability that the true SR exceeds `sr_benchmark`.
        Interpret > 0.95 as "survives the multiple-testing correction".

    Notes
    -----
    The deflated benchmark is  SR* = sr_benchmark + E[max SR under the null].
    DSR is then just the PSR evaluated against SR* instead of sr_benchmark.
    """
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    if var_trials is None:
        var_trials = 1.0 / (n_obs - 1.0)
    sr_star = sr_benchmark + expected_max_sharpe(n_trials, var_trials)
    return probabilistic_sharpe_ratio(sharpe, n_obs, skew, kurt, sr_benchmark=sr_star)


# ---------------------------------------------------------------------------
# 2. Probability of Backtest Overfitting (CSCV)
# ---------------------------------------------------------------------------
def probability_of_backtest_overfitting(
    perf_matrix,
    n_splits: int = 16,
    max_combinations: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Dict[str, object]:
    """PBO by Combinatorially Symmetric Cross-Validation (Bailey et al. 2017).

    Parameters
    ----------
    perf_matrix : (T observations x N strategies) array / DataFrame of
        per-period returns (NOT cumulative).  Rows with any NaN are dropped.
    n_splits : number of contiguous sub-periods S (must be even).  All
        C(S, S/2) balanced train/test partitions are evaluated.
    max_combinations : if set and C(S, S/2) exceeds it, a random subsample of
        that many partitions is used (keeps S=20+ tractable).
    random_state : seed for the subsample.

    Method
    ------
    For every balanced partition: rank the N strategies by in-sample Sharpe,
    take the best one, find its rank among the N out-of-sample Sharpes
    (rank 1 = worst, N = best), map to the relative rank w = rank/(N+1) and the
    logit lambda = ln(w / (1-w)).  PBO = P(lambda <= 0) = P(the IS-best strategy
    lands in the bottom half OOS).

    Returns
    -------
    dict with keys:
        'pbo'            : float, the probability of backtest overfitting
        'logits'         : (M,) array of lambda values
        'oos_ranks'      : (M,) array of the OOS rank of the IS-best strategy
        'relative_ranks' : (M,) array of w in (0,1)
        'best_is'        : (M,) array of the strategy index chosen in-sample
        'n_combinations' : M
        'n_strategies'   : N
    """
    X = perf_matrix.values if isinstance(perf_matrix, pd.DataFrame) else np.asarray(perf_matrix)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("perf_matrix must be 2-D (T observations x N strategies)")
    X = X[~np.isnan(X).any(axis=1)]
    T, N = X.shape
    if N < 2:
        raise ValueError("need at least 2 strategies to rank")
    if n_splits % 2 != 0:
        raise ValueError("n_splits must be even")
    if T < n_splits * 2:
        raise ValueError(f"need at least {n_splits * 2} observations for n_splits={n_splits}")

    # --- per-sub-period sufficient statistics: the Sharpe of any union of
    #     sub-periods is an exact function of (count, sum, sum of squares).
    groups = np.array_split(np.arange(T), n_splits)
    cnt = np.array([len(g) for g in groups], dtype=float)                    # (S,)
    s1 = np.vstack([X[g].sum(axis=0) for g in groups])                       # (S,N)
    s2 = np.vstack([(X[g] ** 2).sum(axis=0) for g in groups])                # (S,N)

    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    if max_combinations is not None and len(combos) > max_combinations:
        rng = np.random.default_rng(random_state)
        pick = rng.choice(len(combos), size=max_combinations, replace=False)
        combos = [combos[i] for i in pick]
    M = len(combos)

    A = np.zeros((M, n_splits), dtype=float)
    for i, c in enumerate(combos):
        A[i, list(c)] = 1.0
    B = 1.0 - A  # complement = test

    def _sharpes(sel: np.ndarray) -> np.ndarray:
        n = sel @ cnt                       # (M,)
        m = (sel @ s1) / n[:, None]         # (M,N)
        v = (sel @ s2) / n[:, None] - m ** 2
        sd = np.sqrt(np.maximum(v, _EPS))
        return m / sd

    sr_is = _sharpes(A)
    sr_oos = _sharpes(B)

    best = np.argmax(sr_is, axis=1)                                # (M,)
    ranks = stats.rankdata(sr_oos, axis=1)                         # 1 = worst
    oos_rank = ranks[np.arange(M), best]
    w = oos_rank / (N + 1.0)
    logits = np.log(w / (1.0 - w))
    pbo = float(np.mean(logits <= 0.0))

    return {
        "pbo": pbo,
        "logits": logits,
        "oos_ranks": oos_rank,
        "relative_ranks": w,
        "best_is": best,
        "n_combinations": M,
        "n_strategies": N,
    }


# ---------------------------------------------------------------------------
# 3. Combinatorial Purged Cross-Validation
# ---------------------------------------------------------------------------
def _contiguous_blocks(idx: np.ndarray) -> List[Tuple[int, int]]:
    """[(start, end_inclusive), ...] for a sorted index array."""
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) != 1)
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def cpcv_splits(
    n_obs: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01,
    purge_pct: Optional[float] = None,
    t1: Optional[Sequence[int]] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Combinatorial Purged Cross-Validation index generator (AFML ch. 7 & 12).

    Observations are cut into `n_groups` contiguous groups; every combination of
    `n_test_groups` groups forms one test set, giving C(n_groups, n_test_groups)
    splits.  For each *contiguous block* of the test set:
      * PURGE : training observations immediately BEFORE the block whose label
        would overlap the test block are dropped.  With `t1` given, an
        observation i is purged iff [i, t1[i]] intersects the block (exact
        purging).  Without `t1`, a fixed window of `purge_pct * n_obs`
        observations before the block is dropped (conservative approximation;
        `purge_pct` defaults to `embargo_pct`).
      * EMBARGO : `embargo_pct * n_obs` training observations immediately AFTER
        the block are dropped, to kill serial-correlation leakage forward.

    Returns
    -------
    list of (train_idx, test_idx) int64 arrays, both sorted, always disjoint.
    """
    if n_obs < n_groups:
        raise ValueError("n_obs must be >= n_groups")
    if not (1 <= n_test_groups < n_groups):
        raise ValueError("need 1 <= n_test_groups < n_groups")
    if not (0.0 <= embargo_pct < 1.0):
        raise ValueError("embargo_pct must be in [0, 1)")

    emb = int(math.ceil(embargo_pct * n_obs))
    pp = embargo_pct if purge_pct is None else purge_pct
    purge = int(math.ceil(pp * n_obs))
    t1_arr = None if t1 is None else np.asarray(t1, dtype=np.int64)
    if t1_arr is not None and t1_arr.shape[0] != n_obs:
        raise ValueError("t1 must have length n_obs")

    groups = np.array_split(np.arange(n_obs, dtype=np.int64), n_groups)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    all_ii = np.arange(n_obs, dtype=np.int64)

    for combo in itertools.combinations(range(n_groups), n_test_groups):
        test_idx = np.sort(np.concatenate([groups[g] for g in combo]))
        keep = np.ones(n_obs, dtype=bool)
        keep[test_idx] = False  # test observations are never in train
        for a, b in _contiguous_blocks(test_idx):
            if t1_arr is not None:
                # exact purge: drop any observation whose label span touches [a, b]
                overlap = (all_ii <= b) & (t1_arr >= a)
                keep &= ~overlap
            elif purge > 0:
                keep[max(0, a - purge):a] = False
            if emb > 0:
                keep[b + 1: min(n_obs, b + 1 + emb)] = False
        train_idx = np.flatnonzero(keep).astype(np.int64)
        out.append((train_idx, test_idx))
    return out


# ---------------------------------------------------------------------------
# 4. Benjamini-Hochberg FDR
# ---------------------------------------------------------------------------
def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg (1995) step-up FDR control under independence / PRDS.

    Returns
    -------
    reject : (m,) bool array, in the ORIGINAL order of `pvals`
    p_adj  : (m,) BH-adjusted p-values (q-values), original order, capped at 1.
             p_adj_(i) = min_{k >= i} ( m/k * p_(k) ), enforced monotone.
    """
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvals must be 1-D")
    if np.any(np.isnan(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("pvals must all lie in [0, 1] and be non-NaN")
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)

    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    ranks = np.arange(1, m + 1, dtype=float)

    # rejection: largest i with p_(i) <= i/m * alpha, then reject 1..i
    below = ps <= (ranks / m) * alpha
    reject_sorted = np.zeros(m, dtype=bool)
    if below.any():
        k = int(np.flatnonzero(below).max())
        reject_sorted[: k + 1] = True

    # adjusted p-values: running minimum from the largest p downwards
    adj_sorted = np.minimum.accumulate((m / ranks * ps)[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    reject = np.empty(m, dtype=bool)
    p_adj = np.empty(m, dtype=float)
    reject[order] = reject_sorted
    p_adj[order] = adj_sorted
    return reject, p_adj


# ---------------------------------------------------------------------------
# 5. Minimum Track Record Length
# ---------------------------------------------------------------------------
def min_track_record_length(
    sharpe: float,
    skew: float = 0.0,
    kurt: float = 3.0,
    target_sharpe: float = 0.0,
    prob: float = 0.95,
) -> float:
    """Number of observations required for the observed (non-annualised) Sharpe
    to be statistically greater than `target_sharpe` with confidence `prob`.

        MinTRL = 1 + (1 - g3*SR + (g4-1)/4 * SR^2) * ( Z^-1(prob) / (SR - SR*) )^2

    Returns np.inf if sharpe <= target_sharpe (no sample size ever suffices).
    The value is a float; round up for a usable observation count.  By
    construction probabilistic_sharpe_ratio(sharpe, MinTRL, ...) == prob.
    """
    if not (0.0 < prob < 1.0):
        raise ValueError("prob must be in (0, 1)")
    if sharpe <= target_sharpe:
        return float("inf")
    d = _sr_denominator(sharpe, skew, kurt)
    z = stats.norm.ppf(prob)
    return float(1.0 + (d ** 2) * (z / (sharpe - target_sharpe)) ** 2)


# ---------------------------------------------------------------------------
# 6. Moving-block bootstrap CI for the mean
# ---------------------------------------------------------------------------
def block_bootstrap_ci(
    returns: Sequence[float],
    n_iter: int = 10000,
    block: int = 10,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
    return_dist: bool = False,
):
    """Moving-block bootstrap (Kunsch 1989) percentile CI for the mean.

    Overlapping blocks of length `block` are drawn with replacement until the
    resample is at least as long as the original series, then truncated to n.
    This preserves dependence up to lag `block-1`, which matters for
    autocorrelated PnL series where an iid bootstrap understates the variance.

    Returns
    -------
    (lo, hi) : two-sided (1-alpha) percentile interval for E[returns].
    If return_dist=True, returns (lo, hi, boot_means).
    """
    x = np.asarray(returns, dtype=float).ravel()
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        raise ValueError("returns is empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    block = int(max(1, min(block, n)))
    n_blocks = int(math.ceil(n / block))
    n_starts = n - block + 1

    rng = np.random.default_rng(random_state)
    starts = rng.integers(0, n_starts, size=(n_iter, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_iter, -1)[:, :n]
    boot_means = x[idx].mean(axis=1)

    lo, hi = np.percentile(boot_means, [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)])
    if return_dist:
        return float(lo), float(hi), boot_means
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# 7. Multiple-testing Sharpe haircuts (Harvey & Liu style)
# ---------------------------------------------------------------------------
def haircut_sharpe(
    sharpe: float,
    n_trials: int,
    n_obs: int,
    other_pvals: Optional[Sequence[float]] = None,
    one_sided: bool = True,
) -> Dict[str, float]:
    """Bonferroni and Holm haircuts on an observed Sharpe -- a simple, purely
    p-value-based cross-check on the DSR.

    The observed (non-annualised) Sharpe gives t = SR * sqrt(n_obs) and a normal
    p-value.  The p-value is inflated for the `n_trials` configurations searched,
    then inverted back into an "adjusted Sharpe"; the haircut is
    1 - SR_adjusted / SR, clipped to [0, 1] (100% = no evidence left).

    IMPORTANT / honest caveat: for a single strategy that is assumed to be the
    BEST of `n_trials`, Holm's step-down adjustment of the smallest p-value is
    algebraically identical to Bonferroni (m * p).  The two only differ when you
    pass `other_pvals` (the p-values of the other trials) and the observed
    strategy is not the most significant one.  With `other_pvals=None` the other
    trials are filled in with their expected null order statistics
    i/(n_trials+1), so holm == bonferroni by construction.

    Returns a dict: p_single, p_bonferroni, p_holm, sharpe_bonferroni,
    sharpe_holm, haircut_bonferroni, haircut_holm.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")

    t = sharpe * math.sqrt(n_obs)
    p = float(stats.norm.sf(t)) if one_sided else float(2.0 * stats.norm.sf(abs(t)))
    p = min(max(p, 0.0), 1.0)

    p_bonf = min(1.0, p * n_trials)

    # Holm step-down over the full family
    if other_pvals is None:
        others = np.arange(2, n_trials + 1, dtype=float) / (n_trials + 1.0)
    else:
        others = np.asarray(other_pvals, dtype=float).ravel()
    fam = np.r_[p, others]
    m = fam.size
    order = np.argsort(fam, kind="mergesort")
    fs = fam[order]
    adj = np.maximum.accumulate((m - np.arange(m)) * fs)
    adj = np.clip(adj, 0.0, 1.0)
    holm_full = np.empty(m)
    holm_full[order] = adj
    p_holm = float(holm_full[0])

    def _invert(p_adj: float) -> float:
        p_adj = min(p_adj, 1.0 - 1e-15)
        tt = stats.norm.isf(p_adj) if one_sided else stats.norm.isf(p_adj / 2.0)
        return float(tt / math.sqrt(n_obs))

    sr_bonf = _invert(p_bonf)
    sr_holm = _invert(p_holm)

    def _hc(sr_adj: float) -> float:
        if sharpe <= 0:
            return 1.0
        return float(min(1.0, max(0.0, 1.0 - sr_adj / sharpe)))

    return {
        "p_single": p,
        "p_bonferroni": p_bonf,
        "p_holm": p_holm,
        "sharpe_bonferroni": sr_bonf,
        "sharpe_holm": sr_holm,
        "haircut_bonferroni": _hc(sr_bonf),
        "haircut_holm": _hc(sr_holm),
    }


# ===========================================================================
# VALIDATION SUITE
# ===========================================================================
if __name__ == "__main__":

    _RESULTS: List[Tuple[str, bool]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        _RESULTS.append((name, bool(ok)))
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}" + (f"  --  {detail}" if detail else ""))

    def section(title: str) -> None:
        print("\n" + "=" * 78)
        print(title)
        print("=" * 78)

    # -------------------------------------------------------------------
    section("1. DEFLATED SHARPE RATIO")
    # -------------------------------------------------------------------
    # (a) deterministic: noise-level Sharpe, huge search -> low probability
    dsr_noise = deflated_sharpe_ratio(sharpe=0.02, n_obs=1000, n_trials=10000,
                                      skew=0.0, kurt=3.0)
    sr_star_noise = expected_max_sharpe(10000, 1.0 / 999.0)
    check("noise Sharpe (0.02/obs, 1000 obs, 10k trials) -> DSR < 0.05",
          dsr_noise < 0.05, f"DSR={dsr_noise:.6f}, SR*={sr_star_noise:.4f}")

    # (b) deterministic: strong signal, few trials -> high probability
    dsr_signal = deflated_sharpe_ratio(sharpe=0.15, n_obs=1000, n_trials=5,
                                       skew=0.0, kurt=3.0)
    check("strong Sharpe (0.15/obs, 1000 obs, 5 trials) -> DSR > 0.99",
          dsr_signal > 0.99, f"DSR={dsr_signal:.6f}")

    # (c) DSR with n_trials=1 must collapse to the plain PSR
    a = deflated_sharpe_ratio(0.08, 500, 1, 0.0, 3.0)
    b = probabilistic_sharpe_ratio(0.08, 500, 0.0, 3.0)
    check("DSR(n_trials=1) == PSR", abs(a - b) < 1e-12, f"{a:.10f} vs {b:.10f}")

    # (d) monotonicity in the number of trials
    seq = [deflated_sharpe_ratio(0.09, 1000, k, 0.0, 3.0) for k in (1, 10, 100, 1000, 10000)]
    check("DSR strictly decreasing in n_trials",
          all(seq[i] > seq[i + 1] for i in range(len(seq) - 1)),
          " > ".join(f"{v:.4f}" for v in seq))

    # (e) non-normality penalties: negative skew and fat tails must hurt
    d_norm = deflated_sharpe_ratio(0.10, 1000, 100, 0.0, 3.0)
    d_negskew = deflated_sharpe_ratio(0.10, 1000, 100, -1.5, 3.0)
    d_fat = deflated_sharpe_ratio(0.10, 1000, 100, 0.0, 12.0)
    check("negative skew lowers DSR", d_negskew < d_norm,
          f"skew=-1.5 -> {d_negskew:.4f} vs normal {d_norm:.4f}")
    check("excess kurtosis lowers DSR", d_fat < d_norm,
          f"kurt=12 -> {d_fat:.4f} vs normal {d_norm:.4f}")

    # (f) Monte-Carlo type-I control: best of M pure-noise strategies
    N_REP, N_STRAT, N_OBS = 200, 200, 500
    rng = np.random.default_rng(11)
    dsr_null = np.zeros(N_REP)
    raw_sig = np.zeros(N_REP)
    for i in range(N_REP):
        R = rng.standard_normal((N_OBS, N_STRAT))
        srs = R.mean(axis=0) / R.std(axis=0, ddof=0)
        j = int(np.argmax(srs))
        r = R[:, j]
        sk = float(stats.skew(r))
        ku = float(stats.kurtosis(r, fisher=False))
        dsr_null[i] = deflated_sharpe_ratio(float(srs[j]), N_OBS, N_STRAT, sk, ku)
        raw_sig[i] = probabilistic_sharpe_ratio(float(srs[j]), N_OBS, sk, ku) > 0.95
    fp_deflated = float(np.mean(dsr_null > 0.95))
    fp_raw = float(np.mean(raw_sig))
    check("MC null: P(DSR > 0.95) <= 0.10 (nominal 5%; DSR is conservative)",
          fp_deflated <= 0.10,
          f"deflated false-positive rate={fp_deflated:.3f} vs UNdeflated PSR rate={fp_raw:.3f}")
    check("MC null: deflation actually bites (raw rate >> deflated rate)",
          fp_raw > 0.8 and fp_deflated < 0.2,
          f"raw={fp_raw:.3f}, deflated={fp_deflated:.3f}, median DSR={np.median(dsr_null):.3f}")

    # (g) Monte-Carlo power: one real strategy, few trials
    rng = np.random.default_rng(12)
    TRUE_SR, N_OBS_P, N_REP_P = 0.15, 1000, 200
    hits = 0
    for _ in range(N_REP_P):
        r = rng.standard_normal(N_OBS_P) + TRUE_SR
        d = deflated_sharpe_ratio(_sharpe(r), N_OBS_P, 5,
                                  float(stats.skew(r)),
                                  float(stats.kurtosis(r, fisher=False)))
        hits += int(d > 0.95)
    power = hits / N_REP_P
    check("MC power: true SR=0.15/obs, 5 trials -> P(DSR > 0.95) > 0.85",
          power > 0.85, f"power={power:.3f}")

    # -------------------------------------------------------------------
    section("2. PROBABILITY OF BACKTEST OVERFITTING (CSCV)")
    # -------------------------------------------------------------------
    # NOTE: PBO for a SINGLE noise dataset is itself a noisy statistic (observed
    # spread ~0.11 to 0.77), because train and test are complementary halves of
    # ONE fixed sample, so their means are negatively coupled across partitions.
    # PBO concentrates on 0.5 only in expectation over datasets -- so the null
    # test averages over independent noise datasets.
    N_DS, T_DS, N_STRAT_DS = 12, 1500, 50
    pbos = []
    for s in range(N_DS):
        R = np.random.default_rng(2024 + s).standard_normal((T_DS, N_STRAT_DS))
        pbos.append(probability_of_backtest_overfitting(R, n_splits=16)["pbo"])
    pbos = np.asarray(pbos)
    check(f"pure noise ({N_DS} independent datasets, 50 strategies) -> mean PBO ~ 0.5",
          0.35 <= pbos.mean() <= 0.65,
          f"mean PBO={pbos.mean():.4f} (per-dataset range "
          f"{pbos.min():.3f}-{pbos.max():.3f})")

    rng = np.random.default_rng(2024)
    noise = rng.standard_normal((1000, 50))
    res_noise = probability_of_backtest_overfitting(noise, n_splits=16)
    check("pure noise: median logit ~ 0 (|median| < 0.6, no systematic edge)",
          abs(float(np.median(res_noise["logits"]))) < 0.6,
          f"single dataset PBO={res_noise['pbo']:.4f} over "
          f"{res_noise['n_combinations']} partitions, "
          f"median logit={float(np.median(res_noise['logits'])):+.4f}")

    rng = np.random.default_rng(7)
    skilled = rng.standard_normal((1000, 20))
    skilled[:, 0] += 0.25  # one genuinely, consistently superior strategy
    res_skill = probability_of_backtest_overfitting(skilled, n_splits=16)
    frac_picked = float(np.mean(res_skill["best_is"] == 0))
    check("one genuinely best strategy -> PBO < 0.05",
          res_skill["pbo"] < 0.05,
          f"PBO={res_skill['pbo']:.4f}, IS-best was strategy 0 in "
          f"{frac_picked * 100:.1f}% of partitions, median logit="
          f"{np.median(res_skill['logits']):+.3f}")

    # structural checks
    check("PBO: number of partitions == C(16, 8) == 12870",
          res_noise["n_combinations"] == math.comb(16, 8),
          f"{res_noise['n_combinations']}")
    check("PBO: relative ranks strictly inside (0, 1)",
          bool(np.all(res_noise["relative_ranks"] > 0)
               and np.all(res_noise["relative_ranks"] < 1)),
          f"min={res_noise['relative_ranks'].min():.4f}, "
          f"max={res_noise['relative_ranks'].max():.4f}")
    # DataFrame input must give an identical answer to ndarray input
    res_df = probability_of_backtest_overfitting(pd.DataFrame(noise), n_splits=16)
    check("PBO: DataFrame input == ndarray input",
          res_df["pbo"] == res_noise["pbo"], f"{res_df['pbo']:.6f}")

    # -------------------------------------------------------------------
    section("3. COMBINATORIAL PURGED CROSS-VALIDATION")
    # -------------------------------------------------------------------
    N_CPCV, NG, NTG, EMB = 1000, 6, 2, 0.02
    splits = cpcv_splits(N_CPCV, n_groups=NG, n_test_groups=NTG, embargo_pct=EMB)
    emb_len = int(math.ceil(EMB * N_CPCV))

    check("split count == C(6, 2) == 15", len(splits) == math.comb(NG, NTG), f"{len(splits)}")

    overlap_bad = [i for i, (tr, te) in enumerate(splits) if np.intersect1d(tr, te).size > 0]
    check("no train index ever overlaps a test index", not overlap_bad,
          f"checked all {len(splits)} splits")
    assert not overlap_bad, "train/test overlap detected"

    emb_violations = 0
    emb_removed = 0
    purge_violations = 0
    purge_removed = 0
    for tr, te in splits:
        tr_set = set(tr.tolist())
        te_set = set(te.tolist())
        for a, b in _contiguous_blocks(te):
            for k in range(b + 1, min(N_CPCV, b + 1 + emb_len)):
                if k in tr_set:
                    emb_violations += 1
                if k not in te_set:
                    emb_removed += 1
            for k in range(max(0, a - emb_len), a):
                if k in tr_set:
                    purge_violations += 1
                if k not in te_set:
                    purge_removed += 1
    check("embargo: no train observation within emb_len AFTER any test block",
          emb_violations == 0, f"violations={emb_violations}, embargo length={emb_len} obs")
    assert emb_violations == 0, "embargo violated"
    check("purge: no train observation within purge_len BEFORE any test block",
          purge_violations == 0, f"violations={purge_violations}")
    assert purge_violations == 0, "purge violated"
    check("embargo actually removes observations (not a no-op)",
          emb_removed > 0, f"{emb_removed} post-block observations embargoed across all splits")
    check("purge actually removes observations (not a no-op)",
          purge_removed > 0, f"{purge_removed} pre-block observations purged across all splits")

    # embargo=0 must yield strictly larger training sets
    splits0 = cpcv_splits(N_CPCV, n_groups=NG, n_test_groups=NTG, embargo_pct=0.0)
    sz0 = sum(len(tr) for tr, _ in splits0)
    sz1 = sum(len(tr) for tr, _ in splits)
    check("embargo/purge shrink the training set vs embargo_pct=0",
          sz1 < sz0, f"total train obs {sz1} (2% embargo) < {sz0} (0% embargo)")

    # every observation must be tested the same number of times
    counts = np.zeros(N_CPCV, dtype=int)
    for _, te in splits:
        counts[te] += 1
    check("every observation is tested exactly C(5,1)=5 times",
          bool(np.all(counts == math.comb(NG - 1, NTG - 1))),
          f"unique test-counts={sorted(set(counts.tolist()))}")

    # exact purging driven by label spans t1
    span = 25
    t1_arr = np.minimum(np.arange(N_CPCV) + span, N_CPCV - 1)
    splits_t1 = cpcv_splits(N_CPCV, n_groups=NG, n_test_groups=NTG,
                            embargo_pct=EMB, t1=t1_arr)
    bad_t1 = 0
    for tr, te in splits_t1:
        te_set = set(te.tolist())
        for i in tr:
            if any(k in te_set for k in range(int(i), int(t1_arr[i]) + 1)):
                bad_t1 += 1
    check("t1 mode: no training label span overlaps the test set",
          bad_t1 == 0, f"overlapping spans={bad_t1} (label span={span} obs)")
    assert bad_t1 == 0, "t1 purging failed"

    # -------------------------------------------------------------------
    section("4. BENJAMINI-HOCHBERG FDR")
    # -------------------------------------------------------------------
    # Worked example from Benjamini & Hochberg (1995), Table 1 (Needleman data)
    bh_p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
            0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]
    rej, padj = benjamini_hochberg(bh_p, alpha=0.05)
    check("BH 1995 worked example: exactly 4 rejections at alpha=0.05",
          int(rej.sum()) == 4 and bool(np.all(rej[:4])) and not bool(np.any(rej[4:])),
          f"rejected indices={np.flatnonzero(rej).tolist()}")
    expect_adj = np.array([0.0015, 0.0030, 0.0095, 0.035625, 0.0603])
    check("BH adjusted p-values match hand computation (first 5)",
          bool(np.allclose(padj[:5], expect_adj, atol=1e-9)),
          f"{np.round(padj[:5], 6).tolist()}")
    check("BH adjusted p-values are monotone in sorted p order",
          bool(np.all(np.diff(padj[np.argsort(bh_p)]) >= -1e-15)))
    check("BH reject <=> p_adj <= alpha", bool(np.array_equal(rej, padj <= 0.05)))
    # unsorted input must give the same answer, permuted
    perm = np.array([7, 0, 14, 3, 9, 1, 11, 2, 5, 13, 4, 8, 12, 6, 10])
    rej_p, padj_p = benjamini_hochberg(np.asarray(bh_p)[perm], alpha=0.05)
    check("BH is order-invariant (permuted input gives permuted output)",
          bool(np.array_equal(rej_p, rej[perm])) and bool(np.allclose(padj_p, padj[perm])))
    # Bonferroni on the same data rejects only 3 -> BH must be at least as powerful
    n_bonf = int(np.sum(np.asarray(bh_p) <= 0.05 / len(bh_p)))
    check("BH is more powerful than Bonferroni on the same data (4 > 3)",
          int(rej.sum()) > n_bonf, f"BH={int(rej.sum())}, Bonferroni={n_bonf}")
    # global null: family-wise rejection rate must be controlled at alpha
    rng = np.random.default_rng(5)
    fdr_rate = float(np.mean([benjamini_hochberg(rng.uniform(size=200), 0.05)[0].any()
                              for _ in range(400)]))
    check("global null: family-wise rejection rate <= alpha",
          fdr_rate <= 0.08,
          f"rate={fdr_rate:.4f} over 400 families of 200 uniform p-values (nominal 0.05)")

    # -------------------------------------------------------------------
    section("5. MINIMUM TRACK RECORD LENGTH")
    # -------------------------------------------------------------------
    SR_T, SK_T, KU_T = 0.08, -0.4, 5.0
    mtrl = min_track_record_length(SR_T, SK_T, KU_T, target_sharpe=0.0, prob=0.95)
    psr_at = probabilistic_sharpe_ratio(SR_T, mtrl, SK_T, KU_T, 0.0)
    check("self-consistency: PSR(SR, n=MinTRL) == prob (0.95)",
          abs(psr_at - 0.95) < 1e-9, f"MinTRL={mtrl:.2f} obs -> PSR={psr_at:.10f}")
    psr_short = probabilistic_sharpe_ratio(SR_T, max(2.0, mtrl * 0.5), SK_T, KU_T, 0.0)
    check("half the MinTRL is not yet significant",
          psr_short < 0.95, f"PSR at {mtrl * 0.5:.0f} obs = {psr_short:.4f}")
    check("MinTRL = inf when Sharpe <= target",
          math.isinf(min_track_record_length(0.05, 0.0, 3.0, target_sharpe=0.05)))
    m_norm = min_track_record_length(0.08, 0.0, 3.0)
    m_bad = min_track_record_length(0.08, -1.0, 8.0)
    check("negative skew / fat tails require a longer track record",
          m_bad > m_norm, f"{m_bad:.0f} obs vs {m_norm:.0f} obs (normal case)")
    # normal case: denominator^2 = 1 - 0*SR + (3-1)/4 * SR^2 = 1 + SR^2/2
    closed = 1.0 + (1.0 + 0.5 * 0.08 ** 2) * (stats.norm.ppf(0.95) / 0.08) ** 2
    check("normal case matches closed form 1 + (1 + SR^2/2)(z_0.95 / SR)^2",
          abs(m_norm - closed) < 1e-9, f"{m_norm:.6f} vs {closed:.6f}")

    # -------------------------------------------------------------------
    section("6. MOVING-BLOCK BOOTSTRAP CI")
    # -------------------------------------------------------------------
    # (a) coverage of the true mean over repeated iid draws
    rng = np.random.default_rng(99)
    N_COV, N_SAMP, TRUE_MU = 300, 400, 0.0
    cov = 0
    for _ in range(N_COV):
        x = rng.standard_normal(N_SAMP)
        lo, hi = block_bootstrap_ci(x, n_iter=300, block=10, alpha=0.05,
                                    random_state=int(rng.integers(1 << 30)))
        cov += int(lo <= TRUE_MU <= hi)
    coverage = cov / N_COV
    check("iid data: 95% CI covers the true mean at ~nominal rate [0.88, 0.99]",
          0.88 <= coverage <= 0.99,
          f"coverage={coverage:.3f} over {N_COV} synthetic draws (nominal 0.95)")

    # (b) autocorrelated data: blocks must beat a naive iid bootstrap
    rng = np.random.default_rng(1234)
    PHI, N_AR, N_AR_REP = 0.6, 400, 200
    cov_blk = cov_iid = 0
    for _ in range(N_AR_REP):
        e = rng.standard_normal(N_AR)
        x = np.empty(N_AR)
        x[0] = e[0]
        for t in range(1, N_AR):
            x[t] = PHI * x[t - 1] + e[t]
        s = int(rng.integers(1 << 30))
        lo, hi = block_bootstrap_ci(x, n_iter=300, block=30, alpha=0.05, random_state=s)
        cov_blk += int(lo <= 0.0 <= hi)
        lo1, hi1 = block_bootstrap_ci(x, n_iter=300, block=1, alpha=0.05, random_state=s)
        cov_iid += int(lo1 <= 0.0 <= hi1)
    c_blk, c_iid = cov_blk / N_AR_REP, cov_iid / N_AR_REP
    check("AR(1) phi=0.6: block bootstrap covers better than an iid bootstrap",
          c_blk > c_iid,
          f"block(30)={c_blk:.3f} vs iid(block=1)={c_iid:.3f} (nominal 0.95)")

    # (c) known-mean recovery and interval sanity
    rng = np.random.default_rng(3)
    x = rng.standard_normal(5000) * 2.0 + 1.0
    lo, hi = block_bootstrap_ci(x, n_iter=4000, block=10, alpha=0.05, random_state=1)
    analytic = 1.96 * x.std(ddof=1) / math.sqrt(x.size)
    check("large sample: CI brackets the true mean and has ~analytic width",
          lo < 1.0 < hi and abs((hi - lo) - 2 * analytic) < 0.35 * (2 * analytic),
          f"CI=({lo:.4f}, {hi:.4f}) width={hi - lo:.4f} vs analytic "
          f"{2 * analytic:.4f}; true mu=1.0")
    wide = block_bootstrap_ci(x, 2000, 10, 0.05, random_state=2)
    narrow = block_bootstrap_ci(x, 2000, 10, 0.20, random_state=2)
    check("larger alpha gives a narrower interval",
          (narrow[1] - narrow[0]) < (wide[1] - wide[0]),
          f"80% width={narrow[1] - narrow[0]:.5f} < 95% width={wide[1] - wide[0]:.5f}")

    # -------------------------------------------------------------------
    section("7. MULTIPLE-TESTING SHARPE HAIRCUTS")
    # -------------------------------------------------------------------
    hc1 = haircut_sharpe(0.08, n_trials=1, n_obs=1000)
    check("n_trials=1 -> zero haircut and p_adj == p_single",
          abs(hc1["haircut_bonferroni"]) < 1e-9
          and abs(hc1["p_bonferroni"] - hc1["p_single"]) < 1e-15,
          f"p={hc1['p_single']:.6f}, haircut={hc1['haircut_bonferroni']:.6f}")

    hcs = [haircut_sharpe(0.08, n_trials=k, n_obs=1000)["haircut_bonferroni"]
           for k in (1, 5, 20, 100, 1000)]
    check("haircut is monotone non-decreasing in n_trials",
          all(hcs[i] <= hcs[i + 1] + 1e-12 for i in range(len(hcs) - 1)),
          " -> ".join(f"{v * 100:.1f}%" for v in hcs))

    hc = haircut_sharpe(0.08, n_trials=20, n_obs=1000)
    check("Holm == Bonferroni for the best-of-N case (documented, not a bug)",
          abs(hc["p_holm"] - hc["p_bonferroni"]) < 1e-12,
          f"p_holm={hc['p_holm']:.6f}, p_bonf={hc['p_bonferroni']:.6f}")

    hc_mid = haircut_sharpe(0.04, n_trials=5, n_obs=1000,
                            other_pvals=[0.0001, 0.0002, 0.0003, 0.20])
    check("Holm < Bonferroni when the observed trial is not the smallest p",
          hc_mid["p_holm"] < hc_mid["p_bonferroni"] - 1e-12,
          f"p_holm={hc_mid['p_holm']:.6f} < p_bonf={hc_mid['p_bonferroni']:.6f}")

    # cross-check vs DSR: both must reject the noise case and keep the real one
    d_n = deflated_sharpe_ratio(0.02, 1000, 10000)
    h_n = haircut_sharpe(0.02, 10000, 1000)
    d_s = deflated_sharpe_ratio(0.15, 1000, 5)
    h_s = haircut_sharpe(0.15, 5, 1000)
    check("cross-check: DSR and Bonferroni agree on the noise case (both reject)",
          d_n < 0.05 and h_n["p_bonferroni"] > 0.05,
          f"DSR={d_n:.4f}, Bonferroni p_adj={h_n['p_bonferroni']:.4f}, "
          f"haircut={h_n['haircut_bonferroni'] * 100:.1f}%")
    check("cross-check: DSR and Bonferroni agree on the real signal (both keep)",
          d_s > 0.99 and h_s["p_bonferroni"] < 0.05,
          f"DSR={d_s:.4f}, Bonferroni p_adj={h_s['p_bonferroni']:.6f}, "
          f"haircut={h_s['haircut_bonferroni'] * 100:.1f}%")

    check("adjusted Sharpe never exceeds the raw Sharpe",
          hc["sharpe_bonferroni"] <= 0.08 + 1e-12,
          f"raw=0.0800, Bonferroni-adjusted={hc['sharpe_bonferroni']:.4f}")

    # -------------------------------------------------------------------
    section("SUMMARY")
    # -------------------------------------------------------------------
    n_pass = sum(1 for _, ok in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    for name, ok in _RESULTS:
        if not ok:
            print(f"  FAILED: {name}")
    print(f"\n  {n_pass}/{len(_RESULTS)} checks passed, {n_fail} failed.")
    print("  OVERALL: " + ("PASS" if n_fail == 0 else "FAIL"))
    raise SystemExit(0 if n_fail == 0 else 1)
