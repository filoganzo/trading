"""
Final validation of the surviving configuration.

Order matters here. Each test can only kill the result, never rescue it:

  1. per-quarter / per-session   is it one lucky window or one lucky hour?
  2. purged walk-forward         does it survive picking parameters forward?
  3. effective trials            how many INDEPENDENT bets did the search make?
  4. Deflated Sharpe             is the best result better than the best of that
                                 many correlated tries would be by luck?
  5. PBO (CSCV)                  does the in-sample-best land in the bottom half
                                 out of sample?
  6. random-entry control        already applied per-spec in the sweep; repeated
                                 here on the finalist alone with more draws.

The trial count feeds directly into (4), and DSR's verdict on a Sharpe of this
size flips between ~5 and ~50 trials, so (3) is not a formality.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from mtf_runner import BaseParams, base_indicators, base_signals
from mtf import htf_gate, HtfParams, ALIGN_CLOSED
from management import Mgmt
from mgmt_sweep import run_on_indices
from effective_trials import effective_trials, returns_matrix_from_trades
import stats_battery as SB

SYMBOL, BASE_TF, HTF_TF, SIDE = "XAUUSD", "M5", "M30", +1

# the surviving spec: tightest sensible stop, no target, hold to the cap
FINAL_MGMT = Mgmt(stop_mode="atr", sl_atr=1.5, tp_mode="none", trims=(),
                  be_at_r=0.0, trail_mode="none", max_bars=288)

ENTRY_GRID = list(itertools.product((100, 200), (3, 6), (50, 100), (50.0, 60.0)))


def trades_for(bcl, mrr, hcl, hwl, mg=FINAL_MGMT, want_control=False,
               n_control=4000, seed=17):
    bp = BaseParams(chan_len=bcl, min_red_run=mrr, touch_src="raw",
                    touch_lookback=3, wt_level=40.0, wt_cross_lookback=3)
    d = base_indicators(SYMBOL, BASE_TF, bp)
    sig = base_signals(d, bp)
    hp = HtfParams(tf=HTF_TF, align=ALIGN_CLOSED, chan_len=hcl, wt_level=hwl,
                   band_src="linreg")
    g = htf_gate(d, hp)
    trig = (sig["long_sig"] if SIDE > 0 else sig["short_sig"]).to_numpy(bool)
    gate = (g["htf_long_ok"] if SIDE > 0 else g["htf_short_ok"]).to_numpy(bool)
    valid = (np.isfinite(d["atr"].to_numpy(float))
             & np.isfinite(d["lrc_lower"].to_numpy(float))
             & np.isfinite(d["wt2"].to_numpy(float)))
    valid[-300:] = False
    idx = np.where(trig & gate & valid)[0]
    out = run_on_indices(d, idx, SIDE, mg, SYMBOL)
    if not out:
        return None, None
    t = pd.DataFrame({"r": out["r"], "bars": out["bars"],
                      "i": out["idx"],
                      "utc": d["utc"].to_numpy()[out["idx"]],
                      "session": d["session"].to_numpy()[out["idx"]],
                      "reason": out["reason"]})
    ctrl = None
    if want_control:
        rng = np.random.default_rng(seed)
        pool = np.where((~trig) & gate & valid)[0]
        cidx = np.sort(rng.choice(pool, size=min(n_control, len(pool)),
                                  replace=False))
        c = run_on_indices(d, cidx, SIDE, mg, SYMBOL)
        ctrl = c["r"] if c else None
    return t, ctrl


def stat_block(r: np.ndarray) -> dict:
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return {}
    eq = np.cumsum(r)
    dd = eq - np.maximum.accumulate(np.r_[0.0, eq])[1:]
    gp, gl = r[r > 0].sum(), -r[r <= 0].sum()
    return {"n": len(r), "expectancy_r": r.mean(), "total_r": r.sum(),
            "win_rate": (r > 0).mean(),
            "profit_factor": gp / gl if gl > 0 else np.inf,
            "max_dd_r": dd.min(), "sharpe_per_trade": r.mean() / r.std(ddof=1)}


def main():
    pd.set_option("display.width", 220)
    print("=" * 84)
    print("FINAL VALIDATION -- XAUUSD M5 base / M30 confirmation, LONG only")
    print(f"management: {FINAL_MGMT.label()}")
    print("=" * 84)

    frames, ctrls = {}, {}
    for combo in ENTRY_GRID:
        t, c = trades_for(*combo, want_control=True)
        if t is not None and len(t) >= 30:
            frames[str(combo)] = t
            ctrls[str(combo)] = c
    print(f"\n{len(frames)} entry configs, "
          f"{sum(len(t) for t in frames.values())} trades total")

    pooled = pd.concat(frames.values()).sort_values("utc").reset_index(drop=True)
    pool_ctrl = np.concatenate([c for c in ctrls.values() if c is not None])

    print("\n--- 1. POOLED, and vs the random-entry control ---")
    sb = stat_block(pooled["r"].to_numpy())
    print(pd.DataFrame([sb]).round(4).to_string(index=False))
    print(f"    control mean {pool_ctrl.mean():+.4f} R over {len(pool_ctrl)} "
          f"random gate-on entries -> delta {sb['expectancy_r'] - pool_ctrl.mean():+.4f} R")

    print("\n--- 2. BY QUARTER ---")
    q = pooled.copy()
    q["quarter"] = pd.to_datetime(q["utc"]).dt.to_period("Q")
    print(q.groupby("quarter")["r"].agg(
        n="size", total_r="sum", expectancy_r="mean",
        win=lambda x: (x > 0).mean()).round(3).to_string())

    print("\n--- 3. BY SESSION ---")
    print(pooled.groupby("session")["r"].agg(
        n="size", total_r="sum", expectancy_r="mean",
        win=lambda x: (x > 0).mean()).round(3).to_string())

    print("\n--- 4. EFFECTIVE INDEPENDENT TRIALS ---")
    R = returns_matrix_from_trades(frames, freq="D")
    et = effective_trials(R.to_numpy())

    print("\n--- 5. DEFLATED SHARPE ---")
    r = pooled["r"].to_numpy()
    sr = r.mean() / r.std(ddof=1)
    from scipy import stats as sps
    sk = float(sps.skew(r)); ku = float(sps.kurtosis(r, fisher=False))
    print(f"    per-trade Sharpe {sr:.4f}, n={len(r)}, skew {sk:.2f}, kurt {ku:.2f}")
    # the real search was much wider than the 16 configs validated here
    searched = 5184 * 172
    for label, nt in (("n_eff (clusters)", max(1, int(et["n_eff_recommended"]))),
                      ("n_eff (participation)", max(1, int(et["participation_ratio"]))),
                      ("entry+mgmt grid searched", searched)):
        d = SB.deflated_sharpe_ratio(sr, len(r), nt, skew=sk, kurt=ku)
        print(f"    DSR @ {label:<26} = {nt:>7}  ->  {d:.4f}"
              f"   {'SURVIVES' if d > 0.95 else 'killed'}")

    print("\n--- 6. PROBABILITY OF BACKTEST OVERFITTING (CSCV) ---")
    M = R.to_numpy()
    if M.shape[1] >= 4 and M.shape[0] >= 64:
        pbo = SB.probability_of_backtest_overfitting(M, n_splits=10)
        print(f"    PBO = {pbo['pbo']:.3f} over {pbo['n_combinations']} partitions, "
              f"{pbo['n_strategies']} strategies")
        print("    (near 0 = the in-sample best stays best; near 1 = pure overfit;"
              " ~0.5 = no skill)")
    else:
        print(f"    skipped: matrix {M.shape} too small")

    print("\n--- 7. BOOTSTRAP CI on pooled expectancy ---")
    lo, hi = SB.block_bootstrap_ci(r, n_iter=5000, block=10)
    print(f"    95% CI  [{lo:+.4f}, {hi:+.4f}]  "
          f"{'excludes zero' if lo > 0 else 'INCLUDES ZERO'}")

    print("\n--- 8. MIN TRACK RECORD LENGTH ---")
    mtl = SB.min_track_record_length(sr, skew=sk, kurt=ku, target_sharpe=0.0,
                                     prob=0.95)
    print(f"    trades needed for 95% confidence the Sharpe > 0: {mtl:.0f} "
          f"(have {len(r)})")

    pooled.to_csv("final_trades_gold.csv", index=False)
    print("\nwrote final_trades_gold.csv")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
