"""
Stage B: management sweep with a matched random-entry control.

Runs on the CORRECTED engine (targets lagged one bar, gap fills at the open,
trims blocked on stop bars). Every management spec is evaluated twice:

  real     the actual setup entries
  control  random entries drawn from bars where the HTF gate is ON for the same
           side but the base trigger did NOT fire

The control is the whole point. Gold rose 48% across this sample, so a long-only
scheme with a wide stop and a long hold makes money whether or not the entry
carries information. Subtracting the control removes both the drift and the
management scheme's own bias, leaving only what the TRIGGER contributed.

The null distribution for the percentile rank is a bootstrap of n_signal-sized
means drawn from the pooled control trades, which is equivalent to re-drawing
control samples but far cheaper.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from mtf_runner import (BaseParams, base_indicators, base_signals, build_paths,
                        POINT, CONTRACT, DEFAULT_POINT, DEFAULT_CONTRACT)
from mtf import htf_gate, HtfParams, ALIGN_CLOSED
from management import Mgmt, simulate_managed, REASON_NAMES


def run_on_indices(d, idx, side, mg: Mgmt, symbol: str,
                   commission_per_lot_rt: float = 7.0,
                   extra_slippage_points: float = 0.0) -> dict:
    """Apply one management spec to an explicit set of entry bars."""
    if len(idx) == 0:
        return {}
    point = POINT.get(symbol, DEFAULT_POINT)
    contract = CONTRACT.get(symbol, DEFAULT_CONTRACT)
    spread = d["spread"].to_numpy(float) if "spread" in d else np.zeros(len(d))
    atr_a = d["atr"].to_numpy(float)
    hz = int(mg.max_bars)

    (H, L, C, OP, MID, UP, LO, eo, sw_lo, sw_hi,
     mid0, up0, lo0) = build_paths(d, idx, hz, swing_lookback=mg.swing_lookback)
    cost = spread[np.minimum(idx + 1, len(d) - 1)] + extra_slippage_points
    entry = eo + side * cost * point
    a = atr_a[idx]
    swing = sw_lo if side > 0 else sw_hi
    res = simulate_managed(H, L, C, entry, a, side, mg,
                           MID=MID, UPPER=UP, LOWER=LO, SWING=swing, O=OP,
                           MID0=mid0, UPPER0=up0, LOWER0=lo0)
    comm_r = commission_per_lot_rt / (res["r_unit"] * contract)
    r = res["r"] - comm_r
    ok = np.isfinite(r)
    return {"r": r[ok], "bars": res["bars"][ok], "idx": idx[ok],
            "reason": np.array([REASON_NAMES[x] for x in res["reason"]])[ok]}


def mgmt_grid_stage1() -> list[Mgmt]:
    """Coverage of the dimensions that matter first: horizon x target x stop."""
    out = []
    stops = [("atr", 1.0), ("atr", 1.5), ("atr", 2.0), ("atr", 3.0), ("none", 2.0)]
    targets = [("none", 0.0), ("atr", 2.0), ("atr", 3.0), ("atr", 5.0),
               ("atr", 8.0), ("mid", 0.0), ("opposite", 0.0)]
    for mb in (24, 48, 96, 144):
        for sm, sl in stops:
            for tm, tp in targets:
                if sm == "none" and tm == "none":
                    continue          # nothing would ever close it but time
                out.append(Mgmt(stop_mode=sm, sl_atr=sl, tp_mode=tm, tp_atr=tp,
                                max_bars=mb))
    return out


def mgmt_grid_stage2(best_mb: int, best_sl: float, best_tm: str,
                     best_tp: float) -> list[Mgmt]:
    """Layer trims / breakeven / trailing onto the best region from stage 1."""
    out = []
    for trims in ((), ((1.0, 0.5),), ((1.0, 0.34), (2.0, 0.33))):
        for be in (0.0, 1.0):
            for tr, ta in (("none", 0.0), ("chandelier", 1.5), ("chandelier", 3.0)):
                for mb in (best_mb, min(288, best_mb * 2)):
                    out.append(Mgmt(stop_mode="atr", sl_atr=best_sl,
                                    tp_mode=best_tm, tp_atr=best_tp,
                                    trims=trims, be_at_r=be, trail_mode=tr,
                                    trail_atr=ta, trail_start_r=1.0,
                                    max_bars=mb))
    return out


def sweep(symbol="XAUUSD", base_tf="M5", htf_tf="M30", side=+1,
          entry_grid=None, mgmt_specs=None, n_control=4000,
          n_boot=2000, seed=17, verbose=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entry_grid = entry_grid or list(itertools.product(
        (100, 200),        # base chan_len
        (3, 6),            # min_red_run
        (50, 100),         # htf chan_len
        (50.0, 60.0),      # htf wt_level
    ))
    mgmt_specs = mgmt_specs or mgmt_grid_stage1()
    t0 = time.time()
    rows = []

    for ei, (bcl, mrr, hcl, hwl) in enumerate(entry_grid):
        bp = BaseParams(chan_len=bcl, min_red_run=mrr, touch_src="raw",
                        touch_lookback=3, wt_level=40.0, wt_cross_lookback=3)
        d = base_indicators(symbol, base_tf, bp)
        sig = base_signals(d, bp)
        hp = HtfParams(tf=htf_tf, align=ALIGN_CLOSED, chan_len=hcl,
                       wt_level=hwl, band_src="linreg")
        g = htf_gate(d, hp)

        trig = (sig["long_sig"] if side > 0 else sig["short_sig"]).to_numpy(bool)
        gate = (g["htf_long_ok"] if side > 0 else g["htf_short_ok"]).to_numpy(bool)
        valid = (np.isfinite(d["atr"].to_numpy(float))
                 & np.isfinite(d["lrc_lower"].to_numpy(float))
                 & np.isfinite(d["wt2"].to_numpy(float)))
        valid[-300:] = False                     # need room for the longest path

        real_idx = np.where(trig & gate & valid)[0]
        pool = np.where((~trig) & gate & valid)[0]
        if len(real_idx) < 40 or len(pool) < 500:
            continue
        ctrl_idx = np.sort(rng.choice(pool, size=min(n_control, len(pool)),
                                      replace=False))

        for mg in mgmt_specs:
            rr = run_on_indices(d, real_idx, side, mg, symbol)
            cc = run_on_indices(d, ctrl_idx, side, mg, symbol)
            if not rr or not cc or len(rr["r"]) < 30 or len(cc["r"]) < 200:
                continue
            r, c = rr["r"], cc["r"]
            n = len(r)
            boot = rng.choice(c, size=(n_boot, n), replace=True).mean(axis=1)
            exp_r = float(r.mean())
            delta = exp_r - float(c.mean())
            gp, gl = r[r > 0].sum(), -r[r <= 0].sum()
            eq = np.cumsum(r)
            dd = eq - np.maximum.accumulate(np.r_[0.0, eq])[1:]
            rows.append({
                "base_chan": bcl, "min_run": mrr, "htf_chan": hcl,
                "htf_wt": hwl,
                "max_bars": mg.max_bars, "stop_mode": mg.stop_mode,
                "sl_atr": mg.sl_atr, "tp_mode": mg.tp_mode, "tp_atr": mg.tp_atr,
                "trims": len(mg.trims), "be": mg.be_at_r,
                "trail": mg.trail_mode, "trail_atr": mg.trail_atr,
                "n": n, "expectancy_r": exp_r,
                "total_r": float(r.sum()),
                "control_mean": float(c.mean()), "delta": delta,
                "pctile_vs_control": float((boot < exp_r).mean()),
                "win_rate": float((r > 0).mean()),
                "profit_factor": float(gp / gl) if gl > 0 else np.inf,
                "max_dd_r": float(dd.min()),
                "avg_bars": float(rr["bars"].mean()),
                "label": mg.label(),
            })
        if verbose:
            print(f"  entry {ei+1}/{len(entry_grid)} "
                  f"(chan{bcl}/run{mrr}/htf{hcl}/wt{hwl:.0f}) "
                  f"n_real={len(real_idx)} rows={len(rows)} {time.time()-t0:.0f}s",
                  flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="XAUUSD")
    ap.add_argument("--base", default="M5")
    ap.add_argument("--htf", default="M30")
    ap.add_argument("--side", type=int, default=1)
    ap.add_argument("--out", default="sweeps/mgmt_XAUUSD_M5_M30_long.parquet")
    a = ap.parse_args()
    print(f"Stage B management sweep: {a.symbol} {a.base}/{a.htf} "
          f"side={'long' if a.side > 0 else 'short'}")
    df = sweep(a.symbol, a.base, a.htf, a.side)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_parquet(a.out)
    print(f"\n{len(df)} rows -> {a.out}")
