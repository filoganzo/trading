import numpy as np, pandas as pd, warnings, time; warnings.filterwarnings("ignore")
from scipy import stats as sps
from stocks_test import run_mgmt
from mtf_runner import BaseParams
from mtf import HtfParams, ALIGN_CLOSED
from management import Mgmt
pd.set_option("display.width", 210)

BP = BaseParams(chan_len=50, min_red_run=3, touch_src="raw", touch_lookback=3,
                wt_level=30.0, wt_cross_lookback=3)
HP = HtfParams(tf="D1", align=ALIGN_CLOSED, chan_len=50, wt_level=40.0,
               band_src="linreg")
SPECS = []
for mb in (12, 24, 48):
    for sl in (1.0, 1.5, 2.0, 3.0):
        for tm, tp in (("none", 0.0), ("atr", 2.0), ("atr", 3.0), ("atr", 5.0),
                       ("mid", 0.0)):
            SPECS.append(Mgmt(stop_mode="atr", sl_atr=sl, tp_mode=tm,
                              tp_atr=tp, max_bars=mb))
rows = []
t0 = time.time()
for mg in SPECS:
    real, ctrl = run_mgmt("H1", "D1", +1, mg, BP, HP, n_control_per_sym=500)
    if len(real) < 100 or len(ctrl) < 500:
        continue
    r = real["r"].to_numpy(); c = ctrl
    boot = np.random.default_rng(3).choice(c, size=(3000, len(r)), replace=True).mean(axis=1)
    # symbol-level sign test on the per-symbol mean
    bysym = real.groupby("symbol")["r"].mean()
    npos = int((bysym > 0).sum())
    rows.append({"max_bars": mg.max_bars, "sl_atr": mg.sl_atr,
                 "tp_mode": mg.tp_mode, "tp_atr": mg.tp_atr,
                 "n": len(r), "syms": real["symbol"].nunique(),
                 "exp_r": r.mean(), "ctrl": c.mean(), "delta": r.mean() - c.mean(),
                 "pct_vs_ctrl": float((boot < r.mean()).mean()),
                 "win": (r > 0).mean(),
                 "pf": r[r > 0].sum() / max(-r[r <= 0].sum(), 1e-9),
                 "syms_pos": npos,
                 "sign_p": sps.binomtest(npos, len(bysym), 0.5).pvalue})
    print(f"  {mg.label()[:52]:<52} n={len(r):>5} delta={rows[-1]['delta']:+.4f} "
          f"[{time.time()-t0:.0f}s]", flush=True)
df = pd.DataFrame(rows)
df.to_parquet("sweeps/stocks_mgmt.parquet")
print("\n=== TOP 12 by delta ===")
print(df.nlargest(12, "delta").round(4).to_string(index=False))
print("\n=== marginal effects ===")
for dim in ("max_bars", "sl_atr", "tp_mode"):
    print(f"\n-- {dim} --")
    print(df.groupby(dim).agg(delta=("delta","mean"), exp_r=("exp_r","mean"),
                              ctrl=("ctrl","mean"), win=("win","mean"),
                              syms_pos=("syms_pos","mean")).round(4).to_string())
