import numpy as np, pandas as pd, time, warnings; warnings.filterwarnings("ignore")
from stocks_test import run_did
from mtf_runner import BaseParams
from mtf import HtfParams, ALIGN_CLOSED
pd.set_option("display.width", 210)
t0 = time.time(); rows = []
PAIRS = ((("H1","D1"),(6,12,24)), (("H1","H4"),(6,12,24)), (("D1","W1"),(3,5,10)))
for (base_tf, htf_tf), hz in PAIRS:
    for cl in (50, 100, 200):
        for mrr in (2, 3):
            for wl in (30.0, 40.0):
                for hcl in (50, 100):
                    for hwl in (40.0, 50.0):
                        bp = BaseParams(chan_len=cl, min_red_run=mrr, touch_src="raw",
                                        touch_lookback=3, wt_level=wl, wt_cross_lookback=3)
                        hp = HtfParams(tf=htf_tf, align=ALIGN_CLOSED, chan_len=hcl,
                                       wt_level=hwl, band_src="linreg")
                        per, summ = run_did(base_tf, htf_tf, +1, hz, bp, hp, verbose=False)
                        if len(summ) == 0:
                            continue
                        for _, s in summ.iterrows():
                            rows.append({"base": base_tf, "htf": htf_tf, "chan": cl,
                                         "run": mrr, "wt": wl, "hchan": hcl,
                                         "hwt": hwl, **s.to_dict()})
        print(f"  {base_tf}/{htf_tf} chan={cl} rows={len(rows)} {time.time()-t0:.0f}s", flush=True)
r = pd.DataFrame(rows)
r.to_parquet("sweeps/stocks_did.parquet")
print(f"\ntotal rows {len(r)}")
