"""Headless bridge runner - the panel's worker loop without the GUI.

    python run_headless.py            # watch only, never arms (safe)
    python run_headless.py --minutes 30

It writes the same bridge_settings.json / bridge_signal.json the panel writes,
but "enabled" stays false, so the EA sees a DISARMED bridge and will not
trade. Use it to confirm the data, the levels and the signal engine are
healthy before putting the panel in front of a real account.
"""
from __future__ import annotations

import argparse
import json
import time
import datetime as dt

import config as C
import feed
import strategy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--write", action="store_true",
                    help="also write the signal files (still DISARMED)")
    args = ap.parse_args()

    cfg = dict(C.DEFAULTS)
    cfg["enabled"] = False              # headless never arms
    eng = strategy.Engine()

    print(f"headless bridge  poll={args.poll}s  for {args.minutes} min  "
          f"write={'yes' if args.write else 'no'}  ARMED=never")
    print(f"gold {C.OANDA_GOLD}@OANDA  silver {C.OANDA_SILVER}@OANDA  "
          f"dxy {C.IC_DXY}@IC  exec {C.VANTAGE_GOLD}@Vantage")
    print()

    end = time.time() + args.minutes * 60
    h4 = d1 = None
    htf_at = 0.0
    last_reason = None

    while time.time() < end:
        t0 = time.time()
        try:
            frames = feed.get_bars(count=max(600, cfg["seed_htf_bars"]))
            bars = feed.align(frames)
            if h4 is None or time.time() - htf_at > 300:
                h4, d1 = feed.get_htf(count=cfg["seed_htf_bars"])
                htf_at = time.time()
            sig = strategy.evaluate(bars, h4, d1, cfg, eng)
        except Exception as exc:
            print(f"{dt.datetime.now():%H:%M:%S}  ERROR {exc}")
            time.sleep(args.poll)
            continue

        d = sig.get("diag") or {}
        lv = d.get("levels", {})
        nlv = sum(v[0] for v in lv.values()) if lv else 0
        band = "UPPER" if d.get("at_upper") else ("LOWER" if d.get("at_lower") else "mid")
        arm = ("ARMED-L" if d.get("armed_long")
               else "ARMED-S" if d.get("armed_short") else "-")
        reason = sig.get("reason", "")

        line = (f"{dt.datetime.now():%H:%M:%S}  {sig['action']:5s} "
                f"px {bars.close.iloc[-1]:8.2f}  wt {d.get('wt', 0):>6}  "
                f"z {d.get('z_close', 0):>5}  {band:5s}  "
                f"t={'Y' if d.get('touch') else 'n'}  "
                f"XAG {d.get('silver', '?')} DXY {d.get('dxy', '?')}  "
                f"{arm:7s}  lv {nlv:3d}  {reason}")
        # print every entry, but a repeated FLAT reason only once
        if sig["action"] == "ENTER" or reason != last_reason:
            print(line)
        last_reason = reason

        if args.write:
            C.COMMON_FILES.mkdir(parents=True, exist_ok=True)
            payload = dict(cfg, _written_utc=time.time())
            C.SETTINGS_FILE.write_text(json.dumps(payload, indent=2))
            out = dict(sig)
            out.update(ts=time.time(), ttl=cfg["signal_ttl_seconds"],
                       armed=False, symbol=C.VANTAGE_GOLD, magic=cfg["magic"],
                       action="FLAT")      # forced: headless must never enter
            out["id"] = f"{int(out['ts'])}-headless"
            C.SIGNAL_FILE.write_text(json.dumps(out, indent=2, default=str))

        time.sleep(max(0.5, args.poll - (time.time() - t0)))

    print("\ndone")


if __name__ == "__main__":
    main()
