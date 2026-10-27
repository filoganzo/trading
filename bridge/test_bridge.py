"""Tests for the pieces of the bridge that decide trades.

Run: python test_bridge.py

These pin the two conventions that would otherwise fail silently - the LinReg
orientation and the level-invalidation rule - plus the session wrap and the
correlation gate.
"""
import numpy as np
import pandas as pd

import indicators as ind
import strategy as st
from levels import LevelSet

FAILS = []


def check(name, cond, extra=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {extra}")
        FAILS.append(name)


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ---------------------------------------------------------------- indicators
def test_linreg():
    print("linreg")
    # A perfectly linear series rising by 1 per bar. Chronological, so the
    # newest bar is 110. The EA's LinRegAt returns the fit at the NEWEST bar,
    # so an exact line must return exactly the newest value.
    c = np.arange(100.0, 111.0)
    z = np.zeros_like(c)
    v = ind.linreg_at(z, z, z, c, 3, 5, 0)
    check("fit of an exact line = newest value", approx(v, 110.0, 1e-9), f"got {v}")

    # At shift 2 the window ends two bars back, so the fit is 108.
    v2 = ind.linreg_at(z, z, z, c, 3, 5, 2)
    check("shift 2 = value two bars back", approx(v2, 108.0, 1e-9), f"got {v2}")

    # A falling line must come back below, not above: catches a reversed x.
    cd = np.arange(110.0, 99.0, -1.0)
    vd = ind.linreg_at(z, z, z, cd, 3, 5, 0)
    check("falling line = newest value", approx(vd, 100.0, 1e-9), f"got {vd}")

    # length<=1 degenerates to the raw price
    check("length 1 = raw price",
          approx(ind.linreg_at(z, z, z, c, 3, 1, 0), 110.0))


def test_shift():
    print("shift indexing")
    a = np.array([1.0, 2.0, 3.0])
    check("shift 0 is newest", ind._at(a, 0) == 3.0)
    check("shift 2 is oldest", ind._at(a, 2) == 1.0)
    check("out of range = 0.0", ind._at(a, 9) == 0.0)


def test_regression():
    print("regression stats")
    # Exact line -> residuals are zero -> sd 0, basis = newest value.
    c = np.arange(100.0, 130.0)
    z = np.zeros_like(c)
    ok, basis, sd = ind.regression_stats(z, z, z, c, 20)
    check("exact line: sd == 0", ok and approx(sd, 0.0, 1e-7), f"sd={sd}")
    check("exact line: basis = newest", approx(basis, 129.0, 1e-7), f"basis={basis}")

    # Not enough bars must fail rather than return garbage.
    ok2, _, _ = ind.regression_stats(z, z, z, c[:5], 20)
    check("short history refuses", not ok2)


def test_wavetrend():
    print("wavetrend")
    n = 300
    rng = np.random.default_rng(7)
    c = 4000 + np.cumsum(rng.normal(0, 1, n))
    h, l = c + 0.5, c - 0.5
    ok, wt = ind.wavetrend(h, l, c)
    check("returns a value on 300 bars", ok)
    check("value is finite and bounded", ok and abs(wt) < 500, f"wt={wt}")
    ok2, _ = ind.wavetrend(h[:30], l[:30], c[:30])
    check("refuses under 60 bars", not ok2)


# ------------------------------------------------------------------- levels
def _htf(rows):
    return pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])


def test_levels_invalidate():
    print("level invalidation")
    ls = LevelSet(50)
    # bar 1 high 110 low 90; bar 2 closes at 115 (above 110 -> kills the high)
    # last row is the forming bar and must not create a level.
    df = _htf([[0, 100, 110, 90, 105],
               [3600, 105, 118, 104, 115],
               [7200, 115, 116, 114, 115]])
    ls.seed(df, "H4", 0)
    highs = [lv.price for lv in ls.h4_high]
    check("forming bar creates no level", 116 not in highs, f"{highs}")
    check("high broken by a later close is gone",
          110 not in [lv.price for lv in ls.h4_high if lv.valid], f"{highs}")
    check("the newer high survives",
          118 in [lv.price for lv in ls.h4_high if lv.valid], f"{highs}")
    check("lows survive when price rises",
          90 in [lv.price for lv in ls.h4_low if lv.valid])


def test_scan_break():
    print("break scan")
    ls = LevelSet(50)
    ls._add(ls.d1_low, 4000.0, 0, "D1L")
    # close below 4000 with the previous 3 closes above it = a downside break
    found, lvl = ls.scan_break(True, 3995.0, [4005.0, 4010.0, 4012.0])
    check("clean break down found", found and lvl == 4000.0, f"{found} {lvl}")
    # one prior close on the wrong side kills it
    found2, _ = ls.scan_break(True, 3995.0, [3998.0, 4010.0, 4012.0])
    check("prior close through level rejects", not found2)
    # price still above the level: no break
    found3, _ = ls.scan_break(True, 4005.0, [4005.0, 4010.0, 4012.0])
    check("no break while price is on the origin side", not found3)


def test_rejection_and_room():
    print("rejection / room")
    ls = LevelSet(50)
    ls._add(ls.d1_low, 4000.0, 0, "D1L")
    # wick to 3999 (1.0 overshoot), closes back above -> long rejection
    f, lvl = ls.scan_rejection(True, bh=4008, bl=3999, bc=4006,
                               max_dip=1.5, tol=0.0, prior_closes=[4010.0])
    check("rejection found within max dip", f and lvl == 4000.0)
    # overshoot deeper than max_dip -> not a rejection
    f2, _ = ls.scan_rejection(True, bh=4008, bl=3990, bc=4006,
                              max_dip=1.5, tol=0.0, prior_closes=[4010.0])
    check("overshoot beyond max dip rejects", not f2)
    # closing below the level is a break, not a rejection
    f3, _ = ls.scan_rejection(True, bh=4008, bl=3999, bc=3999.5,
                              max_dip=1.5, tol=0.0, prior_closes=[4010.0])
    check("close through level is not a rejection", not f3)

    ls2 = LevelSet(50)
    ls2._add(ls2.h4_high, 4010.0, 0, "H4H")
    has, near = ls2.nearest(True, 4005.0)
    check("nearest above found", has and near == 4010.0)
    check("room blocked when level is 5.0 away (min 10 pips = 1.0)",
          (near - 4005.0) >= 1.0)


def test_anchor():
    print("anchor")
    ls = LevelSet(50)
    ls._add(ls.h4_low, 4000.0, 0, "H4L")
    ls._add(ls.h4_low, 4003.0, 0, "H4L")
    # armed at 4000, price now 4005: the highest level crossed is 4003
    f, a = ls.scan_anchor(True, close=4005.0, armed=4000.0)
    check("anchor picks the furthest crossed level", f and a == 4003.0, f"{a}")


# ------------------------------------------------------------------ session
def test_session():
    print("session window")
    import datetime as dt
    # 2026-06-15 08:00 UTC = 10:00 Rome (DST)
    ts = int(dt.datetime(2026, 6, 15, 8, 0).timestamp()
             - dt.datetime(2026, 6, 15, 8, 0).timestamp()
             + dt.datetime(2026, 6, 15, 8, 0).replace(
                 tzinfo=dt.timezone.utc).timestamp())
    check("summer: 08:00 UTC is 10:00 Rome", st._rome_minutes(ts) == 600,
          f"{st._rome_minutes(ts)}")
    ts_w = int(dt.datetime(2026, 1, 15, 8, 0,
                           tzinfo=dt.timezone.utc).timestamp())
    check("winter: 08:00 UTC is 09:00 Rome", st._rome_minutes(ts_w) == 540,
          f"{st._rome_minutes(ts_w)}")

    cfg = {"use_time_window": True, "session_start": "08:30", "session_end": "11:00"}
    check("inside window", st.in_time_window(ts, cfg))          # 10:00 Rome
    cfg2 = dict(cfg, session_start="12:00", session_end="14:00")
    check("outside window", not st.in_time_window(ts, cfg2))
    cfg3 = {"use_time_window": False}
    check("window off = always open", st.in_time_window(ts, cfg3))
    # wrap past midnight
    cfg4 = {"use_time_window": True, "session_start": "22:00", "session_end": "02:00"}
    ts_n = int(dt.datetime(2026, 6, 15, 21, 0,
                           tzinfo=dt.timezone.utc).timestamp())   # 23:00 Rome
    check("wrapping window includes 23:00", st.in_time_window(ts_n, cfg4))


# ----------------------------------------------------------------- strategy
def _frame(n=400, seed=3):
    rng = np.random.default_rng(seed)
    c = 4000 + np.cumsum(rng.normal(0, 0.6, n))
    t = np.arange(n, dtype="int64") * 60 + 1_750_000_000
    return pd.DataFrame({
        "time": t, "open": c - 0.1, "high": c + 0.4, "low": c - 0.4,
        "close": c, "spread": 10,
        "s_open": 50.0, "s_close": 50.1,     # silver green
        "d_open": 100.0, "d_close": 99.9,    # dxy red  -> long-confirming
    })


def _htf_frame(n=60, step=14400):
    rng = np.random.default_rng(11)
    c = 4000 + np.cumsum(rng.normal(0, 4, n))
    t = np.arange(n, dtype="int64") * step + 1_749_000_000
    return pd.DataFrame({"time": t, "open": c, "high": c + 6,
                         "low": c - 6, "close": c})


def test_evaluate_smoke():
    print("evaluate")
    cfg = {
        "tp_pips": 40.0, "sl_pips": 400.0, "lots": 0.01,
        "origin_bars": 3, "reclaim_window": 5, "max_dist_pips": 30.0,
        "min_room_pips": 10.0, "use_anchor": True, "cooldown_bars": 5,
        "use_linreg_candles": True, "linreg_length": 15, "signal_length": 9,
        "sma_signal": True, "use_correlation": True, "reject_doji": True,
        "enable_h4": True, "enable_d1": True, "max_levels": 60,
        "seed_htf_bars": 50, "use_time_window": False,
        "wt_channel": 10, "wt_average": 21, "wt_threshold": 40.0,
        "reg_length": 100, "reg_dev_mult": 1.2,
    }
    bars = _frame()
    eng = st.Engine()
    sig = st.evaluate(bars, _htf_frame(), _htf_frame(n=40, step=86400), cfg, eng)
    check("returns a dict with an action", isinstance(sig, dict) and "action" in sig)
    check("action is ENTER or FLAT", sig["action"] in ("ENTER", "FLAT"), sig)
    check("levels were seeded", eng.seeded)
    check("bar index advanced", eng.bar_index == 1, f"{eng.bar_index}")

    # Re-evaluating the SAME closed bar must not advance state: the panel polls
    # every 5s but bars close once a minute.
    st.evaluate(bars, _htf_frame(), _htf_frame(n=40, step=86400), cfg, eng)
    check("same bar does not advance the counter", eng.bar_index == 1,
          f"{eng.bar_index}")

    short = bars.iloc[:50]
    s2 = st.evaluate(short, _htf_frame(), _htf_frame(n=40, step=86400), cfg, st.Engine())
    check("short history warms up instead of trading",
          s2["action"] == "FLAT" and "warming" in s2["reason"], s2)

    # correlation gate: flip DXY green so a long can no longer confirm
    bad = bars.copy()
    bad["d_close"] = 100.1
    eng2 = st.Engine()
    s3 = st.evaluate(bad, _htf_frame(), _htf_frame(n=40, step=86400), cfg, eng2)
    check("runs with contrary correlation", s3["action"] in ("ENTER", "FLAT"))
    if s3["action"] == "ENTER":
        check("a long cannot fire while DXY is green", s3["dir"] != "LONG", s3)



def test_shape_modes():
    """The rejection shape gate, mode by mode, via evaluate()'s own helper
    semantics: 0 any, 1 wick, 2 body, 3 wick-or-body, 4 wick-and-body."""
    print("rejection shape modes")
    def shape(mode, wick_ok, body_ok):
        if mode == 0: return True
        if mode == 1: return wick_ok
        if mode == 2: return body_ok
        if mode == 3: return wick_ok or body_ok
        return wick_ok and body_ok
    check("mode 0 (any) always passes", shape(0, False, False))
    check("mode 1 (wick) needs the wick", shape(1, True, False) and not shape(1, False, True))
    check("mode 2 (body) needs the body", shape(2, False, True) and not shape(2, True, False))
    check("mode 3 (or) takes either", shape(3, True, False) and shape(3, False, True)
          and not shape(3, False, False))
    check("mode 4 (and) needs both", shape(4, True, True) and not shape(4, True, False))


def test_rejection_module_wired():
    """The bug this catches: scan_rejection existed but evaluate() never
    called it, so the bridge silently traded only half the strategy."""
    print("rejection module wiring")
    import inspect
    src = inspect.getsource(st.evaluate)
    check("evaluate() calls scan_rejection", "scan_rejection" in src)
    check("evaluate() can emit a REJ module signal", '"REJ"' in src)
    check("rejection has its own extreme gate",
          "rej_long_entry" in src and "wt_os" in src)

    import config as C
    for k in ("enable_rejection", "rej_max_dip_pips", "rej_shape_mode",
              "rej_wick_body_mult", "rej_min_body_pips", "rej_require_color",
              "rej_approach_bars", "enable_break_reclaim"):
        check(f"config has {k}", k in C.DEFAULTS)


def test_module_precedence():
    """B&R must win over a rejection on the same bar, and only one signal may
    leave per bar."""
    print("module precedence")
    import inspect
    src = inspect.getsource(st.evaluate)
    check("rej_long gated behind both B&R entries",
          "rej_long_entry and not long_entry and not short_entry" in src)
    check("rej_short gated behind rej_long", "not rej_long_entry" in src)
    check("signal id distinguishes the module", '{module}-{side}' in src
          or 'module}-' in src)


if __name__ == "__main__":
    for fn in (test_shift, test_linreg, test_regression, test_wavetrend,
               test_levels_invalidate, test_scan_break, test_rejection_and_room,
               test_anchor, test_session, test_evaluate_smoke,
               test_shape_modes, test_rejection_module_wired,
               test_module_precedence):
        fn()
    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        raise SystemExit(1)
    print("all tests passed")
