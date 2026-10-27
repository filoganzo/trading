"""
LinReg-Reversal strategy: signals, path-based exit simulation, statistics.

Strategy (long; short is the exact mirror)
-----------------------------------------
At the close of an M5 bar, go long iff

  1. the LinReg candle is GREEN  (bopen < bclose),
  2. it is preceded by a run of >= ``min_red_run`` RED LinReg candles,
  3. price has touched or is sitting below the LOWER LinReg-Channel band
     (mult ``chan_mult``, default 1.3) -- a wick / max-excursion touch counts,
     and the touch may occur on the signal bar or anywhere inside the red run,
  4. WaveTrend wt1 has crossed ABOVE wt2 while wt1 <= -``wt_level`` (default 40),
     within ``wt_cross_lookback`` bars of the signal bar.

Optionally the same signal must be present on XAGUSD for the gold trade to be
taken (``use_silver``).

Execution model
---------------
- The signal is *evaluated* on the closed bar i and *filled* at the open of bar
  i+1. Filling at bar i's own close would be look-ahead: an EA cannot transact
  at a closing print. (``entry_mode='signal_close'`` exists for comparison only.)
- The full spread is charged once, at entry, and SL/TP are then evaluated
  against raw (bid) highs and lows. Round-trip cost = 1 x spread, which is the
  correct total for a bid-quoted bar series.
- When a bar contains BOTH the stop and the target, the stop is assumed to fill
  first. This is deliberately pessimistic; the M1 resolver measures how much
  that assumption costs using true intrabar paths.
- Results are reported primarily in R (multiples of the initial stop distance),
  which keeps the 17-month sample comparable across a period in which gold ran
  from roughly $3,000 to $4,400.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from indicators import linreg_candles, lrc_channel, wavetrend, atr

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "linreg_rev")

SERVER_UTC_OFFSET = 3          # ICMarketsEU server clock = UTC+3

CONTRACT = {"XAUUSD": 100.0, "XAGUSD": 1000.0}
POINT = {"XAUUSD": 0.01, "XAGUSD": 0.001}


# --------------------------------------------------------------------- params

@dataclass
class Params:
    # --- indicator geometry -------------------------------------------------
    cand_len: int = 11             # Humble LinReg Candles length
    chan_len: int = 100            # LinReg Channel length
    chan_mult: float = 1.3         # band multiplier (user-specified 1.3)
    chan_source: str = "close"     # 'close' (Pine default) | 'bclose'
    wt_n1: int = 10
    wt_n2: int = 21
    atr_period: int = 14

    # --- entry conditions ---------------------------------------------------
    min_red_run: int = 3           # "a long series of previously red candles"
    touch_src: str = "linreg"      # 'linreg' (blow/bhigh) | 'raw' (low/high)
    touch_lookback: int = 0        # 0 => touch must fall inside the red run
    wt_level: float = 40.0         # cross must occur at/beyond this level
    wt_cross_lookback: int = 1     # 1 => cross must be on the signal bar
    require_counter_slope: bool = False   # channel must lean against the trade
    min_abs_pearson: float = 0.0   # 0 => off

    # --- silver confirmation ------------------------------------------------
    use_silver: bool = False
    silver_mode: str = "signal"    # 'signal' (full mirror) | 'zone' (band+colour)
    silver_lookback: int = 3

    # --- exits --------------------------------------------------------------
    sl_atr: float = 1.5
    tp_atr: float = 2.0
    max_bars: int = 60             # time stop, in M5 bars
    be_at_r: float = 0.0           # move stop to breakeven after +xR (0=off)
    trail_atr: float = 0.0         # ATR trailing stop (0=off)

    # --- scope --------------------------------------------------------------
    # 'reversal' = the strategy as originally specified: the lower-band setup
    #              buys and the upper-band setup sells.
    # 'momentum' = the same setups traded the other way round. The excess-drift
    #              scan says this is the direction gold actually travels: the
    #              upper-band "exhaustion" setup precedes continuation, not
    #              reversal, over this sample.
    mode: str = "reversal"
    allow_long: bool = True
    allow_short: bool = True
    sessions: tuple = ()           # () => all
    entry_mode: str = "next_open"  # 'next_open' | 'signal_close'
    cooldown_bars: int = 0         # min bars between entries on the same side

    # --- costs --------------------------------------------------------------
    commission_per_lot_rt: float = 7.0   # $ per 1.0 lot round turn
    extra_slippage_points: float = 0.0

    def key(self) -> tuple:
        return tuple(sorted((k, tuple(v) if isinstance(v, list) else v)
                            for k, v in asdict(self).items()))


SESSIONS = {                      # UTC hour ranges, [start, end)
    "asian":   (0, 7),
    "london":  (7, 12),
    "overlap": (12, 16),
    "ny_pm":   (16, 21),
    "off":     (21, 24),
}


def session_of(utc_hour: np.ndarray) -> np.ndarray:
    out = np.full(len(utc_hour), "off", dtype=object)
    for name, (a, b) in SESSIONS.items():
        out[(utc_hour >= a) & (utc_hour < b)] = name
    return out


# ----------------------------------------------------------------------- data

_CACHE: dict = {}


def load(symbol: str, tf: str = "M5") -> pd.DataFrame:
    k = (symbol, tf)
    if k not in _CACHE:
        df = pd.read_parquet(os.path.join(DATA_DIR, f"{symbol}_{tf}.parquet"))
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["time"], unit="s")          # server clock
        df["utc"] = df["dt"] - pd.Timedelta(hours=SERVER_UTC_OFFSET)
        df["utc_hour"] = df["utc"].dt.hour
        df["session"] = session_of(df["utc_hour"].to_numpy())
        df["dow"] = df["utc"].dt.dayofweek
        _CACHE[k] = df
    return _CACHE[k].copy()


# ----------------------------------------------------------------- indicators

_IND_CACHE: dict = {}


def add_indicators(df: pd.DataFrame, p: Params, symbol: str) -> pd.DataFrame:
    k = (symbol, len(df), int(df["time"].iloc[0]), p.cand_len, p.chan_len,
         p.chan_mult, p.chan_source, p.wt_n1, p.wt_n2, p.atr_period)
    if k in _IND_CACHE:
        return _IND_CACHE[k]
    out = df.copy()
    out = pd.concat([out, linreg_candles(out, length=p.cand_len)], axis=1)
    src = "close" if p.chan_source == "close" else "bclose"
    out = pd.concat([out, lrc_channel(out, length=p.chan_len, mult=p.chan_mult,
                                      source=src)], axis=1)
    out = pd.concat([out, wavetrend(out, p.wt_n1, p.wt_n2)], axis=1)
    out["atr"] = atr(out, p.atr_period)
    if len(_IND_CACHE) > 16:
        _IND_CACHE.clear()
    _IND_CACHE[k] = out
    return out


# -------------------------------------------------------------------- helpers

BIG = np.int64(10 ** 9)


def _bars_since(flag: np.ndarray) -> np.ndarray:
    """
    Bars since flag was last True (0 = True on this bar; BIG = not yet true).

    Vectorised via a running maximum of the "index where true" -- the scan
    evaluates thousands of parameter sets over 100k bars, so the Python-loop
    version dominated total runtime.
    """
    flag = np.asarray(flag, dtype=bool)
    n = len(flag)
    ar = np.arange(n, dtype=np.int64)
    last = np.maximum.accumulate(np.where(flag, ar, np.int64(-1)))
    out = ar - last
    return np.where(last < 0, BIG, out)


def _run_length_of_false(flag: np.ndarray) -> np.ndarray:
    """
    For each i, how many consecutive bars immediately BEFORE i had flag=False.
    Used to count the red run preceding a green candle.

    Equals ``_bars_since(flag)`` shifted forward one bar: if the previous bar was
    True the run is 0, otherwise it is the gap back to the last True bar.
    """
    since = _bars_since(flag)
    out = np.empty(len(flag), dtype=np.int64)
    out[0] = 0
    out[1:] = np.minimum(since[:-1], BIG)
    return out


# -------------------------------------------------------------------- signals

def make_signals(d: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Boolean long_sig / short_sig plus the diagnostic components."""
    green = d["lrc_green"].to_numpy(bool)
    red = ~green

    # (2) preceding run of opposite-colour candles
    red_run = _run_length_of_false(green)      # reds immediately before a green
    green_run = _run_length_of_false(red)      # greens immediately before a red

    # (3) band touch -- a wick / max-excursion touch counts
    if p.touch_src == "linreg":
        lo_src, hi_src = d["blow"].to_numpy(float), d["bhigh"].to_numpy(float)
    else:
        lo_src, hi_src = d["low"].to_numpy(float), d["high"].to_numpy(float)
    lower, upper = d["lrc_lower"].to_numpy(float), d["lrc_upper"].to_numpy(float)
    touch_lo = lo_src <= lower
    touch_hi = hi_src >= upper
    since_lo, since_hi = _bars_since(touch_lo), _bars_since(touch_hi)

    if p.touch_lookback > 0:
        lo_ok = since_lo <= p.touch_lookback
        hi_ok = since_hi <= p.touch_lookback
    else:                                      # touch inside the run, or now
        lo_ok = since_lo <= red_run
        hi_ok = since_hi <= green_run

    # (4) WaveTrend cross at/beyond the level
    wt1, wt2 = d["wt1"].to_numpy(float), d["wt2"].to_numpy(float)
    pw1, pw2 = np.r_[np.nan, wt1[:-1]], np.r_[np.nan, wt2[:-1]]
    bull_x = (wt1 > wt2) & (pw1 <= pw2) & (wt1 <= -p.wt_level)
    bear_x = (wt1 < wt2) & (pw1 >= pw2) & (wt1 >= p.wt_level)
    lb = max(1, int(p.wt_cross_lookback))
    wt_long_ok = _bars_since(bull_x) <= lb - 1
    wt_short_ok = _bars_since(bear_x) <= lb - 1

    long_sig = green & (red_run >= p.min_red_run) & lo_ok & wt_long_ok
    short_sig = red & (green_run >= p.min_red_run) & hi_ok & wt_short_ok

    # optional: channel must lean against the trade (i.e. we fade the trend)
    if p.require_counter_slope:
        slope = d["lrc_slope"].to_numpy(float)
        long_sig &= slope < 0
        short_sig &= slope > 0
    if p.min_abs_pearson > 0:
        ap = np.abs(d["pearson_r"].to_numpy(float))
        long_sig &= ap >= p.min_abs_pearson
        short_sig &= ap >= p.min_abs_pearson

    # need a valid ATR, a valid channel, and a next bar to fill on
    ok = np.isfinite(d["atr"].to_numpy(float)) & np.isfinite(lower) \
        & np.isfinite(wt2)
    ok[-1] = False
    long_sig &= ok
    short_sig &= ok

    if p.sessions:
        in_sess = d["session"].isin(list(p.sessions)).to_numpy(bool)
        long_sig &= in_sess
        short_sig &= in_sess

    return pd.DataFrame({"long_sig": long_sig, "short_sig": short_sig,
                         "red_run": red_run, "green_run": green_run,
                         "since_lo": since_lo, "since_hi": since_hi},
                        index=d.index)


def silver_gate(gold: pd.DataFrame, p: Params) -> pd.DataFrame:
    """
    Align XAGUSD onto the gold bar index and return per-bar boolean gates.

    'signal' mode : silver must produce the same-direction signal within
                    ``silver_lookback`` bars.
    'zone'   mode : looser -- silver must merely be outside the same band on
                    the same side and printing the same candle colour.
    """
    sv = load("XAGUSD", "M5")
    sv = add_indicators(sv, p, "XAGUSD")
    ssig = make_signals(sv, p)
    sv = pd.concat([sv, ssig], axis=1)

    if p.silver_mode == "signal":
        s_long, s_short = sv["long_sig"].to_numpy(bool), sv["short_sig"].to_numpy(bool)
    else:
        g = sv["lrc_green"].to_numpy(bool)
        lo = sv["blow"].to_numpy(float) <= sv["lrc_lower"].to_numpy(float)
        hi = sv["bhigh"].to_numpy(float) >= sv["lrc_upper"].to_numpy(float)
        s_long, s_short = g & lo, (~g) & hi

    lb = max(1, int(p.silver_lookback))
    s_long_recent = _bars_since(s_long) <= lb - 1
    s_short_recent = _bars_since(s_short) <= lb - 1

    gate = pd.DataFrame({"time": sv["time"].to_numpy(),
                         "sv_long": s_long_recent, "sv_short": s_short_recent})
    merged = gold[["time"]].merge(gate, on="time", how="left")
    # a gold bar with no matching silver bar cannot be confirmed
    return pd.DataFrame(
        {"sv_long": merged["sv_long"].fillna(False).to_numpy(bool),
         "sv_short": merged["sv_short"].fillna(False).to_numpy(bool)},
        index=gold.index)


# --------------------------------------------------------- path-based exits

def build_paths(d: pd.DataFrame, sig_idx: np.ndarray, horizon: int):
    """
    Forward OHLC matrices for each signal, shape (n_signals, horizon).

    Row k covers bars ``sig_idx[k]+1 .. sig_idx[k]+horizon`` -- i.e. it starts on
    the fill bar, so a stop can be hit on the entry bar itself. Bars beyond the
    end of the series are NaN.
    """
    o, h, l, c = (d[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(d)
    k = len(sig_idx)
    H = np.full((k, horizon), np.nan)
    L = np.full((k, horizon), np.nan)
    C = np.full((k, horizon), np.nan)
    for j in range(horizon):
        src = sig_idx + 1 + j
        m = src < n
        H[m, j] = h[src[m]]
        L[m, j] = l[src[m]]
        C[m, j] = c[src[m]]
    entry_open = np.full(k, np.nan)
    m = (sig_idx + 1) < n
    entry_open[m] = o[sig_idx[m] + 1]
    return H, L, C, entry_open


def simulate_side(H, L, C, entry, sl_dist, tp_dist, side: int,
                  be_at_r: float = 0.0, trail_dist=None):
    """
    Resolve each path to an exit. Vectorised over trades, looped over bars.

    ``side`` = +1 long, -1 short. ``sl_dist``/``tp_dist`` are per-trade absolute
    price distances. Returns (r_mult, exit_bar, reason, mfe_r, mae_r) where the
    excursions are measured over the life of the trade.

    Stop-before-target on a same-bar collision (pessimistic).
    """
    k, hz = H.shape
    r = np.full(k, np.nan)
    exit_bar = np.full(k, -1, dtype=np.int64)
    reason = np.zeros(k, dtype=np.int8)          # 1=TP 2=SL 3=time 4=trail/BE
    mfe = np.zeros(k)
    mae = np.zeros(k)
    live = np.isfinite(entry) & np.isfinite(sl_dist) & (sl_dist > 0)
    stop = np.where(live, entry - side * sl_dist, np.nan)
    targ = np.where(live, entry + side * tp_dist, np.nan)
    be_done = np.zeros(k, dtype=bool)

    for j in range(hz):
        act = live & (exit_bar < 0) & np.isfinite(H[:, j])
        if not act.any():
            break
        fav = side * (H[:, j] - entry) if side > 0 else side * (L[:, j] - entry)
        adv = side * (L[:, j] - entry) if side > 0 else side * (H[:, j] - entry)
        with np.errstate(invalid="ignore"):
            mfe[act] = np.maximum(mfe[act], fav[act] / sl_dist[act])
            mae[act] = np.minimum(mae[act], adv[act] / sl_dist[act])

        if side > 0:
            hit_sl = act & (L[:, j] <= stop)
            hit_tp = act & (H[:, j] >= targ)
        else:
            hit_sl = act & (H[:, j] >= stop)
            hit_tp = act & (L[:, j] <= targ)

        # pessimistic collision handling: stop wins
        sl_only = hit_sl
        tp_only = hit_tp & ~hit_sl
        idx = np.where(sl_only)[0]
        if len(idx):
            r[idx] = side * (stop[idx] - entry[idx]) / sl_dist[idx]
            exit_bar[idx] = j
            reason[idx] = np.where(be_done[idx], 4, 2)
        idx = np.where(tp_only)[0]
        if len(idx):
            r[idx] = side * (targ[idx] - entry[idx]) / sl_dist[idx]
            exit_bar[idx] = j
            reason[idx] = 1

        still = live & (exit_bar < 0) & np.isfinite(C[:, j])
        if be_at_r > 0:
            promote = still & ~be_done & (mfe >= be_at_r)
            stop[promote] = entry[promote]
            be_done[promote] = True
        if trail_dist is not None:
            if side > 0:
                cand = H[:, j] - trail_dist
                stop = np.where(still & (cand > stop), cand, stop)
            else:
                cand = L[:, j] + trail_dist
                stop = np.where(still & (cand < stop), cand, stop)

    # time stop: close at the last available bar of the horizon
    open_still = live & (exit_bar < 0)
    if open_still.any():
        lastval = np.full(k, np.nan)
        for j in range(hz - 1, -1, -1):
            need = open_still & ~np.isfinite(lastval)
            if not need.any():
                break
            lastval[need] = C[need, j]
        idx = np.where(open_still & np.isfinite(lastval))[0]
        r[idx] = side * (lastval[idx] - entry[idx]) / sl_dist[idx]
        exit_bar[idx] = hz - 1
        reason[idx] = 3
    return r, exit_bar, reason, mfe, mae


# ------------------------------------------------------------------- backtest

def backtest(symbol: str, p: Params, df: pd.DataFrame | None = None,
             collect_paths: bool = False) -> pd.DataFrame:
    """Run one parameter set. Returns a trades DataFrame (may be empty)."""
    base = load(symbol, "M5") if df is None else df
    d = add_indicators(base, p, symbol)
    sig = make_signals(d, p)
    # make_signals names its outputs after the SETUP (lower-band vs upper-band),
    # not after the trade direction. The mapping to a direction happens here.
    lower_setup = sig["long_sig"].to_numpy(bool).copy()
    upper_setup = sig["short_sig"].to_numpy(bool).copy()

    if p.use_silver:
        g = silver_gate(d, p)
        lower_setup &= g["sv_long"].to_numpy(bool)
        upper_setup &= g["sv_short"].to_numpy(bool)

    if p.mode == "momentum":
        long_sig, short_sig = upper_setup, lower_setup
    else:
        long_sig, short_sig = lower_setup, upper_setup
    if not p.allow_long:
        long_sig = np.zeros_like(long_sig)
    if not p.allow_short:
        short_sig = np.zeros_like(short_sig)

    if p.cooldown_bars > 0:
        for arr in (long_sig, short_sig):
            last = -(10 ** 9)
            for i in np.where(arr)[0]:
                if i - last < p.cooldown_bars:
                    arr[i] = False
                else:
                    last = i

    point = POINT[symbol]
    contract = CONTRACT[symbol]
    spread_pts = d["spread"].to_numpy(float)
    atr_arr = d["atr"].to_numpy(float)
    hz = int(p.max_bars)

    rows = []
    for side, mask in ((+1, long_sig), (-1, short_sig)):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        H, L, C, entry_open = build_paths(d, idx, hz)
        if p.entry_mode == "signal_close":
            raw_entry = d["close"].to_numpy(float)[idx]
        else:
            raw_entry = entry_open
        # cost: full spread (plus optional slippage) charged once, at entry
        cost_pts = spread_pts[np.minimum(idx + 1, len(d) - 1)] \
            + p.extra_slippage_points
        entry = raw_entry + side * cost_pts * point

        a = atr_arr[idx]
        sl_dist = p.sl_atr * a
        tp_dist = p.tp_atr * a
        trail = p.trail_atr * a if p.trail_atr > 0 else None
        r, ebar, reason, mfe, mae = simulate_side(
            H, L, C, entry, sl_dist, tp_dist, side,
            be_at_r=p.be_at_r, trail_dist=trail)

        # full-horizon excursions, independent of the exit -- used to choose
        # TP/SL from the actual distribution rather than only from sweep P&L
        with np.errstate(invalid="ignore"):
            if side > 0:
                fh_mfe = (np.nanmax(H, axis=1) - entry) / sl_dist
                fh_mae = (np.nanmin(L, axis=1) - entry) / sl_dist
            else:
                fh_mfe = (entry - np.nanmin(L, axis=1)) / sl_dist
                fh_mae = (entry - np.nanmax(H, axis=1)) / sl_dist

        commission_r = p.commission_per_lot_rt / (sl_dist * contract)
        rows.append(pd.DataFrame({
            "i": idx,
            "dt": d["dt"].to_numpy()[idx],
            "utc": d["utc"].to_numpy()[idx],
            "setup": ("upper" if (side > 0) == (p.mode == "momentum") else "lower"),
            "session": d["session"].to_numpy()[idx],
            "dow": d["dow"].to_numpy()[idx],
            "side": side,
            "dir": "long" if side > 0 else "short",
            "entry": entry,
            "atr": a,
            "sl_dist": sl_dist,
            "tp_dist": tp_dist,
            "spread_pts": cost_pts,
            "r_gross": r,
            "r": r - commission_r,
            "usd": r * sl_dist * contract - p.commission_per_lot_rt,
            "exit_bar": ebar,
            "bars_held": ebar + 1,
            "reason": pd.Series(reason).map(
                {1: "tp", 2: "sl", 3: "time", 4: "be_trail"}).to_numpy(),
            "mfe_r": mfe,
            "mae_r": mae,
            "fh_mfe_r": fh_mfe,
            "fh_mae_r": fh_mae,
            "red_run": sig["red_run"].to_numpy()[idx],
            "green_run": sig["green_run"].to_numpy()[idx],
            "dev_sigma": d["dev_sigma"].to_numpy()[idx],
            "pearson_r": d["pearson_r"].to_numpy()[idx],
            "wt1": d["wt1"].to_numpy()[idx],
            "slope": d["lrc_slope"].to_numpy()[idx],
        }))

    if not rows:
        return pd.DataFrame()
    t = pd.concat(rows).sort_values("i").reset_index(drop=True)
    return t.dropna(subset=["r"]).reset_index(drop=True)


# ------------------------------------------------------------------ statistics

def stats(t: pd.DataFrame, label: str = "") -> dict:
    if t is None or len(t) == 0:
        return {"label": label, "n": 0, "expectancy_r": np.nan, "total_r": 0.0,
                "win_rate": np.nan, "profit_factor": np.nan, "max_dd_r": np.nan,
                "sharpe_pt": np.nan, "total_usd": 0.0, "avg_bars": np.nan}
    r = t["r"].to_numpy(float)
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    dd = eq - peak
    wins, losses = r[r > 0], r[r <= 0]
    gp, gl = wins.sum(), -losses.sum()
    return {
        "label": label,
        "n": int(len(r)),
        "expectancy_r": float(r.mean()),
        "total_r": float(r.sum()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(gp / gl) if gl > 0 else np.inf,
        "max_dd_r": float(dd.min()) if len(dd) else 0.0,
        "sharpe_pt": float(r.mean() / r.std(ddof=1)) if len(r) > 2 and r.std(ddof=1) > 0 else np.nan,
        "total_usd": float(t["usd"].sum()),
        "avg_bars": float(t["bars_held"].mean()),
        "tp_share": float((t["reason"] == "tp").mean()),
        "sl_share": float((t["reason"] == "sl").mean()),
        "time_share": float((t["reason"] == "time").mean()),
    }


def slice_stats(t: pd.DataFrame) -> pd.DataFrame:
    """
    Per-slice breakdown. Aggregates alone are not enough: every report must be
    sliceable by direction and by session.
    """
    out = [stats(t, "ALL")]
    for d in ("long", "short"):
        out.append(stats(t[t["dir"] == d], f"dir={d}"))
    for s in ("asian", "london", "overlap", "ny_pm", "off"):
        out.append(stats(t[t["session"] == s], f"sess={s}"))
    for d in ("long", "short"):
        for s in ("asian", "london", "overlap", "ny_pm", "off"):
            sub = t[(t["dir"] == d) & (t["session"] == s)]
            if len(sub):
                out.append(stats(sub, f"{d}/{s}"))
    return pd.DataFrame(out)
