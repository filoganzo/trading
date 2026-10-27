import sys
sys.path.insert(0, r'path/to/repo/hft')

from stocks.backtest.config import BT_CACHE_DIR, BT_CACHE_M15_DIR, BT_CACHE_W1_DIR
from stocks.backtest.data.extender import load_bt_bars, load_bt_bars_m15, load_bt_bars_w1
from stocks.backtest.signals.wt_lrc_signal import _lrc_vectorized, _lrc_candle_close_vectorized, _atr_vectorized
from stocks.backtest.signals.htf_signal import precompute_weekly_levels
import numpy as np
import pandas as pd

SYMBOLS = ['CSR.NYSE', 'SBAC.NAS', 'WM.NYSE', 'LAZ.NYSE']
ENTRY_DATE = '2026-04-24'  # primary entry date to check M15 bars

SEP  = '=' * 70
SEP2 = '-' * 70

def classify(lrc_c_val, upper_val, lower_val):
    if lrc_c_val > upper_val:
        return 'SHORT (lrc_c > upper)'
    elif lrc_c_val < lower_val:
        return 'LONG  (lrc_c < lower)'
    else:
        return 'INSIDE band'

for sym in SYMBOLS:
    print(SEP)
    print(f'SYMBOL: {sym}')
    print(SEP)

    # ------------------------------------------------------------------ #
    # 1. Load daily bars
    # ------------------------------------------------------------------ #
    try:
        daily = load_bt_bars(sym)
    except Exception as e:
        print(f'  [ERROR] Could not load daily bars: {e}')
        continue

    print(f'  Daily bars loaded: {len(daily)} rows  |  {daily.index[0].date()} to {daily.index[-1].date()}')

    c = daily['Close'].values.astype(float)
    h = daily['High'].values.astype(float)
    l = daily['Low'].values.astype(float)

    # ------------------------------------------------------------------ #
    # 2. Compute indicators
    # ------------------------------------------------------------------ #
    mid, upper, lower, dsig = _lrc_vectorized(c, 80, 1.5)
    lrc_c             = _lrc_candle_close_vectorized(c, 15, 9)
    atr               = _atr_vectorized(h, l, c, 14)

    # ------------------------------------------------------------------ #
    # 3. Last 10 bars — full diagnostic table
    # ------------------------------------------------------------------ #
    print()
    print('  --- LAST 10 DAILY BARS ---')
    print(f"  {'Date':<14} {'Close':>8} {'lrc_c':>8} {'upper':>8} {'lower':>8} {'ATR':>7}  {'Signal'}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}  {'-'*25}")

    last10_idx = range(len(daily) - 10, len(daily))
    for i in last10_idx:
        date_str  = str(daily.index[i].date())
        raw_close = c[i]
        lc        = lrc_c[i]
        up        = upper[i]
        lo        = lower[i]
        at        = atr[i]
        sig       = classify(lc, up, lo)
        print(f"  {date_str:<14} {raw_close:>8.2f} {lc:>8.2f} {up:>8.2f} {lo:>8.2f} {at:>7.2f}  {sig}")

    # ------------------------------------------------------------------ #
    # 4. Signal bar: find where lrc_c last crossed outside band
    # ------------------------------------------------------------------ #
    print()
    print('  --- SIGNAL BAR DETECTION ---')

    # look back last 20 bars for any cross outside band
    lookback = 20
    start_i  = max(0, len(daily) - lookback)

    signal_bars = []
    for i in range(start_i, len(daily)):
        lc = lrc_c[i]
        up = upper[i]
        lo = lower[i]
        if lc > up or lc < lo:
            direction = 'SHORT' if lc > up else 'LONG'
            band_edge = up if lc > up else lo
            pct_from_edge = (lc - band_edge) / band_edge * 100.0
            signal_bars.append({
                'i':           i,
                'date':        daily.index[i].date(),
                'close':       c[i],
                'lrc_c':       lc,
                'upper':       up,
                'lower':       lo,
                'direction':   direction,
                'band_edge':   band_edge,
                'pct_edge':    pct_from_edge,
            })

    if signal_bars:
        for sb in signal_bars:
            print(f"  Signal bar : {sb['date']}  ({sb['direction']})")
            print(f"    raw_close  = {sb['close']:.4f}")
            print(f"    lrc_c      = {sb['lrc_c']:.4f}")
            print(f"    upper      = {sb['upper']:.4f}")
            print(f"    lower      = {sb['lower']:.4f}")
            print(f"    band_edge  = {sb['band_edge']:.4f}")
            print(f"    lrc_c dist from edge = {sb['pct_edge']:+.4f}%")
            print(f"    close dist from edge = {(sb['close'] - sb['band_edge']) / sb['band_edge'] * 100:+.4f}%")
            print()
    else:
        print('  No signal bars found in last 20 daily bars.')

    # ------------------------------------------------------------------ #
    # 5. lrc_c vs raw close comparison (last 10 bars)
    # ------------------------------------------------------------------ #
    print('  --- LRC_C vs RAW CLOSE DIVERGENCE (last 10 bars) ---')
    print(f"  {'Date':<14} {'Close':>8} {'lrc_c':>8} {'Diff':>8} {'Diff%':>8}  {'lrc_c inside?'}")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*25}")
    for i in last10_idx:
        date_str  = str(daily.index[i].date())
        raw_close = c[i]
        lc        = lrc_c[i]
        diff      = lc - raw_close
        diff_pct  = diff / raw_close * 100
        inside    = classify(lc, upper[i], lower[i])
        close_inside = classify(raw_close, upper[i], lower[i])
        match_flag = '  <-- MISMATCH' if inside != close_inside else ''
        print(f"  {date_str:<14} {raw_close:>8.2f} {lc:>8.2f} {diff:>8.2f} {diff_pct:>7.2f}%  lrc_c={inside.split()[0]}  close={close_inside.split()[0]}{match_flag}")

    # ------------------------------------------------------------------ #
    # 6. M15 bars for entry date (check April 23 and 24)
    # ------------------------------------------------------------------ #
    print()
    print('  --- M15 BARS FOR ENTRY DATE (Apr 23 & 24) ---')
    try:
        m15 = load_bt_bars_m15(sym)
        for target_date in ['2026-04-23', '2026-04-24']:
            day_bars = m15[m15.index.date == pd.Timestamp(target_date).date()]
            if len(day_bars) == 0:
                print(f"  [{target_date}] No M15 bars found.")
                continue
            print(f"  [{target_date}] {len(day_bars)} M15 bars total. Showing last 5:")
            show = day_bars.tail(5)
            for ts, row in show.iterrows():
                print(f"    {str(ts):<35}  O={row['Open']:>8.2f}  H={row['High']:>8.2f}  L={row['Low']:>8.2f}  C={row['Close']:>8.2f}  V={int(row['Volume'])}")
    except Exception as e:
        print(f"  [ERROR] Could not load M15 bars: {e}")

    # ------------------------------------------------------------------ #
    # 7. Weekly swing levels (top 5 by score)
    # ------------------------------------------------------------------ #
    print()
    print('  --- WEEKLY SWING LEVELS (top 5 by score, touch_frac=0.005) ---')
    try:
        w1 = load_bt_bars_w1(sym)
        wh = w1['High'].values.astype(float)
        wl = w1['Low'].values.astype(float)
        levels = precompute_weekly_levels(wh, wl, touch_fracs=[0.005, 0.01])
        for tf in [0.005, 0.01]:
            top5 = levels[tf][:5]
            print(f"  touch_frac={tf}:")
            if top5:
                for px, sc in top5:
                    print(f"    level={px:.4f}   score={sc:.1f}")
            else:
                print('    (none)')
    except Exception as e:
        print(f"  [ERROR] Could not compute weekly levels: {e}")

    print()

print(SEP)
print('DIAGNOSTIC COMPLETE')
print(SEP)
