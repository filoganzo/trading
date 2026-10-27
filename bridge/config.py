"""Shared configuration and file contract for the OANDA->Vantage bridge.

Data flow
---------
  OANDA   : GOLD.pro / SILVER.pro   (the price series the strategy was built on)
  IC      : DXY_Z6                  (OANDA has no dollar index at all)
  Vantage : XAUUSD+                 (execution only - LIVE)

The panel computes everything in Python and writes two files into the MetaQuotes
Common\\Files directory, which every terminal can read via FILE_COMMON:

  bridge_settings.json  - all bot inputs, written whenever the panel saves
  bridge_signal.json    - the current command, rewritten every poll

The EA on the Vantage chart reads both, validates freshness, and executes.

Clock alignment
---------------
OANDA and IC run on different server clocks (measured: OANDA is 1h behind IC).
Never compare raw bar timestamps across brokers - always align first. The panel
measures the offset live rather than hard-coding it, because it changes with DST.
"""
from pathlib import Path

# ---- terminals -------------------------------------------------------------
OANDA_EXE = r"C:\Program Files\OANDA TMS MT5 Terminal\terminal64.exe"
IC_EXE = r"C:\Program Files\MetaTrader 5 IC Markets EU\terminal64.exe"
VANTAGE_EXE = r"C:\Program Files\Vantage International MT5\terminal64.exe"

# ---- symbols ---------------------------------------------------------------
OANDA_GOLD = "GOLD.pro"
OANDA_SILVER = "SILVER.pro"
IC_DXY = "DXY_Z6"
VANTAGE_GOLD = "XAUUSD+"

# ---- shared files ----------------------------------------------------------
COMMON_FILES = Path(r"C:\Users\<user>\AppData\Roaming\MetaQuotes\Terminal\Common\Files")
SETTINGS_FILE = COMMON_FILES / "bridge_settings.json"
SIGNAL_FILE = COMMON_FILES / "bridge_signal.json"
HEARTBEAT_FILE = COMMON_FILES / "bridge_heartbeat.json"
STATE_FILE = Path(__file__).with_name("panel_state.json")

# ---- instrument conventions ------------------------------------------------
PIP = 0.1                # 1 pip on gold = 0.10 price units
USD_PER_PIP_PER_LOT = 10.0   # verified from 11 live IC fills

# ---- defaults: mirror the .set preset ---------------------------------------
DEFAULTS = {
    # master switch - the EA refuses to trade unless this is true
    "enabled": False,

    # instrument / sizing
    "lots": 0.01,
    "magic": 880155,
    "slippage_points": 20,
    "max_spread_points": 0,      # 0 = off
    "max_trades_day": 0,         # 0 = unlimited
    "one_position_only": True,

    # risk
    "tp_pips": 40.0,
    "sl_pips": 400.0,
    "use_breakeven": True,
    "be_trigger_pips": 35.0,
    "be_buffer_pips": 1.0,

    # LinReg candles
    "use_linreg_candles": True,
    "linreg_length": 15,
    "signal_length": 9,
    "sma_signal": True,

    # correlation confirmation
    "use_correlation": True,
    "require_fresh": True,
    "reject_doji": True,

    # entry modules - both on, as in the live preset
    "enable_break_reclaim": True,
    "enable_rejection": True,

    # break & reclaim
    "origin_bars": 3,
    "reclaim_window": 5,
    "max_dist_pips": 30.0,
    "min_room_pips": 10.0,
    "use_anchor": True,

    # rejection module (daily levels only)
    # shape modes: 0 any, 1 wick, 2 body, 3 wick-or-body, 4 wick-and-body
    "rej_max_dip_pips": 15.0,
    "rej_approach_bars": 1,
    "rej_shape_mode": 3,
    "rej_wick_body_mult": 2.0,
    "rej_min_body_pips": 15.0,
    "rej_require_color": True,

    # cooldown
    "cooldown_bars": 5,

    # level engine
    "enable_h4": True,
    "enable_d1": True,
    "max_levels": 60,
    "seed_htf_bars": 250,

    # time window (Rome clock, inclusive start, exclusive end; "" = always on)
    "use_time_window": False,
    "session_start": "08:30",
    "session_end": "11:00",

    # wavetrend
    "wt_channel": 10,
    "wt_average": 21,
    "wt_threshold": 40.0,

    # linreg deviation
    "reg_length": 100,
    "reg_dev_mult": 1.2,

    # bridge behaviour
    "poll_seconds": 5,
    "signal_ttl_seconds": 90,    # EA ignores a signal older than this
}
