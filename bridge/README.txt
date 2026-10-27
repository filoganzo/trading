GOLD BRIDGE  -  OANDA data / Vantage execution
==============================================

WHAT IT IS
----------
Your Break & Reclaim strategy, split in two:

  GoldBridge.exe   Python panel. Reads gold + silver from OANDA and DXY from
                   IC Markets, aligns the three clocks to true UTC, computes
                   the signal, and writes it to a shared file.

  BridgeExecutor   MT5 expert on the Vantage chart. Computes NOTHING. It reads
                   the panel's file and places the market order.

OANDA has no dollar index of any kind (checked: 1,763 symbols, none), which is
why DXY comes from IC Markets. The strategy keeps using the OANDA prices it was
built on; Vantage only executes.

They talk through two files in
  C:\Users\<user>\AppData\Roaming\MetaQuotes\Terminal\Common\Files\
      bridge_settings.json   every bot setting
      bridge_signal.json     the current command, rewritten every few seconds


FIRST RUN
---------
1. Open the Vantage terminal.

2. Enable algo trading:
       Tools -> Options -> Expert Advisors -> "Allow Algo Trading"
   and make sure the "Algo Trading" toolbar button is green.
   (It is currently OFF. Nothing can trade until you turn it on.)

3. Open an XAUUSD+ chart, any timeframe.

4. Drag "BridgeExecutor" onto it. In the inputs:
       InpAllowLiveTrading = false     <-- LEAVE FALSE FOR THE FIRST SESSION
   Allow it in the Common tab. It will log signals without sending orders,
   so you can watch it agree with your chart before it touches money.

5. Run GoldBridge.exe. It opens DISARMED (red). Nothing is sent while red.

6. Watch the diagnostics row. It shows wavetrend, the z-score, whether the
   LinReg band is hit, whether a level is touched, the silver and DXY candle
   colours, and any armed setup. This is the EA's own dashboard, recomputed
   in Python.

7. When you are satisfied it matches, set InpAllowLiveTrading = true and use
   the ARM button when you see a setup coming.


THE SWITCH
----------
ARMED / DISARMED is the big button. Disarming writes a FLAT command
immediately, as does closing the panel, so entries stop the moment you hit it.

DISARM DOES NOT CLOSE AN OPEN TRADE. It stops new entries. Use CLOSE ALL for
an open position. A live trade keeps its TP, SL and breakeven, which the EA
manages by itself - so if the panel or the PC dies mid-trade, the position is
still protected.

Four gates must all be open before an order is sent:
    1. InpAllowLiveTrading  = true   (EA input)
    2. the panel is ARMED
    3. the signal is younger than its TTL (90s)
    4. it is not a signal already executed
Any one shut means nothing happens.


THE TWO ENTRY MODULES
---------------------
Both are on by default, matching your live preset.

  BREAK & RECLAIM  Price closes through an H4/D1 level (with the last
                   origin_bars closes on the other side), that arms a setup,
                   and a later close back through it inside reclaim_window
                   bars fires the entry. Needs a level touch + wavetrend
                   extreme + LinReg band on the ARMING bar.

  REJECTION        Daily levels only. A single candle reaches a level,
                   overshoots by no more than rej_max_dip_pips, and closes
                   back on the origin side. Nothing is armed - it is one bar.
                   Needs the wavetrend extreme and LinReg band on that same
                   bar, so it fires less often than the setup count suggests.

  rej_shape_mode   0 any, 1 wick, 2 body, 3 wick-or-body (default), 4 both.
                   "Wick" means the wick pointing back into the level is at
                   least rej_wick_body_mult x the body.

If both fire on the same bar, Break & Reclaim wins - it carries an armed
setup that would otherwise expire, while a rejection recurs on its own terms.
Only one order ever leaves per bar.


SETTINGS
--------
Defaults mirror your live preset: 0.01 lots, TP 40 pips, SL 400 pips,
breakeven at 35 pips with a 1 pip buffer, magic 880155.

Edit a field, press "Apply settings". Changing a level input (max_levels,
seed_htf_bars, enable_h4, enable_d1) re-seeds the level engine.

"lots" is fixed size. It does NOT scale with the account - at 0.01 lots a
400-pip stop is about $40.


WHAT TO WATCH FOR
-----------------
DXY prints a doji on ~21% of M1 bars (it is a futures index, thinner than
gold). With reject_doji on, those bars can never produce an entry. This is the
same in your backtest (22.7% historically), so it is expected, not a fault -
but it is why signals are rarer than the setup count suggests.

The correlation gate needs silver and DXY to disagree in the right direction:
    LONG  wants silver GREEN and DXY RED
    SHORT wants silver RED   and DXY GREEN
About a third of non-doji bars confirm each way.

The panel needs all three terminals running and logged in. If one is closed,
the status line shows the error and no signal is written - the EA then sees a
stale file and refuses to act, which is the intended failure mode.


IF YOU CHANGE THE PYTHON
------------------------
The .exe is built from these .py files. To rebuild after editing:

    python -m PyInstaller --noconfirm --onefile --windowed --name GoldBridge ^
      --hidden-import MetaTrader5 --collect-submodules MetaTrader5 ^
      --distpath . panel.py

Or just run it from source:   python panel.py
Headless, never arms:         python run_headless.py --minutes 30
Tests:                        python test_bridge.py


FILES
-----
  GoldBridge.exe      the panel (standalone, no Python needed)
  BridgeExecutor.mq5  EA source
  BridgeExecutor.ex5  EA compiled (already installed in the Vantage terminal)
  panel.py            GUI + on/off switch
  strategy.py         the port of ProcessBar() - the decision
  levels.py           H4/D1 level engine
  indicators.py       WaveTrend, LinReg candles, regression channel
  feed.py             multi-terminal data with clock alignment
  config.py           terminals, symbols, defaults
  run_headless.py     watch-only runner
  test_bridge.py      38 tests over the signal logic
