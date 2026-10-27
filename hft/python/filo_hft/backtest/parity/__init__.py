"""Backtest parity shims — pure-Python ports of MQL5 gate logic.

These modules exist so backtests can replicate, tick-for-tick, the
veto decisions the live EA makes. No live code should import from here.
"""
