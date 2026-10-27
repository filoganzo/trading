"""filo_hft — research + tooling for the FiloHFT XAUUSD low-latency platform.

This package is the off-path half of the system. The hot path lives in MQL5
and runs inside an MT5 terminal on the VPS. This package:

  - extracts historical ticks and bars from MT5
  - replays them through an event-driven backtest that mirrors the MQL5
    engine's decisions
  - runs parameter sweeps and segmentation reports
  - trains the ML scoring layer
  - serves a Streamlit dashboard that reads live telemetry CSVs

All tooling shares configuration with the MQL5 side via config/default.ini.
See filo_hft.config.
"""

__version__ = "0.1.0"
