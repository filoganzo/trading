# Trading system

My personal trading stack. For years I studied markets by hand: I read everything I could, talked to anyone who would talk to me, and traded manually. When AI coding tools got good enough, and I understood the market well enough, I turned that knowledge into software. The software helps me decide and executes trades under my supervision.

The repo contains source code only. There are no account details, server names, credentials, market data or results.

## What's here

| Folder | What it does |
|---|---|
| `hft/` | Event-driven intraday platform on MetaTrader 5. An MQL5 engine handles time-critical work: level detection, setup triggers, hidden stops, risk kill-switches. A Python layer handles backtesting, parameter sweeps, walk-forward validation, ML re-scoring, telemetry and dashboards. See `hft/README.md`, `hft/ARCHITECTURE.md`, `hft/METHODOLOGY.md`. |
| `hft/GoldStraddle/` | Level-triggered straddle EA driven from a browser control panel. EA and panel talk through a file-based JSON contract (`CONTRACT.md`). |
| `hft/strategies/`, `hft/stocks/`, `hft/lrc_bot/` | Linear-regression-channel strategies and research on index and stock CFDs, including split detection in broker data, a screener, and a packaged desktop bot. |
| `bridge/` | Multi-source bridge. Takes prices from one broker and the dollar index from another (which the first broker doesn't carry), puts their server clocks on one true-UTC axis, computes the signal in Python, and sends execution to a third terminal. The executor EA computes nothing itself and must pass four independent gates before it trades. |

## Design principles

- **Every live order needs a human.** Every live path starts disarmed and has to be explicitly armed.
- **Try to break each strategy before trusting it:** walk-forward splits, cost and slippage stress, and random baselines.
- **Timestamps are a first-class problem.** Brokers run on different server clocks, and the code measures and corrects the offsets before merging data.

## Stack

Python (pandas, NumPy, scikit-learn, Streamlit, Tkinter/PyQt), MQL5, MetaTrader 5 Python API.
