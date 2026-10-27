# FiloHFT — XAUUSD MT5 Low-Latency Platform

A retail low-latency event-driven trading platform for XAUUSD on IC Markets MT5.

**Not** true exchange-side HFT. This is an honest retail implementation that takes
the HFT architectural playbook (siloed engines, explicit latency budgets, hidden
order intent) and adapts it to what is actually possible through MT5 and broker
routing.

## What it does

Trades microstructural behavior around **Key Levels** (KLs) derived from
multi-timeframe prior highs/lows, using acceleration bursts, regression/stdev
channels, failed-auction detection, and engulfing-with-acceleration patterns.
All broker-visible logic is reduced to market orders triggered by internal
intent; stops and targets are hidden inside the EA.

## Layout

```
FINAL HFT/
├── README.md                       ← you are here
├── ARCHITECTURE.md                 ← component map, latency budget, data flow
├── docs/
│   ├── build-plan.md               ← phased delivery plan (Phase 1-9)
│   ├── latency-budget.md           ← where each millisecond goes
│   ├── research-questions.md       ← open questions to answer empirically
│   ├── config-reference.md         ← every tunable parameter
│   └── telemetry-schema.md         ← CSV schemas for every event stream
├── config/
│   ├── default.ini                 ← canonical parameter defaults (MT5+Python)
│   ├── sessions.json               ← session definitions (Asia/London/NY/overlap)
│   └── live_overrides.json         ← runtime overrides (hot-reload)
├── mql5/                           ← VPS hot path, runs inside MT5 terminal
│   ├── Experts/
│   │   └── FiloHFT.mq5             ← main EA entry point
│   ├── Indicators/
│   │   └── KeyLevelTracker.mq5     ← visual KL indicator (existing, upgraded)
│   └── Include/FiloHFT/
│       ├── Config.mqh              ← parameters + ini parsing
│       ├── Logger.mqh              ← structured CSV logger
│       ├── Telemetry.mqh           ← high-frequency event export
│       ├── EventBus.mqh            ← in-process pub/sub
│       ├── KeyLevels.mqh           ← KL engine (all timeframes, zones, weights)
│       ├── Features.mqh            ← channel, acceleration, bursts, trend
│       ├── Correlation.mqh         ← XAGUSD / DXY proxy correlation
│       ├── Setups.mqh              ← setup detection and scoring
│       ├── ExecutionIntent.mqh     ← internal order intent state machine
│       ├── Execution.mqh           ← broker order submission
│       ├── TradeManagement.mqh     ← hidden SL/TP, trailing, break-even
│       ├── Risk.mqh                ← hard limits, kill switches
│       ├── Safety.mqh              ← anomaly detection (spread, stall, slippage)
│       └── Common.mqh              ← shared types, enums, utilities
├── python/                         ← off-path research and tooling
│   ├── pyproject.toml
│   ├── filo_hft/
│   │   ├── config.py               ← ini loader, mirrors MQL5 Config.mqh
│   │   ├── data/                   ← MT5 extraction + file loaders
│   │   ├── backtest/               ← event-driven replay engine
│   │   ├── sweeps/                 ← parameter sweep runner + segmentation
│   │   ├── ml/                     ← feature extraction, training, inference
│   │   └── gui/                    ← Streamlit dashboard
│   └── scripts/                    ← CLI entry points
└── telemetry/                      ← CSV output from live EA (gitignored)
```

## Where work lives

| Concern                          | Home           | Why                                  |
|----------------------------------|----------------|--------------------------------------|
| Tick ingestion, hot features     | MQL5 (VPS)     | sub-ms in-process; no IPC overhead   |
| KL engine, setup detection       | MQL5 (VPS)     | fires on tick, must see every tick   |
| Execution, hidden SL/TP, trailing| MQL5 (VPS)     | broker bridge is in the same process |
| Risk limits, kill switches       | MQL5 (VPS)     | must hard-gate every order           |
| Historical data extraction       | Python         | `MetaTrader5` Python package         |
| Backtest / replay                | Python         | no latency constraint, fast iteration|
| Parameter sweeps                 | Python         | parallelizable, heavy compute        |
| ML training + inference          | Python         | sklearn / lightgbm ecosystem         |
| Research dashboard               | Python         | Streamlit, off-VPS                   |
| Live monitor GUI                 | MQL5 chart obj | no IPC dependency for live state     |

## Getting started

See [docs/build-plan.md](docs/build-plan.md) for the full phased build and
[ARCHITECTURE.md](ARCHITECTURE.md) for the component-by-component design.

### 1. Deploy the MT5 hot path (VPS side)

1. Copy [mql5/Include/FiloHFT/](mql5/Include/FiloHFT/), [mql5/Experts/FiloHFT.mq5](mql5/Experts/FiloHFT.mq5),
   and [mql5/Indicators/KeyLevelTracker.mq5](mql5/Indicators/KeyLevelTracker.mq5) into your
   MT5 `MQL5/` data folder (Tools → Open Data Folder).
2. Compile `FiloHFT.mq5` in MetaEditor. It should build clean against IC Markets MT5 Demo.
3. Copy [config/default.ini](config/default.ini), [config/sessions.json](config/sessions.json),
   and [config/live_overrides.json](config/live_overrides.json) into `MQL5/Files/FiloHFT/`.
4. Attach the EA to an XAUUSD M1 chart. Leave `InpEnableTrading=false` on first run —
   the EA will stream telemetry into `MQL5/Files/FiloHFT/telemetry/<YYYYMMDD>/*.csv`
   without sending a single order. Flip the switch only after you've verified the
   stream looks sane in the dashboard.

### 2. Install the Python toolchain (research / off-path)

```bash
cd python
pip install -e .
```

This registers five CLIs: `filo-extract`, `filo-backtest`, `filo-sweep`, `filo-train`, `filo-dashboard`.

### 3. Pull historical data

```bash
python -m filo_hft.scripts.extract_history \
    --from 2024-01-01 --to 2024-06-30 \
    --credentials ../credentials.local.ini
```

Writes `data/<symbol>/ticks_*.parquet` and `data/<symbol>/bars_<TF>_*.parquet` for
the primary symbol plus XAGUSD and dxy_m6 proxies.

### 4. Run a backtest

```bash
python -m filo_hft.scripts.run_backtest \
    --config ../config/default.ini \
    --data ../data --symbol XAUUSD \
    --from 2024-01-01 --to 2024-06-30 \
    --out ../backtest_out
```

Produces `trades.parquet`, `candidates.parquet` (with `block_reason`), `equity.parquet`,
and `metrics.json`.

### 5. Sweep parameters

```bash
python -m filo_hft.scripts.run_sweep \
    --config ../config/default.ini \
    --data ../data --symbol XAUUSD \
    --from 2024-01-01 --to 2024-06-30 \
    --param channel.length 60 80 100 120 \
    --param acceleration.min_bursts 1 2 3 \
    --metric profit_factor --top-n 10 \
    --out ../sweeps_out
```

Joblib-parallelised grid. Capped at 200 points by default (`--no-cap` to override).
Emits `sweep_results.parquet` + per-point trade logs + `sweep_summary.md`.

### 6. Train the ML scoring layer

Collect a few weeks of live telemetry (or backtest telemetry) first, then:

```bash
python -m filo_hft.scripts.train_ml \
    --telemetry ../telemetry \
    --out ../models/baseline.joblib
```

Joins `setups → intents → fills → trades` to build `(features, realized_usd > 0)`
pairs, trains LightGBM binary classifier, saves a `TrainedModel` bundle that
`filo_hft.ml.infer.BaselinePredictor` can load on the hot path later.

### 7. Launch the live dashboard

```bash
python -m filo_hft.scripts.run_dashboard \
    --telemetry ../telemetry --port 8501
```

Streamlit app with 10 panels over a rolling window of the EA's telemetry CSVs —
price+channel+KLs, acceleration, trend/correlation, candidate queue, live trades,
execution histograms, safety/risk log, session badge. Read-only; safe to run
against live telemetry while the EA is trading.

## Status

Phase 1-9 scaffold delivered. Every engine has a working implementation — no
stubs. Empirical tuning (channel length, stdev multiplier, burst thresholds,
session behaviors) is **research work** done by running the sweep harness and
training the ML layer against real data — see [docs/research-questions.md](docs/research-questions.md).
