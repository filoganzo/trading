# Build plan

This plan mirrors §18 of the requirements doc but is mapped to the actual file
layout delivered in this drop. Each phase is a self-contained milestone —
you can stop after any phase and still have something runnable.

## Phase 1 — Core skeleton  ✅ delivered

- [x] [config/default.ini](../config/default.ini) — canonical parameter set
- [x] [mql5/Include/FiloHFT/Common.mqh](../mql5/Include/FiloHFT/Common.mqh) — shared types, enums, math helpers
- [x] [mql5/Include/FiloHFT/Config.mqh](../mql5/Include/FiloHFT/Config.mqh) — ini loader + live overrides
- [x] [mql5/Include/FiloHFT/Logger.mqh](../mql5/Include/FiloHFT/Logger.mqh) — plaintext logger
- [x] [mql5/Include/FiloHFT/Telemetry.mqh](../mql5/Include/FiloHFT/Telemetry.mqh) — structured CSV streams
- [x] [mql5/Include/FiloHFT/EventBus.mqh](../mql5/Include/FiloHFT/EventBus.mqh) — in-process pub/sub

## Phase 2 — KL engine  ✅ delivered

- [x] [mql5/Include/FiloHFT/KeyLevels.mqh](../mql5/Include/FiloHFT/KeyLevels.mqh) — all timeframes, zone merging, weights, body-cross invalidation
- [x] [mql5/Indicators/KeyLevelTracker.mq5](../mql5/Indicators/KeyLevelTracker.mq5) — upgraded visual indicator

## Phase 3 — Feature engine  ✅ delivered

- [x] [mql5/Include/FiloHFT/Features.mqh](../mql5/Include/FiloHFT/Features.mqh) — channel, acceleration, burst detection, trend state
- [x] [mql5/Include/FiloHFT/Correlation.mqh](../mql5/Include/FiloHFT/Correlation.mqh) — XAGUSD / dxy_m6 directional agreement

## Phase 4 — Setup engine  ✅ delivered

- [x] [mql5/Include/FiloHFT/Setups.mqh](../mql5/Include/FiloHFT/Setups.mqh) — breakout / retest / rejection / fakeout / failed-auction / engulfing-accel / mean-reversion, each with intrinsic score

## Phase 5 — Execution + trade management  ✅ delivered

- [x] [mql5/Include/FiloHFT/ExecutionIntent.mqh](../mql5/Include/FiloHFT/ExecutionIntent.mqh) — internal intent state machine
- [x] [mql5/Include/FiloHFT/Execution.mqh](../mql5/Include/FiloHFT/Execution.mqh) — spread/slippage gating, OrderSend wrapper
- [x] [mql5/Include/FiloHFT/TradeManagement.mqh](../mql5/Include/FiloHFT/TradeManagement.mqh) — hidden SL/TP, break-even, trailing, concurrency

## Phase 6 — Risk + safety  ✅ delivered

- [x] [mql5/Include/FiloHFT/Risk.mqh](../mql5/Include/FiloHFT/Risk.mqh) — hard limits, per-bucket loss tracking, kill switches
- [x] [mql5/Include/FiloHFT/Safety.mqh](../mql5/Include/FiloHFT/Safety.mqh) — spread/stall/latency/slippage anomaly detection

## Phase 7 — Research harness  ✅ delivered

- [x] [python/filo_hft/data](../python/filo_hft/data) — MT5 history extraction
- [x] [python/filo_hft/backtest](../python/filo_hft/backtest) — event-driven replay mirroring MQL5 logic
- [x] [python/filo_hft/sweeps](../python/filo_hft/sweeps) — parameter sweep runner
- [x] [python/scripts/extract_history.py](../python/scripts/extract_history.py)
- [x] [python/scripts/run_backtest.py](../python/scripts/run_backtest.py)
- [x] [python/scripts/run_sweep.py](../python/scripts/run_sweep.py)

## Phase 8 — ML layer  ✅ delivered

- [x] [python/filo_hft/ml](../python/filo_hft/ml) — feature extraction from telemetry, baseline lightgbm classifier, blending
- [x] [python/scripts/train_ml.py](../python/scripts/train_ml.py)

## Phase 9 — GUI / dashboard  ✅ delivered

- [x] [python/filo_hft/gui](../python/filo_hft/gui) — Streamlit dashboard reading live telemetry
- [x] [python/scripts/run_dashboard.py](../python/scripts/run_dashboard.py)

## What's _not_ delivered

- **Tuned parameters.** Every number in `default.ini` is a starting hypothesis.
  The whole point of the sweep runner is to find the real values.
- **Trained ML model.** The training pipeline is wired but no model is shipped.
  You must run `train_ml.py` against extracted history + backtest labels.
- **News/blackout integration.** `sessions.json` has the schema; the actual
  feed integration (e.g. TradingEconomics, Forex Factory scrape) is a later
  addition.
- **Production hardening.** No auto-restart, no Windows service wrapper, no
  watchdog. Run inside a supervised MT5 terminal on the VPS.
