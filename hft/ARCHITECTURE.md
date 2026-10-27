# Architecture

## Principle: retail low-latency, not exchange-side HFT

The requirement doc asks for a 0.65 ms VPS latency floor and ~0.5 ms polling.
Those numbers are only defensible **inside the MT5 terminal process**. Any
component that crosses the MT5 API boundary or goes through the broker bridge
is bounded by the terminal's own cycle, not by our code. So the architecture
puts **everything decision-critical inside an MQL5 EA**, and pushes everything
else (research, training, dashboards) to Python that runs out-of-band.

## Two-tier deployment

```
┌─────────────────────────────────────────────────────────────────┐
│                            VPS                                 │
│                                                                 │
│   ┌──────────────────── MT5 Terminal ──────────────────────┐    │
│   │                                                        │    │
│   │   ┌─── FiloHFT.mq5 (EA, hot path) ──────────────┐      │    │
│   │   │                                             │      │    │
│   │   │   OnTick:                                   │      │    │
│   │   │     1. Features.Update(tick)                │      │    │
│   │   │     2. KeyLevels.UpdateIfDue(tick)          │      │    │
│   │   │     3. Correlation.Update(tick)             │      │    │
│   │   │     4. Setups.Scan(features, KLs)           │      │    │
│   │   │     5. Execution.ProcessIntents(features)   │      │    │
│   │   │     6. TradeManagement.Step(features)       │      │    │
│   │   │     7. Risk.Check() / Safety.Check()        │      │    │
│   │   │     8. Telemetry.Flush() (batched)          │      │    │
│   │   │                                             │      │    │
│   │   └─────────────────────────────────────────────┘      │    │
│   │           │                                            │    │
│   │           ▼ writes CSV                                 │    │
│   │   telemetry/*.csv (append-only)                        │    │
│   │                                                        │    │
│   └────────────────────────────────────────────────────────┘    │
│           │                                                     │
│           ▼ read by                                             │
│   Python scripts (can run on VPS or off-VPS):                   │
│     - extract_history.py    (one-shot, pulls history)           │
│     - run_backtest.py       (offline replay)                    │
│     - run_sweep.py          (param sweeps)                      │
│     - run_dashboard.py      (Streamlit, live telemetry reader)  │
│     - train_ml.py           (offline ML)                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Engine map (from requirements §12)

| Engine | Implementation                     | Runs on  | Update cadence   |
|--------|------------------------------------|----------|------------------|
| A — Market data / features | [mql5/Include/FiloHFT/Features.mqh](mql5/Include/FiloHFT/Features.mqh) | VPS  | every tick |
| B — Key Levels             | [mql5/Include/FiloHFT/KeyLevels.mqh](mql5/Include/FiloHFT/KeyLevels.mqh) | VPS  | on new bar per TF |
| C — Setup detection        | [mql5/Include/FiloHFT/Setups.mqh](mql5/Include/FiloHFT/Setups.mqh)       | VPS  | every tick |
| D — Execution              | [mql5/Include/FiloHFT/Execution.mqh](mql5/Include/FiloHFT/Execution.mqh) | VPS  | every tick (when intents exist) |
| E — Trade management       | [mql5/Include/FiloHFT/TradeManagement.mqh](mql5/Include/FiloHFT/TradeManagement.mqh) | VPS | every tick + 100ms timer |
| F — Risk / safety          | [mql5/Include/FiloHFT/Risk.mqh](mql5/Include/FiloHFT/Risk.mqh), [Safety.mqh](mql5/Include/FiloHFT/Safety.mqh) | VPS | every tick + 1s timer |
| G — Research / replay      | [python/filo_hft/backtest](python/filo_hft/backtest), [sweeps](python/filo_hft/sweeps) | off-path | offline |
| H — GUI / observability    | [mql5/Include/FiloHFT/Telemetry.mqh](mql5/Include/FiloHFT/Telemetry.mqh) + [python/filo_hft/gui](python/filo_hft/gui) | both | live |

## Latency budget

See [docs/latency-budget.md](docs/latency-budget.md) for the per-stage numbers.
Summary:

| Stage                              | Budget | Realistic |
|------------------------------------|--------|-----------|
| Tick arrives at terminal           | —      | —         |
| OnTick handler entry               | 0.01 ms| ~0.02 ms  |
| Features update                    | 0.05 ms| ~0.1 ms   |
| KL zone lookup                     | 0.02 ms| ~0.05 ms  |
| Setup scan (all families)          | 0.10 ms| ~0.2 ms   |
| Intent → OrderSend call            | 0.02 ms| ~0.05 ms  |
| OrderSend returns (broker RTT)     | —      | 1-5 ms    |
| Risk/safety checks                 | 0.05 ms| ~0.1 ms   |
| Telemetry write (batched)          | 0.02 ms| ~0.05 ms  |
| **Total in-process decision path** | ~0.3ms | **~0.6 ms**|
| **Plus broker RTT**                |        | **+1-5 ms**|

The "0.65 ms" in the requirements is achievable for the **decision path inside
the EA**. The actual round-trip including OrderSend is bounded by the broker
bridge, which is outside our control. That is why the execution model is
"internal intent + market order" rather than resting orders: we want the
decision to be made as late as possible, on the freshest tick, so broker RTT
is applied to the most current price.

## Event bus

`EventBus.mqh` is a minimal in-process publish/subscribe. It is not a
distributed message bus — it is a fixed-size ring buffer of event structs with
typed subscribers. It exists so that engines remain testable in isolation and
so that telemetry export has a single chokepoint.

Events:

- `EVT_TICK` — new tick
- `EVT_BAR_CLOSE` — bar closed on any tracked timeframe
- `EVT_KL_UPDATE` — key levels refreshed
- `EVT_FEATURE_UPDATE` — feature snapshot ready
- `EVT_SETUP_CANDIDATE` — setup detected (accepted or blocked)
- `EVT_INTENT_CREATED` — execution intent created
- `EVT_INTENT_TRIGGERED` — intent converted to order
- `EVT_FILL` — order filled
- `EVT_TRADE_MODIFIED` — SL/TP/BE state change
- `EVT_TRADE_CLOSED` — position closed
- `EVT_RISK_ALERT` — risk limit violation
- `EVT_SAFETY_ALERT` — anomaly detected
- `EVT_KILL_SWITCH` — kill switch engaged

## Configuration

Canonical parameters live in [config/default.ini](config/default.ini). The MQL5
side parses this on `OnInit` via `Config.mqh`. The Python side parses the same
file via `filo_hft/config.py`. EA inputs (visible in MT5 dialog) serve as
overrides that win over the ini. Live overrides via
`config/live_overrides.json` are reloaded on a 5-second timer.

## Telemetry contract

Every decision is logged. Schema is defined in
[docs/telemetry-schema.md](docs/telemetry-schema.md). Logs are plain CSV,
append-only, one file per stream per trading day, written from the EA into
`telemetry/YYYYMMDD/`. Python tools tail these files.

The contract: **every signal fire and every signal block is explainable from
the telemetry alone**. If you cannot answer "why didn't the EA trade at 14:03?"
from the CSVs, that is a bug in the telemetry layer, not a feature request.

## Research-first mindset

The requirements doc contains many numbers (100-bar channel, 1.2 stdev, 2-pip
break-even, 0.8-pip BE buffer, etc.). **Every single one is a hypothesis, not
a fact.** The backtest harness exists so those numbers can be tuned against
real data, segmented by session/hour/regime, before going live.

See [docs/research-questions.md](docs/research-questions.md) for the full list
of questions that must be answered empirically.
