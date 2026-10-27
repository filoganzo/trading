# Telemetry schema

All live EA output is written as plain CSV into
`telemetry/YYYYMMDD/<stream>.csv`, one file per stream per UTC trading day.
Files are append-only with a fixed header written on first open. Python
tooling reads these files (tailing for live, full-read for research).

The guiding contract: **every signal fire and every signal block is
explainable from the telemetry alone.**

## Streams

### `features.csv`

Periodic snapshot of the feature state (written on a gated interval to avoid
flooding, default ~250 ms).

| column             | type    | meaning                                      |
|--------------------|---------|----------------------------------------------|
| ts_ms              | int64   | epoch milliseconds UTC                       |
| bid                | double  | best bid                                     |
| ask                | double  | best ask                                     |
| spread_pips        | double  | (ask - bid) in pips                          |
| mid                | double  | (bid + ask)/2                                |
| chan_mid           | double  | regression channel centerline                |
| chan_upper         | double  | centerline + stdev_mult * sigma              |
| chan_lower         | double  | centerline - stdev_mult * sigma              |
| chan_pos_norm      | double  | normalized position inside channel [-1.5..1.5]|
| chan_slope         | double  | slope in pips/bar                            |
| trend_state        | string  | `up` / `down` / `flat`                       |
| velocity           | double  | pips per second (rolling)                    |
| burst_count_up     | int     | bullish bursts in window                     |
| burst_count_dn     | int     | bearish bursts in window                     |
| corr_xagusd        | double  | directional agreement [-1..1]                |
| corr_dxy           | double  | directional agreement [-1..1] (raw)          |
| session            | string  | current session name                         |

### `keylevels.csv`

Written on every KL refresh (new bar on any tracked TF).

| column        | type    | meaning                                         |
|---------------|---------|-------------------------------------------------|
| ts_ms         | int64   |                                                 |
| event         | string  | `add` / `remove` / `merge` / `invalidate`       |
| zone_id       | string  | stable id within session                        |
| center        | double  | zone center price                               |
| half_width    | double  | half-width in points                            |
| origin_tf     | string  | `D1`/`H4`/`H1`/`M30`/`M15`/`M5`                  |
| weight        | double  | timeframe weight                                |
| role          | string  | `high` / `low` / `merged`                       |
| age_bars      | int     | bars since formation on origin_tf               |
| touch_count   | int     | optional — times price has touched the zone    |

### `setups.csv`

Every candidate setup: accepted, gated, or rejected.

| column           | type   | meaning                                         |
|------------------|--------|-------------------------------------------------|
| ts_ms            | int64  |                                                 |
| family           | string | `breakout`/`retest`/`rejection`/`fakeout`/`failed_auction`/`engulfing_accel`/`mean_reversion` |
| direction        | string | `long` / `short`                                |
| kl_zone_id       | string | nearest relevant KL                             |
| kl_weight        | double |                                                 |
| kl_distance_pips | double |                                                 |
| intrinsic_score  | double | deterministic score [0..1]                      |
| ml_score         | double | if enabled, else -1                             |
| decision         | string | `accepted`/`blocked`                            |
| block_reason     | string | human-readable reason if blocked                |
| chan_pos_norm    | double | snapshot at decision time                       |
| spread_pips      | double | snapshot                                         |
| velocity         | double | snapshot                                         |
| corr_xagusd      | double | snapshot                                         |
| corr_dxy         | double | snapshot                                         |

### `intents.csv`

Execution intents created, triggered, or expired.

| column        | type   | meaning                                   |
|---------------|--------|-------------------------------------------|
| ts_ms         | int64  |                                           |
| intent_id     | string | monotonic within session                  |
| family        | string |                                           |
| direction     | string |                                           |
| state         | string | `created`/`triggered`/`expired`/`cancelled`|
| trigger_price | double |                                           |
| hidden_sl     | double |                                           |
| hidden_tp     | double |                                           |
| ttl_ms        | int    |                                           |
| setup_score   | double |                                           |

### `fills.csv`

| column         | type   | meaning                                  |
|----------------|--------|------------------------------------------|
| ts_ms          | int64  |                                          |
| intent_id      | string |                                          |
| ticket         | int64  | MT5 ticket                               |
| direction      | string |                                          |
| requested_price| double |                                          |
| actual_price   | double |                                          |
| slippage_pips  | double | signed                                   |
| lots           | double |                                          |
| latency_ms     | double | OrderSend RTT                            |

### `trades.csv`

Lifecycle entries for every open trade.

| column        | type   | meaning                                   |
|---------------|--------|-------------------------------------------|
| ts_ms         | int64  |                                           |
| ticket        | int64  |                                           |
| event         | string | `open`/`be`/`trail`/`close`               |
| direction     | string |                                           |
| price         | double | price at event                            |
| hidden_sl     | double |                                           |
| hidden_tp     | double |                                           |
| unrealized_pips| double|                                           |
| realized_usd  | double | on close                                  |

### `risk.csv`

Risk limit activity — every breach, every kill-switch trip.

| column      | type   | meaning                                    |
|-------------|--------|--------------------------------------------|
| ts_ms       | int64  |                                            |
| limit       | string | name of the limit                          |
| value       | double | current value                              |
| threshold   | double |                                            |
| action      | string | `warn`/`halt`/`flatten`                    |

### `safety.csv`

| column      | type   | meaning                                    |
|-------------|--------|--------------------------------------------|
| ts_ms       | int64  |                                            |
| category    | string | `spread`/`stall`/`latency`/`slippage`/`data_gap` |
| value       | double |                                            |
| threshold   | double |                                            |
| action      | string | `warn`/`halt`                              |

### `latency.csv`

Per-stage microbenchmarks written on a 1s interval for load, plus every
outlier.

| column          | type  | meaning                            |
|-----------------|-------|------------------------------------|
| ts_ms           | int64 |                                    |
| stage           | string| `tick_handler`/`features`/`kl`/`setups`/`order_send`/`management` |
| elapsed_ms      | double|                                    |
| p50_ms          | double| rolling                            |
| p95_ms          | double| rolling                            |
| p99_ms          | double| rolling                            |

### `errors.csv`

| column      | type   |
|-------------|--------|
| ts_ms       | int64  |
| severity    | string |
| component   | string |
| code        | int    |
| message     | string |

## Day rollover

At UTC midnight, files close and a new `telemetry/YYYYMMDD/` directory is
created. The EA opens new handles lazily on first write of the new day.

## Ordering and ids

Within a file, rows are strictly in `ts_ms` order (the single-writer EA
guarantees this). `intent_id` and `zone_id` are monotonic strings scoped to
a session — Python tooling joins streams on these plus `ts_ms`.
