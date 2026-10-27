# Latency budget

The requirements doc states a VPS target of ~0.65 ms and off-VPS of ~90 ms.
That single number is misleading unless we split it by stage. Here is where
each millisecond realistically lives.

## Stages of a tick → decision → order cycle

```
 T0: tick arrives at broker data center
      │
      │  (not our problem; bounded by broker infra)
      │
 T1: tick arrives at MT5 terminal on VPS
      │
      │  ~0.01-0.05 ms   terminal delivers to EA
      │
 T2: OnTick() handler entry
      │
      │  ~0.05-0.1 ms    Features.Update
      │  ~0.02-0.05 ms   KeyLevels lookup (zones indexed)
      │  ~0.02 ms        Correlation update (numeric only)
      │  ~0.10-0.20 ms   Setup scan (all enabled families)
      │  ~0.05 ms        Risk + safety checks
      │
 T3: decision made — either: skip / create intent / fire market order
      │
      │  ~0.02-0.05 ms   OrderSend assembly
      │
 T4: OrderSend() call made
      │
      │  1-5 ms          terminal → broker → ack RTT (this is the wall)
      │
 T5: fill ack received
```

## Budgets vs. realistic numbers

| Stage                               | Target  | Realistic | Notes                                |
|-------------------------------------|---------|-----------|--------------------------------------|
| Terminal → EA dispatch              | 0.05 ms | 0.05 ms   | out of our control                   |
| Features.Update                     | 0.05 ms | 0.10 ms   | ring buffer + rolling stats          |
| KeyLevels lookup                    | 0.02 ms | 0.05 ms   | binary search into sorted zone index |
| Correlation update                  | 0.02 ms | 0.02 ms   | single-symbol EWM update             |
| Setup scan                          | 0.10 ms | 0.20 ms   | early-out when no KL in range        |
| Risk/safety checks                  | 0.05 ms | 0.10 ms   | bucket counters, no I/O              |
| OrderSend assembly                  | 0.02 ms | 0.05 ms   | struct fill                          |
| **In-process decision path total**  | **0.31 ms** | **0.57 ms** | **under the 0.65 ms target** |
| OrderSend → broker ack              | —       | 1-5 ms    | bounded by terminal + broker         |
| **Total entry round trip**          | —       | **1.5-6 ms** | | 

## Why we don't hit 0.5 ms polling for trade management

The requirements ask for ~0.5 ms polling of hidden SL/TP. Inside the EA we
can loop at that cadence, but the MT5 API is not sub-millisecond for
`PositionModify` / `PositionClose` calls. Realistic modification RTT is the
same 1-5 ms window as entry. So the trade management engine runs its *logic*
every tick (effectively sub-ms when ticks are arriving fast), but its *broker
interaction* is gated to a 10 ms minimum interval by `management_poll_ms` in
config. When an exit is needed, it fires immediately; the gate is only on
re-evaluation, not on triggered actions.

## Off-VPS

Running off-VPS (~90 ms RTT) means the in-process decision path still takes
the same ~0.6 ms, but the `OnTick` wakeup itself lags by 90 ms. This is
catastrophic for breakout trades (price has already moved) and manageable
for mean reversion. The config provides `trade_offhours = false` and the
safety engine's `latency_alert_ms = 25` will halt the EA if it detects a
90 ms wakeup pattern.

## Where the "0.65 ms" number actually comes from

The in-process decision path. Not end-to-end. End-to-end is bounded by the
broker bridge and there is no MT5-level trick that makes it faster.

## Things we did to stay inside budget

- **No file I/O on the hot path.** Telemetry writes are batched and flushed
  on a 500 ms timer in a separate code path.
- **No `ObjectCreate` / UI calls on the hot path.** The live indicator is a
  separate `.mq5` file that subscribes to shared globals, not part of the EA.
- **No `iClose`/`iHigh` etc. on the hot path.** Features maintain their own
  ring buffers, updated incrementally from `OnTick`. The only `iClose` calls
  are in the KL engine, and those fire on bar close, not on every tick.
- **KL zones indexed by price.** Setup scanning does a binary search into a
  sorted array, not a linear scan.
- **Spread/slippage gates before any work.** If spread is out of range, we
  skip the rest of the handler.
