# Open research questions

These are the questions the requirements doc flags as needing empirical
answers. The platform exists in large part to answer them. Each has a
suggested method and which harness component answers it.

## Channel model

**Q1. What is the optimal channel length around the 100-bar baseline?**
- Method: sweep `channel.length` across [40, 60, 80, 100, 120, 150, 200].
- Metric: net expectancy per setup family, segmented by session.
- Harness: `python/scripts/run_sweep.py --param channel.length`.

**Q2. Is `stdev_mult = 1.2` robust across sessions?**
- Method: sweep `channel.stdev_mult` across [0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75].
- Metric: extreme-block hit rate vs false-reject rate, per session.
- Harness: `run_sweep.py --param channel.stdev_mult`.

## Acceleration

**Q3. How many same-direction bursts best distinguish real acceleration from
liquidity grabs?**
- Method: sweep `acceleration.min_bursts` across [1, 2, 3] and
  `burst_sigma` across [1.8, 2.0, 2.2, 2.5, 3.0].
- Metric: precision/recall against hand-labeled burst events in training
  data; then downstream: setup success rate conditioned on burst count.
- Harness: `run_sweep.py --cross channel.stdev_mult,acceleration.min_bursts`.

**Q4. Is there a reliable way to detect liquidity grabs in MT5 tick data?**
- Method: define liquidity grab as a rapid wick that reverses within
  `liquidity_grab_reject_ms`. Backtest with and without the reject filter.
- Harness: A/B in the backtest runner.

## Failed auction

**Q5. What is the best failed-auction definition given the data we have?**
- Method: three candidate definitions (a) break of channel + rejection
  wick, (b) break of prior value area proxy + insufficient follow-through,
  (c) breakout that retraces >70% inside 3 bars. Backtest all three.
- Harness: implementations live in `Setups.mqh` under `DetectFailedAuction_*`
  stubs — enable one at a time via config.

## KL distance

**Q6. How far from a KL is still tradeable, by setup type?**
- Method: compute distance from nearest KL at intent creation; bucket
  outcomes by distance.
- Harness: `run_sweep.py --param keylevels.max_setup_distance_pips`.

## Exhaustion

**Q7. After how many candles / pips is continuation exhausted?**
- Method: sweep `setups.exhaustion_bars` and `setups.exhaustion_pips`.
- Metric: conditional win rate of continuation trades taken after N bars of
  directional movement.

## Session edges

**Q8. Which sessions favor breakout vs mean reversion?**
- Method: segment backtest results by session and setup family; compute
  expectancy per combination.
- Harness: `sweeps/segmentation.py` produces the session × family matrix.

## Hidden vs visible orders

**Q9. Does hidden target logic materially outperform broker-visible stops?**
- Method: backtest the same strategy twice, once with SL/TP submitted to
  broker, once with hidden SL/TP managed by EA. Compare slippage, fill
  quality, and adverse selection.
- Harness: flag `execution.hidden_exits` (to be added). Run two backtests.

## KL timeframe profitability

**Q10. Which combinations of KL timeframe and setup family are profitable
after costs?**
- Method: segment by `kl_timeframe × setup_family`.
- Harness: `sweeps/segmentation.py`.

## ML contribution

**Q11. Does the ML scorer add information beyond the deterministic score?**
- Method: compare backtest results with `setups.min_setup_score` gate alone
  vs deterministic gate blended with ML probability.
- Harness: `train_ml.py` to build the model, then `run_backtest.py --ml`.

## How to prioritize

Answer these in order: **Q1 → Q2 → Q6 → Q8 → Q3 → Q7 → Q5 → Q9 → Q10 → Q11 → Q4**.
The first four give you a robust channel model and a sense of where the edge
lives by session. The rest refines each component incrementally.

**Do not go live before Q1, Q2, Q6, and Q8 are answered.**
