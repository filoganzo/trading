"""
stocks/backtest/runner.py

Master backtest runner.

Stages:
  0. Extend MT5 cache to 3yr (main process, before fork)
  1. Load all bars + compute ATR
  2. Pre-compute LRC signals for each (symbol, lrc_length)
  3. For each entry_config, build a CandidateFrame (parallelised)
  4. For each (variant × max_concurrent), run portfolio_sim (parallelised)
  5. Aggregate metrics + save CSVs
  6. Auto-run robustness on top-5 variants

Usage:
    cd "FINAL HFT"
    python -m stocks.backtest.runner
    python -m stocks.backtest.runner --skip-extend --top-n 10 --workers 8
"""

import argparse
import io
import logging
import multiprocessing as mp
import os
import sys
import warnings
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            str(__import__("pathlib").Path(__file__).parent.parent / "data" / "backtest_run.log"),
            encoding="utf-8"
        ),
    ],
)
log = logging.getLogger(__name__)

from stocks.backtest.config import (
    RESULTS_DIR, MIN_BARS, MAX_CONCURRENTS, NOTIONAL,
    LRC_LENGTHS, SIGMA_MULTS, MIN_SCORES, TOUCH_PCTS, VEL_REQUIRED,
    SUFFIX_BENCHMARK, ALL_BENCHMARKS,
)
from stocks.backtest.data.universe import build_universe
from stocks.backtest.data.extender import extend_cache
from stocks.backtest.data.loader import load_all
from stocks.backtest.signals.lrc_signal import compute_lrc_signals
from stocks.backtest.signals.entry_signal import compute_entry_signals
from stocks.backtest.simulation.param_grid import TRADE_VARIANTS, ENTRY_GRID
from stocks.backtest.simulation.portfolio_sim import run_portfolio_sim
from stocks.backtest.reporting.metrics import compute_trade_metrics, compute_portfolio_metrics
from stocks.backtest.reporting.report_builder import save_results, print_summary, top_variants, best_by_segment
from stocks.backtest.robustness import run_robustness, print_robustness_summary


# =============================================================================
# Step 2: Pre-compute LRC signal frames for all (symbol, lrc_length)
# =============================================================================

def build_lrc_cache(
    all_bars: Dict[str, pd.DataFrame]
) -> Dict[Tuple[str, int], pd.DataFrame]:
    """
    Returns dict keyed by (symbol, lrc_length) → SignalFrame.
    Only 3 LRC lengths so each symbol computed 3× total.
    """
    lrc_cache: Dict[Tuple[str, int], pd.DataFrame] = {}
    for sym, df in all_bars.items():
        for llen in LRC_LENGTHS:
            sf = compute_lrc_signals(df, llen, sigma_mult=1.0)   # raw, sigma applied per entry_config
            lrc_cache[(sym, llen)] = sf
    log.info(f"LRC cache built: {len(lrc_cache)} (symbol×length) frames")
    return lrc_cache


# =============================================================================
# Step 3: Build CandidateFrame for one entry_config
# =============================================================================

def build_candidates(
    entry_cfg: Dict[str, Any],
    all_bars: Dict[str, pd.DataFrame],
    lrc_cache: Dict[Tuple[str, int], pd.DataFrame],
    sym_meta: pd.DataFrame,
    bench_bars: Dict[str, pd.DataFrame],
    entry_signal_cache: Optional[Dict] = None,
) -> pd.DataFrame:
    llen     = entry_cfg["lrc_length"]
    sigma    = entry_cfg["sigma_mult"]
    min_sc   = entry_cfg["min_score"]
    touch    = entry_cfg["touch_pct"]
    vel_req  = entry_cfg["vel_required"]

    meta_idx = sym_meta.set_index("symbol")
    rows: List[Dict] = []

    for sym, df in all_bars.items():
        if sym in ALL_BENCHMARKS:
            continue
        sf = lrc_cache.get((sym, llen))
        if sf is None:
            continue

        # Apply sigma threshold
        pos_mask = (
            ((sf["dev_sigma"] >  sigma) & (sf["position"] == +1)) |
            ((sf["dev_sigma"] < -sigma) & (sf["position"] == -1))
        )
        if not pos_mask.any():
            continue

        # Entry signal enrichment — use pre-computed cache when available
        if entry_signal_cache is not None:
            ef = entry_signal_cache.get((sym, touch, vel_req))
            if ef is None:
                continue
        else:
            ef = compute_entry_signals(df, touch, vel_req)

        # Benchmark for beta/rel-perf (simplified: not recomputed in BT, just for tagging)
        sym_info  = meta_idx.loc[sym] if sym in meta_idx.index else None
        exchange  = sym_info["exchange"] if sym_info is not None else ""
        sector    = sym_info["sector"]   if sym_info is not None else "Other"
        benchmark = sym_info["benchmark"] if sym_info is not None else "US500"

        # Combined score = mean of lrc_score + level_score (both 0-100 range)
        lrc_score     = sf["lrc_score"].clip(0, 100)
        level_score   = ef["level_score"].clip(0, 100)
        combined_score = (lrc_score + level_score) / 2.0

        # Signal bars: beyond band AND valid entry
        signal_mask = pos_mask & ef["valid"] & (combined_score >= min_sc)
        signal_dates = df.index[signal_mask]
        if len(signal_dates) == 0:
            continue

        closes  = df["Close"]
        atrs    = df["atr"]
        opens   = df["Open"]

        for sig_date in signal_dates:
            i = df.index.get_loc(sig_date)
            if i + 1 >= len(df):
                continue  # no next bar — skip (no look-ahead)

            next_date = df.index[i + 1]
            ep        = float(opens.iloc[i + 1])   # entry = next open
            if ep <= 0 or np.isnan(ep):
                continue

            direction = +1 if sf["position"].iloc[i] == +1 else -1   # +1=short, -1=long
            atr_val   = float(atrs.iloc[i])
            lrc_mid   = float(sf["lrc_mid"].iloc[i])
            sl_raw    = float(ef["level_price"].iloc[i])   # key level price (before ATR buffer)

            if np.isnan(sl_raw) or sl_raw <= 0:
                sl_raw = ep * (1 - direction * 0.02)   # fallback 2% from entry

            rows.append({
                "date":          next_date,       # date position opens
                "signal_date":   sig_date,
                "symbol":        sym,
                "exchange":      exchange,
                "sector":        sector,
                "benchmark":     benchmark,
                "direction":     direction,
                "entry_price":   ep,
                "sl_raw":        sl_raw,
                "atr_at_entry":  atr_val,
                "lrc_mid":       lrc_mid,
                "combined_score": float(combined_score.iloc[i]),
                "dev_sigma":     float(sf["dev_sigma"].iloc[i]),
                "entry_id":      entry_cfg["entry_id"],
            })

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    log.debug(f"  entry_id={entry_cfg['entry_id']}: {len(df_out)} candidates")
    return df_out


# =============================================================================
# Step 4: Worker — runs all 50 variants × all max_concurrent for one CandidateFrame
# =============================================================================

def _worker_task(args) -> List[Dict]:
    """
    args = (entry_cfg, candidates_parquet_path, all_bars_path, variant, max_concurrent)
    Returns list of metric dicts.
    """
    entry_cfg, cand_path, bars_path, variant, max_concurrent = args
    try:
        candidates = pd.read_parquet(cand_path)
        all_bars   = pd.read_pickle(bars_path)   # pre-serialised dict

        if candidates.empty:
            return []

        trades, equity = run_portfolio_sim(candidates, all_bars, variant, max_concurrent)

        results = []
        # All-universe metrics
        if not trades.empty:
            # Merge exchange/sector/vol_tier from candidates (not in _close_record output)
            meta_cols = ["symbol", "exchange", "sector"] + (["vol_tier"] if "vol_tier" in candidates.columns else [])
            sym_meta = candidates[meta_cols].drop_duplicates("symbol")
            trades = trades.merge(sym_meta, on="symbol", how="left")

            results.append({**compute_trade_metrics(trades, "all", "ALL", variant["variant_id"]), "entry_id": entry_cfg["entry_id"]})
            # Per exchange
            if "exchange" in trades.columns:
                for exch, grp in trades.groupby("exchange"):
                    results.append(compute_trade_metrics(grp, "exchange", str(exch), variant["variant_id"]))
            # Per sector
            if "sector" in trades.columns:
                for sect, grp in trades.groupby("sector"):
                    results.append(compute_trade_metrics(grp, "sector", str(sect), variant["variant_id"]))
            # Per vol_tier
            if "vol_tier" in trades.columns:
                for vt, grp in trades.groupby("vol_tier"):
                    if str(vt).strip() and str(vt) != "unknown":
                        results.append(compute_trade_metrics(grp, "vol_tier", str(vt), variant["variant_id"]))

        # Portfolio metrics
        pm = compute_portfolio_metrics(equity, variant["variant_id"], max_concurrent)
        if pm:
            results.append({**pm, "entry_id": entry_cfg["entry_id"]})

        return results
    except Exception as e:
        return [{"error": str(e), "variant_id": variant["variant_id"]}]


# =============================================================================
# Main orchestrator
# =============================================================================

def run(args) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- MT5 ---
    if not args.skip_extend:
        log.info("Stage 0: Extending MT5 cache to 3yr...")
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                log.error("MT5 init failed — run with --skip-extend if bars already cached")
                sys.exit(1)
        except ImportError:
            log.error("MetaTrader5 not installed")
            sys.exit(1)
        universe_df = build_universe()
        extend_cache(universe_df["symbol"].tolist(), ALL_BENCHMARKS)
        mt5.shutdown()
    else:
        log.info("Stage 0: Skipping MT5 extend (--skip-extend)")
        universe_df = build_universe()

    # --- Load bars ---
    log.info("Stage 1: Loading all 3yr bars...")
    all_syms = universe_df["symbol"].tolist() + ALL_BENCHMARKS
    all_bars = load_all(list(dict.fromkeys(all_syms)))
    bench_bars = {b: all_bars[b] for b in ALL_BENCHMARKS if b in all_bars}
    stock_bars = {s: all_bars[s] for s in all_bars if s not in ALL_BENCHMARKS}
    log.info(f"  {len(stock_bars)} stocks | {len(bench_bars)} benchmarks")

    # --- LRC cache ---
    log.info("Stage 2: Pre-computing LRC signals...")
    lrc_cache = build_lrc_cache({**stock_bars, **bench_bars})

    # --- Build CandidateFrames ---
    log.info("Stage 3: Building CandidateFrames for all entry configs...")
    import pickle
    tmp_dir = RESULTS_DIR / "_tmp"
    tmp_dir.mkdir(exist_ok=True)

    # Serialise all_bars once for workers
    bars_pkl = tmp_dir / "all_bars.pkl"
    if not bars_pkl.exists():
        with open(bars_pkl, "wb") as f:
            pickle.dump(all_bars, f)

    # Pre-compute entry signals for the 4 unique (touch_pct × vel_required) combos — 24× speedup
    # Skip if all parquets already exist
    from stocks.backtest.config import TOUCH_PCTS, VEL_REQUIRED
    from itertools import product as _iproduct
    entry_signal_cache: Dict = {}
    needs_build = any(
        not (tmp_dir / f"cands_{ecfg['entry_id']}.parquet").exists()
        for ecfg in ENTRY_GRID
    )
    if needs_build:
        unique_combos = list(_iproduct(TOUCH_PCTS, VEL_REQUIRED))
        log.info(f"  Pre-computing entry signals: {len(unique_combos)} combos × {len(stock_bars)} symbols...")
        for ci, (touch, vel) in enumerate(unique_combos):
            log.info(f"    combo {ci+1}/{len(unique_combos)}: touch={touch} vel_req={vel}")
            for sym, df in stock_bars.items():
                entry_signal_cache[(sym, touch, vel)] = compute_entry_signals(df, touch, vel)
        log.info("  Entry signal cache ready.")
    else:
        log.info("  All candidate parquets cached — skipping entry signal pre-computation.")

    cand_paths: Dict[int, Any] = {}
    for ecfg in ENTRY_GRID:
        eid   = ecfg["entry_id"]
        path  = tmp_dir / f"cands_{eid}.parquet"
        if not path.exists():
            cdf = build_candidates(ecfg, stock_bars, lrc_cache, universe_df, bench_bars, entry_signal_cache)
            if not cdf.empty:
                cdf.to_parquet(path)
            else:
                pd.DataFrame().to_parquet(path)
        cand_paths[eid] = path
        if eid % 12 == 0:
            log.info(f"  entry config {eid}/{len(ENTRY_GRID)}")

    log.info("Stage 3 complete.")

    # --- Liquid filter (optional) ---
    if getattr(args, "liquid", False):
        log.info("Liquid mode: filtering candidates to top-25% by dollar volume...")
        liq_scores = {}
        for sym, df in stock_bars.items():
            if "Volume" not in df.columns or df.empty:
                continue
            dvol = (df["Close"] * df["Volume"]).replace(0, np.nan).dropna()
            if len(dvol) >= 200:
                liq_scores[sym] = float(dvol.median())
        liq_series   = pd.Series(liq_scores)
        liq_threshold = float(liq_series.quantile(0.75))
        liquid_syms  = set(liq_series[liq_series >= liq_threshold].index)
        log.info(f"  Liquid universe: {len(liquid_syms)} stocks (threshold={liq_threshold:,.0f})")

        tmp_dir_liq = RESULTS_DIR / "_tmp_liq"
        tmp_dir_liq.mkdir(exist_ok=True)

        # Filtered bars pickle (liquid stocks + benchmarks)
        liq_bars_pkl = tmp_dir_liq / "all_bars.pkl"
        if not liq_bars_pkl.exists():
            liq_bars = {s: df for s, df in all_bars.items()
                        if s in liquid_syms or s in ALL_BENCHMARKS}
            import pickle as _pkl
            with open(liq_bars_pkl, "wb") as f:
                _pkl.dump(liq_bars, f)
            log.info(f"  Saved liquid all_bars.pkl ({len(liq_bars)} symbols)")

        # Filter each candidate parquet
        liq_cand_paths: Dict[int, Any] = {}
        for ecfg in ENTRY_GRID:
            eid = ecfg["entry_id"]
            src = tmp_dir / f"cands_{eid}.parquet"
            dst = tmp_dir_liq / f"cands_{eid}.parquet"
            if not dst.exists():
                if src.exists() and src.stat().st_size > 512:
                    cdf = pd.read_parquet(src)
                    if "symbol" in cdf.columns and not cdf.empty:
                        cdf_liq = cdf[cdf["symbol"].isin(liquid_syms)]
                    else:
                        cdf_liq = pd.DataFrame()
                    (cdf_liq if not cdf_liq.empty else pd.DataFrame()).to_parquet(dst)
                else:
                    pd.DataFrame().to_parquet(dst)
            liq_cand_paths[eid] = dst

        cand_paths = liq_cand_paths
        bars_pkl   = liq_bars_pkl
        log.info("  Liquid candidate parquets ready.")

    # --- US-only + Vol Tier filter ---
    if getattr(args, "us_only", False):
        US_EXCHANGES = {"NYSE", "NAS"}
        log.info("US-only mode: filtering to NYSE + NAS + computing vol tiers...")

        us_syms = set(
            universe_df[universe_df["exchange"].isin(US_EXCHANGES)]["symbol"]
        ) & set(stock_bars.keys())
        if getattr(args, "liquid", False):
            us_syms &= liquid_syms
        log.info(f"  US liquid universe: {len(us_syms)} stocks")

        # ATR% vol tiers (14-day ATR / Close, median per symbol)
        atr_pct: Dict[str, float] = {}
        for sym in us_syms:
            df = stock_bars.get(sym)
            if df is None or len(df) < 50 or "High" not in df.columns:
                continue
            atr14 = (df["High"] - df["Low"]).rolling(14).mean()
            med = (atr14 / df["Close"]).dropna().median()
            if not np.isnan(med):
                atr_pct[sym] = float(med)

        atr_s = pd.Series(atr_pct)
        low_thresh  = float(atr_s.quantile(0.33))
        high_thresh = float(atr_s.quantile(0.67))
        log.info(f"  Vol tiers — low: <{low_thresh:.3%}, mid: <{high_thresh:.3%}, high: above")

        def _vol_tier(sym: str) -> str:
            v = atr_pct.get(sym)
            if v is None:   return "unknown"
            if v <= low_thresh:  return "low_vol"
            if v <= high_thresh: return "mid_vol"
            return "high_vol"

        tmp_dir_us = RESULTS_DIR / "_tmp_us"
        tmp_dir_us.mkdir(exist_ok=True)

        us_bars_pkl = tmp_dir_us / "all_bars.pkl"
        if not us_bars_pkl.exists():
            import pickle as _pkl
            us_bars = {s: df for s, df in all_bars.items()
                       if s in us_syms or s in ALL_BENCHMARKS}
            with open(us_bars_pkl, "wb") as f:
                _pkl.dump(us_bars, f)
            log.info(f"  Saved US all_bars.pkl ({len(us_bars)} symbols)")

        src_dir = cand_paths  # already liquid-filtered if --liquid was set
        us_cand_paths: Dict[int, Any] = {}
        for ecfg in ENTRY_GRID:
            eid = ecfg["entry_id"]
            src = src_dir[eid]
            dst = tmp_dir_us / f"cands_{eid}.parquet"
            if not dst.exists():
                if src.exists() and src.stat().st_size > 512:
                    cdf = pd.read_parquet(src)
                    if "symbol" in cdf.columns and not cdf.empty:
                        cdf_us = cdf[cdf["symbol"].isin(us_syms)].copy()
                        cdf_us["vol_tier"] = cdf_us["symbol"].map(_vol_tier)
                    else:
                        cdf_us = pd.DataFrame()
                    (cdf_us if not cdf_us.empty else pd.DataFrame()).to_parquet(dst)
                else:
                    pd.DataFrame().to_parquet(dst)
            us_cand_paths[eid] = dst

        cand_paths = us_cand_paths
        bars_pkl   = us_bars_pkl
        log.info("  US + vol-tier candidate parquets ready.")

    # --- Sweep ---
    log.info(f"Stage 4: Sweep — {len(ENTRY_GRID)} entry × {len(TRADE_VARIANTS)} variants × {len(MAX_CONCURRENTS)} concurrencies")
    tasks = [
        (ecfg, cand_paths[ecfg["entry_id"]], bars_pkl, variant, mc)
        for ecfg in ENTRY_GRID
        for variant in TRADE_VARIANTS
        for mc in MAX_CONCURRENTS
        if cand_paths[ecfg["entry_id"]].stat().st_size > 512
    ]
    log.info(f"  Total tasks: {len(tasks)}")

    n_workers = args.workers or max(1, os.cpu_count() - 1)
    log.info(f"  Workers: {n_workers}")

    all_metrics      = []
    portfolio_metrics = []
    entry_metrics    = []
    done = 0

    with mp.Pool(processes=n_workers) as pool:
        for result_list in pool.imap_unordered(_worker_task, tasks, chunksize=4):
            done += 1
            if done % 500 == 0:
                log.info(f"  Tasks done: {done}/{len(tasks)}")
            for r in result_list:
                if "error" in r:
                    continue
                if r.get("segment_type") == "portfolio":
                    portfolio_metrics.append(r)
                elif r.get("segment_type") == "all":
                    all_metrics.append(r)
                    # Entry-level metric = same row tagged with entry_id
                    em = {**r, "entry_id": r.get("entry_id", -1)}
                    entry_metrics.append(em)
                else:
                    all_metrics.append(r)

    log.info("Stage 4 complete.")

    # --- Save results ---
    log.info("Stage 5: Saving results...")
    trade_df = pd.DataFrame(all_metrics)
    port_df  = pd.DataFrame(portfolio_metrics)
    entry_df = pd.DataFrame(entry_metrics)
    liq_tag = "_liq" if getattr(args, "liquid", False) else ""
    us_tag  = "_us"  if getattr(args, "us_only", False) else ""
    tag = datetime.utcnow().strftime("%Y%m%d_%H%M") + liq_tag + us_tag
    save_results(all_metrics, portfolio_metrics, entry_metrics, run_tag=tag)
    print_summary(trade_df, port_df)

    # Per-segment best variants
    seg_best = best_by_segment(trade_df)
    if not seg_best.empty:
        seg_best.to_csv(RESULTS_DIR / f"best_by_segment_{tag}.csv", index=False)
        print("\n" + "=" * 70)
        print("  BEST VARIANT PER EXCHANGE")
        print("=" * 70)
        exch = seg_best[seg_best["seg_type"] == "exchange"][
            ["segment_value", "variant_name", "total_trades", "win_rate",
             "avg_R", "sharpe_R", "profit_factor", "composite"]
        ].sort_values("sharpe_R", ascending=False)
        print(exch.to_string(index=False))
        print("\n" + "=" * 70)
        print("  BEST VARIANT PER SECTOR")
        print("=" * 70)
        sect = seg_best[seg_best["seg_type"] == "sector"][
            ["segment_value", "variant_name", "total_trades", "win_rate",
             "avg_R", "sharpe_R", "profit_factor", "composite"]
        ].sort_values("sharpe_R", ascending=False)
        print(sect.to_string(index=False))

    # --- Robustness ---
    log.info("Stage 6: Robustness tests on top-5 variants...")
    # Use the highest-signal entry config for robustness
    best_eid = 0
    if not entry_df.empty and "entry_id" in entry_df.columns:
        top_e = entry_df[entry_df.get("segment_value", pd.Series()) == "ALL"] if "segment_value" in entry_df else entry_df
        if not top_e.empty and "total_trades" in top_e.columns:
            best_eid = int(top_e.nlargest(1, "total_trades")["entry_id"].iloc[0])

    best_cands_path = cand_paths.get(best_eid)
    if best_cands_path and best_cands_path.exists():
        best_cands = pd.read_parquet(best_cands_path)
        rob = run_robustness(
            trade_df, best_cands, all_bars,
            max_concurrent=5,
            top_n=args.top_n,
        )
        print_robustness_summary(rob)

        # Save robustness CSVs
        for name, df in rob.items():
            if not df.empty:
                df.to_csv(RESULTS_DIR / f"robustness_{name}_{tag}.csv", index=False)

    log.info("Backtest complete.")


def main():
    ap = argparse.ArgumentParser(description="Stocks backtest sweep")
    ap.add_argument("--skip-extend", action="store_true", help="Skip MT5 3yr cache extension")
    ap.add_argument("--workers",     type=int, default=None, help="Parallel workers (default: cpu_count-1)")
    ap.add_argument("--top-n",       type=int, default=5,    help="Variants to robustness-test")
    ap.add_argument("--liquid",      action="store_true",    help="Filter to top-25% liquid stocks only")
    ap.add_argument("--us-only",     action="store_true",    help="Further filter to NYSE/NAS + compute vol tiers")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
