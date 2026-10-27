"""Central config for the stocks backtest sweep."""
from pathlib import Path

ROOT             = Path(__file__).parent.parent          # stocks/
CACHE_DIR        = ROOT / "data" / "cache"               # screener daily cache (2yr)
BT_CACHE_DIR     = ROOT / "data" / "bt_cache"            # 3yr daily bars for backtest
BT_CACHE_M15_DIR = ROOT / "data" / "bt_cache_m15"        # 3yr M15 bars for v5/v6
BT_CACHE_W1_DIR  = ROOT / "data" / "bt_cache_w1"         # weekly bars for HTF levels
RESULTS_DIR      = ROOT / "data" / "backtest_results"
TICKERS_JSON  = ROOT / "config" / "tickers.json"

HISTORY_DAYS  = 1095    # 3 years
ATR_PERIOD    = 14
MIN_BARS      = 170     # minimum bars for a symbol to enter backtest

# LRC param space
LRC_LENGTHS   = [80, 100, 120]
SIGMA_MULTS   = [1.0, 1.3, 1.5, 2.0]

# Entry filter param space
MIN_SCORES    = [40, 60]
TOUCH_PCTS    = [0.75, 1.5]        # % distance from key level
VEL_REQUIRED  = [False, True]

# Portfolio
MAX_CONCURRENTS = [3, 5, 10]
NOTIONAL        = 1_000.0          # USD allocated per full portfolio slot

# Suffix → benchmark index
SUFFIX_BENCHMARK = {
    "NYSE": "US500", "NAS": "US500", "LSE": "UK100",
    "ETR": "DE40", "PAR": "F40", "IT": "IT40",
    "MAD": "ES35", "AMS": "NETH25", "IE": "STOXX50", "SWX": "SWI20",
}
ALL_BENCHMARKS = list(dict.fromkeys(SUFFIX_BENCHMARK.values()))
