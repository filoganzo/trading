"""
Pull all stock symbols from MT5 terminal and categorize by region.
Run this with MT5 terminal open and logged into your broker account.

Usage:
    python stocks/scripts/list_mt5_stocks.py
    python stocks/scripts/list_mt5_stocks.py --output stocks/config/tickers.json
"""

import argparse
import io
import json
import sys
from pathlib import Path

# Force UTF-8 stdout so broker descriptions with non-ASCII don't crash on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    import MetaTrader5 as mt5
except ImportError:
    sys.exit("MetaTrader5 package not found. Run: pip install MetaTrader5")

# ---------------------------------------------------------------------------
# Region heuristics — based on broker suffix conventions and exchange codes
# These cover the most common MT5 broker naming patterns. Extend as needed.
# ---------------------------------------------------------------------------

# Exchange/suffix → region mapping (checked against symbol suffix first)
SUFFIX_REGION = {
    # US
    ".US": "US", ".NAS": "US", ".NYSE": "US", ".AMEX": "US",
    # UK
    ".UK": "Europe", ".LSE": "Europe",
    # Germany
    ".DE": "Europe", ".XETR": "Europe", ".FFM": "Europe",
    # France
    ".FR": "Europe", ".PAR": "Europe",
    # Netherlands
    ".NL": "Europe", ".AMS": "Europe",
    # Spain
    ".ES": "Europe", ".MC": "Europe",
    # Italy
    ".IT": "Europe", ".MIL": "Europe",
    # Switzerland
    ".CH": "Europe", ".SWX": "Europe",
    # Nordic
    ".SE": "Europe", ".NO": "Europe", ".DK": "Europe", ".FI": "Europe",
    # Japan
    ".JP": "Asia", ".TYO": "Asia", ".TSE": "Asia",
    # Hong Kong
    ".HK": "Asia", ".HKEX": "Asia",
    # China
    ".CN": "Asia", ".SHH": "Asia", ".SHZ": "Asia",
    # Australia
    ".AU": "Asia", ".ASX": "Asia",
    # Singapore
    ".SG": "Asia", ".SGX": "Asia",
    # South Korea
    ".KR": "Asia", ".KSE": "Asia",
    # India
    ".IN": "Asia", ".BSE": "Asia", ".NSE": "Asia",
    # Canada
    ".CA": "Other", ".TSX": "Other",
    # Brazil
    ".BR": "Other", ".BVSP": "Other",
}

# Path prefixes (some brokers put region in the group/path, e.g. "Stocks\US\AAPL")
PATH_REGION_KEYWORDS = {
    "US": "US", "NYSE": "US", "NASDAQ": "US", "NAS": "US", "AMEX": "US",
    "UK": "Europe", "LSE": "Europe",
    "GER": "Europe", "DEU": "Europe", "XETR": "Europe",
    "FRA": "Europe", "PAR": "Europe",
    "EUR": "Europe", "EU": "Europe",
    "JPN": "Asia", "JPY": "Asia", "JAPAN": "Asia",
    "HKG": "Asia", "HK": "Asia",
    "CHN": "Asia", "CHINA": "Asia",
    "AUS": "Asia", "ASX": "Asia",
    "SGP": "Asia", "SG": "Asia",
    "KOR": "Asia", "KRW": "Asia",
    "IND": "Asia", "NSE": "Asia",
    "CA": "Other", "CAN": "Other",
    "BR": "Other", "BRA": "Other",
}

# Well-known US tickers (fallback if broker uses no suffix)
KNOWN_US_TICKERS = {
    "AAPL","MSFT","GOOGL","GOOG","AMZN","META","NVDA","TSLA","BRK.B","BRK.A",
    "JPM","V","UNH","XOM","JNJ","WMT","MA","PG","HD","CVX","MRK","ABBV","LLY",
    "PEP","KO","AVGO","COST","TMO","MCD","ACN","DHR","ABT","NEE","TXN","LIN",
    "ADBE","NFLX","CRM","AMD","QCOM","PM","UPS","SBUX","GS","MS","BAC","C",
    "WFC","INTC","IBM","ORCL","CSCO","F","GM","BA","CAT","RTX","HON","MMM",
    "GE","T","VZ","DIS","CMCSA","PYPL","UBER","LYFT","SNAP","TWTR","SPOT",
    "SHOP","SQ","PLTR","RBLX","HOOD","COIN","RIVN","LCID","NIO","BABA","JD",
    "PDD","BIDU","NTES","SE","GRAB","DIDI","LI","XPEV","TME","BILI",
}


def classify_symbol(sym) -> str:
    """Return region string for an mt5.SymbolInfo object."""
    name: str = sym.name
    path: str = getattr(sym, "path", "") or ""

    # 1. Check suffix (most reliable)
    name_upper = name.upper()
    for suffix, region in SUFFIX_REGION.items():
        if name_upper.endswith(suffix.upper()):
            return region

    # 2. Check broker path / group
    path_upper = path.upper()
    for keyword, region in PATH_REGION_KEYWORDS.items():
        if keyword in path_upper:
            return region

    # 3. Known US tickers (no suffix)
    base = name_upper.split(".")[0]
    if base in KNOWN_US_TICKERS:
        return "US"

    return "Unknown"


def is_stock(sym) -> bool:
    """Filter to equity/stock instruments only, excluding FX, crypto, indices, commodities."""
    # MT5 calc_mode 0 = Forex, 5 = Stocks (CFD), 6 = Futures, etc.
    # Some brokers also flag via path containing "Stocks" or "Equities"
    name: str = sym.name.upper()
    path: str = (getattr(sym, "path", "") or "").upper()

    # Explicit path keywords for stocks
    if any(k in path for k in ("STOCK", "EQUIT", "SHARE")):
        return True

    # Exclude obvious non-stocks by name pattern
    non_stock_prefixes = (
        "XAU", "XAG", "XPT", "XPD",        # metals
        "BTC", "ETH", "LTC", "XRP", "BNB",  # crypto
        "US30", "US500", "NAS100", "DE40",   # indices
        "USOIL", "UKOIL", "NGAS",            # energy
        "EUR", "GBP", "USD", "JPY", "AUD",   # FX majors
    )
    if any(name.startswith(p) for p in non_stock_prefixes):
        return False

    # calc_mode 5 is the standard MT5 enum for CFD on stocks
    if getattr(sym, "trade_calc_mode", -1) == 5:
        return True

    # Last resort: if the symbol has a stock-like suffix from our map
    for suffix in SUFFIX_REGION:
        if name.endswith(suffix.upper()):
            return True

    return False


def fetch_stocks(credentials: dict | None = None) -> dict[str, list[dict]]:
    """Connect to MT5, pull all stock symbols, return dict keyed by region."""
    init_kwargs: dict = {}
    if credentials:
        init_kwargs = {k: v for k, v in credentials.items() if v}

    if not mt5.initialize(**init_kwargs):
        err = mt5.last_error()
        sys.exit(f"MT5 initialize() failed: {err}")

    print(f"MT5 connected — terminal: {mt5.terminal_info().name}")
    print(f"Account: {mt5.account_info().login}  |  Server: {mt5.account_info().server}")

    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        mt5.shutdown()
        sys.exit("mt5.symbols_get() returned None — no symbols available.")

    print(f"Total symbols in terminal: {len(all_symbols)}")

    stocks = [s for s in all_symbols if is_stock(s)]
    print(f"Stock instruments identified: {len(stocks)}")

    by_region: dict[str, list[dict]] = {
        "US": [], "Europe": [], "Asia": [], "Other": [], "Unknown": []
    }

    for sym in stocks:
        region = classify_symbol(sym)
        by_region[region].append({
            "name": sym.name,
            "description": sym.description,
            "currency_base": sym.currency_base,
            "currency_profit": sym.currency_profit,
            "digits": sym.digits,
            "spread": sym.spread,
            "trade_contract_size": sym.trade_contract_size,
            "path": getattr(sym, "path", ""),
        })

    # Sort each region alphabetically
    for region in by_region:
        by_region[region].sort(key=lambda x: x["name"])

    mt5.shutdown()
    return by_region


def print_report(by_region: dict[str, list[dict]]) -> None:
    total = sum(len(v) for v in by_region.values())
    print(f"\n{'='*60}")
    print(f"  MT5 STOCK TICKER INVENTORY  ({total} instruments)")
    print(f"{'='*60}\n")

    for region in ("US", "Europe", "Asia", "Other", "Unknown"):
        tickers = by_region.get(region, [])
        if not tickers:
            continue
        print(f"--- {region} ({len(tickers)}) ---")
        for t in tickers:
            desc = f"  [{t['description']}]" if t["description"] else ""
            print(f"  {t['name']:<20}{desc}")
        print()


def main():
    parser = argparse.ArgumentParser(description="List MT5 stock tickers by region")
    parser.add_argument("--output", default="stocks/config/tickers.json",
                        help="Path to write JSON output (default: stocks/config/tickers.json)")
    parser.add_argument("--login", type=int, default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--server", default=None)
    args = parser.parse_args()

    creds = {"login": args.login, "password": args.password, "server": args.server}
    creds = {k: v for k, v in creds.items() if v is not None}

    by_region = fetch_stocks(creds or None)
    print_report(by_region)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        region: {
            "count": len(tickers),
            "tickers": [t["name"] for t in tickers],
            "details": tickers,
        }
        for region, tickers in by_region.items()
        if tickers
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
