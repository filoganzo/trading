"""CLI: extract historical ticks and bars from MT5 into the data/ folder.

    python -m filo_hft.scripts.extract_history \
        --symbol XAUUSD --from 2024-01-01 --to 2024-06-30

Runs all configured symbols (primary + silver + dxy proxy) by default.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import click

from filo_hft.config import load_config
from filo_hft.data import mt5_export


@click.command()
@click.option("--symbol", "symbols", multiple=True, help="Override symbols (repeatable)")
@click.option("--from", "start", required=True, type=click.DateTime(["%Y-%m-%d"]))
@click.option("--to",   "end",   required=True, type=click.DateTime(["%Y-%m-%d"]))
@click.option("--config", "config_path", default="../config/default.ini",
              type=click.Path(exists=True))
@click.option("--credentials", "creds_path", default="../credentials.local.ini",
              type=click.Path(exists=True))
@click.option("--out", "out_dir", default="../data", type=click.Path())
@click.option("--what", default="ticks,bars",
              help="Comma-separated subset of {ticks,bars}")
def main(symbols, start, end, config_path, creds_path, out_dir, what):
    cfg = load_config(config_path)
    if not symbols:
        symbols = (cfg.symbols.primary, cfg.symbols.silver_proxy, cfg.symbols.dxy_proxy)

    creds = mt5_export.load_credentials(creds_path)
    mt5_export.initialize(creds)
    try:
        start_utc = start.replace(tzinfo=timezone.utc)
        end_utc   = end.replace(tzinfo=timezone.utc)
        for sym in symbols:
            print(f"[extract] {sym} {start_utc} → {end_utc}")
            mt5_export.extract_range(
                symbol=sym,
                start=start_utc,
                end=end_utc,
                out_root=Path(out_dir),
                what=tuple(w.strip() for w in what.split(",")),
            )
    finally:
        mt5_export.shutdown()


if __name__ == "__main__":
    main()
