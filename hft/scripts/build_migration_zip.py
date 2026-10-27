"""Build a single-file migration archive for moving FiloHFT to a new PC.

Packs:
  project/        — full FINAL HFT repo (minus caches/venvs/logs)
  mt5_sandbox/    — IC Markets EU MT5 FiloHFT files (EA + includes + indicator + profiles)
  telemetry/      — every live telemetry day captured so far

Emits a MANIFEST.txt with file count and per-section sizes, plus a
README_MIGRATION.md with the new-PC checklist.

Writes to:
  path/to/output/filohft_migration_<YYYYMMDD>.zip
"""

from __future__ import annotations

import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("path/to/repo/hft")
MT5_ROOT = Path("C:/Users/<user>/AppData/Roaming/MetaQuotes/Terminal/<TERMINAL_ID>")
TELEMETRY_ROOT = Path("C:/Users/<user>/AppData/Roaming/MetaQuotes/Terminal/Common/Files/telemetry")

STAMP = datetime.now().strftime("%Y%m%d")
OUT_ZIP = Path(f"path/to/output/filohft_migration_{STAMP}.zip")

EXCLUDE_DIRS = {
    "__pycache__", ".venv", "venv", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "node_modules", ".git", ".idea", ".vscode",
    "logs", "Logs", "temp", ".DS_Store",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log"}

# MT5 files we actually want (anything else in the MT5 tree is broker/platform junk)
MT5_INCLUDE = [
    ("MQL5/Experts/FiloHFT.mq5", "mt5_sandbox/MQL5/Experts/FiloHFT.mq5"),
    ("MQL5/Experts/FiloHFT.ex5", "mt5_sandbox/MQL5/Experts/FiloHFT.ex5"),
    ("MQL5/Indicators/KeyLevelTracker.mq5", "mt5_sandbox/MQL5/Indicators/KeyLevelTracker.mq5"),
    ("MQL5/Indicators/KeyLevelTracker.ex5", "mt5_sandbox/MQL5/Indicators/KeyLevelTracker.ex5"),
    ("MQL5/Profiles/Tester/FiloHFT.set", "mt5_sandbox/MQL5/Profiles/Tester/FiloHFT.set"),
    ("MQL5/Profiles/Tester/FiloHFT.XAUUSD.M1.last_year.000.ini", "mt5_sandbox/MQL5/Profiles/Tester/FiloHFT.XAUUSD.M1.last_year.000.ini"),
    ("MQL5/Profiles/Templates/godopoli.tpl", "mt5_sandbox/MQL5/Profiles/Templates/godopoli.tpl"),
]
# Whole dirs from MT5
MT5_INCLUDE_DIRS = [
    ("MQL5/Include/FiloHFT", "mt5_sandbox/MQL5/Include/FiloHFT"),
    ("MQL5/Profiles/Charts/Default", "mt5_sandbox/MQL5/Profiles/Charts/Default"),
]

README_TEXT = f"""# FiloHFT migration archive — {STAMP}

Contents
--------
project/        Full FINAL HFT project tree (python/, mql5/ source, scripts, docs, configs, artifacts, data)
mt5_sandbox/    The IC Markets EU terminal's FiloHFT-relevant files:
                  - MQL5/Experts/FiloHFT.{{mq5,ex5}}
                  - MQL5/Include/FiloHFT/* (17 .mqh modules)
                  - MQL5/Indicators/KeyLevelTracker.{{mq5,ex5}}
                  - MQL5/Profiles/Tester/FiloHFT.set + last_year .ini
                  - MQL5/Profiles/Templates/godopoli.tpl
                  - MQL5/Profiles/Charts/Default/*
telemetry/      Every live telemetry day captured so far (per-day CSVs:
                setups, intents, fills, trades, tm_lab, shadow_trades, safety, errors, features)

Excluded on purpose
-------------------
- __pycache__, .venv, caches, logs, .git
- MT5 terminal binary (download fresh from IC Markets)
- Broker credentials / account login (enter on new PC)
- Other brokers' MT5 terminals (OANDA, FTMO, Tickmill, Vantage — not FiloHFT)

New-PC checklist
----------------
1. Install Python 3.11+ and MetaTrader 5 (from IC Markets EU).
2. Unzip this archive anywhere. Recommended target: `C:/Users/<you>/Desktop/Filo/Trading & personal finance/FINAL HFT/`
   (keeps hardcoded paths in scripts working).
3. Copy `mt5_sandbox/MQL5/*` into the new MT5 terminal's MQL5 folder
   (`C:/Users/<you>/AppData/Roaming/MetaQuotes/Terminal/<TERMINAL_ID>/MQL5/`).
4. Copy `telemetry/*` into the new MT5 Common/Files folder
   (`C:/Users/<you>/AppData/Roaming/MetaQuotes/Terminal/Common/Files/telemetry/`).
5. In the project, create the venv and install deps:
     python -m venv .venv
     .venv\\Scripts\\activate
     pip install -r python/filo_hft/requirements.txt  (if present, else reinstall: numpy pandas pyarrow matplotlib MetaTrader5 joblib scikit-learn)
6. Patch hardcoded paths if your Windows username changed. Files to scan:
     - python/filo_hft/scripts/day_recap_v2.py   (LIVE_ROOT, OUT_DIR)
     - python/filo_hft/scripts/mt5_history_dump.py (OUT_DIR)
     - python/filo_hft/scripts/day_recap.py       (LIVE_ROOT, OUT_DIR)
     - python/filo_hft/gui/dashboard_tk.py        (telemetry root)
7. Open MT5, log in to the FRESH account. Open XAUUSD M1 chart, attach FiloHFT EA,
   load FiloHFT.set, enable AutoTrading. Confirm a new telemetry/<YYYYMMDD>/ folder appears.
8. Disable Windows sleep/hibernate/auto-update restart. Add a UPS if possible.

Notes
-----
- credentials.local.ini in project/ contains OLD account settings — irrelevant for
  the fresh account. You'll enter new credentials in MT5 directly, not in this file.
- The old broker-side deal history is NOT in this zip (lives on IC Markets servers,
  tied to the old login). If you want it, run `mt5_history_dump.py` on the old PC
  BEFORE closing the account, and add the resulting parquet to the archive manually.
- Every source edit to MQL5/ must be duplicated into the MT5 sandbox copy, or
  the compiled .ex5 silently runs stale code. This is a known trap.

Archive built at: {datetime.now().isoformat()}
"""


def should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    return False


def add_tree(zf: zipfile.ZipFile, src_root: Path, arc_prefix: str, counter: dict) -> None:
    if not src_root.exists():
        print(f"  WARN: {src_root} does not exist, skipping")
        return
    for root, dirs, files in os.walk(src_root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        root_p = Path(root)
        for name in files:
            fp = root_p / name
            rel = fp.relative_to(src_root)
            if should_skip(rel):
                continue
            arc = f"{arc_prefix}/{rel.as_posix()}"
            try:
                zf.write(fp, arc)
                counter["files"] += 1
                counter["bytes"] += fp.stat().st_size
            except (OSError, PermissionError) as e:
                print(f"  SKIP (locked): {fp}: {e}")


def main() -> None:
    print(f"Building {OUT_ZIP}")
    section_stats: dict[str, dict] = {}

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # README + will-be-filled manifest placeholder
        zf.writestr("README_MIGRATION.md", README_TEXT)

        # 1. Project repo
        print("  [1/3] project/ ...")
        c = {"files": 0, "bytes": 0}
        add_tree(zf, PROJECT_ROOT, "project", c)
        section_stats["project"] = c
        print(f"        {c['files']} files, {c['bytes']/1e6:.1f} MB")

        # 2. MT5 FiloHFT assets
        print("  [2/3] mt5_sandbox/ ...")
        c = {"files": 0, "bytes": 0}
        for src_rel, arc in MT5_INCLUDE:
            src = MT5_ROOT / src_rel
            if src.exists():
                zf.write(src, arc)
                c["files"] += 1
                c["bytes"] += src.stat().st_size
            else:
                print(f"        MISSING: {src}")
        for src_rel, arc_prefix in MT5_INCLUDE_DIRS:
            add_tree(zf, MT5_ROOT / src_rel, arc_prefix, c)
        section_stats["mt5_sandbox"] = c
        print(f"        {c['files']} files, {c['bytes']/1e6:.1f} MB")

        # 3. Telemetry
        print("  [3/3] telemetry/ ...")
        c = {"files": 0, "bytes": 0}
        add_tree(zf, TELEMETRY_ROOT, "telemetry", c)
        section_stats["telemetry"] = c
        print(f"        {c['files']} files, {c['bytes']/1e6:.1f} MB")

        # Manifest
        manifest_lines = [
            f"FiloHFT migration archive",
            f"built: {datetime.now().isoformat()}",
            f"source host user: {os.environ.get('USERNAME', '?')}",
            f"",
            f"{'section':<20} {'files':>8} {'size_MB':>10}",
            f"{'-'*20} {'-'*8} {'-'*10}",
        ]
        total_files = 0
        total_bytes = 0
        for sec, c in section_stats.items():
            manifest_lines.append(f"{sec:<20} {c['files']:>8} {c['bytes']/1e6:>10.1f}")
            total_files += c["files"]
            total_bytes += c["bytes"]
        manifest_lines.append(f"{'-'*20} {'-'*8} {'-'*10}")
        manifest_lines.append(f"{'TOTAL':<20} {total_files:>8} {total_bytes/1e6:>10.1f}")
        zf.writestr("MANIFEST.txt", "\n".join(manifest_lines))

    final_size = OUT_ZIP.stat().st_size
    print()
    print(f"DONE: {OUT_ZIP}")
    print(f"uncompressed: {total_bytes/1e6:.1f} MB across {total_files} files")
    print(f"zip on disk : {final_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
