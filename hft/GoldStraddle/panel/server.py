#!/usr/bin/env python3
"""GoldStraddle panel bridge server.

Implements CONTRACT.md sections 1 (transport + atomicity), 7 (HTTP API) and
8 (validation).  Python 3.13, **standard library only**.

What this process is
--------------------
A ``ThreadingHTTPServer`` bound to 127.0.0.1 that does exactly two jobs:

1. Serves the static browser UI out of ``panel/ui/``.
2. Bridges that UI to the MT5 Expert Advisor through three JSON files in the
   MT5 *Common Files* folder (see CONTRACT.md section 1):

       config.json   panel -> EA   (settings; EA polls it every 250 ms)
       state.json    EA -> panel   (telemetry; EA rewrites it every 500 ms)
       command.json  panel -> EA   (one-shot actions)

There is no database, no session, no auth: the security model is "loopback
only, and nothing outside ``ui/`` is ever served".

Two design decisions deserve an explanation up front, because getting them
wrong breaks the EA in ways that are very hard to debug from a chart window:

*Why the server owns ``config_version``*
    The EA re-reads and re-applies ``config.json`` **only** when
    ``config_version`` differs from the value it last applied.  That integer is
    therefore the entire change-notification mechanism between the two
    processes.  If a browser tab held a stale copy of the config and echoed its
    old version back on save, the EA would silently ignore the new settings --
    the user would move a level, see the file change on disk, and the EA would
    keep trading the old one.  So the version number is never accepted from a
    client: it is stripped from every incoming body and re-assigned here as
    ``last_written_version + 1``.  The counter is seeded from the existing
    ``config.json`` at startup (or 0) and, at every write, from the value
    currently on disk, so it stays strictly monotonic across server restarts,
    across two panels being opened in sequence, and across a config file that
    somebody edited by hand.

*Why writes are atomic*
    The EA polls these files on a 250 ms timer with no locking.  A plain
    ``open(path, "w")`` truncates the file first, so there is a window of a few
    hundred microseconds in which a reader sees an empty or half-written
    document.  The EA is required to keep its last good value on a parse
    failure (section 1), but relying on that as the normal path would mean the
    EA regularly skips a config generation.  Every write here therefore goes to
    ``<name>.tmp`` in the *same directory* (same volume, so the rename is a
    metadata operation) and is then moved onto the target with ``os.replace``,
    which is atomic on NTFS.  A reader sees either the whole old file or the
    whole new file, never a mixture.

Usage
-----
    python server.py                      # serve on 127.0.0.1:8733
    python server.py --port 9000
    python server.py --ipc-dir "D:\\tmp\\gs"
    python server.py --selftest           # run the in-process test suite
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlsplit

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

BIND_HOST = "127.0.0.1"          # CONTRACT.md section 7: never 0.0.0.0.
DEFAULT_PORT = 8733
MAX_BODY_BYTES = 256 * 1024      # 256 KB; a 32-level config is ~4 KB.
EA_ALIVE_MS = 3000               # section 7: state.json fresher than 3 s.
STATE_REREAD_DELAY_S = 0.025     # one retry when we catch a mid-write file.

MAX_LEVELS = 32                  # section 2 / section 8.
MIN_LOTS = 0.01                  # section 8.
MAX_LOTS = 100.0                 # section 8.
MAX_MAGIC = 2147483647           # section 8.
MAX_LABEL_CHARS = 24             # section 2: ASCII, <= 24 chars.
FLOAT_EPS = 1e-9                 # lots/limits are compared with a tolerance.

SYMBOL_RE = re.compile(r"^[A-Za-z0-9._#-]{1,32}$")      # section 8, verbatim.
LEVEL_ID_RE = re.compile(r"^[A-Za-z0-9._#-]{1,32}$")
HHMM_RE = re.compile(r"^\d{2}:\d{2}$")                  # section 8, verbatim.

# Top-level config keys, in CONTRACT.md section 2 order.  Used both to order
# the written file and to tell a client which fields it did not send.
CONFIG_KEYS: tuple[str, ...] = (
    "enabled", "symbol", "magic", "unit", "pip_size_mode", "pip_size_manual",
    "time_window", "entry", "long", "short", "levels",
)
SERVER_OWNED_KEYS: tuple[str, ...] = ("config_version", "updated_at")

UNITS = ("pips", "points", "price")
PIP_SIZE_MODES = ("auto", "manual")
TIMEZONES = ("server", "local")
TRIGGER_MODES = ("cross", "touch")
DIRECTIONS = ("either", "from_below", "from_above")

# section 6: exactly six commands, and the shape of each one's args.
COMMAND_ARG_KIND: dict[str, str] = {
    "close_all": "none",
    "close_level": "level_id",
    "close_side": "side",
    "rearm_level": "level_id",
    "rearm_all": "none",
    "panic": "none",
}

# Only these extensions are ever served from ui/.
ALLOWED_EXT = {".html", ".js", ".css", ".svg", ".png", ".ico", ".woff2", ".json"}

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".json": "application/json; charset=utf-8",
}

# Windows eats these names regardless of extension ("nul.html" opens the null
# device), so they never reach the filesystem layer.
WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

# Host header values we answer to (DNS-rebinding guard, section 7).  A rebinding
# attack always arrives with an attacker-controlled *domain name* in Host, so
# accepting only loopback literals plus "localhost" is sufficient.
ALLOWED_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def utc_stamp(now: datetime | None = None) -> str:
    """Return an ISO-8601 ``Z`` timestamp, second resolution (section 2)."""
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_ipc_dir(override: str | None = None) -> Path:
    """Resolve the MT5 Common Files IPC directory and make sure it exists.

    Default is ``%APPDATA%\\MetaQuotes\\Terminal\\Common\\Files\\GoldStraddle``.
    ``%APPDATA%`` is read from the environment because that is where MT5 itself
    gets it from; if the variable is missing (stripped environment, service
    account, a shell that clears it) we fall back to the conventional
    ``~/AppData/Roaming`` -- documented here so the fallback is never a
    surprise when the printed path looks unexpected.
    """
    if override:
        base = Path(override).expanduser()
    else:
        appdata = os.environ.get("APPDATA")
        roaming = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        base = roaming / "MetaQuotes" / "Terminal" / "Common" / "Files" / "GoldStraddle"
    base.mkdir(parents=True, exist_ok=True)
    return base.resolve()


def atomic_write_json(path: Path, obj: Any) -> None:
    """Write ``obj`` as JSON onto ``path`` atomically (section 1).

    ``<name>.tmp`` is written in the same directory -- same volume, so
    ``os.replace`` is a rename and not a copy -- flushed and fsync'd, then
    moved onto the target.  ``os.replace`` overwrites atomically on NTFS, so a
    concurrently polling EA sees either the complete old file or the complete
    new file.

    ``ensure_ascii=True`` and ``allow_nan=False`` enforce the encoding rules of
    section 1 and section 3: the MQL5 side reads with ``FILE_ANSI``, so a
    non-ASCII byte would come back as mojibake, and its JSON parser has no
    concept of ``NaN``/``Infinity``.
    """
    text = json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    # newline="\n": no CRLF translation, so byte counts match what MQL5 reads.
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json_file(path: Path) -> tuple[Any | None, str | None]:
    """Read and parse a JSON file.

    Returns ``(obj, None)`` on success, ``(None, reason)`` on failure.  Never
    raises: every caller here is on a polling path where the correct response
    to a bad read is "keep the last good value and try again".
    """
    try:
        raw = path.read_text(encoding="utf-8-sig")  # tolerate a BOM on input
    except FileNotFoundError:
        return None, "missing"
    except OSError as exc:
        return None, f"io error: {exc}"
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as exc:
        return None, f"parse error: {exc}"


def file_age_ms(path: Path) -> int | None:
    """Age of ``path`` in milliseconds, or ``None`` if it does not exist."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    # Clamp: a clock adjustment (or a file written by a machine whose clock is
    # slightly ahead) must not produce a negative age.
    return max(0, int((time.time() - mtime) * 1000))


def minutes_of_day(hhmm: Any) -> int | None:
    """Parse ``"HH:MM"`` into minutes since midnight, or ``None`` if invalid."""
    if not isinstance(hhmm, str) or not HHMM_RE.match(hhmm):
        return None
    hh, mm = int(hhmm[:2]), int(hhmm[3:])
    if hh >= 24 or mm >= 60:
        return None
    return hh * 60 + mm


def dow_contract(moment: datetime) -> int:
    """Day of week in CONTRACT.md numbering: 0 = Sunday .. 6 = Saturday."""
    return (moment.weekday() + 1) % 7


def time_window_open(start: Any, end: Any, days: Sequence[Any], now: datetime) -> bool:
    """Evaluate a ``time_window`` exactly as section 2 mandates.

    Wrap-around is required: when ``start > end`` the window spans midnight,
    and the *day* test then uses the day the window **started** on, so a Friday
    22:00-06:00 window stays open into Saturday morning.

    A zero-length window (``start == end``) is closed, which follows from the
    inclusive-start / exclusive-end rule rather than being a special case.
    This mirrors the EA so the panel can show an honest "would fire now" hint;
    the EA remains the authority.
    """
    m_start = minutes_of_day(start)
    m_end = minutes_of_day(end)
    if m_start is None or m_end is None:
        return False
    try:
        allowed = {int(d) for d in days if isinstance(d, int) and not isinstance(d, bool)}
    except (TypeError, ValueError):
        return False
    m_now = now.hour * 60 + now.minute
    today = dow_contract(now)

    if m_start <= m_end:
        in_time = m_start <= m_now < m_end
        start_day = today
    elif m_now >= m_start:                      # evening half of a wrap window
        in_time, start_day = True, today
    elif m_now < m_end:                         # morning half: started yesterday
        in_time, start_day = True, dow_contract(now - timedelta(days=1))
    else:
        in_time, start_day = False, today
    return in_time and start_day in allowed


def host_header_ok(host: str | None, port: int) -> bool:
    """DNS-rebinding guard (section 7).

    Accepts ``127.0.0.1``, ``localhost`` and ``[::1]``, with either no port or
    exactly the port we are listening on.  Anything else -- notably any
    attacker-owned domain name that has been re-pointed at 127.0.0.1 -- is
    refused, which stops a malicious page in the user's browser from driving
    this trading panel.
    """
    if not host:
        return False
    value = host.strip()
    if value.startswith("["):                   # bracketed IPv6 literal
        close = value.find("]")
        if close < 0:
            return False
        name = value[1:close]
        rest = value[close + 1:]
        if rest and not rest.startswith(":"):
            return False
        port_text = rest[1:] if rest.startswith(":") else ""
    elif value.count(":") > 1:                  # bare IPv6 is not valid in Host
        return False
    else:
        name, _, port_text = value.partition(":")
    if name.lower() not in ALLOWED_HOSTNAMES:
        return False
    if port_text:
        if not port_text.isdigit() or int(port_text) != port:
            return False
    return True


def safe_static_path(ui_dir: Path, url_path: str) -> Path | None:
    """Map a URL path to a file inside ``ui_dir``, or ``None`` if unsafe.

    Defence in depth, because "serve no path outside ui/" (section 7) has a
    long history of one-line bypasses on Windows:

    * single percent-decode only -- decoding twice is what turns ``%252e%252e``
      into ``..``; after one pass any remaining ``%`` means the request was
      double-encoded, and we refuse those outright (no real asset name here
      contains a percent sign);
    * ``\\`` is a path separator on Windows, so it must not survive into a
      "relative" path;
    * ``:`` covers both drive letters (``/C:/Windows/win.ini``) and NTFS
      alternate data streams (``app.js::$DATA``);
    * a leading ``//`` is a UNC prefix (``//server/share``);
    * trailing dots and spaces are silently stripped by the Win32 layer, so
      ``index.html.`` and ``index.html `` would otherwise reach ``index.html``
      through a check that compared strings;
    * Windows device names are rejected because ``nul.html`` opens the null
      device rather than a file;
    * and finally the candidate is ``resolve()``d -- which follows symlinks and
      junctions -- and required to live under the resolved ``ui/`` root, so a
      symlink inside ui/ pointing at ``C:\\`` gets caught too.

    The extension whitelist is the last gate: even a legitimate file under
    ui/ is not served unless it is one of the eight allowed asset types.
    """
    if not isinstance(url_path, str) or not url_path.startswith("/"):
        return None
    try:
        decoded = unquote(url_path, encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, UnicodeError):
        return None
    if "%" in decoded:                          # double-encoded -> refuse
        return None
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in decoded):
        return None                             # NUL and other control bytes
    if "\\" in decoded or ":" in decoded:
        return None
    if decoded.startswith("//"):
        return None
    rel = decoded.lstrip("/")
    if not rel:
        return None                             # "/" is handled by the caller
    parts = rel.split("/")
    for part in parts:
        if part in ("", ".", ".."):
            return None                         # empty catches "a//b"
        if part != part.strip():
            return None                         # leading/trailing whitespace
        if part.endswith("."):
            return None                         # Windows trailing-dot quirk
        if len(part) > 128:
            return None
        if part.split(".")[0].lower() in WINDOWS_RESERVED:
            return None
    root = ui_dir.resolve()
    candidate = (root / Path(*parts)).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        return None
    if candidate.suffix.lower() not in ALLOWED_EXT:
        return None
    return candidate


def content_type_for(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


# --------------------------------------------------------------------------- #
# Default config (section 2)
# --------------------------------------------------------------------------- #

def default_config() -> dict[str, Any]:
    """A fresh, deliberately harmless config.

    ``enabled`` is ``false`` -- a new install must never auto-arm -- there are
    no levels, and both legs sit at the smallest lot size the contract allows.
    ``config_version`` is a placeholder here; the caller assigns the real one.
    """
    leg = {
        "enabled": True,
        "lots": 0.01,          # smallest allowed: a fresh config cannot hurt
        "tp": 50.0,
        "sl": 0.0,
        "be_enabled": True,
        "be_trigger": 30.0,
        "be_offset": 5.0,
    }
    return {
        "config_version": 0,
        "updated_at": utc_stamp(),
        "enabled": False,      # master arm OFF on first run
        "symbol": "XAUUSD",
        "magic": 770101,
        "unit": "pips",
        "pip_size_mode": "auto",
        "pip_size_manual": 0.10,
        "time_window": {
            "enabled": True,
            "timezone": "server",
            "start": "08:00",
            "end": "17:30",
            "days": [1, 2, 3, 4, 5],
        },
        "entry": {
            "trigger_mode": "cross",
            "touch_tolerance": 0.0,
            "one_shot": True,
            "cooldown_sec": 600,
            "max_slippage_points": 30,
            "max_spread_points": 60,
            "max_concurrent_straddles": 3,
        },
        "long": dict(leg),
        "short": dict(leg),
        "levels": [],
    }


# --------------------------------------------------------------------------- #
# Validation (section 8)
# --------------------------------------------------------------------------- #

def _as_bool(value: Any, field: str, errors: list[str], fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    errors.append(f"{field}: expected true or false")
    return fallback


def _as_number(value: Any, field: str, errors: list[str], fallback: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{field}: expected a number")
        return fallback
    num = float(value)
    if num != num or num in (float("inf"), float("-inf")):
        errors.append(f"{field}: must be a finite number")
        return fallback
    return num


def _as_int(value: Any, field: str, errors: list[str], fallback: int) -> int:
    if isinstance(value, bool):
        errors.append(f"{field}: expected an integer")
        return fallback
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    errors.append(f"{field}: expected an integer")
    return fallback


def _as_str(value: Any, field: str, errors: list[str], fallback: str) -> str:
    if isinstance(value, str):
        return value
    errors.append(f"{field}: expected a string")
    return fallback


def _as_enum(value: Any, field: str, errors: list[str],
             allowed: Sequence[str], fallback: str) -> str:
    if isinstance(value, str) and value in allowed:
        return value
    errors.append(f"{field}: must be one of {', '.join(allowed)}")
    return fallback


def _non_negative(num: float, field: str, errors: list[str]) -> None:
    """section 8: ``tp``, ``sl``, ``be_trigger`` >= 0."""
    if num < -FLOAT_EPS:
        errors.append(f"{field}: must be >= 0")


def _validate_leg(raw: Any, base: dict[str, Any], side: str,
                  errors: list[str], warnings: list[str]) -> dict[str, Any]:
    """Validate one of the two identical ``long``/``short`` blocks."""
    leg = dict(base)
    if raw is None:
        return leg
    if not isinstance(raw, dict):
        errors.append(f"{side}: expected an object")
        return leg

    leg["enabled"] = _as_bool(raw.get("enabled", base["enabled"]),
                              f"{side}.enabled", errors, base["enabled"])

    lots = _as_number(raw.get("lots", base["lots"]), f"{side}.lots", errors, base["lots"])
    # section 8: lots within [0.01, 100].  Enforced even for a disabled leg --
    # the rule is unconditional, and a leg can be re-enabled from the UI in one
    # click without passing through validation again.
    if not (MIN_LOTS - FLOAT_EPS <= lots <= MAX_LOTS + FLOAT_EPS):
        errors.append(f"{side}.lots: must be between {MIN_LOTS} and {MAX_LOTS:g}")
    leg["lots"] = lots

    tp = _as_number(raw.get("tp", base["tp"]), f"{side}.tp", errors, base["tp"])
    _non_negative(tp, f"{side}.tp", errors)
    leg["tp"] = tp

    sl = _as_number(raw.get("sl", base["sl"]), f"{side}.sl", errors, base["sl"])
    _non_negative(sl, f"{side}.sl", errors)
    leg["sl"] = sl

    leg["be_enabled"] = _as_bool(raw.get("be_enabled", base["be_enabled"]),
                                 f"{side}.be_enabled", errors, base["be_enabled"])

    be_trigger = _as_number(raw.get("be_trigger", base["be_trigger"]),
                            f"{side}.be_trigger", errors, base["be_trigger"])
    _non_negative(be_trigger, f"{side}.be_trigger", errors)
    leg["be_trigger"] = be_trigger

    be_offset = _as_number(raw.get("be_offset", base["be_offset"]),
                           f"{side}.be_offset", errors, base["be_offset"])
    # section 8: be_offset in [-1000, 1000].  Negative is legal and means
    # "de-risk to a small loss", so this is a range check, not a sign check.
    if not (-1000.0 - FLOAT_EPS <= be_offset <= 1000.0 + FLOAT_EPS):
        errors.append(f"{side}.be_offset: must be between -1000 and 1000")
    leg["be_offset"] = be_offset

    # section 8, warn only: a BE trigger beyond TP can never fire.
    if be_trigger > 0 and tp > 0 and be_trigger > tp + FLOAT_EPS:
        warnings.append(
            f"{side}: be_trigger ({be_trigger:g}) is beyond tp ({tp:g}); "
            "breakeven can never fire"
        )
    return leg


def _validate_levels(raw: Any, errors: list[str],
                     warnings: list[str]) -> list[dict[str, Any]]:
    """Validate the ``levels`` array (sections 2 and 8)."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        errors.append("levels: expected an array")
        return []
    if len(raw) > MAX_LEVELS:
        errors.append(f"levels: at most {MAX_LEVELS} entries (got {len(raw)})")

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw[:MAX_LEVELS]):
        tag = f"levels[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{tag}: expected an object")
            continue

        level_id = _as_str(item.get("id", ""), f"{tag}.id", errors, "")
        if not LEVEL_ID_RE.match(level_id):
            errors.append(f"{tag}.id: must match {LEVEL_ID_RE.pattern}")
        elif level_id in seen:
            # section 8: duplicate level id -> reject.  Ids key runtime state
            # in the EA and tag positions via the "GS:<id>" comment, so a
            # duplicate would make two levels fight over the same positions.
            errors.append(f"{tag}.id: duplicate id {level_id!r}")
        else:
            seen.add(level_id)

        label = _as_str(item.get("label", ""), f"{tag}.label", errors, "")
        if len(label) > MAX_LABEL_CHARS:
            errors.append(f"{tag}.label: at most {MAX_LABEL_CHARS} characters")
        if not label.isascii():
            # section 1: the MQL5 side reads with FILE_ANSI, so non-ASCII would
            # arrive corrupted on the EA side.
            errors.append(f"{tag}.label: ASCII characters only")

        enabled = _as_bool(item.get("enabled", True), f"{tag}.enabled", errors, True)
        price = _as_number(item.get("price", 0.0), f"{tag}.price", errors, 0.0)
        # section 8: price > 0 for *enabled* levels.  A disabled placeholder row
        # with price 0 is allowed so the UI can hold an empty draft row.
        if enabled and price <= 0:
            errors.append(f"{tag}.price: must be > 0 for an enabled level")

        direction = _as_enum(item.get("direction", "either"), f"{tag}.direction",
                             errors, DIRECTIONS, "either")

        out.append({
            "id": level_id,
            "label": label,
            "price": price,
            "enabled": enabled,
            "direction": direction,
        })
    return out


def validate_config(incoming: Any) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate and normalise a client config body.

    Returns ``(config, errors, warnings)``.  When ``errors`` is non-empty the
    caller must reject with HTTP 400 and write nothing.

    Missing keys are filled from :func:`default_config`, so the UI may PATCH a
    subset without having to echo the whole document back; unknown keys are
    dropped rather than persisted, which keeps ``config.json`` matching
    section 2 exactly.  ``config_version`` and ``updated_at`` are ignored on
    input -- the server owns both (see the module docstring).
    """
    errors: list[str] = []
    warnings: list[str] = []
    base = default_config()

    if not isinstance(incoming, dict):
        return base, ["body: expected a JSON object"], warnings

    cfg = default_config()

    cfg["enabled"] = _as_bool(incoming.get("enabled", base["enabled"]),
                              "enabled", errors, base["enabled"])

    symbol = _as_str(incoming.get("symbol", base["symbol"]), "symbol", errors,
                     base["symbol"])
    if not SYMBOL_RE.match(symbol):
        errors.append(f"symbol: must match {SYMBOL_RE.pattern}")
    cfg["symbol"] = symbol

    magic = _as_int(incoming.get("magic", base["magic"]), "magic", errors,
                    base["magic"])
    if not (1 <= magic <= MAX_MAGIC):
        errors.append(f"magic: must be between 1 and {MAX_MAGIC}")
    cfg["magic"] = magic

    cfg["unit"] = _as_enum(incoming.get("unit", base["unit"]), "unit", errors,
                           UNITS, base["unit"])
    cfg["pip_size_mode"] = _as_enum(incoming.get("pip_size_mode", base["pip_size_mode"]),
                                    "pip_size_mode", errors, PIP_SIZE_MODES,
                                    base["pip_size_mode"])
    pip_manual = _as_number(incoming.get("pip_size_manual", base["pip_size_manual"]),
                            "pip_size_manual", errors, base["pip_size_manual"])
    if cfg["pip_size_mode"] == "manual" and pip_manual <= 0:
        errors.append("pip_size_manual: must be > 0 when pip_size_mode is manual")
    cfg["pip_size_manual"] = pip_manual

    # ---- time_window -----------------------------------------------------
    tw_base = base["time_window"]
    tw_raw = incoming.get("time_window")
    tw = dict(tw_base)
    if tw_raw is None:
        pass
    elif not isinstance(tw_raw, dict):
        errors.append("time_window: expected an object")
    else:
        tw["enabled"] = _as_bool(tw_raw.get("enabled", tw_base["enabled"]),
                                 "time_window.enabled", errors, tw_base["enabled"])
        tw["timezone"] = _as_enum(tw_raw.get("timezone", tw_base["timezone"]),
                                  "time_window.timezone", errors, TIMEZONES,
                                  tw_base["timezone"])
        for key in ("start", "end"):
            value = tw_raw.get(key, tw_base[key])
            # section 8: ^\d{2}:\d{2}$ with h < 24 and m < 60.
            if minutes_of_day(value) is None:
                errors.append(
                    f"time_window.{key}: must be HH:MM (24h, hour < 24, minute < 60)"
                )
                tw[key] = tw_base[key]
            else:
                tw[key] = value
        days_raw = tw_raw.get("days", tw_base["days"])
        if not isinstance(days_raw, list):
            errors.append("time_window.days: expected an array of integers 0..6")
        else:
            clean_days: list[int] = []
            for day in days_raw:
                if isinstance(day, bool) or not isinstance(day, int) or not 0 <= day <= 6:
                    errors.append("time_window.days: values must be integers 0..6")
                    continue
                if day not in clean_days:
                    clean_days.append(day)
            tw["days"] = clean_days
            # Not a section 8 rule; warn-only nicety because an empty array
            # silently means "nothing ever passes" (section 2).
            if tw["enabled"] and not clean_days:
                warnings.append("time_window.days is empty: no day will ever pass")
    cfg["time_window"] = tw

    # ---- entry -----------------------------------------------------------
    entry_base = base["entry"]
    entry_raw = incoming.get("entry")
    entry = dict(entry_base)
    if entry_raw is None:
        pass
    elif not isinstance(entry_raw, dict):
        errors.append("entry: expected an object")
    else:
        entry["trigger_mode"] = _as_enum(
            entry_raw.get("trigger_mode", entry_base["trigger_mode"]),
            "entry.trigger_mode", errors, TRIGGER_MODES, entry_base["trigger_mode"])
        tol = _as_number(entry_raw.get("touch_tolerance", entry_base["touch_tolerance"]),
                         "entry.touch_tolerance", errors, entry_base["touch_tolerance"])
        _non_negative(tol, "entry.touch_tolerance", errors)
        entry["touch_tolerance"] = tol
        entry["one_shot"] = _as_bool(entry_raw.get("one_shot", entry_base["one_shot"]),
                                     "entry.one_shot", errors, entry_base["one_shot"])
        for key in ("cooldown_sec", "max_slippage_points", "max_spread_points",
                    "max_concurrent_straddles"):
            value = _as_int(entry_raw.get(key, entry_base[key]), f"entry.{key}",
                            errors, entry_base[key])
            if value < 0:
                errors.append(f"entry.{key}: must be >= 0")
            entry[key] = value
    cfg["entry"] = entry

    # ---- legs and levels -------------------------------------------------
    cfg["long"] = _validate_leg(incoming.get("long"), base["long"], "long",
                                errors, warnings)
    cfg["short"] = _validate_leg(incoming.get("short"), base["short"], "short",
                                 errors, warnings)
    cfg["levels"] = _validate_levels(incoming.get("levels"), errors, warnings)

    # ---- body-completeness warnings --------------------------------------
    # Section 7 says the POST body is the *full* config object.  Missing keys
    # are still filled from the defaults rather than rejected (a save must not
    # fail because one block was omitted), but silence would be dangerous:
    # omitting "levels" resets the level list, and omitting "enabled" disarms.
    # Both are the fail-safe direction, and both are now said out loud.
    missing = [key for key in CONFIG_KEYS if key not in incoming]
    if missing:
        warnings.append(
            "not sent by the client, so reset to defaults: " + ", ".join(missing)
        )
    unknown = sorted(set(incoming) - set(CONFIG_KEYS) - set(SERVER_OWNED_KEYS))
    if unknown:
        warnings.append("ignored unknown field(s): " + ", ".join(unknown))

    # ---- cross-field warnings (section 8, warn only) ---------------------
    if cfg["enabled"] and not any(lv["enabled"] for lv in cfg["levels"]):
        warnings.append("enabled is true but no level is enabled: nothing can fire")
    if cfg["enabled"] and not (cfg["long"]["enabled"] or cfg["short"]["enabled"]):
        warnings.append("enabled is true but neither leg is enabled: nothing can fire")

    return cfg, errors, warnings


def validate_command(incoming: Any) -> tuple[str, dict[str, Any], list[str]]:
    """Validate a ``POST /api/command`` body against section 6.

    Returns ``(cmd, args, errors)``.  ``id`` and ``issued_at`` are server-owned
    and ignored on input, exactly like ``config_version``.
    """
    errors: list[str] = []
    if not isinstance(incoming, dict):
        return "", {}, ["body: expected a JSON object"]

    cmd = incoming.get("cmd")
    if not isinstance(cmd, str) or cmd not in COMMAND_ARG_KIND:
        allowed = ", ".join(sorted(COMMAND_ARG_KIND))
        return "", {}, [f"cmd: must be one of {allowed}"]

    raw_args = incoming.get("args", {})
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, dict):
        return cmd, {}, ["args: expected an object"]

    kind = COMMAND_ARG_KIND[cmd]
    args: dict[str, Any] = {}
    if kind == "none":
        expected: set[str] = set()
    elif kind == "level_id":
        expected = {"level_id"}
        level_id = raw_args.get("level_id")
        if not isinstance(level_id, str) or not LEVEL_ID_RE.match(level_id):
            errors.append(f"args.level_id: required, must match {LEVEL_ID_RE.pattern}")
        else:
            args["level_id"] = level_id
    else:  # "side"
        expected = {"side"}
        side = raw_args.get("side")
        if side not in ("long", "short"):
            errors.append('args.side: required, must be "long" or "short"')
        else:
            args["side"] = side

    unexpected = sorted(set(raw_args) - expected)
    if unexpected:
        # Strict on purpose: a typo like {"levelId": "L1"} must not silently
        # become "close every level".
        errors.append(f"args: unexpected key(s) for {cmd}: {', '.join(unexpected)}")
    return cmd, args, errors


# --------------------------------------------------------------------------- #
# Bridge: owns the three IPC files
# --------------------------------------------------------------------------- #

class Bridge:
    """All file I/O against the MT5 Common Files directory.

    One instance per server process.  Every mutation goes through ``_lock`` so
    two concurrent POSTs (the UI debounces, but a second browser window or a
    double-click can still overlap) cannot assign the same ``config_version``
    or the same command ``id``.
    """

    def __init__(self, ipc_dir: Path) -> None:
        self.ipc_dir = ipc_dir
        self.config_path = ipc_dir / "config.json"
        self.state_path = ipc_dir / "state.json"
        self.command_path = ipc_dir / "command.json"
        self._lock = threading.Lock()
        self._version = self._seed_version()
        self._command_id = self._seed_command_id()
        # Last successfully parsed state, served while the EA is mid-write.
        self._last_good_state: dict[str, Any] | None = None
        self.notes: list[str] = []   # startup remarks worth printing

    # -- seeding ----------------------------------------------------------

    def _disk_version(self) -> int:
        cfg, _ = read_json_file(self.config_path)
        if isinstance(cfg, dict):
            value = cfg.get("config_version")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
            if isinstance(value, float) and value.is_integer() and value >= 0:
                return int(value)
        return 0

    def _seed_version(self) -> int:
        """Seed the version counter from whatever is already on disk.

        Restarting the panel must never hand the EA a version it has already
        applied, otherwise the next edit looks like "no change" and is ignored.
        """
        return self._disk_version()

    def _disk_command_id(self) -> int:
        best = 0
        cmd, _ = read_json_file(self.command_path)
        if isinstance(cmd, dict) and isinstance(cmd.get("id"), int):
            best = max(best, int(cmd["id"]))
        # The EA's own view matters too: if command.json is stale (deleted,
        # restored from a backup, or written by an older panel) but the EA has
        # already executed id 20, a fresh counter starting at 1 would either
        # replay old commands or -- worse -- issue ids the EA considers already
        # seen and therefore silently drops.
        state, _ = read_json_file(self.state_path)
        if isinstance(state, dict) and isinstance(state.get("last_command_id"), int):
            best = max(best, int(state["last_command_id"]))
        return best

    def _seed_command_id(self) -> int:
        return self._disk_command_id()

    # -- config -----------------------------------------------------------

    @property
    def version(self) -> int:
        """Last ``config_version`` this server wrote (or found on disk)."""
        return self._version

    def ensure_config(self) -> dict[str, Any]:
        """Return the current config, creating a default one on first run.

        A ``config.json`` that exists but does not parse is moved aside to
        ``config.json.bad-<timestamp>`` and replaced with defaults: it is
        unusable to both the panel and the EA, and keeping it in place would
        wedge the UI on every load.  The original is preserved for inspection
        rather than deleted.
        """
        with self._lock:
            cfg, err = read_json_file(self.config_path)
            if isinstance(cfg, dict):
                return cfg
            if err and err != "missing" and self.config_path.exists():
                stamp = time.strftime("%Y%m%d-%H%M%S")
                backup = self.config_path.with_name(f"config.json.bad-{stamp}")
                try:
                    os.replace(self.config_path, backup)
                    self.notes.append(
                        f"config.json was unreadable ({err}); moved to {backup.name} "
                        "and wrote a fresh default"
                    )
                except OSError as exc:            # pragma: no cover - rare
                    self.notes.append(f"could not set aside broken config.json: {exc}")
            fresh = default_config()
            fresh["config_version"] = self._next_version_locked()
            fresh["updated_at"] = utc_stamp()
            atomic_write_json(self.config_path, fresh)
            return fresh

    def _next_version_locked(self) -> int:
        """Assign the next config version.  Caller must hold ``_lock``.

        ``max(in-memory, on-disk) + 1`` rather than ``in-memory + 1``: the disk
        value can be ahead of us if a second panel instance or a hand edit
        touched the file while we were running, and a version that goes
        backwards would make the EA ignore the write.
        """
        self._version = max(self._version, self._disk_version()) + 1
        return self._version

    def write_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """Stamp version + timestamp onto ``cfg`` and write it atomically.

        Returns the exact object written, so the UI can reconcile its local
        state (in particular the new ``config_version``) with the file the EA
        will read.
        """
        with self._lock:
            out = dict(cfg)
            out["config_version"] = self._next_version_locked()
            out["updated_at"] = utc_stamp()
            # Rebuild in section 2 key order so the file stays human-readable.
            ordered = {key: out[key]
                       for key in SERVER_OWNED_KEYS + CONFIG_KEYS
                       if key in out}
            atomic_write_json(self.config_path, ordered)
            return ordered

    # -- command ----------------------------------------------------------

    def write_command(self, cmd: str, args: dict[str, Any]) -> int:
        """Write ``command.json`` with a fresh monotonic id; return that id."""
        with self._lock:
            self._command_id = max(self._command_id, self._disk_command_id()) + 1
            payload = {
                "id": self._command_id,
                "cmd": cmd,
                "args": args,
                "issued_at": utc_stamp(),
            }
            atomic_write_json(self.command_path, payload)
            return self._command_id

    # -- state ------------------------------------------------------------

    def read_state(self) -> tuple[dict[str, Any] | None, int | None, bool]:
        """Read ``state.json``.

        Returns ``(state, age_ms, parse_error)``.

        Catching the EA mid-write is *normal*, not an error: the panel polls
        twice a second while the EA rewrites the file twice a second.  One
        short retry usually lands on the completed file; if it does not, the
        last good state is returned with the real age from the file's mtime and
        ``parse_error`` set, so the UI can show a transient blip instead of a
        blank panel.
        """
        for attempt in (0, 1):
            parsed, err = read_json_file(self.state_path)
            if isinstance(parsed, dict):
                with self._lock:
                    self._last_good_state = parsed
                return parsed, file_age_ms(self.state_path), False
            if err == "missing" and not self.state_path.exists():
                return None, None, False          # EA has never run: state null
            if attempt == 0:
                time.sleep(STATE_REREAD_DELAY_S)
        with self._lock:
            last = self._last_good_state
        return last, file_age_ms(self.state_path), True

    def bridge_block(self, state: dict[str, Any] | None, age_ms: int | None,
                     parse_error: bool) -> dict[str, Any]:
        """Build the ``bridge`` block of section 7."""
        seen = 0
        if isinstance(state, dict):
            raw_seen = state.get("config_version_seen")
            if isinstance(raw_seen, int) and not isinstance(raw_seen, bool):
                seen = raw_seen
        version = self._version
        return {
            "state_age_ms": age_ms,
            "ea_alive": age_ms is not None and age_ms < EA_ALIVE_MS,
            "ipc_dir": str(self.ipc_dir),
            "config_version": version,
            "config_version_seen": seen,
            "in_sync": version == seen,
            # Extra flag (not in section 7): tells the UI that the numbers it is
            # showing came from the last good read, not from a fresh parse.
            "parse_error": parse_error,
        }


# --------------------------------------------------------------------------- #
# HTTP layer (section 7)
# --------------------------------------------------------------------------- #

MISSING_UI_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>GoldStraddle panel - UI files missing</title>
<style>
 body {{ font: 15px/1.55 "Segoe UI", system-ui, sans-serif; margin: 0;
         background: #14171c; color: #e6e8ea; }}
 main {{ max-width: 680px; margin: 8vh auto; padding: 28px 32px;
         background: #1b1f26; border: 1px solid #2b313b; border-radius: 10px; }}
 h1 {{ font-size: 20px; margin: 0 0 6px; color: #f0b429; }}
 code {{ background: #11141a; padding: 2px 6px; border-radius: 4px;
         font-family: Consolas, monospace; }}
 ul {{ padding-left: 20px; }}
 .dim {{ color: #98a0ab; font-size: 13px; }}
</style></head><body><main>
<h1>The panel UI files are not there yet</h1>
<p>The bridge server is running fine &mdash; the API is live &mdash; but it could
not find <code>index.html</code>.</p>
<p>Expected location:</p>
<p><code>{ui_dir}</code></p>
<p>These three files need to exist in that folder:</p>
<ul><li><code>index.html</code></li><li><code>app.js</code></li>
<li><code>style.css</code></li></ul>
<p class="dim">The API endpoints work regardless, so you can check the bridge with
<code>/api/health</code> or <code>/api/state</code>. Reload this page once the
files are in place.</p>
</main></body></html>
"""


class PanelServer(ThreadingHTTPServer):
    """Threaded loopback HTTP server carrying the bridge + ui dir."""

    daemon_threads = True
    # HTTPServer defaults this to 1.  On Windows SO_REUSEADDR lets a second
    # process bind a port that is already in use, so two panels could end up
    # half-serving the same port; refusing reuse makes the launcher's port
    # probe honest and makes a real conflict fail loudly at startup.
    allow_reuse_address = False

    def __init__(self, addr: tuple[str, int], bridge: Bridge, ui_dir: Path) -> None:
        self.bridge = bridge
        self.ui_dir = ui_dir
        super().__init__(addr, PanelHandler)


class PanelHandler(BaseHTTPRequestHandler):
    """Request handler for the panel API and static UI."""

    server_version = "GoldStraddlePanel/1.0"
    sys_version = ""                    # do not advertise the Python version
    protocol_version = "HTTP/1.1"       # keep-alive; every reply sets a length

    # -- plumbing ---------------------------------------------------------

    @property
    def bridge(self) -> Bridge:
        return self.server.bridge          # type: ignore[attr-defined]

    @property
    def ui_dir(self) -> Path:
        return self.server.ui_dir          # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        # /api/state is polled twice a second; logging it would bury everything
        # else. Errors on it still surface through log_error -> here with a
        # non-2xx status, which we keep.
        line = fmt % args
        if "/api/state" in line and " 200 " in line:
            return
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {line}\n")

    def _send_bytes(self, status: int, body: bytes, content_type: str,
                    cache: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        # Deliberately no CORS headers anywhere: same-origin only.
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8", "no-store")

    def _send_text(self, status: int, text: str) -> None:
        self._send_bytes(status, text.encode("utf-8"), "text/plain; charset=utf-8",
                         "no-store")

    def _guard(self) -> bool:
        """Origin checks.  Returns False (and answers) if the request dies."""
        port = self.server.server_port
        if not host_header_ok(self.headers.get("Host"), port):
            self._send_text(403, "403 forbidden: unexpected Host header\n")
            return False
        # An absolute-form request target ("GET http://evil/x HTTP/1.1") or a
        # protocol-relative one ("//host/x") carries its own authority, which
        # RFC 7230 says takes precedence over Host.  urlsplit surfaces it as
        # netloc, so it gets exactly the same treatment.
        target = urlsplit(self.path)
        if target.scheme and target.scheme.lower() != "http":
            self._send_text(403, "403 forbidden: unexpected request scheme\n")
            return False
        if target.netloc and not host_header_ok(target.netloc, port):
            self._send_text(403, "403 forbidden: unexpected request authority\n")
            return False
        return True

    def _is_loopback_peer(self) -> bool:
        return self.client_address and self.client_address[0] in ("127.0.0.1", "::1")

    def _read_body(self) -> tuple[Any, list[str] | None]:
        """Read and JSON-parse the request body.

        Returns ``(obj, None)`` or ``(None, errors)``.  A body over
        :data:`MAX_BODY_BYTES` is refused with 413 by the caller.
        """
        if self.headers.get("Transfer-Encoding", "").lower().strip() == "chunked":
            return None, ["chunked request bodies are not supported"]
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            return None, ["Content-Length header is required"]
        try:
            length = int(raw_len)
        except ValueError:
            return None, ["Content-Length is not an integer"]
        if length < 0:
            return None, ["Content-Length is negative"]
        if length > MAX_BODY_BYTES:
            return None, ["__TOO_LARGE__"]
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return None, ["body is empty"]
        try:
            return json.loads(raw.decode("utf-8")), None
        except UnicodeDecodeError:
            return None, ["body is not valid UTF-8"]
        except json.JSONDecodeError as exc:
            return None, [f"body is not valid JSON: {exc}"]

    # -- routing ----------------------------------------------------------

    def do_GET(self) -> None:      # noqa: N802 (stdlib naming)
        if not self._guard():
            return
        path = urlsplit(self.path).path
        try:
            if path == "/api/state":
                self._api_state()
            elif path == "/api/config":
                self._api_get_config()
            elif path == "/api/health":
                self._api_health()
            elif path.startswith("/api/"):
                self._send_json(404, {"ok": False, "errors": [f"no such endpoint: {path}"]})
            else:
                self._serve_static(path)
        except Exception as exc:                       # pragma: no cover
            self.log_message("unhandled GET error on %s: %r", path, exc)
            self._send_json(500, {"ok": False, "errors": ["internal server error"]})

    def do_HEAD(self) -> None:     # noqa: N802
        # Same routing as GET; _send_bytes skips the body for HEAD.
        self.do_GET()

    def do_POST(self) -> None:     # noqa: N802
        if not self._guard():
            return
        path = urlsplit(self.path).path
        try:
            if path == "/api/config":
                self._api_post_config()
            elif path == "/api/command":
                self._api_post_command()
            elif path == "/api/shutdown":
                self._api_shutdown()
            else:
                self._send_json(404, {"ok": False, "errors": [f"no such endpoint: {path}"]})
        except Exception as exc:                       # pragma: no cover
            self.log_message("unhandled POST error on %s: %r", path, exc)
            self._send_json(500, {"ok": False, "errors": ["internal server error"]})

    # -- endpoints --------------------------------------------------------

    def _api_state(self) -> None:
        """``GET /api/state`` -- must never return an error page.

        The UI polls this twice a second and treats it as the source of truth
        for the whole screen; a 500 would blank the panel while the user is
        looking at open positions.  So everything is caught here and reported
        inside a 200 body via ``bridge.parse_error`` / ``bridge.error``.
        """
        try:
            state, age_ms, parse_error = self.bridge.read_state()
            block = self.bridge.bridge_block(state, age_ms, parse_error)
            self._send_json(200, {"ok": True, "state": state, "bridge": block})
        except Exception as exc:
            self.log_message("state read failed: %r", exc)
            block = {
                "state_age_ms": None,
                "ea_alive": False,
                "ipc_dir": str(self.bridge.ipc_dir),
                "config_version": self.bridge.version,
                "config_version_seen": 0,
                "in_sync": False,
                "parse_error": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
            self._send_json(200, {"ok": True, "state": None, "bridge": block})

    def _api_get_config(self) -> None:
        cfg = self.bridge.ensure_config()
        self._send_json(200, {"ok": True, "config": cfg})

    def _api_post_config(self) -> None:
        body, errors = self._read_body()
        if errors:
            if errors == ["__TOO_LARGE__"]:
                self.close_connection = True    # we never drained the body
                self._send_json(413, {"ok": False,
                                      "errors": [f"body exceeds {MAX_BODY_BYTES} bytes"]})
                return
            self._send_json(400, {"ok": False, "errors": errors})
            return
        if isinstance(body, dict):
            # The server owns these two; drop whatever the client sent so a
            # stale tab cannot roll the version backwards (see module docstring).
            body = {k: v for k, v in body.items()
                    if k not in ("config_version", "updated_at")}
        cfg, verrors, warnings = validate_config(body)
        if verrors:
            self._send_json(400, {"ok": False, "errors": verrors, "warnings": warnings})
            return
        written = self.bridge.write_config(cfg)
        self._send_json(200, {"ok": True, "config": written, "warnings": warnings})

    def _api_post_command(self) -> None:
        body, errors = self._read_body()
        if errors:
            if errors == ["__TOO_LARGE__"]:
                self.close_connection = True
                self._send_json(413, {"ok": False,
                                      "errors": [f"body exceeds {MAX_BODY_BYTES} bytes"]})
                return
            self._send_json(400, {"ok": False, "errors": errors})
            return
        cmd, args, verrors = validate_command(body)
        if verrors:
            self._send_json(400, {"ok": False, "errors": verrors})
            return
        cmd_id = self.bridge.write_command(cmd, args)
        self.log_message('command %s id=%d args=%s', cmd, cmd_id, json.dumps(args))
        self._send_json(200, {"ok": True, "id": cmd_id})

    def _api_health(self) -> None:
        self._send_json(200, {
            "ok": True,
            "ipc_dir": str(self.bridge.ipc_dir),
            "config_exists": self.bridge.config_path.exists(),
            "state_age_ms": file_age_ms(self.bridge.state_path),
        })

    def _api_shutdown(self) -> None:
        """``POST /api/shutdown`` -- used by the launcher when the window closes."""
        if not self._is_loopback_peer():
            self._send_text(403, "403 forbidden\n")
            return
        self._send_json(200, {"ok": True, "stopping": True})
        # shutdown() blocks until serve_forever() returns, so it must not be
        # called from the thread that is currently handling this request.
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    # -- static -----------------------------------------------------------

    def _send_missing_ui(self) -> None:
        page = MISSING_UI_HTML.format(ui_dir=html.escape(str(self.ui_dir)))
        self._send_bytes(200, page.encode("utf-8"), "text/html; charset=utf-8",
                         "no-store")

    def _serve_static(self, path: str) -> None:
        if path in ("", "/"):
            target = self.ui_dir / "index.html"
        else:
            safe = safe_static_path(self.ui_dir, path)
            if safe is None:
                self.log_message("rejected static path %r", path)
                self._send_text(403, "403 forbidden\n")
                return
            target = safe
        if not target.is_file():
            if target.name == "index.html":
                self._send_missing_ui()
                return
            self._send_text(404, "404 not found\n")
            return
        try:
            data = target.read_bytes()
        except OSError as exc:
            self.log_message("cannot read %s: %s", target, exc)
            self._send_text(404, "404 not found\n")
            return
        # "no-cache" (not "no-store"): the browser may keep the file but must
        # revalidate, so the other agent's edits to app.js show up on reload.
        self._send_bytes(200, data, content_type_for(target), "no-cache")


# --------------------------------------------------------------------------- #
# Startup banner + entry point
# --------------------------------------------------------------------------- #

def build_server(bridge: Bridge, port: int, ui_dir: Path) -> PanelServer:
    return PanelServer((BIND_HOST, port), bridge, ui_dir)


def print_banner(bridge: Bridge, port: int, ui_dir: Path) -> None:
    """Print the information the user needs to debug the bridge at a glance."""
    cfg, cfg_err = read_json_file(bridge.config_path)
    version = cfg.get("config_version") if isinstance(cfg, dict) else "?"
    age = file_age_ms(bridge.state_path)
    if age is None:
        state_line = "MISSING (EA has not written state.json yet)"
    elif age < EA_ALIVE_MS:
        state_line = f"FRESH  (age {age} ms - EA looks alive)"
    else:
        state_line = f"STALE  (age {age} ms - EA not writing)"

    index = ui_dir / "index.html"
    ui_line = f"{ui_dir}  ({'index.html present' if index.is_file() else 'index.html MISSING'})"

    line = "=" * 72
    print(line)
    print(" GoldStraddle panel bridge")
    print("-" * 72)
    print(f" URL        : http://{BIND_HOST}:{port}/")
    print(f" IPC dir    : {bridge.ipc_dir}")
    print(f" config.json: {'present' if bridge.config_path.exists() else 'absent'}"
          f"  config_version={version}"
          + (f"  [{cfg_err}]" if cfg_err and cfg_err != 'missing' else ""))
    print(f" state.json : {state_line}")
    print(f" command.json: next id = {bridge._command_id + 1}")
    print(f" UI dir     : {ui_line}")
    if isinstance(cfg, dict):
        tw = cfg.get("time_window") or {}
        if isinstance(tw, dict) and tw.get("enabled"):
            now_local = datetime.now()
            hint = time_window_open(tw.get("start"), tw.get("end"),
                                    tw.get("days") or [], now_local)
            print(f" window     : {tw.get('start')}-{tw.get('end')} "
                  f"days={tw.get('days')} tz={tw.get('timezone')}"
                  f"  -> {'OPEN' if hint else 'CLOSED'} vs this PC's clock "
                  f"({now_local:%a %H:%M}); the EA judges 'server' time itself")
        else:
            print(" window     : disabled (levels can fire 24/5)")
        if cfg.get("enabled") is False:
            print(" master arm : OFF (enabled=false) - the EA will open nothing")
        else:
            print(" master arm : ON  (enabled=true)")
    for note in bridge.notes:
        print(f" note       : {note}")
    print("-" * 72)
    print(" Ctrl+C to stop, or POST /api/shutdown")
    print(line, flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GoldStraddle panel bridge server (localhost only).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"TCP port on 127.0.0.1 (default {DEFAULT_PORT})")
    parser.add_argument("--ipc-dir", default=None,
                        help="override the MT5 Common Files IPC directory")
    parser.add_argument("--ui-dir", default=None,
                        help="override the static UI directory (debug)")
    parser.add_argument("--selftest", action="store_true",
                        help="run the in-process test suite and exit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return run_selftest()

    ipc_dir = resolve_ipc_dir(args.ipc_dir)
    ui_dir = Path(args.ui_dir).expanduser().resolve() if args.ui_dir \
        else (Path(__file__).resolve().parent / "ui")
    bridge = Bridge(ipc_dir)
    bridge.ensure_config()

    try:
        httpd = build_server(bridge, args.port, ui_dir)
    except OSError as exc:
        print(f"ERROR: cannot bind {BIND_HOST}:{args.port} - {exc}", file=sys.stderr)
        print("Another panel may already be running; try --port 8734.",
              file=sys.stderr)
        return 2

    print_banner(bridge, args.port, ui_dir)
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nstopping (Ctrl+C)")
    finally:
        httpd.server_close()
    return 0


# --------------------------------------------------------------------------- #
# Self test (--selftest)
# --------------------------------------------------------------------------- #

NASTY_PATHS: tuple[str, ...] = (
    "/../secret.txt",
    "/../../CONTRACT.md",
    "/..%2fsecret.txt",
    "/%2e%2e/%2e%2e/config.json",
    "/%252e%252e/secret.txt",
    "/..\\secret.txt",
    "/\\..\\secret.txt",
    "/sub\\..\\..\\secret.txt",
    "/C:/Windows/win.ini",
    "/c:\\windows\\win.ini",
    "//server/share/evil.html",
    "/app.js:hidden",
    "/app.js::$DATA",
    "/index.html.",
    "/index.html ",
    "/ index.html",
    "/sub/../../out.html",
    "/./../app.js",
    "/%00app.js",
    "/app.js%00.png",
    "/nul.html",
    "/con.js",
    "/COM1.css",
    "/app.php",
    "/.env",
    "/secret.txt",
    "/sub//app.js",
    "/app.js/../../out.html",
    "app.js",
    "",
    "/..",
    "/.",
    "/" + "a" * 200 + ".js",
)


class _SelfTest(unittest.TestCase):
    """In-process checks for everything that is easy to get subtly wrong."""

    def setUp(self) -> None:
        import tempfile
        self._tmp = tempfile.TemporaryDirectory(prefix="gs-selftest-")
        self.root = Path(self._tmp.name)
        self.ipc = self.root / "ipc"
        self.ipc.mkdir()
        self.ui = self.root / "ui"
        self.ui.mkdir()
        (self.ui / "index.html").write_text("<h1>ui</h1>", encoding="utf-8")
        (self.ui / "app.js").write_text("// ui", encoding="utf-8")
        (self.ui / "style.css").write_text("body{}", encoding="utf-8")
        (self.ui / "secret.txt").write_text("not servable", encoding="utf-8")
        (self.ui / "sub").mkdir()
        (self.ui / "sub" / "icon.svg").write_text("<svg/>", encoding="utf-8")
        (self.root / "secret.txt").write_text("outside ui", encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    # -- section 1: atomic write ------------------------------------------

    def test_atomic_write_round_trip(self) -> None:
        path = self.ipc / "round.json"
        payload = {"a": 1, "b": [1, 2, 3], "c": {"d": 0.5}, "s": "XAUUSD"}
        atomic_write_json(path, payload)
        back, err = read_json_file(path)
        self.assertIsNone(err)
        self.assertEqual(back, payload)
        # no temp file left behind
        self.assertFalse(path.with_name(path.name + ".tmp").exists())
        raw = path.read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"), "must be UTF-8 without BOM")
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"\r\n", raw, "LF only, MQL5 reads with FILE_ANSI")

    def test_atomic_write_overwrites_and_stays_parseable(self) -> None:
        path = self.ipc / "round.json"
        atomic_write_json(path, {"v": 1})
        # a stale .tmp from a crashed write must not block the next one
        path.with_name(path.name + ".tmp").write_text("garbage", encoding="utf-8")
        atomic_write_json(path, {"v": 2})
        back, _ = read_json_file(path)
        self.assertEqual(back, {"v": 2})

    def test_atomic_write_rejects_nan(self) -> None:
        # section 3 forbids NaN/Infinity; the writer must refuse rather than
        # emit JSON the MQL5 parser cannot read.
        with self.assertRaises(ValueError):
            atomic_write_json(self.ipc / "nan.json", {"x": float("nan")})

    def test_read_json_file_reports_reasons(self) -> None:
        missing, err = read_json_file(self.ipc / "nope.json")
        self.assertIsNone(missing)
        self.assertEqual(err, "missing")
        bad = self.ipc / "bad.json"
        bad.write_text('{"half": ', encoding="utf-8")
        obj, err = read_json_file(bad)
        self.assertIsNone(obj)
        self.assertIn("parse error", err or "")

    # -- section 8: accept cases ------------------------------------------

    def test_default_config_validates_clean(self) -> None:
        cfg, errors, warnings = validate_config(default_config())
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])
        self.assertFalse(cfg["enabled"], "a fresh config must never be armed")
        self.assertEqual(cfg["long"]["lots"], 0.01)
        self.assertEqual(cfg["symbol"], "XAUUSD")
        self.assertEqual(cfg["magic"], 770101)
        self.assertEqual(cfg["levels"], [])
        self.assertNotIn("config_version", [k for k in cfg if k == "nope"])

    def test_accepts_full_contract_example(self) -> None:
        cfg = default_config()
        cfg["enabled"] = True
        cfg["levels"] = [
            {"id": "L1", "label": "PDH", "price": 3548.50, "enabled": True,
             "direction": "either"},
            {"id": "L2", "label": "Asia low", "price": 3501.0, "enabled": True,
             "direction": "from_above"},
        ]
        out, errors, warnings = validate_config(cfg)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])
        self.assertEqual(len(out["levels"]), 2)
        self.assertEqual(out["levels"][1]["direction"], "from_above")

    def test_accepts_boundary_values(self) -> None:
        cfg = default_config()
        cfg["long"]["lots"] = MIN_LOTS
        cfg["short"]["lots"] = MAX_LOTS
        cfg["long"]["be_offset"] = -1000.0
        cfg["short"]["be_offset"] = 1000.0
        cfg["magic"] = MAX_MAGIC
        cfg["time_window"]["start"] = "00:00"
        cfg["time_window"]["end"] = "23:59"
        cfg["time_window"]["days"] = [0, 1, 2, 3, 4, 5, 6]
        cfg["long"]["tp"] = 0.0
        cfg["long"]["sl"] = 0.0
        cfg["long"]["be_trigger"] = 0.0
        _, errors, _ = validate_config(cfg)
        self.assertEqual(errors, [])

    def test_accepts_32_levels_and_partial_body(self) -> None:
        cfg = default_config()
        cfg["levels"] = [
            {"id": f"L{i}", "label": f"lvl{i}", "price": 3000.0 + i,
             "enabled": True, "direction": "either"}
            for i in range(1, MAX_LEVELS + 1)
        ]
        out, errors, _ = validate_config(cfg)
        self.assertEqual(errors, [])
        self.assertEqual(len(out["levels"]), MAX_LEVELS)
        # a partial body is filled from defaults, not rejected
        out2, errors2, _ = validate_config({"symbol": "XAUUSD.a"})
        self.assertEqual(errors2, [])
        self.assertEqual(out2["symbol"], "XAUUSD.a")
        self.assertEqual(out2["entry"]["trigger_mode"], "cross")

    def test_disabled_level_may_have_zero_price(self) -> None:
        cfg = default_config()
        cfg["levels"] = [{"id": "L1", "label": "draft", "price": 0.0,
                          "enabled": False, "direction": "either"}]
        _, errors, _ = validate_config(cfg)
        self.assertEqual(errors, [])

    # -- section 8: reject cases ------------------------------------------

    def _expect_reject(self, mutate, needle: str) -> None:
        cfg = default_config()
        mutate(cfg)
        _, errors, _ = validate_config(cfg)
        self.assertTrue(errors, f"expected a rejection mentioning {needle!r}")
        joined = " | ".join(errors)
        self.assertIn(needle, joined, f"errors were: {joined}")

    def test_rejects_lots_out_of_range(self) -> None:
        self._expect_reject(lambda c: c["long"].__setitem__("lots", 0.009), "long.lots")
        self._expect_reject(lambda c: c["long"].__setitem__("lots", 0.0), "long.lots")
        self._expect_reject(lambda c: c["short"].__setitem__("lots", 100.01), "short.lots")
        self._expect_reject(lambda c: c["short"].__setitem__("lots", -1.0), "short.lots")
        self._expect_reject(lambda c: c["long"].__setitem__("lots", "0.1"), "long.lots")

    def test_rejects_negative_distances(self) -> None:
        self._expect_reject(lambda c: c["long"].__setitem__("tp", -0.1), "long.tp")
        self._expect_reject(lambda c: c["long"].__setitem__("sl", -1.0), "long.sl")
        self._expect_reject(lambda c: c["short"].__setitem__("be_trigger", -5.0),
                            "short.be_trigger")
        self._expect_reject(lambda c: c["entry"].__setitem__("touch_tolerance", -0.1),
                            "entry.touch_tolerance")

    def test_rejects_be_offset_out_of_range(self) -> None:
        self._expect_reject(lambda c: c["long"].__setitem__("be_offset", 1000.5),
                            "long.be_offset")
        self._expect_reject(lambda c: c["short"].__setitem__("be_offset", -1000.5),
                            "short.be_offset")

    def test_rejects_enabled_level_without_price(self) -> None:
        self._expect_reject(
            lambda c: c.__setitem__("levels", [{"id": "L1", "label": "x",
                                                "price": 0.0, "enabled": True}]),
            "levels[0].price")
        self._expect_reject(
            lambda c: c.__setitem__("levels", [{"id": "L1", "label": "x",
                                                "price": -3.0, "enabled": True}]),
            "levels[0].price")

    def test_rejects_duplicate_level_ids(self) -> None:
        self._expect_reject(
            lambda c: c.__setitem__("levels", [
                {"id": "L1", "label": "a", "price": 10.0, "enabled": True},
                {"id": "L1", "label": "b", "price": 20.0, "enabled": True}]),
            "duplicate id")

    def test_rejects_more_than_32_levels(self) -> None:
        self._expect_reject(
            lambda c: c.__setitem__("levels", [
                {"id": f"L{i}", "label": "x", "price": 3000.0 + i, "enabled": True}
                for i in range(40)]),
            "at most 32")

    def test_rejects_bad_level_id_and_label(self) -> None:
        self._expect_reject(
            lambda c: c.__setitem__("levels", [{"id": "", "label": "x",
                                                "price": 1.0, "enabled": True}]),
            "levels[0].id")
        self._expect_reject(
            lambda c: c.__setitem__("levels", [{"id": "L 1", "label": "x",
                                                "price": 1.0, "enabled": True}]),
            "levels[0].id")
        self._expect_reject(
            lambda c: c.__setitem__("levels", [{"id": "L1", "label": "y" * 25,
                                                "price": 1.0, "enabled": True}]),
            "at most 24 characters")
        self._expect_reject(
            lambda c: c.__setitem__("levels", [{"id": "L1", "label": "caf\u00e9",
                                                "price": 1.0, "enabled": True}]),
            "ASCII characters only")

    def test_rejects_bad_time_window(self) -> None:
        self._expect_reject(lambda c: c["time_window"].__setitem__("start", "8:00"),
                            "time_window.start")
        self._expect_reject(lambda c: c["time_window"].__setitem__("start", "25:00"),
                            "time_window.start")
        self._expect_reject(lambda c: c["time_window"].__setitem__("end", "17:60"),
                            "time_window.end")
        self._expect_reject(lambda c: c["time_window"].__setitem__("end", "173:0"),
                            "time_window.end")
        self._expect_reject(lambda c: c["time_window"].__setitem__("days", [1, 7]),
                            "time_window.days")
        self._expect_reject(lambda c: c["time_window"].__setitem__("days", [-1]),
                            "time_window.days")
        self._expect_reject(lambda c: c["time_window"].__setitem__("days", "12345"),
                            "time_window.days")
        self._expect_reject(lambda c: c["time_window"].__setitem__("timezone", "utc"),
                            "time_window.timezone")

    def test_rejects_bad_magic_and_symbol(self) -> None:
        self._expect_reject(lambda c: c.__setitem__("magic", 0), "magic")
        self._expect_reject(lambda c: c.__setitem__("magic", -5), "magic")
        self._expect_reject(lambda c: c.__setitem__("magic", MAX_MAGIC + 1), "magic")
        self._expect_reject(lambda c: c.__setitem__("magic", 1.5), "magic")
        self._expect_reject(lambda c: c.__setitem__("symbol", ""), "symbol")
        self._expect_reject(lambda c: c.__setitem__("symbol", "XAU USD"), "symbol")
        self._expect_reject(lambda c: c.__setitem__("symbol", "X" * 33), "symbol")
        self._expect_reject(lambda c: c.__setitem__("symbol", "XAU/USD"), "symbol")

    def test_rejects_bad_enums_and_types(self) -> None:
        self._expect_reject(lambda c: c.__setitem__("unit", "lots"), "unit")
        self._expect_reject(lambda c: c.__setitem__("pip_size_mode", "guess"),
                            "pip_size_mode")
        self._expect_reject(lambda c: c["entry"].__setitem__("trigger_mode", "poke"),
                            "entry.trigger_mode")
        self._expect_reject(
            lambda c: c.__setitem__("levels", [{"id": "L1", "label": "x", "price": 1.0,
                                                "enabled": True, "direction": "up"}]),
            "levels[0].direction")
        self._expect_reject(lambda c: c.__setitem__("enabled", "yes"), "enabled")
        self._expect_reject(lambda c: c["entry"].__setitem__("cooldown_sec", -1),
                            "entry.cooldown_sec")
        self._expect_reject(lambda c: c.__setitem__("levels", "L1"), "levels")
        self._expect_reject(lambda c: c.__setitem__("long", 3), "long")
        _, errors, _ = validate_config(["not", "an", "object"])
        self.assertTrue(errors)
        _, errors, _ = validate_config(None)
        self.assertTrue(errors)

    def test_rejects_non_finite_numbers(self) -> None:
        # json.loads accepts Infinity/NaN by default, so a client really can
        # send them; they must not reach config.json.
        body = json.loads('{"long": {"tp": Infinity}}')
        _, errors, _ = validate_config(body)
        self.assertTrue(any("finite" in e for e in errors), errors)

    # -- section 8: warn-only cases ---------------------------------------

    def test_warns_be_trigger_beyond_tp(self) -> None:
        cfg = default_config()
        cfg["long"]["tp"] = 50.0
        cfg["long"]["be_trigger"] = 80.0
        out, errors, warnings = validate_config(cfg)
        self.assertEqual(errors, [], "must be warn-only, never a rejection")
        self.assertTrue(any("be_trigger" in w for w in warnings), warnings)
        self.assertEqual(out["long"]["be_trigger"], 80.0, "still saved as sent")
        # equal is fine, and tp == 0 (no TP) must not warn
        cfg["long"]["be_trigger"] = 50.0
        _, _, warnings = validate_config(cfg)
        self.assertEqual(warnings, [])
        cfg["long"]["tp"] = 0.0
        cfg["long"]["be_trigger"] = 80.0
        _, _, warnings = validate_config(cfg)
        self.assertEqual(warnings, [])

    def test_warns_enabled_without_levels(self) -> None:
        cfg = default_config()
        cfg["enabled"] = True
        _, errors, warnings = validate_config(cfg)
        self.assertEqual(errors, [])
        self.assertTrue(any("no level is enabled" in w for w in warnings), warnings)
        # a present but disabled level still counts as "nothing can fire"
        cfg["levels"] = [{"id": "L1", "label": "x", "price": 3000.0,
                          "enabled": False}]
        _, errors, warnings = validate_config(cfg)
        self.assertEqual(errors, [])
        self.assertTrue(any("no level is enabled" in w for w in warnings), warnings)

    def test_warns_when_no_leg_enabled(self) -> None:
        cfg = default_config()
        cfg["enabled"] = True
        cfg["levels"] = [{"id": "L1", "label": "x", "price": 3000.0, "enabled": True}]
        cfg["long"]["enabled"] = False
        cfg["short"]["enabled"] = False
        _, errors, warnings = validate_config(cfg)
        self.assertEqual(errors, [])
        self.assertTrue(any("neither leg" in w for w in warnings), warnings)

    def test_warns_about_missing_and_unknown_keys(self) -> None:
        # A partial body is accepted, but the fields it did not send are reset
        # to defaults -- the user must be told, or a one-field save silently
        # wipes the level list.
        out, errors, warnings = validate_config({"enabled": False, "colour": "red"})
        self.assertEqual(errors, [])
        self.assertEqual(out["levels"], [])
        joined = " | ".join(warnings)
        self.assertIn("reset to defaults", joined)
        self.assertIn("levels", joined)
        self.assertIn("colour", joined)
        # A complete body warns about nothing.
        _, _, clean = validate_config(default_config())
        self.assertEqual(clean, [])

    # -- version ownership -------------------------------------------------

    def test_version_monotonic_and_client_value_ignored(self) -> None:
        bridge = Bridge(self.ipc)
        self.assertEqual(bridge.version, 0, "seed is 0 with no config.json")
        first = bridge.ensure_config()
        self.assertEqual(first["config_version"], 1)

        cfg, errors, _ = validate_config(default_config())
        self.assertEqual(errors, [])
        self.assertEqual(bridge.write_config(cfg)["config_version"], 2)
        self.assertEqual(bridge.write_config(cfg)["config_version"], 3)

        # a client that echoes an old version back must not roll us back
        stale = dict(cfg)
        stale["config_version"] = 1
        written = bridge.write_config({k: v for k, v in stale.items()
                                       if k != "config_version"})
        self.assertEqual(written["config_version"], 4)

        # a new server process (restart) keeps counting from the file
        restarted = Bridge(self.ipc)
        self.assertEqual(restarted.version, 4)
        self.assertEqual(restarted.write_config(cfg)["config_version"], 5)

        # and a hand edit that jumps the version forward is respected
        on_disk, _ = read_json_file(self.ipc / "config.json")
        on_disk["config_version"] = 99
        atomic_write_json(self.ipc / "config.json", on_disk)
        self.assertEqual(restarted.write_config(cfg)["config_version"], 100)

    def test_version_survives_corrupt_config(self) -> None:
        (self.ipc / "config.json").write_text("{oops", encoding="utf-8")
        bridge = Bridge(self.ipc)
        cfg = bridge.ensure_config()
        self.assertEqual(cfg["config_version"], 1)
        self.assertTrue(any("unreadable" in n for n in bridge.notes), bridge.notes)
        self.assertTrue(list(self.ipc.glob("config.json.bad-*")),
                        "the broken file must be preserved, not deleted")
        self.assertEqual(read_json_file(self.ipc / "config.json")[1], None)

    def test_written_config_key_order_matches_contract(self) -> None:
        bridge = Bridge(self.ipc)
        cfg, _, _ = validate_config(default_config())
        written = bridge.write_config(cfg)
        self.assertEqual(list(written)[:4],
                         ["config_version", "updated_at", "enabled", "symbol"])
        raw, err = read_json_file(self.ipc / "config.json")
        self.assertIsNone(err)
        self.assertEqual(raw, written)

    # -- command ids -------------------------------------------------------

    def test_command_id_assignment(self) -> None:
        bridge = Bridge(self.ipc)
        self.assertEqual(bridge.write_command("close_all", {}), 1)
        self.assertEqual(bridge.write_command("rearm_all", {}), 2)
        payload, err = read_json_file(self.ipc / "command.json")
        self.assertIsNone(err)
        self.assertEqual(payload["cmd"], "rearm_all")
        self.assertEqual(payload["args"], {})
        self.assertTrue(payload["issued_at"].endswith("Z"))

        # restart: seed from the existing command.json
        again = Bridge(self.ipc)
        self.assertEqual(again.write_command("panic", {}), 3)

    def test_command_id_seeds_from_ea_last_command_id(self) -> None:
        # A stale command.json (id 3) plus an EA that has already executed 20:
        # the next id must be 21, otherwise the EA sees an id <= last_command_id
        # and either ignores it or replays an old one.
        atomic_write_json(self.ipc / "command.json",
                          {"id": 3, "cmd": "close_all", "args": {},
                           "issued_at": utc_stamp()})
        atomic_write_json(self.ipc / "state.json", {"schema": 1,
                                                    "last_command_id": 20})
        bridge = Bridge(self.ipc)
        self.assertEqual(bridge._command_id, 20)
        self.assertEqual(bridge.write_command("close_all", {}), 21)

    def test_command_validation_accepts(self) -> None:
        for body, want_args in (
            ({"cmd": "close_all", "args": {}}, {}),
            ({"cmd": "close_all"}, {}),
            ({"cmd": "rearm_all", "args": None}, {}),
            ({"cmd": "panic", "args": {}}, {}),
            ({"cmd": "close_level", "args": {"level_id": "L1"}}, {"level_id": "L1"}),
            ({"cmd": "rearm_level", "args": {"level_id": "L12"}}, {"level_id": "L12"}),
            ({"cmd": "close_side", "args": {"side": "long"}}, {"side": "long"}),
            ({"cmd": "close_side", "args": {"side": "short"}}, {"side": "short"}),
        ):
            cmd, args, errors = validate_command(body)
            self.assertEqual(errors, [], f"{body} -> {errors}")
            self.assertEqual(cmd, body["cmd"])
            self.assertEqual(args, want_args)

    def test_command_validation_rejects(self) -> None:
        for body in (
            {"cmd": "close_everything", "args": {}},
            {"cmd": "", "args": {}},
            {"cmd": None},
            {},
            {"cmd": "CLOSE_ALL"},
            {"cmd": "close_level", "args": {}},
            {"cmd": "close_level", "args": {"level_id": ""}},
            {"cmd": "close_level", "args": {"levelId": "L1"}},
            {"cmd": "close_level", "args": {"level_id": 1}},
            {"cmd": "close_side", "args": {"side": "up"}},
            {"cmd": "close_side", "args": {}},
            {"cmd": "close_all", "args": {"level_id": "L1"}},
            {"cmd": "close_all", "args": []},
            "close_all",
            None,
        ):
            _, _, errors = validate_command(body)
            self.assertTrue(errors, f"{body!r} should have been rejected")

    def test_all_six_commands_are_known(self) -> None:
        self.assertEqual(sorted(COMMAND_ARG_KIND),
                         ["close_all", "close_level", "close_side", "panic",
                          "rearm_all", "rearm_level"])

    # -- state reading -----------------------------------------------------

    def test_state_missing_then_good_then_corrupt(self) -> None:
        bridge = Bridge(self.ipc)
        state, age, parse_error = bridge.read_state()
        self.assertIsNone(state)
        self.assertIsNone(age)
        self.assertFalse(parse_error)
        block = bridge.bridge_block(state, age, parse_error)
        self.assertFalse(block["ea_alive"])
        self.assertEqual(block["config_version_seen"], 0)

        good = {"schema": 1, "config_version_seen": 7, "last_command_id": 2}
        atomic_write_json(bridge.state_path, good)
        state, age, parse_error = bridge.read_state()
        self.assertEqual(state, good)
        self.assertFalse(parse_error)
        self.assertIsNotNone(age)
        self.assertLess(age, EA_ALIVE_MS)
        block = bridge.bridge_block(state, age, parse_error)
        self.assertTrue(block["ea_alive"])
        self.assertEqual(block["config_version_seen"], 7)
        self.assertFalse(block["in_sync"], "version 0 != seen 7")

        # now simulate catching the EA mid-write
        bridge.state_path.write_text('{"schema": 1, "conf', encoding="utf-8")
        state, age, parse_error = bridge.read_state()
        self.assertTrue(parse_error, "must flag the bad parse")
        self.assertEqual(state, good, "must fall back to the last good state")
        self.assertIsNotNone(age, "age must still come from the real mtime")

    def test_in_sync_reporting(self) -> None:
        bridge = Bridge(self.ipc)
        bridge.ensure_config()                       # version 1
        atomic_write_json(bridge.state_path, {"schema": 1, "config_version_seen": 1})
        state, age, parse_error = bridge.read_state()
        block = bridge.bridge_block(state, age, parse_error)
        self.assertTrue(block["in_sync"])
        self.assertEqual(block["ipc_dir"], str(bridge.ipc_dir))

    # -- time window -------------------------------------------------------

    def test_time_window_plain(self) -> None:
        mon = datetime(2026, 9, 7)                   # a Monday -> contract day 1
        self.assertEqual(dow_contract(mon), 1)
        days = [1, 2, 3, 4, 5]
        self.assertTrue(time_window_open("08:00", "17:30",
                                         days, mon.replace(hour=8, minute=0)))
        self.assertTrue(time_window_open("08:00", "17:30",
                                        days, mon.replace(hour=17, minute=29)))
        # inclusive start, exclusive end
        self.assertFalse(time_window_open("08:00", "17:30",
                                          days, mon.replace(hour=17, minute=30)))
        self.assertFalse(time_window_open("08:00", "17:30",
                                          days, mon.replace(hour=7, minute=59)))
        # day gating
        sun = datetime(2026, 9, 6, 10, 0)
        self.assertEqual(dow_contract(sun), 0)
        self.assertFalse(time_window_open("08:00", "17:30", days, sun))
        self.assertFalse(time_window_open("08:00", "17:30", [], mon.replace(hour=9)))

    def test_time_window_wraps_midnight(self) -> None:
        # Friday 22:00 -> 06:00, days = [Friday]
        fri = datetime(2026, 9, 11)
        self.assertEqual(dow_contract(fri), 5)
        days = [5]
        self.assertTrue(time_window_open("22:00", "06:00",
                                         days, fri.replace(hour=22, minute=0)))
        self.assertTrue(time_window_open("22:00", "06:00",
                                         days, fri.replace(hour=23, minute=59)))
        sat = datetime(2026, 9, 12)
        self.assertEqual(dow_contract(sat), 6)
        # Saturday 02:00 belongs to the window that STARTED on Friday
        self.assertTrue(time_window_open("22:00", "06:00",
                                         days, sat.replace(hour=2, minute=0)))
        self.assertFalse(time_window_open("22:00", "06:00",
                                          days, sat.replace(hour=6, minute=0)),
                         "exclusive end")
        self.assertFalse(time_window_open("22:00", "06:00",
                                          days, sat.replace(hour=12, minute=0)))
        # Saturday evening would start a Saturday window, which is not allowed
        self.assertFalse(time_window_open("22:00", "06:00",
                                          days, sat.replace(hour=23, minute=0)))
        # ...unless Saturday is in the day list
        self.assertTrue(time_window_open("22:00", "06:00", [5, 6],
                                         sat.replace(hour=23, minute=0)))

    def test_time_window_degenerate(self) -> None:
        mon = datetime(2026, 9, 7, 12, 0)
        self.assertFalse(time_window_open("12:00", "12:00", [1], mon),
                         "zero-length window is closed")
        self.assertFalse(time_window_open("bogus", "12:00", [1], mon))
        self.assertFalse(time_window_open("24:00", "12:00", [1], mon))
        self.assertFalse(time_window_open("08:00", None, [1], mon))
        self.assertEqual(minutes_of_day("00:00"), 0)
        self.assertEqual(minutes_of_day("23:59"), 1439)
        self.assertIsNone(minutes_of_day("23:60"))
        self.assertIsNone(minutes_of_day("1:00"))
        self.assertIsNone(minutes_of_day(800))

    # -- path traversal ----------------------------------------------------

    def test_rejects_nasty_static_paths(self) -> None:
        for candidate in NASTY_PATHS:
            with self.subTest(path=candidate):
                self.assertIsNone(
                    safe_static_path(self.ui, candidate),
                    f"{candidate!r} must be refused")

    def test_traversal_cannot_reach_outside_files(self) -> None:
        # belt and braces: even if a rule above changed, nothing may resolve to
        # the file we planted outside ui/
        outside = (self.root / "secret.txt").resolve()
        for candidate in NASTY_PATHS:
            got = safe_static_path(self.ui, candidate)
            self.assertNotEqual(got, outside, candidate)
            if got is not None:                     # pragma: no cover
                self.assertTrue(got.is_relative_to(self.ui.resolve()), candidate)

    def test_accepts_legitimate_static_paths(self) -> None:
        for candidate, expect in (
            ("/index.html", self.ui / "index.html"),
            ("/app.js", self.ui / "app.js"),
            ("/style.css", self.ui / "style.css"),
            ("/sub/icon.svg", self.ui / "sub" / "icon.svg"),
        ):
            with self.subTest(path=candidate):
                got = safe_static_path(self.ui, candidate)
                self.assertIsNotNone(got, candidate)
                self.assertEqual(got, expect.resolve())

    def test_extension_whitelist(self) -> None:
        # a real file inside ui/ with a non-whitelisted extension is refused
        self.assertTrue((self.ui / "secret.txt").is_file())
        self.assertIsNone(safe_static_path(self.ui, "/secret.txt"))
        for ext in sorted(ALLOWED_EXT):
            probe = self.ui / f"probe{ext}"
            probe.write_text("x", encoding="utf-8")
            self.assertIsNotNone(safe_static_path(self.ui, f"/probe{ext}"), ext)

    # -- host header -------------------------------------------------------

    def test_host_header_guard(self) -> None:
        for host in ("127.0.0.1", "127.0.0.1:8733", "localhost", "localhost:8733",
                     "LocalHost:8733", "[::1]:8733", "[::1]"):
            self.assertTrue(host_header_ok(host, 8733), host)
        for host in (None, "", "evil.com", "evil.com:8733", "127.0.0.1.evil.com",
                     "127.0.0.1:9999", "0.0.0.0:8733", "192.168.1.5:8733",
                     "localhost:8733:8733", "[::1", "attacker.localhost",
                     "127.0.0.1:abc", "::1"):
            self.assertFalse(host_header_ok(host, 8733), repr(host))


def run_selftest() -> int:
    """Run the suite, print a PASS/FAIL summary, return a shell exit code."""
    print("GoldStraddle panel server -- selftest")
    print("=" * 72)
    suite = unittest.TestLoader().loadTestsFromTestCase(_SelfTest)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    print("=" * 72)
    total = result.testsRun
    bad = len(result.failures) + len(result.errors)
    print(f" tests run : {total}")
    print(f" failures  : {len(result.failures)}")
    print(f" errors    : {len(result.errors)}")
    print(f" skipped   : {len(result.skipped)}")
    print("=" * 72)
    if bad == 0:
        print(f"RESULT: PASS ({total} tests)")
        return 0
    print(f"RESULT: FAIL ({bad} of {total} tests did not pass)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
