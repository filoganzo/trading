#!/usr/bin/env python3
"""GoldStraddle panel launcher.

Starts :mod:`server` on a free loopback port and opens the UI in a Chromium
*app-mode* window so it behaves like a desktop application: no tabs, no
omnibox, no extensions, its own taskbar entry.  When that window is closed the
launcher shuts the server down and exits, so the whole panel has a single
lifetime and there is no orphaned Python process holding port 8733.

Standard library only.

    python launch.py                  # pick a port, open a browser window
    python launch.py --port 9000
    python launch.py --no-browser     # headless: just run the server
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Sequence

# server.py lives next to this file; make it importable no matter what the
# current working directory is (a .bat double-click starts in system32 often
# enough that this matters).
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import server as srv  # noqa: E402  (path shim above must run first)

PORT_PROBE_LIMIT = 40           # 8733 .. 8772
WINDOW_SIZE = "1280,900"

# Chromium-family browsers, in preference order.  Chrome first because its
# --app mode is the most reliable, then Edge, which is always present on
# Windows 11.  Both paths below were confirmed on this machine.
BROWSER_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("chrome", r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    ("chrome", r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ("chrome", r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ("edge", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ("edge", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ("edge", r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
)


def find_browser() -> tuple[str, str] | None:
    """Return ``(kind, exe_path)`` for the first Chromium browser found."""
    for kind, raw in BROWSER_CANDIDATES:
        path = Path(os.path.expandvars(raw))
        if path.is_file():
            return kind, str(path)
    # Last resort: whatever is on PATH.
    for kind, exe in (("chrome", "chrome.exe"), ("edge", "msedge.exe")):
        found = shutil.which(exe)
        if found:
            return kind, found
    return None


def port_is_free(port: int, host: str = srv.BIND_HOST) -> bool:
    """True if we can bind ``host:port`` right now.

    No ``SO_REUSEADDR``: on Windows that option can let a second bind succeed
    on a port that is already serving, which would make the probe lie and give
    us two half-working panels.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(preferred: int) -> int:
    """Return ``preferred`` if free, else probe upward."""
    for offset in range(PORT_PROBE_LIMIT):
        candidate = preferred + offset
        if candidate > 65535:
            break
        if port_is_free(candidate):
            return candidate
    raise SystemExit(
        f"ERROR: no free port in {preferred}..{preferred + PORT_PROBE_LIMIT - 1} "
        "on 127.0.0.1. Is another panel already running?"
    )


def browser_profile_dir() -> Path:
    """A dedicated Chromium profile directory for the panel window.

    Using ``--user-data-dir`` is what actually guarantees a separate window
    with no tab strip and no extensions, and it keeps the panel out of the
    user's real browser profile (no history entries, no session restore
    surprises, and closing the panel never closes their other windows).
    """
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Local"
    profile = root / "GoldStraddlePanel" / "browser-profile"
    profile.mkdir(parents=True, exist_ok=True)
    return profile


def launch_app_window(exe: str, url: str) -> subprocess.Popen[bytes] | None:
    """Open ``url`` in a Chromium app window.  Returns the process, or None."""
    args = [
        exe,
        f"--app={url}",
        f"--user-data-dir={browser_profile_dir()}",
        f"--window-size={WINDOW_SIZE}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-features=Translate,ChromeWhatsNewUI",
        "--no-service-autorun",
        "--disable-background-mode",
        "--disable-extensions",
        "--disable-background-networking",
    ]
    try:
        return subprocess.Popen(args)
    except OSError as exc:
        print(f"WARNING: could not start {exe}: {exc}", file=sys.stderr)
        return None


def wait_until_serving(port: int, timeout: float = 10.0) -> bool:
    """Poll ``/api/health`` until the server answers (or we give up)."""
    deadline = time.monotonic() + timeout
    url = f"http://{srv.BIND_HOST}:{port}/api/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.05)
    return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the GoldStraddle panel (server + app window).")
    parser.add_argument("--port", type=int, default=srv.DEFAULT_PORT,
                        help=f"preferred port (default {srv.DEFAULT_PORT}); "
                             "the launcher probes upward if it is taken")
    parser.add_argument("--no-browser", action="store_true",
                        help="run the server only; do not open a window")
    parser.add_argument("--ipc-dir", default=None,
                        help="override the MT5 Common Files IPC directory")
    parser.add_argument("--ui-dir", default=None,
                        help="override the static UI directory (debug)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    port = pick_port(args.port)
    if port != args.port:
        print(f"note: port {args.port} was busy, using {port} instead")

    ipc_dir = srv.resolve_ipc_dir(args.ipc_dir)
    ui_dir = Path(args.ui_dir).expanduser().resolve() if args.ui_dir else HERE / "ui"
    bridge = srv.Bridge(ipc_dir)
    bridge.ensure_config()

    try:
        httpd = srv.build_server(bridge, port, ui_dir)
    except OSError as exc:
        print(f"ERROR: cannot bind {srv.BIND_HOST}:{port} - {exc}", file=sys.stderr)
        return 2

    srv.print_banner(bridge, port, ui_dir)

    # The server runs in a background thread so this thread stays free to
    # supervise the browser process (and to receive Ctrl+C, which on Windows
    # is only delivered to the main thread).
    thread = threading.Thread(target=httpd.serve_forever,
                              kwargs={"poll_interval": 0.2},
                              name="panel-http", daemon=True)
    thread.start()

    stopped = threading.Event()

    def stop_server() -> None:
        if stopped.is_set():
            return
        stopped.set()
        with contextlib.suppress(Exception):
            httpd.shutdown()          # unblocks serve_forever()
        with contextlib.suppress(Exception):
            httpd.server_close()      # releases the listening socket
        thread.join(timeout=5.0)

    atexit.register(stop_server)      # covers unexpected exits too

    url = f"http://{srv.BIND_HOST}:{port}/"
    proc: subprocess.Popen[bytes] | None = None

    try:
        if args.no_browser:
            print("running headless (--no-browser). Ctrl+C to stop.", flush=True)
            while thread.is_alive():
                thread.join(timeout=0.5)
            return 0

        if not wait_until_serving(port):
            print("WARNING: the server did not answer /api/health in time; "
                  "opening the window anyway", file=sys.stderr)

        found = find_browser()
        if found:
            kind, exe = found
            print(f"opening app window via {kind}: {exe}", flush=True)
            proc = launch_app_window(exe, url)
        else:
            print("no Chrome/Edge found at the standard paths", file=sys.stderr)

        if proc is None:
            # Fallback: the default browser, as a normal tab.  We cannot watch
            # a tab's lifetime, so the panel then stays up until Ctrl+C.
            print(f"falling back to the default browser: {url}", flush=True)
            webbrowser.open(url)
            print("close this console (or press Ctrl+C) to stop the panel.",
                  flush=True)
            while thread.is_alive():
                thread.join(timeout=0.5)
            return 0

        # Closing the app window ends the whole panel: that is what makes this
        # feel like a desktop app rather than a server plus a stray tab.
        proc.wait()
        print("browser window closed - shutting the panel down", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\nstopping (Ctrl+C)", flush=True)
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.terminate()
        return 0
    finally:
        stop_server()


if __name__ == "__main__":
    sys.exit(main())
