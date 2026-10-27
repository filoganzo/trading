"""Native Tkinter dashboard for FiloHFT.

Reads telemetry CSVs directly from the MT5 Common Files folder and
refreshes every 500 ms. No HTTP, no Streamlit, no external services —
pure stdlib Tk.

Usage:
    python -m filo_hft.gui.dashboard_tk
    python -m filo_hft.gui.dashboard_tk --telemetry-root <path>

If --telemetry-root is omitted, auto-locates
%APPDATA%/MetaQuotes/Terminal/Common/Files/telemetry on Windows.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import tkinter as tk
from tkinter import ttk

_DAY_RE = re.compile(r"^\d{8}$")


# ----------------------------------------------------------------------
# telemetry tail-reader

@dataclass
class _TailState:
    path: Path
    offset: int = 0
    header: list[str] = field(default_factory=list)
    rows: deque = field(default_factory=lambda: deque(maxlen=5000))
    last_advance_wall: float = 0.0  # time.monotonic() at last offset advance


class TelemetryTail:
    """Incrementally reads CSV files from a telemetry day folder.

    Remembers byte offsets so each refresh only parses new lines. If the
    day folder rolls (UTC midnight), the tail state is rebuilt.
    """

    STREAMS = (
        "features", "keylevels", "setups", "intents",
        "fills", "trades", "risk", "safety", "latency", "errors",
    )

    def __init__(self, root: Path) -> None:
        self.root = root
        self.day_dir: Path | None = None
        self.state: dict[str, _TailState] = {}

    def _current_day_dir(self) -> Path | None:
        # Always pick the most-recently-modified YYYYMMDD subdir rather than
        # trusting Python's UTC clock. The EA writes in broker-server time
        # (IC Markets = GMT+3), so during the window between broker-midnight
        # and real UTC midnight the "today" UTC folder is stale and the fresh
        # data is already in "tomorrow" UTC. Using mtime dodges all TZ math.
        if not self.root.exists():
            return None
        subs = [d for d in self.root.iterdir()
                if d.is_dir() and _DAY_RE.match(d.name)]
        if not subs:
            return None
        def _last_mtime(d: Path) -> float:
            best = d.stat().st_mtime
            try:
                for f in d.iterdir():
                    if f.is_file():
                        m = f.stat().st_mtime
                        if m > best:
                            best = m
            except OSError:
                pass
            return best
        subs.sort(key=_last_mtime, reverse=True)
        return subs[0]

    def refresh(self) -> None:
        day = self._current_day_dir()
        if day != self.day_dir:
            self.day_dir = day
            self.state.clear()
        if self.day_dir is None:
            return
        for name in self.STREAMS:
            path = self.day_dir / f"{name}.csv"
            if not path.exists():
                continue
            st = self.state.get(name)
            if st is None:
                st = _TailState(path=path)
                self.state[name] = st
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size < st.offset:
                # file truncated / rolled — restart
                st.offset = 0
                st.header = []
                st.rows.clear()
            if size == st.offset:
                continue
            try:
                with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                    fh.seek(st.offset)
                    chunk = fh.read()
                    st.offset = fh.tell()
            except OSError:
                continue
            if not chunk:
                continue
            st.last_advance_wall = time.monotonic()
            lines = chunk.splitlines()
            if not st.header and lines:
                hdr_line = lines[0]
                st.header = [c.strip() for c in hdr_line.split(",")]
                lines = lines[1:]
            reader = csv.DictReader(lines, fieldnames=st.header)
            for row in reader:
                if row:
                    st.rows.append(row)

    def latest(self, stream: str) -> dict | None:
        st = self.state.get(stream)
        if st is None or not st.rows:
            return None
        return st.rows[-1]

    def seconds_since_advance(self, stream: str) -> float | None:
        st = self.state.get(stream)
        if st is None or st.last_advance_wall == 0.0:
            return None
        return time.monotonic() - st.last_advance_wall

    def rows(self, stream: str) -> list[dict]:
        st = self.state.get(stream)
        return list(st.rows) if st else []


# ----------------------------------------------------------------------
# styling

BG = "#0b0f14"
PANEL = "#131a22"
BORDER = "#1f2a35"
FG = "#d6deeb"
DIM = "#7a8699"
ACCENT = "#5dc1ff"
GREEN = "#22c55e"
RED = "#ef4444"
YELLOW = "#f5c518"
FONT = ("Consolas", 10)
FONT_BOLD = ("Consolas", 10, "bold")
FONT_TITLE = ("Consolas", 11, "bold")
FONT_LARGE = ("Consolas", 16, "bold")


def _panel(parent: tk.Widget, title: str) -> tk.Frame:
    outer = tk.Frame(parent, bg=BORDER, bd=0)
    inner = tk.Frame(outer, bg=PANEL, bd=0)
    inner.pack(fill="both", expand=True, padx=1, pady=1)
    label = tk.Label(inner, text=title, bg=PANEL, fg=ACCENT,
                     font=FONT_TITLE, anchor="w")
    label.pack(fill="x", padx=8, pady=(6, 2))
    sep = tk.Frame(inner, bg=BORDER, height=1)
    sep.pack(fill="x", padx=6)
    body = tk.Frame(inner, bg=PANEL)
    body.pack(fill="both", expand=True, padx=8, pady=(4, 8))
    outer.body = body  # type: ignore[attr-defined]
    return outer


def _kv(parent: tk.Widget, key: str, row: int) -> tk.Label:
    tk.Label(parent, text=key, bg=PANEL, fg=DIM, font=FONT,
             anchor="w").grid(row=row, column=0, sticky="w", padx=(0, 10))
    val = tk.Label(parent, text="--", bg=PANEL, fg=FG, font=FONT_BOLD,
                   anchor="e")
    val.grid(row=row, column=1, sticky="e")
    parent.grid_columnconfigure(1, weight=1)
    return val


# ----------------------------------------------------------------------
# the app

class Dashboard:
    def __init__(self, root: tk.Tk, tele: TelemetryTail) -> None:
        self.root = root
        self.tele = tele
        self.root.title("FiloHFT — live dashboard")
        self.root.configure(bg=BG)
        self.root.geometry("1400x820")
        self.root.minsize(1200, 700)

        self._build_header()
        self._build_body()
        self._build_footer()

        self.root.after(200, self._tick)

    # -- layout ---------------------------------------------------------
    def _build_header(self) -> None:
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=12, pady=(10, 4))

        # big price block
        left = tk.Frame(top, bg=BG)
        left.pack(side="left")
        tk.Label(left, text="XAUUSD", bg=BG, fg=DIM, font=FONT).pack(anchor="w")
        self.lbl_bid_ask = tk.Label(left, text="-- / --", bg=BG, fg=FG, font=FONT_LARGE)
        self.lbl_bid_ask.pack(anchor="w")
        self.lbl_mid_spread = tk.Label(left, text="mid -- spread --",
                                       bg=BG, fg=DIM, font=FONT)
        self.lbl_mid_spread.pack(anchor="w")

        # status cluster
        right = tk.Frame(top, bg=BG)
        right.pack(side="right")
        self.lbl_clock = tk.Label(right, text="--", bg=BG, fg=FG, font=FONT_BOLD)
        self.lbl_clock.pack(anchor="e")
        self.lbl_last_tick = tk.Label(right, text="telemetry: --",
                                      bg=BG, fg=DIM, font=FONT)
        self.lbl_last_tick.pack(anchor="e")
        self.lbl_telemetry_dir = tk.Label(
            right,
            text=f"{self.tele.root}",
            bg=BG, fg=DIM, font=FONT,
        )
        self.lbl_telemetry_dir.pack(anchor="e")

    def _build_body(self) -> None:
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=4)

        body.grid_columnconfigure(0, weight=1, uniform="col")
        body.grid_columnconfigure(1, weight=1, uniform="col")
        body.grid_columnconfigure(2, weight=1, uniform="col")
        body.grid_rowconfigure(0, weight=0)
        body.grid_rowconfigure(1, weight=1)

        # row 0 — risk / safety / open positions
        risk_panel = _panel(body, "RISK")
        risk_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        rb = risk_panel.body  # type: ignore[attr-defined]
        self.lbl_kill = _kv(rb, "kill switch", 0)
        self.lbl_last_risk = _kv(rb, "last trip", 1)
        self.lbl_last_risk_val = _kv(rb, "value / cap", 2)

        safety_panel = _panel(body, "SAFETY")
        safety_panel.grid(row=0, column=1, sticky="nsew", padx=6, pady=(0, 6))
        sb = safety_panel.body  # type: ignore[attr-defined]
        self.lbl_safety = _kv(sb, "halt", 0)
        self.lbl_last_safety = _kv(sb, "last trip", 1)
        self.lbl_last_safety_val = _kv(sb, "value / cap", 2)

        pos_panel = _panel(body, "POSITIONS")
        pos_panel.grid(row=0, column=2, sticky="nsew", padx=(6, 0), pady=(0, 6))
        pb = pos_panel.body  # type: ignore[attr-defined]
        self.lbl_open_count = _kv(pb, "open", 0)
        self.lbl_fills_total = _kv(pb, "fills (session)", 1)
        self.lbl_closes_total = _kv(pb, "closes (session)", 2)

        # row 1 — KL list / setups feed / fills+trades feed
        kl_panel = _panel(body, "KEY LEVELS")
        kl_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.tree_kl = self._tree(kl_panel.body,  # type: ignore[attr-defined]
                                  columns=("center", "tf", "wt", "dist"),
                                  widths=(90, 40, 55, 60))

        setups_panel = _panel(body, "SETUPS — accepted / blocked")
        setups_panel.grid(row=1, column=1, sticky="nsew", padx=6)
        self.tree_setups = self._tree(
            setups_panel.body,  # type: ignore[attr-defined]
            columns=("ts", "family", "dir", "decision", "why"),
            widths=(60, 90, 40, 70, 160),
        )

        trades_panel = _panel(body, "FILLS + TRADES")
        trades_panel.grid(row=1, column=2, sticky="nsew", padx=(6, 0))
        self.tree_trades = self._tree(
            trades_panel.body,  # type: ignore[attr-defined]
            columns=("ts", "event", "dir", "price", "pnl"),
            widths=(60, 60, 40, 80, 70),
        )

    def _build_footer(self) -> None:
        foot = tk.Frame(self.root, bg=BG)
        foot.pack(fill="x", padx=12, pady=(4, 10))
        self.lbl_latency = tk.Label(
            foot,
            text="latency — tick handler: --   order send: --   errors: 0",
            bg=BG, fg=DIM, font=FONT,
        )
        self.lbl_latency.pack(side="left")
        self.lbl_status = tk.Label(
            foot, text="waiting for telemetry...", bg=BG, fg=YELLOW, font=FONT,
        )
        self.lbl_status.pack(side="right")

    def _tree(self, parent: tk.Widget, columns: tuple[str, ...],
              widths: tuple[int, ...]) -> ttk.Treeview:
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Filo.Treeview",
            background=PANEL, foreground=FG, fieldbackground=PANEL,
            bordercolor=BORDER, borderwidth=0, font=FONT, rowheight=20,
        )
        style.configure(
            "Filo.Treeview.Heading",
            background=BORDER, foreground=ACCENT, font=FONT_BOLD,
            relief="flat",
        )
        style.map("Filo.Treeview",
                  background=[("selected", "#1e3a5f")],
                  foreground=[("selected", FG)])
        tree = ttk.Treeview(parent, columns=columns, show="headings",
                            style="Filo.Treeview", height=14)
        for c, w in zip(columns, widths):
            tree.heading(c, text=c.upper())
            tree.column(c, width=w, anchor="w", stretch=True)
        tree.pack(fill="both", expand=True)
        tree.tag_configure("accepted", foreground=GREEN)
        tree.tag_configure("blocked", foreground=DIM)
        tree.tag_configure("long", foreground=GREEN)
        tree.tag_configure("short", foreground=RED)
        tree.tag_configure("open", foreground=ACCENT)
        tree.tag_configure("close_win", foreground=GREEN)
        tree.tag_configure("close_loss", foreground=RED)
        tree.tag_configure("be", foreground=YELLOW)
        tree.tag_configure("trail", foreground=YELLOW)
        return tree

    # -- periodic refresh ----------------------------------------------
    def _tick(self) -> None:
        try:
            self.tele.refresh()
            self._update_header()
            self._update_risk()
            self._update_safety()
            self._update_positions()
            self._update_kl()
            self._update_setups()
            self._update_trades()
            self._update_latency()
        except Exception as exc:  # pragma: no cover
            self.lbl_status.configure(
                text=f"refresh error: {exc}", fg=RED
            )
        self.root.after(500, self._tick)

    # -- updaters ------------------------------------------------------
    def _update_header(self) -> None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.lbl_clock.configure(text=now)

        f = self.tele.latest("features")
        if f is None:
            self.lbl_status.configure(text="waiting for telemetry...", fg=YELLOW)
            return
        try:
            bid = float(f.get("bid", 0) or 0)
            ask = float(f.get("ask", 0) or 0)
            mid = float(f.get("mid", 0) or 0)
            spread = float(f.get("spread_pips", 0) or 0)
        except ValueError:
            return
        self.lbl_bid_ask.configure(text=f"{bid:.2f} / {ask:.2f}")
        self.lbl_mid_spread.configure(
            text=f"mid {mid:.2f}   spread {spread:.2f} pips   {f.get('session', '')}"
        )
        # Freshness = how long since we last saw the features file grow.
        # EA timestamps are in broker-local time and can't be compared to
        # Python's UTC clock directly.
        age = self.tele.seconds_since_advance("features")
        if age is None:
            self.lbl_status.configure(text="no data", fg=YELLOW)
            self.lbl_last_tick.configure(text="features: --")
        elif age < 2:
            self.lbl_status.configure(text=f"live ({age:.1f}s)", fg=GREEN)
            self.lbl_last_tick.configure(text=f"features updated {age:.1f}s ago")
        elif age < 10:
            self.lbl_status.configure(text=f"stale ({age:.1f}s)", fg=YELLOW)
            self.lbl_last_tick.configure(text=f"features updated {age:.1f}s ago")
        else:
            self.lbl_status.configure(text=f"dead ({age:.0f}s)", fg=RED)
            self.lbl_last_tick.configure(text=f"features updated {age:.0f}s ago")

    def _update_risk(self) -> None:
        rows = self.tele.rows("risk")
        trips = [r for r in rows if (r.get("action") or "").lower() == "halt"]
        if trips:
            last = trips[-1]
            self.lbl_kill.configure(text="ENGAGED", fg=RED)
            self.lbl_last_risk.configure(text=last.get("limit", "--"))
            self.lbl_last_risk_val.configure(
                text=f"{last.get('value', '--')} / {last.get('threshold', '--')}"
            )
        else:
            self.lbl_kill.configure(text="clear", fg=GREEN)
            self.lbl_last_risk.configure(text="--")
            self.lbl_last_risk_val.configure(text="--")

    def _update_safety(self) -> None:
        rows = self.tele.rows("safety")
        halts = [r for r in rows if (r.get("action") or "").lower() == "halt"]
        warns = [r for r in rows if (r.get("action") or "").lower() == "warn"]

        # Derive "now" from the latest feature tick (EA-clock, timezone-agnostic).
        # A halt row is only *active* if its age on the EA clock is below
        # halt_seconds. Safety halts are sliding — they auto-expire.
        f_latest = self.tele.latest("features")
        try:
            ea_now_ms = int((f_latest or {}).get("ts_ms", 0) or 0)
        except (TypeError, ValueError):
            ea_now_ms = 0
        halt_window_ms = 30 * 1000  # matches default.ini safety.halt_seconds

        active = None
        if halts and ea_now_ms > 0:
            last = halts[-1]
            try:
                last_ts = int(last.get("ts_ms", 0) or 0)
                if 0 <= ea_now_ms - last_ts < halt_window_ms:
                    active = last
            except (TypeError, ValueError):
                pass

        if active is not None:
            self.lbl_safety.configure(text="HALTED", fg=RED)
            self.lbl_last_safety.configure(text=active.get("category", "--"))
            self.lbl_last_safety_val.configure(
                text=f"{active.get('value', '--')} / {active.get('threshold', '--')}"
            )
        elif halts:
            last = halts[-1]
            self.lbl_safety.configure(text="clear", fg=GREEN)
            self.lbl_last_safety.configure(
                text=f"{last.get('category', '--')} (expired)"
            )
            self.lbl_last_safety_val.configure(
                text=f"{last.get('value', '--')} / {last.get('threshold', '--')}"
            )
        elif warns:
            last = warns[-1]
            self.lbl_safety.configure(text="warn", fg=YELLOW)
            self.lbl_last_safety.configure(text=last.get("category", "--"))
            self.lbl_last_safety_val.configure(
                text=f"{last.get('value', '--')} / {last.get('threshold', '--')}"
            )
        else:
            self.lbl_safety.configure(text="clear", fg=GREEN)
            self.lbl_last_safety.configure(text="--")
            self.lbl_last_safety_val.configure(text="--")

    def _update_positions(self) -> None:
        fills = self.tele.rows("fills")
        trades = self.tele.rows("trades")
        opens = sum(1 for t in trades if (t.get("event") or "") == "open")
        closes = sum(1 for t in trades if (t.get("event") or "") == "close")
        self.lbl_open_count.configure(
            text=str(opens - closes),
            fg=GREEN if opens > closes else FG,
        )
        self.lbl_fills_total.configure(text=str(len(fills)))
        self.lbl_closes_total.configure(text=str(closes))

    def _update_kl(self) -> None:
        rows = self.tele.rows("keylevels")
        # collapse events to latest state per zone_id
        latest: dict[str, dict] = {}
        for r in rows:
            zid = r.get("zone_id")
            if not zid:
                continue
            latest[zid] = r
        f = self.tele.latest("features")
        try:
            mid = float(f.get("mid", 0)) if f else 0.0
        except ValueError:
            mid = 0.0
        entries: list[tuple[float, dict]] = []
        for r in latest.values():
            try:
                center = float(r.get("center", 0) or 0)
            except ValueError:
                continue
            if center <= 0:
                continue
            dist = abs(center - mid) if mid > 0 else 1e9
            entries.append((dist, r))
        entries.sort(key=lambda x: x[0])

        self.tree_kl.delete(*self.tree_kl.get_children())
        for dist, r in entries[:30]:
            # display distance in pips (XAUUSD: 1 pip = 0.10 price)
            dist_pips = dist / 0.10 if mid > 0 else 0.0
            try:
                wt = float(r.get("weight", 0) or 0)
            except ValueError:
                wt = 0.0
            self.tree_kl.insert(
                "", "end",
                values=(
                    f"{float(r.get('center', 0)):.2f}",
                    r.get("origin_tf", "--"),
                    f"{wt:.2f}",
                    f"{dist_pips:.1f}" if mid > 0 else "--",
                ),
            )

    def _update_setups(self) -> None:
        rows = self.tele.rows("setups")
        self.tree_setups.delete(*self.tree_setups.get_children())
        for r in list(rows)[-30:][::-1]:
            decision = (r.get("decision") or "").lower()
            tag = "accepted" if decision == "accepted" else "blocked"
            ts_short = _short_ts(r.get("ts_ms"))
            why = r.get("block_reason") or ""
            if decision == "accepted":
                why = f"score={r.get('intrinsic_score', '--')}"
            self.tree_setups.insert(
                "", "end",
                values=(ts_short, r.get("family", "--"),
                        r.get("direction", "--"), decision, why[:40]),
                tags=(tag,),
            )

    def _update_trades(self) -> None:
        fills = self.tele.rows("fills")
        trades = self.tele.rows("trades")
        merged: list[tuple[int, str, dict]] = []
        for r in fills:
            try:
                merged.append((int(r.get("ts_ms", 0) or 0), "fill", r))
            except ValueError:
                pass
        for r in trades:
            try:
                merged.append((int(r.get("ts_ms", 0) or 0), "trade", r))
            except ValueError:
                pass
        merged.sort(key=lambda x: x[0])

        self.tree_trades.delete(*self.tree_trades.get_children())
        for ts_ms, kind, r in merged[-40:][::-1]:
            ts_short = _short_ts(str(ts_ms))
            if kind == "fill":
                ev = "fill"
                direction = r.get("direction", "--")
                price = r.get("actual_price", "--")
                pnl = f"slip {r.get('slippage_pips', '--')}p"
                tag = direction.lower()
            else:
                ev = r.get("event", "--")
                direction = r.get("direction", "--")
                price = r.get("price", "--")
                try:
                    realized = float(r.get("realized_usd", 0) or 0)
                    pnl = f"${realized:+.2f}"
                    if ev == "close":
                        tag = "close_win" if realized > 0 else "close_loss"
                    else:
                        tag = ev
                except ValueError:
                    pnl = "--"
                    tag = ev
            self.tree_trades.insert(
                "", "end",
                values=(ts_short, ev, direction, price, pnl),
                tags=(tag,),
            )

    def _update_latency(self) -> None:
        rows = self.tele.rows("latency")
        tick_samples = [
            float(r["elapsed_ms"])
            for r in rows[-200:]
            if (r.get("stage") or "") == "tick_handler"
            and (r.get("elapsed_ms") or "").replace(".", "", 1).replace("-", "", 1).isdigit()
        ]
        order_samples = [
            float(r["elapsed_ms"])
            for r in rows[-200:]
            if (r.get("stage") or "") == "order_send"
            and (r.get("elapsed_ms") or "").replace(".", "", 1).replace("-", "", 1).isdigit()
        ]

        def _p95(xs: list[float]) -> str:
            if not xs:
                return "--"
            xs = sorted(xs)
            return f"{xs[int(len(xs) * 0.95)]:.2f}ms"

        errors = len(self.tele.rows("errors"))
        self.lbl_latency.configure(
            text=(
                f"latency — tick handler p95: {_p95(tick_samples)}   "
                f"order send p95: {_p95(order_samples)}   "
                f"errors: {errors}"
            )
        )


def _short_ts(ts_ms_str) -> str:
    try:
        ms = int(ts_ms_str or 0)
        if ms <= 0:
            return "--"
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return "--"


# ----------------------------------------------------------------------
# entry point

def _auto_root() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        p = Path(appdata) / "MetaQuotes" / "Terminal" / "Common" / "Files" / "telemetry"
        if p.exists():
            return p
    return Path("telemetry")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--telemetry-root",
        type=Path,
        default=None,
        help="Telemetry root (contains YYYYMMDD/*.csv). "
             "Defaults to MT5 Common Files folder.",
    )
    args = parser.parse_args()
    root_path = args.telemetry_root or _auto_root()
    tele = TelemetryTail(root_path)
    root = tk.Tk()
    Dashboard(root, tele)
    root.mainloop()


if __name__ == "__main__":
    main()
