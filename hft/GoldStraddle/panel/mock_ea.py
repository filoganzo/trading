"""
GoldStraddle - mock EA (test harness, NOT for trading)
======================================================

Stands in for GoldStraddle.mq5 so the panel can be driven end to end without
MetaTrader running: it reads config.json, random-walks a fake XAUUSD price,
runs the same trigger / TP / breakeven state machine described in CONTRACT.md,
and publishes state.json on the same 500 ms cadence as the real EA.

Use it to check the panel's wiring, the level state badges, the breakeven
display and the command buttons. It never talks to a broker and it knows
nothing about your account.

    python mock_ea.py                 # walk around the last real price, or 3550
    python mock_ea.py --price 3548.5  # start somewhere specific
    python mock_ea.py --vol 0.8       # livelier ticks, to make levels fire
    python mock_ea.py --sweep         # slowly sweep price across all levels

Ctrl+C to stop. It writes a final state.json marking itself stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

EA_VERSION = "1.0.0-mock"
SCHEMA = 1
HEARTBEAT_MS = 500

# Pretend to be an IC Markets EU demo box.
FAKE_SERVER_GMT_OFFSET = 2.0
FAKE_ACCOUNT = {
    "login": 52000001,
    "server": "<broker-server>",
    "currency": "EUR",
    "balance": 100000.0,
    "leverage": 500,
    "margin_mode": "hedging",
    "hedging_ok": True,
}

# XAUUSD as IC Markets quotes it: 2 digits, 100 oz contract.
SYM_DIGITS = 2
SYM_POINT = 0.01
SYM_PIP = 0.10
SYM_CONTRACT = 100.0
SPREAD_POINTS_BASE = 22


# --------------------------------------------------------------------------
#  paths
# --------------------------------------------------------------------------
def default_ipc_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return base / "MetaQuotes" / "Terminal" / "Common" / "Files" / "GoldStraddle"


def read_json(path: Path) -> dict[str, Any] | None:
    """Tolerant read - a torn file mid-write is normal, so just return None."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def write_json_atomic(path: Path, obj: dict[str, Any]) -> None:
    """Same .tmp + replace dance the real EA does, so readers never tear."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
#  unit resolution - must match CONTRACT.md section 2 exactly
# --------------------------------------------------------------------------
def price_distance(x: float, unit: str, pip: float) -> float:
    if unit == "price":
        return x
    if unit == "points":
        return x * SYM_POINT
    return x * pip  # "pips"


def money_per_price(lots: float) -> float:
    """Account-currency P/L for one unit of price move. EUR account, so the
    USD contract value is fudged by a flat 1.0 - close enough for a mock."""
    return lots * SYM_CONTRACT


# --------------------------------------------------------------------------
#  the mock
# --------------------------------------------------------------------------
class MockEA:
    def __init__(self, ipc: Path, price: float, vol: float, sweep: bool) -> None:
        self.ipc = ipc
        self.cfg_path = ipc / "config.json"
        self.state_path = ipc / "state.json"
        self.cmd_path = ipc / "command.json"

        self.bid = price
        self.vol = vol
        self.sweep = sweep
        self.sweep_dir = 1.0
        self.prev_bid = price          # seeded, never left at 0 (real EA bug class)
        self.tick_n = 0

        self.cfg: dict[str, Any] = {}
        self.cfg_version_seen = -1
        self.cfg_error = ""

        self.levels: dict[str, dict[str, Any]] = {}   # id -> runtime state
        self.positions: list[dict[str, Any]] = []
        self.next_ticket = 500000001
        self.log: list[dict[str, str]] = []
        self.last_command_id = 0
        self.last_command_result = ""
        self.panic_latch = False
        self.session_fires = 0
        self.session_closed = 0.0
        self.running = True

        self.say("info", f"mock EA started, IPC={ipc}")

    # ---- logging ---------------------------------------------------------
    def say(self, lvl: str, msg: str) -> None:
        self.log.append({"t": datetime.now().strftime("%H:%M:%S"), "lvl": lvl, "msg": msg})
        del self.log[:-60]
        print(f"[{lvl:5}] {msg}")

    # ---- config ----------------------------------------------------------
    def poll_config(self) -> None:
        raw = read_json(self.cfg_path)
        if raw is None:
            self.cfg_error = "config.json unreadable" if self.cfg_path.exists() else "no config.json yet"
            return
        self.cfg_error = ""
        ver = int(raw.get("config_version", 0))
        if ver == self.cfg_version_seen:
            return
        self.cfg = raw
        self.cfg_version_seen = ver
        self.say("info", f"config v{ver} applied")
        if raw.get("enabled"):
            self.panic_latch = False
        self.sync_levels()

    def sync_levels(self) -> None:
        """Re-arm new/changed levels, drop removed ones. Mirrors the real EA:
        a level whose price the market is already sitting on goes to 'waiting',
        never straight to 'armed'."""
        cfg_levels = self.cfg.get("levels") or []
        seen = set()
        tol = self.tol_price()
        for lv in cfg_levels:
            lid = str(lv.get("id", ""))
            if not lid:
                continue
            seen.add(lid)
            rt = self.levels.get(lid)
            if rt is None or abs(float(rt["price"]) - float(lv.get("price", 0))) > 1e-9:
                self.levels[lid] = self.fresh_level(lv, tol)
            else:
                rt["label"] = lv.get("label", "")
                rt["enabled"] = bool(lv.get("enabled", True))
                rt["direction"] = lv.get("direction", "either")
        for lid in list(self.levels):
            if lid not in seen:
                del self.levels[lid]

    def fresh_level(self, lv: dict[str, Any], tol: float) -> dict[str, Any]:
        price = float(lv.get("price", 0))
        if self.bid < price - tol:
            side, st = "below", "armed"
        elif self.bid > price + tol:
            side, st = "above", "armed"
        else:
            side, st = "unknown", "waiting"      # the anti-instant-fire rule
        return {
            "id": lv.get("id"), "label": lv.get("label", ""), "price": price,
            "enabled": bool(lv.get("enabled", True)),
            "direction": lv.get("direction", "either"),
            "state": st, "arm_side": side,
            "last_fire_utc": "", "fire_count": 0, "cooldown_until": 0.0,
        }

    # ---- helpers ---------------------------------------------------------
    def unit(self) -> str:
        return self.cfg.get("unit", "pips")

    def pip(self) -> float:
        if self.cfg.get("pip_size_mode") == "manual":
            return float(self.cfg.get("pip_size_manual", SYM_PIP)) or SYM_PIP
        return SYM_PIP

    def tol_price(self) -> float:
        entry = self.cfg.get("entry") or {}
        return price_distance(float(entry.get("touch_tolerance", 0) or 0), self.unit(), self.pip())

    @property
    def ask(self) -> float:
        return round(self.bid + SPREAD_POINTS_BASE * SYM_POINT, SYM_DIGITS)

    def window_open(self) -> tuple[bool, str]:
        tw = self.cfg.get("time_window") or {}
        if not tw.get("enabled", False):
            return True, "no time restriction"
        now = self.server_now() if tw.get("timezone", "server") == "server" else datetime.now()
        try:
            sh, sm = (int(v) for v in str(tw.get("start", "00:00")).split(":"))
            eh, em = (int(v) for v in str(tw.get("end", "23:59")).split(":"))
        except ValueError:
            return False, "bad window times"
        cur, start, end = now.hour * 60 + now.minute, sh * 60 + sm, eh * 60 + em
        if start <= end:
            in_win, day_ref = (start <= cur < end), now
        else:
            # wrap-around: after midnight we still belong to yesterday's window
            in_win = cur >= start or cur < end
            day_ref = now if cur >= start else now - timedelta(days=1)
        days = tw.get("days")
        if days is None:
            days = [1, 2, 3, 4, 5]
        # python weekday(): Mon=0..Sun=6  ->  contract: Sun=0..Sat=6
        if ((day_ref.weekday() + 1) % 7) not in days:
            return False, "day not enabled"
        return (in_win, "in window" if in_win else "outside time window")

    def server_now(self) -> datetime:
        return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=FAKE_SERVER_GMT_OFFSET)

    def blockers(self) -> list[str]:
        out = []
        if not self.cfg:
            out.append("no config")
        if self.cfg_error:
            out.append(self.cfg_error)
        if not self.cfg.get("enabled"):
            out.append("master switch off")
        if self.panic_latch:
            out.append("panic latch - re-arm to clear")
        ok, why = self.window_open()
        if not ok:
            out.append(why)
        if not FAKE_ACCOUNT["hedging_ok"]:
            out.append("netting account - straddles impossible")
        return out

    # ---- price -----------------------------------------------------------
    def tick(self) -> None:
        self.prev_bid = self.bid
        self.tick_n += 1
        if self.sweep:
            # deterministic slow sweep so every level gets visited
            self.bid += self.sweep_dir * self.vol * 0.05
            span = [l["price"] for l in self.levels.values()] or [self.bid]
            if self.bid > max(span) + 2 or self.bid < min(span) - 2:
                self.sweep_dir *= -1
        else:
            drift = math.sin(self.tick_n / 240.0) * 0.004
            self.bid += random.gauss(drift, self.vol * 0.02)
        self.bid = round(self.bid, SYM_DIGITS)

    # ---- trigger machine -------------------------------------------------
    def eval_levels(self) -> None:
        armed = not self.blockers()
        tol = self.tol_price()
        entry = self.cfg.get("entry") or {}
        one_shot = bool(entry.get("one_shot", True))
        cooldown = float(entry.get("cooldown_sec", 60) or 0)
        max_conc = int(entry.get("max_concurrent_straddles", 0) or 0)
        mode = entry.get("trigger_mode", "cross")
        now = time.time()

        live_ids = {p["level_id"] for p in self.positions}

        for rt in self.levels.values():
            if not rt["enabled"]:
                rt["state"] = "disabled"
                continue
            if rt["state"] == "disabled":
                rt["state"] = "waiting"

            if rt["state"] == "cooldown" and now >= rt["cooldown_until"]:
                rt.update(self.fresh_level(rt, tol))
                self.say("info", f"{rt['id']} re-armed after cooldown")

            # a 'waiting' level arms as soon as price clears the band
            if rt["state"] == "waiting":
                if self.bid < rt["price"] - tol:
                    rt["state"], rt["arm_side"] = "armed", "below"
                    self.say("info", f"{rt['id']} armed from below")
                elif self.bid > rt["price"] + tol:
                    rt["state"], rt["arm_side"] = "armed", "above"
                    self.say("info", f"{rt['id']} armed from above")

            if rt["id"] in live_ids and rt["state"] not in ("spent", "cooldown"):
                rt["state"] = "live"

            if rt["state"] != "armed" or not armed:
                continue

            hit = False
            if mode == "touch":
                hit = abs(self.bid - rt["price"]) <= max(tol, SYM_POINT)
            else:
                if rt["arm_side"] == "below":
                    hit = self.bid >= rt["price"] - tol
                elif rt["arm_side"] == "above":
                    hit = self.bid <= rt["price"] + tol

            if not hit:
                continue
            d = rt["direction"]
            if d == "from_below" and rt["arm_side"] != "below":
                continue
            if d == "from_above" and rt["arm_side"] != "above":
                continue
            if max_conc and len({p["level_id"] for p in self.positions}) >= max_conc:
                self.say("warn", f"{rt['id']} hit but max_concurrent_straddles reached")
                continue

            self.fire(rt)
            rt["fire_count"] += 1
            rt["last_fire_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            self.session_fires += 1
            if one_shot:
                rt["state"] = "spent"
            else:
                rt["state"], rt["cooldown_until"] = "cooldown", now + cooldown

    def fire(self, rt: dict[str, Any]) -> None:
        for side in ("long", "short"):
            leg = self.cfg.get(side) or {}
            if not leg.get("enabled", False):
                continue
            lots = float(leg.get("lots", 0.01) or 0.01)
            fill = self.ask if side == "long" else self.bid
            sgn = 1 if side == "long" else -1
            tp_d = price_distance(float(leg.get("tp", 0) or 0), self.unit(), self.pip())
            sl_d = price_distance(float(leg.get("sl", 0) or 0), self.unit(), self.pip())
            pos = {
                "ticket": self.next_ticket, "level_id": rt["id"], "side": side,
                "lots": lots, "open_price": round(fill, SYM_DIGITS),
                "sl": round(fill - sgn * sl_d, SYM_DIGITS) if sl_d else 0.0,
                "tp": round(fill + sgn * tp_d, SYM_DIGITS) if tp_d else 0.0,
                "be_done": False,
                "open_time": self.server_now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.next_ticket += 1
            self.positions.append(pos)
            self.say("trade", f"{side.upper()} {lots} @ {pos['open_price']} "
                              f"tp={pos['tp']} sl={pos['sl']} [{rt['id']}]")

    # ---- position management --------------------------------------------
    def manage(self) -> None:
        for pos in list(self.positions):
            sgn = 1 if pos["side"] == "long" else -1
            close_px = self.bid if pos["side"] == "long" else self.ask
            prof = (close_px - pos["open_price"]) * sgn

            # broker-side TP / SL
            if pos["tp"] and ((pos["side"] == "long" and self.bid >= pos["tp"])
                              or (pos["side"] == "short" and self.ask <= pos["tp"])):
                self.close(pos, "TP")
                continue
            if pos["sl"] and ((pos["side"] == "long" and self.bid <= pos["sl"])
                              or (pos["side"] == "short" and self.ask >= pos["sl"])):
                self.close(pos, "SL")
                continue

            # breakeven: once only, never backwards
            leg = self.cfg.get(pos["side"]) or {}
            if pos["be_done"] or not leg.get("be_enabled", False):
                continue
            trig = price_distance(float(leg.get("be_trigger", 0) or 0), self.unit(), self.pip())
            if trig <= 0 or prof < trig:
                continue
            off = price_distance(float(leg.get("be_offset", 0) or 0), self.unit(), self.pip())
            target = round(pos["open_price"] + sgn * off, SYM_DIGITS)
            cur = pos["sl"]
            better = (target > cur) if pos["side"] == "long" else (target < cur or cur == 0)
            if cur and not better:
                pos["be_done"] = True
                continue
            pos["sl"] = target
            pos["be_done"] = True
            self.say("trade", f"{pos['side']} #{pos['ticket']} breakeven -> SL {target}")

    def close(self, pos: dict[str, Any], why: str) -> None:
        sgn = 1 if pos["side"] == "long" else -1
        close_px = self.bid if pos["side"] == "long" else self.ask
        money = (close_px - pos["open_price"]) * sgn * money_per_price(pos["lots"])
        self.session_closed += money
        self.positions.remove(pos)
        self.say("trade", f"closed {pos['side']} #{pos['ticket']} ({why}) {money:+.2f}")

    # ---- commands --------------------------------------------------------
    def poll_commands(self) -> None:
        raw = read_json(self.cmd_path)
        if not raw:
            return
        cid = int(raw.get("id", 0))
        if cid == self.last_command_id or cid == 0:
            return
        cmd, args = raw.get("cmd", ""), (raw.get("args") or {})
        n = 0
        if cmd == "close_all":
            n = len(self.positions)
            for p in list(self.positions):
                self.close(p, "close_all")
        elif cmd == "close_level":
            for p in [x for x in self.positions if x["level_id"] == args.get("level_id")]:
                self.close(p, "close_level")
                n += 1
        elif cmd == "close_side":
            for p in [x for x in self.positions if x["side"] == args.get("side")]:
                self.close(p, "close_side")
                n += 1
        elif cmd == "rearm_level":
            rt = self.levels.get(str(args.get("level_id")))
            if rt:
                rt.update(self.fresh_level(rt, self.tol_price()))
                n = 1
        elif cmd == "rearm_all":
            for rt in self.levels.values():
                rt.update(self.fresh_level(rt, self.tol_price()))
                n += 1
            self.panic_latch = False
        elif cmd == "panic":
            n = len(self.positions)
            for p in list(self.positions):
                self.close(p, "panic")
            self.panic_latch = True
        else:
            self.last_command_id = cid
            self.last_command_result = f"{cmd}: unknown command"
            self.say("error", self.last_command_result)
            return
        self.last_command_id = cid
        self.last_command_result = f"{cmd}: {n}"
        self.say("info", f"command #{cid} {cmd} -> {n}")

    # ---- publish ---------------------------------------------------------
    def publish(self, stopped: bool = False) -> None:
        blockers = self.blockers()
        win_open, win_why = self.window_open()
        pip = self.pip()
        live_counts: dict[str, int] = {}
        for p in self.positions:
            live_counts[p["level_id"]] = live_counts.get(p["level_id"], 0) + 1

        floating = 0.0
        positions = []
        for p in self.positions:
            sgn = 1 if p["side"] == "long" else -1
            close_px = self.bid if p["side"] == "long" else self.ask
            pp = (close_px - p["open_price"]) * sgn
            money = pp * money_per_price(p["lots"])
            floating += money
            positions.append({
                "ticket": p["ticket"], "level_id": p["level_id"], "side": p["side"],
                "lots": p["lots"], "open_price": p["open_price"],
                "sl": p["sl"], "tp": p["tp"], "current": round(close_px, SYM_DIGITS),
                "profit_price": round(pp, SYM_DIGITS),
                "profit_pips": round(pp / pip, 1) if pip else 0.0,
                "profit_money": round(money, 2),
                "be_done": p["be_done"], "open_time": p["open_time"],
            })

        now_utc = datetime.now(timezone.utc)
        state = {
            "schema": SCHEMA, "ea_version": EA_VERSION,
            "ts_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ts_server": self.server_now().strftime("%Y-%m-%d %H:%M:%S"),
            "ts_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "server_gmt_offset_hours": FAKE_SERVER_GMT_OFFSET,
            "heartbeat_ms": HEARTBEAT_MS,
            "config_version_seen": self.cfg_version_seen,
            "config_error": self.cfg_error,
            "terminal": {
                "company": "IC Markets (EU) Ltd [MOCK]", "name": "MetaTrader 5",
                "build": 4755, "connected": True, "trade_allowed": True,
                "expert_enabled": True, "dll_allowed": False,
            },
            "account": {
                **{k: v for k, v in FAKE_ACCOUNT.items()},
                "equity": round(FAKE_ACCOUNT["balance"] + self.session_closed + floating, 2),
                "margin_free": round(FAKE_ACCOUNT["balance"] * 0.98, 2),
            },
            "symbol": {
                "name": self.cfg.get("symbol", "XAUUSD"), "found": True,
                "bid": self.bid, "ask": self.ask, "spread_points": SPREAD_POINTS_BASE,
                "digits": SYM_DIGITS, "point": SYM_POINT, "pip_size": pip,
                "tick_size": SYM_POINT, "tick_value": 1.0, "contract_size": SYM_CONTRACT,
                "volume_min": 0.01, "volume_max": 50.0, "volume_step": 0.01,
                "stops_level_points": 0, "freeze_level_points": 0,
                "trade_mode": "full", "filling": "IOC",
                "candidates": ["XAUUSD", "XAUUSD.a"],
            },
            "window": {
                "open": win_open, "reason": win_why, "next_change": "",
                "timezone": (self.cfg.get("time_window") or {}).get("timezone", "server"),
            },
            "armed": not blockers and not stopped,
            "blockers": blockers if not stopped else ["mock EA stopped"],
            "levels": [{
                "id": rt["id"], "label": rt["label"], "price": rt["price"],
                "enabled": rt["enabled"], "state": rt["state"], "arm_side": rt["arm_side"],
                "distance_price": round(rt["price"] - self.bid, SYM_DIGITS),
                "distance_pips": round((rt["price"] - self.bid) / pip, 1) if pip else 0.0,
                "last_fire_utc": rt["last_fire_utc"], "fire_count": rt["fire_count"],
                "open_positions": live_counts.get(rt["id"], 0),
                "cooldown_remaining_sec": max(0, int(rt["cooldown_until"] - time.time()))
                if rt["state"] == "cooldown" else 0,
            } for rt in self.levels.values()],
            "positions": positions,
            "totals": {
                "open_positions": len(positions),
                "open_lots": round(sum(p["lots"] for p in positions), 2),
                "floating_money": round(floating, 2),
                "session_closed_money": round(self.session_closed, 2),
                "session_fires": self.session_fires,
            },
            "last_command_id": self.last_command_id,
            "last_command_result": self.last_command_result,
            "log": self.log[-60:],
        }
        write_json_atomic(self.state_path, state)

    # ---- main loop -------------------------------------------------------
    def run(self) -> None:
        last_publish = 0.0
        while self.running:
            self.poll_config()
            self.poll_commands()
            self.tick()
            self.eval_levels()
            self.manage()
            now = time.time()
            if now - last_publish >= HEARTBEAT_MS / 1000.0:
                self.publish()
                last_publish = now
            time.sleep(0.1)
        self.say("warn", "mock EA stopped")
        self.publish(stopped=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Mock EA for the GoldStraddle panel (no trading).")
    ap.add_argument("--ipc-dir", type=Path, default=None)
    ap.add_argument("--price", type=float, default=None, help="starting bid")
    ap.add_argument("--vol", type=float, default=1.0, help="tick volatility multiplier")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep price across all configured levels instead of random walk")
    a = ap.parse_args()

    ipc = a.ipc_dir or default_ipc_dir()
    ipc.mkdir(parents=True, exist_ok=True)

    price = a.price
    if price is None:
        prev = read_json(ipc / "state.json") or {}
        price = float((prev.get("symbol") or {}).get("bid") or 0) or 3550.00

    ea = MockEA(ipc, price, a.vol, a.sweep)

    def stop(*_: Any) -> None:
        ea.running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    print("=" * 62)
    print("  MOCK EA - simulation only, no broker, no orders")
    print(f"  ipc dir : {ipc}")
    print(f"  price   : {price}  vol x{a.vol}{'  [sweep]' if a.sweep else ''}")
    print("=" * 62)
    ea.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
