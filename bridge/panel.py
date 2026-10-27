"""Control panel for the OANDA -> Vantage bridge.

  * Big ARM / DISARM switch. Nothing is ever sent while DISARMED.
  * Every bot setting editable, persisted to panel_state.json.
  * Pulls GOLD.pro / SILVER.pro from OANDA and DXY_Z6 from IC, aligns clocks.
  * Writes bridge_settings.json and bridge_signal.json into the MetaQuotes
    Common\\Files folder, which the Vantage EA reads.

SAFETY
------
The account behind this is REAL MONEY (<live-server>). Three
independent gates must all be open before any order can be placed:
    1. this panel is ARMED,
    2. the signal file is fresh (TTL), and
    3. the EA's own AllowLiveTrading input is true.
Disarming immediately writes a FLAT command, so closing the panel or hitting
the switch stops new entries at once.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import config as C
import feed
import strategy

BG = "#12151c"; FG = "#e8e8e8"; MUT = "#8b93a7"
OK = "#4ade80"; WARN = "#fbbf24"; BAD = "#f87171"; ACC = "#4aa3ff"
CARD = "#1a1f2b"


class Panel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gold Bridge  -  OANDA data / Vantage execution")
        self.geometry("1080x760")
        self.configure(bg=BG)
        self.settings = dict(C.DEFAULTS)
        self._load_state()
        self.settings["enabled"] = False      # never start armed
        self.vars: dict[str, tk.Variable] = {}
        self.q: queue.Queue = queue.Queue()
        self.worker_stop = threading.Event()
        self.last_signal = None
        self.last_error = ""
        self.engine = strategy.Engine()
        self._build()
        self._write_settings()
        self._write_signal({"action": "FLAT", "reason": "panel starting"})
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()
        self.after(250, self._drain)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- persistence ----------------
    def _load_state(self):
        try:
            if C.STATE_FILE.exists():
                self.settings.update(json.loads(C.STATE_FILE.read_text()))
        except Exception:
            pass

    def _save_state(self):
        try:
            C.STATE_FILE.write_text(json.dumps(self.settings, indent=2))
        except Exception:
            pass

    # ---------------- file contract ----------------
    def _write_settings(self):
        C.COMMON_FILES.mkdir(parents=True, exist_ok=True)
        payload = dict(self.settings)
        payload["_written_utc"] = time.time()
        tmp = C.SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(C.SETTINGS_FILE)        # atomic

    def _write_signal(self, sig: dict):
        C.COMMON_FILES.mkdir(parents=True, exist_ok=True)
        sig = dict(sig)
        sig["ts"] = time.time()
        sig["ttl"] = self.settings.get("signal_ttl_seconds", 90)
        sig["armed"] = bool(self.settings.get("enabled"))
        sig["symbol"] = C.VANTAGE_GOLD
        sig["magic"] = self.settings.get("magic")
        if not sig["armed"]:
            sig["action"] = "FLAT"
        sig["id"] = f"{int(sig['ts'])}-{sig['action']}-{sig.get('dir','')}"
        tmp = C.SIGNAL_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(sig, indent=2))
        tmp.replace(C.SIGNAL_FILE)
        self.last_signal = sig

    # ---------------- ui ----------------
    def _card(self, parent, title):
        f = tk.Frame(parent, bg=CARD, bd=0)
        tk.Label(f, text=title, bg=CARD, fg=ACC,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(8, 4))
        return f

    def _build(self):
        top = tk.Frame(self, bg=BG); top.pack(fill="x", padx=14, pady=(12, 6))

        self.arm_btn = tk.Button(top, text="DISARMED", width=18, height=2,
                                 font=("Segoe UI", 16, "bold"),
                                 bg=BAD, fg="#0b0d12", bd=0, activebackground="#ef4444",
                                 command=self._toggle)
        self.arm_btn.pack(side="left")

        info = tk.Frame(top, bg=BG); info.pack(side="left", padx=18)
        self.state_lbl = tk.Label(info, text="No orders will be sent", bg=BG, fg=BAD,
                                  font=("Segoe UI", 12, "bold"), anchor="w")
        self.state_lbl.pack(anchor="w")
        self.acct_lbl = tk.Label(info, text="account -", bg=BG, fg=MUT,
                                 font=("Consolas", 9), anchor="w", justify="left")
        self.acct_lbl.pack(anchor="w")
        self.diag_lbl = tk.Label(info, text="signal -", bg=BG, fg=MUT,
                                 font=("Consolas", 9), anchor="w", justify="left")
        self.diag_lbl.pack(anchor="w")

        self.flat_btn = tk.Button(top, text="CLOSE ALL", width=12, height=2,
                                  font=("Segoe UI", 10, "bold"),
                                  bg="#374151", fg=FG, bd=0, command=self._flatten)
        self.flat_btn.pack(side="right")

        body = tk.Frame(self, bg=BG); body.pack(fill="both", expand=True, padx=14, pady=6)
        left = tk.Frame(body, bg=BG); left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(body, bg=BG); right.pack(side="right", fill="both", expand=True, padx=(12, 0))

        groups = [
            ("Risk", ["lots", "tp_pips", "sl_pips", "use_breakeven",
                      "be_trigger_pips", "be_buffer_pips", "max_spread_points",
                      "max_trades_day", "one_position_only"]),
            ("Modules", ["enable_break_reclaim", "enable_rejection"]),
            ("Break & Reclaim", ["origin_bars", "reclaim_window", "max_dist_pips",
                                 "min_room_pips", "use_anchor", "cooldown_bars"]),
            ("Rejection (D1 levels)", ["rej_max_dip_pips", "rej_approach_bars",
                                       "rej_shape_mode", "rej_wick_body_mult",
                                       "rej_min_body_pips", "rej_require_color"]),
            ("Indicators", ["use_linreg_candles", "linreg_length", "signal_length",
                            "sma_signal", "wt_channel", "wt_average", "wt_threshold",
                            "reg_length", "reg_dev_mult"]),
            ("Filters", ["use_correlation", "require_fresh", "reject_doji",
                         "enable_h4", "enable_d1", "max_levels", "seed_htf_bars"]),
            ("Session (Rome)", ["use_time_window", "session_start", "session_end"]),
            ("Bridge", ["magic", "slippage_points", "poll_seconds", "signal_ttl_seconds"]),
        ]
        for i, (title, keys) in enumerate(groups):
            parent = left if i % 2 == 0 else right
            card = self._card(parent, title)
            card.pack(fill="x", pady=5)
            grid = tk.Frame(card, bg=CARD); grid.pack(fill="x", padx=10, pady=(0, 8))
            for r, k in enumerate(keys):
                tk.Label(grid, text=k, bg=CARD, fg=MUT, font=("Consolas", 9),
                         anchor="w", width=20).grid(row=r, column=0, sticky="w", pady=1)
                val = self.settings.get(k, C.DEFAULTS.get(k))
                if isinstance(val, bool):
                    v = tk.BooleanVar(value=val)
                    ttk.Checkbutton(grid, variable=v).grid(row=r, column=1, sticky="w")
                else:
                    v = tk.StringVar(value=str(val))
                    tk.Entry(grid, textvariable=v, width=14, bg="#0e1218", fg=FG,
                             insertbackground=FG, bd=0,
                             font=("Consolas", 9)).grid(row=r, column=1, sticky="w")
                self.vars[k] = v

        btns = tk.Frame(self, bg=BG); btns.pack(fill="x", padx=14, pady=(0, 6))
        tk.Button(btns, text="Apply settings", bg=ACC, fg="#0b0d12", bd=0,
                  font=("Segoe UI", 10, "bold"), command=self._apply).pack(side="left")
        tk.Label(btns, text="  settings apply immediately; the EA re-reads them every tick",
                 bg=BG, fg=MUT, font=("Segoe UI", 8)).pack(side="left")

        logc = self._card(self, "Live"); logc.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        self.log = tk.Text(logc, height=11, bg="#0e1218", fg=FG, bd=0,
                           font=("Consolas", 9), wrap="none")
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # ---------------- actions ----------------
    def _apply(self):
        before = dict(self.settings)
        for k, v in self.vars.items():
            raw = v.get()
            default = C.DEFAULTS.get(k)
            try:
                if isinstance(default, bool):
                    self.settings[k] = bool(raw)
                elif isinstance(default, int) and not isinstance(default, bool):
                    self.settings[k] = int(float(raw))
                elif isinstance(default, float):
                    self.settings[k] = float(raw)
                else:
                    self.settings[k] = str(raw)
            except Exception:
                messagebox.showerror("Bad value", f"{k}: cannot parse {raw!r}")
                return
        # level inputs changed -> the seeded set is no longer the one these
        # settings describe, so force a re-seed on the next poll
        if any(self.settings.get(k) != before.get(k)
               for k in ("max_levels", "seed_htf_bars", "enable_h4", "enable_d1")):
            self.engine.seeded = False
            self._log("level inputs changed - re-seeding")
        self._save_state(); self._write_settings()
        self._log(f"settings applied  lots={self.settings['lots']} "
                  f"tp={self.settings['tp_pips']} sl={self.settings['sl_pips']}")

    def _toggle(self):
        if not self.settings["enabled"]:
            if not messagebox.askyesno(
                "ARM the bridge?",
                "This account is REAL MONEY (<live-server>).\n\n"
                f"Lots: {self.settings['lots']}\n"
                f"TP: {self.settings['tp_pips']} pips    SL: {self.settings['sl_pips']} pips\n\n"
                "While ARMED the EA will place market orders on signals.\nProceed?"):
                return
            self.settings["enabled"] = True
            self.arm_btn.config(text="ARMED", bg=OK)
            self.state_lbl.config(text="LIVE - orders will be sent", fg=OK)
            self._log("*** ARMED ***")
        else:
            self.settings["enabled"] = False
            self.arm_btn.config(text="DISARMED", bg=BAD)
            self.state_lbl.config(text="No orders will be sent", fg=BAD)
            self._write_signal({"action": "FLAT", "reason": "disarmed"})
            self._log("*** DISARMED - FLAT written ***")
        self._save_state(); self._write_settings()

    def _flatten(self):
        if messagebox.askyesno("Close all?", "Send CLOSE_ALL to the EA?"):
            self._write_signal({"action": "CLOSE_ALL", "reason": "manual"})
            self._log("CLOSE_ALL sent")

    def _log(self, msg):
        self.log.insert("end", f"{time.strftime('%H:%M:%S')}  {msg}\n")
        self.log.see("end")

    # ---------------- worker ----------------
    def _loop(self):
        h4 = d1 = None
        htf_at = 0.0
        while not self.worker_stop.is_set():
            try:
                frames = feed.get_bars(count=max(600, self.settings["seed_htf_bars"]))
                bars = feed.align(frames)
                # H4/D1 only move on their own schedule; refreshing them every
                # poll would mean four extra terminal round-trips a minute for
                # data that is identical. 5 minutes is well inside an H4 bar.
                if h4 is None or time.time() - htf_at > 300:
                    h4, d1 = feed.get_htf(count=self.settings["seed_htf_bars"])
                    htf_at = time.time()
                sig = strategy.evaluate(bars, h4, d1, self.settings, self.engine)
                acct = feed.vantage_account()
                self.q.put(("tick", (sig, acct, frames["offsets"], len(bars))))
            except Exception as e:          # never let the worker die silently
                self.q.put(("err", str(e)))
            self.worker_stop.wait(max(2, int(self.settings.get("poll_seconds", 5))))

    def _drain(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "err":
                    if payload != self.last_error:
                        self._log(f"ERROR: {payload}")
                        self.last_error = payload
                else:
                    sig, acct, offs, nbars = payload
                    self.last_error = ""
                    self._write_signal(sig)
                    pos = acct.get("positions", [])
                    self.acct_lbl.config(
                        text=f"{acct.get('login')} {acct.get('server')}  "
                             f"bal {acct.get('balance')} {acct.get('currency')}  "
                             f"eq {acct.get('equity')}  algo={acct.get('trade_allowed')}  "
                             f"pos={len(pos)}  bars={nbars}  off O{offs['oanda_h']}/IC{offs['ic_h']}")
                    d = sig.get("diag") or {}
                    if d:
                        lv = d.get("levels", {})
                        nlv = sum(v[0] for v in lv.values()) if lv else 0
                        arm = ("ARMED-L" if d.get("armed_long") else
                               "ARMED-S" if d.get("armed_short") else
                               "REJ-L" if d.get("rej_long") else
                               "REJ-S" if d.get("rej_short") else "-")
                        band = ("UPPER" if d.get("at_upper") else
                                "LOWER" if d.get("at_lower") else "mid")
                        self.diag_lbl.config(
                            text=f"wt {d.get('wt'):>6}  z {d.get('z_close'):>5}  "
                                 f"{band:5s}  touch={'Y' if d.get('touch') else 'n'}  "
                                 f"XAG {d.get('silver')} DXY {d.get('dxy')}  "
                                 f"{arm}  levels {nlv}  |  {sig.get('reason','') or 'idle'}")
                    # log entries always; log FLAT only when the reason changes,
                    # otherwise a 5s poll floods the window with the same line
                    reason = sig.get("reason", "")
                    if sig.get("action") == "ENTER":
                        self._log(f"ENTER     {sig.get('dir','')} {reason}")
                    elif reason and reason != getattr(self, "_last_reason", None):
                        self._log(f"          {reason}")
                    self._last_reason = reason
        except queue.Empty:
            pass
        self.after(400, self._drain)

    def _on_close(self):
        self.settings["enabled"] = False
        self._write_signal({"action": "FLAT", "reason": "panel closed"})
        self._write_settings()
        self.worker_stop.set()
        self.destroy()


if __name__ == "__main__":
    Panel().mainloop()
