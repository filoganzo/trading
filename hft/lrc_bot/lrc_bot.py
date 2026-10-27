"""
LRC Bot  v1.0  —  IC Markets edition
Self-contained mean-reversion trading bot.
Pulls historical + live data directly from MT5. No local data files needed.

Package to .exe:  run build.bat
"""
import sys, os, json, math, logging, subprocess, tempfile, atexit
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import pandas as pd

from PyQt6.QtCore  import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui   import QColor, QBrush, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWizard, QWizardPage, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QFileDialog, QTextEdit, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QCheckBox, QDoubleSpinBox, QSpinBox, QMessageBox, QSizePolicy,
)

# ── Version + constants ───────────────────────────────────────────────────────
VERSION      = "1.0.0"
MAGIC        = 20260423      # identifies bot's orders in MT5
LRC_LENGTH   = 120
SIGMA_MULT   = 1.0
TOUCH_PCT    = 1.5
SL_ATR_MULT  = 0.3
TP_RR        = 5.0
MIN_SCORE    = 40
MIN_LIVE_R   = 1.5           # skip if live R already below this
TRIGGER_SEC  = 10            # seconds between trigger checks
SCAN_HOUR    = 23            # local hour for nightly scan

IC_SERVERS = [
    "<broker-server>", "<broker-server>", "<broker-server>",
    "<broker-server>", "<broker-server>",
    "<broker-server>",   "<broker-server>",
    "<broker-server>",
]

IC_MT5_PATHS = [
    r"C:\Program Files\IC Markets Global MetaTrader 5\terminal64.exe",
    r"C:\Program Files\ICMarkets MetaTrader 5\terminal64.exe",
    r"C:\Program Files\IC Markets - MetaTrader 5\terminal64.exe",
    r"C:\Program Files (x86)\IC Markets Global MetaTrader 5\terminal64.exe",
]

# ── Paths ─────────────────────────────────────────────────────────────────────
def _res(name):
    """Resolve bundled resource path (works both raw and via PyInstaller)."""
    base = getattr(sys, "_MEIPASS", Path(__file__).parent)
    return Path(base) / name

CONFIG_FILE = Path.home() / "lrc_bot_config.json"
TRADES_FILE = Path.home() / "lrc_bot_trades.csv"
LOG_FILE    = Path.home() / "lrc_bot.log"
TICKERS_F   = _res("tickers.json")

logging.basicConfig(
    filename=str(LOG_FILE), level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("lrc_bot")

# ── Theme ─────────────────────────────────────────────────────────────────────
CRUST    = "#11111b"; MANTLE  = "#181825"; BASE    = "#1e1e2e"
SURFACE0 = "#313244"; SURFACE1= "#45475a"; OVERLAY0= "#6c7086"
OVERLAY1 = "#7f849c"; TEXT    = "#cdd6f4"; SUBTEXT = "#a6adc8"
LAVENDER = "#b4befe"; BLUE    = "#89b4fa"; TEAL    = "#94e2d5"
GREEN    = "#a6e3a1"; YELLOW  = "#f9e2af"; PEACH   = "#fab387"
MAROON   = "#eba0ac"; RED     = "#f38ba8"; MAUVE   = "#cba6f7"

QSS = f"""
QWidget {{ background:{BASE}; color:{TEXT};
  font-family:'Cascadia Code','Consolas',monospace; font-size:12px; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
  background:{SURFACE0}; color:{TEXT}; border:1px solid {SURFACE1};
  border-radius:4px; padding:4px 8px; }}
QPushButton {{
  background:{SURFACE0}; color:{TEXT}; border:1px solid {SURFACE1};
  border-radius:4px; padding:6px 16px; }}
QPushButton:hover  {{ background:{SURFACE1}; color:{LAVENDER}; }}
QPushButton#primary {{ background:{MAUVE}; color:{CRUST}; font-weight:bold; border:none; }}
QPushButton#primary:hover {{ background:{LAVENDER}; }}
QPushButton#danger  {{ color:{MAROON}; border-color:{MAROON}; }}
QPushButton#danger:hover  {{ background:{MAROON}; color:{CRUST}; }}
QFrame#card {{ background:{MANTLE}; border:1px solid {SURFACE1}; border-radius:6px; }}
QLabel#head {{ color:{LAVENDER}; font-size:14px; font-weight:bold; }}
QLabel#section {{ color:{SUBTEXT}; font-size:11px; font-weight:bold;
  text-transform:uppercase; letter-spacing:1px; }}
QProgressBar {{ background:{SURFACE0}; border-radius:3px; height:8px; color:transparent; }}
QProgressBar::chunk {{ background:{MAUVE}; border-radius:3px; }}
QTextEdit {{ background:{MANTLE}; border:1px solid {SURFACE0}; border-radius:4px; }}
QTableWidget {{ background:{MANTLE}; gridline-color:{SURFACE0}; border:none;
  selection-background-color:{SURFACE1}; }}
QHeaderView::section {{ background:{SURFACE0}; color:{SUBTEXT}; padding:3px 8px;
  border:none; font-weight:bold; font-size:11px; }}
QTableWidget::item {{ padding:2px 8px; }}
QWizard {{ background:{BASE}; }}
QWizardPage {{ background:{BASE}; }}
"""

# ═════════════════════════════════════════════════════════════════════════════
#  INLINE SIGNAL ENGINE  (no project imports)
# ═════════════════════════════════════════════════════════════════════════════

def _lrc(close: np.ndarray, length: int, sigma_mult: float):
    """Linear Regression Channel. Returns (mid, upper, lower, dev_sigma)."""
    n   = len(close)
    mid = np.full(n, np.nan); upper = np.full(n, np.nan)
    lower = np.full(n, np.nan); dsig = np.full(n, np.nan)
    x    = np.arange(length, dtype=float)
    xm   = x.mean(); xd = x - xm; xd2 = (xd**2).sum()
    for i in range(length - 1, n):
        y   = close[i - length + 1: i + 1]
        ym  = y.mean(); sl = (xd * (y - ym)).sum() / xd2
        yh  = sl * x + (ym - sl * xm)
        res = y - yh; std = res.std(ddof=0)
        m = yh[-1]; mid[i] = m
        upper[i] = m + sigma_mult * std; lower[i] = m - sigma_mult * std
        if std > 0: dsig[i] = (close[i] - m) / std
    return mid, upper, lower, dsig


def _atr(df: pd.DataFrame, p: int = 14) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n = len(c); res = np.full(n, np.nan)
    if n < p + 1: return res
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    res[p] = tr[:p].mean()
    for i in range(p+1, n): res[i] = (res[i-1]*(p-1) + tr[i-1]) / p
    return res


def _key_level(df: pd.DataFrame, lookback: int = 60):
    """Nearest swing high/low to last close. Returns (level_price, dist_pct)."""
    h = df["high"].values; l = df["low"].values; c = df["close"].values
    last = c[-1]; levels = []
    window = min(lookback, len(c) - 3)
    for i in range(2, window):
        if h[-i] > h[-i-1] and h[-i] > h[-i+1]: levels.append(h[-i])
        if l[-i] < l[-i-1] and l[-i] < l[-i+1]: levels.append(l[-i])
    if not levels: return float("nan"), float("nan")
    pairs = sorted((abs(lv - last) / last * 100, lv) for lv in levels)
    return pairs[0][1], pairs[0][0]


def _m15_ok(mt5, sym: str, direction: str) -> bool:
    """
    M15 velocity check: True = momentum exhausting → OK to enter.
    Looks for: correct approach direction + (fading range OR reversal candle OR wick rejection).
    """
    try:
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M15, 0, 8)
        if rates is None or len(rates) < 5: return True
        df    = pd.DataFrame(rates)
        rngs  = df["high"] - df["low"]
        last  = df.iloc[-1]
        fading = float(rngs.iloc[-1]) < float(rngs.iloc[:-1].mean()) * 0.75
        if direction == "LONG":
            approached = float(df["close"].iloc[-4]) >= float(df["close"].iloc[-2])
            reversal   = float(last["close"]) > float(last["open"])
            wick       = (float(last["open"]) - float(last["low"])) > \
                          abs(float(last["close"]) - float(last["open"])) * 1.5
        else:
            approached = float(df["close"].iloc[-4]) <= float(df["close"].iloc[-2])
            reversal   = float(last["close"]) < float(last["open"])
            wick       = (float(last["high"]) - float(last["open"])) > \
                          abs(float(last["close"]) - float(last["open"])) * 1.5
        return approached and (fading or reversal or wick)
    except Exception:
        return True


def analyse(df: pd.DataFrame):
    """
    Full LRC signal analysis on daily OHLC DataFrame.
    Returns signal dict or None if price inside band.
    """
    if len(df) < LRC_LENGTH + 10: return None
    c = df["close"].values
    mid, upper, lower, dsig = _lrc(c, LRC_LENGTH, SIGMA_MULT)
    atr_arr = _atr(df)
    last_c = c[-1]; last_dsig = dsig[-1]; last_atr = atr_arr[-1]
    if math.isnan(mid[-1]) or math.isnan(last_atr): return None

    if   last_c > upper[-1]: direction, pos = "SHORT", +1
    elif last_c < lower[-1]: direction, pos = "LONG",  -1
    else:                    direction, pos = None,     0

    level_px, level_dist = _key_level(df)
    sigma_prox = min(100.0, abs(last_dsig) / SIGMA_MULT * 100) if not math.isnan(last_dsig) else 0
    if not math.isnan(level_dist):
        level_prox = ((TOUCH_PCT - level_dist) / TOUCH_PCT * 100 if level_dist <= TOUCH_PCT
                      else max(0.0, (3.5 - level_dist) / 3.5 * 40))
    else:
        level_prox = 0
    readiness = sigma_prox * 0.6 + level_prox * 0.4

    lrc_met   = pos != 0
    level_met = (not math.isnan(level_dist)) and level_dist <= TOUCH_PCT
    is_signal = lrc_met and level_met and readiness >= MIN_SCORE

    sl = tp = None
    if is_signal and not math.isnan(level_px):
        sl_price = level_px + pos * SL_ATR_MULT * last_atr
        risk     = abs(last_c - sl_price)
        if risk > 0.001:
            sl = round(sl_price, 4)
            tp = round(last_c - pos * TP_RR * risk, 4)

    return {
        "direction": direction or ("SHORT?" if last_dsig > 0 else "LONG?"),
        "is_signal": is_signal, "lrc_met": lrc_met, "level_met": level_met,
        "readiness": round(readiness, 1),
        "dev_sigma": round(float(last_dsig), 3) if not math.isnan(last_dsig) else 0.0,
        "level_dist": round(float(level_dist), 2) if not math.isnan(level_dist) else 99.0,
        "close": round(last_c, 4), "atr": round(last_atr, 4),
        "lrc_mid": round(float(mid[-1]), 4),
        "sl": sl, "tp": tp,
    }

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG  +  PASSWORD
# ═════════════════════════════════════════════════════════════════════════════

def load_cfg() -> dict:
    try:   return json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}
    except: return {}

def save_cfg(cfg: dict):
    safe = {k: v for k, v in cfg.items() if k != "_pw"}
    CONFIG_FILE.write_text(json.dumps(safe, indent=2))

def save_pw(login: int, pw: str):
    try:
        import keyring; keyring.set_password("lrc_bot", str(login), pw)
    except Exception:
        cfg = load_cfg(); cfg["_pw"] = pw; CONFIG_FILE.write_text(json.dumps(cfg, indent=2))

def load_pw(login: int) -> str:
    try:
        import keyring; return keyring.get_password("lrc_bot", str(login)) or ""
    except Exception:
        return load_cfg().get("_pw", "")

def log_trade(row: dict):
    exists = TRADES_FILE.exists()
    with open(TRADES_FILE, "a", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not exists: w.writeheader()
        w.writerow(row)

# ═════════════════════════════════════════════════════════════════════════════
#  MT5 HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def mt5_connect(path, login, password, server):
    """Returns (ok, message). Leaves mt5 initialized on success."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize(path=path, login=int(login), password=password, server=server):
            return False, f"MT5 error: {mt5.last_error()}"
        info = mt5.account_info()
        if info is None: return False, "Connected but no account info returned"
        broker = info.company or ""
        if "IC" not in broker.upper() and "IC Markets" not in broker:
            mt5.shutdown()
            return False, f"Broker mismatch: '{broker}' — this bot requires IC Markets"
        return True, f"{info.name}  #{info.login}  ${info.balance:,.2f}  ({info.server})"
    except Exception as e:
        return False, str(e)

def mt5_bars(mt5, sym: str, n: int = 220) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 0, n)
    if rates is None or len(rates) < 150: return pd.DataFrame()
    df = pd.DataFrame(rates)
    df.rename(columns={"tick_volume":"volume"}, inplace=True)
    return df[["open","high","low","close","volume"]]

def mt5_live_price(mt5, sym: str) -> float:
    tick = mt5.symbol_info_tick(sym)
    if tick and tick.bid > 0:
        return (tick.bid + tick.ask) / 2 if tick.ask > 0 else tick.bid
    return float("nan")

def mt5_calc_volume(mt5, sym, entry, sl, balance, risk_pct) -> float:
    info = mt5.symbol_info(sym)
    if not info: return 0.0
    risk_usd = balance * risk_pct / 100
    risk_ps  = abs(entry - sl)
    if risk_ps < 0.001: return 0.0
    shares   = risk_usd / risk_ps
    vol      = shares / info.trade_contract_size
    step     = info.volume_step
    vol      = max(info.volume_min, round(round(vol / step) * step, 2))
    return min(info.volume_max, vol)

def mt5_place_order(mt5, sym, direction, price, sl, tp, volume, paper=True):
    if paper:
        return True, f"[PAPER] {direction} {sym} qty={volume:.2f} @ {price:.2f}  SL={sl:.2f}  TP={tp:.2f}"
    order_type = mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL
    # Market orders require exact bid (SELL) or ask (BUY) — mid-price gets rejected
    tick = mt5.symbol_info_tick(sym)
    if not tick:
        return False, f"No tick for {sym}"
    fill_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": sym,
        "volume": float(volume), "type": order_type, "price": fill_price,
        "sl": sl, "tp": tp, "deviation": 50, "magic": MAGIC,
        "comment": "LRC_bot", "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        return True, f"FILLED {direction} {sym} qty={volume:.2f} @ {res.price:.2f}"
    code = res.retcode if res else "no response"
    comment = res.comment if res else ""
    return False, f"Order failed: {code} {comment}"

# ═════════════════════════════════════════════════════════════════════════════
#  WORKERS
# ═════════════════════════════════════════════════════════════════════════════

class ScanWorker(QThread):
    progress = pyqtSignal(int, int, str)   # done, total, sym
    finished = pyqtSignal(dict)            # {sym: signal_dict}
    error    = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg

    def run(self):
        try:
            import MetaTrader5 as mt5
            path = self.cfg.get("mt5_path","")
            if not mt5.initialize(path=path, login=int(self.cfg["login"]),
                                  password=load_pw(self.cfg["login"]),
                                  server=self.cfg["server"]):
                self.error.emit(f"MT5 init failed: {mt5.last_error()}"); return

            # Load symbol list
            try:
                data  = json.loads(Path(TICKERS_F).read_text())
                syms  = [t for t in data.get("US",{}).get("tickers",[]) if "-" not in t]
            except Exception:
                syms = [s.name for s in (mt5.symbols_get() or [])
                        if any(s.name.endswith(x) for x in (".NAS",".NYSE"))]

            # Pre-filter to symbols MT5 actually knows — avoids 1000+ wasted IPC calls
            known = {s.name for s in (mt5.symbols_get() or [])}
            syms  = [s for s in syms if s in known]

            results = {}
            for i, sym in enumerate(syms):
                if i % 20 == 0:   # throttle GUI updates — every 20 symbols
                    self.progress.emit(i+1, len(syms), sym)
                mt5.symbol_select(sym, True)
                df = mt5_bars(mt5, sym)
                if df.empty: continue
                sig = analyse(df)
                if sig: results[sym] = sig

            mt5.shutdown()
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class TriggerWorker(QThread):
    triggered = pyqtSignal(str, str, float, float, float, float)  # sym, dir, price, sl, tp, vol
    checked   = pyqtSignal(str, str)   # sym, reason (for log)
    status    = pyqtSignal(str)        # status line update
    error     = pyqtSignal(str)

    def __init__(self, cfg, signals: dict):
        super().__init__()
        self.cfg     = cfg
        self.signals = signals   # {sym: signal_dict} from last scan
        self._stop   = False

    def stop(self): self._stop = True

    def run(self):
        try:
            import MetaTrader5 as mt5
            path = self.cfg.get("mt5_path","")
            if not mt5.initialize(path=path, login=int(self.cfg["login"]),
                                  password=load_pw(self.cfg["login"]),
                                  server=self.cfg["server"]):
                self.error.emit(f"MT5 init: {mt5.last_error()}"); return

            paper     = self.cfg.get("paper_mode", True)
            max_pos   = int(self.cfg.get("max_positions", 3))
            risk_pct  = float(self.cfg.get("risk_pct", 1.0))

            while not self._stop:
                try:
                    acct = mt5.account_info()
                    if acct is None:
                        self.status.emit("MT5 disconnected — retrying…"); self.sleep(30); continue
                    balance = acct.balance
                    open_pos = mt5.positions_get(magic=MAGIC) or []
                    n_open   = len(open_pos)
                    open_syms = {p.symbol for p in open_pos}

                    self.status.emit(
                        f"Balance ${balance:,.0f}  Equity ${acct.equity:,.0f}  "
                        f"Positions {n_open}/{max_pos}  Watching {len(self.signals)}")

                    for sym, sig in self.signals.items():
                        if self._stop: break
                        if not sig.get("is_signal"): continue
                        if sym in open_syms:
                            self.checked.emit(sym, "already in position"); continue
                        if n_open >= max_pos:
                            self.checked.emit(sym, f"max positions ({max_pos}) reached"); continue

                        live = mt5_live_price(mt5, sym)
                        if math.isnan(live): continue

                        d  = sig["direction"]
                        sl = sig.get("sl"); tp = sig.get("tp")
                        if sl is None or tp is None: continue

                        # Recompute live R
                        risk_live = abs(live - sl)
                        reward_live = (tp - live) if "LONG" in d else (live - tp)
                        live_r = reward_live / risk_live if risk_live > 0.001 else -1

                        if live_r < MIN_LIVE_R:
                            self.checked.emit(sym, f"live R {live_r:.1f} < {MIN_LIVE_R}"); continue

                        # M15 velocity check
                        if not _m15_ok(mt5, sym, d):
                            self.checked.emit(sym, "M15 momentum still strong — waiting"); continue

                        # All clear — trigger
                        vol = mt5_calc_volume(mt5, sym, live, sl, balance, risk_pct)
                        if vol <= 0:
                            self.checked.emit(sym, "volume calc failed"); continue

                        ok, msg = mt5_place_order(mt5, sym, d, live, sl, tp, vol, paper=paper)
                        log.info(msg)
                        if ok:
                            self.triggered.emit(sym, d, live, sl, tp, vol)
                            log_trade({"time": datetime.now().isoformat(), "symbol": sym,
                                       "direction": d, "entry": live, "sl": sl, "tp": tp,
                                       "volume": vol, "paper": paper})
                            open_syms.add(sym); n_open += 1
                        else:
                            self.checked.emit(sym, f"order failed: {msg}")

                except Exception as e:
                    self.error.emit(f"Trigger loop error: {e}")

                self.sleep(TRIGGER_SEC)

            mt5.shutdown()
        except Exception as e:
            self.error.emit(str(e))

# ═════════════════════════════════════════════════════════════════════════════
#  SETUP WIZARD
# ═════════════════════════════════════════════════════════════════════════════

def _lbl(txt, color=TEXT, bold=False):
    l = QLabel(txt); l.setStyleSheet(f"color:{color};{'font-weight:bold;' if bold else ''}")
    return l

def _field(placeholder="", echo_pw=False):
    f = QLineEdit(); f.setPlaceholderText(placeholder)
    if echo_pw: f.setEchoMode(QLineEdit.EchoMode.Password)
    return f


class PathPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 1 — MT5 Terminal")
        self.setSubTitle("Locate your IC Markets MT5 installation.")
        lay = QVBoxLayout(self)
        lay.addWidget(_lbl("MT5 terminal64.exe path:"))
        row = QHBoxLayout()
        self.path_edit = _field(r"C:\Program Files\IC Markets Global MetaTrader 5\terminal64.exe")
        row.addWidget(self.path_edit)
        browse = QPushButton("Browse…"); browse.clicked.connect(self._browse)
        row.addWidget(browse)
        lay.addLayout(row)
        self.test_btn = QPushButton("Test path"); self.test_btn.clicked.connect(self._test)
        lay.addWidget(self.test_btn)
        self.result_lbl = _lbl("")
        lay.addWidget(self.result_lbl)
        lay.addStretch()
        self.registerField("mt5_path*", self.path_edit)
        # Auto-detect
        for p in IC_MT5_PATHS:
            if Path(p).exists(): self.path_edit.setText(p); break

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select terminal64.exe", "", "terminal64.exe (terminal64.exe)")
        if p: self.path_edit.setText(p)

    def _test(self):
        p = self.path_edit.text().strip()
        if Path(p).exists():
            self.result_lbl.setText("✓ File found"); self.result_lbl.setStyleSheet(f"color:{GREEN};")
        else:
            self.result_lbl.setText("✗ File not found"); self.result_lbl.setStyleSheet(f"color:{RED};")


class CredPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 2 — Account credentials")
        self.setSubTitle("Enter your IC Markets MT5 login details.")
        lay = QGridLayout(self)
        lay.addWidget(_lbl("Login (account number):"), 0, 0)
        self.login_edit = _field("12345678"); lay.addWidget(self.login_edit, 0, 1)
        lay.addWidget(_lbl("Password:"), 1, 0)
        self.pw_edit = _field("••••••••", echo_pw=True); lay.addWidget(self.pw_edit, 1, 1)
        lay.addWidget(_lbl("Server:"), 2, 0)
        self.server_cb = QComboBox(); self.server_cb.addItems(IC_SERVERS)
        lay.addWidget(self.server_cb, 2, 1)
        self.verify_btn = QPushButton("Connect & Verify")
        self.verify_btn.setObjectName("primary"); self.verify_btn.clicked.connect(self._verify)
        lay.addWidget(self.verify_btn, 3, 0, 1, 2)
        self.result_lbl = _lbl(""); lay.addWidget(self.result_lbl, 4, 0, 1, 2)
        lay.addWidget(_lbl("Password is stored in Windows Credential Manager — never in plain text.",
                           OVERLAY1), 5, 0, 1, 2)
        self.registerField("login*",  self.login_edit)
        self.registerField("password*", self.pw_edit)
        self._verified = False

    def _verify(self):
        path   = self.wizard().field("mt5_path")
        login  = self.login_edit.text().strip()
        pw     = self.pw_edit.text()
        server = self.server_cb.currentText()
        self.result_lbl.setText("Connecting…"); self.result_lbl.setStyleSheet(f"color:{YELLOW};")
        QApplication.processEvents()
        ok, msg = mt5_connect(path, login, pw, server)
        if ok:
            self.result_lbl.setText(f"✓ {msg}"); self.result_lbl.setStyleSheet(f"color:{GREEN};")
            self._verified = True
            try:
                import MetaTrader5 as mt5; mt5.shutdown()
            except Exception: pass
        else:
            self.result_lbl.setText(f"✗ {msg}"); self.result_lbl.setStyleSheet(f"color:{RED};")
            self._verified = False
        self.completeChanged.emit()

    def isComplete(self): return self._verified


class StrategyPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Step 3 — Strategy settings")
        self.setSubTitle("Configure risk parameters. Start in paper mode to verify first.")
        lay = QGridLayout(self)
        lay.addWidget(_lbl("Account size ($):"), 0, 0)
        self.acc = QDoubleSpinBox(); self.acc.setRange(1000, 10_000_000)
        self.acc.setValue(100_000); self.acc.setSingleStep(5000); self.acc.setPrefix("$")
        lay.addWidget(self.acc, 0, 1)
        lay.addWidget(_lbl("Risk per trade (%):"), 1, 0)
        self.risk = QDoubleSpinBox(); self.risk.setRange(0.1, 5.0)
        self.risk.setValue(1.0); self.risk.setSingleStep(0.25); self.risk.setSuffix("%")
        lay.addWidget(self.risk, 1, 1)
        lay.addWidget(_lbl("Max concurrent positions:"), 2, 0)
        self.maxp = QSpinBox(); self.maxp.setRange(1, 20); self.maxp.setValue(3)
        lay.addWidget(self.maxp, 2, 1)
        lay.addWidget(QLabel(""), 3, 0)
        self.paper_cb = QCheckBox("Paper mode (log signals, place no real orders)")
        self.paper_cb.setChecked(True)
        self.paper_cb.setStyleSheet(f"color:{YELLOW}; font-weight:bold;")
        lay.addWidget(self.paper_cb, 4, 0, 1, 2)
        lay.addWidget(_lbl(
            "⚠  Switch to live trading only after verifying paper mode results.",
            MAROON), 5, 0, 1, 2)
        lay.setRowStretch(6, 1)


class SetupWizard(QWizard):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LRC Bot — Setup")
        self.setMinimumSize(520, 400)
        self.page_path  = PathPage();    self.addPage(self.page_path)
        self.page_cred  = CredPage();    self.addPage(self.page_cred)
        self.page_strat = StrategyPage();self.addPage(self.page_strat)
        self.setButtonText(QWizard.WizardButton.FinishButton, "Save & Start Bot")

    def accept(self):
        login = self.field("login").strip()
        pw    = self.field("password")
        cfg = {
            "mt5_path":      self.field("mt5_path"),
            "login":         int(login),
            "server":        self.page_cred.server_cb.currentText(),
            "account_size":  self.page_strat.acc.value(),
            "risk_pct":      self.page_strat.risk.value(),
            "max_positions": self.page_strat.maxp.value(),
            "paper_mode":    self.page_strat.paper_cb.isChecked(),
        }
        save_cfg(cfg); save_pw(int(login), pw)
        super().accept()

# ═════════════════════════════════════════════════════════════════════════════
#  STATUS WINDOW
# ═════════════════════════════════════════════════════════════════════════════

def _c(h): return QColor(h)
def _item(txt, col=TEXT):
    it = QTableWidgetItem(str(txt)); it.setForeground(QBrush(_c(col)))
    it.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
    return it

class StatusWindow(QMainWindow):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg      = cfg
        self.signals  = {}   # {sym: signal_dict}
        self.scan_w   = None
        self.trig_w   = None
        self._running = True   # auto-start trigger after first scan completes
        self.setWindowTitle(f"LRC Bot  v{VERSION}  ·  IC Markets")
        self.setMinimumSize(480, 640)
        self._build_ui()
        self._schedule_nightly()
        self._start_scan()   # initial scan on launch; trigger starts in _on_scan_done

    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        lay  = QVBoxLayout(root); lay.setContentsMargins(12,12,12,12); lay.setSpacing(8)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QFrame(); hdr.setObjectName("card")
        hl  = QHBoxLayout(hdr); hl.setContentsMargins(10,8,10,8)
        left = QVBoxLayout()
        self.conn_lbl = QLabel(f"LRC Bot  v{VERSION}  ·  IC Markets")
        self.conn_lbl.setObjectName("head"); left.addWidget(self.conn_lbl)
        self.acct_lbl = QLabel("Connecting…")
        self.acct_lbl.setStyleSheet(f"color:{SUBTEXT}; font-size:11px;"); left.addWidget(self.acct_lbl)
        hl.addLayout(left); hl.addStretch()
        self.paper_btn = QPushButton("● PAPER" if self.cfg.get("paper_mode") else "● LIVE")
        self.paper_btn.setStyleSheet(
            f"color:{YELLOW}; font-weight:bold;" if self.cfg.get("paper_mode")
            else f"color:{GREEN}; font-weight:bold;")
        self.paper_btn.clicked.connect(self._toggle_paper)
        hl.addWidget(self.paper_btn)
        lay.addWidget(hdr)

        # ── Status bar ────────────────────────────────────────────────────────
        self.status_lbl = QLabel("Initialising…")
        self.status_lbl.setStyleSheet(f"color:{OVERLAY1}; font-size:11px;")
        self.status_lbl.setWordWrap(True); lay.addWidget(self.status_lbl)

        # ── Scan progress ─────────────────────────────────────────────────────
        self.prog_bar = QProgressBar(); self.prog_bar.setVisible(False)
        lay.addWidget(self.prog_bar)

        # ── Signals table ─────────────────────────────────────────────────────
        lay.addWidget(_lbl("ACTIVE SIGNALS", SUBTEXT))
        self.sig_tbl = QTableWidget(0, 6)
        self.sig_tbl.setHorizontalHeaderLabels(["Symbol","Dir","Live $","Live R","Score","Conditions"])
        self.sig_tbl.verticalHeader().setVisible(False)
        self.sig_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.sig_tbl.horizontalHeader().setStretchLastSection(True)
        self.sig_tbl.setFixedHeight(150)
        lay.addWidget(self.sig_tbl)

        # ── Positions table ───────────────────────────────────────────────────
        lay.addWidget(_lbl("OPEN POSITIONS", SUBTEXT))
        self.pos_tbl = QTableWidget(0, 5)
        self.pos_tbl.setHorizontalHeaderLabels(["Symbol","Dir","Entry","Live $","P&L $"])
        self.pos_tbl.verticalHeader().setVisible(False)
        self.pos_tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.pos_tbl.horizontalHeader().setStretchLastSection(True)
        self.pos_tbl.setFixedHeight(130)
        lay.addWidget(self.pos_tbl)

        # ── Log ───────────────────────────────────────────────────────────────
        lay.addWidget(_lbl("LOG", SUBTEXT))
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(f"color:{TEXT}; font-size:11px;")
        lay.addWidget(self.log_box)

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        self.start_btn = QPushButton("■ Stop Bot"); self.start_btn.setObjectName("danger")
        self.start_btn.clicked.connect(self._toggle_bot)
        ctrl.addWidget(self.start_btn)
        scan_btn = QPushButton("⟳ Scan Now"); scan_btn.clicked.connect(self._start_scan)
        ctrl.addWidget(scan_btn)
        settings_btn = QPushButton("⚙ Settings"); settings_btn.clicked.connect(self._open_settings)
        ctrl.addWidget(settings_btn)
        lay.addLayout(ctrl)

        # ── Emergency controls ────────────────────────────────────────────────
        emg = QHBoxLayout()
        close_all_btn = QPushButton("⚠ Stop & Close All Positions")
        close_all_btn.setObjectName("danger")
        close_all_btn.setStyleSheet(
            f"background:{MAROON}; color:{CRUST}; font-weight:bold; border:none; padding:8px;")
        close_all_btn.clicked.connect(self._stop_and_close_all)
        emg.addWidget(close_all_btn)
        lay.addLayout(emg)

        # ── Refresh positions timer ───────────────────────────────────────────
        self._pos_timer = QTimer(); self._pos_timer.timeout.connect(self._refresh_positions)
        self._pos_timer.start(15_000)

    # ── Bot control ───────────────────────────────────────────────────────────
    def _start_scan(self):
        if self.scan_w and self.scan_w.isRunning(): return
        # Pause trigger so scan has exclusive MT5 access
        if self.trig_w and self.trig_w.isRunning():
            self.trig_w.stop(); self.trig_w.wait(5000)
        self._log("Starting scan…", YELLOW)
        self.prog_bar.setValue(0); self.prog_bar.setVisible(True)
        self.scan_w = ScanWorker(self.cfg)
        self.scan_w.progress.connect(self._on_scan_progress)
        self.scan_w.finished.connect(self._on_scan_done)
        self.scan_w.error.connect(lambda e: self._log(f"Scan error: {e}", RED))
        self.scan_w.start()

    def _on_scan_progress(self, done, total, sym):
        self.prog_bar.setMaximum(total); self.prog_bar.setValue(done)
        self.status_lbl.setText(f"Scanning {done}/{total}  —  {sym}")

    def _on_scan_done(self, results: dict):
        self.signals = results
        self.prog_bar.setVisible(False)
        n_sig = sum(1 for s in results.values() if s.get("is_signal"))
        self._log(f"Scan done — {len(results)} analysed, {n_sig} signals", GREEN)
        self._refresh_signals_table()
        if self._running:
            self._restart_trigger()

    def _start_trigger(self):
        if self.trig_w and self.trig_w.isRunning(): return
        self.trig_w = TriggerWorker(self.cfg, self.signals)
        self.trig_w.triggered.connect(self._on_triggered)
        self.trig_w.checked.connect(lambda s, r: self._log(f"  {s}: {r}", OVERLAY1))
        self.trig_w.status.connect(lambda s: self.status_lbl.setText(s))
        self.trig_w.error.connect(lambda e: self._log(f"Trigger error: {e}", RED))
        self.trig_w.start()
        self._running = True
        self.start_btn.setText("■ Stop Bot")
        self.start_btn.setObjectName("danger")
        self._log("Bot running — checking triggers every 10s", TEAL)

    def _restart_trigger(self):
        if self.trig_w and self.trig_w.isRunning():
            self.trig_w.stop(); self.trig_w.wait(3000)
        self.trig_w = None
        self._start_trigger()

    def _toggle_bot(self):
        if self._running:
            if self.trig_w: self.trig_w.stop()
            self._running = False
            self.start_btn.setText("▶ Start Bot"); self.start_btn.setObjectName("")
            self._log("Bot stopped", YELLOW)
        else:
            self._start_trigger()

    def _stop_and_close_all(self):
        """Stop the trigger loop then market-close every position opened by this bot."""
        reply = QMessageBox.warning(
            self, "Close all positions?",
            f"This will MARKET-CLOSE every position tagged with the bot's magic number "
            f"({MAGIC}).\n\nManually placed trades are NOT affected.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # 1. Stop the trigger loop first
        if self.trig_w and self.trig_w.isRunning():
            self.trig_w.stop()
            self.trig_w.wait(4000)
        self._running = False
        self.start_btn.setText("▶ Start Bot")
        self._log("Bot stopped.", YELLOW)

        # 2. Connect to MT5 and close each position
        paper = self.cfg.get("paper_mode", True)
        try:
            import MetaTrader5 as mt5
            mt5.initialize(
                path=self.cfg.get("mt5_path", ""),
                login=int(self.cfg["login"]),
                password=load_pw(self.cfg["login"]),
                server=self.cfg["server"],
            )
            positions = mt5.positions_get(magic=MAGIC) or []
            if not positions:
                self._log("No open bot positions found.", TEAL)
                mt5.shutdown(); return

            self._log(f"Closing {len(positions)} position(s)…", MAROON)
            closed = 0; failed = 0
            for p in positions:
                if paper:
                    self._log(f"  [PAPER] would close {p.symbol} qty={p.volume}", YELLOW)
                    closed += 1; continue
                # Determine close order type (opposite of open)
                close_type = (mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY
                              else mt5.ORDER_TYPE_BUY)
                tick = mt5.symbol_info_tick(p.symbol)
                if not tick:
                    self._log(f"  {p.symbol}: no tick — skipped", RED); failed += 1; continue
                price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
                req = {
                    "action":       mt5.TRADE_ACTION_DEAL,
                    "symbol":       p.symbol,
                    "volume":       p.volume,
                    "type":         close_type,
                    "position":     p.ticket,
                    "price":        price,
                    "deviation":    50,
                    "magic":        MAGIC,
                    "comment":      "LRC_bot_close_all",
                    "type_time":    mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                res = mt5.order_send(req)
                if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                    self._log(f"  ✓ Closed {p.symbol}  @ {res.price:.2f}", GREEN)
                    closed += 1
                else:
                    code = res.retcode if res else "no response"
                    self._log(f"  ✗ {p.symbol} failed: {code}", RED)
                    failed += 1

            mt5.shutdown()
            summary = f"Done — {closed} closed"
            if failed: summary += f", {failed} failed (check MT5 manually)"
            self._log(summary, GREEN if not failed else YELLOW)
            self._refresh_positions()

        except Exception as e:
            self._log(f"Close-all error: {e}", RED)

    def _toggle_paper(self):
        self.cfg["paper_mode"] = not self.cfg.get("paper_mode", True)
        save_cfg(self.cfg)
        is_paper = self.cfg["paper_mode"]
        self.paper_btn.setText("● PAPER" if is_paper else "● LIVE")
        self.paper_btn.setStyleSheet(
            f"color:{YELLOW}; font-weight:bold;" if is_paper else f"color:{GREEN}; font-weight:bold;")
        if not is_paper:
            if QMessageBox.warning(self, "Go LIVE?",
                "This will place REAL orders on your IC Markets account.\n\nAre you sure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) != QMessageBox.StandardButton.Yes:
                self.cfg["paper_mode"] = True; save_cfg(self.cfg)
                self.paper_btn.setText("● PAPER")
                self.paper_btn.setStyleSheet(f"color:{YELLOW}; font-weight:bold;"); return
        if self.trig_w: self.trig_w.cfg["paper_mode"] = self.cfg["paper_mode"]
        mode = "PAPER" if is_paper else "LIVE"
        self._log(f"Switched to {mode} mode", YELLOW if is_paper else GREEN)

    def _open_settings(self):
        wiz = SetupWizard()
        if wiz.exec():
            self.cfg = load_cfg()
            self._log("Settings saved. Restart for full effect.", YELLOW)

    # ── Tables ────────────────────────────────────────────────────────────────
    def _refresh_signals_table(self):
        # Reuse whatever MT5 connection is currently open — never init/shutdown here
        import MetaTrader5 as mt5
        sigs = [(sym, s) for sym, s in self.signals.items() if s.get("is_signal")]
        self.sig_tbl.setRowCount(len(sigs))
        for r, (sym, s) in enumerate(sigs):
            try:   live = mt5_live_price(mt5, sym)
            except: live = float("nan")
            sl = s.get("sl") or 0; tp = s.get("tp") or 0; d = s.get("direction","")
            if not math.isnan(live) and sl and tp:
                risk_l = abs(live-sl); rew_l = (tp-live) if "LONG" in d else (live-tp)
                live_r = rew_l/risk_l if risk_l>0.001 else 0
            else: live_r = 0
            d_c = GREEN if "LONG" in d else RED
            lr_c = GREEN if live_r>=3 else (YELLOW if live_r>=1.5 else RED)
            cond = ("LRC ✓ LVL ✓" if s.get("lrc_met") and s.get("level_met")
                    else "LRC ✓ LVL ✗" if s.get("lrc_met") else "LRC ✗ LVL ✓")
            for c, (txt, col) in enumerate([
                (sym, LAVENDER), (d, d_c),
                (f"${live:,.2f}" if not math.isnan(live) else "—", BLUE),
                (f"{live_r:.1f}R", lr_c),
                (f"{s.get('readiness',0):.0f}", MAUVE),
                (cond, GREEN if "✓ LVL ✓" in cond else YELLOW),
            ]):
                self.sig_tbl.setItem(r, c, _item(txt, col))
            self.sig_tbl.setRowHeight(r, 24)

    def _refresh_positions(self):
        # Reuse whatever MT5 connection is currently open — never init/shutdown here
        try:
            import MetaTrader5 as mt5
            acct = mt5.account_info()
            if acct is None:
                return   # MT5 not ready yet (scan still running)
            positions = mt5.positions_get(magic=MAGIC) or []
            self.pos_tbl.setRowCount(len(positions))
            self.acct_lbl.setText(
                f"#{self.cfg['login']}  Balance ${acct.balance:,.0f}  "
                f"Equity ${acct.equity:,.0f}  "
                f"{'[PAPER]' if self.cfg.get('paper_mode') else '[LIVE]'}")
            for r, p in enumerate(positions):
                is_long = p.type == mt5.POSITION_TYPE_BUY
                pnl_c = GREEN if p.profit >= 0 else RED
                for c, (txt, col) in enumerate([
                    (p.symbol, LAVENDER),
                    ("▲ LONG" if is_long else "▼ SHORT", GREEN if is_long else RED),
                    (f"${p.price_open:,.2f}", TEXT),
                    (f"${p.price_current:,.2f}", BLUE),
                    (f"${p.profit:+,.0f}", pnl_c),
                ]):
                    self.pos_tbl.setItem(r, c, _item(txt, col))
                self.pos_tbl.setRowHeight(r, 24)
        except Exception: pass

    # ── Trigger callback ──────────────────────────────────────────────────────
    def _on_triggered(self, sym, d, price, sl, tp, vol):
        mode = "[PAPER]" if self.cfg.get("paper_mode") else "[LIVE]"
        self._log(
            f"{mode} {d} {sym}  qty={vol:.2f}  @ ${price:.2f}  SL=${sl:.2f}  TP=${tp:.2f}",
            YELLOW if self.cfg.get("paper_mode") else GREEN)
        self._refresh_positions()

    # ── Log helper ────────────────────────────────────────────────────────────
    def _log(self, msg: str, color: str = TEXT):
        ts  = datetime.now().strftime("%H:%M:%S")
        html = (f'<span style="color:{OVERLAY1}">{ts}</span> '
                f'<span style="color:{color}">{msg}</span>')
        self.log_box.append(html)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
        log.info(msg)

    # ── Nightly scan ──────────────────────────────────────────────────────────
    def _schedule_nightly(self):
        now = datetime.now()
        nxt = now.replace(hour=SCAN_HOUR, minute=0, second=0, microsecond=0)
        if nxt <= now: nxt += timedelta(days=1)
        ms = int((nxt - now).total_seconds() * 1000)
        QTimer.singleShot(ms, self._nightly_tick)

    def _nightly_tick(self):
        self._log(f"Nightly scan starting ({SCAN_HOUR}:00)", LAVENDER)
        self._start_scan(); self._schedule_nightly()

    def closeEvent(self, ev):
        if self.trig_w: self.trig_w.stop()
        ev.accept()

# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

_LOCK = os.path.join(tempfile.gettempdir(), "lrc_bot.lock")

def _kill_previous():
    """Kill any previously running LRC Bot instance."""
    try:
        if os.path.exists(_LOCK):
            pid = int(Path(_LOCK).read_text().strip())
            if pid != os.getpid():
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               capture_output=True)
        Path(_LOCK).write_text(str(os.getpid()))
        atexit.register(lambda: Path(_LOCK).unlink(missing_ok=True))
    except Exception:
        pass
    # Also reset any stale MT5 session
    try:
        import MetaTrader5 as _mt5; _mt5.shutdown()
    except Exception:
        pass


def main():
    _kill_previous()
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)

    cfg = load_cfg()
    if not cfg.get("login"):
        wiz = SetupWizard()
        if wiz.exec() != QWizard.DialogCode.Accepted:
            sys.exit(0)
        cfg = load_cfg()

    win = StatusWindow(cfg)
    win.setWindowFlags(win.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
    win.show()
    win.raise_()
    win.activateWindow()
    # Drop always-on-top after MT5 settles, but keep window visible
    def _drop_ontop():
        win.setWindowFlags(win.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
        win.show()
        win.raise_()
    QTimer.singleShot(4000, _drop_ontop)
    # Trigger starts automatically in _on_scan_done once scan finishes
    sys.exit(app.exec())

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        _app = QApplication.instance() or QApplication(sys.argv)
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "LRC Bot — Fatal Error", traceback.format_exc())
        sys.exit(1)
