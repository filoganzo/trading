"""
LRC Signal Dashboard — PyQt6 professional GUI.
Run:  python stocks/dashboard/gui.py

Architecture:
  - Price feed : MT5 live prices for all tickers.json symbols, every 60s
  - Scan       : LRC signal engine, runs at 23:00 or on demand
  - Confidence : base_score × (live_R/5) × (1 − speed_penalty)
"""
import sys, json, math
from pathlib import Path
from datetime import datetime, timezone, time as dtime
from collections import deque

sys.path.insert(0, 'path/to/repo/hft')

import pandas as pd
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QLabel, QFrame, QProgressBar, QPushButton, QGridLayout,
    QMessageBox, QDialog, QFormLayout, QLineEdit, QComboBox,
    QDialogButtonBox,
)

# ── Catppuccin Mocha ──────────────────────────────────────────────────────────
CRUST    = "#11111b"; MANTLE  = "#181825"; BASE    = "#1e1e2e"
SURFACE0 = "#313244"; SURFACE1= "#45475a"; SURFACE2= "#585b70"
OVERLAY0 = "#6c7086"; OVERLAY1= "#7f849c"; TEXT    = "#cdd6f4"
SUBTEXT  = "#a6adc8"; LAVENDER= "#b4befe"; BLUE    = "#89b4fa"
TEAL     = "#94e2d5"; GREEN   = "#a6e3a1"; YELLOW  = "#f9e2af"
PEACH    = "#fab387"; MAROON  = "#eba0ac"; RED     = "#f38ba8"
MAUVE    = "#cba6f7"; SKY     = "#89dceb"

TRADES_FILE      = Path(__file__).parent / "open_trades.json"
TICKERS_FILE     = Path(__file__).parent.parent / "config" / "tickers.json"
NIGHTLY_HOUR     = 23
MARKET_OPEN_UTC  = dtime(14, 30)
MARKET_CLOSE_UTC = dtime(21,  0)

# ── Load MT5 symbol list from tickers.json ────────────────────────────────────
def _load_mt5_symbols() -> list[str]:
    try:
        data = json.loads(TICKERS_FILE.read_text())
        all_t = data.get("US", {}).get("tickers", [])
        return [t for t in all_t if "-" not in t]   # drop roll/futures (-24 etc.)
    except Exception:
        return []

ALL_MT5_SYMBOLS: list[str] = _load_mt5_symbols()

# ── Stylesheet ────────────────────────────────────────────────────────────────
QSS = f"""
QMainWindow, QWidget {{
    background:{BASE}; color:{TEXT};
    font-family:'Cascadia Code','Fira Code','Consolas',monospace; font-size:12px;
}}
QTabWidget::pane {{ border:1px solid {SURFACE1}; background:{MANTLE}; }}
QTabBar::tab {{
    background:{SURFACE0}; color:{SUBTEXT};
    padding:6px 18px; border-top-left-radius:4px; border-top-right-radius:4px;
}}
QTabBar::tab:selected {{ background:{MANTLE}; color:{LAVENDER}; border-bottom:2px solid {LAVENDER}; }}
QTabBar::tab:hover {{ color:{TEXT}; }}
QTableWidget {{
    background:{MANTLE}; gridline-color:{SURFACE0}; border:none;
    selection-background-color:{SURFACE1}; selection-color:{TEXT};
    alternate-background-color:{BASE};
}}
QHeaderView::section {{
    background:{SURFACE0}; color:{SUBTEXT}; padding:4px 8px;
    border:none; font-weight:bold; font-size:11px;
}}
QTableWidget::item {{ padding:3px 8px; }}
QScrollBar:vertical   {{ background:{SURFACE0}; width:8px;  border-radius:4px; }}
QScrollBar:horizontal {{ background:{SURFACE0}; height:8px; border-radius:4px; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background:{OVERLAY0}; border-radius:4px; min-height:20px; min-width:20px;
}}
QPushButton {{
    background:{SURFACE0}; color:{TEXT};
    border:1px solid {SURFACE1}; border-radius:4px; padding:5px 14px;
}}
QPushButton:hover  {{ background:{SURFACE1}; color:{LAVENDER}; }}
QPushButton:pressed {{ background:{SURFACE2}; }}
QPushButton#primary {{ background:{MAUVE}; color:{CRUST}; font-weight:bold; border:none; }}
QPushButton#primary:hover {{ background:{LAVENDER}; }}
QPushButton#danger  {{ background:{SURFACE0}; color:{MAROON}; border:1px solid {MAROON}; }}
QPushButton#danger:hover  {{ background:{MAROON}; color:{CRUST}; }}
QFrame#card {{ background:{MANTLE}; border:1px solid {SURFACE1}; border-radius:6px; }}
QProgressBar {{
    background:{SURFACE0}; border-radius:3px; height:8px;
    text-align:center; color:transparent;
}}
QProgressBar::chunk {{ border-radius:3px; }}
QSplitter::handle {{ background:{SURFACE1}; }}
QSplitter::handle:horizontal {{ width:3px; }}
QSplitter::handle:vertical   {{ height:3px; }}
QLineEdit, QComboBox {{
    background:{SURFACE0}; color:{TEXT};
    border:1px solid {SURFACE1}; border-radius:4px; padding:4px 8px;
}}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def _c(h): return QColor(h)
def _nan(v):
    try: return math.isnan(float(v))
    except: return True

def _item(text, color=TEXT, align=Qt.AlignmentFlag.AlignLeft):
    it = QTableWidgetItem(str(text))
    it.setForeground(QBrush(_c(color)))
    it.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
    it.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
    return it

def _num(text, color=TEXT):
    return _item(text, color, Qt.AlignmentFlag.AlignRight)

def _is_market_hours() -> bool:
    now = datetime.now(timezone.utc).time()
    return datetime.now(timezone.utc).weekday() < 5 and MARKET_OPEN_UTC <= now <= MARKET_CLOSE_UTC

def _drift_and_r(live, entry, sl, tp, direction):
    """Returns (drift_pct_of_range, live_r).  Positive drift = moved toward TP."""
    try:
        live, entry, sl, tp = float(live), float(entry), float(sl), float(tp)
        if any(_nan(x) for x in [live, entry, sl, tp]):
            return float("nan"), float("nan")
        if "LONG" in direction:
            total  = tp - entry
            drift  = (live - entry) / total * 100 if abs(total) > 0.001 else 0.0
            risk   = live - sl
            reward = tp - live
        else:
            total  = entry - tp
            drift  = (entry - live) / total * 100 if abs(total) > 0.001 else 0.0
            risk   = sl - live
            reward = live - tp
        live_r = reward / risk if risk > 0.001 else float("nan")
        return drift, live_r
    except Exception:
        return float("nan"), float("nan")

def _confidence(base_score, live_r, speed_pct_per_min) -> float:
    """
    Decaying confidence score (0-100).
    - Starts at base_score from the scan.
    - Scales linearly with remaining R (5R = full, 0R = zero).
    - Speed penalty: price rushing toward TP erodes confidence further.
    """
    if _nan(live_r) or live_r <= 0:
        return 0.0
    r_factor     = max(0.0, min(1.0, live_r / 5.0))
    speed_factor = max(0.5, 1.0 - max(0.0, speed_pct_per_min) / 15.0)  # 15%/min = -50%
    return max(0.0, min(100.0, base_score * r_factor * speed_factor))

# ── Price feed worker ─────────────────────────────────────────────────────────
class PriceWorker(QThread):
    """
    Fetches live prices for ALL tickers.json symbols from the running MT5 terminal.
    Symbols are already in MT5 format (MSFT.NAS, AAPL.NAS, etc.) — direct lookup.
    symbol_select() is called once per symbol to add it to Market Watch.
    """
    finished   = pyqtSignal(dict)   # {mt5_sym: float}
    mt5_status = pyqtSignal(str)

    _selected: set = set()   # class-level: symbols already added to Market Watch

    def __init__(self, symbols: list[str]):
        super().__init__()
        self.symbols = symbols

    def run(self):
        try:
            import MetaTrader5 as mt5
        except ImportError:
            self.mt5_status.emit("missing")
            self.finished.emit({})
            return

        if not mt5.initialize():
            self.mt5_status.emit("offline")
            self.finished.emit({})
            return

        prices = {}
        for sym in self.symbols:
            if sym not in PriceWorker._selected:
                mt5.symbol_select(sym, True)
                PriceWorker._selected.add(sym)
            tick = mt5.symbol_info_tick(sym)
            if tick and tick.bid > 0:
                prices[sym] = (tick.bid + tick.ask) / 2 if tick.ask > 0 else tick.bid

        mt5.shutdown()
        self.mt5_status.emit(f"ok:{len(prices)}")
        self.finished.emit(prices)

# ── Scan worker ───────────────────────────────────────────────────────────────
class ScanWorker(QThread):
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)
    def __init__(self, params):
        super().__init__()
        self.params = params
    def run(self):
        try:
            from stocks.dashboard.signals import scan_all
            self.finished.emit(scan_all(**self.params))
        except Exception as e:
            self.error.emit(str(e))

# ── Base table ────────────────────────────────────────────────────────────────
class BaseTable(QTableWidget):
    row_selected = pyqtSignal(dict)

    def __init__(self, col_defs, parent=None):
        super().__init__(parent)
        self.col_defs = col_defs
        self._data: list[dict] = []
        self.setColumnCount(len(col_defs))
        self.setHorizontalHeaderLabels([c[1] for c in col_defs])
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setHighlightSections(False)
        self.setSortingEnabled(True)
        for i, (_, _, w) in enumerate(col_defs):
            if w == 0:
                self.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
            else:
                self.setColumnWidth(i, w)
        self.currentCellChanged.connect(self._on_sel)

    def _on_sel(self, row, *_):
        if 0 <= row < len(self._data):
            self.row_selected.emit(self._data[row])

    def populate(self, rows: list[dict]):
        self._data = rows
        self.setSortingEnabled(False)
        self.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self._fill(r, row)
            self.setRowHeight(r, 26)
        self.setSortingEnabled(True)

    def update_prices(self, prices: dict):
        """Update live price in row data, then redraw preserving sort."""
        for row in self._data:
            sym = row.get("symbol", "")
            if sym in prices:
                row["_live"] = prices[sym]
        col   = self.horizontalHeader().sortIndicatorSection()
        order = self.horizontalHeader().sortIndicatorOrder()
        self.setSortingEnabled(False)
        for i, row in enumerate(self._data):
            self._fill(i, row)
        self.setSortingEnabled(True)
        if col >= 0:
            self.sortByColumn(col, order)

    def _fill(self, r: int, row: dict):
        raise NotImplementedError

# ── Setups table ──────────────────────────────────────────────────────────────
_SETUP_COLS = [
    ("symbol",     "Symbol",     90),
    ("live_status","Live",       75),
    ("direction",  "Dir",        55),
    ("live",       "Live $",     82),
    ("drift",      "Drift %",    65),
    ("speed",      "Speed",      55),
    ("live_r",     "Live R",     60),
    ("confidence", "Conf",       52),
    ("entry",      "Entry Est",  82),
    ("sl",         "Stop",       78),
    ("tp",         "Target",     78),
    ("shares",     "Shares",     62),
    ("risk_usd",   "Risk $",     68),
    ("condition",  "Conditions",  0),
    ("vol_tier",   "Tier",       50),
    ("data_date",  "Date",       85),
]

class SetupsTable(BaseTable):
    def __init__(self, parent=None):
        super().__init__(_SETUP_COLS, parent)

    def _fill(self, r, row):
        d     = row.get("direction", "")
        entry = float(row.get("close", float("nan")))
        sl    = float(row.get("sl",    float("nan")))
        tp    = float(row.get("tp",    float("nan")))
        live  = float(row.get("_live", entry))
        speed = float(row.get("_speed", 0.0))
        base  = float(row.get("readiness", 0))

        drift, live_r = _drift_and_r(live, entry, sl, tp, d)
        conf = _confidence(base, live_r, speed) if not _nan(live_r) else float("nan")

        # Live status
        if _nan(live_r):
            lst, lst_c, expired = "—", OVERLAY0, False
        elif live_r <= 0 or (not _nan(drift) and drift > 100):
            lst, lst_c, expired = "✕ EXPIRED", RED, True
        elif live_r < 1.5:
            lst, lst_c, expired = "⚠ SKIP",    PEACH, False
        elif live_r < 2.5:
            lst, lst_c, expired = "~ WEAK",    YELLOW, False
        else:
            lst, lst_c, expired = "● ACTIVE",  GREEN, False

        # Drift
        drift_s = f"{drift:+.1f}%" if not _nan(drift) else "—"
        drift_c = GREEN if (not _nan(drift) and drift <= 10) else (YELLOW if (not _nan(drift) and drift <= 35) else RED)

        # Speed (% of TP range per minute moving toward TP)
        if speed > 4:
            spd_s, spd_c = "→→→", RED
        elif speed > 1.5:
            spd_s, spd_c = "→→",  YELLOW
        elif speed > 0.4:
            spd_s, spd_c = "→",   PEACH
        elif speed < -1:
            spd_s, spd_c = "←",   TEAL   # drifting back toward SL — better entry coming
        else:
            spd_s, spd_c = "·",   OVERLAY0

        # Live R
        lr_s = f"{live_r:.1f}R" if not _nan(live_r) else "—"
        lr_c = GREEN if (not _nan(live_r) and live_r >= 4) else (
               YELLOW if (not _nan(live_r) and live_r >= 2.5) else (
               PEACH  if (not _nan(live_r) and live_r >= 1.5) else RED))

        # Confidence
        conf_s = f"{conf:.0f}" if not _nan(conf) else "—"
        conf_c = GREEN if (not _nan(conf) and conf >= 60) else (
                 YELLOW if (not _nan(conf) and conf >= 35) else RED)

        cond   = row.get("condition", "—")
        cond_c = GREEN if "✓  LVL ✓" in cond else (
                 YELLOW if "LRC ✓" in cond else (
                 PEACH  if "LVL ✓" in cond else OVERLAY0))

        tier_l = {"low_vol":"Low","mid_vol":"Mid","high_vol":"High"}.get(row.get("vol_tier",""), "?")
        tier_c = {"low_vol":SKY, "mid_vol":YELLOW,"high_vol":RED}.get(row.get("vol_tier",""), SUBTEXT)

        dim    = OVERLAY1 if expired else None
        def _tc(c): return dim or c

        items = [
            _item(row.get("symbol",""),                         _tc(LAVENDER)),
            _item(lst, lst_c),
            _item(("▲ " if "LONG" in d else "▼ ") + d,        _tc(GREEN if "LONG" in d else RED)),
            _num(f"${live:,.2f}"  if not _nan(live)  else "—", _tc(BLUE)),
            _num(drift_s, drift_c),
            _item(spd_s, spd_c),
            _num(lr_s, lr_c),
            _num(conf_s, conf_c),
            _num(f"${entry:,.2f}" if not _nan(entry) else "—", _tc(TEXT)),
            _num(f"${sl:,.2f}"   if not _nan(sl)    else "—", _tc(MAROON)),
            _num(f"${tp:,.2f}"   if not _nan(tp)    else "—", _tc(TEAL)),
            _num(f"{int(row.get('shares',0)):,}",               _tc(TEXT)),
            _num(f"${row.get('risk_usd',0):,.0f}",              _tc(TEXT)),
            _item(cond, _tc(cond_c)),
            _item(tier_l, _tc(tier_c)),
            _item(str(row.get("data_date",""))[:10],            OVERLAY1),
        ]
        if expired:
            bg = QColor(RED); bg.setAlpha(18)
            for it in items: it.setBackground(QBrush(bg))
        for c, it in enumerate(items):
            self.setItem(r, c, it)

# ── Pipeline table (NEAR/WATCH rows) ─────────────────────────────────────────
_PIPE_COLS = [
    ("symbol",     "Symbol",    90),
    ("direction",  "Dir",       55),
    ("status",     "Status",    65),
    ("condition",  "Conditions", 0),
    ("readiness",  "Score",     55),
    ("dev_sigma",  "Dev σ",     62),
    ("level_dist", "Lvl Dist%", 72),
    ("live",       "Live $",    82),
    ("lrc_mid",    "LRC Mid",   82),
    ("sector",     "Sector",    90),
    ("vol_tier",   "Tier",      50),
    ("data_date",  "Date",      85),
]

class PipelineTable(BaseTable):
    def __init__(self, parent=None):
        super().__init__(_PIPE_COLS, parent)

    def _fill(self, r, row):
        d   = row.get("direction", "")
        d_c = GREEN if "LONG" in d else (RED if "SHORT" in d else OVERLAY1)
        st  = row.get("status", "")
        st_c = {"SIGNAL":GREEN,"NEAR":YELLOW,"WATCH":PEACH,"COLD":OVERLAY0}.get(st, TEXT)
        cond  = row.get("condition", "—")
        cond_c = GREEN if "✓  LVL ✓" in cond else (YELLOW if "LRC ✓" in cond else PEACH if "LVL ✓" in cond else OVERLAY0)
        dev   = float(row.get("dev_sigma", 0))
        ld    = float(row.get("level_dist", 99))
        live  = float(row.get("_live", row.get("close", float("nan"))))
        tier_l = {"low_vol":"Low","mid_vol":"Mid","high_vol":"High"}.get(row.get("vol_tier",""), "?")
        tier_c = {"low_vol":SKY, "mid_vol":YELLOW,"high_vol":RED}.get(row.get("vol_tier",""), SUBTEXT)
        items = [
            _item(row.get("symbol",""), LAVENDER),
            _item(("▲ " if "LONG" in d else "▼ " if "SHORT" in d else "") + d, d_c),
            _item(st, st_c),
            _item(cond, cond_c),
            _num(f"{float(row.get('readiness',0)):.0f}", MAUVE),
            _num(f"{dev:+.2f}σ", RED if dev > 0 else GREEN),
            _num(f"{ld:.2f}%",   GREEN if ld <= 1.5 else (YELLOW if ld <= 3.0 else OVERLAY1)),
            _num(f"${live:,.2f}" if not _nan(live) else "—", BLUE),
            _num(f"${float(row.get('lrc_mid',0)):,.2f}"),
            _item(row.get("sector",""), SUBTEXT),
            _item(tier_l, tier_c),
            _item(str(row.get("data_date",""))[:10], OVERLAY1),
        ]
        for c, it in enumerate(items): self.setItem(r, c, it)

# ── Universe table ────────────────────────────────────────────────────────────
_UNIV_COLS = [
    ("symbol",     "Symbol",    90),
    ("status",     "Status",    65),
    ("direction",  "Dir",       75),
    ("condition",  "Conditions", 0),
    ("readiness",  "Score",     55),
    ("dev_sigma",  "Dev σ",     62),
    ("level_dist", "Lvl Dist%", 72),
    ("close",      "Close",     80),
    ("lrc_mid",    "LRC Mid",   80),
    ("sector",     "Sector",    90),
    ("exchange",   "Exch",      50),
    ("vol_tier",   "Tier",      50),
    ("data_date",  "Date",      85),
]

class UniverseTable(BaseTable):
    def __init__(self, parent=None):
        super().__init__(_UNIV_COLS, parent)

    def _fill(self, r, row):
        d   = row.get("direction", "")
        st  = row.get("status", "")
        cond  = row.get("condition", "—")
        dev   = float(row.get("dev_sigma", 0))
        ld    = float(row.get("level_dist", 99))
        tier_l = {"low_vol":"Low","mid_vol":"Mid","high_vol":"High"}.get(row.get("vol_tier",""), "?")
        tier_c = {"low_vol":SKY, "mid_vol":YELLOW,"high_vol":RED}.get(row.get("vol_tier",""), SUBTEXT)
        items = [
            _item(row.get("symbol",""), LAVENDER),
            _item(st, {"SIGNAL":GREEN,"NEAR":YELLOW,"WATCH":PEACH,"COLD":OVERLAY0}.get(st, TEXT)),
            _item(("▲ " if "LONG" in d else "▼ " if "SHORT" in d else "") + d,
                  GREEN if "LONG" in d else (RED if "SHORT" in d else OVERLAY1)),
            _item(cond, GREEN if "✓  LVL ✓" in cond else (YELLOW if "LRC ✓" in cond else PEACH if "LVL ✓" in cond else OVERLAY0)),
            _num(f"{float(row.get('readiness',0)):.0f}", MAUVE),
            _num(f"{dev:+.2f}σ", RED if dev > 0 else GREEN),
            _num(f"{ld:.2f}%",   GREEN if ld <= 1.5 else (YELLOW if ld <= 3.0 else OVERLAY1)),
            _num(f"${float(row.get('close',0)):,.2f}"),
            _num(f"${float(row.get('lrc_mid',0)):,.2f}"),
            _item(row.get("sector",""), SUBTEXT),
            _item(row.get("exchange",""), OVERLAY1),
            _item(tier_l, tier_c),
            _item(str(row.get("data_date",""))[:10], OVERLAY1),
        ]
        for c, it in enumerate(items): self.setItem(r, c, it)

# ── Open Trades table ─────────────────────────────────────────────────────────
_OT_COLS = [
    ("symbol",  "Symbol",  88), ("direction","Dir",   55),
    ("entry",   "Entry $", 80), ("live",     "Live $",80),
    ("sl",      "Stop $",  78), ("tp",       "Target",78),
    ("shares",  "Shares",  62), ("pnl_usd",  "P&L $", 82),
    ("pnl_r",   "P&L R",  62), ("drift",    "Drift%", 65),
    ("live_r",  "Live R",  62), ("opened",   "Opened",105),
    ("close_btn","",       65),
]

class OpenTradesTable(QTableWidget):
    trade_closed = pyqtSignal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._trades: list[dict] = []
        self.setColumnCount(len(_OT_COLS))
        self.setHorizontalHeaderLabels([c[1] for c in _OT_COLS])
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setHighlightSections(False)
        self.setSortingEnabled(False)
        for i, (_, _, w) in enumerate(_OT_COLS): self.setColumnWidth(i, w)

    def populate(self, trades, prices={}):
        self._trades = trades
        self.setRowCount(len(trades))
        for r, t in enumerate(trades): self._fill(r, t, prices)

    def update_prices(self, prices):
        for r, t in enumerate(self._trades):
            if t.get("symbol","") in prices:
                t["_live"] = prices[t["symbol"]]
            self._fill(r, t, prices)

    def _fill(self, r, t, prices={}):
        sym   = t.get("symbol","")
        d     = t.get("direction","LONG")
        entry = float(t.get("entry",0) or 0)
        sl    = float(t.get("sl",0) or 0)
        tp    = float(t.get("tp",0) or 0)
        shrs  = int(float(t.get("shares",0) or 0))
        live  = float(t.get("_live", prices.get(sym, entry)) or entry)
        t["_live"] = live
        is_long = "LONG" in d
        risk_ps = abs(entry - sl)
        pnl_usd = (live - entry) * shrs * (1 if is_long else -1)
        pnl_r   = pnl_usd / (risk_ps * shrs) if risk_ps * shrs > 0.001 else 0.0
        pnl_c   = GREEN if pnl_usd > 0 else (RED if pnl_usd < 0 else TEXT)
        drift, live_r = _drift_and_r(live, entry, sl, tp, d)
        items = [
            _item(sym, LAVENDER),
            _item(("▲ " if is_long else "▼ ") + d, GREEN if is_long else RED),
            _num(f"${entry:,.2f}"),
            _num(f"${live:,.2f}", BLUE),
            _num(f"${sl:,.2f}", MAROON),
            _num(f"${tp:,.2f}", TEAL),
            _num(f"{shrs:,}"),
            _num(f"${pnl_usd:+,.0f}", pnl_c),
            _num(f"{pnl_r:+.2f}R", pnl_c),
            _num(f"{drift:+.1f}%" if not _nan(drift) else "—",
                 GREEN if (not _nan(drift) and drift <= 10) else (YELLOW if (not _nan(drift) and drift <= 35) else RED)),
            _num(f"{live_r:.1f}R" if not _nan(live_r) else "—",
                 GREEN if (not _nan(live_r) and live_r >= 4) else (YELLOW if (not _nan(live_r) and live_r >= 2.5) else RED)),
            _item(str(t.get("opened",""))[:16], OVERLAY1),
        ]
        for c, it in enumerate(items): self.setItem(r, c, it)
        btn = QPushButton("Close"); btn.setObjectName("danger"); btn.setFixedSize(56, 20)
        btn.clicked.connect(lambda _, idx=r: self.trade_closed.emit(idx))
        self.setCellWidget(r, len(_OT_COLS)-1, btn)
        self.setRowHeight(r, 26)

# ── Trade Card ────────────────────────────────────────────────────────────────
class TradeCard(QFrame):
    take_trade = pyqtSignal(dict)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card"); self.setMinimumWidth(220)
        self._row = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12,12,12,12); lay.setSpacing(6)
        self._header = QLabel("Select a row"); self._header.setObjectName("card_header")
        lay.addWidget(self._header)
        self._grid = QGridLayout(); self._grid.setColumnStretch(1,1)
        lay.addLayout(self._grid)
        self._bar = QProgressBar(); self._bar.setRange(0,100); self._bar.setFixedHeight(8)
        lay.addWidget(self._bar)
        self._conf_bar = QProgressBar(); self._conf_bar.setRange(0,100); self._conf_bar.setFixedHeight(6)
        lay.addWidget(self._conf_bar)
        self._take_btn = QPushButton("Take Trade")
        self._take_btn.setObjectName("primary"); self._take_btn.setEnabled(False)
        self._take_btn.clicked.connect(lambda: self.take_trade.emit(self._row))
        lay.addWidget(self._take_btn); lay.addStretch()
        self._fields: dict[str, QLabel] = {}
        spec = [
            ("Symbol","symbol"),("Direction","direction"),("Conditions","condition"),
            ("Live Price","live"),("Entry Est","entry"),("Stop Loss","sl"),
            ("Take Profit","tp"),("Shares","shares"),("Risk $","risk_usd"),
            ("Reward $","reward"),("Live R","live_r"),("Drift %","drift"),
            ("Confidence","conf"),("Speed","speed"),("Sector","sector"),
            ("Data Date","data_date"),
        ]
        for i, (lbl, key) in enumerate(spec):
            ql = QLabel(lbl+":"); ql.setStyleSheet(f"color:{OVERLAY1};font-size:11px;")
            qv = QLabel("—"); qv.setStyleSheet(f"color:{TEXT};"); qv.setWordWrap(True)
            self._grid.addWidget(ql,i,0); self._grid.addWidget(qv,i,1)
            self._fields[key] = qv

    def update_row(self, row: dict, prices: dict = {}, speeds: dict = {}):
        self._row = row
        sym   = row.get("symbol","")
        d     = row.get("direction","")
        close = float(row.get("close", float("nan")))
        sl    = float(row.get("sl",    float("nan")))
        tp    = float(row.get("tp",    float("nan")))
        rr    = float(row.get("risk_usd",0))
        live  = float(prices.get(sym, row.get("_live", close)))
        speed = float(speeds.get(sym, row.get("_speed", 0.0)))
        score = float(row.get("readiness",0))
        drift, live_r = _drift_and_r(live, close, sl, tp, d)
        conf  = _confidence(score, live_r, speed)

        d_col = GREEN if "LONG" in d else RED
        self._header.setText(f"{sym}  {d}")
        self._header.setStyleSheet(f"color:{d_col};font-size:14px;font-weight:bold;")
        self._fields["symbol"].setText(sym)
        self._fields["direction"].setText(d)
        self._fields["condition"].setText(row.get("condition","—"))
        self._fields["live"].setText(f"${live:,.2f}" if not _nan(live) else "—")
        self._fields["entry"].setText(f"${close:,.2f}" if not _nan(close) else "—")
        self._fields["sl"].setText(f"${sl:,.2f}" if not _nan(sl) else "—")
        self._fields["tp"].setText(f"${tp:,.2f}" if not _nan(tp) else "—")
        self._fields["shares"].setText(f"{int(row.get('shares',0)):,}")
        self._fields["risk_usd"].setText(f"${rr:,.0f}")
        self._fields["reward"].setText(f"${rr*5:,.0f}")
        self._fields["live_r"].setText(f"{live_r:.1f}R" if not _nan(live_r) else "—")
        self._fields["drift"].setText(f"{drift:+.1f}%" if not _nan(drift) else "—")
        self._fields["conf"].setText(f"{conf:.0f}/100")
        spd_txt = ("→→→ fast" if speed>4 else "→→ medium" if speed>1.5 else "→ slow" if speed>0.4
                   else "← retreating" if speed<-1 else "· stable")
        self._fields["speed"].setText(spd_txt)
        self._fields["sector"].setText(row.get("sector","—"))
        self._fields["data_date"].setText(str(row.get("data_date","—"))[:10])

        self._bar.setValue(int(score))
        chunk = GREEN if score>=70 else (YELLOW if score>=40 else RED)
        self._bar.setStyleSheet(f"QProgressBar::chunk{{background:{chunk};border-radius:3px;}}")
        self._conf_bar.setValue(int(conf))
        cc = GREEN if conf>=60 else (YELLOW if conf>=35 else RED)
        self._conf_bar.setStyleSheet(f"QProgressBar::chunk{{background:{cc};border-radius:3px;}}")

        expired = (not _nan(live_r) and live_r <= 0) or (not _nan(drift) and drift > 100)
        is_signal = row.get("status","") == "SIGNAL" and not expired
        self._take_btn.setEnabled(is_signal)
        self._take_btn.setText("Take Trade" if not expired else "✕ Expired")

    def update_live(self, prices, speeds):
        if self._row: self.update_row(self._row, prices, speeds)

# ── Sector Heat ───────────────────────────────────────────────────────────────
class SectorHeat(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self); lay.setContentsMargins(8,8,8,8); lay.setSpacing(4)
        hdr = QLabel("Sector Heat"); hdr.setStyleSheet(f"color:{LAVENDER};font-weight:bold;")
        lay.addWidget(hdr)
        self._tbl = QTableWidget()
        self._tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.setColumnCount(4)
        self._tbl.setHorizontalHeaderLabels(["Sector","Sig","Dev σ","Near"])
        self._tbl.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self._tbl)

    def update_data(self, df: pd.DataFrame, open_syms: set = set()):
        if df.empty: return
        rows = []
        for sec, grp in df.groupby("sector"):
            rows.append((str(sec),
                         int((grp["status"]=="SIGNAL").sum()),
                         float(grp["dev_sigma"].mean()),
                         int((grp["status"]=="NEAR").sum()),
                         int(grp["symbol"].isin(open_syms).sum())))
        rows.sort(key=lambda x: -x[1])
        self._tbl.setRowCount(len(rows))
        for r, (sec, nsig, avgd, nnear, nopen) in enumerate(rows):
            bg = QColor(GREEN if nsig>0 else (YELLOW if nnear>0 else SURFACE0)); bg.setAlpha(45)
            for c, (txt, col) in enumerate([
                (("◉ " if nopen>0 else "")+sec, BLUE if nopen>0 else TEXT),
                (str(nsig),  GREEN  if nsig>0  else OVERLAY0),
                (f"{avgd:+.2f}σ", RED if avgd>0 else GREEN),
                (str(nnear), YELLOW if nnear>0 else OVERLAY0),
            ]):
                it = _item(txt, col); it.setBackground(QBrush(bg))
                self._tbl.setItem(r, c, it)
            self._tbl.setRowHeight(r, 22)

# ── Add Trade Dialog ──────────────────────────────────────────────────────────
class AddTradeDialog(QDialog):
    def __init__(self, prefill={}, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Log Trade"); self.setMinimumWidth(300); self.setModal(True)
        form = QFormLayout(self); self._fields = {}
        for key, lbl, typ in [
            ("symbol","Symbol",str),("direction","Direction","combo"),
            ("entry","Entry $",float),("sl","Stop Loss $",float),
            ("tp","Target $",float),("shares","Shares",int),
        ]:
            val = prefill.get(key,"")
            if typ == "combo":
                w = QComboBox(); w.addItems(["LONG","SHORT"])
                if str(val)=="SHORT": w.setCurrentIndex(1)
            else:
                w = QLineEdit(str(val) if val!="" else "")
            form.addRow(lbl+":", w); self._fields[key] = w
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        form.addRow(btns)

    def values(self):
        out = {}
        for key, w in self._fields.items():
            if isinstance(w, QComboBox): out[key] = w.currentText()
            else:
                raw = w.text().strip()
                out[key] = raw.upper() if key=="symbol" else (float(raw) if raw else "")
        return out

# ── Main Window ───────────────────────────────────────────────────────────────
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LRC Dashboard  ·  FiloHFT")
        self.resize(1500, 900)
        self._df:     pd.DataFrame = pd.DataFrame()
        self._prices: dict         = {}    # mt5_sym -> float
        self._speeds: dict         = {}    # mt5_sym -> float (% TP-range/min)
        self._history: dict        = {}    # mt5_sym -> deque[(datetime, float)]
        self._trades: list[dict]   = self._load_trades()
        self._scan_worker          = None
        self._price_worker         = None
        self._params               = dict(account=100_000, risk_pct=0.01)
        self._build_ui()
        self._schedule_nightly()
        self._price_tick()      # start price feed immediately
        self._start_scan()      # initial scan on launch

    # ── Persistence ───────────────────────────────────────────────────────────
    def _load_trades(self):
        try: return json.loads(TRADES_FILE.read_text()) if TRADES_FILE.exists() else []
        except: return []
    def _save_trades(self):
        TRADES_FILE.write_text(json.dumps(self._trades, indent=2, default=str))

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # Header
        hbar = QFrame()
        hbar.setStyleSheet(f"background:{CRUST};border-bottom:1px solid {SURFACE0};")
        hbar.setFixedHeight(50)
        hl = QHBoxLayout(hbar); hl.setContentsMargins(16,0,16,0)
        title = QLabel("LRC DASHBOARD  ·  FiloHFT")
        title.setStyleSheet(f"color:{LAVENDER};font-size:15px;font-weight:bold;letter-spacing:2px;")
        hl.addWidget(title); hl.addStretch()
        self._mkt_lbl = QLabel("○ Market closed")
        self._mkt_lbl.setStyleSheet(f"color:{OVERLAY0};font-size:11px;")
        hl.addWidget(self._mkt_lbl)
        self._price_lbl = QLabel("MT5: connecting…")
        self._price_lbl.setStyleSheet(f"color:{OVERLAY0};font-size:11px;margin-left:16px;")
        hl.addWidget(self._price_lbl)
        self._status_lbl = QLabel("Scan: waiting")
        self._status_lbl.setStyleSheet(f"color:{OVERLAY1};font-size:11px;margin-left:16px;")
        hl.addWidget(self._status_lbl)
        scan_btn = QPushButton("⟳  Scan Now"); scan_btn.setObjectName("primary")
        scan_btn.setFixedWidth(110); scan_btn.clicked.connect(self._start_scan)
        hl.addWidget(scan_btn)
        root.addWidget(hbar)

        # Body
        body = QSplitter(Qt.Orientation.Horizontal); body.setChildrenCollapsible(False)
        root.addWidget(body)
        self._tabs = QTabWidget(); body.addWidget(self._tabs)
        right = QSplitter(Qt.Orientation.Vertical); right.setChildrenCollapsible(False)
        right.setMinimumWidth(240); right.setMaximumWidth(370)
        body.addWidget(right)
        self._card = TradeCard(); self._card.take_trade.connect(self._on_take_trade)
        right.addWidget(self._card)
        self._heat = SectorHeat(); right.addWidget(self._heat)
        right.setSizes([520,280]); body.setStretchFactor(0,1); body.setStretchFactor(1,0)

        # Tab 1: Setups
        self._setup_tbl = SetupsTable()
        self._setup_tbl.row_selected.connect(
            lambda r: self._card.update_row(r, self._prices, self._speeds))
        self._tabs.addTab(self._setup_tbl, "⚡ Setups")

        # Tab 2: Pipeline
        pw = QWidget(); pl = QVBoxLayout(pw); pl.setContentsMargins(0,0,0,0)
        pl.addWidget(QLabel("  Approaching signal — one condition missing"))
        self._pipe_tbl = PipelineTable()
        self._pipe_tbl.row_selected.connect(
            lambda r: self._card.update_row(r, self._prices, self._speeds))
        pl.addWidget(self._pipe_tbl); self._tabs.addTab(pw, "📡 Pipeline")

        # Tab 3: Longs
        self._long_tbl = PipelineTable()
        self._long_tbl.row_selected.connect(
            lambda r: self._card.update_row(r, self._prices, self._speeds))
        self._tabs.addTab(self._long_tbl, "▲ Longs")

        # Tab 4: Shorts
        self._short_tbl = PipelineTable()
        self._short_tbl.row_selected.connect(
            lambda r: self._card.update_row(r, self._prices, self._speeds))
        self._tabs.addTab(self._short_tbl, "▼ Shorts")

        # Tab 5: Open Trades
        ow = QWidget(); ol = QVBoxLayout(ow); ol.setContentsMargins(0,0,0,0)
        ob = QHBoxLayout()
        ob.addWidget(QLabel("  Manually logged positions — live P&L"))
        ob.addStretch()
        add_btn = QPushButton("+ Log Trade"); add_btn.clicked.connect(lambda: self._on_take_trade({}))
        ob.addWidget(add_btn); ol.addLayout(ob)
        self._ot_tbl = OpenTradesTable(); self._ot_tbl.trade_closed.connect(self._on_close_trade)
        ol.addWidget(self._ot_tbl); self._tabs.addTab(ow, "📂 Open Trades")

        # Tab 6: Universe
        self._univ_tbl = UniverseTable()
        self._univ_tbl.row_selected.connect(
            lambda r: self._card.update_row(r, self._prices, self._speeds))
        self._tabs.addTab(self._univ_tbl, "🌐 Universe")

        # Footer
        foot = QFrame()
        foot.setStyleSheet(f"background:{CRUST};border-top:1px solid {SURFACE0};")
        foot.setFixedHeight(26)
        fl = QHBoxLayout(foot); fl.setContentsMargins(16,0,16,0)
        lbl = QLabel(
            "Strategy: lev+0.3atr|rr5|none  ·  backtest stats: local only  ·  "
            "Conf = scan_score × (LiveR/5) × (1−speed_penalty)  ·  "
            "⚠ Entry est = last close — fill at next open")
        lbl.setStyleSheet(f"color:{OVERLAY0};font-size:10px;"); fl.addWidget(lbl)
        root.addWidget(foot)

    # ── Scan ──────────────────────────────────────────────────────────────────
    def _start_scan(self):
        if self._scan_worker and self._scan_worker.isRunning(): return
        self._status_lbl.setText("Scanning…")
        self._status_lbl.setStyleSheet(f"color:{YELLOW};font-size:11px;margin-left:16px;")
        self._scan_worker = ScanWorker(self._params)
        self._scan_worker.finished.connect(self._on_scan_done)
        self._scan_worker.error.connect(self._on_scan_error)
        self._scan_worker.start()

    def _on_scan_done(self, df: pd.DataFrame):
        self._df = df
        ts = datetime.now().strftime("%H:%M:%S")
        n  = int((df["status"]=="SIGNAL").sum()) if not df.empty else 0
        self._status_lbl.setText(f"Scan: {ts}  {n} signal{'s' if n!=1 else ''}")
        self._status_lbl.setStyleSheet(f"color:{GREEN};font-size:11px;margin-left:16px;")
        self._populate_all(df)
        self._heat.update_data(df, {t.get("symbol") for t in self._trades})
        self._ot_tbl.populate(self._trades, self._prices)
        # Apply any prices we already have to the newly scanned rows
        if self._prices:
            self._apply_prices_to_tables()

    def _on_scan_error(self, msg):
        self._status_lbl.setText(f"Scan error: {msg[:60]}")
        self._status_lbl.setStyleSheet(f"color:{RED};font-size:11px;margin-left:16px;")

    def _populate_all(self, df: pd.DataFrame):
        if df.empty:
            for t in [self._setup_tbl, self._pipe_tbl, self._long_tbl,
                      self._short_tbl, self._univ_tbl]:
                t.populate([])
            return
        rows  = df.to_dict("records")
        sig   = [r for r in rows if r.get("status")=="SIGNAL"]
        pipe  = [r for r in rows if r.get("status") in ("NEAR","WATCH")]
        longs = [r for r in rows if r.get("status") in ("SIGNAL","NEAR","WATCH")
                 and "LONG" in str(r.get("direction",""))]
        shrts = [r for r in rows if r.get("status") in ("SIGNAL","NEAR","WATCH")
                 and "SHORT" in str(r.get("direction",""))]
        self._setup_tbl.populate(sig)
        self._pipe_tbl.populate(pipe)
        self._long_tbl.populate(longs)
        self._short_tbl.populate(shrts)
        self._univ_tbl.populate(rows)
        self._tabs.setTabText(0, f"⚡ Setups ({len(sig)})")
        self._tabs.setTabText(1, f"📡 Pipeline ({len(pipe)})")
        self._tabs.setTabText(2, f"▲ Longs ({len(longs)})")
        self._tabs.setTabText(3, f"▼ Shorts ({len(shrts)})")
        self._tabs.setTabText(4, f"📂 Open Trades ({len(self._trades)})")

    # ── Price feed ────────────────────────────────────────────────────────────
    def _price_tick(self):
        """Fire a price fetch, then reschedule."""
        if not (self._price_worker and self._price_worker.isRunning()):
            self._price_worker = PriceWorker(ALL_MT5_SYMBOLS)
            self._price_worker.finished.connect(self._on_prices)
            self._price_worker.mt5_status.connect(self._on_mt5_status)
            self._price_worker.start()
        interval = 60_000 if _is_market_hours() else 300_000
        QTimer.singleShot(interval, self._price_tick)

    def _on_mt5_status(self, status: str):
        mh = _is_market_hours()
        self._mkt_lbl.setText("● Market open" if mh else "○ Market closed")
        self._mkt_lbl.setStyleSheet(f"color:{GREEN if mh else OVERLAY0};font-size:11px;")
        if status == "offline":
            self._price_lbl.setText("MT5 ✕ offline")
            self._price_lbl.setStyleSheet(f"color:{MAROON};font-size:11px;margin-left:16px;")
        elif status == "missing":
            self._price_lbl.setText("MetaTrader5 not installed")
            self._price_lbl.setStyleSheet(f"color:{RED};font-size:11px;margin-left:16px;")

    def _on_prices(self, prices: dict):
        if not prices: return
        ts = datetime.now()

        # Update history for speed calculation
        for sym, price in prices.items():
            if sym not in self._history:
                self._history[sym] = deque(maxlen=10)
            self._history[sym].append((ts, price))

        self._prices.update(prices)

        # Compute speed (% of TP range per minute, toward TP) for signal rows
        self._speeds = {}
        if not self._df.empty:
            sig_rows = self._df[self._df["status"]=="SIGNAL"]
            for _, row in sig_rows.iterrows():
                sym  = row.get("symbol","")
                hist = self._history.get(sym)
                if not hist or len(hist) < 2: continue
                t0, p0 = hist[0]; t1, p1 = hist[-1]
                dt_min = (t1 - t0).total_seconds() / 60
                if dt_min < 0.1: continue
                entry = float(row.get("close", float("nan")))
                tp    = float(row.get("tp",    float("nan")))
                d     = row.get("direction","")
                if _nan(entry) or _nan(tp): continue
                total = abs(tp - entry)
                if total < 0.001: continue
                moved = (p1-p0) if "LONG" in d else (p0-p1)
                self._speeds[sym] = (moved / total) * 100 / dt_min

            # Inject speed back into row data for table refresh
            for row in self._setup_tbl._data:
                row["_speed"] = self._speeds.get(row.get("symbol",""), 0.0)

        self._apply_prices_to_tables()

        n_got = len(prices)
        n_req = len(ALL_MT5_SYMBOLS)
        self._price_lbl.setText(f"MT5 ● {ts.strftime('%H:%M:%S')}  {n_got}/{n_req}")
        self._price_lbl.setStyleSheet(f"color:{TEAL};font-size:11px;margin-left:16px;")

    def _apply_prices_to_tables(self):
        for tbl in [self._setup_tbl, self._pipe_tbl, self._long_tbl,
                    self._short_tbl, self._univ_tbl]:
            tbl.update_prices(self._prices)
        self._ot_tbl.update_prices(self._prices)
        self._card.update_live(self._prices, self._speeds)

    # ── Nightly scan ──────────────────────────────────────────────────────────
    def _schedule_nightly(self):
        import datetime as _dt
        now = datetime.now()
        nxt = now.replace(hour=NIGHTLY_HOUR, minute=0, second=0, microsecond=0)
        if nxt <= now: nxt += _dt.timedelta(days=1)
        QTimer.singleShot(int((nxt-now).total_seconds()*1000), self._nightly_tick)
    def _nightly_tick(self):
        self._start_scan(); self._schedule_nightly()

    # ── Open Trades ───────────────────────────────────────────────────────────
    def _on_take_trade(self, row: dict):
        dlg = AddTradeDialog({
            "symbol":    row.get("symbol",""),
            "direction": row.get("direction","LONG"),
            "entry":     row.get("close",""),
            "sl":        row.get("sl",""),
            "tp":        row.get("tp",""),
            "shares":    row.get("shares",""),
        }, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            vals = dlg.values(); vals["opened"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            self._trades.append(vals); self._save_trades()
            self._ot_tbl.populate(self._trades, self._prices)
            self._heat.update_data(self._df, {t.get("symbol") for t in self._trades})
            self._tabs.setTabText(4, f"📂 Open Trades ({len(self._trades)})")

    def _on_close_trade(self, idx: int):
        if 0 <= idx < len(self._trades):
            t = self._trades[idx]
            if QMessageBox.question(self, "Close trade?",
                f"Remove {t.get('symbol','')} {t.get('direction','')} from tracking?",
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                self._trades.pop(idx); self._save_trades()
                self._ot_tbl.populate(self._trades, self._prices)
                self._heat.update_data(self._df, {t.get("symbol") for t in self._trades})
                self._tabs.setTabText(4, f"📂 Open Trades ({len(self._trades)})")

# ── Entry ─────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = Dashboard()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
