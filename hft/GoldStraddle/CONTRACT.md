# GoldStraddle — IPC Contract v1

Authoritative interface between the **panel** (Python + browser UI on the PC) and the
**EA** (`GoldStraddle.mq5` running on an XAUUSD chart in MT5 / IC Markets EU).

Nobody may change field names or semantics without updating this file.

---

## 1. Transport

Plain JSON files in the MT5 **Common Files** folder, so the panel and any terminal
share them without DLLs, sockets, or permission changes:

```
C:\Users\<user>\AppData\Roaming\MetaQuotes\Terminal\Common\Files\GoldStraddle\
    config.json     panel  -> EA    (settings; EA polls it)
    state.json      EA     -> panel (telemetry; panel polls it)
    command.json    panel  -> EA    (one-shot actions)
```

From MQL5 this folder is reached with the `FILE_COMMON` flag and the relative path
`GoldStraddle\config.json`. MT5 creates intermediate directories on write.

**Encoding:** UTF-8 without BOM. MQL5 side uses `FILE_TXT | FILE_ANSI` (ASCII-safe
content only — no non-ASCII in any string field).

### Write atomicity (mandatory)

A reader must never see a half-written file.

* **Panel writes:** write `config.json.tmp` in the same directory, flush, then
  `os.replace()` onto `config.json`. Same for `command.json`.
* **EA writes:** write `state.json.tmp` with `FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON`,
  `FileClose`, then `FileMove(..., FILE_COMMON, "GoldStraddle\\state.json", FILE_REWRITE)`.
* **Both readers:** open with `FILE_SHARE_READ|FILE_SHARE_WRITE`, parse, and on any
  parse failure **keep the last good value** and retry next poll. Never act on a
  partial parse. Never crash.

### Cadence

| Actor | Action | Period |
|---|---|---|
| EA | poll `config.json` + `command.json` | `OnTimer`, 250 ms |
| EA | write `state.json` | `OnTimer`, 500 ms (and immediately after any trade action) |
| EA | evaluate level triggers | `OnTick` (never on the timer) |
| Panel | poll `state.json` via `GET /api/state` | 500 ms |
| Panel | write `config.json` | on user edit (debounced 300 ms) |

The EA reloads config only when `config_version` differs from what it last applied.

---

## 2. `config.json` — panel -> EA

```json
{
  "config_version": 17,
  "updated_at": "2026-09-07T14:22:03Z",
  "enabled": false,
  "symbol": "XAUUSD",
  "magic": 770101,
  "unit": "pips",
  "pip_size_mode": "auto",
  "pip_size_manual": 0.10,

  "time_window": {
    "enabled": true,
    "timezone": "server",
    "start": "08:00",
    "end": "17:30",
    "days": [1, 2, 3, 4, 5]
  },

  "entry": {
    "trigger_mode": "cross",
    "touch_tolerance": 0.0,
    "one_shot": true,
    "cooldown_sec": 60,
    "max_slippage_points": 30,
    "max_spread_points": 60,
    "max_concurrent_straddles": 3
  },

  "long":  { "enabled": true, "lots": 0.10, "tp": 50.0, "sl": 0.0,
             "be_enabled": true, "be_trigger": 30.0, "be_offset": 5.0 },
  "short": { "enabled": true, "lots": 0.10, "tp": 50.0, "sl": 0.0,
             "be_enabled": true, "be_trigger": 30.0, "be_offset": 5.0 },

  "levels": [
    { "id": "L1", "label": "PDH", "price": 3548.50, "enabled": true, "direction": "either" }
  ]
}
```

### Field semantics

**Top level**

| Field | Type | Meaning |
|---|---|---|
| `config_version` | int, monotonically increasing | Panel bumps by 1 on every write. EA re-applies only when this changes. |
| `updated_at` | ISO-8601 Z string | Informational; panel clock. |
| `enabled` | bool | **Master arm.** `false` = EA opens nothing. Existing positions are still managed (TP/BE keep working). Never closes anything on its own. |
| `symbol` | string | Trading symbol, e.g. `XAUUSD`. May carry a broker suffix. |
| `magic` | int | Magic number. The EA touches **only** positions with this magic **and** this symbol. |
| `unit` | `"pips"` \| `"points"` \| `"price"` | Unit that **every distance field** below (`tp`, `sl`, `be_trigger`, `be_offset`, `touch_tolerance`) is expressed in. |
| `pip_size_mode` | `"auto"` \| `"manual"` | How to resolve the price value of 1 pip. |
| `pip_size_manual` | double | Used when mode is `manual`. |

**Unit resolution — single source of truth (EA implements, panel mirrors for display)**

```
point = SYMBOL_POINT                       (XAUUSD @ 2 digits -> 0.01)

pip_size (auto):
    if symbol name contains "XAU"      -> 0.10        // gold convention: 1 pip = $0.10
    else if digits == 3 or digits == 5 -> point * 10
    else                               -> point

price_distance(x) =
    unit == "price"  -> x
    unit == "points" -> x * point
    unit == "pips"   -> x * pip_size
```

So with `unit: "pips"` on 2-digit XAUUSD, `tp: 50` means **$5.00** of price move.
The panel MUST display the resolved `$` distance and money value next to every
distance input so this is never ambiguous.

**`time_window`**

| Field | Meaning |
|---|---|
| `enabled` | `false` = no time restriction, levels can fire 24/5. |
| `timezone` | `"server"` = compare against `TimeCurrent()` (broker time). `"local"` = compare against `TimeLocal()` (the PC clock). |
| `start`, `end` | `"HH:MM"`, 24h, inclusive start / exclusive end. |
| `days` | Array of ints, `0`=Sunday .. `6`=Saturday, taken from the window's own timezone. Empty array = nothing passes. |

**Wrap-around is required.** If `start > end` the window spans midnight
(e.g. `"22:00"`-`"06:00"` = 22:00 to 06:00 next morning). Implement as:

```
if start <= end:  open = (mins_start <= mins_now < mins_end)
else:             open = (mins_now >= mins_start) || (mins_now < mins_end)
```

For a wrap-around window the day test uses the day the window **started** on, so a
Friday 22:00-06:00 window stays open into Saturday morning.

The time window gates **entries only**. TP, SL and breakeven management run
unconditionally, 24/5, regardless of the window or of `enabled`.

**`entry`**

| Field | Meaning |
|---|---|
| `trigger_mode` | `"cross"` (default, recommended) or `"touch"`. See section 4. |
| `touch_tolerance` | Distance (in `unit`) added around the level for touch detection. `0` = exact. |
| `one_shot` | `true` = a level fires once then goes `spent` until manually re-armed. `false` = it re-arms automatically after `cooldown_sec`. |
| `cooldown_sec` | Seconds after a fire before the same level may fire again (only when `one_shot` is false). |
| `max_slippage_points` | Deviation passed to the order request, in **points**. |
| `max_spread_points` | Entry is skipped (and logged) if current spread in points exceeds this. `0` = no check. |
| `max_concurrent_straddles` | Max number of distinct levels that may hold open EA positions at once. `0` = unlimited. |

**`long` / `short`** — independent blocks, identical shape.

| Field | Meaning |
|---|---|
| `enabled` | `false` = that leg is not opened. If only one leg is enabled the EA opens a single-sided trade, not a straddle. |
| `lots` | Volume for that leg. EA must clamp to `SYMBOL_VOLUME_MIN/MAX` and round to `SYMBOL_VOLUME_STEP`. |
| `tp` | Take profit distance from **that leg's actual fill price**, in `unit`. `0` = no TP. |
| `sl` | Initial stop loss distance from fill, in `unit`. `0` = no SL. |
| `be_enabled` | Enable breakeven for that leg. |
| `be_trigger` | Once the leg is this far **in profit** (measured on the closing side of the market — bid for long, ask for short), move the stop. |
| `be_offset` | Where the stop goes, as a **signed offset from the fill price in the profitable direction**. Long: `SL = open + price_distance(be_offset)`. Short: `SL = open - price_distance(be_offset)`. `0` = exactly breakeven. Negative values are allowed and mean "still slightly at a loss" (a partial de-risk). |

Breakeven is **one-way and once-only**: it fires at most once per position and must
never move a stop backwards (further from profit) than where it already is.

**`levels`** — array, order preserved, max 32 entries.

| Field | Meaning |
|---|---|
| `id` | Stable unique string, panel-assigned (`L1`, `L2`, ...). Used to key runtime state and tag positions. Must survive edits and reorders. |
| `label` | Free-text note shown in the UI (`PDH`, `Asia low`, ...). ASCII only, <= 24 chars. Never used in logic. |
| `price` | The trigger price. |
| `enabled` | `false` = ignored entirely. |
| `direction` | `"either"` (default), `"from_below"` (fire only when price rises into the level), or `"from_above"` (fire only when price falls into it). |

---

## 3. `state.json` — EA -> panel

Written by the EA every ~500 ms. The panel treats it as **read-only truth**.

```json
{
  "schema": 1,
  "ea_version": "1.0.0",
  "ts_utc": "2026-09-07T14:22:03Z",
  "ts_server": "2026-09-07 16:22:03",
  "ts_local": "2026-09-07 16:22:03",
  "server_gmt_offset_hours": 2.0,
  "heartbeat_ms": 500,

  "config_version_seen": 17,
  "config_error": "",

  "terminal": { "company": "IC Markets (EU) Ltd", "name": "MetaTrader 5",
                "build": 4755, "connected": true,
                "trade_allowed": true, "expert_enabled": true, "dll_allowed": false },

  "account": { "login": 12345678, "server": "<broker-server>",
               "currency": "EUR", "balance": 100000.0, "equity": 100120.5,
               "margin_free": 99000.0, "leverage": 500,
               "margin_mode": "hedging", "hedging_ok": true },

  "symbol": {
    "name": "XAUUSD", "found": true,
    "bid": 3548.42, "ask": 3548.65, "spread_points": 23,
    "digits": 2, "point": 0.01, "pip_size": 0.10,
    "tick_size": 0.01, "tick_value": 1.0, "contract_size": 100.0,
    "volume_min": 0.01, "volume_max": 50.0, "volume_step": 0.01,
    "stops_level_points": 0, "freeze_level_points": 0,
    "trade_mode": "full", "filling": "IOC",
    "candidates": ["XAUUSD", "XAUUSD.a"]
  },

  "window": { "open": true, "reason": "in window",
              "next_change": "2026-09-07 17:30:00", "timezone": "server" },

  "armed": true,
  "blockers": [],

  "levels": [
    { "id": "L1", "label": "PDH", "price": 3548.50, "enabled": true,
      "state": "armed", "arm_side": "below", "distance_price": 0.08,
      "distance_pips": 0.8, "last_fire_utc": "", "fire_count": 0,
      "open_positions": 0, "cooldown_remaining_sec": 0 }
  ],

  "positions": [
    { "ticket": 123456789, "level_id": "L1", "side": "long",
      "lots": 0.10, "open_price": 3548.65, "sl": 0.0, "tp": 3553.65,
      "current": 3548.42, "profit_price": -0.23, "profit_pips": -2.3,
      "profit_money": -2.3, "be_done": false, "open_time": "2026-09-07 16:22:03" }
  ],

  "totals": { "open_positions": 2, "open_lots": 0.20, "floating_money": -4.6,
              "session_closed_money": 0.0, "session_fires": 0 },

  "last_command_id": 4,
  "last_command_result": "close_all: closed 2",
  "log": [
    { "t": "16:22:03", "lvl": "info", "msg": "L1 armed from below" }
  ]
}
```

**Rules**

* `armed` = `config.enabled && window.open && hedging_ok && trade_allowed && !config_error`.
  It is the single boolean the panel shows as the big status light.
* `blockers` = array of short human-readable strings explaining why `armed` is false
  (e.g. `["master switch off", "outside time window"]`). Empty when armed.
* `symbol.candidates` = every Market Watch symbol whose name contains `XAU`, so the
  panel can offer a dropdown when the configured name is not found. Max 12.
* `account.margin_mode` is `"hedging"` / `"netting"` / `"exchange"`.
  `hedging_ok` is `true` only for `"hedging"` — see section 5.
* `level.state` is one of:
  `"disabled"` (config off), `"waiting"` (armed side not yet established / price sitting
  on the level at arm time), `"armed"` (ready to fire), `"cooldown"`, `"spent"`
  (one-shot, already fired), `"live"` (fired and holding open positions).
* `level.arm_side` is `"below"` (price is currently below the level, so the level will be
  approached from below) / `"above"` / `"unknown"`.
* `log` = last 60 entries, newest **last**. `lvl` in `info|warn|error|trade`.
* `profit_money` is in **account currency**.
* Numbers are JSON numbers, never strings. No `NaN`/`Infinity` — emit `0` instead.
* Absent/unknown numeric values are `0`; absent strings are `""`.

---

## 4. Trigger semantics

Evaluated in `OnTick` only, per enabled level, and only while `state.armed` is true.

**`trigger_mode: "cross"` (default and recommended)**

On each tick the EA keeps `prev_bid` (previous tick's bid). A level with price `L` and
tolerance `tol` (in price terms) fires when:

* `arm_side == "below"` and `bid >= L - tol`  (price rose into the level), or
* `arm_side == "above"` and `bid <= L + tol`  (price fell into the level)

and `direction` permits it (`"either"`, or `"from_below"` matching `arm_side=="below"`,
or `"from_above"` matching `arm_side=="above"`).

`arm_side` is established when the level is **armed** (config load, re-arm command, or
cooldown expiry), from the sign of `bid - L`:

* `bid < L - tol` -> `arm_side = "below"`
* `bid > L + tol` -> `arm_side = "above"`
* otherwise       -> `state = "waiting"`, and the level only becomes `armed` once price
  has moved clear of the tolerance band. **This is what prevents an instant fire on a
  level the price is already sitting on** — the single most important safety rule here.

**`trigger_mode: "touch"`** — fires whenever `|bid - L| <= tol`, ignoring `arm_side`.
Still respects `direction` using the sign of `bid - prev_bid`. Offered because some
users want it; `cross` is the default.

**On fire**, in one pass:

1. Re-check `armed`, spread, and `max_concurrent_straddles`.
2. Open the enabled legs at market with `magic`, deviation `max_slippage_points`,
   and comment `GS:<level_id>` (this comment is how positions are attributed to levels,
   so it must be written exactly).
3. Read each leg's **actual fill price** from the deal, then set that leg's TP/SL from
   the fill (not from the level price).
4. Set `fire_count++`, `last_fire_utc`, and `state` to `live`.
5. Append a `trade`-level log line per leg.

If one leg fails to open, the other still stands. Log the failure with the retcode;
do not retry automatically.

---

## 5. Hard safety requirements (EA)

1. **Hedging account check.** Long + short cannot coexist on a netting account — they
   would net to zero. On init and on every config load, if
   `ACCOUNT_MARGIN_MODE != ACCOUNT_MARGIN_MODE_RETAIL_HEDGING`, set `hedging_ok=false`,
   push a blocker, refuse to open any straddle, and log an `error`. The panel must
   show this as a loud red banner.
2. **Stops level.** Clamp every SL/TP to at least `SYMBOL_TRADE_STOPS_LEVEL` points from
   the current market. If a requested TP/SL is inside the broker's minimum distance,
   push it to the minimum and log a `warn`.
3. **Freeze level.** Skip stop modification while price is inside
   `SYMBOL_TRADE_FREEZE_LEVEL` of it; retry next tick.
4. **Filling mode.** Pick from `SYMBOL_FILLING_MODE` — prefer `IOC`, then `FOK`, then
   `RETURN`. IC Markets Raw is ECN and generally wants `IOC`.
5. **Ownership.** Only ever modify or close positions matching **both** `magic` and
   `symbol`. Never touch a manual trade.
6. **Idempotent stop writes.** Before `TRADE_ACTION_SLTP`, compare against the current
   SL/TP normalized to `digits`; skip the call if unchanged. This avoids hammering the
   server with identical modifies every tick.
7. **No auto-close.** The EA never closes a position except on an explicit panel command
   or a broker-side TP/SL hit. `enabled: false` stops new entries only.
8. **Normalize** every price with `NormalizeDouble(p, digits)` before sending.
9. Trade only when `TERMINAL_TRADE_ALLOWED`, `MQLInfoInteger(MQL_TRADE_ALLOWED)` and
   `SYMBOL_TRADE_MODE == FULL`. Otherwise blocker + no entries.
10. **Survive restarts.** On `OnInit`, rebuild runtime state by scanning open positions
    for the magic and parsing `GS:<level_id>` comments, so a terminal restart or chart
    reload does not orphan or double-manage anything.

---

## 6. `command.json` — panel -> EA

One-shot actions. The panel bumps `id` by 1 each time; the EA executes only when
`id != last_command_id`, then publishes `last_command_id` + `last_command_result`
in `state.json`.

```json
{ "id": 5, "cmd": "close_all", "args": {}, "issued_at": "2026-09-07T14:30:00Z" }
```

| `cmd` | `args` | Effect |
|---|---|---|
| `close_all` | `{}` | Market-close every EA position on the symbol. |
| `close_level` | `{"level_id":"L1"}` | Close both legs of that level. |
| `close_side` | `{"side":"long"}` | Close every EA long (or short). |
| `rearm_level` | `{"level_id":"L1"}` | Reset that level to `armed`/`waiting`, clear `spent`. |
| `rearm_all` | `{}` | Re-arm every enabled level. |
| `panic` | `{}` | `close_all` **and** set an internal latch that blocks all new entries until the panel writes `enabled:true` again with a new `config_version`. |

Commands execute regardless of the time window or `enabled`. Closing must always work.

---

## 7. Panel HTTP API (localhost only)

Python stdlib server bound to **`127.0.0.1`** (never `0.0.0.0`), default port **8733**.

| Method | Path | Body / Returns |
|---|---|---|
| `GET` | `/` | The UI (`ui/index.html`). |
| `GET` | `/api/state` | `{"ok":true,"state":<state.json>,"bridge":{...}}`. `state` is `null` if the file is missing. |
| `GET` | `/api/config` | `{"ok":true,"config":<config.json>}` — creates a default config on first run. |
| `POST` | `/api/config` | Body = full config object. Validates, bumps `config_version`, writes atomically. Returns `{"ok":true,"config":<written>,"warnings":[...]}` or `{"ok":false,"errors":[...]}` with HTTP 400. |
| `POST` | `/api/command` | `{"cmd":"close_all","args":{}}` -> assigns `id`, writes atomically, returns `{"ok":true,"id":5}`. |
| `GET` | `/api/health` | `{"ok":true,"ipc_dir":"...","config_exists":true,"state_age_ms":320}` |

`bridge` block in `/api/state`:

```json
{ "state_age_ms": 320, "ea_alive": true, "ipc_dir": "C:\\...\\GoldStraddle",
  "config_version": 17, "config_version_seen": 17, "in_sync": true }
```

* `ea_alive` = `state.json` modified less than **3000 ms** ago.
* `in_sync` = `config_version == config_version_seen`. The UI shows a "pending" pill
  when false — that is how the user knows the EA has actually picked up an edit.
* The server **owns** `config_version`: clients never send it, the server increments.
* Reject any request whose `Host` header is not `127.0.0.1`/`localhost` (DNS-rebinding
  guard), and serve no path outside `ui/`.

---

## 8. Validation rules (panel server enforces, EA re-checks)

| Rule | On violation |
|---|---|
| `lots` within `[0.01, 100]` | reject |
| `tp`, `sl`, `be_trigger` >= 0 | reject |
| `be_offset` in `[-1000, 1000]` | reject |
| `price` > 0 for enabled levels | reject |
| duplicate level `id` | reject |
| `<= 32` levels | reject |
| `start`/`end` match `^\d{2}:\d{2}$`, h<24, m<60 | reject |
| `days` subset of `0..6` | reject |
| `magic` in `[1, 2147483647]` | reject |
| `symbol` matches `^[A-Za-z0-9._#-]{1,32}$` | reject |
| `be_trigger <= tp` when both > 0 | **warn only** (still saved) — a BE trigger beyond TP can never fire |
| `enabled:true` with no enabled levels | **warn only** |

Warnings are returned as `{"ok":true,"warnings":[...]}` and shown in the UI as amber
text; they never block a save.
