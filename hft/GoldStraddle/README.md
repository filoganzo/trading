# GoldStraddle

A level-triggered straddle bot for XAUUSD on IC Markets EU MT5, driven from a
desktop control panel that runs on your PC.

You type in key gold levels. When price reaches one inside your chosen time
window, the EA opens **a long and a short at the same time**, then manages each
leg independently — its own take profit, its own breakeven trigger, its own
breakeven offset.

```
┌──────────────────────────┐        JSON files in          ┌──────────────────────┐
│  Panel (Chrome app win)  │   MT5\Common\Files\           │   GoldStraddle EA    │
│  ──────────────────────  │   GoldStraddle\               │   ──────────────     │
│  levels, TP, BE, window  │  ── config.json  ──────────>  │  watches levels      │
│  live status + P/L       │  <────── state.json  ──────   │  opens long + short  │
│  close / re-arm / panic  │  ── command.json ──────────>  │  TP + breakeven      │
└──────────────────────────┘                               └──────────────────────┘
        server.py                                            on an XAUUSD chart
```

The two halves talk through three JSON files in MT5's **Common Files** folder.
No DLLs, no sockets, no `Allow DLL imports`, nothing to poke a hole in. The
panel can be closed and reopened at any time — the EA keeps trading from the
last config it read.

---

## Layout

| Path | What it is |
|---|---|
| `mt5/GoldStraddle.mq5` | The EA. Executes and manages trades. |
| `panel/server.py` | Local bridge: serves the UI, reads/writes the JSON files. |
| `panel/launch.py` | Starts the server and opens the panel window. |
| `panel/ui/` | The panel itself (HTML/CSS/JS, no dependencies). |
| `Start Panel.bat` | Double-click this to open the panel. |
| `deploy_ea.ps1` | Copies the EA into the terminal and compiles it. |
| `CONTRACT.md` | The panel↔EA interface spec. Read this before changing either side. |

---

## Setup — once

### 1. Install the EA

```powershell
powershell -ExecutionPolicy Bypass -File "deploy_ea.ps1"
```

This copies `mt5/GoldStraddle.mq5` into

```
%APPDATA%\MetaQuotes\Terminal\<TERMINAL_ID>\MQL5\Experts\
```

(that GUID is the IC Markets EU terminal) and compiles it with MetaEditor,
printing the compiler log.

> **This copy step is not optional.** MT5 compiles the file inside its own
> sandbox, not the one in this repo. Editing the repo copy and pressing F7 in
> MetaEditor recompiles the *old* sandbox file, and you end up debugging code
> that isn't running. Always run `deploy_ea.ps1` after any EA edit.

### 2. Check the account is **hedging**

The whole strategy depends on holding a long and a short at once. On a
**netting** account they cancel out and you end up flat — the strategy is
impossible, not just suboptimal.

In MT5: **Tools → Options → Server**, or just look at the panel — it shows the
margin mode and throws up a red banner if the account is netting. IC Markets
MT5 accounts are hedging by default.

### 3. Attach the EA

1. In MT5, open an **XAUUSD** chart (any timeframe — the EA works off ticks).
2. Make sure **AutoTrading** is on (the button in the toolbar must be green).
3. Navigator → Expert Advisors → drag **GoldStraddle** onto the chart.
4. In the dialog, tick **Allow Algo Trading**. DLL imports are *not* needed.

A smiley face in the top-right of the chart means it's live. The chart shows a
small status readout; the panel shows everything.

### 4. Open the panel

Double-click **`Start Panel.bat`**.

It starts the local bridge on `127.0.0.1:8733` and opens the panel in its own
Chrome window with no tabs or address bar. Closing the window shuts the bridge
down too.

The panel and the EA find each other automatically through the Common Files
folder — there is nothing to pair or configure.

---

## Daily use

1. Open the panel. Confirm the header shows **EA live** and a green sync pill.
2. Type your gold levels into the levels table. You can paste several at once
   (one per line, or comma separated) — they all become levels.
3. Set the time window. Pick **server** or **local** time deliberately: IC
   Markets EU server time is GMT+2 in winter, GMT+3 in summer, so it is
   usually 1–2 hours ahead of Italian local time. The panel shows both clocks
   side by side so you can see exactly which one you're setting.
4. Set the long and short parameters. Every distance field shows what it means
   in dollars and in money at your lot size — see the note on pips below.
5. Flip the master switch to **ARM**.

The status light tells you whether entries are actually live. If it's off, the
blocker chips next to it say why — master switch off, outside the window,
netting account, AutoTrading disabled, spread too wide.

### Pips on gold — read this once

Gold quotes to 2 decimals on IC Markets, and "1 pip" conventionally means
**$0.10** of price, i.e. 10 points.

So with the unit set to **pips**:

| You enter | Price move | On 0.10 lots (10 oz) |
|---|---|---|
| `tp: 10` | $1.00 | $10 |
| `tp: 50` | $5.00 | $50 |
| `tp: 100` | $10.00 | $100 |

The panel shows the resolved dollar move and money value next to every field,
live, so you never have to hold this in your head. If you'd rather work in raw
points or in dollars of price directly, switch the **unit** setting.

### How a level fires

A level arms on the side price is currently on, and fires when price comes
*into* it from that side. If you add a level at a price the market is already
sitting on, it goes to **waiting** instead of firing instantly — it only arms
once price moves clear of the tolerance band. That's deliberate: it's what
stops a mistyped level from opening a position the moment you hit Enter.

`direction` narrows this further: `from below` only fires when price rises into
the level, `from above` only when it falls into it.

With `one_shot` on (the default), a level fires once and then shows **spent**
until you re-arm it. With it off, it re-arms itself after the cooldown.

### Breakeven

Each side independently: once the leg is `be_trigger` in profit, the stop moves
to `entry ± be_offset`.

- `be_offset = 0` → exactly breakeven.
- `be_offset = 5` → locks in 5 pips ($0.50 on gold).
- `be_offset = -5` → leaves a small residual loss; de-risks without being
  stopped out by noise sitting right at your entry.

It fires once per position and never moves a stop backwards.

### Closing things

`Close all`, `Close longs`, `Close shorts`, and **PANIC** (closes everything and
latches entries off until you re-arm). All of them need a second confirming
click. They work regardless of the time window or the master switch — closing
is never blocked.

---

## Notes and limits

- **The time window gates entries only.** Take profit, stop loss and breakeven
  run 24/5 on anything already open. Disarming the master switch stops new
  entries; it never closes what you're holding.
- **Windows can span midnight.** `22:00 → 06:00` is read as an overnight
  window, and the panel says "spans midnight" so you can see it was understood
  that way.
- **TP and SL sit at the broker.** They're real server-side orders, so they
  still trigger if MT5 is closed or your PC drops off. Breakeven needs the EA
  running — it's the EA that moves the stop.
- **The EA only touches its own trades**, matched on magic number *and* symbol.
  Your manual positions are never modified or closed.
- **Restarts are safe.** The EA rebuilds its level→position mapping from the
  position comments on init, so a terminal restart or chart reload doesn't
  orphan or double-manage anything.
- The panel binds to `127.0.0.1` only — nothing is exposed to the network.
- One EA instance per symbol. Running two on the same symbol with the same
  magic number will have them fight over the same positions.

## If something's wrong

| Symptom | Cause |
|---|---|
| Panel says **EA not responding** | EA isn't on a chart, or AutoTrading is off. The panel dims live prices rather than showing stale ones. |
| Panel says **bridge offline** | `Start Panel.bat` isn't running. |
| Sync pill stuck on **syncing…** | EA isn't reading config — check it's attached and the terminal isn't frozen. |
| Red **netting account** banner | Straddles are impossible on that account. Nothing will open. |
| Nothing fires despite price hitting a level | Check the blocker chips: window closed, spread over `max_spread_points`, level `spent`, or `max_concurrent_straddles` reached. |
| Edited the EA, behaviour unchanged | You didn't re-run `deploy_ea.ps1`. The terminal is still running the old `.ex5`. |

The panel's log pane mirrors the EA's own log, which is also in the MT5
**Experts** tab. Every entry, stop move and rejection is recorded there with
the broker's return code.
