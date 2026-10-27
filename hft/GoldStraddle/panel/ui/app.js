/* ===========================================================================
   GoldStraddle panel - front end
   Implements CONTRACT.md v1 (sections 2, 3, 6, 7, 8).

   Contents
     01  constants and defaults
     02  safe access, numbers, formatting
     03  tooltip
     04  toasts
     05  confirm helpers (two-click / hold)
     06  store
     07  unit resolution (mirrors CONTRACT section 2)
     08  config binding framework  (focus-safe)
     09  levels editor
     10  long / short parameter cards + inline scale
     11  time window
     12  entry / execution
     13  positions
     14  log
     15  header, banners, meta
     16  validation (mirrors CONTRACT section 8)
     17  network: poll / load / save / command
     18  wiring and boot
   =========================================================================== */
'use strict';

/* ------------------------------------------------------------------ 01 */

var POLL_MS = 500;
var POLL_MAX_MS = 5000;
var POLL_TIMEOUT_MS = 4000;
var SAVE_DEBOUNCE_MS = 300;
var CONFIRM_MS = 3000;
var PANIC_HOLD_MS = 900;
var MAX_LEVELS = 32;
var DASH = '—';

var LEVEL_STATES = ['disabled', 'waiting', 'armed', 'cooldown', 'spent', 'live'];

var STATE_TIPS = {
  waiting:
    'WAITING - price is sitting inside the tolerance band around this level, so the arm side ' +
    'is not established yet. The level arms by itself as soon as price moves clear of the band. ' +
    'This is what prevents an instant fire on a level price is already on.',
  armed: 'ARMED - ready to fire when price moves into the level from the arm side.',
  cooldown: 'COOLDOWN - fired recently. It re-arms automatically when the cooldown expires.',
  spent: 'SPENT - one-shot level that has already fired. Use re-arm to reset it.',
  live: 'LIVE - this level fired and is holding open positions.',
  disabled: 'DISABLED - switched off in the config, ignored entirely.',
  unknown: 'The EA has not reported a state for this level yet.'
};

var DAY_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

var DEFAULT_CONFIG = {
  enabled: false,
  symbol: 'XAUUSD',
  magic: 770101,
  unit: 'pips',
  pip_size_mode: 'auto',
  pip_size_manual: 0.10,
  time_window: { enabled: true, timezone: 'server', start: '08:00', end: '17:30', days: [1, 2, 3, 4, 5] },
  entry: {
    trigger_mode: 'cross', touch_tolerance: 0.0, one_shot: true, cooldown_sec: 600,
    max_slippage_points: 30, max_spread_points: 60, max_concurrent_straddles: 3
  },
  long: { enabled: true, lots: 0.10, tp: 50.0, sl: 0.0, be_enabled: true, be_trigger: 30.0, be_offset: 5.0 },
  short: { enabled: true, lots: 0.10, tp: 50.0, sl: 0.0, be_enabled: true, be_trigger: 30.0, be_offset: 5.0 },
  levels: []
};

/* ------------------------------------------------------------------ 02 */

function $(id) { return document.getElementById(id); }

/** Defensive deep read. State may be absent or partial at any moment. */
function get(obj, path, fallback) {
  if (obj === null || obj === undefined) return fallback;
  var parts = String(path).split('.');
  var cur = obj;
  for (var i = 0; i < parts.length; i++) {
    if (cur === null || cur === undefined) return fallback;
    cur = cur[parts[i]];
  }
  return (cur === undefined || cur === null) ? fallback : cur;
}

function num(v, fallback) {
  var n = (typeof v === 'number') ? v : parseFloat(v);
  return Number.isFinite(n) ? n : fallback;
}

function int(v, fallback) {
  var n = num(v, NaN);
  return Number.isFinite(n) ? Math.round(n) : fallback;
}

function esc(s) {
  return String(s === undefined || s === null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

/** Fixed decimals, never throws, never prints NaN. */
function fmt(v, d) {
  var n = num(v, NaN);
  if (!Number.isFinite(n)) return DASH;
  return n.toFixed(d === undefined ? 2 : d);
}

function fmtSigned(v, d) {
  var n = num(v, NaN);
  if (!Number.isFinite(n)) return DASH;
  return (n > 0 ? '+' : n < 0 ? '−' : '') + Math.abs(n).toFixed(d === undefined ? 2 : d);
}

/** Trailing-zero-free short number, for labels. */
function trimNum(v) {
  var n = num(v, NaN);
  if (!Number.isFinite(n)) return DASH;
  var s = n.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
  return s === '-0' ? '0' : s;
}

function fmtInt(v) {
  var n = num(v, NaN);
  if (!Number.isFinite(n)) return DASH;
  return Math.round(n).toLocaleString('en-US');
}

/** Money in the account currency. Always carries the currency code. */
function fmtMoney(v, signed) {
  var n = num(v, NaN);
  if (!Number.isFinite(n)) return DASH;
  var ccy = String(get(S.state, 'account.currency', '') || '');
  var body = (signed ? (n > 0 ? '+' : n < 0 ? '−' : '') : '') + Math.abs(n).toFixed(2);
  return ccy ? body + ' ' + ccy : body;
}

function fmtPrice(v) {
  var n = num(v, NaN);
  if (!Number.isFinite(n)) return DASH;
  var d = clamp(int(get(S.state, 'symbol.digits', 2), 2), 0, 8);
  return n.toFixed(d);
}

/** "16:22:03" out of "2026-09-07 16:22:03" or an ISO string. */
function timePart(s) {
  var m = /(\d{2}:\d{2}:\d{2})/.exec(String(s || ''));
  if (m) return m[1];
  m = /(\d{2}:\d{2})/.exec(String(s || ''));
  return m ? m[1] : '--:--:--';
}

function gmtLabel(h) {
  var n = num(h, NaN);
  if (!Number.isFinite(n)) return 'GMT?';
  var sign = n < 0 ? '-' : '+';
  var a = Math.abs(n);
  var whole = Math.floor(a);
  var mins = Math.round((a - whole) * 60);
  return 'GMT' + sign + whole + (mins ? ':' + String(mins).padStart(2, '0') : '');
}

function hhmmToMins(s) {
  var m = /^(\d{1,2}):(\d{2})$/.exec(String(s || '').trim());
  if (!m) return null;
  var h = parseInt(m[1], 10), mi = parseInt(m[2], 10);
  if (!(h >= 0 && h < 24 && mi >= 0 && mi < 60)) return null;
  return h * 60 + mi;
}

function normHHMM(s) {
  var mins = hhmmToMins(s);
  if (mins === null) return '';
  return String(Math.floor(mins / 60)).padStart(2, '0') + ':' + String(mins % 60).padStart(2, '0');
}

function localHHMMSS() {
  var d = new Date();
  return String(d.getHours()).padStart(2, '0') + ':' +
    String(d.getMinutes()).padStart(2, '0') + ':' +
    String(d.getSeconds()).padStart(2, '0');
}

function iconSvg(name, cls) {
  return '<svg class="i' + (cls ? ' ' + cls : '') + '" aria-hidden="true"><use href="#i-' + name + '"/></svg>';
}

/** '$' only when the quote currency really is USD. */
function priceSign() {
  var nm = String(get(S.state, 'symbol.name', '') || get(S, 'cfg.symbol', '') || '').toUpperCase();
  return /USD/.test(nm) ? '$' : '';
}

/* ------------------------------------------------------------------ 03 */

var Tip = (function () {
  var el = null, cur = null;

  function show(target) {
    if (!el) el = $('tip');
    if (!el) return;
    var text = target.getAttribute('data-tip');
    if (!text) return;
    cur = target;
    el.textContent = text;
    el.hidden = false;
    var r = target.getBoundingClientRect();
    var tr = el.getBoundingClientRect();
    var x = clamp(r.left + r.width / 2 - tr.width / 2, 8, window.innerWidth - tr.width - 8);
    var y = r.bottom + 7;
    if (y + tr.height > window.innerHeight - 8) y = Math.max(8, r.top - tr.height - 7);
    el.style.left = Math.round(x) + 'px';
    el.style.top = Math.round(y) + 'px';
  }

  function hide() { if (el) { el.hidden = true; } cur = null; }

  function init() {
    el = $('tip');
    document.addEventListener('mouseover', function (e) {
      var t = e.target && e.target.closest ? e.target.closest('[data-tip]') : null;
      if (t && t !== cur) show(t); else if (!t && cur) hide();
    });
    document.addEventListener('mouseout', function (e) {
      var t = e.target && e.target.closest ? e.target.closest('[data-tip]') : null;
      if (t && t === cur) hide();
    });
    document.addEventListener('focusin', function (e) {
      var t = e.target && e.target.closest ? e.target.closest('[data-tip]') : null;
      if (t) show(t); else hide();
    });
    document.addEventListener('focusout', hide);
    window.addEventListener('scroll', hide, true);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') hide(); });
  }

  return { init: init, hide: hide };
})();

/* ------------------------------------------------------------------ 04 */

function toast(kind, msg) {
  var box = $('toasts');
  if (!box) return;
  var icon = kind === 'err' ? 'alert' : kind === 'warn' ? 'alert' : 'check';
  var el = document.createElement('div');
  el.className = 'toast toast-' + (kind === 'err' ? 'err' : kind === 'warn' ? 'warn' : 'ok');
  el.innerHTML = iconSvg(icon) + '<span></span>';
  el.lastChild.textContent = String(msg);
  box.appendChild(el);
  var ttl = kind === 'err' ? 7000 : 4000;
  setTimeout(function () {
    el.classList.add('out');
    setTimeout(function () { if (el.parentNode) el.parentNode.removeChild(el); }, 200);
  }, ttl);
  while (box.children.length > 5) box.removeChild(box.firstChild);
}

/* ------------------------------------------------------------------ 05 */

var Confirm = (function () {
  var active = null;   // { btn, timer, restore }

  function cancel() {
    if (!active) return;
    clearTimeout(active.timer);
    active.restore();
    active = null;
  }

  /**
   * Two-click confirmation. Never fires a destructive command on one click.
   * opts: { label, confirmLabel, iconOnly }
   */
  function attach(btn, opts, onConfirm) {
    if (!btn) return;
    opts = opts || {};
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      if (active && active.btn === btn) {
        cancel();
        onConfirm();
        return;
      }
      cancel();
      var prevHtml = btn.innerHTML;
      var prevAria = btn.getAttribute('aria-label') || '';
      var prevTip = btn.getAttribute('data-tip') || '';
      var confirmText = opts.confirmLabel || btn.getAttribute('data-confirm') ||
        ('Confirm ' + (opts.label || 'action'));
      btn.classList.add('confirming');
      btn.setAttribute('aria-label', confirmText);
      btn.setAttribute('data-tip', confirmText + ' - click again within 3s');
      if (!opts.iconOnly) btn.textContent = confirmText;
      var restore = function () {
        btn.classList.remove('confirming');
        if (!opts.iconOnly) btn.innerHTML = prevHtml;
        if (prevAria) btn.setAttribute('aria-label', prevAria); else btn.removeAttribute('aria-label');
        if (prevTip) btn.setAttribute('data-tip', prevTip); else btn.removeAttribute('data-tip');
        Tip.hide();
      };
      active = { btn: btn, restore: restore, timer: setTimeout(function () { active = null; restore(); }, CONFIRM_MS) };
    });
    btn.addEventListener('blur', function () { if (active && active.btn === btn) cancel(); });
  }

  return { attach: attach, cancel: cancel };
})();

/* ------------------------------------------------------------------ 06 */

var S = {
  cfg: null,
  cfgLoaded: false,
  state: null,
  bridge: {},
  bridgeOk: false,
  failStreak: 0,
  pollTimer: null,

  editSeq: 0,
  savePending: false,
  saveTimer: null,
  saving: false,
  resaveNeeded: false,
  awaitVersion: null,
  syncedAt: 0,
  saveErrors: [],
  saveWarnings: [],
  clientErrors: [],
  clientWarnings: [],
  badFields: {},

  sortMode: 'manual',
  lastBid: NaN,
  lastAsk: NaN,
  expectCmdId: null,
  shownCmdId: null,
  panicLatched: false,
  dismissedBanners: {},

  sig: { banners: '', log: '', levels: '', pos: '', meta: '', sym: '' }
};

/* ------------------------------------------------------------------ 07 */

/**
 * Unit resolution - mirrors the single source of truth in CONTRACT section 2.
 *   pip_size (auto): XAU -> 0.10 ; digits 3 or 5 -> point*10 ; else point
 *   price_distance(x): price -> x ; points -> x*point ; pips -> x*pip_size
 */
function uctx() {
  var digits = clamp(int(get(S.state, 'symbol.digits', 2), 2), 0, 8);
  var point = num(get(S.state, 'symbol.point', 0), 0) || Math.pow(10, -digits);
  var tickSize = num(get(S.state, 'symbol.tick_size', 0), 0) || point;
  var tickValue = num(get(S.state, 'symbol.tick_value', 0), 0);
  var contract = num(get(S.state, 'symbol.contract_size', 0), 0);

  var mode = get(S.cfg, 'pip_size_mode', 'auto');
  var manual = num(get(S.cfg, 'pip_size_manual', 0), 0);
  var pip = num(get(S.state, 'symbol.pip_size', 0), 0);

  if (mode === 'manual') {
    pip = manual > 0 ? manual : pip;
  }
  if (!(pip > 0)) {
    var nm = String(get(S.state, 'symbol.name', '') || get(S.cfg, 'symbol', '') || '').toUpperCase();
    if (nm.indexOf('XAU') >= 0) pip = 0.10;
    else if (digits === 3 || digits === 5) pip = point * 10;
    else pip = point;
  }

  return {
    digits: digits, point: point, pip: pip, tickSize: tickSize,
    tickValue: tickValue, contract: contract,
    unit: get(S.cfg, 'unit', 'pips'),
    live: !!get(S.state, 'symbol.found', false)
  };
}

/** Distance in the configured unit -> price move. */
function priceDist(x, u) {
  var n = num(x, NaN);
  if (!Number.isFinite(n)) return NaN;
  if (u.unit === 'price') return n;
  if (u.unit === 'points') return n * u.point;
  return n * u.pip;
}

/** Price move -> account-currency money for `lots`. */
function moneyOf(priceMove, lots, u) {
  var pm = num(priceMove, NaN), lt = num(lots, NaN);
  if (!Number.isFinite(pm) || !Number.isFinite(lt)) return NaN;
  if (u.tickValue > 0 && u.tickSize > 0) return (pm / u.tickSize) * u.tickValue * lt;
  if (u.contract > 0) return pm * u.contract * lt;
  return NaN;
}

/** Price move -> pips. */
function pipsOf(priceMove, u) {
  var pm = num(priceMove, NaN);
  if (!Number.isFinite(pm) || !(u.pip > 0)) return NaN;
  return pm / u.pip;
}

/**
 * The readout under every distance input: resolved price move, money at the
 * configured lots, and pips when the input is not already in pips.
 */
function distReadout(value, lots, zeroText) {
  var u = uctx();
  var v = num(value, NaN);
  if (!Number.isFinite(v)) return '';
  if (v === 0 && zeroText) return '<span class="warnres">' + esc(zeroText) + '</span>';

  var pm = priceDist(v, u);
  var parts = [];
  parts.push('<span class="m">' + esc(priceSign() + trimNum(Math.abs(pm))) + '</span> price');
  if (u.unit !== 'pips') {
    var p = pipsOf(pm, u);
    if (Number.isFinite(p)) parts.push('<span class="m">' + esc(fmt(Math.abs(p), 1)) + '</span> pips');
  }
  var m = moneyOf(Math.abs(pm), lots, u);
  if (Number.isFinite(m)) parts.push('<span class="m">' + esc(fmtMoney(m, false)) + '</span> @ ' + esc(trimNum(lots)) + ' lots');
  else parts.push('<span class="warnres">money needs the EA symbol spec</span>');
  return '= ' + parts.join(' · ');
}

/* ------------------------------------------------------------------ 08 */
/* Focus-safe config binding framework.
 *
 * The DOM is written only from S.cfg. S.cfg is written by the user (directly,
 * so the DOM is already right) or by a server response. A server response is
 * adopted ONLY if the user has not typed since the request went out
 * (S.editSeq unchanged), and even then refresh() never writes into the element
 * that currently has focus. That closes both clobber paths.
 */

var bindings = [];

function readCtl(el, kind, opts) {
  if (kind === 'check') return el.checked;
  if (kind === 'number') {
    var raw = el.value.trim();
    if (raw === '') return undefined;              // mid-edit: leave cfg alone
    var n = Number(raw);
    if (!Number.isFinite(n)) return undefined;     // "-", ".", "1e" ...
    return opts && opts.int ? Math.round(n) : n;
  }
  if (kind === 'time') {
    var t = normHHMM(el.value);
    return t === '' ? undefined : t;
  }
  return el.value;
}

function writeCtl(el, kind, v) {
  if (kind === 'check') { el.checked = !!v; return; }
  if (kind === 'number') {
    var n = num(v, NaN);
    el.value = Number.isFinite(n) ? String(n) : '';
    return;
  }
  if (kind === 'time') { el.value = normHHMM(v); return; }
  el.value = (v === undefined || v === null) ? '' : String(v);
}

function bind(el, kind, getter, setter, opts) {
  if (!el) return null;
  opts = opts || {};
  var b = { el: el, kind: kind, get: getter, set: setter, opts: opts };
  bindings.push(b);

  if (kind === 'check') {
    el.addEventListener('change', function () { setter(el.checked); onEdit(opts.immediate); });
  } else {
    el.addEventListener('input', function () {
      var v = readCtl(el, kind, opts);
      if (v === undefined) { touch(); renderDerived(); return; }
      setter(v);
      onEdit(false);
    });
    el.addEventListener('focus', function () { el.dataset.orig = el.value; });
    el.addEventListener('blur', function () { refreshOne(b); });
    el.addEventListener('keydown', function (e) { fieldKeys(e, el, kind, opts); });
  }
  return b;
}

function refreshOne(b) {
  if (!b || !b.el) return;
  if (b.el === document.activeElement) return;   // never clobber the field in use
  if (!b.el.isConnected) return;
  writeCtl(b.el, b.kind, b.get());
}

function refreshAll() {
  for (var i = 0; i < bindings.length; i++) refreshOne(bindings[i]);
  renderDerived();
}

function dropBindings(pred) {
  bindings = bindings.filter(function (b) { return !pred(b); });
}

/** Enter commits, Escape reverts, Shift+Arrow steps by 10x. */
function fieldKeys(e, el, kind, opts) {
  if (e.key === 'Enter') {
    e.preventDefault();
    var v = readCtl(el, kind, opts);
    if (v !== undefined) { /* already written by input handler */ }
    flushSave();
    el.dataset.orig = el.value;
    return;
  }
  if (e.key === 'Escape') {
    e.preventDefault();
    e.stopPropagation();
    var orig = el.dataset.orig;
    if (orig !== undefined && orig !== el.value) {
      el.value = orig;
      var rv = readCtl(el, kind, opts);
      if (rv !== undefined) { bindingFor(el, function (b) { b.set(rv); }); onEdit(false); }
    }
    return;
  }
  if (kind === 'number' && e.shiftKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
    e.preventDefault();
    var step = num(el.step, 1) || 1;
    var cur = num(el.value, 0);
    var next = cur + (e.key === 'ArrowUp' ? 1 : -1) * step * 10;
    var lo = num(el.min, NaN), hi = num(el.max, NaN);
    if (Number.isFinite(lo)) next = Math.max(lo, next);
    if (Number.isFinite(hi)) next = Math.min(hi, next);
    el.value = String(Math.round(next * 1e6) / 1e6);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }
}

function bindingFor(el, fn) {
  for (var i = 0; i < bindings.length; i++) if (bindings[i].el === el) { fn(bindings[i]); return; }
}

/* --- path helpers for the static (non-level) fields ------------------- */

function cfgGet(path, fb) { return get(S.cfg, path, fb); }

function cfgSet(path, v) {
  if (!S.cfg) return;
  var parts = String(path).split('.');
  var cur = S.cfg;
  for (var i = 0; i < parts.length - 1; i++) {
    if (typeof cur[parts[i]] !== 'object' || cur[parts[i]] === null) cur[parts[i]] = {};
    cur = cur[parts[i]];
  }
  cur[parts[parts.length - 1]] = v;
}

function bindPath(id, kind, path, opts) {
  var el = $(id);
  if (!el) return null;
  return bind(el, kind,
    function () { return cfgGet(path, undefined); },
    function (v) { cfgSet(path, v); },
    opts);
}

/* ------------------------------------------------------------------ 09 */
/* Levels editor */

var rows = new Map();   // level id -> { tr, refs }

function levels() {
  if (!S.cfg) return [];
  if (!Array.isArray(S.cfg.levels)) S.cfg.levels = [];
  var base = S.cfg.levels.slice();
  var live = get(S.state, 'levels', []);
  if (Array.isArray(live)) live.forEach(function (l) {
    if ((l.level_type === 'D1' || l.level_type === 'H4') && !base.some(function (x) { return x.id === l.id; })) {
      base.push({ id: l.id, label: l.label || (l.level_type + ' validated'), price: l.price, enabled: l.enabled !== false, direction: 'either', level_type: l.level_type, automatic: true });
    }
  });
  return base;
}

function findLevel(id) {
  var ls = levels();
  for (var i = 0; i < ls.length; i++) if (ls[i] && ls[i].id === id) return ls[i];
  return null;
}

function stateForLevel(id) {
  var arr = get(S.state, 'levels', []);
  if (!Array.isArray(arr)) return null;
  for (var i = 0; i < arr.length; i++) if (get(arr[i], 'id', null) === id) return arr[i];
  return null;
}

function nextLevelId() {
  var maxN = 0;
  var ls = levels();
  for (var i = 0; i < ls.length; i++) {
    var m = /^L(\d+)$/.exec(String(get(ls[i], 'id', '')));
    if (m) maxN = Math.max(maxN, parseInt(m[1], 10));
  }
  var n = maxN + 1;
  var taken = {};
  for (var j = 0; j < ls.length; j++) taken[get(ls[j], 'id', '')] = true;
  while (taken['L' + n]) n++;
  return 'L' + n;
}

function addLevels(prices, labelHint) {
  var ls = levels();
  var room = MAX_LEVELS - ls.length;
  if (room <= 0) { toast('warn', 'Level limit reached (' + MAX_LEVELS + '). Delete one first.'); return 0; }
  var added = 0;
  for (var i = 0; i < prices.length && added < room; i++) {
    var p = num(prices[i], NaN);
    if (!Number.isFinite(p) || p <= 0) continue;
    ls.push({
      id: nextLevelId(),
      label: labelHint && prices.length === 1 ? String(labelHint).slice(0, 24) : '',
      price: p,
      enabled: true,
      direction: 'either'
    });
    added++;
  }
  if (added) {
    onEdit(true);
    syncLevels(true);
  }
  if (prices.length > added) {
    toast('warn', 'Added ' + added + ' of ' + prices.length + ' - the rest did not fit or were not valid prices.');
  } else if (added > 1) {
    toast('ok', 'Added ' + added + ' levels.');
  }
  return added;
}

function parsePrices(text) {
  var out = [];
  var toks = String(text || '').split(/[\s,;|]+/);
  for (var i = 0; i < toks.length; i++) {
    var t = toks[i].replace(/[^0-9.\-]/g, '');
    if (!t) continue;
    var n = Number(t);
    if (Number.isFinite(n) && n > 0) out.push(n);
  }
  return out;
}

function deleteLevel(id) {
  var ls = levels();
  for (var i = 0; i < ls.length; i++) {
    if (ls[i] && ls[i].id === id) { ls.splice(i, 1); break; }
  }
  var entry = rows.get(id);
  if (entry) {
    dropBindings(function (b) { return entry.tr.contains(b.el); });
    if (entry.tr.parentNode) entry.tr.parentNode.removeChild(entry.tr);
    rows.delete(id);
  }
  onEdit(true);
  syncLevels(true);
}

function moveLevel(id, delta) {
  var ls = levels();
  var i = -1;
  for (var k = 0; k < ls.length; k++) if (ls[k] && ls[k].id === id) { i = k; break; }
  if (i < 0) return;
  var j = i + delta;
  if (j < 0 || j >= ls.length) return;
  var tmp = ls[i]; ls[i] = ls[j]; ls[j] = tmp;
  onEdit(true);
  syncLevels(true);
}

function moveLevelTo(id, targetId, after) {
  var ls = levels();
  var from = -1, to = -1;
  for (var k = 0; k < ls.length; k++) {
    if (ls[k] && ls[k].id === id) from = k;
    if (ls[k] && ls[k].id === targetId) to = k;
  }
  if (from < 0 || to < 0 || from === to) return;
  var item = ls.splice(from, 1)[0];
  if (from < to) to--;
  ls.splice(after ? to + 1 : to, 0, item);
  onEdit(true);
  syncLevels(true);
}

function buildLevelRow(id) {
  var tr = document.createElement('tr');
  tr.className = 'lvl';
  tr.dataset.id = id;
  tr.innerHTML =
    '<td class="c-ord">' +
      '<div class="ordcell">' +
        '<span class="grip" data-role="grip" aria-hidden="true">' + iconSvg('grip') + '</span>' +
        '<span class="ordbtns">' +
          '<button type="button" class="ibtn" data-role="up" aria-label="Move level up">' + iconSvg('up') + '</button>' +
          '<button type="button" class="ibtn" data-role="dn" aria-label="Move level down">' + iconSvg('down') + '</button>' +
        '</span>' +
      '</div>' +
      '<span class="lvl-id" data-role="id"></span>' +
    '</td>' +
    '<td class="c-on"><span class="cell-chk">' +
      '<input type="checkbox" class="cb" data-role="on">' +
    '</span></td>' +
    '<td class="c-price"><input type="number" class="mono" data-role="price" step="0.1" min="0" inputmode="decimal"></td>' +
    '<td class="c-label"><input type="text" data-role="label" maxlength="24" autocomplete="off" spellcheck="false" placeholder="note"></td>' +
    '<td class="c-dir"><select data-role="dir">' +
      '<option value="either">either</option>' +
      '<option value="from_below">from below</option>' +
      '<option value="from_above">from above</option>' +
    '</select></td>' +
    '<td class="c-dist num"><span class="dist" data-role="dist"></span></td>' +
    '<td class="c-state"><span class="state-cell">' +
      '<span class="badge badge-unknown" data-role="badge">' + DASH + '</span>' +
      '<span class="state-sub" data-role="statesub"></span>' +
    '</span></td>' +
    '<td class="c-act"><span class="acts">' +
      '<button type="button" class="ibtn" data-role="rearm" aria-label="Re-arm this level" data-tip="rearm_level: reset this level to armed / waiting and clear spent.">' + iconSvg('rearm') + '</button>' +
      '<button type="button" class="ibtn ibtn-danger" data-role="closelvl" aria-label="Close this level’s positions" data-tip="close_level: market-close both legs opened by this level. Needs a second click.">' + iconSvg('close') + '</button>' +
      '<button type="button" class="ibtn ibtn-danger" data-role="del" aria-label="Delete this level" data-tip="Remove this level from the config. Needs a second click.">' + iconSvg('trash') + '</button>' +
    '</span></td>';

  var q = function (role) { return tr.querySelector('[data-role="' + role + '"]'); };
  var refs = {
    id: q('id'), on: q('on'), price: q('price'), label: q('label'), dir: q('dir'),
    dist: q('dist'), badge: q('badge'), statesub: q('statesub'),
    up: q('up'), dn: q('dn'), grip: q('grip')
  };
  refs.id.textContent = id;
  refs.on.setAttribute('aria-label', 'Level ' + id + ' enabled');
  refs.price.setAttribute('aria-label', 'Level ' + id + ' price');
  refs.label.setAttribute('aria-label', 'Level ' + id + ' label');
  refs.dir.setAttribute('aria-label', 'Level ' + id + ' direction');

  bind(refs.on, 'check',
    function () { return get(findLevel(id), 'enabled', false); },
    function (v) { var l = findLevel(id); if (l) l.enabled = !!v; },
    { immediate: true });

  bind(refs.price, 'number',
    function () { return get(findLevel(id), 'price', undefined); },
    function (v) { var l = findLevel(id); if (l) l.price = v; });

  bind(refs.label, 'text',
    function () { return get(findLevel(id), 'label', ''); },
    function (v) { var l = findLevel(id); if (l) l.label = String(v).slice(0, 24); });

  bind(refs.dir, 'text',
    function () { return get(findLevel(id), 'direction', 'either'); },
    function (v) { var l = findLevel(id); if (l) l.direction = v; });

  refs.up.addEventListener('click', function () { moveLevel(id, -1); });
  refs.dn.addEventListener('click', function () { moveLevel(id, 1); });

  Confirm.attach(q('del'), { iconOnly: true, confirmLabel: 'Delete level ' + id },
    function () { deleteLevel(id); });
  Confirm.attach(q('closelvl'), { iconOnly: true, confirmLabel: 'Close positions of ' + id },
    function () { command('close_level', { level_id: id }, 'Close ' + id); });
  q('rearm').addEventListener('click', function () { command('rearm_level', { level_id: id }, 'Re-arm ' + id); });

  /* drag reorder - draggable is only switched on from the grip so text
     selection inside the row inputs keeps working */
  refs.grip.addEventListener('mousedown', function () { tr.draggable = true; });
  refs.grip.addEventListener('mouseup', function () { tr.draggable = false; });
  tr.addEventListener('dragstart', function (e) {
    if (S.sortMode !== 'manual') { e.preventDefault(); return; }
    e.dataTransfer.setData('text/plain', id);
    e.dataTransfer.effectAllowed = 'move';
    tr.classList.add('dragging');
  });
  tr.addEventListener('dragend', function () {
    tr.draggable = false;
    tr.classList.remove('dragging');
    clearDropMarks();
  });
  tr.addEventListener('dragover', function (e) {
    if (S.sortMode !== 'manual') return;
    e.preventDefault();
    var r = tr.getBoundingClientRect();
    var after = (e.clientY - r.top) > r.height / 2;
    clearDropMarks();
    tr.classList.add(after ? 'dropafter' : 'dropbefore');
  });
  tr.addEventListener('drop', function (e) {
    e.preventDefault();
    var src = e.dataTransfer.getData('text/plain');
    var r = tr.getBoundingClientRect();
    var after = (e.clientY - r.top) > r.height / 2;
    clearDropMarks();
    if (src && src !== id) moveLevelTo(src, id, after);
  });

  rows.set(id, { tr: tr, refs: refs });
  return tr;
}

function clearDropMarks() {
  var marked = document.querySelectorAll('tr.dropbefore, tr.dropafter');
  for (var i = 0; i < marked.length; i++) marked[i].classList.remove('dropbefore', 'dropafter');
}

/** Display order: config order, or price-sorted with a current-price marker. */
function displayOrder() {
  var ls = levels().slice();
  if (S.sortMode === 'price') {
    ls.sort(function (a, b) { return num(get(b, 'price', 0), 0) - num(get(a, 'price', 0), 0); });
  }
  return ls;
}

function syncLevels(force) {
  var body = $('levelsBody');
  if (!body || !S.cfg) return;

  var order = displayOrder();
  var bid = num(get(S.state, 'symbol.bid', NaN), NaN);
  var showMark = S.sortMode === 'price' && Number.isFinite(bid) && bid > 0 && order.length > 0;
  var sig = order.map(function (l) { return get(l, 'id', '?'); }).join('|') + '#' + S.sortMode + '#' + (showMark ? '1' : '0');

  if (force || sig !== S.sig.levels) {
    S.sig.levels = sig;

    /* drop rows for levels that no longer exist */
    var alive = {};
    for (var i = 0; i < order.length; i++) alive[get(order[i], 'id', '?')] = true;
    rows.forEach(function (entry, id) {
      if (!alive[id]) {
        dropBindings(function (b) { return entry.tr.contains(b.el); });
        if (entry.tr.parentNode) entry.tr.parentNode.removeChild(entry.tr);
        rows.delete(id);
      }
    });

    /* rebuild the running order */
    var frag = document.createDocumentFragment();
    var markPlaced = false;
    for (var k = 0; k < order.length; k++) {
      var id = get(order[k], 'id', null);
      if (!id) continue;
      if (showMark && !markPlaced && num(get(order[k], 'price', 0), 0) < bid) {
        frag.appendChild(priceMarkRow(bid));
        markPlaced = true;
      }
      var entry = rows.get(id) || { tr: buildLevelRow(id) };
      frag.appendChild(entry.tr);
    }
    if (showMark && !markPlaced) frag.appendChild(priceMarkRow(bid));
    body.textContent = '';
    body.appendChild(frag);
  } else if (showMark) {
    var pm = body.querySelector('.pricemark .mono');
    if (pm) pm.textContent = fmtPrice(bid);
  }

  /* live values + focus-safe input refresh */
  for (var n = 0; n < order.length; n++) updateLevelRow(order[n]);

  var count = levels().length;
  var cEl = $('levelsCount');
  if (cEl) cEl.textContent = count + '/' + MAX_LEVELS;
  var empty = $('levelsEmpty');
  if (empty) empty.hidden = count > 0;
}

function priceMarkRow(bid) {
  var tr = document.createElement('tr');
  tr.className = 'pricemark';
  tr.innerHTML = '<td colspan="8"><span class="pricemark-in">' +
    '<span>current bid</span><span class="mono">' + esc(fmtPrice(bid)) + '</span>' +
    '<span>levels below</span></span></td>';
  return tr;
}

function updateLevelRow(lvl) {
  var id = get(lvl, 'id', null);
  var entry = id ? rows.get(id) : null;
  if (!entry) return;
  var refs = entry.refs, tr = entry.tr;
  var u = uctx();
  var st = stateForLevel(id);

  /* inputs (skips whatever has focus) */
  refs.price.step = String(Math.max(u.point * 10, 0.0001));
  bindingFor(refs.price, refreshOne);
  bindingFor(refs.label, refreshOne);
  bindingFor(refs.dir, refreshOne);
  bindingFor(refs.on, refreshOne);

  var enabled = !!get(lvl, 'enabled', false);
  tr.classList.toggle('off', !enabled);

  /* distance from price */
  var bid = num(get(S.state, 'symbol.bid', NaN), NaN);
  var price = num(get(lvl, 'price', NaN), NaN);
  var dPrice, dPips, known = false;
  if (Number.isFinite(bid) && bid > 0 && Number.isFinite(price) && price > 0) {
    dPrice = price - bid;
    dPips = pipsOf(dPrice, u);
    known = true;
  } else if (st) {
    dPrice = num(get(st, 'distance_price', NaN), NaN);
    dPips = num(get(st, 'distance_pips', NaN), NaN);
    known = Number.isFinite(dPrice);
  }

  var above = known && dPrice > 0;
  tr.classList.toggle('above', known && above && enabled);
  tr.classList.toggle('below', known && !above && enabled);

  if (!known) {
    refs.dist.className = 'dist';
    refs.dist.innerHTML = '<span class="dist-main">' + DASH + '</span><span class="dist-sub">no price</span>';
  } else {
    refs.dist.className = 'dist ' + (above ? 'up' : 'down');
    refs.dist.innerHTML =
      '<span class="dist-main">' + iconSvg(above ? 'caret-up' : 'caret-down') +
      esc(fmtSigned(dPips, 1)) + ' p</span>' +
      '<span class="dist-sub">' + esc(priceSign() + fmt(Math.abs(dPrice), Math.min(u.digits, 2))) + '</span>';
    refs.dist.setAttribute('data-tip', above
      ? 'This level is above the current bid: price has to rise ' + fmt(Math.abs(dPips), 1) + ' pips to reach it.'
      : 'This level is below the current bid: price has to fall ' + fmt(Math.abs(dPips), 1) + ' pips to reach it.');
  }

  /* state badge */
  var sname = String(get(st, 'state', '') || '');
  if (!enabled) sname = 'disabled';
  var known2 = LEVEL_STATES.indexOf(sname) >= 0;
  refs.badge.className = 'badge badge-' + (known2 ? sname : 'unknown');
  refs.badge.textContent = known2 ? sname : (S.state ? 'unknown' : 'no EA');

  var armSide = String(get(st, 'arm_side', '') || 'unknown');
  var tipBase = STATE_TIPS[known2 ? sname : 'unknown'] || STATE_TIPS.unknown;
  if (sname === 'armed' && (armSide === 'below' || armSide === 'above')) {
    tipBase = 'ARMED - arm side is ' + armSide + ', so it fires when price moves into the level from ' + armSide + '.';
  }
  refs.badge.setAttribute('data-tip', tipBase);

  var sub = [];
  var cd = num(get(st, 'cooldown_remaining_sec', 0), 0);
  if (sname === 'cooldown' && cd > 0) sub.push(Math.round(cd) + 's');
  var op = int(get(st, 'open_positions', 0), 0);
  if (op > 0) sub.push(op + ' pos');
  var fc = int(get(st, 'fire_count', 0), 0);
  if (fc > 0) sub.push(fc + 'x');
  if (armSide === 'below' || armSide === 'above') sub.push('arm ' + armSide);
  refs.statesub.textContent = sub.join(' · ');
  tr.classList.toggle('livepos', op > 0);
}

/* ------------------------------------------------------------------ 10 */
/* Long / short parameter cards */

function renderSide(side) {
  var p = get(S.cfg, side, {}) || {};
  var u = uctx();
  var lots = num(p.lots, 0);

  /* unit chips */
  var chips = document.querySelectorAll('#' + side + 'Card [data-unit]');
  for (var i = 0; i < chips.length; i++) chips[i].textContent = u.unit;

  /* lots readout */
  var lotsRes = $(side + '_lots_res');
  if (lotsRes) {
    var vmin = num(get(S.state, 'symbol.volume_min', NaN), NaN);
    var vmax = num(get(S.state, 'symbol.volume_max', NaN), NaN);
    var vstep = num(get(S.state, 'symbol.volume_step', NaN), NaN);
    var lotEl = $(side + '_lots');
    if (lotEl) {
      if (Number.isFinite(vstep) && vstep > 0) lotEl.step = String(vstep);
      if (Number.isFinite(vmin) && vmin > 0) lotEl.min = String(vmin);
      if (Number.isFinite(vmax) && vmax > 0) lotEl.max = String(Math.min(vmax, 100));
    }
    var oneUnit = moneyOf(u.pip, lots, u);
    var bits = [];
    if (Number.isFinite(oneUnit)) bits.push('1 pip = <span class="m">' + esc(fmtMoney(oneUnit, false)) + '</span>');
    if (Number.isFinite(vmin) && Number.isFinite(vmax) && vmax > 0) {
      bits.push('broker allows ' + esc(trimNum(vmin)) + ' – ' + esc(trimNum(vmax)) +
        (Number.isFinite(vstep) && vstep > 0 ? ' step ' + esc(trimNum(vstep)) : ''));
    }
    lotsRes.innerHTML = bits.length ? bits.join(' · ') : '<span class="warnres">waiting for the EA symbol spec</span>';
  }

  setHTML($(side + '_tp_res'), distReadout(p.tp, lots, 'no take profit'));
  setHTML($(side + '_sl_res'), distReadout(p.sl, lots, 'no stop loss - this leg runs unprotected'));
  setHTML($(side + '_be_trigger_res'), distReadout(p.be_trigger, lots, 'breakeven can never trigger at 0'));

  /* be_offset - signed, explain the sign explicitly */
  var boRes = $(side + '_be_offset_res');
  var bo = num(p.be_offset, NaN);
  if (boRes) {
    if (!Number.isFinite(bo)) setHTML(boRes, '');
    else if (bo === 0) setHTML(boRes, '<span class="m">exact breakeven</span> - stop lands on the fill price');
    else setHTML(boRes, distReadout(Math.abs(bo), lots, ''));
  }

  var expl = $(side + '_be_explain');
  if (expl) {
    if (!get(p, 'be_enabled', false)) {
      expl.innerHTML = 'Breakeven off: the initial stop stays where it was placed.';
    } else if (!Number.isFinite(bo)) {
      expl.innerHTML = '';
    } else {
      var pm = Math.abs(priceDist(bo, u));
      var money = moneyOf(pm, lots, u);
      var moneyTxt = Number.isFinite(money) ? ' (' + fmtMoney(money, false) + ')' : '';
      if (bo > 0) {
        expl.innerHTML = 'Offset is <b>positive</b>: once ' + esc(trimNum(p.be_trigger)) + ' ' + esc(u.unit) +
          ' in profit the stop moves to <b>lock in ' + esc(trimNum(bo)) + ' ' + esc(u.unit) + '</b>' + esc(moneyTxt) + ' of profit.';
      } else if (bo === 0) {
        expl.innerHTML = 'Offset is <b>zero</b>: the stop moves to <b>exact breakeven</b> - the fill price, no profit locked.';
      } else {
        expl.innerHTML = 'Offset is <b>negative</b>: the stop moves close to entry but still <b>' +
          esc(trimNum(Math.abs(bo))) + ' ' + esc(u.unit) + '</b>' + esc(moneyTxt) +
          ' at a loss - a partial de-risk rather than a full breakeven.';
      }
    }
  }

  /* step sizes follow the unit */
  var distStep = u.unit === 'price' ? 0.1 : 1;
  ['tp', 'sl', 'be_trigger', 'be_offset'].forEach(function (f) {
    var el = $(side + '_' + f);
    if (el) el.step = String(distStep);
  });

  drawScale($(side + '_scale'), side);
  renderPreview(side);
}

function setHTML(el, html) { if (el) el.innerHTML = html; }

/** "if a leg fired right now" - resolves distances into real prices. */
function renderPreview(side) {
  var el = $(side + '_preview');
  if (!el) return;
  var p = get(S.cfg, side, {}) || {};
  var u = uctx();
  var isLong = side === 'long';
  var fill = num(get(S.state, 'symbol.' + (isLong ? 'ask' : 'bid'), NaN), NaN);

  if (!Number.isFinite(fill) || fill <= 0) {
    el.innerHTML = '<span class="warnres">No live price - resolved TP / SL prices appear once the EA is running.</span>';
    return;
  }
  if (!get(p, 'enabled', false)) {
    el.innerHTML = 'Leg disabled - nothing would open.';
    return;
  }

  var sgn = isLong ? 1 : -1;
  var tp = num(p.tp, 0), sl = num(p.sl, 0);
  var out = ['If it fired now at <span class="m">' + esc(fmtPrice(fill)) + '</span>:'];
  out.push('TP <span class="m">' + (tp > 0 ? esc(fmtPrice(fill + sgn * priceDist(tp, u))) : 'none') + '</span>');
  out.push('SL <span class="m">' + (sl > 0 ? esc(fmtPrice(fill - sgn * priceDist(sl, u))) : 'none') + '</span>');
  if (get(p, 'be_enabled', false)) {
    var beStop = fill + sgn * priceDist(num(p.be_offset, 0), u);
    var beAt = fill + sgn * priceDist(num(p.be_trigger, 0), u);
    out.push('BE stop <span class="m">' + esc(fmtPrice(beStop)) + '</span> once price reaches <span class="m">' + esc(fmtPrice(beAt)) + '</span>');
  }
  var stops = num(get(S.state, 'symbol.stops_level_points', 0), 0);
  if (stops > 0) {
    out.push('broker minimum stop distance <span class="m">' + esc(fmt(stops, 0)) + ' pts</span>');
  }
  el.innerHTML = out.join(' · ');
}

/**
 * Inline scale: entry -> BE trigger -> BE stop -> TP on the profit axis.
 * The short card is mirrored (profit runs left) because a short gains as
 * price falls.
 */
function drawScale(svg, side) {
  if (!svg) return;
  var p = get(S.cfg, side, {}) || {};
  var u = uctx();
  var tp = num(p.tp, 0), sl = num(p.sl, 0);
  var beOn = !!get(p, 'be_enabled', false);
  var bt = num(p.be_trigger, 0), bo = num(p.be_offset, 0);

  if (!(tp > 0) && !(sl > 0) && !beOn) {
    svg.innerHTML = '<text class="sc-none" x="200" y="50" text-anchor="middle" font-size="10">' +
      'no TP, no SL, no breakeven - this leg would run unmanaged</text>';
    return;
  }

  var marks = [{ v: 0, cls: 'entry', label: 'ENTRY', row: 'b1' }];
  if (tp > 0) marks.push({ v: tp, cls: 'tp', label: 'TP ' + trimNum(tp), row: 'a1' });
  if (sl > 0) marks.push({ v: -sl, cls: 'sl', label: 'SL ' + trimNum(sl), row: 'a1' });
  if (beOn) {
    if (bt > 0) marks.push({ v: bt, cls: 'be', label: 'BE trig ' + trimNum(bt), row: 'a2' });
    if (Number.isFinite(bo)) marks.push({ v: bo, cls: 'be', label: 'BE stop ' + trimNum(bo), row: 'b2' });
  }

  var vals = marks.map(function (m) { return m.v; });
  var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
  var span = (hi - lo) || 1;
  lo -= span * 0.16; hi += span * 0.16;

  var X0 = 20, X1 = 380, Y = 52;
  var flip = side === 'short';
  var X = function (v) {
    var t = (v - lo) / (hi - lo);
    return flip ? (X1 - (X1 - X0) * t) : (X0 + (X1 - X0) * t);
  };
  var rowY = { a2: 22, a1: 38, b1: 70, b2: 85 };

  var parts = [];
  var x0 = X(0);

  /* profit / loss bands */
  var xhi = X(hi), xlo = X(lo);
  parts.push('<rect class="sc-band-p" x="' + fmt(Math.min(x0, xhi), 1) + '" y="' + (Y - 7) +
    '" width="' + fmt(Math.abs(xhi - x0), 1) + '" height="14" />');
  if (lo < 0) {
    parts.push('<rect class="sc-band-l" x="' + fmt(Math.min(x0, xlo), 1) + '" y="' + (Y - 7) +
      '" width="' + fmt(Math.abs(x0 - xlo), 1) + '" height="14" />');
  }

  parts.push('<line class="sc-axis" x1="' + X0 + '" y1="' + Y + '" x2="' + X1 + '" y2="' + Y + '" />');

  /* profit direction arrow at the profit end */
  var ax = flip ? X0 + 2 : X1 - 2;
  var back = flip ? 12 : -12;
  parts.push('<path class="sc-dir" d="M' + fmt(ax + back, 1) + ' ' + (Y - 4) + 'L' + fmt(ax, 1) + ' ' + Y +
    'L' + fmt(ax + back, 1) + ' ' + (Y + 4) + '" />');
  parts.push('<text class="sc-dir-lab" x="' + fmt(flip ? X0 + 16 : X1 - 16, 1) + '" y="' + (Y - 12) +
    '" text-anchor="' + (flip ? 'start' : 'end') + '" font-size="8">profit ' +
    (side === 'long' ? 'as price rises' : 'as price falls') + '</text>');

  for (var i = 0; i < marks.length; i++) {
    var m = marks[i];
    var x = X(m.v);
    var isEntry = m.cls === 'entry';
    parts.push('<line class="sc-tick sc-' + m.cls + '" x1="' + fmt(x, 1) + '" y1="' + (Y - (isEntry ? 13 : 9)) +
      '" x2="' + fmt(x, 1) + '" y2="' + (Y + (isEntry ? 13 : 9)) + '" />');
    var lx = clamp(x, 30, 370);
    parts.push('<text class="sc-lab sc-lab-' + m.cls + '" x="' + fmt(lx, 1) + '" y="' + rowY[m.row] +
      '" text-anchor="middle" font-size="9">' + esc(m.label) + '</text>');
  }

  svg.innerHTML = parts.join('');
  svg.setAttribute('aria-label',
    (side === 'long' ? 'Long' : 'Short') + ' leg scale. Entry at fill price. ' +
    (tp > 0 ? 'Take profit ' + trimNum(tp) + ' ' + u.unit + '. ' : 'No take profit. ') +
    (sl > 0 ? 'Stop loss ' + trimNum(sl) + ' ' + u.unit + '. ' : 'No stop loss. ') +
    (beOn ? 'Breakeven triggers at ' + trimNum(bt) + ' and moves the stop to ' + trimNum(bo) + '.' : 'Breakeven off.'));
}

/* ------------------------------------------------------------------ 11 */
/* Time window */

function renderWindow() {
  var tw = get(S.cfg, 'time_window', {}) || {};
  var startM = hhmmToMins(tw.start);
  var endM = hhmmToMins(tw.end);
  var wraps = (startM !== null && endM !== null && startM > endM);

  var wn = $('twWrapNote');
  if (wn) wn.hidden = !wraps;

  var srvNow = timePart(get(S.state, 'ts_server', ''));
  var locNow = localHHMMSS();
  var sEl = $('tzServerNow');
  if (sEl) sEl.textContent = srvNow.slice(0, 5) + ' ' + gmtLabel(get(S.state, 'server_gmt_offset_hours', NaN));
  var lEl = $('tzLocalNow');
  if (lEl) lEl.textContent = locNow.slice(0, 5);

  var note = $('daysTzNote');
  if (note) note.textContent = '(' + (tw.timezone === 'local' ? 'local PC time' : 'broker server time') + ')';

  /* day toggles */
  var days = Array.isArray(tw.days) ? tw.days : [];
  for (var d = 0; d < 7; d++) {
    var cb = $('day' + d);
    if (cb && cb !== document.activeElement) cb.checked = days.indexOf(d) >= 0;
    if (cb) cb.setAttribute('aria-label', DAY_NAMES[d]);
  }

  /* live window state from the EA */
  var led = $('winLed'), txt = $('winTxt'), next = $('winNext');
  var haveWin = !!get(S.state, 'window', null);
  if (!S.cfg) return;
  if (!get(tw, 'enabled', false)) {
    if (led) led.className = 'win-led open';
    if (txt) txt.textContent = 'No time restriction - entries allowed 24/5.';
    if (next) next.textContent = '';
  } else if (!haveWin) {
    if (led) led.className = 'win-led';
    if (txt) txt.textContent = 'Window state unknown - the EA is not reporting.';
    if (next) next.textContent = '';
  } else {
    var open = !!get(S.state, 'window.open', false);
    if (led) led.className = 'win-led ' + (open ? 'open' : 'closed');
    var reason = String(get(S.state, 'window.reason', '') || (open ? 'in window' : 'outside window'));
    var wtz = String(get(S.state, 'window.timezone', tw.timezone || 'server'));
    if (txt) txt.textContent = (open ? 'OPEN' : 'CLOSED') + ' · ' + reason + ' · ' + wtz + ' time';
    var nc = String(get(S.state, 'window.next_change', '') || '');
    if (next) next.textContent = nc ? 'next change ' + timePart(nc) : '';
  }
}

function setDay(d, on) {
  if (!S.cfg) return;
  if (!S.cfg.time_window) S.cfg.time_window = {};
  var days = Array.isArray(S.cfg.time_window.days) ? S.cfg.time_window.days.slice() : [];
  var i = days.indexOf(d);
  if (on && i < 0) days.push(d);
  if (!on && i >= 0) days.splice(i, 1);
  days.sort(function (a, b) { return a - b; });
  S.cfg.time_window.days = days;
  onEdit(false);
}

/* ------------------------------------------------------------------ 12 */
/* Entry / execution */

function renderEntry() {
  var u = uctx();
  var en = get(S.cfg, 'entry', {}) || {};

  var chips = document.querySelectorAll('#entryCard [data-unit]');
  for (var i = 0; i < chips.length; i++) chips[i].textContent = u.unit;

  /* cooldown does not apply when one_shot is on */
  var oneShot = !!get(en, 'one_shot', false);
  var cdEl = $('en_cooldown_sec');
  var cdField = $('cooldownField');
  if (cdEl) cdEl.disabled = oneShot;
  if (cdField) cdField.classList.toggle('off', oneShot);
  setHTML($('en_cooldown_res'), oneShot
    ? '<span class="warnres">not used while one shot is on</span>'
    : 'a fired level re-arms ' + esc(trimNum(get(en, 'cooldown_sec', 0))) + 's later');

  setHTML($('en_trigger_res'), get(en, 'trigger_mode', 'cross') === 'cross'
    ? 'arm side is fixed when the level arms; fires when price crosses in from that side'
    : '<span class="warnres">fires on any touch inside the tolerance, arm side ignored</span>');

  setHTML($('en_touch_res'), distReadout(get(en, 'touch_tolerance', 0), num(get(S.cfg, 'long.lots', 0), 0), 'exact price, no tolerance band'));
  var tolEl = $('en_touch_tolerance');
  if (tolEl) tolEl.step = String(u.unit === 'price' ? 0.01 : 0.1);

  /* spread check in context */
  var maxSpread = num(get(en, 'max_spread_points', 0), 0);
  var curSpread = num(get(S.state, 'symbol.spread_points', NaN), NaN);
  var sr = $('en_spread_res');
  if (sr) {
    if (maxSpread === 0) setHTML(sr, '0 = no spread check');
    else if (!Number.isFinite(curSpread)) setHTML(sr, 'entries skipped above this spread');
    else if (curSpread > maxSpread) setHTML(sr, '<span class="errres">now ' + esc(fmt(curSpread, 0)) + ' pts - entries blocked</span>');
    else setHTML(sr, 'now <span class="m">' + esc(fmt(curSpread, 0)) + '</span> pts - ok');
  }

  var conc = num(get(en, 'max_concurrent_straddles', 0), 0);
  var liveLvls = 0;
  var sl = get(S.state, 'levels', []);
  if (Array.isArray(sl)) for (var k = 0; k < sl.length; k++) if (int(get(sl[k], 'open_positions', 0), 0) > 0) liveLvls++;
  setHTML($('en_conc_res'), conc === 0
    ? '0 = unlimited'
    : esc(liveLvls) + ' of ' + esc(conc) + ' level slots in use');

  /* pip size in effect */
  var pn = $('pipNote');
  if (pn) {
    pn.textContent = '1 pip = ' + trimNum(u.pip) + ' price' +
      (get(S.cfg, 'pip_size_mode', 'auto') === 'manual' ? ' (manual)' : ' (auto)');
  }
  var manualField = $('pipManualField');
  var manualEl = $('cfg_pip_manual');
  var isManual = get(S.cfg, 'pip_size_mode', 'auto') === 'manual';
  if (manualEl) manualEl.disabled = !isManual;
  if (manualField) manualField.classList.toggle('off', !isManual);

  /* symbol found / candidates */
  var found = get(S.state, 'symbol.found', null);
  var fr = $('symFoundRes');
  if (fr) {
    if (!S.state) setHTML(fr, 'EA not running - cannot verify the symbol yet');
    else if (found === true) setHTML(fr, 'found · ' + esc(String(get(S.state, 'symbol.name', ''))) +
      ' · digits <span class="m">' + esc(fmt(u.digits, 0)) + '</span> · point <span class="m">' + esc(trimNum(u.point)) + '</span>');
    else setHTML(fr, '<span class="errres">not found in Market Watch</span>');
  }

  var cands = get(S.state, 'symbol.candidates', []);
  var box = $('symCandidates');
  var list = $('symCandList');
  var showCands = found === false && Array.isArray(cands) && cands.length > 0;
  if (box) box.hidden = !showCands;
  if (list) {
    var csig = showCands ? cands.join('|') : '';
    if (csig !== S.sig.sym) {
      S.sig.sym = csig;
      list.textContent = '';
      if (showCands) {
        cands.slice(0, 12).forEach(function (c) {
          var b = document.createElement('button');
          b.type = 'button';
          b.className = 'cand';
          b.textContent = String(c);
          b.addEventListener('click', function () {
            cfgSet('symbol', String(c));
            onEdit(true);
            refreshAll();
            toast('ok', 'Symbol set to ' + c);
          });
          list.appendChild(b);
        });
      }
    }
  }

  renderSymbolSpec();
}

function renderSymbolSpec() {
  var el = $('symTable');
  if (!el) return;
  var sy = get(S.state, 'symbol', null);
  var rowsData;
  if (!sy) {
    rowsData = [['status', 'no EA state', 'bad']];
  } else {
    rowsData = [
      ['name', String(get(sy, 'name', DASH)), get(sy, 'found', false) ? 'good' : 'bad'],
      ['found', get(sy, 'found', false) ? 'yes' : 'no', get(sy, 'found', false) ? 'good' : 'bad'],
      ['digits', fmt(get(sy, 'digits', NaN), 0), ''],
      ['point', trimNum(get(sy, 'point', NaN)), ''],
      ['pip_size', trimNum(get(sy, 'pip_size', NaN)), ''],
      ['tick_size', trimNum(get(sy, 'tick_size', NaN)), ''],
      ['tick_value', trimNum(get(sy, 'tick_value', NaN)), ''],
      ['contract_size', trimNum(get(sy, 'contract_size', NaN)), ''],
      ['volume min/step/max', trimNum(get(sy, 'volume_min', NaN)) + ' / ' + trimNum(get(sy, 'volume_step', NaN)) + ' / ' + trimNum(get(sy, 'volume_max', NaN)), ''],
      ['stops level', fmt(get(sy, 'stops_level_points', NaN), 0) + ' pts', ''],
      ['freeze level', fmt(get(sy, 'freeze_level_points', NaN), 0) + ' pts', ''],
      ['trade mode', String(get(sy, 'trade_mode', DASH)), String(get(sy, 'trade_mode', '')) === 'full' ? 'good' : 'warn'],
      ['filling', String(get(sy, 'filling', DASH)), '']
    ];
  }
  var html = rowsData.map(function (r) {
    return '<dt>' + esc(r[0]) + '</dt><dd' + (r[2] ? ' class="' + r[2] + '"' : '') + '>' + esc(r[1]) + '</dd>';
  }).join('');
  if (el.dataset.sig !== html) { el.dataset.sig = html; el.innerHTML = html; }
}

/* ------------------------------------------------------------------ 13 */
/* Positions */

var posRows = new Map();

function renderPositions() {
  var body = $('posBody');
  if (!body) return;
  var list = get(S.state, 'positions', []);
  if (!Array.isArray(list)) list = [];

  var sig = list.map(function (p) { return get(p, 'ticket', '?'); }).join('|');
  if (sig !== S.sig.pos) {
    S.sig.pos = sig;
    var keep = {};
    for (var i = 0; i < list.length; i++) keep[String(get(list[i], 'ticket', '?'))] = true;
    posRows.forEach(function (tr, tk) {
      if (!keep[tk]) { if (tr.parentNode) tr.parentNode.removeChild(tr); posRows.delete(tk); }
    });
    var frag = document.createDocumentFragment();
    for (var k = 0; k < list.length; k++) {
      var tk = String(get(list[k], 'ticket', '?'));
      var tr = posRows.get(tk);
      if (!tr) {
        tr = document.createElement('tr');
        tr.innerHTML = '<td></td><td></td><td></td><td class="num"></td><td class="num"></td>' +
          '<td class="num"></td><td class="num"></td><td class="num"></td>' +
          '<td class="num"></td><td class="num"></td><td class="c-be"></td><td></td>';
        posRows.set(tk, tr);
      }
      frag.appendChild(tr);
    }
    body.textContent = '';
    body.appendChild(frag);
  }

  var sumPips = 0, havePips = false;
  for (var n = 0; n < list.length; n++) {
    var p = list[n];
    var tr2 = posRows.get(String(get(p, 'ticket', '?')));
    if (!tr2) continue;
    var c = tr2.children;
    var side = String(get(p, 'side', '')).toLowerCase();
    var pips = num(get(p, 'profit_pips', NaN), NaN);
    var money = num(get(p, 'profit_money', NaN), NaN);
    if (Number.isFinite(pips)) { sumPips += pips; havePips = true; }

    c[0].textContent = String(get(p, 'ticket', DASH));
    c[1].textContent = String(get(p, 'level_id', DASH));
    c[2].innerHTML = side === 'long' || side === 'short'
      ? '<span class="sidetag sidetag-' + side + '">' + iconSvg(side === 'long' ? 'caret-up' : 'caret-down') +
        (side === 'long' ? 'LONG' : 'SHORT') + '</span>'
      : DASH;
    c[3].textContent = fmt(get(p, 'lots', NaN), 2);
    c[4].textContent = fmtPrice(get(p, 'open_price', NaN));
    c[5].textContent = fmtPrice(get(p, 'current', NaN));
    var slv = num(get(p, 'sl', 0), 0);
    var tpv = num(get(p, 'tp', 0), 0);
    c[6].textContent = slv > 0 ? fmtPrice(slv) : DASH;
    c[7].textContent = tpv > 0 ? fmtPrice(tpv) : DASH;
    c[8].textContent = fmtSigned(pips, 1);
    c[8].className = 'num ' + plClass(pips);
    c[9].textContent = fmtMoney(money, true);
    c[9].className = 'num ' + plClass(money);
    var be = !!get(p, 'be_done', false);
    c[10].innerHTML = '<span class="bedot' + (be ? ' on' : '') + '" data-tip="' +
      (be ? 'Breakeven has been applied to this position.' : 'Breakeven not applied yet.') + '">BE</span>';
    c[11].textContent = timePart(get(p, 'open_time', ''));
  }

  var cnt = $('posCount');
  if (cnt) cnt.textContent = String(list.length);
  var empty = $('posEmpty');
  if (empty) empty.hidden = list.length > 0;

  /* totals */
  var t = get(S.state, 'totals', {}) || {};
  setText($('tot_lots'), S.state ? fmt(get(t, 'open_lots', NaN), 2) : DASH);
  var fl = num(get(t, 'floating_money', NaN), NaN);
  var flEl = $('tot_float');
  if (flEl) { flEl.textContent = fmtMoney(fl, true); flEl.className = 'num mono live ' + plClass(fl); }
  var tpEl = $('tot_pips');
  if (tpEl) {
    tpEl.textContent = havePips ? fmtSigned(sumPips, 1) : DASH;
    tpEl.className = 'num mono live ' + plClass(havePips ? sumPips : NaN);
  }
  setText($('tot_session'), 'closed ' + (S.state ? fmtMoney(get(t, 'session_closed_money', NaN), true) : DASH));
  setText($('tot_fires'), 'fires ' + (S.state ? fmt(get(t, 'session_fires', NaN), 0) : DASH));
}

function plClass(v) {
  var n = num(v, NaN);
  if (!Number.isFinite(n)) return 'pl-flat';
  return n > 0 ? 'pl-pos' : n < 0 ? 'pl-neg' : 'pl-flat';
}

function setText(el, s) { if (el) el.textContent = s; }

/* ------------------------------------------------------------------ 14 */
/* Log - newest first (labelled in the dock header) */

function logFilters() {
  return {
    info: !!($('lf_info') && $('lf_info').checked),
    trade: !!($('lf_trade') && $('lf_trade').checked),
    warn: !!($('lf_warn') && $('lf_warn').checked),
    error: !!($('lf_error') && $('lf_error').checked)
  };
}

function renderLog() {
  var el = $('logList');
  if (!el) return;
  var raw = get(S.state, 'log', []);
  if (!Array.isArray(raw)) raw = [];
  var f = logFilters();

  var items = [];
  for (var i = raw.length - 1; i >= 0; i--) {
    var e = raw[i];
    var lvl = String(get(e, 'lvl', 'info')).toLowerCase();
    if (['info', 'warn', 'error', 'trade'].indexOf(lvl) < 0) lvl = 'info';
    if (!f[lvl]) continue;
    items.push({ t: String(get(e, 't', '')), lvl: lvl, msg: String(get(e, 'msg', '')) });
  }

  var sig = String(raw.length) + '#' + JSON.stringify(f) + '#' +
    items.map(function (x) { return x.t + x.lvl + x.msg; }).join('');
  if (sig === S.sig.log) return;
  S.sig.log = sig;

  setText($('logCount'), String(raw.length));

  if (!items.length) {
    el.innerHTML = '<li class="log-empty">' +
      (S.state ? (raw.length ? 'No entries match the current filter.' : 'No log entries from the EA yet.')
               : 'No EA state - nothing logged yet.') + '</li>';
    return;
  }

  var html = '';
  for (var k = 0; k < items.length; k++) {
    html += '<li class="log-row log-' + items[k].lvl + '">' +
      '<span class="log-t">' + esc(items[k].t) + '</span>' +
      '<span class="log-l">' + esc(items[k].lvl) + '</span>' +
      '<span class="log-m" title="' + esc(items[k].msg) + '">' + esc(items[k].msg) + '</span>' +
      '</li>';
  }
  el.innerHTML = html;
  el.scrollTop = 0;
}

/* ------------------------------------------------------------------ 15 */
/* Header, banners, meta */

function renderHeader() {
  var alive = !!get(S.bridge, 'ea_alive', false) && !!S.state;
  var mode = !S.bridgeOk ? 'offline' : 'ok';
  var eaMode = !S.state ? 'absent' : (alive ? 'alive' : 'stale');
  document.body.dataset.bridge = mode;
  document.body.dataset.ea = eaMode;
  document.body.dataset.sort = S.sortMode;

  /* master switch */
  var armed = !!get(S.cfg, 'enabled', false);
  var btn = $('armBtn');
  if (btn) {
    btn.setAttribute('aria-pressed', armed ? 'true' : 'false');
    setText($('armText'), armed ? 'ARMED' : 'DISARMED');
    setText($('armHint'), armed ? 'entries allowed - click to disarm' : 'no new entries - click to arm');
  }
  document.body.dataset.armed = armed ? '1' : '0';

  /* effective status light from state.armed */
  var led = $('armedLed'), txt = $('armedText');
  var blockers = get(S.state, 'blockers', []);
  if (!Array.isArray(blockers)) blockers = [];
  if (!S.bridgeOk) {
    if (led) led.className = 'led led-err';
    setText(txt, 'BRIDGE OFFLINE');
  } else if (!S.state) {
    if (led) led.className = 'led led-off';
    setText(txt, 'NO EA STATE');
  } else if (!alive) {
    if (led) led.className = 'led led-err';
    setText(txt, 'EA NOT RESPONDING');
  } else if (get(S.state, 'armed', false)) {
    if (led) led.className = 'led led-on';
    setText(txt, 'LIVE - EA IS ARMED');
  } else {
    if (led) led.className = 'led led-warn';
    setText(txt, 'NOT ARMED');
  }

  /* blocker chips */
  var chipHost = $('blockers');
  if (chipHost) {
    var csig = (alive ? '1' : '0') + '|' + blockers.join('|');
    if (chipHost.dataset.sig !== csig) {
      chipHost.dataset.sig = csig;
      if (!S.state || !alive || !blockers.length) {
        chipHost.textContent = '';
      } else {
        chipHost.innerHTML = blockers.slice(0, 8).map(function (b) {
          var bad = /hedg|not allowed|error|disabled|not connected/i.test(String(b));
          return '<span class="chip' + (bad ? ' chip-err' : '') + '">' + esc(String(b)) + '</span>';
        }).join('');
      }
    }
  }

  /* sync pill */
  var sp = $('syncPill');
  if (sp) {
    var inSync = get(S.bridge, 'in_sync', null);
    if (S.bridgeOk && inSync === false) {
      sp.hidden = false;
      sp.className = 'pill pill-sync';
      sp.innerHTML = '<span class="pdot"></span>SYNCING…';
      sp.setAttribute('data-tip', 'The panel has written config v' + fmt(get(S.bridge, 'config_version', NaN), 0) +
        ' but the EA has only applied v' + fmt(get(S.bridge, 'config_version_seen', NaN), 0) + '.');
    } else {
      sp.hidden = true;
    }
  }

  /* quotes with directional flash */
  var bid = num(get(S.state, 'symbol.bid', NaN), NaN);
  var ask = num(get(S.state, 'symbol.ask', NaN), NaN);
  var showLive = S.bridgeOk && !!S.state;
  flashQuote($('bidV'), bid, 'lastBid', showLive);
  flashQuote($('askV'), ask, 'lastAsk', showLive);

  var spr = $('spreadV');
  if (spr) {
    var sp2 = num(get(S.state, 'symbol.spread_points', NaN), NaN);
    var abs = (Number.isFinite(bid) && Number.isFinite(ask)) ? (ask - bid) : NaN;
    spr.textContent = (showLive && Number.isFinite(sp2))
      ? fmt(sp2, 0) + ' pts' + (Number.isFinite(abs) ? '  ' + priceSign() + fmt(abs, Math.min(uctx().digits, 2)) : '')
      : DASH;
  }

  /* stale / offline marker over the quote block */
  var stale = $('quoteStale');
  var staleTxt = $('quoteStaleText');
  if (stale && staleTxt) {
    if (!S.bridgeOk) {
      stale.hidden = false;
      staleTxt.textContent = 'BRIDGE OFFLINE - NO LIVE PRICE';
    } else if (!S.state) {
      stale.hidden = false;
      staleTxt.textContent = 'EA NOT RUNNING - NO PRICE';
    } else if (!alive) {
      stale.hidden = false;
      var age = num(get(S.bridge, 'state_age_ms', NaN), NaN);
      staleTxt.textContent = 'EA NOT RESPONDING - LAST UPDATE ' +
        (Number.isFinite(age) ? fmt(age / 1000, 1) + 'S AGO' : 'UNKNOWN') + ' - NOT LIVE';
    } else {
      stale.hidden = true;
    }
  }

  /* clocks */
  setText($('clkServer'), S.state ? timePart(get(S.state, 'ts_server', '')) : '--:--:--');
  setText($('clkServerTz'), gmtLabel(get(S.state, 'server_gmt_offset_hours', NaN)));
  setText($('clkLocal'), localHHMMSS());

  /* connection pills */
  pill($('pillEa'), alive ? 'ok' : (S.state ? 'err' : 'mute'), 'EA',
    alive ? 'EA heartbeat is fresh (state.json written ' + fmt(num(get(S.bridge, 'state_age_ms', NaN), NaN), 0) + ' ms ago).'
          : 'No EA heartbeat. state.json is older than 3000 ms or missing.');
  var inS = get(S.bridge, 'in_sync', null);
  pill($('pillSync'), inS === true ? 'ok' : inS === false ? 'warn' : 'mute', 'SYNC',
    'config_version ' + fmt(get(S.bridge, 'config_version', NaN), 0) + ' written / ' +
    fmt(get(S.bridge, 'config_version_seen', NaN), 0) + ' applied by the EA.');
  var conn = get(S.state, 'terminal.connected', null);
  pill($('pillConn'), conn === true ? 'ok' : conn === false ? 'err' : 'mute', 'LINK',
    'terminal.connected - is MetaTrader connected to the broker.');
  var ta = get(S.state, 'terminal.trade_allowed', null);
  pill($('pillTrade'), ta === true ? 'ok' : ta === false ? 'err' : 'mute', 'TRADE',
    'terminal.trade_allowed - the AutoTrading button in MetaTrader.');
  var ee = get(S.state, 'terminal.expert_enabled', null);
  pill($('pillExperts'), ee === true ? 'ok' : ee === false ? 'err' : 'mute', 'EXPERTS',
    'terminal.expert_enabled - algo trading permitted for expert advisors.');

  /* account */
  var acc = get(S.state, 'account', null);
  setText($('accLogin'), acc ? fmtInt(get(acc, 'login', NaN)) : DASH);
  setText($('accServer'), acc ? String(get(acc, 'server', DASH)) : DASH);
  setText($('accBal'), acc ? fmtMoney(get(acc, 'balance', NaN), false) : DASH);
  setText($('accEq'), acc ? fmtMoney(get(acc, 'equity', NaN), false) : DASH);
  var floatv = num(get(S.state, 'totals.floating_money', NaN), NaN);
  if (!Number.isFinite(floatv) && acc) {
    var b = num(get(acc, 'balance', NaN), NaN), eq = num(get(acc, 'equity', NaN), NaN);
    if (Number.isFinite(b) && Number.isFinite(eq)) floatv = eq - b;
  }
  var fEl = $('accFloat');
  if (fEl) { fEl.textContent = fmtMoney(floatv, true); fEl.className = 'stat-val mono live ' + (num(floatv, 0) > 0 ? 'pos' : num(floatv, 0) < 0 ? 'neg' : ''); }
  setText($('accFree'), acc ? fmtMoney(get(acc, 'margin_free', NaN), false) : DASH);
  setText($('accLev'), acc ? '1:' + fmt(get(acc, 'leverage', NaN), 0) : DASH);
  var mm = $('accMode');
  if (mm) {
    var mode2 = acc ? String(get(acc, 'margin_mode', DASH)) : DASH;
    mm.textContent = mode2;
    mm.className = 'stat-val' + (acc && get(acc, 'hedging_ok', false) === false ? ' neg' : '');
  }

  setText($('brandSymbol'), String(get(S.cfg, 'symbol', 'XAUUSD') || 'XAUUSD'));
  setText($('brandVenue'), String(get(S.state, 'terminal.company', 'MetaTrader 5') || 'MetaTrader 5'));

  /* config version box */
  var cv = $('cfgVerBox');
  if (cv) cv.textContent = 'v' + fmt(get(S.bridge, 'config_version', NaN), 0) + ' / ' + fmt(get(S.bridge, 'config_version_seen', NaN), 0);

  renderSaveStatus();
}

function pill(el, kind, label, tip) {
  if (!el) return;
  el.className = 'pill pill-' + kind;
  el.innerHTML = '<span class="pdot"></span>' + esc(label);
  if (tip) el.setAttribute('data-tip', tip);
}

function flashQuote(el, v, key, showLive) {
  if (!el) return;
  if (!showLive || !Number.isFinite(v)) {
    el.textContent = DASH;
    el.classList.remove('up', 'dn');
    S[key] = NaN;
    return;
  }
  var prev = S[key];
  el.textContent = fmtPrice(v);
  if (Number.isFinite(prev) && v !== prev) {
    el.classList.remove('up', 'dn');
    /* force a restart of the tint transition */
    void el.offsetWidth;
    el.classList.add(v > prev ? 'up' : 'dn');
    if (el._flashT) clearTimeout(el._flashT);
    el._flashT = setTimeout(function () { el.classList.remove('up', 'dn'); }, 240);
  }
  S[key] = v;
}

function renderSaveStatus() {
  var el = $('saveStatus');
  if (!el) return;
  var cls = 'save save-idle', txt = 'ready';

  if (!S.cfgLoaded) {
    cls = 'save save-err'; txt = 'config not loaded - edits are NOT saved';
  } else if (S.saveErrors.length) {
    cls = 'save save-err'; txt = 'save rejected - see banner';
  } else if (S.saving) {
    cls = 'save save-saving'; txt = 'saving…';
  } else if (S.savePending) {
    cls = 'save save-saving'; txt = 'edited - saving in a moment';
  } else if (S.awaitVersion !== null) {
    var seen = num(get(S.bridge, 'config_version_seen', NaN), NaN);
    if (Number.isFinite(seen) && seen >= S.awaitVersion) {
      if (!S.syncedAt) S.syncedAt = Date.now();
      if (Date.now() - S.syncedAt < 5000) { cls = 'save save-live'; txt = 'EA picked it up'; }
      else { cls = 'save save-saved'; txt = 'in sync'; S.awaitVersion = null; S.syncedAt = 0; }
    } else {
      cls = 'save save-saved'; txt = 'saved - waiting for the EA';
    }
  } else if (S.saveWarnings.length) {
    cls = 'save save-saved'; txt = 'saved with warnings';
  }
  if (el.className !== cls) el.className = cls;
  if (el.textContent !== txt) el.textContent = txt;
}

function banner(kind, iconName, title, body, list) {
  var key = title.replace(/[^a-z0-9]+/gi, '-').toLowerCase();
  if (S.dismissedBanners && S.dismissedBanners[key]) return '';
  var h = '<div class="banner banner-' + kind + '" data-banner-key="' + esc(key) + '">' + iconSvg(iconName) + '<div class="banner-body">';
  h += '<b>' + esc(title) + '</b>' + (body ? ' ' + esc(body) : '');
  if (list && list.length) {
    h += '<ul class="banner-list">';
    for (var i = 0; i < list.length; i++) h += '<li>' + esc(String(list[i])) + '</li>';
    h += '</ul>';
  }
  return h + '</div><button type="button" class="banner-close" aria-label="Dismiss warning" title="Dismiss">&times;</button></div>';
}

function renderBanners() {
  var host = $('banners');
  if (!host) return;
  var out = [];

  if (!S.bridgeOk) {
    out.push(banner('err', 'alert', 'Panel bridge offline.',
      'The local Python server is not answering. Nothing below is live and no edit can be saved. ' +
      'Start the panel launcher again, then this banner clears by itself. Retrying every ' +
      fmt(pollDelay() / 1000, 1) + 's.'));
  }

  if (!S.cfgLoaded && S.bridgeOk) {
    out.push(banner('err', 'alert', 'Configuration not loaded.',
      'GET /api/config did not return a config, so the fields below show defaults and edits are not being saved. Retrying.'));
  }

  if (S.state && get(S.bridge, 'ea_alive', false) === false) {
    var age = num(get(S.bridge, 'state_age_ms', NaN), NaN);
    out.push(banner('err', 'alert', 'EA not responding.',
      'The last state.json is ' + (Number.isFinite(age) ? fmt(age / 1000, 1) + 's' : 'more than 3s') +
      ' old, so every live number on this screen is the last known value, not the market. ' +
      'Check the chart still has GoldStraddle attached and that MetaTrader is running.'));
  }

  if (S.state && get(S.state, 'account.hedging_ok', true) === false) {
    out.push(banner('err', 'alert', 'This account is not hedging.',
      'Margin mode is "' + String(get(S.state, 'account.margin_mode', 'unknown')) +
      '". A long and a short cannot coexist, so they would net to zero. The EA refuses to open any straddle ' +
      'until this account is a hedging account.'));
  }

  var cerr = String(get(S.state, 'config_error', '') || '');
  if (cerr) out.push(banner('err', 'alert', 'The EA rejected the configuration.', cerr));

  if (S.state && get(S.state, 'terminal.connected', true) === false) {
    out.push(banner('err', 'alert', 'MetaTrader is not connected to the broker.', 'No prices, no execution.'));
  }
  if (S.state && (get(S.state, 'terminal.trade_allowed', true) === false || get(S.state, 'terminal.expert_enabled', true) === false)) {
    out.push(banner('warn', 'alert', 'Algo trading is switched off in MetaTrader.',
      'Turn the AutoTrading button on (toolbar) and allow algo trading for experts. Until then the EA cannot open anything.'));
  }
  if (S.state && get(S.state, 'symbol.found', true) === false) {
    out.push(banner('warn', 'alert', 'Symbol "' + String(get(S.cfg, 'symbol', '')) + '" is not in Market Watch.',
      'Pick one of the suggestions in the Entry card, or add the symbol in MetaTrader.'));
  }
  if (S.panicLatched) {
    out.push(banner('warn', 'alert', 'PANIC latch is set.',
      'The EA will not open anything until this panel arms again. Use the ARM switch when you are ready.'));
  }

  if (S.saveErrors.length) {
    out.push(banner('err', 'alert', 'The panel server rejected this configuration.',
      'It was not written, so the EA is still running the previous version. Fix these and it saves automatically:', S.saveErrors));
  }
  var warns = S.clientWarnings.concat(S.saveWarnings);
  if (warns.length) {
    out.push(banner('warn', 'alert', 'Saved with warnings.', '', warns));
  }
  if (S.clientErrors.length && !S.saveErrors.length) {
    out.push(banner('warn', 'alert', 'These values will be rejected on save.', '', S.clientErrors));
  }

  var html = out.join('');
  if (html !== S.sig.banners) { S.sig.banners = html; host.innerHTML = html; }
}

function renderMeta() {
  var el = $('metaTable');
  if (!el) return;
  var age = num(get(S.bridge, 'state_age_ms', NaN), NaN);
  var rowsData = [
    ['bridge', S.bridgeOk ? 'online' : 'offline', S.bridgeOk ? 'good' : 'bad'],
    ['ea heartbeat', Number.isFinite(age) ? fmt(age, 0) + ' ms ago' : DASH,
      get(S.bridge, 'ea_alive', false) ? 'good' : 'bad'],
    ['write period', fmt(get(S.state, 'heartbeat_ms', NaN), 0) + ' ms', ''],
    ['ea version', String(get(S.state, 'ea_version', DASH)), ''],
    ['state schema', fmt(get(S.state, 'schema', NaN), 0), ''],
    ['config version', fmt(get(S.bridge, 'config_version', NaN), 0) + ' written', ''],
    ['seen by ea', fmt(get(S.bridge, 'config_version_seen', NaN), 0),
      get(S.bridge, 'in_sync', false) ? 'good' : 'warn'],
    ['ts utc', String(get(S.state, 'ts_utc', DASH)), ''],
    ['terminal', String(get(S.state, 'terminal.name', DASH)) + ' b' + fmt(get(S.state, 'terminal.build', NaN), 0), ''],
    ['company', String(get(S.state, 'terminal.company', DASH)), ''],
    ['dll allowed', S.state ? (get(S.state, 'terminal.dll_allowed', false) ? 'yes' : 'no') : DASH, ''],
    ['ipc folder', String(get(S.bridge, 'ipc_dir', DASH)), '']
  ];
  var html = rowsData.map(function (r) {
    return '<dt>' + esc(r[0]) + '</dt><dd' + (r[2] ? ' class="' + r[2] + '"' : '') + '>' + esc(r[1]) + '</dd>';
  }).join('');
  if (S.sig.meta !== html) { S.sig.meta = html; el.innerHTML = html; }

  var lc = $('lastCmd');
  if (lc) {
    var res = String(get(S.state, 'last_command_result', '') || '');
    var lcid = get(S.state, 'last_command_id', null);
    if (res) {
      lc.hidden = false;
      lc.textContent = 'last command #' + fmt(lcid, 0) + ': ' + res;
    } else {
      lc.hidden = true;
    }
    if (S.expectCmdId !== null && lcid === S.expectCmdId && S.shownCmdId !== lcid) {
      S.shownCmdId = lcid;
      S.expectCmdId = null;
      toast(/fail|error|refus|reject/i.test(res) ? 'err' : 'ok', res || 'command executed');
    }
  }

  var dir = $('eaEmptyDir');
  if (dir) dir.textContent = String(get(S.bridge, 'ipc_dir', DASH));
  var empty = $('eaEmpty');
  if (empty) empty.hidden = !(S.bridgeOk && !S.state);
}

/* ------------------------------------------------------------------ 16 */
/* Client-side mirror of CONTRACT section 8. The server is the authority;
   this only gives instant feedback. */

function validateClient() {
  var errs = [], warns = [], bad = {};
  if (!S.cfg) { S.clientErrors = []; S.clientWarnings = []; S.badFields = {}; return; }

  ['long', 'short'].forEach(function (side) {
    var p = get(S.cfg, side, {}) || {};
    var lots = num(p.lots, NaN);
    if (!Number.isFinite(lots) || lots < 0.01 || lots > 100) {
      errs.push(side + '.lots must be between 0.01 and 100');
      bad[side + '_lots'] = 1;
    }
    ['tp', 'sl', 'be_trigger'].forEach(function (f) {
      var v = num(p[f], NaN);
      if (!Number.isFinite(v) || v < 0) { errs.push(side + '.' + f + ' must be 0 or more'); bad[side + '_' + f] = 1; }
    });
    var bo = num(p.be_offset, NaN);
    if (!Number.isFinite(bo) || bo < -1000 || bo > 1000) {
      errs.push(side + '.be_offset must be between -1000 and 1000');
      bad[side + '_be_offset'] = 1;
    }
    var tp = num(p.tp, 0), bt = num(p.be_trigger, 0);
    if (tp > 0 && bt > 0 && bt > tp) {
      warns.push(side + ': be_trigger (' + trimNum(bt) + ') is beyond tp (' + trimNum(tp) + ') so breakeven can never fire');
    }
  });

  var ls = levels();
  if (ls.length > MAX_LEVELS) errs.push('at most ' + MAX_LEVELS + ' levels (' + ls.length + ' configured)');
  var seen = {};
  var enabledCount = 0;
  for (var i = 0; i < ls.length; i++) {
    var l = ls[i] || {};
    var id = String(l.id || '');
    if (seen[id]) errs.push('duplicate level id "' + id + '"');
    seen[id] = 1;
    if (l.enabled) {
      enabledCount++;
      if (!(num(l.price, 0) > 0)) errs.push('level ' + (id || i + 1) + ': price must be greater than 0 while enabled');
    }
  }
  if (get(S.cfg, 'enabled', false) && enabledCount === 0) {
    warns.push('the master switch is on but no level is enabled - nothing can fire');
  }

  var tw = get(S.cfg, 'time_window', {}) || {};
  if (hhmmToMins(tw.start) === null) { errs.push('time window start must be HH:MM'); bad.tw_start = 1; }
  if (hhmmToMins(tw.end) === null) { errs.push('time window end must be HH:MM'); bad.tw_end = 1; }
  if (get(tw, 'enabled', false) && Array.isArray(tw.days) && tw.days.length === 0) {
    warns.push('the time window is on with no days selected - nothing can ever fire');
  }

  var magic = num(get(S.cfg, 'magic', NaN), NaN);
  if (!Number.isFinite(magic) || magic < 1 || magic > 2147483647) {
    errs.push('magic must be between 1 and 2147483647'); bad.cfg_magic = 1;
  }
  var sym = String(get(S.cfg, 'symbol', ''));
  if (!/^[A-Za-z0-9._#-]{1,32}$/.test(sym)) { errs.push('symbol must be 1-32 of A-Z a-z 0-9 . _ # -'); bad.cfg_symbol = 1; }

  S.clientErrors = errs;
  S.clientWarnings = warns;

  /* paint field state */
  Object.keys(S.badFields).forEach(function (id) {
    if (!bad[id]) { var e = $(id); if (e) e.classList.remove('bad'); }
  });
  Object.keys(bad).forEach(function (id) { var e = $(id); if (e) e.classList.add('bad'); });
  S.badFields = bad;
}

/* ------------------------------------------------------------------ 17 */
/* Network */

function pollDelay() {
  if (S.failStreak === 0) return POLL_MS;
  return Math.min(POLL_MAX_MS, POLL_MS * Math.pow(2, S.failStreak));
}

/** Self-scheduling loop: the next request only goes out after this one lands. */
function pollLoop() {
  fetchState().then(function () {
    S.pollTimer = setTimeout(pollLoop, pollDelay());
  });
}

function fetchState() {
  var opts = { cache: 'no-store', headers: { 'Accept': 'application/json' } };
  if (typeof AbortSignal !== 'undefined' && AbortSignal.timeout) opts.signal = AbortSignal.timeout(POLL_TIMEOUT_MS);
  return fetch('/api/state', opts).then(function (r) {
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  }).then(function (j) {
    S.bridgeOk = true;
    S.failStreak = 0;
    S.state = (j && j.state) ? j.state : null;
    S.bridge = (j && j.bridge) ? j.bridge : {};
    renderLive();
  }).catch(function () {
    S.failStreak++;
    S.bridgeOk = false;
    /* keep the last state object so the log and tables stay readable, but the
       whole live layer is dimmed and labelled by renderHeader/renderBanners */
    renderLive();
  });
}

function loadConfig() {
  return fetch('/api/config', { cache: 'no-store', headers: { 'Accept': 'application/json' } })
    .then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(function (j) {
      var c = j && j.config ? j.config : null;
      if (!c) throw new Error('no config in response');
      S.cfg = normalizeConfig(c);
      S.cfgLoaded = true;
      S.sortMode = ($('sortPrice') && $('sortPrice').checked) ? 'price' : 'manual';
      refreshAll();
      syncLevels(true);
      renderLive();
    })
    .catch(function () {
      if (!S.cfg) { S.cfg = normalizeConfig(DEFAULT_CONFIG); refreshAll(); syncLevels(true); }
      S.cfgLoaded = false;
      renderLive();
      setTimeout(loadConfig, 3000);
    });
}

function normalizeConfig(raw) {
  var out = JSON.parse(JSON.stringify(DEFAULT_CONFIG));
  function merge(dst, src) {
    if (!src || typeof src !== 'object') return dst;
    Object.keys(src).forEach(function (k) {
      var v = src[k];
      if (v && typeof v === 'object' && !Array.isArray(v) && dst[k] && typeof dst[k] === 'object' && !Array.isArray(dst[k])) {
        merge(dst[k], v);
      } else {
        dst[k] = v;
      }
    });
    return dst;
  }
  merge(out, raw);

  /* the server owns config_version - keep it out of our object entirely so it
     is impossible to send it back (CONTRACT section 7) */
  delete out.config_version;
  delete out.updated_at;

  if (!Array.isArray(out.levels)) out.levels = [];
  var used = {};
  out.levels = out.levels.filter(function (l) { return l && typeof l === 'object'; }).map(function (l, i) {
    var id = String(l.id || '');
    if (!id || used[id]) { id = 'L' + (i + 1); while (used[id]) id = id + 'x'; }
    used[id] = 1;
    return {
      id: id,
      label: String(l.label === undefined || l.label === null ? '' : l.label).slice(0, 24),
      price: num(l.price, 0),
      enabled: !!l.enabled,
      direction: ['either', 'from_below', 'from_above'].indexOf(l.direction) >= 0 ? l.direction : 'either'
    };
  }).slice(0, MAX_LEVELS);

  if (!Array.isArray(get(out, 'time_window.days', null))) out.time_window.days = [];
  out.time_window.days = out.time_window.days
    .map(function (d) { return int(d, -1); })
    .filter(function (d) { return d >= 0 && d <= 6; });
  out.time_window.start = normHHMM(out.time_window.start) || '08:00';
  out.time_window.end = normHHMM(out.time_window.end) || '17:30';
  if (['server', 'local'].indexOf(out.time_window.timezone) < 0) out.time_window.timezone = 'server';
  if (['pips', 'points', 'price'].indexOf(out.unit) < 0) out.unit = 'pips';
  if (['auto', 'manual'].indexOf(out.pip_size_mode) < 0) out.pip_size_mode = 'auto';
  if (['cross', 'touch'].indexOf(get(out, 'entry.trigger_mode', '')) < 0) out.entry.trigger_mode = 'cross';
  return out;
}

/** Bump the edit sequence without scheduling a write (mid-typing states). */
function touch() { S.editSeq++; }

function onEdit(immediate) {
  S.editSeq++;
  S.saveErrors = [];
  validateClient();
  renderDerived();
  if (!S.cfgLoaded) { renderSaveStatus(); return; }
  if (immediate) flushSave();
  else scheduleSave();
}

function scheduleSave() {
  S.savePending = true;
  if (S.saveTimer) clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(function () { S.saveTimer = null; doSave(); }, SAVE_DEBOUNCE_MS);
  renderSaveStatus();
}

function flushSave() {
  if (S.saveTimer) { clearTimeout(S.saveTimer); S.saveTimer = null; }
  if (!S.cfgLoaded) return;
  doSave();
}

function configPayload() {
  var c = JSON.parse(JSON.stringify(S.cfg || {}));
  delete c.config_version;                       // the server owns it
  c.updated_at = new Date().toISOString().replace(/\.\d+Z$/, 'Z');
  return c;
}

function doSave() {
  if (!S.cfg || !S.cfgLoaded) return;
  if (S.saving) { S.resaveNeeded = true; return; }
  S.saving = true;
  S.savePending = false;
  var seqAtRequest = S.editSeq;
  renderSaveStatus();

  fetch('/api/config', {
    method: 'POST',
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    body: JSON.stringify(configPayload())
  }).then(function (r) {
    return r.json().catch(function () { return {}; }).then(function (j) { return { r: r, j: j }; });
  }).then(function (res) {
    var r = res.r, j = res.j;
    if (r.ok && j && j.ok) {
      S.saveErrors = [];
      S.saveWarnings = Array.isArray(j.warnings) ? j.warnings.map(String) : [];
      var v = num(get(j, 'config.config_version', NaN), NaN);
      if (Number.isFinite(v)) { S.awaitVersion = v; S.syncedAt = 0; }
      /* Adopt the server-normalised config ONLY if the user has not typed
         since this request went out. Otherwise the local object stays the
         truth and the pending save will converge. */
      if (j.config && S.editSeq === seqAtRequest) {
        S.cfg = normalizeConfig(j.config);
        refreshAll();
        syncLevels(false);
      }
    } else {
      S.saveErrors = (j && Array.isArray(j.errors) && j.errors.length)
        ? j.errors.map(String)
        : ['The panel server rejected the write (HTTP ' + r.status + ').'];
      S.saveWarnings = [];
    }
  }).catch(function () {
    S.saveErrors = ['Could not reach the panel bridge, so this change was not saved.'];
  }).then(function () {
    S.saving = false;
    renderSaveStatus();
    renderBanners();
    if (S.resaveNeeded) { S.resaveNeeded = false; scheduleSave(); }
  });
}

function command(cmd, args, label) {
  var name = label || cmd;
  fetch('/api/command', {
    method: 'POST',
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
    body: JSON.stringify({ cmd: cmd, args: args || {} })
  }).then(function (r) {
    return r.json().catch(function () { return {}; }).then(function (j) { return { r: r, j: j }; });
  }).then(function (res) {
    if (res.r.ok && res.j && res.j.ok) {
      var id = num(res.j.id, NaN);
      S.expectCmdId = Number.isFinite(id) ? id : null;
      toast('ok', name + ' sent to the EA');
    } else {
      var err = get(res.j, 'errors.0', null) || get(res.j, 'error', null) || ('HTTP ' + res.r.status);
      toast('err', name + ' was not sent: ' + err);
    }
  }).catch(function () {
    toast('err', name + ' was not sent - the panel bridge is not answering.');
  });
}

/* ------------------------------------------------------------------ 18 */
/* Render orchestration */

/** Anything that depends only on the config (safe to call on every keypress). */
function renderDerived() {
  if (!S.cfg) return;
  renderSide('long');
  renderSide('short');
  renderWindow();
  renderEntry();
  renderBanners();
  renderSaveStatus();
}

/** Everything, after a poll. */
function renderLive() {
  renderHeader();
  if (S.cfg) {
    syncLevels(false);
    renderSide('long');
    renderSide('short');
    renderWindow();
    renderEntry();
  }
  renderPositions();
  renderLog();
  renderMeta();
  renderBanners();
}

/* --- wiring ---------------------------------------------------------- */

function wireStaticBindings() {
  bindPath('long_enabled', 'check', 'long.enabled', { immediate: true });
  bindPath('long_lots', 'number', 'long.lots');
  bindPath('long_tp', 'number', 'long.tp');
  bindPath('long_sl', 'number', 'long.sl');
  bindPath('long_be_enabled', 'check', 'long.be_enabled');
  bindPath('long_be_trigger', 'number', 'long.be_trigger');
  bindPath('long_be_offset', 'number', 'long.be_offset');

  bindPath('short_enabled', 'check', 'short.enabled', { immediate: true });
  bindPath('short_lots', 'number', 'short.lots');
  bindPath('short_tp', 'number', 'short.tp');
  bindPath('short_sl', 'number', 'short.sl');
  bindPath('short_be_enabled', 'check', 'short.be_enabled');
  bindPath('short_be_trigger', 'number', 'short.be_trigger');
  bindPath('short_be_offset', 'number', 'short.be_offset');

  bindPath('tw_enabled', 'check', 'time_window.enabled', { immediate: true });
  bindPath('tw_start', 'time', 'time_window.start');
  bindPath('tw_end', 'time', 'time_window.end');

  bindPath('en_trigger_mode', 'text', 'entry.trigger_mode');
  bindPath('en_touch_tolerance', 'number', 'entry.touch_tolerance');
  bindPath('en_one_shot', 'check', 'entry.one_shot');
  bindPath('en_cooldown_sec', 'number', 'entry.cooldown_sec', { int: true });
  bindPath('en_max_slippage_points', 'number', 'entry.max_slippage_points', { int: true });
  bindPath('en_max_spread_points', 'number', 'entry.max_spread_points', { int: true });
  bindPath('en_max_concurrent', 'number', 'entry.max_concurrent_straddles', { int: true });

  bindPath('cfg_symbol', 'text', 'symbol');
  bindPath('cfg_magic', 'number', 'magic', { int: true });
  bindPath('cfg_unit', 'text', 'unit');
  bindPath('cfg_pip_mode', 'text', 'pip_size_mode');
  bindPath('cfg_pip_manual', 'number', 'pip_size_manual');

  /* timezone radios */
  ['server', 'local'].forEach(function (tz) {
    var el = $('tw_tz_' + tz);
    if (!el) return;
    el.addEventListener('change', function () {
      if (el.checked) { cfgSet('time_window.timezone', tz); onEdit(true); }
    });
    bindings.push({
      el: el, kind: 'radio',
      get: function () { return get(S.cfg, 'time_window.timezone', 'server'); },
      set: function () {},
      opts: {}
    });
  });

  /* day toggles */
  for (var d = 0; d < 7; d++) {
    (function (day) {
      var cb = $('day' + day);
      if (!cb) return;
      cb.addEventListener('change', function () { setDay(day, cb.checked); });
    })(d);
  }
}

/* the radio bindings need their own refresh rule */
var _refreshOne = refreshOne;
refreshOne = function (b) {
  if (!b || !b.el) return;
  if (b.kind === 'radio') {
    if (b.el === document.activeElement) return;
    b.el.checked = (b.el.value === b.get());
    return;
  }
  _refreshOne(b);
};

function wireActions() {
  var bannerHost = $('banners');
  if (bannerHost) bannerHost.addEventListener('click', function (e) {
    var close = e.target.closest('.banner-close');
    if (!close) return;
    var item = close.closest('.banner');
    if (item) { S.dismissedBanners[item.dataset.bannerKey] = true; item.remove(); }
  });
  /* master arm */
  var armBtn = $('armBtn');
  if (armBtn) {
    armBtn.addEventListener('click', function () {
      if (!S.cfg) return;
      var next = !get(S.cfg, 'enabled', false);
      S.cfg.enabled = next;
      if (next) S.panicLatched = false;
      onEdit(true);
      renderHeader();
      toast(next ? 'ok' : 'warn', next
        ? 'Armed. The EA may open positions when a level triggers.'
        : 'Disarmed. No new entries. Open positions are still managed.');
    });
  }

  /* quick add */
  var qa = $('qaInput');
  var qaBtn = $('qaAddBtn');
  function commitQuickAdd() {
    if (!qa) return;
    var prices = parsePrices(qa.value);
    if (!prices.length) { toast('warn', 'No usable price in that text.'); return; }
    var n = addLevels(prices, null);
    if (n) { qa.value = ''; qa.classList.remove('parsed'); updateQaHint(); qa.focus(); }
  }
  function updateQaHint() {
    var hint = $('qaHint');
    if (!hint || !qa) return;
    var prices = parsePrices(qa.value);
    if (!qa.value.trim()) {
      hint.textContent = 'Type a price and press Enter. Paste a list (newline, comma, space or tab separated) to add several at once.';
      qa.classList.remove('parsed');
    } else if (prices.length === 0) {
      hint.textContent = 'No price recognised in that text yet.';
      qa.classList.remove('parsed');
    } else {
      hint.textContent = prices.length === 1
        ? 'Enter adds one level at ' + trimNum(prices[0]) + '.'
        : 'Enter adds ' + prices.length + ' levels: ' + prices.slice(0, 8).map(trimNum).join(', ') +
          (prices.length > 8 ? ' and ' + (prices.length - 8) + ' more' : '') + '.';
      qa.classList.add('parsed');
    }
  }
  if (qa) {
    qa.addEventListener('input', updateQaHint);
    qa.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') { e.preventDefault(); commitQuickAdd(); }
      else if (e.key === 'Escape') { qa.value = ''; updateQaHint(); }
    });
    /* multi-price paste goes straight in - a single-line input would otherwise
       mangle the newlines */
    qa.addEventListener('paste', function (e) {
      var txt = (e.clipboardData || window.clipboardData);
      if (!txt) return;
      var data = txt.getData('text');
      var prices = parsePrices(data);
      if (prices.length > 1) {
        e.preventDefault();
        addLevels(prices, null);
        qa.value = '';
        updateQaHint();
      }
    });
  }
  if (qaBtn) qaBtn.addEventListener('click', commitQuickAdd);

  /* level sort mode */
  ['sortManual', 'sortPrice'].forEach(function (id) {
    var el = $(id);
    if (!el) return;
    el.addEventListener('change', function () {
      if (!el.checked) return;
      S.sortMode = (id === 'sortPrice') ? 'price' : 'manual';
      document.body.dataset.sort = S.sortMode;
      syncLevels(true);
    });
  });

  var ra = $('rearmAllBtn');
  if (ra) ra.addEventListener('click', function () { command('rearm_all', {}, 'Re-arm all'); });

  /* danger zone */
  Confirm.attach($('btnCloseAll'), { label: 'close all' }, function () { command('close_all', {}, 'Close all'); });
  Confirm.attach($('btnCloseLongs'), { label: 'close longs' }, function () { command('close_side', { side: 'long' }, 'Close longs'); });
  Confirm.attach($('btnCloseShorts'), { label: 'close shorts' }, function () { command('close_side', { side: 'short' }, 'Close shorts'); });

  wirePanic();

  /* log dock */
  var lt = $('logToggle');
  if (lt) {
    lt.addEventListener('click', function () {
      var dock = $('logdock');
      var collapsed = dock.dataset.collapsed === '1';
      dock.dataset.collapsed = collapsed ? '0' : '1';
      lt.setAttribute('aria-expanded', collapsed ? 'true' : 'false');
    });
  }
  ['lf_info', 'lf_trade', 'lf_warn', 'lf_error'].forEach(function (id) {
    var el = $(id);
    if (el) el.addEventListener('change', function () { S.sig.log = ''; renderLog(); });
  });
}

function wirePanic() {
  var btn = $('btnPanic');
  var fill = $('panicFill');
  var sub = $('panicSub');
  if (!btn) return;

  var raf = null, start = 0, fired = false, keyArmed = false, keyTimer = null;

  function firePanic() {
    if (!S.cfg) return;
    command('panic', {}, 'PANIC');
    S.cfg.enabled = false;
    S.panicLatched = true;
    onEdit(true);
    renderHeader();
    toast('warn', 'PANIC sent. Everything is being closed and the EA is latched until you arm again.');
  }

  function reset() {
    if (raf) cancelAnimationFrame(raf);
    raf = null;
    btn.classList.remove('holding');
    if (fill) fill.style.width = '0%';
    if (sub) sub.textContent = 'hold 900 ms';
  }

  function step() {
    var p = clamp((Date.now() - start) / PANIC_HOLD_MS, 0, 1);
    if (fill) fill.style.width = (p * 100).toFixed(1) + '%';
    if (sub) sub.textContent = p >= 1 ? 'releasing…' : 'keep holding…';
    if (p >= 1) {
      if (!fired) { fired = true; firePanic(); }
      reset();
      return;
    }
    raf = requestAnimationFrame(step);
  }

  btn.addEventListener('pointerdown', function (e) {
    e.preventDefault();
    btn.focus();
    fired = false;
    start = Date.now();
    btn.classList.add('holding');
    raf = requestAnimationFrame(step);
  });
  ['pointerup', 'pointercancel', 'pointerleave'].forEach(function (ev) {
    btn.addEventListener(ev, function () { if (!fired) reset(); });
  });
  btn.addEventListener('click', function (e) { e.preventDefault(); });

  /* keyboard equivalent: press once to arm, again within 3s to fire */
  btn.addEventListener('keydown', function (e) {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    if (keyArmed) {
      clearTimeout(keyTimer);
      keyArmed = false;
      if (sub) sub.textContent = 'hold 900 ms';
      firePanic();
      return;
    }
    keyArmed = true;
    if (sub) sub.textContent = 'press again to confirm';
    keyTimer = setTimeout(function () {
      keyArmed = false;
      if (sub) sub.textContent = 'hold 900 ms';
    }, CONFIRM_MS);
  });
  btn.addEventListener('blur', function () {
    clearTimeout(keyTimer);
    keyArmed = false;
    reset();
  });
}

function wireGlobalKeys() {
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') Confirm.cancel();
    /* Alt+A toggles the master switch from anywhere; deliberately not a bare
       key so it cannot be hit by accident while typing in a field. */
    if (e.altKey && (e.key === 'a' || e.key === 'A')) {
      e.preventDefault();
      var b = $('armBtn');
      if (b) b.click();
    }
    if (e.altKey && (e.key === 'l' || e.key === 'L')) {
      e.preventDefault();
      var qa = $('qaInput');
      if (qa) { qa.focus(); qa.select(); }
    }
  });
}

function boot() {
  Tip.init();
  wireStaticBindings();
  wireActions();
  wireGlobalKeys();

  document.body.dataset.sort = S.sortMode;

  renderHeader();
  renderPositions();
  renderLog();
  renderMeta();
  renderBanners();

  loadConfig();
  pollLoop();

  /* the PC clock keeps running even when the EA is dead - that is part of how
     the user sees that the server clock has frozen */
  setInterval(function () {
    setText($('clkLocal'), localHHMMSS());
    var l = $('tzLocalNow');
    if (l) l.textContent = localHHMMSS().slice(0, 5);
    renderSaveStatus();
  }, 500);
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
