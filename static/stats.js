// static/stats.js — first-party script of the /stats page (not vendored;
// see VENDOR.md for the third-party bundles beside it).
//
// Loaded as a classic script right after the grid markup and BEFORE the
// page's inline dashboard IIFE, so the top-level declarations here
// (GS_LAYOUT_KEY, grid) are the globals that IIFE reads; the usage section
// is an IIFE of its own and only touches its own elements.
//
// Cache-busting: the page links this file as /static/stats.js?v=<version>;
// /static answers with ETag/Last-Modified and no no-store, so a new build
// is what fetches a new copy.

// --- GridStack: layout presets, explicit edit mode, keyboard move/resize ---
// Layout state persists in localStorage PER PRESET (ops / usage / both); the
// active preset is remembered too. Tiles never move unless "edit layout" is
// on — a hover-paused chart must not turn into an accidental drag. GridStack
// ships no keyboard reorder, so edit mode adds Alt+arrows (move) and
// Alt+Shift+arrows (resize) on the focused tile title.
// uPlot sparklines re-fit on size changes via their own ResizeObservers.
// v6: the usage half gained tiles (headline, stages, hours); a v5 layout
// would remove them on load, so the key moved and v5 layouts are ignored.
const GS_KEY_BASE = 'whisper-stats-layout-v11';
const GS_PRESET_KEY = 'whisper-stats-preset';
const GS_PRESETS = {
  ops:   ['gpu', 'cpu', 'ram', 'process', 'activity', 'errors', 'latency',
          'endpoints', 'models', 'recent'],
  usage: ['headline', 'usage', 'stages', 'hours', 'turnaround', 'failures', 'models', 'recent'],
  both:  null,      // every tile
};
let GS_LAYOUT_KEY = GS_KEY_BASE + ':both';
let gsPreset = 'both';
const grid = GridStack.init({
  column: 12,
  // Responsive: collapse to a single stacked column on phones/narrow tablets.
  // breakpointForWindow keys off the viewport width (not the grid container),
  // and layout:'list' keeps tiles in their saved order when reflowing.
  // (GridStack 10+ has responsive OFF by default, so this is required.)
  columnOpts: {
    breakpointForWindow: true,
    breakpoints: [{ w: 700, c: 1 }],
    layout: 'list',
  },
  // String form so cells track --fs-base (the scale picker). At 100% scale,
  // 4rem = 60px (matches the previous fixed value); at 130% it's ~78px.
  // Saved layouts (column units) preserve unchanged across scale changes.
  cellHeight: '4rem',
  margin: 6,
  // Gravity on: tiles pack upward, so a gap left by a shrunk or removed
  // tile closes by itself and a full-width tile (recent jobs) can be
  // dragged above its neighbours, which move down to make room. The GPU
  // card is one item with swapped inner content, so the old float:true
  // workaround for a hidden second item is no longer needed.
  float: false,
  resizable: { handles: 'se,s,e' },
  draggable: { handle: '.card h3' },
  alwaysShowResizeHandle: false,
  // Read-mostly by default; [✎ edit layout] lifts this.
  staticGrid: true,
});
// Persist on every change (debounced via setTimeout to coalesce rapid drags).
let _saveTimer = null;
function _saveLayout() {
  if (_saveTimer) clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    try { localStorage.setItem(GS_LAYOUT_KEY, JSON.stringify(grid.save(false))); } catch (_) {}
  }, 200);
}
grid.on('change added removed', _saveLayout);

// Show the preset's tiles (re-attaching hidden ones), hide the rest, then
// restore that preset's saved positions. `persist=false` for the own-scope
// path, which forces "both" without overwriting the remembered choice.
function setPreset(name, persist = true) {
  if (!(name in GS_PRESETS)) name = 'both';
  gsPreset = name;
  GS_LAYOUT_KEY = GS_KEY_BASE + ':' + name;
  document.body.classList.toggle('preset-usage', name === 'usage');
  document.body.classList.toggle('preset-ops', name === 'ops');
  const allowed = GS_PRESETS[name];
  grid.batchUpdate();
  document.querySelectorAll('.grid-stack > .grid-stack-item[gs-id]').forEach(el => {
    const id = el.getAttribute('gs-id');
    const show = !allowed || allowed.includes(id);
    const managed = !!el.gridstackNode;
    if (show && !managed) {
      el.hidden = false;
      grid.makeWidget(el);
    } else if (!show && managed) {
      grid.removeWidget(el, false);
      delete el.gridstackNode;
      el.hidden = true;
    }
  });
  grid.batchUpdate(false);
  try {
    const saved = localStorage.getItem(GS_LAYOUT_KEY);
    if (saved) grid.load(JSON.parse(saved), false);
  } catch (_) {}
  if (persist) { try { localStorage.setItem(GS_PRESET_KEY, name); } catch (_) {} }
  const seg = document.getElementById('layout-preset');
  if (seg) seg.querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.v === name));
}
window._fwSetPreset = setPreset;
(() => {
  let initial = 'both';
  try { initial = localStorage.getItem(GS_PRESET_KEY) || 'both'; } catch (_) {}
  setPreset(initial, false);
  const seg = document.getElementById('layout-preset');
  if (seg) seg.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (b && seg.contains(b)) setPreset(b.dataset.v);
  });
})();

// Edit mode: drag handles + resize corners live, tile titles focusable for
// the keyboard, the header button reads "done".
let layoutEditing = false;
const editLayoutBtn = document.getElementById('edit-layout-btn');
function setLayoutEditing(on) {
  layoutEditing = !!on;
  grid.setStatic(!layoutEditing);
  document.body.classList.toggle('layout-edit', layoutEditing);
  document.querySelectorAll('.grid-stack-item .card > h3, .grid-stack-item .card .usage-toolbar > h3, .grid-stack-item .card .rj-toolbar > h3')
    .forEach(h => { if (layoutEditing) h.setAttribute('tabindex', '0'); else h.removeAttribute('tabindex'); });
  if (editLayoutBtn) {
    editLayoutBtn.textContent = layoutEditing ? '✓' : '✎';
    editLayoutBtn.setAttribute('aria-label', layoutEditing ? 'done editing layout' : 'edit layout');
    editLayoutBtn.classList.toggle('active', layoutEditing);
    editLayoutBtn.setAttribute('aria-pressed', layoutEditing ? 'true' : 'false');
  }
  announceLayout(layoutEditing
    ? 'Layout editing on. Drag a tile title, or focus one and use Alt with the arrow keys to move, Alt Shift arrows to resize.'
    : 'Layout saved.');
}
if (editLayoutBtn) editLayoutBtn.addEventListener('click', () => setLayoutEditing(!layoutEditing));
function announceLayout(text) {
  const el = document.getElementById('layout-live');
  if (el) el.textContent = text;
}
document.addEventListener('keydown', (e) => {
  if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
  if ((e.key === 'e' || e.key === 'E') && !e.altKey && !e.ctrlKey && !e.metaKey) {
    setLayoutEditing(!layoutEditing); return;
  }
  if (!layoutEditing || !e.altKey) return;
  const dir = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
  if (!dir) return;
  const item = e.target && e.target.closest && e.target.closest('.grid-stack-item');
  const node = item && item.gridstackNode;
  if (!node) return;
  e.preventDefault();
  const cols = grid.getColumn();
  if (e.shiftKey) {
    const w = Math.max(2, Math.min(cols, (node.w || 1) + dir[0]));
    const h = Math.max(2, (node.h || 1) + dir[1]);
    grid.update(item, { w, h });
    announceLayout(item.getAttribute('gs-id') + ' resized to ' + w + ' by ' + h);
  } else {
    const x = Math.max(0, Math.min(cols - (node.w || 1), (node.x || 0) + dir[0]));
    const y = Math.max(0, (node.y || 0) + dir[1]);
    grid.update(item, { x, y });
    announceLayout(item.getAttribute('gs-id') + ' moved to column ' + (x + 1) + ', row ' + (y + 1));
  }
});

// Header reset-layout button: clears the ACTIVE preset's saved positions.
const resetLayoutBtn = document.getElementById('reset-layout-btn');
if (resetLayoutBtn) {
  resetLayoutBtn.addEventListener('click', () => {
    if (!confirm('Reset the "' + gsPreset + '" tile layout to its defaults?')) return;
    localStorage.removeItem(GS_LAYOUT_KEY);
    location.reload();
  });
}

// --- Usage section (independent of the live SSE dashboard) ------------------
// One document from GET /stats/usage (v2) feeds every usage card: the
// headline strip, the chart (stacked bars by kind, lines otherwise, a dashed
// compare line), the legend / table twin, the leaderboard, the pipeline
// stages card, the busy-hours grid, and the audio/RTF columns of the
// loaded-models table (via window.__statsUsage for the inline renderer).
//
// The scope bar in the page header (range, compare, kind, "with stage" chips,
// click-to-filter chips) is the single filter that scopes all of them, and
// its state is mirrored to the URL so a view can be pasted and reopened.
//
// The chart formats its own axis/tooltip dates inline rather than via the
// shared TIME_HELPERS_JS fmtWhen: its x-values are day*86400 = UTC midnight
// of each bucket's calendar date, so labels read the date with getUTC*
// (correct on any operator timezone) — fmtWhen would apply the local clock,
// which is meaningless for a whole-day bucket.
(() => {
'use strict';
const $ = id => document.getElementById(id);
const chartEl = $('usage-plot');
const tipEl = $('usage-tip');
if (!chartEl || typeof uPlot === 'undefined') return;

// ---------------------------------------------------------------- helpers
// Ported from the frontend's usageDerive.ts so both surfaces bucket, level
// and read the URL the same way.
const KINDS = ['dictation', 'file', 'url', 'text'];
const KIND_LABEL = { dictation: 'dictation', file: 'files', url: 'links', text: 'text' };
// Chart steps of the page's own hues (deeper than the pastel UI tokens so
// stacked segments pass the colour-vision checks). The values are the
// shared --kind-* tokens (web_common.KIND_COLORS = the desktop app's
// --c-chart-* palette), resolved once for canvas/SVG use.
const KIND_FALLBACK = { dictation: '#cf7b00', file: '#3e96ea', url: '#d76797', text: '#6f675c' };
const KIND_COLOR = (() => {
  const cs = getComputedStyle(document.documentElement), out = { unknown: '#6e7681' };
  for (const k in KIND_FALLBACK) out[k] = (cs.getPropertyValue('--kind-' + k) || '').trim() || KIND_FALLBACK[k];
  return out;
})();
// Pipeline stages, in pipeline order, sharing the glyph strip's hues.
// Stage hues are the shared --stage-* tokens (web_common.STAGE_COLORS, the
// desktop app's app.css values), resolved once because uPlot strokes and
// SVG fills need literal colours. The fallbacks only matter if the tokens
// are missing, i.e. never on a served page.
const STAGE_FALLBACK = { downloading: '#d9a45b', separating: '#6faed9', vad: '#a493e8',
                         transcribing: '#93b76f', diarizing: '#c68fb4', translating: '#4dd0c4' };
const STAGE_COLOR = (() => {
  const cs = getComputedStyle(document.documentElement), out = {};
  for (const k in STAGE_FALLBACK) out[k] = (cs.getPropertyValue('--stage-' + k) || '').trim() || STAGE_FALLBACK[k];
  return out;
})();
const STAGE_LABEL = { translating: 'Translation', diarizing: 'Speaker diarization',
                      vad: 'Silence skipping', separating: 'Music separation',
                      transcribing: 'Transcription', downloading: 'Download' };
// The whole pipeline in run order, grouped by what the stage is for. The
// meter denominator per stage = sessions of the kinds it can run on
// (mirrors usage_store.STAGE_ELIGIBLE).
const STAGE_GROUPS = [
  ['Core', ['transcribing']],
  ['Preparation · before the model', ['downloading', 'separating', 'vad']],
  ['Enrichment · after the model', ['diarizing', 'translating']],
];
const STAGE_KINDS = { downloading: ['url'], separating: ['file', 'url'], vad: ['file', 'url'],
                      transcribing: ['file', 'url', 'dictation'], diarizing: ['file', 'url'],
                      translating: ['dictation', 'file', 'url', 'text'] };
const WITH_CHIPS = [['translating', 'translated'], ['diarizing', 'diarized'],
                    ['separating', 'music separated'], ['vad', 'silence skipped']];
const PALETTE = ['#388bfd', '#bb8009', '#2ea043', '#8957e5', '#db61a2',
                 '#56d4dd', '#e3b341', '#ff9bce'];
const OTHERS_COLOR = '#6e7681';
const RANGE_PRESETS = ['7', '30', '90', '180', '365', 'all'];
// The global measure (scope bar `#sb-metric`, Q.metric): every usage card
// counts the same thing. proc_s is "processing", not "GPU": it is wall
// time inside the pipeline on whatever device ran it.
const METRIC_LABEL = { audio_s: 'audio duration', words: 'words', requests: 'requests',
                       errors: 'errors', proc_s: 'processing time', sessions: 'sessions' };
const METRIC_ORDER = ['audio_s', 'words', 'sessions', 'requests', 'proc_s', 'errors'];
// Reading a measure off one busy-hours slot: words sit flat on the slot
// (the desktop app's shape), the others are nested per-kind splits.
function slotMeasure(h, metric) {
  if (metric === 'words') return Number(kindScoped(h) || 0);
  return Number(kindScoped(h[metric]) || 0);
}
const DAY_MS = 86400000;

function remPx(n) {
  const base = parseFloat(getComputedStyle(document.documentElement).fontSize) || 15;
  return Math.round(n * base);
}
const MONO = 'Consolas, "Cascadia Code", "JetBrains Mono", Menlo, ui-monospace, monospace';

function fmtCount(n) {
  n = Number(n || 0);
  if (n >= 1e9) return (n / 1e9).toFixed(1).replace(/\.0$/, '') + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'k';
  return String(Math.round(n));
}
function fmtDur(sec) {
  sec = Number(sec || 0);
  if (sec < 60) return sec.toFixed(sec < 10 ? 1 : 0) + 's';
  if (sec < 3600) return (sec / 60).toFixed(1).replace(/\.0$/, '') + 'm';
  if (sec < 86400) return (sec / 3600).toFixed(1).replace(/\.0$/, '') + 'h';
  return (sec / 86400).toFixed(1).replace(/\.0$/, '') + 'd';
}
function fmtMetric(metric, v) {
  return (metric === 'audio_s' || metric === 'proc_s') ? fmtDur(v) : fmtCount(v);
}
function fmtDate(ts) {
  const d = new Date(ts * 1000), p2 = n => ('0' + n).slice(-2);
  return d.getUTCFullYear() + '.' + p2(d.getUTCMonth() + 1) + '.' + p2(d.getUTCDate());
}
function fmtDay(day) { return fmtDate(day * 86400); }
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
function fmtDayShort(day, withYear) {
  const d = new Date(day * DAY_MS);
  return d.getUTCDate() + ' ' + MON[d.getUTCMonth()] + (withYear ? ' ' + d.getUTCFullYear() : '');
}
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function todayDay() {
  const d = new Date();
  return Math.floor(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()) / DAY_MS);
}
function isoOfDay(day) { return new Date(day * DAY_MS).toISOString().slice(0, 10); }
function dayOfIso(iso) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso || '');
  if (!m) return null;
  return Math.floor(Date.UTC(+m[1], +m[2] - 1, +m[3]) / DAY_MS);
}
// Quartile levels over the ACTIVE cells (the frontend's quantileBreaks): a
// quiet week still gets a full ramp, and one giant day cannot flatten it.
// The desktop app's levelling (usageDerive.ts quantileBreaks): the active
// slots ranked into four equal-count groups, each break the top of its
// group; a duplicate break is clamped up so the steps stay ordered. Same
// algorithm here so a slot is the same shade on both.
function quantileBreaks(values) {
  const a = values.filter(v => Number.isFinite(v) && v > 0).sort((x, y) => x - y);
  const n = a.length;
  if (!n) return null;
  const b = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    const level = 4 - Math.floor(((n - 1 - i) * 4) / n);
    if (level <= 3) b[level - 1] = a[i];
  }
  for (let k = 1; k < 3; k++) if (b[k] < b[k - 1]) b[k] = b[k - 1];
  return b;
}
// Legend copy for the five steps, as the app prints it: 0 · 1–b1 · b1–b2 · b2–b3 · b3+.
function legendRanges(b, fmt) {
  return ['0', fmt(1) + '–' + fmt(b[0]), fmt(b[0]) + '–' + fmt(b[1]), fmt(b[1]) + '–' + fmt(b[2]), fmt(b[2]) + '+'];
}
function levelOf(v, br) {
  if (!br || !(v > 0)) return 0;
  return v <= br[0] ? 1 : v <= br[1] ? 2 : v <= br[2] ? 3 : 4;
}

// ---------------------------------------------------------------- state
// Everything the scope bar + the usage card's own controls decide. Mirrored
// to the URL (parsePageQuery / pageQueryParams), defaults omitted.
// Filters are lists (empty = no filter): `kinds` is OR across job kinds,
// `with` is AND across stages, `users` / `keys` are "one of" as picked in
// the who / keys pickers. `model` is a single click-to-filter chip.
const Q = {
  range: '30', from: null, to: null, compare: 'off',
  kinds: [], with: [], users: [], keys: [], model: null,
  bucket: 'auto', metric: 'audio_s', by: 'kind', rhythm: 'hours',
};
// Display names for picked user / key ids (from the picker or the
// leaderboard), so the filter sentences and the recent-jobs table can say
// who was picked without another round trip.
const pickLabels = { user: {}, key: {} };
const DEFAULTS = JSON.parse(JSON.stringify(Q));
const ownScope = () => document.body.classList.contains('scope-own');

function parsePageQuery(search) {
  const p = new URLSearchParams(search || '');
  const r = p.get('range');
  if (r === 'custom') {
    const f = parseInt(p.get('from'), 10), t = parseInt(p.get('to'), 10);
    if (Number.isFinite(f) && Number.isFinite(t) && f <= t && t - f < 3650) {
      Q.range = 'custom'; Q.from = f; Q.to = t;
    }
  } else if (r && RANGE_PRESETS.includes(r)) Q.range = r;
  const c = p.get('compare'); if (c === 'prev' || c === 'yoy') Q.compare = c;
  const csv = name => (p.get(name) || '').split(',').map(s => s.trim()).filter(Boolean);
  // `kind` / `user` / `key` are the single-valued spellings of older links.
  Q.kinds = csv('kinds').concat(csv('kind')).filter(k => KINDS.includes(k));
  Q.kinds = Array.from(new Set(Q.kinds));
  if (Q.kinds.length === KINDS.length) Q.kinds = [];
  Q.with = csv('with').filter(s => WITH_CHIPS.some(c => c[0] === s));
  Q.users = Array.from(new Set(csv('users').concat(csv('user'))));
  Q.keys = Array.from(new Set(csv('keys').concat(csv('key'))));
  Q.model = p.get('model') || null;
  const b = p.get('bucket'); if (['auto', 'day', 'week', 'month'].includes(b)) Q.bucket = b;
  const rh = p.get('rhythm'); if (['hours', 'days', 'months'].includes(rh)) Q.rhythm = rh;
  const m = p.get('metric'); if (m && METRIC_LABEL[m]) Q.metric = m;
  const by = p.get('by'); if (['kind', 'user', 'key', 'model', 'stage'].includes(by)) Q.by = by;
}
function pageQueryParams() {
  const p = new URLSearchParams();
  if (Q.range === 'custom') { p.set('range', 'custom'); p.set('from', Q.from); p.set('to', Q.to); }
  else if (Q.range !== DEFAULTS.range) p.set('range', Q.range);
  for (const k of ['compare', 'bucket', 'metric', 'by', 'rhythm']) {
    if (Q[k] !== DEFAULTS[k]) p.set(k, Q[k]);
  }
  for (const k of ['kinds', 'with', 'users', 'keys']) if (Q[k].length) p.set(k, Q[k].join(','));
  if (Q.model) p.set('model', Q.model);
  return p;
}
function syncUrl() {
  const q = pageQueryParams().toString();
  const url = location.pathname + (q ? '?' + q : '') + location.hash;
  try { history.replaceState(null, '', url); } catch (_) {}
}
function isFiltered() {
  return Q.range !== '30' || filterCount() > 0 || Q.compare !== 'off';
}
// How many narrowing filters are on (kinds, stages, users, keys, model):
// the card window chips show it so a screenshot still says "filtered".
function filterCount() {
  return (Q.kinds.length ? 1 : 0) + (Q.with.length ? 1 : 0) + (Q.users.length ? 1 : 0)
    + (Q.keys.length ? 1 : 0) + (Q.model ? 1 : 0);
}
const kindsLabel = () => Q.kinds.map(k => KIND_LABEL[k] || k).join(' + ');
const pickLabel = (dim, id) => pickLabels[dim][id] || id;

// ---------------------------------------------------------------- scope bar
function seg(id) { return $(id); }
function setSeg(id, val) {
  const g = seg(id); if (!g) return;
  g.querySelectorAll('button').forEach(b => b.classList.toggle('active', b.dataset.v === String(val)));
}
function onSeg(id, fn) {
  const g = seg(id); if (!g) return;
  g.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b || !g.contains(b)) return;
    fn(b.dataset.v, b);
  });
}
function renderChips() {
  const kinds = $('sb-kind');
  if (kinds) kinds.querySelectorAll('.chip').forEach(c => {
    c.classList.toggle('on', c.dataset.v === 'all' ? Q.kinds.length === 0 : Q.kinds.includes(c.dataset.v));
  });
  const w = $('sb-with');
  if (w) w.querySelectorAll('.chip').forEach(c => {
    c.classList.toggle('on', Q.with.includes(c.dataset.v));
  });
  // Every active narrowing as a sentence: the chip groups and pickers
  // are where you act, this row is where you read what is on.
  const f = $('sb-filters');
  if (f) {
    const chips = [];
    const chip = (dim, text, title) => chips.push('<button type="button" class="chip filter" data-dim="' + dim
      + '" title="' + esc(title || 'remove this filter') + '">' + text + ' <span class="x">×</span></button>');
    if (Q.kinds.length) chip('kinds', 'kind: ' + esc(kindsLabel()), 'any of these kinds · remove');
    if (Q.with.length) chip('with', 'stage: ' + esc(Q.with.map(s => (WITH_CHIPS.find(c => c[0] === s) || [s, s])[1]).join(' + ')), 'ran every one of these stages · remove');
    if (Q.users.length) chip('users', 'user: ' + esc(Q.users.map(u => pickLabel('user', u)).join(', ')), 'one of these users · remove');
    if (Q.keys.length) chip('keys', 'key: ' + esc(Q.keys.map(k => pickLabel('key', k)).join(', ')), 'one of these keys · remove');
    if (Q.model) chip('model', 'model: ' + esc(Q.model));
    if (isFiltered()) chips.push('<button type="button" class="chip clear" id="sb-clear">clear</button>');
    f.innerHTML = chips.length ? chips.join('')
      : '<span class="sb-none">none</span>';
  }
  renderPickerButtons();
  publishFilter();
  const summary = $('sb-summary');
  if (summary && lastDoc) {
    const rg = lastDoc.range || {};
    let s = fmtDay(rg.from) + ' – ' + fmtDay(rg.to) + ' · ' + rg.days + ' d · by ' + lastDoc.bucket;
    if (Q.range === 'all') s += ' · since the first run';
    if (Q.range === 'custom') s += ' · custom';
    if (lastDoc.compare) s += ' · vs ' + (lastDoc.compare.mode === 'yoy' ? 'a year earlier' : 'previous ' + rg.days + ' d');
    if (lastDoc.tz && lastDoc.tz !== 'local') s += ' · ' + lastDoc.tz;
    if (lastDoc.range && lastDoc.range.source === 'jobs') {
      s += ' · stage filters cover the last ' + (rg.jobs_retention_days || 365) + ' days';
    }
    summary.textContent = s;
  }
}
// ---- who / keys pickers: a searchable checklist of the users (keys) with
// usage in the window, ranked by the measure (/stats/pick). Changes apply
// as they are made; "clear" empties the pick. The keys list narrows to the
// picked users' keys.
let _pickOpen = null;
function renderPickerButtons() {
  [['sb-who', 'users', 'user'], ['sb-keys', 'keys', 'key']].forEach(([id, list, word]) => {
    const n = Q[list].length, btn = document.querySelector('#' + id + ' > button');
    if (!btn) return;
    // Update the count span in place: replacing the button's HTML while a
    // click on it is still bubbling detaches the click's target, and the
    // outside-click handler then reads it as "outside" and closes the pop.
    const span = btn.querySelector('.n');
    if (span) span.textContent = n ? n + ' picked' : 'any';
    btn.setAttribute('aria-expanded', _pickOpen === id ? 'true' : 'false');
  });
}
function openPicker(id, dim, list) {
  const wrap = $(id), pop = wrap && wrap.querySelector('.pick-pop');
  if (!pop) return;
  if (_pickOpen === id) { closePickers(); return; }
  closePickers();
  _pickOpen = id;
  pop.hidden = false;
  renderPickerButtons();
  const q = pop.querySelector('input[type=search]'), body = pop.querySelector('.pick-list');
  q.value = '';
  body.innerHTML = '<div class="pick-note">loading…</div>';
  const p = new URLSearchParams();
  if (Q.range === 'custom') { p.set('from', Q.from); p.set('to', Q.to); }
  else if (Q.range === 'all') p.set('all', '1');
  else p.set('days', Q.range);
  p.set('dim', dim); p.set('metric', Q.metric);
  filterParams(p, list);          // rank by the slice, minus this dimension
  if (dim === 'key') p.delete('keys'); else p.delete('users');
  try { p.set('tz', Intl.DateTimeFormat().resolvedOptions().timeZone || ''); } catch (_) {}
  fetch('/stats/pick?' + p.toString(), { cache: 'no-store' })
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(j => {
      if (_pickOpen !== id) return;
      const rows = j.rows || [];
      rows.forEach(r => { pickLabels[dim][r.id] = r.label; });
      // Picked ids that fell out of the window still show, so they can be un-picked.
      Q[list].forEach(pid => { if (!rows.some(r => r.id === pid)) rows.push({ id: pid, label: pickLabel(dim, pid), value: 0, stale: true }); });
      const max = Math.max(1, ...rows.map(r => Number(r.value) || 0));
      const draw = (needle) => {
        const vis = rows.filter(r => !needle || (r.label + ' ' + (r.user_label || '')).toLowerCase().includes(needle));
        body.innerHTML = vis.length ? vis.map(r =>
          '<label class="pick-opt' + (r.stale ? ' stale' : '') + '"><input type="checkbox" data-id="' + esc(r.id) + '"' + (Q[list].includes(r.id) ? ' checked' : '') + '>'
          + '<span class="name">' + esc(r.label) + (r.me ? ' <span class="badge ok">you</span>' : '')
          + (r.user_label ? '<span class="sub">' + esc(r.user_label) + '</span>' : '') + '</span>'
          + '<span class="bar"><i style="width:' + ((Number(r.value) || 0) / max * 100).toFixed(0) + '%"></i></span>'
          + '<span class="v">' + fmtMetric(j.metric, r.value) + '</span></label>').join('')
          : '<div class="pick-note">nothing matches</div>';
        pop.querySelector('.pick-foot span').textContent = Q[list].length + ' of ' + rows.length + ' picked';
      };
      draw('');
      q.oninput = () => draw(q.value.trim().toLowerCase());
      body.onchange = (e) => {
        const cid = e.target && e.target.dataset.id; if (!cid) return;
        Q[list] = e.target.checked ? Q[list].concat([cid]) : Q[list].filter(x => x !== cid);
        pop.querySelector('.pick-foot span').textContent = Q[list].length + ' of ' + rows.length + ' picked';
        renderPickerButtons();
        pickerLoad();
      };
      q.focus();
    })
    .catch(() => { body.innerHTML = '<div class="pick-note">not available for your scope</div>'; });
}
let _pickTimer = null;
function pickerLoad() {     // coalesce a burst of checkbox clicks into one fetch
  clearTimeout(_pickTimer);
  _pickTimer = setTimeout(load, 250);
}
function closePickers() {
  document.querySelectorAll('.pick-pop').forEach(p => { p.hidden = true; });
  _pickOpen = null;
  renderPickerButtons();
}
function wirePickers() {
  [['sb-who', 'user', 'users'], ['sb-keys', 'key', 'keys']].forEach(([id, dim, list]) => {
    const wrap = $(id); if (!wrap) return;
    wrap.querySelector('button').addEventListener('click', () => openPicker(id, dim, list));
    const clr = wrap.querySelector('.pick-clear');
    if (clr) clr.addEventListener('click', () => {
      if (!Q[list].length) return;
      Q[list] = [];
      wrap.querySelectorAll('.pick-list input').forEach(i => { i.checked = false; });
      renderPickerButtons(); load();
    });
  });
  document.addEventListener('click', (e) => {
    if (_pickOpen && e.target.isConnected && !e.target.closest('.picker')) closePickers();
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && _pickOpen) closePickers(); });
}
// The recent-jobs table (inline dashboard) follows the kind and user
// filters: it reads this and re-renders from its last snapshot.
function publishFilter() {
  window.__statsFilter = { kinds: Q.kinds.slice(), users: Q.users.map(u => pickLabel('user', u)) };
  if (typeof window._fwRerenderJobs === 'function') { try { window._fwRerenderJobs(); } catch (_) {} }
}

function wireScopeBar() {
  onSeg('sb-range', (v) => {
    if (v === 'custom') { openCustom(); return; }
    Q.range = v; Q.from = Q.to = null; setSeg('sb-range', v); load();
  });
  onSeg('sb-compare', (v) => { Q.compare = v; setSeg('sb-compare', v); load(); });
  // Kind chips: OR across the chosen kinds; "all" is the empty selection,
  // choosing every kind collapses to it, Alt-click isolates one (like the
  // chart legend).
  const kinds = $('sb-kind');
  if (kinds) kinds.addEventListener('click', (e) => {
    const c = e.target.closest('.chip'); if (!c) return;
    const v = c.dataset.v;
    if (v === 'all') Q.kinds = [];
    else if (e.altKey) Q.kinds = [v];
    else Q.kinds = Q.kinds.includes(v) ? Q.kinds.filter(k => k !== v) : Q.kinds.concat([v]);
    if (Q.kinds.length === KINDS.length) Q.kinds = [];
    setKind();
  });
  const w = $('sb-with');
  if (w) w.addEventListener('click', (e) => {
    const c = e.target.closest('.chip'); if (!c) return;
    const v = c.dataset.v;
    Q.with = Q.with.includes(v) ? Q.with.filter(x => x !== v) : Q.with.concat([v]);
    load();
  });
  const f = $('sb-filters');
  if (f) f.addEventListener('click', (e) => {
    const c = e.target.closest('.chip'); if (!c) return;
    if (c.id === 'sb-clear') {
      Object.assign(Q, { range: '30', from: null, to: null, compare: 'off', kinds: [],
                         with: [], users: [], keys: [], model: null });
      setSeg('sb-range', '30'); setSeg('sb-compare', 'off');
    } else if (c.dataset.dim === 'model') {
      Q.model = null;
    } else if (c.dataset.dim) {
      Q[c.dataset.dim] = [];
    }
    load();
  });
  // Custom span popover: two dates bounded by each other + span shortcuts.
  const pop = $('sb-custom');
  if (pop) {
    pop.addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      if (b.dataset.span) { const [f0, t0] = spanPreset(b.dataset.span); $('sb-from').value = isoOfDay(f0); $('sb-to').value = isoOfDay(t0); validateCustom(); }
      else if (b.id === 'sb-custom-cancel') closeCustom();
      else if (b.id === 'sb-custom-apply') {
        const f0 = dayOfIso($('sb-from').value), t0 = dayOfIso($('sb-to').value);
        if (f0 == null || t0 == null || f0 > t0 || t0 - f0 >= 3650) return;
        Q.range = 'custom'; Q.from = f0; Q.to = t0; setSeg('sb-range', 'custom');
        closeCustom(); load();
      }
    });
    ['sb-from', 'sb-to'].forEach(id => $(id).addEventListener('input', validateCustom));
  }
}
function spanPreset(name) {
  const d = new Date(); const y = d.getUTCFullYear(), m = d.getUTCMonth();
  const day = (yy, mm, dd) => Math.floor(Date.UTC(yy, mm, dd) / DAY_MS);
  const lastOf = (yy, mm) => day(yy, mm + 1, 0);
  const t = todayDay();
  if (name === 'month') return [day(y, m, 1), t];
  if (name === 'lastmonth') return [day(y, m - 1, 1), lastOf(y, m - 1)];
  if (name === 'quarter') return [day(y, m - (m % 3), 1), t];
  if (name === 'year') return [day(y, 0, 1), t];
  if (name === 'lastyear') return [day(y - 1, 0, 1), day(y - 1, 11, 31)];
  return [t - 29, t];
}
function openCustom() {
  const pop = $('sb-custom'); if (!pop) return;
  const t = todayDay();
  $('sb-from').value = isoOfDay(Q.from != null ? Q.from : t - 29);
  $('sb-to').value = isoOfDay(Q.to != null ? Q.to : t);
  pop.classList.remove('hidden'); validateCustom();
}
function closeCustom() { const pop = $('sb-custom'); if (pop) pop.classList.add('hidden'); }
function validateCustom() {
  const f0 = dayOfIso($('sb-from').value), t0 = dayOfIso($('sb-to').value);
  const ok = f0 != null && t0 != null && f0 <= t0 && t0 - f0 < 3650;
  $('sb-custom-apply').disabled = !ok;
  $('sb-custom-note').textContent = ok
    ? (t0 - f0 + 1) + ' days · shown by ' + ((t0 - f0 + 1) <= 120 ? 'day' : (t0 - f0 + 1) <= 730 ? 'week' : 'month')
    : 'pick a start on or before the end (at most 10 years)';
}

// ---------------------------------------------------------------- fetch
let lastDoc = null;
let lastTail = null;
let _seq = 0;
const usageCards = () => document.querySelectorAll('.usage-fed');

function queryString() {
  const p = new URLSearchParams();
  if (Q.range === 'custom') { p.set('from', Q.from); p.set('to', Q.to); }
  else if (Q.range === 'all') p.set('all', '1');
  else p.set('days', Q.range);
  p.set('bucket', Q.bucket); p.set('metric', Q.metric);
  // Own scope has no per-user board; the server refuses by=user there.
  p.set('by', (ownScope() && Q.by === 'user') ? 'key' : Q.by);
  if (Q.compare !== 'off') p.set('compare', Q.compare);
  filterParams(p);
  try { p.set('tz', Intl.DateTimeFormat().resolvedOptions().timeZone || ''); } catch (_) {}
  return '?' + p.toString();
}
// The comma-list filters every usage endpoint understands (/stats/usage,
// /stats/tail, /stats/pick); users are dropped for own scope (403 there).
function filterParams(p, skip) {
  if (Q.kinds.length) p.set('kinds', Q.kinds.join(','));
  if (Q.with.length) p.set('with', Q.with.join(','));
  if (Q.users.length && !ownScope() && skip !== 'users') p.set('users', Q.users.join(','));
  if (Q.keys.length && skip !== 'keys') p.set('keys', Q.keys.join(','));
  return p;
}
function tailQuery() {
  const p = new URLSearchParams();
  if (Q.range === 'custom') { p.set('from', Q.from); p.set('to', Q.to); }
  else if (Q.range === 'all') p.set('all', '1');
  else p.set('days', Q.range);
  if (Q.kinds.length) p.set('kind', Q.kinds.join(','));
  filterParams(p);
  p.delete('with');   // the tail has no stage filter
  try { p.set('tz', Intl.DateTimeFormat().resolvedOptions().timeZone || ''); } catch (_) {}
  return '?' + p.toString();
}
function loadTail(seq) {
  fetch('/stats/tail' + tailQuery(), { cache: 'no-store' })
    .then(r => r.ok ? r.json() : null)
    .then(j => {
      if (seq !== _seq || !j) return;
      lastTail = j;
      // The usage document may still be in flight: its arrival renders
      // everything (renderAll), so only refresh the tail-fed cards when it
      // is already here.
      if (lastDoc) { renderHeadline(); renderTurnaround(); renderFailures(); }
    })
    .catch(err => console.warn('[stats] tail fetch failed', err));
}
// Kinds narrow the document's per-kind splits on the client, but the
// by-user / key / model breakdowns, the leaderboard and the tail are
// aggregated on the server: fetch again (cheap, and one code path).
function setKind() { load(); }
function load() {
  renderChips(); syncUrl();
  usageCards().forEach(el => el.classList.add('updating'));
  const mine = ++_seq;
  loadTail(mine);
  fetch('/stats/usage' + queryString(), { cache: 'no-store' })
    .then(r => {
      if (r.status === 403) {
        // Own scope asked for the per-user board (the only row would be the
        // viewer); the page routes own scope to `key` itself, so this is a
        // hand-edited URL. Say so instead of "unavailable".
        $('usage-board-rows').innerHTML =
          '<tr><td colspan="8" class="empty">— not available for your scope —</td></tr>';
        return null;
      }
      return r.ok ? r.json() : null;
    })
    .then(j => {
      if (mine !== _seq) return;      // stale response — a newer change won
      usageCards().forEach(el => el.classList.remove('updating'));
      if (!j) { showError(); return; }
      hideError();
      lastDoc = j;
      renderChips();
      renderAll();
    })
    .catch(err => {
      console.warn('[stats] usage fetch failed', err);
      if (mine !== _seq) return;
      usageCards().forEach(el => el.classList.remove('updating'));
      showError();
    });
}
function showError() {
  const el = $('usage-error'); if (!el) return;
  el.classList.remove('hidden');
  el.innerHTML = 'usage unavailable' + (lastDoc ? ' — showing the last successful load' : '')
    + ' <button type="button" id="usage-retry">Retry</button>';
  $('usage-retry').addEventListener('click', load);
}
function hideError() { const el = $('usage-error'); if (el) el.classList.add('hidden'); }

// ---------------------------------------------------------------- render
function kindScoped(split) {
  // A per-kind split ({all, dictation, file, url, text}) narrowed to the
  // chosen kinds: `all` when none are chosen, else the chosen ones summed
  // (cells are numbers or {sessions, requests, …} objects).
  if (!split) return null;
  if (!Q.kinds.length) return split.all == null ? null : split.all;
  let out = null;
  for (const k of Q.kinds) {
    const v = split[k];
    if (v == null) continue;
    if (typeof v === 'number') out = (out || 0) + v;
    else { out = out || {}; for (const m in v) out[m] = (out[m] || 0) + (Number(v[m]) || 0); }
  }
  return out;
}
function renderAll() {
  if (!lastDoc) return;
  renderHeadline();
  renderWindowChips();
  // Everything that takes vertical space in the usage card renders BEFORE
  // the chart, which measures whatever height is left; drawing it first
  // would freeze a canvas sized against a one-row "loading" board.
  prepareLines();
  renderLegend();
  renderBoard();
  renderStages();
  renderHours();
  renderTurnaround();
  renderFailures();
  publishModels();
  renderChart();
  renderTable();
}

// ---- turnaround: fixed-edge histogram of end-to-end time (proc + wait),
// the queue-wait share hatched inside each bar, p50 / p95 marked; wait
// p50 / p95 by day as a small line beneath.
function fmtEdge(s) { return s >= 60 ? (s / 60) + 'm' : s + 's'; }
function renderTurnaround() {
  const el = $('turnaround-hist'); if (!el) return;
  const t = lastTail && lastTail.turnaround;
  const tag = $('turnaround-tag'); const note = $('turnaround-note'); const wv = $('turnaround-wait');
  if (!t || !t.n) {
    el.innerHTML = '<div class="usage-empty" style="position:static">No finished jobs in this window.</div>';
    if (tag) tag.textContent = ''; if (note) note.textContent = ''; if (wv) wv.innerHTML = '';
    return;
  }
  if (tag) tag.textContent = 'p50 ' + fmtDur(t.p50) + ' · p95 ' + fmtDur(t.p95) + ' · ' + fmtCount(t.n) + ' jobs';
  // Drawn at the box's pixel size (no preserveAspectRatio stretch, which
  // distorted glyphs and dashes); a ResizeObserver redraws on tile resize.
  const W = Math.max(120, Math.floor(el.clientWidth || 420));
  const H = Math.max(56, Math.floor(el.clientHeight || 110));
  const pl = 4, pb = 16, pt = 14, iw = W - pl * 2, ih = H - pb - pt;
  const n = t.edges_s.length, bw = iw / n, mx = Math.max(1, ...t.counts);
  const labelEvery = bw >= 34 ? 1 : 2;
  let s = '<defs><pattern id="ta-hatch" width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
    + '<line x1="0" y1="0" x2="0" y2="4" stroke="#f0f6fc" stroke-width="1.2" opacity=".55"/></pattern></defs>';
  // Bars stack per job kind in the kind chips' colours (file / url /
  // text / dictation); a kind-filtered window is a single colour.
  const kinds = Object.keys(t.by_kind || {}).sort((a, b) => KINDS.indexOf(a) - KINDS.indexOf(b));
  t.counts.forEach((c, i) => {
    const h = c / mx * ih, y = pt + ih - h, x = pl + i * bw + 1;
    const share = t.wait_share[i] || 0;
    const lbl = fmtEdge(t.edges_s[i]) + (i + 1 < n ? '–' + fmtEdge(t.edges_s[i + 1]) : '+');
    const attrs = ' data-i="' + i + '" data-tip="1"';
    if (kinds.length > 1 && c > 0) {
      let yTop = pt + ih;
      s += '<g' + attrs + '>';
      kinds.forEach(k => {
        const kc = t.by_kind[k][i] || 0; if (!kc) return;
        const kh = kc / c * h; yTop -= kh;
        s += '<rect x="' + x.toFixed(1) + '" y="' + yTop.toFixed(1) + '" width="' + (bw - 2).toFixed(1) + '" height="' + Math.max(0, kh - 1).toFixed(1) + '" fill="' + (KIND_COLOR[k] || KIND_COLOR.unknown) + '" rx="1.5"/>';
      });
      s += '</g>';
    } else {
      s += '<rect' + attrs + ' x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '" width="' + (bw - 2).toFixed(1) + '" height="' + h.toFixed(1) + '" fill="' + (KIND_COLOR[kinds[0]] || '#388bfd') + '" rx="3"/>';
    }
    // an invisible full-height hit target so thin bars and empty buckets still answer hover
    s += '<rect' + attrs + ' x="' + (pl + i * bw).toFixed(1) + '" y="' + pt + '" width="' + bw.toFixed(1) + '" height="' + ih + '" fill="transparent"/>';
    if (share > 0) s += '<rect x="' + x.toFixed(1) + '" y="' + (pt + ih - h * share).toFixed(1) + '" width="' + (bw - 2).toFixed(1) + '" height="' + (h * share).toFixed(1) + '" fill="url(#ta-hatch)" rx="2"/>';
    if (i % labelEvery === 0) s += '<text x="' + (x + bw / 2 - 1).toFixed(1) + '" y="' + (H - 4) + '" text-anchor="middle">' + esc(fmtEdge(t.edges_s[i])) + '</text>';
    // job count: inside the bar when it is tall enough, else just above it
    if (c > 0 && bw >= 16) {
      const inside = h >= 16;
      s += '<text class="' + (inside ? 'cnt in' : 'cnt') + '" x="' + (x + bw / 2 - 1).toFixed(1) + '" y="' + (inside ? y + 11 : y - 3).toFixed(1) + '" text-anchor="middle">' + c + '</text>';
    }
  });
  const xOf = (v) => {
    // position within the bucket that contains v (linear inside the bucket)
    let i = 0; for (let j = 0; j < n; j++) if (v >= t.edges_s[j]) i = j;
    const lo = t.edges_s[i], hi = i + 1 < n ? t.edges_s[i + 1] : lo * 2 || 1;
    return pl + i * bw + Math.min(1, Math.max(0, (v - lo) / Math.max(1e-9, hi - lo))) * bw;
  };
  // p50 / p95 markers: the label sits above the plot so it never lands on
  // a bar; when the two are close, p95's label is nudged right of p50's.
  let lastLabelEnd = -1e9;
  [['p50', t.p50], ['p95', t.p95]].forEach(([k, v]) => {
    const x = xOf(v);
    // label is the quantile name only — the values sit in the card title
    const text = k;
    const tw = text.length * 6.6;
    const lx = Math.min(W - tw, Math.max(x + 3, lastLabelEnd + 6));
    lastLabelEnd = lx + tw;
    s += '<line class="q" x1="' + x.toFixed(1) + '" x2="' + x.toFixed(1) + '" y1="' + (pt - 2) + '" y2="' + (pt + ih) + '"/>'
      + '<text class="q" x="' + lx.toFixed(1) + '" y="' + (pt - 4) + '">' + esc(text) + '</text>';
  });
  el.innerHTML = '<svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '">' + s + '</svg>';
  wireTips(el, '[data-tip]', (target) => {
    const i = Number(target.getAttribute('data-i')); const tt = lastTail && lastTail.turnaround; if (!tt) return '';
    const c = tt.counts[i] || 0;
    const lbl = fmtEdge(tt.edges_s[i]) + (i + 1 < tt.edges_s.length ? '–' + fmtEdge(tt.edges_s[i + 1]) : '+');
    let html = '<div class="tip-date">turnaround ' + esc(lbl) + '</div>';
    const ks = Object.keys(tt.by_kind || {}).sort((a, b) => KINDS.indexOf(a) - KINDS.indexOf(b));
    ks.forEach(k => { const kc = tt.by_kind[k][i] || 0; if (kc) html += tipRow(KIND_COLOR[k] || KIND_COLOR.unknown, KIND_LABEL[k] || k, kc + ' jobs'); });
    html += tipRow(null, 'total', c + ' jobs', ks.length > 1 ? 'tot' : '');
    html += tipRow(null, 'queue wait', Math.round((tt.wait_share[i] || 0) * 100) + ' % of the time', 'cmp');
    return html;
  });
  if (!el._ro && typeof ResizeObserver !== 'undefined') {
    let raf = 0;
    el._ro = new ResizeObserver(() => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        const svg = el.querySelector('svg');
        if (svg && (Math.abs(el.clientWidth - svg.width.baseVal.value) > 1 || Math.abs(el.clientHeight - svg.height.baseVal.value) > 1)) renderTurnaround();
      });
    });
    el._ro.observe(el);
  }
  const w = lastTail.wait || {};
  if (wv) wv.innerHTML = '';
  if (note) {
    const r = lastTail.range || {};
    const anyWait = (w.max || 0) > 0;
    note.innerHTML = 'Hatched = waiting for a GPU slot'
      + (anyWait ? ' (queue wait p50 <b>' + esc(fmtDur(w.p50 || 0)) + '</b> · p95 <b>' + esc(fmtDur(w.p95 || 0)) + '</b> · max ' + esc(fmtDur(w.max || 0)) + ')' : ' (no queueing in this window)')
      + '. Turnaround = processing + wait.'
      + (r.truncated_to_days ? ' Per-job rows cover the last ' + r.truncated_to_days + ' days only.' : '');
  }
}

// ---- failures: by stage · class, terminal + soft-failed, with counts.
function renderFailures() {
  const el = $('failures-list'); if (!el) return;
  const f = lastTail && lastTail.failures;
  const tag = $('failures-tag');
  if (!f) { el.innerHTML = '<span class="empty">— loading —</span>'; return; }
  if (tag) tag.textContent = f.failed + ' of ' + fmtCount(f.jobs) + ' jobs failed'
    + (f.jobs ? ' · ' + (f.failed / f.jobs * 100).toFixed(1) + ' %' : '');
  const rows = [];
  Object.entries(f.by_stage || {}).forEach(([stage, classes]) => {
    Object.entries(classes).forEach(([cls, n]) => rows.push({ stage, cls, n }));
  });
  if (!rows.length) {
    el.innerHTML = '<span class="empty">No failures in this window' + (Q.kinds.length ? ' for ' + esc(kindsLabel()) : '') + '.</span>';
    return;
  }
  rows.sort((a, b) => b.n - a.n || a.stage.localeCompare(b.stage));
  const total = rows.reduce((a, r) => a + r.n, 0);
  const cmp = lastTail.compare && lastTail.compare.errors;
  el.innerHTML = rows.map(r =>
    '<div><span><span class="stage-sw" style="background:' + (STAGE_COLOR[r.stage] || '#6e7681') + '"></span>'
    + esc(r.stage === '(job)' ? 'job' : r.stage) + ' · <span class="cls">' + esc(r.cls) + '</span></span>'
    + '<span class="n"><b>' + r.n + '</b> · ' + Math.round(r.n / total * 100) + ' %</span>'
    + '<div class="m"><i style="width:' + (r.n / total * 100).toFixed(0) + '%"></i></div></div>').join('')
    + (cmp && lastDoc && lastDoc.compare ? '<div class="meta">' + cmp.cur + ' failed jobs vs ' + cmp.prev + ' ' + esc(cmpWord()) + '</div>' : '');
}

// Headline strip: five numbers for the window (+ deltas vs the compare
// window). Words, time saved and wpm stay a desktop-app story: on a server
// page volume is minutes.
function renderHeadline() {
  const el = $('headline-strip'); if (!el) return;
  const tot = kindScoped(lastDoc.totals) || {};
  const cmp = lastDoc.compare ? kindScoped(lastDoc.compare.totals) : null;
  const rtf = tot.audio_s > 0 ? tot.proc_s / tot.audio_s : null;
  const crtf = cmp && cmp.audio_s > 0 ? cmp.proc_s / cmp.audio_s : null;
  const failed = tot.requests > 0 ? tot.errors / tot.requests : 0;
  const cfailed = cmp && cmp.requests > 0 ? cmp.errors / cmp.requests : null;
  const delta = (a, b, inverse) => {
    if (b == null || !lastDoc.compare) return '';
    if (!(b > 0)) return '<span class="delta flat">— vs ' + cmpWord() + '</span>';
    const d = (a - b) / b * 100;
    const arrow = d > 0.5 ? '▲' : d < -0.5 ? '▼' : '—';
    const good = inverse ? d < -0.5 : d > 0.5;
    const bad = inverse ? d > 0.5 : d < -0.5;
    return '<span class="delta ' + (good ? 'good' : bad ? 'bad' : 'flat') + '">'
      + arrow + ' ' + Math.abs(d).toFixed(0) + ' % vs ' + cmpWord() + '</span>';
  };
  const ta = lastTail && lastTail.turnaround;
  const tc = lastTail && lastTail.compare;
  const taDelta = (ta && tc && lastDoc.compare)
    ? '<span class="delta ' + (tc.turnaround_p50.delta < -0.05 ? 'good' : tc.turnaround_p50.delta > 0.05 ? 'bad' : 'flat') + '">'
      + (tc.turnaround_p50.delta > 0.05 ? '▲' : tc.turnaround_p50.delta < -0.05 ? '▼' : '—') + ' '
      + fmtDur(Math.abs(tc.turnaround_p50.delta)) + ' vs ' + cmpWord() + '</span>'
    : '';
  // [label, value, sub, delta, measure the tile stands for (click picks it)]
  const cells = [
    ['audio duration', fmtDur(tot.audio_s), '· ' + fmtCount(tot.words) + ' words',
     delta(tot.audio_s, cmp && cmp.audio_s), Q.metric === 'words' ? 'words' : 'audio_s'],
    ['sessions', fmtCount(tot.sessions), '· ' + fmtCount(tot.requests) + ' requests',
     delta(tot.sessions, cmp && cmp.sessions), Q.metric === 'requests' ? 'requests' : 'sessions'],
    ['failed', (failed * 100).toFixed(1), '% · ' + fmtCount(tot.errors) + ' errors', delta(failed, cfailed, true), 'errors'],
    ['turnaround p50', ta && ta.n ? fmtDur(ta.p50) : '—',
     ta && ta.n ? '· p95 ' + fmtDur(ta.p95) : '', taDelta, null],
    ['RTF', rtf == null ? '—' : rtf.toFixed(2), rtf == null ? '' : '× · ' + (1 / rtf).toFixed(0) + '× live',
     delta(rtf, crtf, true), null],
    ['processing time', fmtDur(tot.proc_s), '', delta(tot.proc_s, cmp && cmp.proc_s), 'proc_s'],
  ];
  el.innerHTML = cells.map(c =>
    '<div class="hl' + (c[4] && c[4] === Q.metric ? ' active' : '') + '"'
    + (c[4] ? ' data-m="' + c[4] + '" tabindex="0" role="button" title="show ' + METRIC_LABEL[c[4]] + ' on every usage card"' : '')
    + '><div class="l">' + esc(c[0]) + '</div><div class="v'
    + (c[0] === 'failed' && failed > 0.02 ? ' warn' : '') + '">' + esc(c[1])
    + '<small>' + esc(c[2]) + '</small></div>' + c[3] + '</div>').join('');
  const tag = $('headline-tag');
  if (tag) tag.textContent = (lastDoc.range ? lastDoc.range.days + ' d' : '')
    + (Q.kinds.length ? ' · ' + kindsLabel() : '');
}
// ---- window chips: every usage card's title row says which window it
// shows (the scope bar's range), pulsing once when that window changes so
// the reader sees which cards follow the control. Ring cards get theirs
// from the inline dashboard (refreshRingChips).
let _winSig = null;
function windowChipHtml() {
  const rg = lastDoc.range || {};
  let html;
  if (Q.range === 'all') html = '<b>all</b> · since ' + esc(fmtDayShort(rg.from, true));
  else {
    const fy = new Date(rg.from * DAY_MS).getUTCFullYear(), ty = new Date(rg.to * DAY_MS).getUTCFullYear();
    const cross = fy !== ty || ty !== new Date().getUTCFullYear();
    html = '<b>' + rg.days + ' d</b> · ' + esc(fmtDayShort(rg.from, cross)) + ' – ' + esc(fmtDayShort(rg.to, cross));
  }
  if (lastDoc.compare) html += ' <em>vs ' + (lastDoc.compare.mode === 'yoy' ? 'last year' : 'previous') + '</em>';
  const n = filterCount();
  if (n) html += ' <span class="fn" title="' + n + ' filter' + (n > 1 ? 's' : '') + ' on: see the FILTERS row">⏷ ' + n + '</span>';
  return html;
}
function renderWindowChips() {
  const sig = [Q.range, Q.from, Q.to, Q.compare, filterCount()].join('|');
  const pulse = _winSig != null && sig !== _winSig;
  _winSig = sig;
  const html = windowChipHtml(), title = ($('sb-summary') || {}).textContent || '';
  document.querySelectorAll('.card .win[data-win="usage"]').forEach(el => {
    el.innerHTML = html; el.title = title + ' — click to jump to the range control';
    if (pulse) { el.classList.remove('pulse'); void el.offsetWidth; el.classList.add('pulse'); }
  });
}
function flashCtl(id) {
  const el = $(id); if (!el) return;
  el.scrollIntoView({ block: 'nearest' });
  el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
}
document.addEventListener('click', (e) => {
  const chip = e.target.closest('.card .win[data-win="usage"]'); if (!chip) return;
  e.preventDefault(); e.stopPropagation();
  if (Q.range === 'custom') { flashCtl('sb-range'); openCustom(); } else flashCtl('sb-range');
});
function cmpWord() { return lastDoc.compare && lastDoc.compare.mode === 'yoy' ? 'last year' : 'prev'; }

// ---- chart
let chart = null;
let curLines = [];         // [{id, label, values, color, others, hatch}] in stack order
let hidden = new Set();    // line ids toggled off in the legend
let stacked = false;
let xs = [];
let cmpTotal = null;       // dashed compare line (sum of the compare lines)

function visibleLines() { return curLines.filter(ln => !hidden.has(ln.id)); }

function prepareLines() {
  const j = lastDoc;
  xs = (j.days || []).map(d => d * 86400);
  stacked = j.by === 'kind';
  let lines = (j.lines || []);
  if (stacked) {
    // Fixed stack order bottom → top, fixed colours (identity, not rank).
    const order = ['file', 'url', 'text', 'dictation', 'unknown'];
    lines = order.map(k => lines.find(ln => ln.id === k)).filter(Boolean)
      .filter(ln => !Q.kinds.length || Q.kinds.includes(ln.id));
    curLines = lines.map(ln => ({
      id: ln.id, label: KIND_LABEL[ln.id] || ln.id, values: ln.values,
      color: KIND_COLOR[ln.id] || OTHERS_COLOR, others: false, me: !!ln.me,
    }));
  } else {
    curLines = lines.map((ln, i) => ({
      id: ln.id, label: ln.label, values: ln.values, others: !!ln.others, me: !!ln.me,
      user_label: ln.user_label,
      color: ln.others ? OTHERS_COLOR
        : (j.by === 'stage' && STAGE_COLOR[ln.id]) || PALETTE[i % PALETTE.length],
    }));
  }
  cmpTotal = null;
  if (j.compare && j.compare.lines && j.compare.lines.length) {
    const n = xs.length;
    const tot = new Array(n).fill(0);
    const keep = new Set(curLines.map(ln => ln.id));
    j.compare.lines.forEach(ln => {
      if (!keep.has(ln.id)) return;
      for (let i = 0; i < n; i++) tot[i] += Number(ln.values[i] || 0);
    });
    cmpTotal = tot;
  }
}
// uPlot has no stacking: feed cumulative sums and draw the TOP of the stack
// first so each lower segment paints over the ones above it. Raw values stay
// on curLines for the tooltip, the table and the legend.
function chartData() {
  const vis = visibleLines();
  const n = xs.length;
  let series;
  if (stacked) {
    const cum = new Array(n).fill(0);
    const rows = vis.map(ln => {
      const out = new Array(n);
      for (let i = 0; i < n; i++) { cum[i] += Number(ln.values[i] || 0); out[i] = cum[i]; }
      return out;
    });
    series = rows.slice().reverse();          // top of the stack first
  } else {
    series = vis.map(ln => ln.values.map(v => v == null ? null : Number(v)));
  }
  if (cmpTotal) series.push(cmpTotal);
  return [xs].concat(series);
}
function seriesSpec() {
  const vis = visibleLines();
  const ordered = stacked ? vis.slice().reverse() : vis;
  const spec = [{ value: (u, ts) => ts == null ? '' : fmtDate(ts) }].concat(
    ordered.map(ln => stacked ? {
      label: ln.label, stroke: ln.color, fill: ln.color, width: 0,
      paths: uPlot.paths.bars({ size: [0.72, 100], align: 0 }),
      points: { show: false },
    } : {
      label: ln.label, stroke: ln.color,
      width: ln.others ? 1.25 : 1.5,
      dash: ln.others ? [4, 3] : undefined,
      fill: vis.length === 1 ? ln.color + '22' : undefined,
      points: { show: vis.length === 1, size: 4 }, spanGaps: true,
    }));
  if (cmpTotal) spec.push({ label: 'compare', stroke: '#8b949e', width: 1.25,
                            dash: [4, 3], points: { show: false }, spanGaps: true });
  return spec;
}
function updateTip(u) {
  const idx = u.cursor.idx;
  if (idx == null || !curLines.length) { tipEl.style.display = 'none'; announce(''); return; }
  const vis = visibleLines();
  let html = '<div class="tip-date">' + fmtDate(xs[idx])
    + (lastDoc.bucket !== 'day' ? ' · ' + lastDoc.bucket : '') + '</div>';
  let total = 0;
  const rows = stacked ? vis.slice().reverse() : vis;
  rows.forEach(ln => {
    const v = Number(ln.values[idx] || 0); total += v;
    html += '<div class="tip-row"><span class="usage-swatch" style="background:' + ln.color + '"></span>'
      + '<span>' + esc(ln.label) + '</span><span class="tip-val">' + fmtMetric(Q.metric, v) + '</span></div>';
  });
  if (rows.length > 1) html += '<div class="tip-row tot"><span>total</span><span class="tip-val">'
    + fmtMetric(Q.metric, total) + '</span></div>';
  if (cmpTotal) {
    const c = cmpTotal[idx] || 0;
    html += '<div class="tip-row cmp"><span>' + cmpWord() + '</span><span class="tip-val">'
      + fmtMetric(Q.metric, c) + (c > 0 ? ' · ' + ((total - c) / c * 100).toFixed(0) + ' %' : '') + '</span></div>';
  }
  const orect = u.over.getBoundingClientRect();
  showTipAt(html, orect.left + u.cursor.left, orect.top + u.cursor.top);
  announce(fmtDate(xs[idx]) + ': ' + rows.map(ln => ln.label + ' ' + fmtMetric(Q.metric, ln.values[idx] || 0)).join(', '));
}
// One tooltip element for every card (usage chart, turnaround bars, busy
// hours cells): fixed-positioned beside the pointer, flipped when it would
// leave the viewport.
function showTipAt(html, cx, cy) {
  tipEl.innerHTML = html;
  tipEl.style.display = 'block';
  const vw = document.documentElement.clientWidth, vh = document.documentElement.clientHeight;
  const tw = tipEl.offsetWidth, th = tipEl.offsetHeight;
  let left = cx + 14;
  if (left + tw + 4 > vw) left = cx - tw - 14;
  let top = cy + 14;
  if (top + th + 4 > vh) top = cy - th - 14;
  tipEl.style.left = Math.max(4, left) + 'px';
  tipEl.style.top = Math.max(4, top) + 'px';
}
function hideTip() { tipEl.style.display = 'none'; }
function tipRow(color, label, val, cls) {
  return '<div class="tip-row' + (cls ? ' ' + cls : '') + '">'
    + (color ? '<span class="usage-swatch" style="background:' + color + '"></span>' : '')
    + '<span>' + esc(label) + '</span><span class="tip-val">' + val + '</span></div>';
}
// Delegated hover + keyboard focus for a card's [data-tip] targets; `build(el)` returns the tooltip html.
function wireTips(host, selector, build) {
  if (!host) return;
  // The builder closes over the current render (measure, kinds…): swap it
  // on every call, wire the listeners once.
  host._tipBuild = build;
  if (host._tips) return;
  host._tips = true;
  const at = (target, e) => {
    const html = host._tipBuild(target); if (!html) { hideTip(); return; }
    if (e && e.clientX != null) showTipAt(html, e.clientX, e.clientY);
    else { const r = target.getBoundingClientRect(); showTipAt(html, r.left + r.width / 2, r.bottom); }
  };
  host.addEventListener('mousemove', (e) => { const t = e.target.closest(selector); if (t && host.contains(t)) at(t, e); else hideTip(); });
  host.addEventListener('mouseleave', hideTip);
  host.addEventListener('focusin', (e) => { const t = e.target.closest(selector); if (t) at(t, null); });
  host.addEventListener('focusout', hideTip);
}
function snapToDataX(u, mLeft, mTop) {
  if (mLeft < 0) return [mLeft, mTop];
  const idx = u.valToIdx(u.posToVal(mLeft, 'x'));
  return [Math.round(u.valToPos(u.data[0][idx], 'x')), mTop];
}
function announce(text) { const a = $('usage-live'); if (a) a.textContent = text; }

function renderChart() {
  prepareLines();
  hidden = new Set([...hidden].filter(id => curLines.some(ln => ln.id === id)));
  if (chart) { chart.destroy(); chart = null; }
  tipEl.style.display = 'none';
  const empty = $('usage-empty');
  if (!curLines.length || !xs.length) {
    if (empty) {
      empty.classList.remove('hidden');
      empty.textContent = 'No ' + METRIC_LABEL[Q.metric] + ' between ' + fmtDay(lastDoc.range.from)
        + ' and ' + fmtDay(lastDoc.range.to) + (Q.with.length ? ' for jobs that ran every chosen stage' : '')
        + '. Widen the range or clear a filter.';
    }
    return;
  }
  if (empty) empty.classList.add('hidden');
  const w = chartEl.clientWidth || 600, h = chartEl.clientHeight || 220;
  const days = lastDoc.range ? lastDoc.range.days : 30;
  chart = new uPlot({
    width: w, height: h,
    padding: [remPx(0.5), remPx(0.6), remPx(0.2), remPx(0.4)],
    legend: { show: false },
    cursor: {
      y: false, drag: { x: false, y: false }, move: snapToDataX,
      points: { size: remPx(0.6), width: remPx(0.13), stroke: '#0d1117',
                fill: (u, si) => { const ordered = stacked ? visibleLines().slice().reverse() : visibleLines();
                                   return (ordered[si - 1] && ordered[si - 1].color) || '#8b949e'; } },
    },
    scales: { x: { time: true }, y: { range: { min: { pad: 0, mode: 1, hard: 0 }, max: { pad: 0.1, mode: 1 } } } },
    hooks: { setCursor: [updateTip] },
    axes: [
      { stroke: '#6e7681', grid: { stroke: '#21262d', width: 1 },
        ticks: { stroke: '#30363d', width: 1, size: 3 },
        font: remPx(0.733) + 'px ' + MONO,
        splits: (u) => {
          // Calendar-grid ticks: a day step from a curated ladder (smallest
          // whose pixel spacing ≥ ~65 px), every tick a UTC midnight.
          const x0 = u.data[0][0], x1 = u.data[0][u.data[0].length - 1];
          const px = u.bbox.width / devicePixelRatio;
          const spanDays = Math.max(1, (x1 - x0) / 86400);
          const ladder = [1, 2, 3, 7, 14, 30, 61, 91, 182, 365];
          let step = ladder[ladder.length - 1];
          for (const s of ladder) { if (px / (spanDays / s) >= 65) { step = s; break; } }
          const out = [];
          for (let t = x0; t <= x1 + 1; t += step * 86400) out.push(t);
          return out;
        },
        values: (u, splits) => splits.map(s => {
          const d = new Date(s * 1000), p2 = n => ('0' + n).slice(-2);
          return days > 400 ? String(d.getUTCFullYear()).slice(2) + '.' + p2(d.getUTCMonth() + 1)
            : p2(d.getUTCMonth() + 1) + '.' + p2(d.getUTCDate());
        }) },
      { stroke: '#6e7681', size: remPx(2.8), gap: 4,
        grid: { stroke: '#21262d', width: 1 },
        ticks: { stroke: '#30363d', width: 1, size: 3 },
        font: remPx(0.733) + 'px ' + MONO,
        values: (u, splits) => splits.map(v => fmtMetric(Q.metric, v)) },
    ],
    series: seriesSpec(),
  }, chartData(), chartEl);
  chart.over.addEventListener('mouseleave', () => { tipEl.style.display = 'none'; announce(''); });
}
function refreshChart() {
  if (!chart) { renderChart(); return; }
  // Legend toggles change the stack itself: rebuild the series set.
  renderChart();
}

let _raf = 0;
new ResizeObserver(() => {
  if (_raf || !chart) return;
  _raf = requestAnimationFrame(() => {
    _raf = 0;
    const cw = chartEl.clientWidth, ch = chartEl.clientHeight;
    if (cw > 0 && ch > 0) chart.setSize({ width: cw, height: ch });
  });
}).observe(chartEl);

// Keyboard scrub: the chart wrapper is focusable; arrows step the cursor
// bucket by bucket, Home/End jump, Escape clears. The aria-live region
// reads the bucket out (see updateTip → announce).
let kbIdx = null;
const wrap = $('usage-chart-wrap');
if (wrap) wrap.addEventListener('keydown', (e) => {
  if (!chart || !xs.length) return;
  const n = xs.length;
  if (e.key === 'Escape') { kbIdx = null; chart.setCursor({ left: -10, top: -10 }); tipEl.style.display = 'none'; return; }
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
  e.preventDefault();
  if (e.key === 'Home') kbIdx = 0;
  else if (e.key === 'End') kbIdx = n - 1;
  else if (kbIdx == null) kbIdx = n - 1;
  else kbIdx = Math.max(0, Math.min(n - 1, kbIdx + (e.key === 'ArrowRight' ? 1 : -1)));
  chart.setCursor({ left: chart.valToPos(xs[kbIdx], 'x'), top: 10 });
});

// ---- legend: one entry per line, click hides/shows, alt-click isolates,
// colours never move with the survivors (identity, not rank).
function renderLegend() {
  const el = $('usage-legend'); if (!el) return;
  const what = METRIC_LABEL[Q.metric] + ' per ' + lastDoc.bucket + ' · '
    + (stacked ? 'stacked by kind' : (lastDoc.by === 'user' || lastDoc.by === 'key')
       ? 'top 8 by ' + lastDoc.by + ', rest folded into “others”' : 'by ' + lastDoc.by);
  el.innerHTML = '<span class="what">' + esc(what) + '</span>' + curLines.map(ln =>
    '<button type="button" data-id="' + esc(ln.id) + '" class="' + (hidden.has(ln.id) ? 'off' : '') + '">'
    + '<span class="usage-swatch" style="background:' + ln.color + '"></span>' + esc(ln.label)
    + (ln.me ? ' <span class="badge ok">you</span>' : '') + '</button>').join('')
    + (cmpTotal ? '<span class="cmp"><i></i>' + esc(cmpWord()) + ' total</span>' : '')
    + '<span class="kb">click a series to hide · alt-click to isolate · ← → scrub · T table</span>';
  el.querySelectorAll('button[data-id]').forEach(b => b.addEventListener('click', (e) => {
    const id = b.dataset.id;
    if (e.altKey || e.metaKey) {
      hidden = new Set(curLines.map(ln => ln.id).filter(x => x !== id));
    } else if (hidden.has(id)) hidden.delete(id); else hidden.add(id);
    if (hidden.size === curLines.length) hidden.clear();   // never blank the chart
    refreshChart(); renderLegend(); renderTable();
  }));
}

// ---- table twin: the same numbers without hovering; ⌘C-friendly.
let tableMode = false;
function renderTable() {
  const el = $('usage-table'); if (!el) return;
  chartEl.parentElement.classList.toggle('hidden', tableMode);
  el.classList.toggle('hidden', !tableMode);
  setSeg('usage-view', tableMode ? 'table' : 'chart');
  if (!tableMode) return;
  const vis = visibleLines();
  let h = '<table class="tbl usage-twin"><thead><tr><th>' + esc(lastDoc.bucket) + '</th>'
    + vis.map(ln => '<th class="num">' + esc(ln.label) + '</th>').join('')
    + (vis.length > 1 ? '<th class="num">total</th>' : '')
    + (cmpTotal ? '<th class="num">' + esc(cmpWord()) + '</th>' : '') + '</tr></thead><tbody>';
  xs.forEach((x, i) => {
    let tot = 0;
    h += '<tr><td>' + fmtDate(x) + '</td>' + vis.map(ln => { const v = Number(ln.values[i] || 0); tot += v;
      return '<td class="num">' + fmtMetric(Q.metric, v) + '</td>'; }).join('')
      + (vis.length > 1 ? '<td class="num"><b>' + fmtMetric(Q.metric, tot) + '</b></td>' : '')
      + (cmpTotal ? '<td class="num dim">' + fmtMetric(Q.metric, cmpTotal[i] || 0) + '</td>' : '') + '</tr>';
  });
  el.innerHTML = h + '</tbody></table>';
}

// ---- leaderboard: the same entities ranked by the metric; rows are
// click-to-filter (user / key / model), kinds toggle the kind chip.
// Column sort for the leaderboard: click a header (again to flip); the
// server's metric order is the default and the rank column follows.
let boardSort = { key: null, dir: -1 };
function renderBoard() {
  const tb = $('usage-board-rows'); if (!tb) return;
  let board = (lastDoc.leaderboard || []).slice();
  if (lastDoc.by === 'user' || lastDoc.by === 'key') board.forEach(r => { if (r.label) pickLabels[lastDoc.by][r.id] = r.label; });
  const by = lastDoc.by;
  const head = $('usage-board-head');
  // The measure leads; the fixed columns skip it so nothing is listed twice.
  const cols = [['label', esc(by), ''], [Q.metric, esc(METRIC_LABEL[Q.metric]), 'num'],
    ...[['sessions', 'sessions', 'num'], ['requests', 'requests', 'num'], ['audio_s', 'audio duration', 'num'],
        ['proc_s', 'processing time', 'num'], ['rtf', 'RTF', 'num'], ['errors', 'err', 'num']]
      .filter(c => c[0] !== Q.metric)];
  if (head) head.innerHTML = '<tr><th class="rank">#</th>' + cols.map(([k, lab, cls]) =>
    '<th class="' + cls + ' sortable' + (boardSort.key === k ? ' on' : '') + '" data-k="' + k + '" title="sort by ' + lab + '">'
    + lab + (boardSort.key === k ? (boardSort.dir < 0 ? ' ▾' : ' ▴') : '') + '</th>').join('') + '</tr>';
  if (head && !head._sortWired) {
    head._sortWired = true;
    head.addEventListener('click', (e) => {
      const th = e.target.closest('th[data-k]'); if (!th) return;
      const k = th.getAttribute('data-k');
      boardSort = boardSort.key === k ? { key: k, dir: -boardSort.dir } : { key: k, dir: k === 'label' ? 1 : -1 };
      renderBoard();
    });
  }
  if (boardSort.key) {
    const val = r => boardSort.key === 'label' ? String(by === 'kind' ? (KIND_LABEL[r.id] || r.label) : (r.label || '')).toLowerCase()
      : boardSort.key === 'rtf' ? (r.rtf == null ? -Infinity : r.rtf) : Number((r.totals || r)[boardSort.key] || 0);
    board.sort((a, b) => { const x = val(a), y = val(b); return (x < y ? -1 : x > y ? 1 : 0) * boardSort.dir; });
  }
  if (!board.length) {
    tb.innerHTML = '<tr><td colspan="9" class="empty">— no usage in this window —</td></tr>';
    return;
  }
  const colorById = {};
  curLines.forEach(ln => { if (!ln.others) colorById[ln.id] = ln.color; });
  const max = Math.max(1, ...board.map(r => Number((r.totals || r)[Q.metric] || 0)));
  tb.innerHTML = board.map((r, i) => {
    const t = r.totals || r;
    const c = colorById[r.id] || (by === 'kind' ? KIND_COLOR[r.id] : null);
    const sw = c ? '<span class="usage-swatch" style="background:' + c + '"></span>' : '';
    const share = '<span class="share" style="width:' + (Number(t[Q.metric] || 0) / max * 60).toFixed(0)
      + 'px;background:' + (c || '#6e7681') + '"></span>';
    const sub = by === 'key' && r.user_label ? '<span class="sub">' + esc(r.user_label) + '</span>' : '';
    const me = r.me ? ' <span class="badge ok">you</span>' : '';
    const label = by === 'kind' ? (KIND_LABEL[r.id] || r.label) : (r.label || '?');
    const rtf = r.rtf == null ? '—' : r.rtf.toFixed(2) + '×';
    const clickable = ['user', 'key', 'model', 'kind'].includes(by) && !(r.id || '').startsWith('(');
    return '<tr' + (clickable ? ' class="pick" tabindex="0" data-id="' + esc(r.id) + '"' : '') + '>'
      + '<td class="rank" data-label="#">' + (i + 1) + '</td>'
      + '<td class="name" data-label="name">' + share + sw + esc(label) + me + sub + '</td>'
      + '<td class="num" data-label="' + esc(METRIC_LABEL[Q.metric]) + '">' + fmtMetric(Q.metric, t[Q.metric]) + '</td>'
      + (Q.metric === 'sessions' ? '' : '<td class="num" data-label="sessions">' + fmtCount(t.sessions) + '</td>')
      + (Q.metric === 'requests' ? '' : '<td class="num" data-label="requests">' + fmtCount(t.requests) + '</td>')
      + (Q.metric === 'audio_s' ? '' : '<td class="num" data-label="audio duration">' + fmtDur(t.audio_s) + '</td>')
      + (Q.metric === 'proc_s' ? '' : '<td class="num" data-label="processing time">' + fmtDur(t.proc_s) + '</td>')
      + '<td class="num" data-label="RTF"><span class="rtf' + (r.rtf > 0.35 ? ' slow' : '') + '">' + rtf + '</span></td>'
      + (Q.metric === 'errors' ? '' : '<td class="num' + (t.errors ? ' err' : '') + '" data-label="err">' + (t.errors ? fmtCount(t.errors) : '—') + '</td>')
      + '</tr>';
  }).join('');
  tb.querySelectorAll('tr.pick').forEach(tr => {
    const pick = () => {
      const id = tr.dataset.id;
      if (by === 'kind') {
        Q.kinds = Q.kinds.includes(id) ? Q.kinds.filter(k => k !== id) : Q.kinds.concat([id]);
        if (Q.kinds.length === KINDS.length) Q.kinds = [];
        setKind(); return;
      }
      if (by === 'user' && ownScope()) return;
      if (by === 'user' || by === 'key') {
        const list = by + 's', row = board.find(r => r.id === id);
        if (row && row.label) pickLabels[by][id] = row.label;
        Q[list] = Q[list].includes(id) ? Q[list].filter(x => x !== id) : Q[list].concat([id]);
      } else {
        Q[by] = Q[by] === id ? null : id;
      }
      load();
    };
    tr.addEventListener('click', pick);
    tr.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
  });
}

// ---- pipeline stages: the whole pipeline in run order, transcription as
// the reference row; unused stages keep their row and say '0 of N'.
function renderStages() {
  const el = $('stages-rows'); if (!el) return;
  const rows = {};
  (lastDoc.stages || []).forEach(s => { rows[s.stage] = s; });
  const tot = lastDoc.totals || {};
  const eligible = s => (STAGE_KINDS[s] || []).reduce((a, k) => a + Number(((tot[k] || {}).sessions) || 0), 0);
  const order = STAGE_GROUPS.flatMap(g => g[1]);
  const totalSecs = order.reduce((a, s) => a + ((rows[s] && rows[s].secs) || 0), 0);
  // The share bar splits the measure across stages: processing → stage
  // seconds, audio → audio the stage saw, sessions/requests → runs; words
  // and errors are not tracked per stage, so those fall back to seconds.
  const shareKey = Q.metric === 'audio_s' ? 'audio_s'
    : (Q.metric === 'sessions' || Q.metric === 'requests') ? 'runs' : 'secs';
  const shareOf = s => Number((rows[s] && rows[s][shareKey]) || 0);
  const shareTot = order.reduce((a, s) => a + shareOf(s), 0);
  const shareFmt = shareKey === 'runs' ? fmtCount : fmtDur;
  const bar = $('stages-bar');
  if (bar) {
    bar.innerHTML = shareTot > 0 ? order.map(s => {
      const v = shareOf(s);
      return v > 0 ? '<span style="flex:' + (v / shareTot) + ';background:' + STAGE_COLOR[s]
        + '" title="' + esc(STAGE_LABEL[s]) + ' ' + (v / shareTot * 100).toFixed(0) + ' % · ' + shareFmt(v) + '"></span>' : '';
    }).join('') : '';
    bar.title = 'share of ' + (shareKey === 'runs' ? 'runs' : shareKey === 'audio_s' ? 'audio duration' : 'processing time') + ' per stage';
  }
  const models = (lastDoc.models || []).slice().sort((a, b) => b.sessions - a.sessions);
  let html = '';
  STAGE_GROUPS.forEach(([title, stages]) => {
    html += '<tr class="grp"><td colspan="6">' + esc(title) + '</td></tr>';
    stages.forEach(s => {
      const r = rows[s], n = eligible(s);
      const sw = '<span class="stage-sw" style="background:' + STAGE_COLOR[s] + '"></span>';
      if (!r || !r.runs) {
        html += '<tr class="dim"><td>' + sw + esc(STAGE_LABEL[s]) + '</td><td class="num">0</td>'
          + '<td class="num"><span class="meter"><i style="width:0"></i></span>0 of ' + fmtCount(n) + '</td>'
          + '<td class="num">—</td><td class="num">—</td><td class="num">—</td></tr>';
        return;
      }
      const of = r.of_runs > 0 ? r.of_runs : n;
      const pct = of > 0 ? Math.round(r.runs / of * 100) : 0;
      const rtf = r.audio_s > 0 ? r.secs / r.audio_s : null;
      let extra = '';
      if (s === 'transcribing' && models.length) {
        extra = models.slice(0, 2).map(m => esc(m.model) + ' · ' + fmtCount(m.sessions) + ' jobs').join(', ')
          + (models.length > 2 ? ', +' + (models.length - 2) + ' more' : '');
      }
      if (s === 'diarizing' && r.speakers_avg != null) extra = r.speakers_avg + ' speakers avg';
      if (s === 'vad' && r.retained_avg != null) extra = ((1 - r.retained_avg) * 100).toFixed(0) + ' % skipped';
      if (s === 'translating') {
        extra = (r.targets || []).slice(0, 4).map(t => esc(t.code.toUpperCase()) + ' '
          + Math.round(t.runs / r.runs * 100) + ' %').join(' · ');
        if (r.kept_original) extra += (extra ? ' · ' : '') + r.kept_original + ' kept original';
      }
      html += '<tr' + (Q.with.includes(s) ? ' class="pinned"' : '') + '>'
        + '<td>' + sw + esc(STAGE_LABEL[s])
        + (Q.with.includes(s) ? ' <span class="badge">filter</span>' : '')
        + (extra ? '<span class="sub">' + extra + '</span>' : '') + '</td>'
        + '<td class="num">' + fmtCount(r.runs) + '</td>'
        + '<td class="num"><span class="meter"><i style="width:' + pct + '%"></i></span>' + pct + ' %</td>'
        + '<td class="num">' + fmtDur(r.audio_s) + '</td>'
        + '<td class="num">' + fmtDur(r.secs) + '</td>'
        + '<td class="num"><span class="rtf' + (rtf > 0.35 ? ' slow' : '') + '">' + (rtf == null ? '—' : rtf.toFixed(2) + '×') + '</span></td>'
        + '</tr>';
    });
  });
  el.innerHTML = html;
  const tag = $('stages-tag');
  if (tag) {
    const tr = (rows.transcribing && rows.transcribing.secs) || 0;
    tag.textContent = totalSecs > 0
      ? fmtDur(totalSecs) + ' processing time' + (tr > 0 ? ' · ' + Math.round(tr / totalSecs * 100) + ' % transcribing' : '')
      : 'no processing time in this window';
  }
}

// ---- busy hours: weekday × hour of the measure summed over the window,
// linear- or quartile-levelled (quantileBreaks), the busiest cell ringed,
// hour-of-day and weekday marginal bars, a phrase for the pattern.
const DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const DOW_LONG = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
// Parts of the day as hour ranges (the title phrase says "Wed 12–18",
// not "Wednesday afternoons").
const DAY_PARTS = [['06–12', 6, 12], ['12–18', 12, 18], ['18–24', 18, 24], ['00–06', 0, 6]];
// How many times each weekday occurs in the window (epoch day 0 was a
// Thursday), so a cell's sum can be said per occurrence.
function weekdayCounts(from, to) {
  const out = [0, 0, 0, 0, 0, 0, 0];
  if (!(from <= to)) return out;
  for (let d = from; d <= to; d++) out[(d + 3) % 7]++;
  return out;
}
// The phrase in the title: the smallest day/part group holding most of the
// measure, most specific first; 'no clear pattern' when nothing does.
function hoursPhrase(cells) {
  const total = cells.reduce((a, v) => a + v, 0);
  if (!(total > 0)) return 'quiet';
  const sum = (days, h0, h1) => days.reduce((a, d) => {
    for (let h = h0; h < h1; h++) a += cells[d * 24 + h]; return a; }, 0);
  let peak = 0; cells.forEach((v, i) => { if (v > cells[peak]) peak = i; });
  // One slot holding half of everything is named as that slot; otherwise
  // the smallest day / hour-range group holding 60 %+ is named, prefixed
  // "mostly" so it reads as a share, not as the peak.
  if (cells[peak] / total >= 0.5) return DOW[Math.floor(peak / 24)] + ' ' + ('0' + (peak % 24)).slice(-2) + '–' + ('0' + (peak % 24 + 1)).slice(-2);
  const groups = [];
  for (let d = 0; d < 7; d++) DAY_PARTS.forEach(([p, a, b]) => groups.push([DOW[d] + ' ' + p, [d], a, b]));
  for (let d = 0; d < 7; d++) groups.push([DOW[d], [d], 0, 24]);
  const wk = [0, 1, 2, 3, 4], we = [5, 6];
  DAY_PARTS.forEach(([p, a, b]) => { groups.push(['Mon–Fri ' + p, wk, a, b]); groups.push(['Sat–Sun ' + p, we, a, b]); });
  DAY_PARTS.forEach(([p, a, b]) => groups.push([p, [0, 1, 2, 3, 4, 5, 6], a, b]));
  groups.push(['Mon–Fri', wk, 0, 24]); groups.push(['Sat–Sun', we, 0, 24]);
  for (const [label, days, a, b] of groups) if (sum(days, a, b) / total >= 0.6) return 'mostly ' + label;
  return '–';
}
// Three rhythms share one renderer: a grid of rows × columns with a
// measure per cell, marginal bars on top (per column) and right (per
// row) each relative to a flat distribution, quantile shading, a peak
// line, a phrase in the title, tooltips split by kind, and compare rows.
//   hours  — weekday × hour of day, from the document's hour slots
//   days   — weekday × week of the window, from the per-day series
//   months — year × month, from the per-day series
const RHYTHMS = ['hours', 'days', 'months'];
const MON_LONG = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                  'August', 'September', 'October', 'November', 'December'];
const dowOfDay = day => (day + 3) % 7;                    // epoch day 0 was a Thursday
const ymOfDay = day => { const d = new Date(day * DAY_MS); return [d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()]; };
const ordinal = n => n + (n % 10 === 1 && n !== 11 ? 'st' : n % 10 === 2 && n !== 12 ? 'nd' : n % 10 === 3 && n !== 13 ? 'rd' : 'th');

// The layout of one rhythm for the current window: how many rows and
// columns, what each is called, and where a source record lands.
function rhythmLayout(mode, rg) {
  const from = rg.from, to = rg.to;
  if (mode === 'days') {
    // Day of month across, hour of day down: the month's shape. A window
    // inside one month shows that month's days; longer windows sum every
    // occurrence of each day of month (31 columns).
    const [fy, fm] = ymOfDay(from), [ty, tm] = ymOfDay(to);
    const oneMonth = fy === ty && fm === tm;
    const cols = oneMonth ? new Date(Date.UTC(fy, fm + 1, 0)).getUTCDate() : 31;
    const occ = new Array(cols).fill(0);
    for (let d = from; d <= to; d++) occ[ymOfDay(d)[2] - 1]++;
    return { rows: 24, cols, rowLabel: r => ('0' + r).slice(-2), rowLong: r => ('0' + r).slice(-2) + '–' + ('0' + (r + 1)).slice(-2),
      colLabel: c => String(c + 1), colLong: c => 'the ' + ordinal(c + 1) + (oneMonth ? ' of ' + MON[fm] : ' of each month'),
      cellName: i => 'the ' + ordinal(i % cols + 1) + (oneMonth ? ' ' + MON[fm] : '') + ' ' + ('0' + Math.floor(i / cols)).slice(-2) + '–' + ('0' + (Math.floor(i / cols) + 1)).slice(-2),
      slotCell: h => (h.dom > cols ? -1 : h.hour * cols + (h.dom - 1)),
      colUnit: 'day of month', rowUnit: 'hour of day', per: 'per slot', colOcc: occ, oneMonth,
      colSteps: [1, 2, 5], rowSteps: [1, 2, 3, 6] };
  }
  if (mode === 'months') {
    const [y0] = ymOfDay(from), [y1] = ymOfDay(to);
    const years = y1 - y0 + 1;
    return { rows: years, cols: 12, rowLabel: r => String(y0 + r), rowLong: r => String(y0 + r),
      colLabel: c => MON[c], colLong: c => MON_LONG[c],
      cellName: i => MON[i % 12] + ' ' + (y0 + Math.floor(i / 12)),
      dayCell: day => { if (day < from || day > to) return -1; const [y, m] = ymOfDay(day); return (y - y0) * 12 + m; },
      colUnit: 'month of year', rowUnit: 'year', per: 'per month', maxRows: years };
  }
  return { rows: 7, cols: 24, rowLabel: r => DOW[r], rowLong: r => DOW_LONG[r] + 's',
    colLabel: c => ('0' + c).slice(-2), colLong: c => ('0' + c).slice(-2) + '–' + ('0' + (c + 1)).slice(-2),
    cellName: i => DOW[Math.floor(i / 24)] + ' ' + ('0' + (i % 24)).slice(-2) + '–' + ('0' + (i % 24 + 1)).slice(-2),
    colUnit: 'hour of day', rowUnit: 'weekday', per: 'per slot', maxRows: 7, colSteps: [1, 2, 3, 6] };
}

// Every column and row carries its label; only when the tile is too
// narrow (or too short) for them all does the density drop, to the
// coarsest step the rhythm allows (every 2nd / 3rd / 6th hour, every 5th
// day) at which neighbouring labels no longer touch. Measured, not
// guessed, and re-run on tile resize.
function fitHoursLabels(el) {
  const steps = el._labelSteps || { col: [1], row: [1] };
  const cell = el.querySelector('i[data-i]'); if (!cell) return;
  const cw = cell.getBoundingClientRect().width, ch = cell.getBoundingClientRect().height;
  // Column labels are grid items stretched to the track, so the text is
  // measured as a Range (scrollWidth never drops below the box); row
  // labels sit centred in their row, so their box height is the text's.
  const fit = (labels, size, allowed, measure) => {
    let need = 0;
    labels.forEach(l => { l.classList.remove('thin'); need = Math.max(need, measure(l)); });
    need += 3;                                        // a hair of air between neighbours
    let step = allowed[allowed.length - 1];
    for (const s of allowed) if (s * size >= need) { step = s; break; }
    labels.forEach((l, i) => l.classList.toggle('thin', i % step !== 0));
  };
  const textW = l => { const r = document.createRange(); r.selectNodeContents(l); return r.getBoundingClientRect().width; };
  fit(Array.from(el.querySelectorAll('.hl')), cw, steps.col, textW);
  fit(Array.from(el.querySelectorAll('.dl')), ch, steps.row, l => l.getBoundingClientRect().height);
}
function renderHours() {
  const el = $('hours-grid'); if (!el) return;
  const mode = RHYTHMS.includes(Q.rhythm) ? Q.rhythm : 'hours';
  const M = Q.metric, ML = METRIC_LABEL[M];
  const fmtM = v => fmtMetric(M, v);
  const fmtAvg = v => (M === 'audio_s' || M === 'proc_s' || v >= 10) ? fmtM(v) : v.toFixed(1);
  const C = M === 'sessions' ? 'proc_s' : 'sessions', CL = METRIC_LABEL[C];
  const fmtC = v => fmtMetric(C, v);
  const rg = lastDoc.range || {};
  const L = rhythmLayout(mode, rg);
  const N = L.rows * L.cols;
  const cells = new Array(N).fill(0), sess = new Array(N).fill(0);
  const byKind = {};
  const addKind = (k, i, pv, sv) => {
    if (!pv && !sv) return;
    const b = byKind[k] || (byKind[k] = [new Array(N).fill(0), new Array(N).fill(0)]);
    b[0][i] += pv; b[1][i] += sv;
  };
  const kindOn = k => !Q.kinds.length || Q.kinds.includes(k);
  // Fill from the source the rhythm reads; `cellOf` maps a compare-window
  // record onto this grid too (days shifted by the window offset).
  const fill = (target, doc, offsetDays) => {
    if (mode === 'hours' || mode === 'days') {
      (mode === 'hours' ? doc.hours : doc.dom_hours || []).forEach(h => {
        const i = mode === 'hours' ? h.dow * 24 + h.hour : L.slotCell(h);
        if (i < 0) return;
        target.c[i] += slotMeasure(h, M); target.s[i] += slotMeasure(h, C);
        if (target.k) KINDS.forEach(k => { if (kindOn(k)) addKind(k, i, Number(M === 'words' ? h[k] || 0 : (h[M] || {})[k] || 0), Number(C === 'words' ? h[k] || 0 : (h[C] || {})[k] || 0)); });
      });
      return;
    }
    (doc.series || []).forEach(p => {
      const i = L.dayCell(p.day + offsetDays); if (i < 0) return;
      const all = kindScoped(p) || {};
      target.c[i] += Number(all[M] || 0); target.s[i] += Number(all[C] || 0);
      if (target.k) KINDS.forEach(k => { if (kindOn(k) && p[k]) addKind(k, i, Number(p[k][M] || 0), Number(p[k][C] || 0)); });
    });
  };
  fill({ c: cells, s: sess, k: true }, lastDoc, 0);
  const cmpSrc = { hours: 'hours', days: 'dom_hours', months: 'series' }[mode];
  const cmp = lastDoc.compare && lastDoc.compare[cmpSrc] ? lastDoc.compare : null;
  const cmpCells = new Array(N).fill(0);
  if (cmp) fill({ c: cmpCells, s: new Array(N).fill(0), k: false }, cmp, mode === 'months' ? (rg.from - cmp.range.from) : 0);
  const br = quantileBreaks(cells);
  let peak = -1, peakV = 0;
  cells.forEach((v, i) => { if (v > peakV) { peakV = v; peak = i; } });
  const colTot = new Array(L.cols).fill(0), rowTot = new Array(L.rows).fill(0);
  cells.forEach((v, i) => { colTot[i % L.cols] += v; rowTot[Math.floor(i / L.cols)] += v; });
  const winSum = cells.reduce((a, v) => a + v, 0), cmpSum = cmpCells.reduce((a, v) => a + v, 0);
  const cmpDelta = (cur, prev) => {
    if (!cmp) return '';
    if (!(prev > 0)) return cur > 0 ? 'new vs ' + cmpWord() : '— vs ' + cmpWord();
    const d = (cur - prev) / prev * 100;
    return (d > 0.5 ? '▲ ' : d < -0.5 ? '▼ ' : '— ') + Math.abs(d).toFixed(0) + ' % vs ' + cmpWord();
  };
  // Occurrences per row in the window (hours: how many Tuesdays; days and
  // months: each cell is its own occurrence).
  const occ = mode === 'hours' ? weekdayCounts(rg.from, rg.to) : new Array(L.rows).fill(1);
  const colOcc = L.colOcc || null;           // days: how often each day of month occurs
  // Marginals: each fills its own track; the dashed tick is where a flat
  // distribution's average falls on that track.
  const colIdx = colTot.map(v => winSum > 0 ? v / (winSum / L.cols) : 0);
  const rowIdx = rowTot.map(v => winSum > 0 ? v / (winSum / L.rows) : 0);
  const cMax = Math.max(1, ...colIdx), rMax = Math.max(1, ...rowIdx);
  const baseC = (100 / cMax).toFixed(1) + '%', baseR = (100 / rMax).toFixed(1) + '%';
  const barLen = (idx, max) => (idx > 0 ? Math.max(4, idx / max * 100) : 0).toFixed(1) + '%';
  const share = v => winSum > 0 ? (v / winSum * 100).toFixed(0) + ' % of the window' : '';

  el.style.gridTemplateColumns = 'auto repeat(' + L.cols + ', 1fr) 2.8rem';   // 0.4rem gap + 2.4rem bar track (.rb margin-left)
  el.style.maxHeight = 'calc(2.8rem + ' + L.rows + ' * ' + ({ hours: 1.8, days: 0.7, months: 2.6 }[mode]) + 'rem + ' + (L.rows + 2) + ' * 2px)';
  el.classList.toggle('dense', L.cols > 30 || L.rows > 12);
  let html = '<span></span>' + colTot.map((v, c) =>
    '<span class="hb' + (colIdx[c] > 1 ? ' hi' : '') + '" data-h="' + c + '" data-tip="h" tabindex="0" role="img" style="--base:' + baseC + '" aria-label="'
    + esc(L.colLong(c)) + ': ' + fmtM(v) + ' ' + ML + ', ' + colIdx[c].toFixed(1) + '× an average ' + L.colUnit + '"><i style="height:' + barLen(colIdx[c], cMax) + '"></i></span>').join('') + '<span></span>';
  html += '<span></span>' + Array.from({ length: L.cols }, (_, c) => '<span class="hl" data-h="' + c + '">' + esc(L.colLabel(c)) + '</span>').join('') + '<span></span>';
  for (let r = 0; r < L.rows; r++) {
    html += '<span class="dl" data-d="' + r + '">' + esc(L.rowLabel(r)) + '</span>';
    for (let c = 0; c < L.cols; c++) {
      const i = r * L.cols + c, v = cells[i];
      const inWin = mode === 'days' ? colOcc[c] > 0 : true;
      const title = L.cellName(i) + (inWin ? ' · ' + fmtM(v) + ' ' + ML + ' · ' + fmtC(sess[i]) + ' ' + CL : ' · not in this window');
      html += '<i tabindex="0" role="img" aria-label="' + esc(title) + '" data-i="' + i + '" data-tip="1"'
        + ' data-l="' + levelOf(v, br) + '"' + (i === peak && peakV > 0 ? ' class="peak"' : '') + (inWin ? '' : ' data-out="1"') + '></i>';
    }
    html += '<span class="rb' + (rowIdx[r] > 1 ? ' hi' : '') + '" data-d="' + r + '" data-tip="d" tabindex="0" role="img" style="--base:' + baseR + '" aria-label="'
      + esc(L.rowLong(r)) + ': ' + fmtM(rowTot[r]) + ' ' + ML + ', ' + rowIdx[r].toFixed(1) + '× an average ' + L.rowUnit + '"><i style="width:' + barLen(rowIdx[r], rMax) + '"></i></span>';
  }
  el.innerHTML = html;
  el._labelSteps = { col: L.colSteps || [1], row: L.rowSteps || [1] };
  fitHoursLabels(el);
  if (!el._ro && typeof ResizeObserver !== 'undefined') {
    let raf = 0;
    el._ro = new ResizeObserver(() => {
      if (raf) return;
      raf = requestAnimationFrame(() => { raf = 0; fitHoursLabels(el); });
    });
    el._ro.observe(el);
  }
  if (!el._hl) {
    el._hl = true;
    const light = (t, on) => {
      if (!t) return;
      let sel;
      if (t.hasAttribute('data-i')) {
        const i = Number(t.getAttribute('data-i')), cols = Number(el.dataset.cols) || 24;
        sel = '[data-d="' + Math.floor(i / cols) + '"], [data-h="' + (i % cols) + '"]';
      } else if (t.hasAttribute('data-h')) sel = '[data-h="' + t.getAttribute('data-h') + '"]';
      else sel = '[data-d="' + t.getAttribute('data-d') + '"]';
      el.querySelectorAll(sel).forEach(x => x.classList.toggle('on', on));
      t.classList.toggle('on', on);
    };
    let cur = null;
    const move = (t) => { if (t === cur) return; light(cur, false); cur = t; light(cur, true); };
    el.addEventListener('mouseover', e => move(e.target.closest('[data-tip]')));
    el.addEventListener('mouseleave', () => move(null));
    el.addEventListener('focusin', e => move(e.target.closest('[data-tip]')));
    el.addEventListener('focusout', () => move(null));
  }
  el.dataset.cols = L.cols;
  // Tooltip rows shared by cells and marginals: per kind, then total
  // (bold, ruled), then the readings, then the compare window (dimmed).
  const kindRows = (pick) => Object.keys(byKind).sort((a, b) => KINDS.indexOf(a) - KINDS.indexOf(b))
    .map(k => [k, pick(byKind[k][0]), pick(byKind[k][1])]).filter(r => r[1] || r[2])
    .map(r => tipRow(KIND_COLOR[r[0]], KIND_LABEL[r[0]] || r[0], fmtM(r[1]) + ' ' + ML + ' · ' + fmtC(r[2]) + ' ' + CL)).join('');
  const body = (head, v, c, rows, extra, pv) => {
    let out = '<div class="tip-date">' + head + '</div>' + rows;
    out += tipRow(null, rows ? 'total' : 'idle', v > 0 ? fmtM(v) + ' ' + ML + ' · ' + fmtC(c) + ' ' + CL : '—', rows ? 'tot' : '');
    if (v > 0) out += extra;
    if (cmp) out += tipRow(null, cmpWord(), fmtM(pv) + ' · ' + cmpDelta(v, pv), 'cmp');
    return out;
  };
  // days: a day of month the window never contains (a 30-day window
  // skips one date; short months have no 29th–31st) is drawn hatched and
  // says so, instead of pretending to be a quiet slot.
  const outOfWindow = (head, c) => '<div class="tip-date">' + head + '</div>'
    + tipRow(null, 'not in this window', 'the ' + ordinal(c + 1) + ' does not occur between ' + fmtDay(rg.from) + ' and ' + fmtDay(rg.to));
  const sumCol = (arr, c) => arr.reduce((a, x, i) => a + (i % L.cols === c ? x : 0), 0);
  const sumRow = (arr, r) => arr.slice(r * L.cols, r * L.cols + L.cols).reduce((a, x) => a + x, 0);
  wireTips(el, '[data-tip]', (target) => {
    const kind = target.getAttribute('data-tip');
    if (kind === 'h') {
      const c = Number(target.getAttribute('data-h')), v = colTot[c];
      if (mode === 'days' && !colOcc[c]) return outOfWindow(esc(L.colLong(c)) + ' · every ' + L.rowUnit, c);
      const n = mode === 'hours' ? occ.reduce((a, x) => a + x, 0) : L.rows;
      let extra = tipRow(null, 'share', share(v)) + tipRow(null, 'vs average', colIdx[c].toFixed(1) + '× an average ' + L.colUnit);
      if (mode === 'hours' && n > 1) extra += tipRow(null, 'per day', '≈ ' + fmtAvg(v / n) + ' ' + ML + ' over ' + n + ' days');
      if (mode === 'days' && colOcc[c] > 1) extra += tipRow(null, 'per month', '≈ ' + fmtAvg(v / colOcc[c]) + ' ' + ML + ' over ' + colOcc[c] + ' months');
      return body(esc(L.colLong(c)) + ' · every ' + L.rowUnit, v, sumCol(sess, c), kindRows(arr => sumCol(arr, c)), extra, sumCol(cmpCells, c));
    }
    if (kind === 'd') {
      const r = Number(target.getAttribute('data-d')), v = rowTot[r];
      let extra = tipRow(null, 'share', share(v)) + tipRow(null, 'vs average', rowIdx[r].toFixed(1) + '× an average ' + L.rowUnit);
      if (mode === 'hours' && occ[r] > 1) extra += tipRow(null, 'per ' + DOW[r], '≈ ' + fmtAvg(v / occ[r]) + ' ' + ML + ' over ' + occ[r] + ' ' + DOW[r]);
      return body(esc(L.rowLong(r)) + ' · all ' + L.colUnit + 's', v, sumRow(sess, r), kindRows(arr => sumRow(arr, r)), extra, sumRow(cmpCells, r));
    }
    const i = Number(target.getAttribute('data-i')), r = Math.floor(i / L.cols);
    if (mode === 'days' && !colOcc[i % L.cols]) return outOfWindow(esc(L.cellName(i)), i % L.cols);
    let extra = '';
    if (mode === 'hours' && occ[r] > 1) extra = tipRow(null, 'per ' + DOW[r], '≈ ' + fmtAvg(cells[i] / occ[r]) + ' ' + ML + ' over ' + occ[r] + ' ' + DOW[r]);
    if (mode === 'days' && colOcc[i % L.cols] > 1) extra = tipRow(null, 'per month', '≈ ' + fmtAvg(cells[i] / colOcc[i % L.cols]) + ' ' + ML + ' over ' + colOcc[i % L.cols] + ' months');
    return body(esc(L.cellName(i)), cells[i], sess[i], kindRows(arr => arr[i]), extra, cmpCells[i]);
  });
  const kindNote = Q.kinds.length ? kindsLabel() + ' only · ' : '';
  const lg = $('hours-legend');
  const cmpNote = cmp ? '<span class="sub">' + cmpDelta(winSum, cmpSum) + '</span>' : '';
  const unitWord = { hours: 'weekday-hour', days: 'day-of-month hour', months: 'month' }[mode];
  const counts = [0, 0, 0, 0, 0];
  if (br) cells.forEach((v, i) => { if (mode !== 'days' || colOcc[i % L.cols] > 0) counts[levelOf(v, br)]++; });
  const steps = br ? legendRanges(br, v => M === 'audio_s' || M === 'proc_s' ? fmtDur(v) : fmtCount(v)).map((txt, l) =>
    '<span class="lv" title="' + counts[l] + ' slot' + (counts[l] === 1 ? '' : 's') + '"><i data-l="' + l + '"></i>' + esc(txt) + '</span>').join('') : '';
  if (lg) lg.innerHTML = br
    ? '<div class="row">' + steps + '<span class="sub" title="the four shades are quarters of the active slots">quartiles · peak ' + fmtM(peakV) + '</span>' + cmpNote + '</div>'
      + '<div class="row"><span title="the bars beside the grid: each ' + L.colUnit + ' (top) and ' + L.rowUnit + ' (right) relative to a flat distribution; the dashed tick is the average"><span class="mg"><i class="b"></i><i class="t"></i></span>side bars vs average</span>'
      + '<span class="what" title="' + esc(ML) + ' per ' + unitWord + '">' + kindNote + esc(ML) + ' · ' + esc(lastDoc.tz === 'local' ? 'server time' : lastDoc.tz) + '</span></div>'
    : '<span class="what">' + kindNote + 'no ' + esc(ML) + ' in this window</span>';
  let phrase = '–';
  if (mode === 'hours') phrase = hoursPhrase(cells);
  else if (mode === 'days') phrase = winSum > 0 ? domPhrase(colTot, rowTot) : 'quiet';
  else phrase = winSum > 0 ? monthPhrase(colTot) : 'quiet';
  const sub = $('hours-sub');
  if (sub) {
    const pr = Math.floor(peak / L.cols);
    sub.innerHTML = peakV > 0
      ? 'Peak ' + esc(L.cellName(peak)) + ' · <b>' + fmtM(peakV) + '</b>'
        + (mode === 'hours' && occ[pr] > 1 ? ' · ≈ ' + fmtAvg(peakV / occ[pr]) + ' per ' + DOW[pr] : '')
        + ' · ' + fmtC(sess[peak]) + ' ' + esc(CL)
        + (phrase !== '–' && !/^quiet$/.test(phrase) ? ' · <span class="mostly" title="the smallest day / hour group holding 60 %+ of the ' + esc(ML) + '">' + esc(phrase) + '</span>' : '')
      : 'No ' + esc(ML) + ' in this window';
    sub.title = peakV > 0 ? fmtM(peakV) + ' ' + ML + ' in the busiest ' + (mode === 'hours' ? 'slot' : mode.slice(0, -1)) + ', ' + L.cellName(peak)
      + (mode === 'hours' && occ[pr] > 1 ? ', summed over ' + occ[pr] + ' ' + DOW_LONG[pr] + 's in the window' : '') : '';
  }
  const title = $('hours-title'); if (title) title.textContent = 'Busy ' + mode;
  setSeg('hours-mode', mode);
}
// Days: the weekday group holding most of the measure (weekdays /
// weekends / one weekday), like hoursPhrase; months: the top month when
// it carries 40 %+, else the top two.
function domPhrase(colTot, rowTot) {
  // Which third of the month, or which part of the day, holds most.
  const total = colTot.reduce((a, v) => a + v, 0);
  const third = n => colTot.slice(n * 10, n === 2 ? 31 : n * 10 + 10).reduce((a, v) => a + v, 0);
  const thirds = [['1st–10th', third(0)], ['11th–20th', third(1)], ['21st–31st', third(2)]];
  for (const [label, v] of thirds) if (v / total >= 0.6) return 'mostly ' + label;
  for (const [label, a, b] of DAY_PARTS) if (rowTot.slice(a, b).reduce((x, v) => x + v, 0) / total >= 0.6) return 'mostly ' + label;
  return '–';
}
function monthPhrase(colTot) {
  const total = colTot.reduce((a, v) => a + v, 0);
  const order = colTot.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
  if (order[0][0] / total >= 0.4) return 'mostly ' + MON[order[0][1]];
  if ((order[0][0] + order[1][0]) / total >= 0.5) return 'mostly ' + MON[order[0][1]] + ' + ' + MON[order[1][1]];
  return '–';
}

// ---- models: audio / RTF per decode model over the window, joined by the
// inline loaded-models renderer (window.__statsUsage.models[name]).
function publishModels() {
  const map = {};
  (lastDoc.models || []).forEach(m => { map[m.model] = m; });
  window.__statsUsage = { models: map, range: lastDoc.range };
  if (typeof window._fwRerenderModels === 'function') {
    try { window._fwRerenderModels(); } catch (_) {}
  }
}

// ---------------------------------------------------------------- wiring
onSeg('usage-bucket', (v) => { Q.bucket = v; setSeg('usage-bucket', v); load(); });
// The measure is global: the scope bar's control, a headline tile, or the
// M key set it, and every usage card re-renders from the same document.
// The document only needs refetching when its ranking (leaderboard) or
// series depend on it, which they do, so load() it is.
function setMetric(v) {
  if (!METRIC_LABEL[v] || v === Q.metric) return;
  Q.metric = v; setSeg('sb-metric', v);
  flashCtl('sb-metric');
  load();
}
onSeg('sb-metric', setMetric);
const hlStrip = $('headline-strip');
if (hlStrip) {
  const pick = (e) => {
    const t = e.target.closest('.hl[data-m]');
    if (!t || !hlStrip.contains(t)) return;
    setMetric(t.dataset.m);
  };
  hlStrip.addEventListener('click', pick);
  hlStrip.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(e); } });
}
onSeg('usage-by', (v) => { Q.by = v; setSeg('usage-by', v); hidden.clear(); load(); });
onSeg('hours-mode', (v) => { Q.rhythm = v; syncUrl(); if (lastDoc) renderHours(); });
onSeg('usage-view', (v) => {
  tableMode = v === 'table'; renderTable();
  if (!tableMode && chart) chart.setSize({ width: chartEl.clientWidth, height: chartEl.clientHeight });
});
document.addEventListener('keydown', (e) => {
  if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
  if (e.key === 't' || e.key === 'T') { tableMode = !tableMode; renderTable(); }
  if (e.key === 'm' || e.key === 'M') setMetric(METRIC_ORDER[(METRIC_ORDER.indexOf(Q.metric) + 1) % METRIC_ORDER.length]);
});

parsePageQuery(location.search);
if (ownScope() && Q.by === 'user') Q.by = 'key';
setSeg('sb-range', Q.range); setSeg('sb-compare', Q.compare);
setSeg('usage-bucket', Q.bucket); setSeg('sb-metric', Q.metric); setSeg('usage-by', Q.by);
wireScopeBar();
wirePickers();
renderChips();
load();
// Own scope is applied by the inline IIFE after the first snapshot; when it
// flips the `by` control to `key`, reload with the corrected query.
window._fwUsageReload = () => {
  if (ownScope()) {
    if (Q.by === 'user') { Q.by = 'key'; setSeg('usage-by', 'key'); }
    Q.users = [];
    const who = $('sb-who'); if (who) who.classList.add('hidden');
  }
  load();
};
})();
