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
const GS_KEY_BASE = 'whisper-stats-layout-v9';
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
    editLayoutBtn.textContent = layoutEditing ? '✓ done' : '✎ edit layout';
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
// The scope bar in the page header (range, compare, kind, "ran" stage chips,
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
// stacked segments pass the colour-vision checks): file = transcribe cyan,
// url = download yellow, text = translate magenta, dictation = dictate green.
const KIND_COLOR = { file: '#388bfd', url: '#bb8009', text: '#8957e5',
                     dictation: '#2ea043', unknown: '#6e7681' };
// Pipeline stages, in pipeline order, sharing the glyph strip's hues.
const STAGE_COLOR = { downloading: '#db61a2', separating: '#bb8009',
                      transcribing: '#388bfd', diarizing: '#2ea043',
                      translating: '#8957e5', vad: '#93b76f' };
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
const METRIC_LABEL = { audio_s: 'audio', words: 'words', requests: 'requests',
                       errors: 'errors', proc_s: 'GPU s', sessions: 'sessions' };
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
function quantileBreaks(values) {
  const act = values.filter(v => v > 0).sort((a, b) => a - b);
  if (!act.length) return null;
  const q = k => act[Math.min(act.length - 1, Math.floor(act.length * k))];
  const br = [q(0.25), q(0.5), q(0.75)];
  // Quartiles of a handful of cells are noise, and equal breaks would map
  // two legend steps to one threshold: fall back to a linear 0..max scale.
  const max = act[act.length - 1];
  if (act.length < 8 || br[0] === br[1] || br[1] === br[2]) return [max / 4, max / 2, max * 3 / 4];
  return br;
}
function levelOf(v, br) {
  if (!br || !(v > 0)) return 0;
  return v <= br[0] ? 1 : v <= br[1] ? 2 : v <= br[2] ? 3 : 4;
}

// ---------------------------------------------------------------- state
// Everything the scope bar + the usage card's own controls decide. Mirrored
// to the URL (parsePageQuery / pageQueryParams), defaults omitted.
const Q = {
  range: '30', from: null, to: null, compare: 'off',
  kind: 'all', with: [], user: null, key: null, model: null,
  bucket: 'auto', metric: 'audio_s', by: 'kind',
};
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
  const k = p.get('kind'); if (k && KINDS.includes(k)) Q.kind = k;
  const w = (p.get('with') || '').split(',').map(s => s.trim()).filter(Boolean);
  Q.with = w.filter(s => WITH_CHIPS.some(c => c[0] === s));
  Q.user = p.get('user') || null; Q.key = p.get('key') || null;
  Q.model = p.get('model') || null;
  const b = p.get('bucket'); if (['auto', 'day', 'week', 'month'].includes(b)) Q.bucket = b;
  const m = p.get('metric'); if (m && METRIC_LABEL[m]) Q.metric = m;
  const by = p.get('by'); if (['kind', 'user', 'key', 'model', 'stage'].includes(by)) Q.by = by;
}
function pageQueryParams() {
  const p = new URLSearchParams();
  if (Q.range === 'custom') { p.set('range', 'custom'); p.set('from', Q.from); p.set('to', Q.to); }
  else if (Q.range !== DEFAULTS.range) p.set('range', Q.range);
  for (const k of ['compare', 'kind', 'bucket', 'metric', 'by']) {
    if (Q[k] !== DEFAULTS[k]) p.set(k, Q[k]);
  }
  if (Q.with.length) p.set('with', Q.with.join(','));
  for (const k of ['user', 'key', 'model']) if (Q[k]) p.set(k, Q[k]);
  return p;
}
function syncUrl() {
  const q = pageQueryParams().toString();
  const url = location.pathname + (q ? '?' + q : '') + location.hash;
  try { history.replaceState(null, '', url); } catch (_) {}
}
function isFiltered() {
  return Q.range !== '30' || Q.kind !== 'all' || Q.with.length > 0
    || !!Q.user || !!Q.key || !!Q.model || Q.compare !== 'off';
}

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
    c.classList.toggle('on', c.dataset.v === Q.kind);
  });
  const w = $('sb-with');
  if (w) w.querySelectorAll('.chip').forEach(c => {
    c.classList.toggle('on', Q.with.includes(c.dataset.v));
  });
  const f = $('sb-filters');
  if (f) {
    const chips = [];
    for (const k of ['user', 'key', 'model']) {
      if (Q[k]) chips.push('<button type="button" class="chip filter" data-dim="' + k + '">'
        + k + ': ' + esc(Q[k]) + ' <span class="x">×</span></button>');
    }
    if (isFiltered()) chips.push('<button type="button" class="chip clear" id="sb-clear">clear</button>');
    f.innerHTML = chips.length ? chips.join('')
      : '<span class="sb-none">none · click a row in the leaderboard</span>';
  }
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
function wireScopeBar() {
  onSeg('sb-range', (v) => {
    if (v === 'custom') { openCustom(); return; }
    Q.range = v; Q.from = Q.to = null; setSeg('sb-range', v); load();
  });
  onSeg('sb-compare', (v) => { Q.compare = v; setSeg('sb-compare', v); load(); });
  const kinds = $('sb-kind');
  if (kinds) kinds.addEventListener('click', (e) => {
    const c = e.target.closest('.chip'); if (!c) return;
    Q.kind = c.dataset.v; setKind();
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
      Object.assign(Q, { range: '30', from: null, to: null, compare: 'off', kind: 'all',
                         with: [], user: null, key: null, model: null });
      setSeg('sb-range', '30'); setSeg('sb-compare', 'off');
    } else if (c.dataset.dim) {
      Q[c.dataset.dim] = null;
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
  if (Q.with.length) p.set('with', Q.with.join(','));
  if (Q.key) p.set('key', Q.key);
  if (Q.user) p.set('user', Q.user);
  try { p.set('tz', Intl.DateTimeFormat().resolvedOptions().timeZone || ''); } catch (_) {}
  return '?' + p.toString();
}
function tailQuery() {
  const p = new URLSearchParams();
  if (Q.range === 'custom') { p.set('from', Q.from); p.set('to', Q.to); }
  else if (Q.range === 'all') p.set('all', '1');
  else p.set('days', Q.range);
  if (Q.kind !== 'all') p.set('kind', Q.kind);
  if (Q.key) p.set('key', Q.key);
  if (Q.user) p.set('user', Q.user);
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
// Kind is a client-side split of the usage document, but the tail
// (turnaround, failures) is computed per kind on the server: re-fetch it.
function setKind() {
  renderChips(); syncUrl(); renderAll();
  loadTail(_seq);
}
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
  // A per-kind split ({all, dictation, file, url, text}) narrowed to Q.kind.
  if (!split) return null;
  return split[Q.kind === 'all' ? 'all' : Q.kind] || null;
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
    el.innerHTML = '<span class="empty">No failures in this window' + (Q.kind !== 'all' ? ' for ' + KIND_LABEL[Q.kind] : '') + '.</span>';
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
  const cells = [
    ['audio', fmtDur(tot.audio_s), '', delta(tot.audio_s, cmp && cmp.audio_s)],
    ['sessions', fmtCount(tot.sessions), '· ' + fmtCount(tot.requests) + ' requests',
     delta(tot.sessions, cmp && cmp.sessions)],
    ['failed', (failed * 100).toFixed(1), '%', delta(failed, cfailed, true)],
    ['turnaround p50', ta && ta.n ? fmtDur(ta.p50) : '—',
     ta && ta.n ? '· p95 ' + fmtDur(ta.p95) : '', taDelta],
    ['RTF', rtf == null ? '—' : rtf.toFixed(2), rtf == null ? '' : '× · ' + (1 / rtf).toFixed(0) + '× live',
     delta(rtf, crtf, true)],
    ['GPU seconds', fmtDur(tot.proc_s), '', delta(tot.proc_s, cmp && cmp.proc_s)],
  ];
  el.innerHTML = cells.map(c =>
    '<div class="hl"><div class="l">' + esc(c[0]) + '</div><div class="v'
    + (c[0] === 'failed' && failed > 0.02 ? ' warn' : '') + '">' + esc(c[1])
    + '<small>' + esc(c[2]) + '</small></div>' + c[3] + '</div>').join('');
  const tag = $('headline-tag');
  if (tag) tag.textContent = (lastDoc.range ? lastDoc.range.days + ' d' : '')
    + (Q.kind !== 'all' ? ' · ' + KIND_LABEL[Q.kind] : '');
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
  return html;
}
function renderWindowChips() {
  const sig = [Q.range, Q.from, Q.to, Q.compare].join('|');
  const pulse = _winSig != null && sig !== _winSig;
  _winSig = sig;
  const html = windowChipHtml(), title = ($('sb-summary') || {}).textContent || '';
  document.querySelectorAll('.card h3 .win[data-win="usage"]').forEach(el => {
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
  const chip = e.target.closest('.card h3 .win[data-win="usage"]'); if (!chip) return;
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
      .filter(ln => Q.kind === 'all' || ln.id === Q.kind);
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
  if (!host || host._tips) return;
  host._tips = true;
  const at = (target, e) => {
    const html = build(target); if (!html) { hideTip(); return; }
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
  const btn = $('usage-table-btn'); if (btn) btn.classList.toggle('active', tableMode);
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
function renderBoard() {
  const tb = $('usage-board-rows'); if (!tb) return;
  const board = lastDoc.leaderboard || [];
  const by = lastDoc.by;
  const head = $('usage-board-head');
  if (head) head.innerHTML = '<tr><th class="rank">#</th><th>' + esc(by) + '</th>'
    + '<th class="num">' + esc(METRIC_LABEL[Q.metric]) + '</th><th class="num">sessions</th>'
    + '<th class="num">requests</th><th class="num">audio</th><th class="num">GPU s</th>'
    + '<th class="num">RTF</th><th class="num">err</th></tr>';
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
      + '<td class="num" data-label="sessions">' + fmtCount(t.sessions) + '</td>'
      + '<td class="num" data-label="requests">' + fmtCount(t.requests) + '</td>'
      + '<td class="num" data-label="audio">' + fmtDur(t.audio_s) + '</td>'
      + '<td class="num" data-label="GPU s">' + fmtDur(t.proc_s) + '</td>'
      + '<td class="num" data-label="RTF"><span class="rtf' + (r.rtf > 0.35 ? ' slow' : '') + '">' + rtf + '</span></td>'
      + '<td class="num' + (t.errors ? ' err' : '') + '" data-label="err">' + (t.errors ? fmtCount(t.errors) : '—') + '</td>'
      + '</tr>';
  }).join('');
  tb.querySelectorAll('tr.pick').forEach(tr => {
    const pick = () => {
      const id = tr.dataset.id;
      if (by === 'kind') { Q.kind = Q.kind === id ? 'all' : id; setKind(); return; }
      if (by === 'user' && ownScope()) return;
      Q[by] = Q[by] === id ? null : id;
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
  const bar = $('stages-bar');
  if (bar) bar.innerHTML = totalSecs > 0 ? order.map(s => {
    const secs = (rows[s] && rows[s].secs) || 0;
    return secs > 0 ? '<span style="flex:' + (secs / totalSecs) + ';background:' + STAGE_COLOR[s]
      + '" title="' + esc(STAGE_LABEL[s]) + ' ' + (secs / totalSecs * 100).toFixed(0) + ' %"></span>' : '';
  }).join('') : '';
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
      ? fmtDur(totalSecs) + ' GPU' + (tr > 0 ? ' · ' + Math.round(tr / totalSecs * 100) + ' % transcribing' : '')
      : 'no GPU time in this window';
  }
}

// ---- busy hours: weekday × hour of GPU seconds summed over the window,
// linear- or quartile-levelled (quantileBreaks), the busiest cell ringed,
// hour-of-day and weekday marginal bars, a phrase for the pattern.
const DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const DOW_LONG = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
const DAY_PARTS = [['mornings', 6, 12], ['afternoons', 12, 18], ['evenings', 18, 24], ['nights', 0, 6]];
// How many times each weekday occurs in the window (epoch day 0 was a
// Thursday), so a cell's sum can be said per occurrence.
function weekdayCounts(from, to) {
  const out = [0, 0, 0, 0, 0, 0, 0];
  if (!(from <= to)) return out;
  for (let d = from; d <= to; d++) out[(d + 3) % 7]++;
  return out;
}
// The phrase in the title: the smallest day/part group holding most of the
// GPU time, most specific first; 'no clear pattern' when nothing does.
function hoursPhrase(cells) {
  const total = cells.reduce((a, v) => a + v, 0);
  if (!(total > 0)) return 'quiet';
  const sum = (days, h0, h1) => days.reduce((a, d) => {
    for (let h = h0; h < h1; h++) a += cells[d * 24 + h]; return a; }, 0);
  let peak = 0; cells.forEach((v, i) => { if (v > cells[peak]) peak = i; });
  if (cells[peak] / total >= 0.5) return DOW[Math.floor(peak / 24)] + ' ' + ('0' + (peak % 24)).slice(-2) + '–' + ('0' + (peak % 24 + 1)).slice(-2);
  const groups = [];
  for (let d = 0; d < 7; d++) DAY_PARTS.forEach(([p, a, b]) => groups.push([DOW_LONG[d] + ' ' + p, [d], a, b]));
  for (let d = 0; d < 7; d++) groups.push([DOW_LONG[d] + 's', [d], 0, 24]);
  const wk = [0, 1, 2, 3, 4], we = [5, 6];
  DAY_PARTS.forEach(([p, a, b]) => { groups.push(['weekday ' + p, wk, a, b]); groups.push(['weekend ' + p, we, a, b]); });
  DAY_PARTS.forEach(([p, a, b]) => groups.push([p, [0, 1, 2, 3, 4, 5, 6], a, b]));
  groups.push(['weekdays', wk, 0, 24]); groups.push(['weekends', we, 0, 24]);
  for (const [label, days, a, b] of groups) if (sum(days, a, b) / total >= 0.6) return label;
  return 'no clear pattern';
}
function renderHours() {
  const el = $('hours-grid'); if (!el) return;
  const cells = new Array(7 * 24).fill(0);
  const sess = new Array(7 * 24).fill(0);
  const byKind = {};   // kind → [proc_s per cell, sessions per cell]
  (lastDoc.hours || []).forEach(h => {
    const v = h.proc_s ? kindScoped(h.proc_s) : 0;
    const s = h.sessions ? kindScoped(h.sessions) : 0;
    const i = h.dow * 24 + h.hour;
    cells[i] += Number(v || 0);
    sess[i] += Number(s || 0);
    KINDS.forEach(k => {
      if (Q.kind !== 'all' && Q.kind !== k) return;
      const pv = Number((h.proc_s || {})[k] || 0), sv = Number((h.sessions || {})[k] || 0);
      if (!pv && !sv) return;
      const b = byKind[k] || (byKind[k] = [new Array(7 * 24).fill(0), new Array(7 * 24).fill(0)]);
      b[0][i] += pv; b[1][i] += sv;
    });
  });
  const br = quantileBreaks(cells);
  let peak = -1, peakV = 0;
  cells.forEach((v, i) => { if (v > peakV) { peakV = v; peak = i; } });
  const hourTot = new Array(24).fill(0), dayTot = new Array(7).fill(0);
  cells.forEach((v, i) => { hourTot[i % 24] += v; dayTot[Math.floor(i / 24)] += v; });
  const hm = Math.max(...hourTot), dm = Math.max(...dayTot);
  const rg = lastDoc.range || {};
  const occ = weekdayCounts(rg.from, rg.to);
  const slot = i => DOW[Math.floor(i / 24)] + ' ' + ('0' + (i % 24)).slice(-2) + '–' + ('0' + (i % 24 + 1)).slice(-2);
  // marginal row: GPU seconds per hour of day (summed over the weekdays)
  let html = '<span></span>' + hourTot.map(v =>
    '<span class="hb"><i style="height:' + (v > 0 ? Math.max(12, v / hm * 100) : 0) + '%"></i></span>').join('') + '<span></span>';
  html += '<span></span>' + Array.from({ length: 24 }, (_, h) =>
    '<span class="hl">' + (h % 6 === 0 ? ('0' + h).slice(-2) : '') + '</span>').join('') + '<span></span>';
  for (let d = 0; d < 7; d++) {
    html += '<span class="dl">' + DOW[d] + '</span>';
    for (let h = 0; h < 24; h++) {
      const i = d * 24 + h, v = cells[i];
      const title = slot(i) + ' · ' + fmtDur(v) + ' GPU · ' + fmtCount(sess[i]) + ' sessions';
      html += '<i tabindex="0" role="img" aria-label="' + esc(title) + '" data-i="' + i + '" data-tip="1"'
        + ' data-l="' + levelOf(v, br) + '"' + (i === peak && peakV > 0 ? ' class="peak"' : '') + '></i>';
    }
    html += '<span class="rb"><i style="width:' + (dayTot[d] > 0 ? Math.max(8, dayTot[d] / dm * 100) : 0) + '%"></i></span>';
  }
  el.innerHTML = html;
  wireTips(el, '[data-tip]', (target) => {
    const i = Number(target.getAttribute('data-i')), d = Math.floor(i / 24);
    let out = '<div class="tip-date">' + slot(i) + '</div>';
    const ks = Object.keys(byKind).filter(k => byKind[k][0][i] || byKind[k][1][i]).sort((a, b) => KINDS.indexOf(a) - KINDS.indexOf(b));
    ks.forEach(k => out += tipRow(KIND_COLOR[k], KIND_LABEL[k] || k, fmtDur(byKind[k][0][i]) + ' GPU · ' + fmtCount(byKind[k][1][i]) + ' sessions'));
    out += tipRow(null, ks.length ? 'total' : 'idle', cells[i] ? fmtDur(cells[i]) + ' GPU · ' + fmtCount(sess[i]) + ' sessions' : '—', ks.length > 1 ? 'tot' : '');
    if (cells[i] > 0 && occ[d] > 1) out += tipRow(null, 'per ' + DOW_LONG[d], '≈ ' + fmtDur(cells[i] / occ[d]) + ' GPU over ' + occ[d] + ' ' + DOW_LONG[d] + 's');
    return out;
  });
  const kindNote = Q.kind !== 'all' ? (KIND_LABEL[Q.kind] || Q.kind) + ' only · ' : '';
  const lg = $('hours-legend');
  if (lg) lg.innerHTML = br
    ? '<span class="ramp">quiet <i></i> busy</span><span>max ' + fmtDur(peakV) + ' per slot</span>'
      + '<span class="what">' + kindNote + 'GPU seconds per weekday-hour · ' + esc(lastDoc.tz === 'local' ? 'server time' : lastDoc.tz) + '</span>'
    : '<span class="what">' + kindNote + 'no GPU time in this window</span>';
  const sub = $('hours-sub');
  if (sub) sub.innerHTML = peakV > 0
    ? 'Busiest slot: ' + slot(peak) + ' · <b>' + fmtDur(peakV) + ' GPU</b>'
      + (occ[Math.floor(peak / 24)] > 1 ? ' over ' + occ[Math.floor(peak / 24)] + ' ' + DOW_LONG[Math.floor(peak / 24)] + 's, ≈ '
        + fmtDur(peakV / occ[Math.floor(peak / 24)]) + ' each' : '')
      + ' · ' + fmtCount(sess[peak]) + ' sessions'
    : 'No GPU time in this window';
  const tag = $('hours-tag');
  if (tag) tag.textContent = hoursPhrase(cells);
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
onSeg('usage-metric', (v) => { Q.metric = v; setSeg('usage-metric', v); load(); });
onSeg('usage-by', (v) => { Q.by = v; setSeg('usage-by', v); hidden.clear(); load(); });
const tableBtn = $('usage-table-btn');
if (tableBtn) tableBtn.addEventListener('click', () => { tableMode = !tableMode; renderTable(); if (!tableMode && chart) chart.setSize({ width: chartEl.clientWidth, height: chartEl.clientHeight }); });
document.addEventListener('keydown', (e) => {
  if (e.target && /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
  if (e.key === 't' || e.key === 'T') { tableMode = !tableMode; renderTable(); }
});

parsePageQuery(location.search);
if (ownScope() && Q.by === 'user') Q.by = 'key';
setSeg('sb-range', Q.range); setSeg('sb-compare', Q.compare);
setSeg('usage-bucket', Q.bucket); setSeg('usage-metric', Q.metric); setSeg('usage-by', Q.by);
wireScopeBar();
renderChips();
load();
// Own scope is applied by the inline IIFE after the first snapshot; when it
// flips the `by` control to `key`, reload with the corrected query.
window._fwUsageReload = () => { if (ownScope() && Q.by === 'user') { Q.by = 'key'; setSeg('usage-by', 'key'); } load(); };
})();
