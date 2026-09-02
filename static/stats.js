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

// --- GridStack init: drag-to-reorder + click-to-resize tiles ---------------
// Layout state persists in localStorage; [↺ layout] in the header clears it.
// uPlot sparklines re-fit on resizestop via setSize().
// `let`: applyScope() appends '-own' before the machine tiles are removed
// for scope=own, so an own-scope layout never restores tiles it lacks and an
// admin's layout in the same browser is left alone.
// v6: the usage half gained tiles (headline, stages, hours); a v5 layout
// would remove them on load, so the key moved and v5 layouts are ignored.
let GS_LAYOUT_KEY = 'whisper-stats-layout-v6';
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
  float: true,
  resizable: { handles: 'se,s,e' },
  draggable: { handle: '.card h3' },
  alwaysShowResizeHandle: false,
});
// On touch devices, freeze drag/resize (the layout stays, but reordering tiles
// by dragging is fiddly on a phone and the dashboard is read-mostly there).
try {
  if (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) {
    grid.setStatic(true);
  }
} catch (_) {}
// Restore saved layout if present (best-effort — schema mismatches are
// silently ignored; user can hit [↺ layout] to recover defaults).
try {
  const saved = localStorage.getItem(GS_LAYOUT_KEY);
  if (saved) grid.load(JSON.parse(saved));
} catch (_) {}
// Persist on every change (debounced via setTimeout to coalesce rapid drags).
let _saveTimer = null;
function _saveLayout() {
  if (_saveTimer) clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    try { localStorage.setItem(GS_LAYOUT_KEY, JSON.stringify(grid.save(false))); } catch (_) {}
  }, 200);
}
grid.on('change added removed', _saveLayout);
// Resize is handled by per-spark ResizeObserver inside makeSpark() — fires on
// GridStack drag-resize, window resize, scale-picker rem changes, and any
// other reflow uniformly. Listening on `resizestop` here would only catch
// GridStack-initiated resizes and would miss the rest.
// Header reset-layout button.
const resetLayoutBtn = document.getElementById('reset-layout-btn');
if (resetLayoutBtn) {
  resetLayoutBtn.addEventListener('click', () => {
    if (!confirm('Reset stats tile layout to defaults?')) return;
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
                      vad: 'Silence skipping', separating: 'Music separation' };
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
  return [q(0.25), q(0.5), q(0.75)];
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
    Q.kind = c.dataset.v; renderChips(); syncUrl(); renderAll();   // client-side
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
function load() {
  renderChips(); syncUrl();
  usageCards().forEach(el => el.classList.add('updating'));
  const mine = ++_seq;
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
  // Everything that takes vertical space in the usage card renders BEFORE
  // the chart, which measures whatever height is left; drawing it first
  // would freeze a canvas sized against a one-row "loading" board.
  prepareLines();
  renderLegend();
  renderBoard();
  renderStages();
  renderHours();
  publishModels();
  renderChart();
  renderTable();
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
  const cells = [
    ['audio', fmtDur(tot.audio_s), '', delta(tot.audio_s, cmp && cmp.audio_s)],
    ['sessions', fmtCount(tot.sessions), '· ' + fmtCount(tot.requests) + ' requests',
     delta(tot.sessions, cmp && cmp.sessions)],
    ['failed', (failed * 100).toFixed(1), '%', delta(failed, cfailed, true)],
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
  tipEl.innerHTML = html;
  tipEl.style.display = 'block';
  const orect = u.over.getBoundingClientRect();
  const vw = document.documentElement.clientWidth, vh = document.documentElement.clientHeight;
  const tw = tipEl.offsetWidth, th = tipEl.offsetHeight;
  let left = orect.left + u.cursor.left + 14;
  if (left + tw + 4 > vw) left = orect.left + u.cursor.left - tw - 14;
  let top = orect.top + u.cursor.top + 14;
  if (top + th + 4 > vh) top = orect.top + u.cursor.top - th - 14;
  tipEl.style.left = Math.max(4, left) + 'px';
  tipEl.style.top = Math.max(4, top) + 'px';
  announce(fmtDate(xs[idx]) + ': ' + rows.map(ln => ln.label + ' ' + fmtMetric(Q.metric, ln.values[idx] || 0)).join(', '));
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
      if (by === 'kind') { Q.kind = Q.kind === id ? 'all' : id; renderChips(); syncUrl(); renderAll(); return; }
      if (by === 'user' && ownScope()) return;
      Q[by] = Q[by] === id ? null : id;
      load();
    };
    tr.addEventListener('click', pick);
    tr.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
  });
}

// ---- pipeline stages: share of eligible runs + speed per optional stage.
const STAGE_ROWS = ['translating', 'diarizing', 'vad', 'separating'];
function renderStages() {
  const el = $('stages-rows'); if (!el) return;
  const rows = {};
  (lastDoc.stages || []).forEach(s => { rows[s.stage] = s; });
  const ordered = STAGE_ROWS.slice().sort((a, b) =>
    (Q.with.includes(b) ? 1 : 0) - (Q.with.includes(a) ? 1 : 0));
  const totalSecs = ordered.reduce((a, s) => a + ((rows[s] && rows[s].secs) || 0), 0);
  const bar = $('stages-bar');
  if (bar) bar.innerHTML = totalSecs > 0 ? ordered.map(s => {
    const secs = (rows[s] && rows[s].secs) || 0;
    return secs > 0 ? '<span style="flex:' + (secs / totalSecs) + ';background:' + STAGE_COLOR[s]
      + '" title="' + esc(STAGE_LABEL[s]) + ' ' + (secs / totalSecs * 100).toFixed(0) + ' %"></span>' : '';
  }).join('') : '';
  el.innerHTML = ordered.map(s => {
    const r = rows[s];
    if (!r || !r.runs) {
      return '<tr class="dim"><td><span class="stage-sw" style="background:' + STAGE_COLOR[s] + '"></span>'
        + esc(STAGE_LABEL[s]) + '</td><td colspan="5" class="empty">not used in this window</td></tr>';
    }
    const pct = r.of_runs > 0 ? Math.round(r.runs / r.of_runs * 100) : 0;
    const rtf = r.audio_s > 0 ? r.secs / r.audio_s : null;
    let extra = '';
    if (s === 'diarizing' && r.speakers_avg != null) extra = r.speakers_avg + ' speakers avg';
    if (s === 'vad' && r.retained_avg != null) extra = ((1 - r.retained_avg) * 100).toFixed(0) + ' % skipped';
    if (s === 'translating') {
      extra = (r.targets || []).slice(0, 4).map(t => esc(t.code.toUpperCase()) + ' '
        + Math.round(t.runs / r.runs * 100) + ' %').join(' · ');
      if (r.kept_original) extra += (extra ? ' · ' : '') + r.kept_original + ' kept original';
    }
    return '<tr' + (Q.with.includes(s) ? ' class="pinned"' : '') + '>'
      + '<td><span class="stage-sw" style="background:' + STAGE_COLOR[s] + '"></span>' + esc(STAGE_LABEL[s])
      + (Q.with.includes(s) ? ' <span class="badge">filter</span>' : '')
      + (extra ? '<span class="sub">' + extra + '</span>' : '') + '</td>'
      + '<td class="num">' + fmtCount(r.runs) + '</td>'
      + '<td class="num"><span class="meter"><i style="width:' + pct + '%"></i></span>' + pct + ' %</td>'
      + '<td class="num">' + fmtDur(r.audio_s) + '</td>'
      + '<td class="num">' + fmtDur(r.secs) + '</td>'
      + '<td class="num"><span class="rtf' + (rtf > 0.35 ? ' slow' : '') + '">' + (rtf == null ? '—' : rtf.toFixed(2) + '×') + '</span></td>'
      + '</tr>';
  }).join('');
  const tag = $('stages-tag');
  if (tag) tag.textContent = fmtDur(totalSecs) + ' GPU in optional stages';
}

// ---- busy hours: weekday × hour of GPU seconds, quartile-levelled over the
// active cells, peak ringed; cells are focusable and describe themselves.
const DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
function renderHours() {
  const el = $('hours-grid'); if (!el) return;
  const cells = new Array(7 * 24).fill(0);
  const sess = new Array(7 * 24).fill(0);
  (lastDoc.hours || []).forEach(h => {
    const v = h.proc_s ? kindScoped(h.proc_s) : 0;
    const s = h.sessions ? kindScoped(h.sessions) : 0;
    cells[h.dow * 24 + h.hour] += Number(v || 0);
    sess[h.dow * 24 + h.hour] += Number(s || 0);
  });
  const br = quantileBreaks(cells);
  let peak = -1, peakV = 0;
  cells.forEach((v, i) => { if (v > peakV) { peakV = v; peak = i; } });
  let html = '<span></span>' + Array.from({ length: 24 }, (_, h) =>
    '<span class="hl">' + (h % 6 === 0 ? ('0' + h).slice(-2) : '') + '</span>').join('');
  for (let d = 0; d < 7; d++) {
    html += '<span class="dl">' + DOW[d] + '</span>';
    for (let h = 0; h < 24; h++) {
      const i = d * 24 + h, v = cells[i];
      const title = DOW[d] + ' ' + ('0' + h).slice(-2) + '–' + ('0' + (h + 1)).slice(-2) + ' · '
        + fmtDur(v) + ' GPU · ' + fmtCount(sess[i]) + ' sessions';
      html += '<i tabindex="0" role="img" aria-label="' + esc(title) + '" title="' + esc(title)
        + '" data-l="' + levelOf(v, br) + '"' + (i === peak && peakV > 0 ? ' class="peak"' : '') + '></i>';
    }
  }
  el.innerHTML = html;
  const lg = $('hours-legend');
  if (lg) lg.innerHTML = br
    ? '<span><i data-l="0"></i>idle</span><span><i data-l="1"></i>≤ ' + fmtDur(br[0]) + '</span>'
      + '<span><i data-l="2"></i>≤ ' + fmtDur(br[1]) + '</span><span><i data-l="3"></i>≤ ' + fmtDur(br[2])
      + '</span><span><i data-l="4"></i>&gt; ' + fmtDur(br[2]) + '</span>'
      + '<span class="what">GPU seconds · quartiles of active hours · ' + esc(lastDoc.tz === 'local' ? 'server time' : lastDoc.tz) + '</span>'
    : '<span class="what">no GPU seconds in this window</span>';
  const tag = $('hours-tag');
  if (tag) tag.textContent = peak >= 0 && peakV > 0
    ? 'peak ' + DOW[Math.floor(peak / 24)] + ' ' + ('0' + (peak % 24)).slice(-2) + '–'
      + ('0' + (peak % 24 + 1)).slice(-2) + ' · ' + fmtDur(peakV) + ' GPU'
    : '';
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
