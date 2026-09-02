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
let GS_LAYOUT_KEY = 'whisper-stats-layout-v5';
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

// --- Usage-over-time section (independent of the live SSE dashboard) -------
// Self-contained IIFE: builds its own uPlot time-series chart + leaderboard,
// fetched once on load and on every selector change. The chart formats its own
// axis/tooltip dates inline rather than via the shared TIME_HELPERS_JS fmtWhen
// (now injected lower down for the recent-transcriptions table): its x-values
// are day*86400 = UTC midnight of each bucket's calendar date, so labels read
// the date with getUTC* (correct on any operator timezone) — fmtWhen would
// apply the local clock, which is meaningless for a whole-day bucket.
(() => {
'use strict';
const $ = id => document.getElementById(id);
const chartEl = $('usage-plot');
const tipEl = $('usage-tip');
if (!chartEl || typeof uPlot === 'undefined') return;

// Per-entity line palette; 'others' is a dim dashed grey. curLines/curMetric
// are shared by buildChart, the tooltip, and the leaderboard swatches.
const PALETTE = ['#79c0ff','#7ee787','#f2cc60','#d2a8ff','#ff7b72','#56d4dd','#e3b341','#ff9bce'];
const OTHERS_COLOR = '#6e7681';
let curLines = [];
let curMetric = 'audio_s';

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
  return metric === 'audio_s' ? fmtDur(v) : fmtCount(v);
}
function fmtDate(ts) {
  const d = new Date(ts * 1000), p2 = n => ('0' + n).slice(-2);
  return d.getUTCFullYear() + '.' + p2(d.getUTCMonth() + 1) + '.' + p2(d.getUTCDate());
}

// Floating cursor tooltip: bucket date + every line's value, with the line
// nearest the cursor (by data-space y distance) bolded. Driven by uPlot's
// setCursor hook; hidden when the cursor leaves the plot (idx == null).
function updateTip(u) {
  const idx = u.cursor.idx;
  if (idx == null || !curLines.length) { tipEl.style.display = 'none'; return; }
  const xs = u.data[0];
  const cy = u.posToVal(u.cursor.top, 'y');
  let focus = -1, best = Infinity;
  for (let s = 0; s < curLines.length; s++) {
    const v = u.data[s + 1][idx];
    if (v == null) continue;
    const d = Math.abs(v - cy);
    if (d < best) { best = d; focus = s; }
  }
  let html = '<div class="tip-date">' + fmtDate(xs[idx]) + '</div>';
  for (let s = 0; s < curLines.length; s++) {
    const ln = curLines[s];
    html += '<div class="tip-row' + (s === focus ? ' focus' : '') + '">'
      + '<span class="usage-swatch" style="background:' + ln.color + '"></span>'
      + '<span>' + esc(ln.label) + '</span>'
      + '<span class="tip-val">' + fmtMetric(curMetric, u.data[s + 1][idx]) + '</span>'
      + '</div>';
  }
  tipEl.innerHTML = html;
  tipEl.style.display = 'block';
  // Position in VIEWPORT space (the tooltip is position:fixed, so the card's
  // overflow:auto can't clip it near the chart edges). uPlot cursor coords are
  // relative to the plot over-element; add its viewport rect.
  const orect = u.over.getBoundingClientRect();
  const vw = document.documentElement.clientWidth;
  const vh = document.documentElement.clientHeight;
  const tw = tipEl.offsetWidth, th = tipEl.offsetHeight;
  let left = orect.left + u.cursor.left + 14;
  if (left + tw + 4 > vw) left = orect.left + u.cursor.left - tw - 14;  // flip left near right edge
  let top = orect.top + u.cursor.top + 14;
  if (top + th + 4 > vh) top = orect.top + u.cursor.top - th - 14;      // flip up near bottom edge
  tipEl.style.left = Math.max(4, left) + 'px';
  tipEl.style.top = Math.max(4, top) + 'px';
}

// cursor.move hook: snap the vertical guide to the nearest data point's x-pixel
// so guide + highlight dot + tooltip coincide (uPlot's canonical snap, per the
// nearest-non-null demo). mLeft < 0 means the cursor is off-plot — leave it.
function snapToDataX(u, mLeft, mTop) {
  if (mLeft < 0) return [mLeft, mTop];
  const idx = u.valToIdx(u.posToVal(mLeft, 'x'));
  return [Math.round(u.valToPos(u.data[0][idx], 'x')), mTop];
}

let chart = null;
function buildChart() {
  if (chart) { chart.destroy(); chart = null; }
  const w = chartEl.clientWidth || 600;
  const h = chartEl.clientHeight || 220;
  const single = curLines.length === 1;
  const series = [{ value: (u, ts) => ts == null ? '' : fmtDate(ts) }].concat(
    curLines.map(ln => ({
      label: ln.label, stroke: ln.color,
      width: ln.others ? 1.25 : 1.5,
      dash: ln.others ? [4, 3] : undefined,
      fill: single ? ln.color + '22' : undefined,
      points: { show: single, size: 4 }, spanGaps: true,
    })));
  chart = new uPlot({
    width: w, height: h,
    padding: [remPx(0.5), remPx(0.6), remPx(0.2), remPx(0.4)],
    legend: { show: false },
    // drag:{x:false,y:false} removes uPlot's default drag-to-zoom.
    // points: a per-series highlight dot on the hovered bucket via uPlot's
    // .u-cursor-pt — leave `show` at its default (an element-FACTORY); passing
    // the boolean `show:true` makes initCursorPt's `instanceof HTMLElement`
    // check fail, so NO dot is ever created. We only style it (rem-sized disc,
    // --bg ring, fill = each line's colour) so every line gets a clear marker.
    // move: snapToDataX locks the vertical guide onto the nearest data point's
    // x-pixel, so the guide line, the dot, cursor.idx and the tooltip all land
    // on the SAME bucket (without it the guide trails the raw mouse and the
    // tooltip flips to the next day at the column's midpoint).
    cursor: {
      y: false, drag: { x: false, y: false }, move: snapToDataX,
      points: {
        size: remPx(0.6), width: remPx(0.13), stroke: '#0d1117',
        fill: (u, si) => (curLines[si - 1] && curLines[si - 1].color) || '#79c0ff',
      },
    },
    scales: { x: { time: true }, y: { range: { min: { pad: 0.05, mode: 1 }, max: { pad: 0.1, mode: 1 } } } },
    hooks: { setCursor: [updateTip] },
    axes: [
      { stroke: '#6e7681', grid: { stroke: '#21262d', width: 1 },
        ticks: { stroke: '#30363d', width: 1, size: 3 },
        font: remPx(0.733) + 'px ' + MONO,
        // Ticks on a clean whole-day calendar grid, NOT on array indices. The
        // data omits empty buckets, so the old "every Nth point" subsample (a) put
        // labels on uneven dates and (b) HALVED the label set at 64px-width
        // boundaries (ceil(n/floor(px/64))) — a stray scrollbar flipped 12 daily
        // labels to every-other-day. Instead pick a day-step from a curated ladder
        // (smallest whose pixel spacing ≥ ~65px) and emit ticks at t0 + k·step·day.
        // t0 is a UTC midnight, so every tick is a UTC midnight → the getUTC
        // `values` formatter below stays calendar-correct on any operator timezone.
        // Thins gracefully (1→2→3→7…) and is deterministic across rebuilds.
        splits: (u) => {
          const xs = u.data[0] || [];
          if (xs.length < 2) return xs.slice();
          // Width source MUST be u.bbox.width, NOT u.over.clientWidth: on every
          // rebuild (any selector change) uPlot runs splits before .u-over is laid
          // out, so over.clientWidth reads 0 and the old `|| 600` fallback forced a
          // narrow width → every-other-day labels that then stuck. bbox.width is set
          // synchronously from the passed width and is reliable; it's in DEVICE
          // pixels, so divide by devicePixelRatio to get CSS px for the gap test.
          const px = (u.bbox && u.bbox.width)
            ? u.bbox.width / (window.devicePixelRatio || 1)
            : (u.over.clientWidth || 600);
          const maxTicks = Math.max(2, Math.floor(px / remPx(4.3)));
          const t0 = xs[0], t1 = xs[xs.length - 1];
          const spanDays = Math.max(1, Math.round((t1 - t0) / 86400));
          const LADDER = [1, 2, 3, 7, 14, 30, 60, 90, 180, 365];
          let step = LADDER[LADDER.length - 1];
          for (const s of LADDER) { if (spanDays / s <= maxTicks) { step = s; break; } }
          const out = [];
          for (let t = t0; t <= t1 + 1; t += step * 86400) out.push(t);
          return out;
        },
        // Fixed MM.DD, locale-independent (matches the project's YYYY.MM.DD norm).
        // x-values are UTC midnight of each bucket day, so getUTC* yields the
        // server-intended calendar date on every operator timezone. The hover
        // tooltip carries the full year-qualified date.
        values: (u, splits) => splits.map(s => {
          const d = new Date(s * 1000);
          const p2 = n => ('0' + n).slice(-2);
          return p2(d.getUTCMonth() + 1) + '.' + p2(d.getUTCDate());
        }) },
      { stroke: '#6e7681', size: remPx(2.8), gap: 4,
        grid: { stroke: '#21262d', width: 1 },
        ticks: { stroke: '#30363d', width: 1, size: 3 },
        font: remPx(0.733) + 'px ' + MONO,
        values: (u, splits) => splits.map(v => fmtMetric(curMetric, v)) },
    ],
    series,
  }, [[]].concat(curLines.map(() => [])), chartEl);
  chart.over.addEventListener('mouseleave', () => { tipEl.style.display = 'none'; });
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

function renderBoard(board, by) {
  const tb = $('usage-board-rows');
  if (!board || !board.length) {
    tb.innerHTML = '<tr><td colspan="6" class="empty">— no usage in this window —</td></tr>';
    return;
  }
  // Swatch the rows that are charted (top-K) so the table ties to the lines.
  const colorById = {};
  curLines.forEach(ln => { if (!ln.others) colorById[ln.id] = ln.color; });
  tb.innerHTML = board.map((r, i) => {
    const sw = colorById[r.id]
      ? '<span class="usage-swatch" style="background:' + colorById[r.id] + '"></span>' : '';
    const sub = by === 'key' && r.user_label
      ? '<span class="sub">' + esc(r.user_label) + '</span>' : '';
    return '<tr>'
      + '<td class="rank" data-label="#">' + (i + 1) + '</td>'
      + '<td class="name" data-label="name">' + sw + esc(r.label || '?') + sub + '</td>'
      + '<td class="num" data-label="requests">' + fmtCount(r.requests) + '</td>'
      + '<td class="num" data-label="words">' + fmtCount(r.words) + '</td>'
      + '<td class="num" data-label="audio">' + fmtDur(r.audio_s) + '</td>'
      + '<td class="num" data-label="err">' + fmtCount(r.errors) + '</td>'
      + '</tr>';
  }).join('');
}
function esc(s) {
  return String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function segVal(id) {
  const g = $(id);
  const b = g && g.querySelector('button.active');
  return b ? b.dataset.v : '';
}

let _seq = 0;
function loadUsage() {
  const days = segVal('usage-range');
  const bucket = segVal('usage-bucket');
  const metric = segVal('usage-metric');
  const by = segVal('usage-by');
  const q = '?days=' + encodeURIComponent(days)
          + '&bucket=' + encodeURIComponent(bucket)
          + '&metric=' + encodeURIComponent(metric)
          + '&by=' + encodeURIComponent(by);
  const mine = ++_seq;
  fetch('/stats/usage' + q, { cache: 'no-store' })
    .then(r => {
      if (r.status === 403) {
        // Own scope asked for the per-user board (the only row would be
        // the viewer). The page switches the control to `key` on its own;
        // say so instead of "unavailable".
        $('usage-board-rows').innerHTML =
          '<tr><td colspan="6" class="empty">— not available for your scope —</td></tr>';
        return null;
      }
      return r.ok ? r.json() : null;
    })
    .then(j => {
      if (!j || mine !== _seq) return;   // stale response — a newer change won
      curMetric = j.metric || metric;
      curLines = (j.lines || []).map((ln, i) => ({
        id: ln.id, label: ln.label, values: ln.values, others: !!ln.others,
        color: ln.others ? OTHERS_COLOR : PALETTE[i % PALETTE.length],
      }));
      const xs = (j.days || []).map(d => d * 86400);
      if (!curLines.length || !xs.length) {
        if (chart) { chart.destroy(); chart = null; }
        tipEl.style.display = 'none';
        renderBoard(j.leaderboard, by);
        return;
      }
      buildChart();
      chart.setData([xs].concat(curLines.map(l => l.values)));
      renderBoard(j.leaderboard, by);
    })
    .catch(err => {
      console.warn('[stats] usage fetch failed', err);
      $('usage-board-rows').innerHTML =
        '<tr><td colspan="6" class="empty">— usage unavailable —</td></tr>';
    });
}

['usage-range', 'usage-bucket', 'usage-metric', 'usage-by'].forEach(id => {
  const g = $(id);
  if (!g) return;
  g.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b || !g.contains(b) || b.classList.contains('active')) return;
    g.querySelectorAll('button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    loadUsage();
  });
});
loadUsage();
})();
