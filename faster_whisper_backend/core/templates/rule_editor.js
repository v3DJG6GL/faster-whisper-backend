<script>
// Default the cb:map entry cap for pages that embed the editor without an
// API payload carrying map_max_entries (/settings). /quick-config's load()
// still overwrites it with the server value -- same schema, same number.
window.__mme = (typeof window.__mme === 'number') ? window.__mme : __MAP_CAP__;
// German-aware, case-insensitive collator for ordering human-visible lists
// (cb:map keys). sensitivity:'base' folds case + accent so 'anemi' sorts next
// to 'Amoxyzylin' and umlauts land in German position (ä≈a); numeric:true gives
// natural number order (item2 before item10). Display only — never used for
// equality/fingerprint/persistence sorts, which stay byte-stable.
const _coll = new Intl.Collator('de', { sensitivity: 'base', numeric: true });
// Display \n, \r, \t, \\ as literal 2-char escape sequences in <input>
// cells. Single-line inputs strip newlines per WHATWG spec, so without
// this the user sees an empty field for any value containing a newline.
function _esc(s) {
  if (s == null) return '';
  return String(s)
    .replace(/\\/g, '\\\\')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t');
}
function _unesc(s) {
  if (s == null) return '';
  let out = '';
  for (let i = 0; i < s.length; i++) {
    if (s[i] === '\\' && i + 1 < s.length) {
      const nxt = s[++i];
      if (nxt === 'n') out += '\n';
      else if (nxt === 'r') out += '\r';
      else if (nxt === 't') out += '\t';
      else if (nxt === '\\') out += '\\';
      else out += '\\' + nxt;  // keep both chars: \1, \d, \w, \. survive intact
    } else {
      out += s[i];
    }
  }
  return out;
}

const _PIPELINE_TYPES = [
  { type: 'regex-list',                  pill: 'regex[]' },
  { type: 'callback:lowercase-wordlist', pill: 'cb:wordlist' },
  { type: 'callback:map',                pill: 'cb:map' },
  { type: 'callback:dedup',              pill: 'cb:dedup' },
  { type: 'callback:upper',              pill: 'cb:upper' },
  { type: 'terminal',                    pill: 'terminal' },
];
const _typePill = (t) => (_PIPELINE_TYPES.find(x => x.type === t) || {}).pill || t;

// Enter commits after a one-tick defer: picking an <input list> suggestion
// with Enter changes the value synchronously, so a changed value means a
// datalist pick (no commit; the next Enter fires it).
function _bindEnterCommit(inp, onEnter) {
  if (!onEnter) return;
  inp.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    e.preventDefault();
    const before = inp.value;
    setTimeout(() => {
      if (inp.value !== before) return;  // datalist pick — defer
      onEnter();
    }, 0);
  });
}

function _helpDiv(text) {
  const d = document.createElement('div');
  d.className = 'help';
  if (text != null) d.textContent = text;
  return d;
}

function _makeMonoLabeledInput(label, val, onInput, onEnter) {
  const lbl = _helpDiv(label + ':');
  const inp = document.createElement('input');
  inp.type = 'text';
  inp.spellcheck = false;
  inp.autocomplete = 'off';
  inp.value = val == null ? '' : val;
  inp.addEventListener('input', () => onInput(inp.value));
  _bindEnterCommit(inp, onEnter);
  const wrap = document.createElement('div');
  wrap.appendChild(lbl); wrap.appendChild(inp);
  return wrap;
}

function _markMapDup(inp, dup, caseOnly) {
  inp.classList.toggle('map-dup', dup);
  if (dup) inp.title = caseOnly
    ? 'Duplicate spoken word (case is ignored): keep only one of these rows, or Save is rejected'
    : 'Duplicate spoken word: only the last row is saved';
  else inp.removeAttribute('title');
}

function _makeMapRow(rule, key, val, commitData, datalistId, onEnter, ts, showDate) {
  const tr = document.createElement('tr');
  const td1 = document.createElement('td');
  const td2 = document.createElement('td');
  const td3 = document.createElement('td');
  td3.style.width = '2.5rem';
  const ki = document.createElement('input');
  ki.type = 'text'; ki.value = _esc(key);
  // /quick-config autocompletes the spoken word from recent transcription
  // FINALs; /settings passes no datalistId (it would clutter long maps).
  if (datalistId) ki.setAttribute('list', datalistId);
  const vi = document.createElement('input');
  vi.type = 'text'; vi.value = _esc(val);
  // A repeated spoken word keeps only the LAST row's value, so every row
  // sharing a key is flagged; otherwise Save silently drops the others.
  // The server lowercases keys (and 422s keys differing only by case), so
  // rows are grouped case-insensitively. Null-prototype m and a Map keep
  // words like "constructor" or "__proto__" from hitting Object.prototype.
  function _readMap(parent) {
    const m = Object.create(null), byKey = new Map();
    parent.querySelectorAll('tr').forEach(r => {
      const kin = r.querySelector('td:first-child input');
      const k = _unesc(kin.value);
      const v = _unesc(r.querySelector('td:nth-child(2) input').value);
      if (k) {
        m[k] = v;
        const lk = k.toLowerCase();
        if (!byKey.has(lk)) byKey.set(lk, []);
        byKey.get(lk).push([kin, k]);
      }
      else _markMapDup(kin, false);
    });
    byKey.forEach(rows => {
      const caseOnly = rows.some(([, k]) => k !== rows[0][1]);
      rows.forEach(([i]) => _markMapDup(i, rows.length > 1, caseOnly));
    });
    return m;
  }
  function rebuild() {
    const parent = tr.parentNode;
    if (!parent) return;
    rule.map = _readMap(parent);
    commitData();
  }
  ki.addEventListener('input', rebuild);
  vi.addEventListener('input', rebuild);
  _bindEnterCommit(ki, onEnter);
  _bindEnterCommit(vi, onEnter);
  const del = document.createElement('button');
  del.type = 'button'; del.textContent = '×';
  del.addEventListener('click', () => {
    const parent = tr.parentNode;
    tr.remove();
    if (parent) {
      rule.map = _readMap(parent);
      commitData();
    }
  });
  td1.appendChild(ki); td2.appendChild(vi); td3.appendChild(del);
  tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3);
  // Inline "added / last-updated" date, /quick-config only (showDate). Appended
  // LAST so the td:first-child / td:nth-child(2) selectors _readMap uses to
  // locate the key/value inputs are unaffected. New rows (no ts yet) show "—"
  // until the next server save stamps map_meta.
  if (showDate) {
    const td4 = document.createElement('td');
    td4.className = 'map-date-cell';
    const span = document.createElement('span');
    span.className = 'map-date';
    stampTs(span, ts);
    td4.appendChild(span);
    tr.appendChild(td4);
  }
  return tr;
}

// ---- regex-list entry editor (shared by /settings + /quick-config) ----
// An ordered batch of {pattern, replacement, label?, note?} entries. The DOM IS the source
// of truth: add / delete / drag-reorder all reduce to "mutate the DOM, then
// _readEntries + commit". Empty-pattern rows are skipped on read (kept in the
// DOM so an in-progress row isn't lost), mirroring cb:map. Helpers live at
// script top level so they survive strict-mode block scoping.
let _rlDragSrc = null;

// Shared reorder affordances: also used by the admin pipeline-rule list, a
// sibling <script> parsed after this one that inherits these globals.
// Edge auto-scroll for native-DnD reorder lists. The browser's built-in drag
// auto-scroll only fires in a sliver at the very viewport edge, crawls at a
// fixed speed, and stalls the instant the pointer holds still. This drives a
// requestAnimationFrame loop from the LAST pointer Y, so parking near an edge
// keeps scrolling, with speed ramping by how deep into the hot-zone the pointer
// sits. begin(listEl) resolves the scroll target (nearest scrollable ancestor,
// else the window) once per drag; feed move(clientY) from dragover; stop() on
// dragend/drop. One native drag at a time per tab, so a single shared instance
// (DRAG_AUTOSCROLL) serves every reorder list on the page.
function makeDragAutoScroll() {
  let raf = null, lastY = 0, scroller = null, tracking = null, topInset = 0;  // scroller === null → window
  function resolve(listEl) {
    let el = listEl && listEl.parentElement;
    while (el) {
      const oy = getComputedStyle(el).overflowY;
      if ((oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight + 1) {
        scroller = el; return;
      }
      el = el.parentElement;
    }
    scroller = null;
  }
  // A sticky/fixed header pinned at top:0 covers the top hot-zone, so the zone
  // starts at the header's BOTTOM (else it swallows the zone). That edge is
  // constant during a drag, so resolve it once.
  function computeTopInset() {
    const hdr = document.querySelector('header');
    if (!hdr) return 0;
    const pos = getComputedStyle(hdr).position;
    if (pos !== 'sticky' && pos !== 'fixed') return 0;
    const r = hdr.getBoundingClientRect();
    return (r.top <= 1 && r.bottom > 0) ? r.bottom : 0;
  }
  function tick() {
    let top, bottom;
    if (scroller) { const r = scroller.getBoundingClientRect(); top = r.top; bottom = r.bottom; }
    else { top = 0; bottom = window.innerHeight; }
    top = Math.max(top, topInset);                // clear a sticky header
    const zone = Math.max(48, Math.min((bottom - top) * 0.18, 140));   // hot-zone px
    let m = 0;                                    // -1..1, sign = scroll direction
    if (lastY < top + zone) m = -Math.min(1, (top + zone - lastY) / zone);
    else if (lastY > bottom - zone) m = Math.min(1, (lastY - (bottom - zone)) / zone);
    if (m) {
      const step = Math.sign(m) * Math.ceil(Math.pow(Math.abs(m), 1.5) * 24);  // ease-in, cap 24px/frame
      if (scroller) scroller.scrollTop += step;
      else window.scrollBy(0, step);
    }
    raf = requestAnimationFrame(tick);
  }
  return {
    begin(listEl) {
      resolve(listEl);
      topInset = computeTopInset();
      // Track the pointer at the document level for the whole drag, so the loop
      // keeps a LIVE y even when the cursor is over the sticky header (where the
      // list's own dragover never fires) — that's what lets the top edge scroll
      // without having to reach the very window edge.
      tracking = (e) => { lastY = e.clientY; if (raf === null) raf = requestAnimationFrame(tick); };
      document.addEventListener('dragover', tracking, true);
    },
    move(y) { lastY = y; if (raf === null) raf = requestAnimationFrame(tick); },
    stop() {
      if (raf !== null) cancelAnimationFrame(raf);
      raf = null; scroller = null; topInset = 0;
      if (tracking) { document.removeEventListener('dragover', tracking, true); tracking = null; }
    },
  };
}
const DRAG_AUTOSCROLL = makeDragAutoScroll();

// Thin stroke chevrons for the ↑/↓ one-position "move" buttons (the keyboard /
// no-drag path that complements drag-to-reorder). Stroke (not filled triangle)
// to read light at rail size and sit consistently beside the inline-SVG lock.
const MOVE_CHEV_UP = '<svg viewBox="0 0 16 16" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M3.5 10.5 8 6l4.5 4.5"/></svg>';
const MOVE_CHEV_DOWN = '<svg viewBox="0 0 16 16" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M3.5 5.5 8 10l4.5-4.5"/></svg>';

function _readEntries(parent) {
  const out = [];
  parent.querySelectorAll('.rl-entry').forEach((en) => {
    // find/repl are regex SOURCE, read verbatim (and shown verbatim by frRow):
    // _unesc would collapse \n/\t/\\ in the regex. Only cb:map (raw strings)
    // and the label escape.
    const pat = (en.querySelector('.e-pattern') || {}).value || '';
    if (!pat) return;
    const lbl = _unesc((en.querySelector('.e-label') || {}).value || '');
    const note = (en.querySelector('.e-note') || {}).value || '';  // textarea: real text
    // Always emit label/note ("" when empty), like the server's schema dump.
    out.push({ pattern: pat,
               replacement: (en.querySelector('.e-repl') || {}).value || '',
               label: lbl, note: note });
  });
  return out;
}

function _renumberEntries(parent) {
  // Renumber ords AND refresh ↑/↓ disabled state — first row can't go up, last
  // can't go down. Runs after every add / delete / drag / button-move, so the
  // end-stops stay correct without each row tracking its own neighbours.
  const rows = parent.querySelectorAll('.rl-entry');
  rows.forEach((en, i) => {
    const o = en.querySelector('.rl-ord');
    if (o) o.textContent = String(i + 1);
    const up = en.querySelector('.rl-move-up');
    const dn = en.querySelector('.rl-move-down');
    if (up) up.disabled = (i === 0);
    if (dn) dn.disabled = (i === rows.length - 1);
  });
}

function _makeEntryRow(rule, entry, commitData, onEnter) {
  entry = entry || {};
  const row = document.createElement('div');
  row.className = 'rl-entry';
  function rebuild() {
    const parent = row.parentNode;
    if (!parent) return;
    rule.entries = _readEntries(parent);
    commitData();
  }
  // rail: drag grip + ▲/ordinal/▼ stepper for one-position nudges; the ordinal
  // and end-stops are maintained by _renumberEntries.
  const rail = document.createElement('div');
  rail.className = 'rl-rail';
  const grip = document.createElement('span');
  grip.className = 'rl-grip'; grip.textContent = '⠿';
  grip.title = 'drag to reorder'; grip.draggable = true;
  let dragOrigNext = null;   // the row's slot at dragstart, for a cancelled drag
  grip.addEventListener('dragstart', (e) => {
    _rlDragSrc = row;
    dragOrigNext = row.nextElementSibling;
    e.dataTransfer.effectAllowed = 'move';
    // Firefox starts a drag only once SOME data is set. A private type, not
    // text/plain: every text field accepts a text drop natively, so a release
    // over another card's pattern input would insert the payload there and
    // count as a real drop (no cancel restore below).
    try { e.dataTransfer.setData('application/x-fwb-regex-entry', '1'); } catch (_) {}
    DRAG_AUTOSCROLL.begin(row.parentNode);
    setTimeout(() => row.classList.add('rl-dragging'), 0);
  });
  grip.addEventListener('dragend', (e) => {
    row.classList.remove('rl-dragging'); _rlDragSrc = null;
    DRAG_AUTOSCROLL.stop();
    const parent = row.parentNode;
    if (!parent) return;
    // dropEffect 'none' = Esc or a release outside the list: the platform's
    // cancel. dragover already moved the row live, so put it back in its
    // dragstart slot and leave rule.entries (and the dirty flag) alone.
    if (e.dataTransfer && e.dataTransfer.dropEffect === 'none') {
      parent.insertBefore(row, dragOrigNext && dragOrigNext.parentNode === parent
        ? dragOrigNext : null);
      _renumberEntries(parent);
      return;
    }
    // A real drop: sync rule.entries with the moved DOM (idempotent after an
    // in-list drop, which already committed).
    rule.entries = _readEntries(parent); _renumberEntries(parent); commitData();
  });
  const ord = document.createElement('span');
  ord.className = 'rl-ord'; ord.textContent = '0';
  // ↑/↓ move the NEIGHBOUR, not this row: reinserting the row would blur the
  // button the keyboard user just pressed, dropping focus to <body>. A move onto
  // an end-stop disables the pressed button (same focus loss), so hand focus to
  // the row's other arrow then.
  function _rlMove(delta) {
    const parent = row.parentNode;
    if (!parent) return;
    const pressed = delta < 0 ? upBtn : dnBtn;
    const hadFocus = document.activeElement === pressed;
    if (delta < 0) {
      const prev = row.previousElementSibling;
      if (prev) parent.insertBefore(prev, row.nextElementSibling);
    } else {
      const next = row.nextElementSibling;
      if (next) parent.insertBefore(next, row);
    }
    rule.entries = _readEntries(parent);
    _renumberEntries(parent);
    if (hadFocus && pressed.disabled) (pressed === upBtn ? dnBtn : upBtn).focus();
    commitData();
  }
  const stepper = document.createElement('div');
  stepper.className = 'rl-stepper';
  const upBtn = document.createElement('button');
  upBtn.type = 'button'; upBtn.className = 'rl-move rl-move-up';
  upBtn.innerHTML = MOVE_CHEV_UP; upBtn.title = 'Move up one position';
  upBtn.setAttribute('aria-label', upBtn.title);
  upBtn.addEventListener('click', () => { if (!upBtn.disabled) _rlMove(-1); });
  const dnBtn = document.createElement('button');
  dnBtn.type = 'button'; dnBtn.className = 'rl-move rl-move-down';
  dnBtn.innerHTML = MOVE_CHEV_DOWN; dnBtn.title = 'Move down one position';
  dnBtn.setAttribute('aria-label', dnBtn.title);
  dnBtn.addEventListener('click', () => { if (!dnBtn.disabled) _rlMove(1); });
  stepper.appendChild(upBtn); stepper.appendChild(ord); stepper.appendChild(dnBtn);
  rail.appendChild(grip); rail.appendChild(stepper);
  const body = document.createElement('div');
  body.className = 'rl-ebody';
  const r1 = document.createElement('div');
  r1.className = 'rl-erow1';
  const lwrap = document.createElement('div');
  lwrap.className = 'rl-elabelwrap';
  const li = document.createElement('input');
  li.type = 'text'; li.className = 'e-label';
  li.spellcheck = false; li.autocomplete = 'off';
  li.placeholder = 'label (optional)';
  li.value = _esc(entry.label || '');
  li.addEventListener('input', rebuild);
  _bindEnterCommit(li, onEnter);
  lwrap.appendChild(li);
  const noteWrap = document.createElement('div');
  noteWrap.className = 'rl-enote';
  const ni = document.createElement('textarea');
  ni.className = 'e-note'; ni.rows = 2; ni.spellcheck = false;
  ni.placeholder = 'note (optional)';
  ni.value = entry.note || '';  // textarea keeps real newlines — no _esc
  ni.addEventListener('input', () => { rebuild(); paintNote(); });
  noteWrap.appendChild(ni);
  const noteBtn = document.createElement('button');
  noteBtn.type = 'button'; noteBtn.className = 'rl-notebtn';
  let noteOpen = !!entry.note;
  function paintNote() {
    noteWrap.style.display = noteOpen ? 'block' : 'none';
    const has = !!ni.value;
    noteBtn.textContent = has ? '⌄ note ●' : '⌄ note';
    noteBtn.classList.toggle('has-note', has);
  }
  noteBtn.addEventListener('click', () => { noteOpen = !noteOpen; paintNote(); });
  const del = document.createElement('button');
  del.type = 'button'; del.className = 'rl-del'; del.textContent = '×';
  del.title = 'delete entry';
  del.addEventListener('click', () => {
    const parent = row.parentNode;
    row.remove();
    if (parent) { rule.entries = _readEntries(parent); _renumberEntries(parent); commitData(); }
  });
  r1.appendChild(lwrap); r1.appendChild(noteBtn); r1.appendChild(del);
  function frRow(gut, cls, val, ph) {
    const fr = document.createElement('div');
    fr.className = 'rl-fr';
    const g = document.createElement('span');
    g.className = 'rl-gut'; g.textContent = gut;
    const inp = document.createElement('input');
    inp.type = 'text'; inp.className = cls;
    inp.spellcheck = false; inp.autocomplete = 'off';
    inp.value = val == null ? '' : val;  // regex source: verbatim, no _esc (see _readEntries)
    inp.placeholder = ph;
    inp.addEventListener('input', rebuild);
    _bindEnterCommit(inp, onEnter);
    fr.appendChild(g); fr.appendChild(inp);
    return fr;
  }
  body.appendChild(r1);
  body.appendChild(frRow('find', 'e-pattern', entry.pattern, 'regex pattern (required)'));
  body.appendChild(frRow('→ repl', 'e-repl', entry.replacement, '(empty = delete match)'));
  body.appendChild(noteWrap);
  paintNote();
  row.appendChild(rail); row.appendChild(body);
  return row;
}

function renderTypeEditor(rule, commitData, opts) {
  // opts (all optional):
  //   datalistId      — cb:map key autocomplete (/quick-config).
  //   makeSaveBtn     — `() => HTMLButtonElement`; adds a per-rule Save button
  //                     (/quick-config; admin /settings has no per-card Save).
  //   commitOnEnter   — fired by Enter in any text input (after a one-tick
  //                     guard for native datalist picks); not the wordlist
  //                     textarea, where Enter inserts a newline.
  //   showNote / showRepeat — admin-only fields (see below).
  //   showMapDates / collapseMapAfter — cb:map date column / fold older rows.
  opts = opts || {};
  const onEnter = opts.commitOnEnter;
  const box = document.createElement('div');
  box.className = 'rule-editor';

  // Rule rationale / documentation. Only the admin /settings editor passes
  // `showNote` — /quick-config omits it (non-admin users cannot patch the
  // `note` field, see _PATCH_ALLOWED_FIELDS in quick_config/routes.py).
  if (opts.showNote) {
    box.appendChild(_helpDiv('note (rationale / documentation):'));
    const noteTa = document.createElement('textarea');
    noteTa.className = 'rule-note';
    noteTa.rows = 2;
    noteTa.spellcheck = false;
    noteTa.style.width = '100%';
    noteTa.style.boxSizing = 'border-box';
    noteTa.value = rule.note == null ? '' : rule.note;
    noteTa.addEventListener('input', () => { rule.note = noteTa.value; commitData(); });
    box.appendChild(noteTa);
  }

  if (rule.type === 'terminal') {
    box.appendChild(_helpDiv('Hardcoded terminal step: lstrip(" \\t\\r") + rstrip(" \\t\\r"). '
      + 'Always runs last. Preserves a leading or trailing newline ("\\n") emitted by '
      + '"neue Zeile" / "neuer Absatz" at the edges of the utterance.'));
    return box;
  }

  // "Repeat until nothing changes" (schema _RepeatableRule.repeat). Admin
  // only, like the note: /quick-config cannot patch it (_PATCH_ALLOWED_FIELDS).
  // Off is deleted, never stored as false — the server dumps it absent, and
  // the editor's fingerprint of every rule without it must not move.
  if (opts.showRepeat) {
    const repWrap = document.createElement('label');
    repWrap.className = 'rule-repeat';
    const repCb = document.createElement('input');
    repCb.type = 'checkbox';
    repCb.className = 'rule-repeat-cb';
    repCb.checked = rule.repeat === true;
    repCb.addEventListener('change', () => {
      if (repCb.checked) rule.repeat = true;
      else delete rule.repeat;
      commitData();
    });
    repWrap.appendChild(repCb);
    const repTxt = document.createElement('span');
    repTxt.textContent = ' Repeat until nothing changes';
    repWrap.appendChild(repTxt);
    box.appendChild(repWrap);
    box.appendChild(_helpDiv('Runs the rule again on its own output (a list: all its entries) '
      + 'until the text stops changing, at most __REPEAT_CAP__ times. For matches that overlap: '
      + 'one pass sees "Hals-nasen-ohren-arzt" only up to the first hyphen it changed.'));
  }

  function _saveBtn() {
    const b = opts.makeSaveBtn();
    b.style.minWidth = '6rem';
    return b;
  }

  // Save row for rule types without a "+ add entry" bar (regex-list and
  // cb:map put Save next to their add button instead).
  function _appendSaveRow(parent) {
    if (!opts.makeSaveBtn) return;
    const saveRow = document.createElement('div');
    saveRow.style.cssText = 'display:flex;justify-content:flex-end;margin-top:0.6rem;';
    saveRow.appendChild(_saveBtn());
    parent.appendChild(saveRow);
  }

  if (rule.type === 'regex-list') {
    box.appendChild(_helpDiv('Ordered find→replace list — entries run top-to-bottom '
      + '(drag ⠿ to reorder). An empty replacement deletes the match.'));
    const entriesBox = document.createElement('div');
    entriesBox.className = 'rl-entries';
    (rule.entries || []).forEach((en) => {
      entriesBox.appendChild(_makeEntryRow(rule, en, commitData, onEnter));
    });
    // Self-contained drag-reorder (the admin DnD closure is out of scope from
    // this shared script). Only react to a drag that started in THIS list.
    entriesBox.addEventListener('dragover', (e) => {
      if (!_rlDragSrc || _rlDragSrc.parentNode !== entriesBox) return;
      e.preventDefault();
      DRAG_AUTOSCROLL.move(e.clientY);
      const after = Array.from(
        entriesBox.querySelectorAll('.rl-entry:not(.rl-dragging)')
      ).find((en) => {
        const r = en.getBoundingClientRect();
        return e.clientY < r.top + r.height / 2;
      });
      if (after) entriesBox.insertBefore(_rlDragSrc, after);
      else entriesBox.appendChild(_rlDragSrc);
      _renumberEntries(entriesBox);
    });
    entriesBox.addEventListener('drop', (e) => {
      if (!_rlDragSrc || _rlDragSrc.parentNode !== entriesBox) return;
      e.preventDefault();
      DRAG_AUTOSCROLL.stop();
      rule.entries = _readEntries(entriesBox);
      _renumberEntries(entriesBox);
      commitData();
    });
    box.appendChild(entriesBox);
    _renumberEntries(entriesBox);
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.textContent = '+ add entry';
    addBtn.style.flex = '1';
    addBtn.addEventListener('click', () => {
      if (!rule.entries) rule.entries = [];
      const row = _makeEntryRow(rule, {}, commitData, onEnter);
      entriesBox.appendChild(row);
      _renumberEntries(entriesBox);
      const pi = row.querySelector('.e-pattern');
      if (pi) pi.focus();
    });
    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display:flex;gap:0.5rem;align-items:stretch;margin-top:0.6rem;';
    btnRow.appendChild(addBtn);
    if (opts.makeSaveBtn) btnRow.appendChild(_saveBtn());
    box.appendChild(btnRow);
    return box;
  }

  if (rule.type === 'callback:lowercase-wordlist') {
    box.appendChild(_makeMonoLabeledInput('pattern', rule.pattern, (v) => {
      rule.pattern = v; commitData();
    }, onEnter));
    box.appendChild(_helpDiv('Wordlist (one entry per line, case-insensitive):'));
    const ta = document.createElement('textarea');
    ta.value = (rule.wordlist || []).join('\n');
    ta.rows = 6;
    ta.addEventListener('input', () => {
      rule.wordlist = ta.value.split('\n').map(s => s.trim()).filter(Boolean);
      commitData();
    });
    // Intentionally NO Enter binding on the wordlist textarea — Enter must
    // insert a newline so users can edit multi-line lists.
    box.appendChild(ta);
    _appendSaveRow(box);
    return box;
  }

  if (rule.type === 'callback:map') {
    box.appendChild(_helpDiv('Pattern auto-built from map keys (longest-first, '
      + 'word-bounded, case-insensitive). Edit entries below.'));
    // "n / cap" readout so a full dictionary is visible BEFORE a save
    // bounces off the server's entry cap. __mme defaults to the MapRule cap
    // (top of this file); /quick-config's load() overwrites it from the API.
    const mapCap = (typeof window.__mme === 'number') ? window.__mme : 0;
    const cnt = box.appendChild(_helpDiv());  // text painted by paintCount() once the table exists
    const showDate = !!opts.showMapDates;
    const meta = rule.map_meta || {};
    const tbl = document.createElement('table');
    tbl.className = 'map-table';
    tbl.style.width = '100%';
    // Order by map_meta (added / last-updated), oldest first → newest last, so
    // the freshest entries sit next to the "+ add entry" bar. Un-stamped keys
    // (factory entries never edited here) sort as oldest, then alphabetically.
    const rows = Object.entries(rule.map || {}).sort((a, b) => {
      const ta = meta[a[0]] || 0, tb = meta[b[0]] || 0;
      if (ta !== tb) return ta - tb;
      return _coll.compare(a[0], b[0]);
    });
    const collapseAfter = opts.collapseMapAfter || 0;
    const hiddenCount = (collapseAfter && rows.length > collapseAfter)
      ? rows.length - collapseAfter : 0;
    if (hiddenCount) {
      const toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'map-toggle';
      let shown = false;
      const paint = () => {
        tbl.classList.toggle('show-all', shown);
        toggle.textContent = shown
          ? '▾ Hide ' + hiddenCount + ' older'
          : '▸ Show ' + hiddenCount + ' older mapping' + (hiddenCount === 1 ? '' : 's');
      };
      toggle.addEventListener('click', () => { shown = !shown; paint(); });
      paint();
      box.appendChild(toggle);
    }
    rows.forEach(([k, v], i) => {
      const tr = _makeMapRow(rule, k, v, commitData, opts.datalistId, onEnter, meta[k] || 0, showDate);
      // Hide the oldest `hiddenCount` rows behind the toggle (CSS display:none,
      // so _readMap still reads every row when building the patch).
      if (i < hiddenCount) tr.classList.add('map-row-collapsed');
      tbl.appendChild(tr);
    });
    box.appendChild(tbl);
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.textContent = '+ add entry';
    addBtn.style.flex = '1';
    addBtn.addEventListener('click', () => {
      // Append directly (no re-render) so the row body stays expanded and
      // other rows keep their input state. No placeholder key and no
      // commitData(): _readMap skips empty-key rows, and the first keystroke
      // commits.
      if (!rule.map) rule.map = {};
      const newTr = _makeMapRow(rule, '', '', commitData, opts.datalistId, onEnter, 0, opts.showMapDates);
      tbl.appendChild(newTr);
      // Open the autocomplete straight away. showPicker() must run
      // synchronously in this click handler to keep transient activation
      // (Chrome 99+, Firefox 149+; older browsers just focus).
      const ki = newTr.querySelector('td:first-child input');
      if (ki) {
        ki.focus();
        try { ki.showPicker(); } catch (_) { /* unsupported */ }
      }
    });
    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display:flex;gap:0.5rem;align-items:stretch;margin-top:0.6rem;';
    btnRow.appendChild(addBtn);
    if (opts.makeSaveBtn) btnRow.appendChild(_saveBtn());
    box.appendChild(btnRow);
    // Row add/remove never re-renders the editor, so repaint on DOM changes.
    // Counts <tr>s (not rule.map) so a blank just-added row counts; a
    // MutationObserver because × detaches its row and no event bubbles to tbl.
    const paintCount = () => {
      const n = tbl.querySelectorAll('tr').length;
      cnt.textContent = n + (mapCap ? ' / ' + mapCap : '') + ' entries'
        + ((mapCap && n >= mapCap)
           ? ' — full: delete entries before adding new ones' : '');
      addBtn.disabled = !!(mapCap && n >= mapCap);
    };
    new MutationObserver(paintCount).observe(tbl, { childList: true });
    paintCount();
    return box;
  }

  if (rule.type === 'callback:dedup' || rule.type === 'callback:upper') {
    box.appendChild(_makeMonoLabeledInput('pattern', rule.pattern, (v) => {
      rule.pattern = v; commitData();
    }, onEnter));
    box.appendChild(_helpDiv(rule.type === 'callback:dedup'
      ? 'Callback: collapse each match — last non-comma wins; pure-comma run → single comma.'
      : 'Callback: uppercase group(2) (or whole match if pattern has fewer than 2 groups).'));
    _appendSaveRow(box);
    return box;
  }

  return box;
}
</script>
