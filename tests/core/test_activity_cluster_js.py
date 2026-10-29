"""Header activity cluster (ACTIVITY_CLUSTER_JS) driven under node.

The script runs against a minimal DOM stub: the popover's innerHTML setter
rebuilds its cancel buttons / footer link from the markup and detaches the
old nodes (blurring a focused one, like a browser), EventSource and fetch are
fakes the drive script controls."""

import json
import shutil
import subprocess

import pytest

from faster_whisper_backend.core import web_common


_DOM_STUB = r"""
const body = { tag: 'body', parentElement: null };
function el(tag, cls, attrs) {
  const e = {
    tag, parentElement: null, attrs: attrs || {}, listeners: {}, kids: [],
    hidden: false, disabled: false, textContent: '', title: '', style: {},
    classList: {
      s: new Set(cls ? cls.split(' ') : []),
      add(c) { this.s.add(c); }, remove(c) { this.s.delete(c); },
      toggle(c, on) { if (on === undefined) on = !this.s.has(c);
                      if (on) this.s.add(c); else this.s.delete(c); return on; },
      contains(c) { return this.s.has(c); },
    },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    hasAttribute(k) { return k in this.attrs; },
    removeAttribute(k) { delete this.attrs[k]; },
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
    contains(n) { for (; n; n = n.parentElement) if (n === this) return true; return false; },
    focus() { if (!this.disabled) document.activeElement = this; },
  };
  return e;
}
const hdr = el('header');
const statsLink = el('a', 'page-link allowed', { 'data-page': 'stats' });
const btn = el('button', 'hdr-activity page-link', { 'data-page': 'stats' });
btn.hidden = true;
const pop = el('div', 'hact-pop');
pop.hidden = true;
let html = '';
Object.defineProperty(pop, 'innerHTML', {
  get() { return html; },
  set(v) {
    html = v;
    pop.kids.forEach(k => {
      if (k.contains(document.activeElement)) document.activeElement = body;
      k.parentElement = null;
    });
    pop.kids = [];
    const re = /<button class="hact-cancel"( disabled)? data-pid="([^"]*)"/g;
    let m;
    while ((m = re.exec(v))) {
      const b = el('button', 'hact-cancel', { 'data-pid': m[2] });
      b.disabled = !!m[1]; b.parentElement = pop; pop.kids.push(b);
    }
    if (v.indexOf('<div class="hact-foot"><a') >= 0) {
      const a = el('a'); a.parentElement = pop; pop.kids.push(a); pop.foot = a;
    }
  },
});
pop.querySelectorAll = sel => sel === '.hact-cancel'
  ? pop.kids.filter(k => k.classList.contains('hact-cancel')) : [];
pop.querySelector = sel => sel === '.hact-foot a' ? (pop.foot && pop.foot.parentElement ? pop.foot : null) : null;
[statsLink, btn, pop].forEach(n => { n.parentElement = hdr; });
hdr.parentElement = body;
const ids = { hact: btn, 'hact-pop': pop };
['hact-jobs', 'hact-gpu', 'hact-gpuv', 'hact-vram', 'hact-vramv'].forEach(
  i => { ids[i] = el('span'); ids[i].parentElement = btn; });
const docListeners = {};
global.document = {
  body, activeElement: body, visibilityState: 'visible',
  getElementById(id) { return ids[id] || null; },
  querySelector(s) {
    if (s === 'header') return hdr;
    if (s === 'header a.page-link[data-page="stats"]') return statsLink;
    return null;
  },
  addEventListener(t, f) { (docListeners[t] = docListeners[t] || []).push(f); },
};
global.window = globalThis;
const winListeners = {};
window.addEventListener = (t, f) => { (winListeners[t] = winListeners[t] || []).push(f); };
window._csrfToken = () => 't';
global.MutationObserver = function () { this.observe = () => {}; };
global.setInterval = () => 0;
const streams = [];
global.EventSource = function (url) { this.url = url; this.readyState = 1;
  this.close = () => { this.readyState = 2; }; streams.push(this); };
let probeStatus = 200;
global.fetch = () => Promise.resolve({ status: probeStatus, ok: true });
const tick = () => new Promise(r => setTimeout(r, 5));
const job = (pid, progress) => ({ kind: 'transcribe', model: 'm',
                                  progress_id: pid, progress });
const openPop = () => btn.listeners.click[0]({ stopPropagation() {} });
"""


def _run(drive: str) -> dict:
    js = web_common.ACTIVITY_CLUSTER_JS.strip()
    js = js[len("<script>"):-len("</script>")]
    out = subprocess.run(["node", "-"],
                         input=_DOM_STUB + js + "\n(async () => {" + drive + "})();",
                         capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_denied_probe_drops_allowed_so_the_css_really_hides_the_button():
    """`.hdr-activity.allowed` / `.page-link.allowed` set display:inline-flex,
    which beats [hidden]: a 401/403 probe that only flipped `hidden` left the
    button painted with frozen numbers. A later auth change restores it."""
    res = _run(r"""
const before = btn.classList.contains('allowed') && !btn.hidden;
probeStatus = 403;
streams[0].readyState = 2; streams[0].onerror();
await tick();
const denied = { hidden: btn.hidden, allowed: btn.classList.contains('allowed'),
                 streams: streams.length };
probeStatus = 200;
winListeners['whisper:auth-changed'][0]();
await tick();
console.log(JSON.stringify({ before, denied,
  back: { hidden: btn.hidden, allowed: btn.classList.contains('allowed'),
          streams: streams.length } }));
""")
    assert res == {"before": True,
                   "denied": {"hidden": True, "allowed": False, "streams": 1},
                   "back": {"hidden": False, "allowed": True, "streams": 2}}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_frame_rebuild_keeps_keyboard_focus_on_the_same_cancel_button():
    """renderPop swaps the popover's innerHTML on every changed 1 Hz frame;
    a keyboard user Tabbed onto a cancel button lost focus to <body>."""
    res = _run(r"""
window._fwFeedActivity({ jobs: [job('p1', 0.41), job('p2', 0.10)] });
openPop();
pop.querySelectorAll('.hact-cancel')[1].focus();
window._fwFeedActivity({ jobs: [job('p1', 0.42), job('p2', 0.11)] });
const a = document.activeElement;
const same = { pid: a.getAttribute && a.getAttribute('data-pid'),
               inPop: pop.contains(a) && a !== pop };
pop.foot.focus();
window._fwFeedActivity({ jobs: [job('p1', 0.43), job('p2', 0.12)] });
const foot = document.activeElement === pop.foot && pop.contains(pop.foot);
pop.querySelectorAll('.hact-cancel')[1].focus();
window._fwFeedActivity({ jobs: [job('p1', 0.44)] });
console.log(JSON.stringify({ same, foot, gone: document.activeElement === btn }));
""")
    assert res == {"same": {"pid": "p2", "inPop": True}, "foot": True, "gone": True}
