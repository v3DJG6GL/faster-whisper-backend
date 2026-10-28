"""Priority+ nav overflow (NAV_OVERFLOW_JS): every relayout (a poller tick
that changes the status rail's width fires the ResizeObserver) used to
re-append every nav link, and moving a focused node blurs it, so a keyboard
user's focus jumped to <body> mid-Tab.

The script runs under node against a minimal DOM stub whose node moves blur
a focused node like a browser does; widths are faked so three 10px links fit
a 25px nav only when one is tucked into the "more" list."""

import json
import shutil
import subprocess

import pytest

from faster_whisper_backend.core import web_common


_DOM_STUB = r"""
const body = { tag: 'body', children: [], parentElement: null };
function rendered(e) {
  for (let n = e; n; n = n.parentElement) {
    if (n === body) return true;
    if (n.hidden) return false;
  }
  return false;
}
function detach(c) {
  const p = c.parentElement;
  if (!p) return;
  if (c.contains(document.activeElement)) document.activeElement = body;
  p.children.splice(p.children.indexOf(c), 1);
  c.parentElement = null;
}
function el(tag, cls) {
  const e = {
    tag, children: [], parentElement: null, attrs: {}, listeners: {},
    hidden: false, textContent: '', scrollWidth: 0, clientWidth: 100,
    classList: {
      s: new Set(cls ? cls.split(' ') : []),
      add(c) { this.s.add(c); }, remove(c) { this.s.delete(c); },
      contains(c) { return this.s.has(c); },
    },
    get isConnected() { return rendered(this) || !!this.parentElement; },
    get offsetParent() { return rendered(this) ? this.parentElement : null; },
    getClientRects() { return rendered(this) ? [{}] : []; },
    getBoundingClientRect() { return { top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0 }; },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
    contains(n) { for (; n; n = n.parentElement) if (n === this) return true; return false; },
    appendChild(c) { return this.insertBefore(c, null); },
    insertBefore(c, ref) {
      detach(c);
      const i = ref ? this.children.indexOf(ref) : this.children.length;
      this.children.splice(i, 0, c); c.parentElement = this; return c;
    },
    get firstChild() { return this.children[0] || null; },
    replaceChildren() { this.children.slice().forEach(detach); },
    querySelectorAll(sel) {
      const out = [], c = sel.slice(1);
      (function walk(n) { n.children.forEach(k => { if (k.classList.contains(c)) out.push(k); walk(k); }); })(this);
      return out;
    },
    querySelector(sel) { return this.querySelectorAll(sel)[0] || null; },
    focus() { document.activeElement = this; },
  };
  return e;
}
const hdr = el('header'), inner = el('div'), nav = el('span', 'navrow');
const more = el('span', 'nav-more'), btn = el('button', 'nav-more-btn');
const cnt = el('span', 'cnt'), list = el('ul', 'nav-more-list');
body.children.push(hdr); hdr.parentElement = body;
hdr.appendChild(inner); inner.appendChild(nav); inner.appendChild(more);
more.appendChild(btn); btn.appendChild(cnt); more.appendChild(list);
list.hidden = true;
const links = ['a', 'b', 'c'].map(t => { const a = el('a', 'navlink'); a.textContent = t; return nav.appendChild(a); });
let navCap = 3;   // links that fit the nav before it overflows
Object.defineProperty(nav, 'scrollWidth', { get() {
  return 10 * nav.children.filter(c => !c.hidden).length; } });
Object.defineProperty(nav, 'clientWidth', { get() { return 10 * navCap + 5; } });
const docListeners = {};
global.document = {
  body, activeElement: body, createElement: el,
  querySelector(s) { return s === 'header' ? hdr : null; },
  getElementById(id) { return id === 'navrow' ? nav : null; },
  addEventListener(t, f) { docListeners[t] = f; },
};
const ros = [];
global.ResizeObserver = function (f) { ros.push(f); this.observe = () => {}; };
global.MutationObserver = function () { this.observe = () => {}; };
global.requestAnimationFrame = f => f();
global.getComputedStyle = () => ({ display: 'none' });
global.window = globalThis;
const relayout = () => ros[0]();
"""


def _run(drive: str) -> dict:
    js = web_common.NAV_OVERFLOW_JS.strip()
    js = js[len("<script>"):-len("</script>")]
    out = subprocess.run(["node", "-"], input=_DOM_STUB + js + drive,
                         capture_output=True, text=True, timeout=30, check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_relayout_keeps_focus_on_a_link_in_the_bar():
    res = _run(r"""
links[1].focus();
relayout();
console.log(JSON.stringify({ focused: document.activeElement.textContent,
                             order: nav.children.map(c => c.textContent) }));
""")
    assert res == {"focused": "b", "order": ["a", "b", "c"]}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_relayout_keeps_focus_on_a_link_in_the_open_more_list():
    res = _run(r"""
navCap = 2; relayout();
btn.listeners.click[0]();
const tucked = list.querySelectorAll('.navlink').map(a => a.textContent);
links[2].focus();
relayout();
console.log(JSON.stringify({ tucked, open: !list.hidden,
                             focused: document.activeElement.textContent }));
""")
    assert res == {"tucked": ["c"], "open": True, "focused": "c"}
