"""Phone nav drawer (NAV_DRAWER_JS): open marks the rest of the page inert,
so every way the drawer can stop being reachable must close it.

The script runs under node against a minimal DOM stub (header with toggle,
nav row and backdrop, plus a <main> sibling) — enough for open()/close() and
the inert bookkeeping, no browser needed."""

import json
import shutil
import subprocess

import pytest

from faster_whisper_backend.core import web_common


_DOM_STUB = r"""
function el(tag) {
  const e = {
    tag, children: [], parentElement: null, attrs: {}, listeners: {},
    offsetParent: {},
    classList: {
      s: new Set(),
      add(c) { this.s.add(c); }, remove(c) { this.s.delete(c); },
      contains(c) { return this.s.has(c); },
    },
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
    hasAttribute(k) { return k in this.attrs; },
    removeAttribute(k) { delete this.attrs[k]; },
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
    fire(t, ev) { (this.listeners[t] || []).forEach(f => f(ev || {})); },
    appendChild(c) { c.parentElement = this; this.children.push(c); return c; },
    querySelectorAll() { return []; },
    closest() { return null; },
    focus() { document.activeElement = this; },
  };
  return e;
}
const body = el('body'), hdr = el('header'), main = el('main');
const btn = el('button'), nav = el('nav'), bd = el('div');
body.appendChild(hdr); body.appendChild(main);
hdr.appendChild(btn); hdr.appendChild(nav); hdr.appendChild(bd);
const docListeners = {};
global.document = {
  body, activeElement: body,
  querySelector(s) {
    return { header: hdr, '.nav-toggle': btn, '.nav-backdrop': bd }[s] || null;
  },
  getElementById(id) { return id === 'navrow' ? nav : null; },
  addEventListener(t, f) { docListeners[t] = f; },
  removeEventListener(t) { delete docListeners[t]; },
};
const mqs = {};
global.window = {
  matchMedia(q) {
    const mq = mqs[q] = mqs[q] || el('mql');
    return mq;
  },
};
"""


def _run(drive: str) -> dict:
    js = web_common.NAV_DRAWER_JS.strip()
    js = js[len("<script>"):-len("</script>")]
    out = subprocess.run(["node", "-"], input=_DOM_STUB + js + drive,
                         capture_output=True, text=True, encoding="utf-8", timeout=30, check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_drawer_closes_and_releases_inert_when_viewport_leaves_breakpoint():
    """Rotating a phone to landscape widens past the 40em drawer breakpoint:
    the toggle and backdrop vanish, but <main> stayed inert, so no tap
    reached the page until a reload."""
    res = _run(r"""
btn.fire('click');
const openInert = main.hasAttribute('inert');
const mq = mqs['(max-width: 40em)'];
if (mq) mq.fire('change', { matches: false });
console.log(JSON.stringify({
  openInert,
  inert: main.hasAttribute('inert'),
  open: hdr.classList.contains('nav-open'),
  expanded: btn.getAttribute('aria-expanded'),
}));
""")
    assert res == {"openInert": True, "inert": False, "open": False,
                   "expanded": "false"}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_drawer_stays_open_while_still_inside_breakpoint():
    res = _run(r"""
btn.fire('click');
mqs['(max-width: 40em)'].fire('change', { matches: true });
console.log(JSON.stringify({
  inert: main.hasAttribute('inert'), open: hdr.classList.contains('nav-open'),
}));
""")
    assert res == {"inert": True, "open": True}


def test_drawer_breakpoint_matches_nav_css():
    # The JS query must track the CSS breakpoint that shows the toggle.
    assert "@media (max-width: 40em) {\n  .nav-toggle { display: inline-flex;" \
        in web_common.NAV_CSS
    assert "matchMedia('(max-width: 40em)')" in web_common.NAV_DRAWER_JS
