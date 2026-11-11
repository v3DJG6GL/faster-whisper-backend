"""reports.html "pipeline" line of a report card, driven under node.

A report's ``stages`` are stored as the submitter sent them, so a stage field
may be an object whose toString/valueOf is not callable. Building the line
must not throw on one (render() builds every card in one forEach, so a throw
loses the whole list), and every primitive stage renders exactly as before."""

from tests._page_helpers import js_upto, needs_node, node_json
from tests.conftest import read_repo_text


def _harness(stages_json: str) -> str:
    html = read_repo_text("faster_whisper_backend", "reports", "templates", "reports.html")
    block = js_upto(html, "    var stages = Array.isArray(r.stages)",
                    "    card.appendChild(textLine('raw'")
    helper = ""
    if "  function _prim(v) {" in html:
        helper = js_upto(html, "  function _prim(v) {", "\n  }\n") + "\n  }\n"
    return helper + r"""
function line(stagesList) {
  var r = { stages: stagesList };
  var out = null;
  var card = { appendChild(x) { out = x; } };
  function textLine(klass, label, value) { return value; }
""" + block + r"""
  return out;
}
const cases = """ + stages_json + r""";
const res = cases.map(function(c) {
  try { return { ok: line(c) }; } catch (e) { return { err: String(e && e.message) }; }
});
console.log(JSON.stringify(res));
"""


@needs_node
def test_object_stage_fields_do_not_throw():
    out = node_json(_harness(r"""[
  [{ name: { toString: 1 } }],
  [{ name: 'x', model: { toString: 1 } }],
  [{ name: 'x', detail: { valueOf: 1, toString: 1 } }],
  [{ name: { toString: 1 } }, { name: 'y', model: 'm' }],
  [{ name: ['a', 'b'] }, { name: 'z', model: { a: 1 }, detail: [1] }]
]"""))
    assert out == [
        {"ok": None},
        {"ok": "x"},
        {"ok": "x"},
        {"ok": "y (m)"},
        {"ok": "z"},
    ]


@needs_node
def test_primitive_stages_render_as_before():
    out = node_json(_harness(r"""[
  [{ name: 'transcribe', model: 'large-v3' },
   { name: 'translate', model: 'm', detail: 'de→fr' }],
  [{ name: 'x', model: 0, detail: true }],
  [{ name: 0 }],
  [{ name: 7, model: 1.5, detail: false }],
  [null, 'str', 5, { name: '' }, { name: 'ok' }],
  []
]"""))
    assert out == [
        {"ok": "transcribe (large-v3)   →   translate (m) · de→fr"},
        {"ok": "x · true"},
        {"ok": None},
        {"ok": "7 (1.5)"},
        {"ok": "ok"},
        {"ok": None},
    ]
