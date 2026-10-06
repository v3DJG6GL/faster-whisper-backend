"""Source pins for the /logs viewer script (admin/logs_routes.py serves it)."""


def test_logs_page_search_skips_fold_controls_and_button_labels(client):
    """The search filter matched el.textContent, which also holds the fold
    controls' "▸ N more segment rows / show all" and a clamped line's
    "Show all · N chars" button, so "show" or "rows" hit every control and
    clipped line, and hide mode hid the controls that reveal folded rows."""
    html = client.get("/logs").text
    f = html[html.index("function applyFilter(el)"):]
    f = f[:f.index("\n  }\n")]
    assert "el.classList.contains('fold-ctl')" in f
    assert "_lineText.get(el)" in f
    assert "el.textContent.toLowerCase()" not in f
    # makeLine records the line's own text for the filter to read.
    m = html[html.index("function makeLine(line, st)"):]
    assert "_lineText.set(el, txt);" in m[:m.index("\n  }\n")]
