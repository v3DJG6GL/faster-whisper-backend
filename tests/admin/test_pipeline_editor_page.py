"""Source pins for the /settings/pipeline editor script (inline JS; the
behaviour itself was exercised in headless Chromium when it was written)."""


def _html(client):
    r = client.get("/settings/pipeline")
    assert r.status_code == 200
    return r.text


def test_promote_diff_is_per_entry(client):
    """A map / entry list used to render as ONE JSON line per side, which hid
    what changed in an 80-key dictation map. The diff walks keys and entries."""
    html = _html(client)
    assert "addRow(k + ' · ' + JSON.stringify(key), js(o[key]), js(n[key]));" in html
    assert "'order changed only'" in html
    assert "oEl.textContent = ov;" not in html


def test_new_config_json_rules_can_be_added_to_a_local_list(client):
    """A saved local rule list replaces the factory list, so a rule that an
    update adds to config.json stayed invisible (2026-09-19: es-punctuation)."""
    html = _html(client)
    assert "function _missingFactoryRules()" in html
    assert "from config.json';" in html
    assert "rules.splice(at, 0, copy);" in html
