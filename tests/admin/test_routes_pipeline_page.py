"""The dedicated /settings/pipeline page reuses the /settings shell; a
`settings-view` meta flag picks which sections the page renders."""


def test_pipeline_page_loopback(client):
    r = client.get("/settings/pipeline")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    # The IIFE branches on this meta to render ONLY the Pipeline section.
    assert '<meta name="settings-view" content="pipeline">' in r.text


def test_settings_page_marks_full_view(client):
    # The same shell on /settings advertises the complementary view so the
    # IIFE renders every section EXCEPT Pipeline (replaced by a link).
    r = client.get("/settings")
    assert r.status_code == 200
    assert '<meta name="settings-view" content="settings">' in r.text


def test_pipeline_tab_in_nav(client):
    r = client.get("/settings")
    assert 'href="/settings/pipeline"' in r.text


def test_pipeline_page_no_store(client):
    r = client.get("/settings/pipeline")
    assert "no-store" in r.headers.get("cache-control", "")


def test_view_branching_and_stub_ship_in_shell(client):
    # The stub is built client-side, so only its code/strings can be asserted.
    r = client.get("/settings/pipeline")
    assert "PIPELINE_ONLY" in r.text
    assert "pipeline-moved-stub" in r.text
    assert "Open the Pipeline page" in r.text


def test_settings_keeps_editor_code_and_links_to_pipeline(client):
    # /settings still ships the rule editor (the per-model + capture scoping
    # pickers reuse makeRuleListEditor) and links to the dedicated page.
    r = client.get("/settings")
    assert "makeRuleListEditor" in r.text
    assert "/settings/pipeline" in r.text


def test_redesigned_card_controls_present(client):
    r = client.get("/settings/pipeline")
    for needle in ("rule-rail", "lock-btn", "color-btn", "color-pop", "head-actions"):
        assert needle in r.text, needle


def test_card_colour_dot_is_built_as_a_node(client):
    # rule.color reaches the dot as a dataset value, never spliced into an
    # innerHTML string (same <span class="color-dot" data-color> markup).
    r = client.get("/settings/pipeline")
    assert "colorBtn.innerHTML" not in r.text
    assert "_dot.dataset.color = rule.color || '';" in r.text
