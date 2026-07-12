"""The three WebUI version surfaces render the build version:
header vtag (every shared-header page) / hub build line / settings card."""

import build_info


def test_header_vtag_on_shared_header_pages(client):
    r = client.get("/logs")
    assert r.status_code == 200
    assert 'id="hdr-vtag"' in r.text
    assert build_info.VERSION_SHORT in r.text
    # The placeholder must be substituted, never leak literally.
    assert "{{HEADER_VTAG}}" not in r.text


def test_hub_hero_build_line(client):
    r = client.get("/")
    assert r.status_code == 200
    assert 'class="buildline"' in r.text
    assert build_info.APP_VERSION in r.text


def test_settings_identity_card(client):
    r = client.get("/settings")
    assert r.status_code == 200
    assert 'id="srv-ident"' in r.text
    assert build_info.APP_VERSION in r.text
    assert "copy report" in r.text


def test_pipeline_view_has_no_identity_card(client):
    r = client.get("/settings/pipeline")
    assert r.status_code == 200
    assert 'id="srv-ident"' not in r.text
    assert "{{SERVER_IDENT}}" not in r.text
