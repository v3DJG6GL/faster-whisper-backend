"""The three WebUI version surfaces render the build version:
header vtag (every shared-header page) / hub build line / settings card."""

import build_info
import config as cfg


def test_header_vtag_shell_carries_no_facts(client):
    """The shared header rides host-gated (keyless) pages, so the chip ships
    as an empty shell — the facts arrive via /auth/whoami."""
    r = client.get("/logs")
    assert r.status_code == 200
    assert 'id="hdr-vtag"' in r.text
    assert 'onclick="_fwCopyBuild(this)"></button>' in r.text
    assert build_info.APP_VERSION not in r.text
    assert build_info.BOOT_ID[:8] not in r.text
    # The placeholder must be substituted, never leak literally.
    assert "{{HEADER_VTAG}}" not in r.text


def test_whoami_carries_build_fields(client):
    """The chip's values ride the authenticated whoami payload, as data (not
    pre-rendered HTML) — the page writes them in with textContent."""
    b = client.get("/auth/whoami").json()["build"]
    assert b["version"] == build_info.APP_VERSION
    assert b["version_short"] == build_info.VERSION_SHORT
    assert b["boot"] == build_info.BOOT_ID[:8]
    assert b["started"] == build_info.STARTED_UTC
    assert b["server"] == build_info.SERVER_NAME


def test_shared_header_page_leaks_no_build_to_unauthenticated_caller(
    client, make_user_key,
):
    """Locked down + no credential: the page shell still renders (it is only
    host-gated) but must not disclose the version or the boot id."""
    make_user_key("admin", is_admin=True)
    r = client.get("/logs")
    assert r.status_code == 200
    assert build_info.APP_VERSION not in r.text
    assert build_info.BOOT_ID[:8] not in r.text
    assert client.get("/auth/whoami").status_code == 401


def test_hub_hero_build_line_shell_carries_no_facts(client):
    """The hub is host-gated but keyless, so the hero's build caption ships as
    an empty shell — the facts arrive via /auth/whoami, like the header chip."""
    r = client.get("/")
    assert r.status_code == 200
    assert '<p class="buildline"></p>' in r.text
    assert build_info.APP_VERSION not in r.text
    assert build_info.BOOT_ID[:8] not in r.text
    # The placeholder must be gone, never leak literally.
    assert "{{BUILD_LINE}}" not in r.text


def test_settings_identity_card_shell_carries_no_facts(client):
    """/settings is host-gated but keyless, so the page ships only the card's
    static chrome — the paths/engine facts must NOT be server-rendered."""
    r = client.get("/settings")
    assert r.status_code == 200
    assert 'id="srv-ident"' in r.text
    assert "copy report" in r.text
    assert cfg._DATA_DIR not in r.text
    assert cfg._DB_DIR not in r.text
    assert build_info.engine_versions() not in r.text


def test_settings_state_carries_identity_fields(client):
    """The card's values ride the admin-gated state route instead, as data
    (not pre-rendered HTML) — the page writes them in with textContent."""
    ident = client.get("/settings/state").json()["server_ident"]
    assert ident["version"] == build_info.APP_VERSION
    assert ident["engine"] == build_info.engine_versions()
    assert ident["data_dir"] == cfg._DATA_DIR
    assert ident["db_dir"] == cfg._DB_DIR
    assert build_info.APP_VERSION in ident["report"]


def test_pipeline_view_has_no_identity_card(client):
    r = client.get("/settings/pipeline")
    assert r.status_code == 200
    assert 'id="srv-ident"' not in r.text
    assert "{{SERVER_IDENT}}" not in r.text
