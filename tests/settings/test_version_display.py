"""The three WebUI version surfaces — header vtag (every shared-header page)
/ hub build line / settings card — ship as empty shells on the host-gated,
keyless pages; the build facts arrive as data on the authenticated
/auth/whoami and /settings/state payloads."""

import importlib.metadata
import platform
import re

from faster_whisper_backend import build_info
from faster_whisper_backend.settings import config as cfg

# The header chip is built once at web_common import and the page render is
# lru_cached, and the hub is a static template: the shell tests pin the exact
# empty markup (a short real version such as "dev", the Dockerfile default,
# is a substring of ordinary markup like "device", so an absence check on it
# proves nothing).
_EMPTY_VTAG = ('<button id="hdr-vtag" class="vtag" type="button" data-tip="" '
               'data-build="" aria-label="Copy server build info" '
               'onclick="_fwCopyBuild(this)"></button>')


def test_header_vtag_shell_carries_no_facts(client):
    """The shared header rides host-gated (keyless) pages, so the chip ships
    as an empty shell — the facts arrive via /auth/whoami."""
    r = client.get("/logs")
    assert r.status_code == 200
    assert _EMPTY_VTAG in r.text
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
    assert _EMPTY_VTAG in r.text
    assert build_info.BOOT_ID[:8] not in r.text
    assert client.get("/auth/whoami").status_code == 401


def test_hub_hero_build_line_shell_carries_no_facts(client):
    """The hub is host-gated but keyless, so the hero's build caption ships as
    an empty shell — the facts arrive via /auth/whoami, like the header chip."""
    r = client.get("/")
    assert r.status_code == 200
    assert '<p class="buildline"></p>' in r.text
    assert build_info.BOOT_ID[:8] not in r.text


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
    # Each engine fact on its own (a per-part render would leave the joined
    # string absent), scoped to the card so an unrelated version string
    # elsewhere on the page cannot make this flaky.
    card = re.search(r'<div id="srv-ident">.*?</section></div>', r.text, re.S)
    assert card is not None
    for fact in (platform.python_version(),
                 importlib.metadata.version("faster-whisper"),
                 importlib.metadata.version("ctranslate2"),
                 "faster-whisper", "CTranslate2"):
        assert fact not in card.group(0), fact


def test_settings_state_carries_identity_fields(client):
    """The card's values ride the admin-gated state route instead, as data
    (not pre-rendered HTML) — the page writes them in with textContent."""
    ident = client.get("/settings/state").json()["server_ident"]
    assert ident["version"] == build_info.APP_VERSION
    assert ident["engine"] == build_info.engine_versions()
    assert ident["data_dir"] == cfg._DATA_DIR
    assert ident["db_dir"] == cfg._DB_DIR
    assert build_info.APP_VERSION in ident["report"]


def test_identity_card_reports_configured_device_not_nvml(app_module, monkeypatch):
    """A box with an NVIDIA card but MODEL_DEVICE=cpu must not claim "gpu — …"
    in the card / copy-report: the device word follows what the server decodes
    on, not NVML merely finding a device."""
    from faster_whisper_backend.admin import routes as admin_routes
    from faster_whisper_backend.runtime import model_registry
    from faster_whisper_backend.runtime import system_stats
    monkeypatch.setattr(system_stats, "gpu_name",
                        lambda: "NVIDIA GeForce RTX 3080")
    monkeypatch.setattr(model_registry, "loaded_models_snapshot", lambda: [])
    monkeypatch.setattr(cfg, "MODEL_DEVICE", "cpu", raising=False)
    ident = admin_routes._server_ident_fields()
    assert "gpu —" not in ident["runs_as"]
    assert "cpu" in ident["runs_as"]
    # the card present-but-unused hint may appear, but never as the device
    assert "gpu —" not in ident["report"]

    # and a loaded cuda model wins over the config fallback
    monkeypatch.setattr(model_registry, "loaded_models_snapshot",
                        lambda: [{"name": "large-v3", "role": "transcribing",
                                  "device": "cuda"}])
    ident = admin_routes._server_ident_fields()
    assert "gpu — NVIDIA GeForce RTX 3080" in ident["runs_as"]


def test_pipeline_view_has_no_identity_card(client):
    r = client.get("/settings/pipeline")
    assert r.status_code == 200
    assert 'id="srv-ident"' not in r.text
    assert "{{SERVER_IDENT}}" not in r.text
