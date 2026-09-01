"""Integration tests for the /settings/api-keys admin endpoints that manage
per-account synced client settings (metadata map, export, import, delete)."""

import json

from conftest import bearer

_API = "/settings/api-keys/api"
_V1 = "/v1/client-settings"


def _seed(client, blob, base=0, device=None, headers=None):
    """Push a blob the way a desktop device would (user-tier /v1 endpoint)."""
    body = {"blob": blob, "base_version": base}
    if device is not None:
        body["device"] = device
    r = client.put(_V1, json=body, headers=headers or {})
    assert r.status_code == 200
    return r.json()


def test_page_ships_sync_ui(client):
    """The served page must carry the sync surface: the import-preview
    modal, the drawer builder, and the header-chip class."""
    html = client.get("/settings/api-keys").text
    assert "cs-import-modal" in html
    assert "buildCsDrawer" in html
    assert "pill sync" in html or ".pill.sync" in html


def test_meta_map_empty(client):
    r = client.get(f"{_API}/client-settings")
    assert r.status_code == 200
    # `unavailable` False keeps "healthy but empty" distinguishable from a
    # store outage (which also serves an empty map).
    assert r.json() == {"by_user": {}, "unavailable": False}


def test_meta_map_after_open_mode_push(client):
    _seed(client, {"general": {"theme": "dark"}}, device="mars-tower")
    j = client.get(f"{_API}/client-settings").json()
    m = j["by_user"]["(open-mode)"]
    assert m["version"] == 1
    assert m["device"] == "mars-tower"
    assert m["bytes"] > 0
    assert m["updated_at"] is not None
    assert "blob" not in m  # metadata only — blob contents never listed


def test_export_download(client, make_user_key):
    uid, key = make_user_key("Dr. Mueller", is_admin=True)
    _seed(client, {"general": {"theme": "dark"}}, headers=bearer(key))
    r = client.get(
        f"{_API}/users/{uid}/client-settings/export", headers=bearer(key)
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    # The blob may carry the account's saved API keys — no browser or
    # intermediary may cache it.
    assert r.headers["cache-control"] == "no-store"
    # Filename carries a sanitized username + the stored version.
    assert (
        'filename="client-settings_Dr.-Mueller_v1.json"'
        in r.headers["content-disposition"]
    )
    # Pretty-printed (a human inspecting/restoring it beats byte-fidelity)
    # but parses back to the identical document.
    assert b'\n  "general"' in r.content
    assert json.loads(r.content) == {"general": {"theme": "dark"}}


def test_export_import_roundtrip_unknown_schema(client, make_user_key):
    """The contract the JS import honours (via the export's filename): a
    server-exported document round-trips through Import even when its
    top-level keys match none of the known desktop categories."""
    uid, key = make_user_key("future", is_admin=True)
    blob = {"futureCategory": {"x": 1}}
    _seed(client, blob, headers=bearer(key))
    r = client.get(
        f"{_API}/users/{uid}/client-settings/export", headers=bearer(key)
    )
    assert r.status_code == 200
    exported = json.loads(r.content)
    assert exported == blob
    r = client.post(
        f"{_API}/users/{uid}/client-settings/import",
        json={"blob": exported},
        headers=bearer(key),
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_export_nothing_stored_404(client, make_user_key):
    uid, key = make_user_key("empty", is_admin=True)
    r = client.get(
        f"{_API}/users/{uid}/client-settings/export", headers=bearer(key)
    )
    assert r.status_code == 404


def test_export_unknown_user_404(client):
    r = client.get(f"{_API}/users/nope/client-settings/export")
    assert r.status_code == 404


def test_import_bumps_and_devices_converge(client, make_user_key):
    """The end-to-end property the feature exists for: an admin import
    always lands as a NEWER version, so a device's stale CAS push 409s
    with the imported document — it converges via its normal merge path."""
    uid, key = make_user_key("carol", is_admin=True)
    _seed(client, {"n": 1}, headers=bearer(key), device="laptop")  # device → v1

    r = client.post(
        f"{_API}/users/{uid}/client-settings/import",
        json={"blob": {"n": 99}},
        headers=bearer(key),
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["version"] == 2

    # Metadata reflects the import (device tag included).
    m = client.get(f"{_API}/client-settings", headers=bearer(key)).json()
    assert m["by_user"][uid]["version"] == 2
    assert m["by_user"][uid]["device"] == "WebUI import"

    # The device still holds v1 — its push conflicts and carries the import.
    r = client.put(
        _V1,
        json={"blob": {"n": 2}, "base_version": 1},
        headers=bearer(key),
    )
    assert r.status_code == 409
    assert r.json()["version"] == 2
    assert r.json()["blob"] == {"n": 99}


def test_import_creates_when_nothing_stored(client, make_user_key):
    uid, key = make_user_key("fresh", is_admin=True)
    r = client.post(
        f"{_API}/users/{uid}/client-settings/import",
        json={"blob": {"seeded": True}},
        headers=bearer(key),
    )
    assert r.status_code == 200
    assert r.json()["version"] == 1
    got = client.get(_V1, headers=bearer(key)).json()
    assert got["blob"] == {"seeded": True}


def test_import_oversize_413(client, make_user_key):
    import client_settings_store as store
    uid, key = make_user_key("big", is_admin=True)
    big = {"x": "a" * (store._CAP_BLOB + 100)}
    r = client.post(
        f"{_API}/users/{uid}/client-settings/import",
        json={"blob": big},
        headers=bearer(key),
    )
    assert r.status_code == 413


def test_import_non_finite_float_422(client, make_user_key):
    """Bare NaN in the import body must be a 422 with nothing force-written
    — not a landed write followed by a 500 (see client_settings_store.InvalidBlob)."""
    uid, key = make_user_key("nan", is_admin=True)
    _seed(client, {"n": 1}, headers=bearer(key))
    r = client.post(
        f"{_API}/users/{uid}/client-settings/import",
        content=b'{"blob": {"x": NaN}}',
        headers={**bearer(key), "content-type": "application/json"},
    )
    assert r.status_code == 422
    assert "NaN" in r.json()["detail"]
    got = client.get(_V1, headers=bearer(key)).json()
    assert got["version"] == 1 and got["blob"] == {"n": 1}


def test_import_malformed_422(client, make_user_key):
    uid, key = make_user_key("mal", is_admin=True)
    base = f"{_API}/users/{uid}/client-settings/import"
    # Non-object blob.
    r = client.post(base, json={"blob": "a string"}, headers=bearer(key))
    assert r.status_code == 422
    # Unknown extra field (extra="forbid").
    r = client.post(
        base, json={"blob": {}, "base_version": 3}, headers=bearer(key)
    )
    assert r.status_code == 422


def test_import_unknown_user_404(client):
    r = client.post(
        f"{_API}/users/nope/client-settings/import", json={"blob": {}}
    )
    assert r.status_code == 404


def test_delete_then_meta_empty(client, make_user_key):
    uid, key = make_user_key("gone", is_admin=True)
    _seed(client, {"n": 1}, headers=bearer(key))
    r = client.delete(
        f"{_API}/users/{uid}/client-settings", headers=bearer(key)
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True, "deleted": True}
    # Idempotent second delete reports deleted=False.
    r = client.delete(
        f"{_API}/users/{uid}/client-settings", headers=bearer(key)
    )
    assert r.json() == {"ok": True, "deleted": False}
    m = client.get(f"{_API}/client-settings", headers=bearer(key)).json()
    assert uid not in m["by_user"]


def test_open_mode_sentinel_manageable(client):
    """The "(open-mode)" row belongs to no listed user but must stay
    exportable/deletable — otherwise a blob stored before lockdown would
    be invisible and immortal."""
    _seed(client, {"from": "open-mode-device"})
    r = client.get(f"{_API}/users/(open-mode)/client-settings/export")
    assert r.status_code == 200
    assert "client-settings_open-mode_v1.json" in r.headers["content-disposition"]
    assert json.loads(r.content) == {"from": "open-mode-device"}
    r = client.delete(f"{_API}/users/(open-mode)/client-settings")
    assert r.json() == {"ok": True, "deleted": True}


def test_admin_gate(client, make_user_key):
    """Locked down: no bearer → 401, non-admin bearer → 403 on every
    endpoint (same tier as the rest of the keys page's API)."""
    uid_admin, _admin_key = make_user_key("admin", is_admin=True)
    _uid, user_key = make_user_key("plain")

    paths = [
        ("GET", f"{_API}/client-settings", None),
        ("GET", f"{_API}/users/{uid_admin}/client-settings/export", None),
        ("POST", f"{_API}/users/{uid_admin}/client-settings/import", {"blob": {}}),
        ("DELETE", f"{_API}/users/{uid_admin}/client-settings", None),
    ]
    for method, path, body in paths:
        r = client.request(method, path, json=body)
        assert r.status_code == 401, (method, path)
        r = client.request(method, path, json=body, headers=bearer(user_key))
        assert r.status_code == 403, (method, path)


def test_admin_endpoints_when_store_unavailable(client, make_user_key, monkeypatch):
    """Store never initialized: the meta map degrades to empty (the page must
    still render its users) but flags `unavailable: true` so the drawers say
    "store down" rather than the falsehood "nothing stored"; export/import/
    delete surface a 503 instead of a bare 500."""
    import client_settings_store

    uid, key = make_user_key("sadmin", is_admin=True)
    monkeypatch.setattr(client_settings_store, "_conn", None)
    h = bearer(key)

    r = client.get(f"{_API}/client-settings", headers=h)
    assert r.status_code == 200
    assert r.json() == {"by_user": {}, "unavailable": True}

    assert (
        client.get(f"{_API}/users/{uid}/client-settings/export", headers=h).status_code
        == 503
    )
    assert (
        client.post(
            f"{_API}/users/{uid}/client-settings/import",
            json={"blob": {"a": 1}},
            headers=h,
        ).status_code
        == 503
    )
    assert (
        client.delete(f"{_API}/users/{uid}/client-settings", headers=h).status_code
        == 503
    )
