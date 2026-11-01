"""Route tests for /settings/overrides (state / resolve) + the per-user and
per-key binding endpoints on /settings/api-keys. Driven through the real app
via the conftest TestClient; no faster-whisper needed."""

import importlib
import json
import shutil
import subprocess
import sys

import pytest

from faster_whisper_backend.pipeline import apply as pl_apply
from tests.conftest import bearer

PERMS = "/settings/api-keys/api/users"
OV = "/settings/overrides"


def _admin(make_user_key):
    uid, raw = make_user_key("admin", is_admin=True)
    return uid, raw, bearer(raw)


def _make_profile(client, h, name="clinic-de", **fields):
    body = {"OVERRIDE_PROFILES": {name: fields}}
    r = client.post(f"{OV}/state", headers=h, json=body)
    assert r.status_code == 200, r.text
    return r


def test_overrides_page_renders(client):
    import re
    r = client.get(OV)
    assert r.status_code == 200
    body = r.text
    # render_page substituted every placeholder (none leak through)
    assert not re.findall(r"\{\{[A-Z_]+\}\}", body)
    for marker in ("panel-profiles", "panel-explorer", "_renderWaterfall",
                   "ov-wrap", "tab-explorer", "/settings/overrides",
                   # save() surfaces the server 409 in-use guard's detail
                   # instead of a bare "save failed (409)"
                   "setStatus(jc.detail || 'profile still in use'",
                   # the What-if row names the decode path's own outcomes
                   "ignored_invalid: '⊘ ignored — invalid value'",
                   # a dropped outer names its locked partner, not itself, and
                   # a moved partner the client never sent shows no "sent —"
                   "ignored_partner_locked: '⊘ ignored — partner gate locked'",
                   "cs.sent == null ? 'fitted to the sent gate'"):
        assert marker in body, marker


def test_state_shape_and_field_meta(client, make_user_key):
    _, _, h = _admin(make_user_key)
    j = client.get(f"{OV}/state", headers=h).json()
    assert set(j) >= {"profiles", "field_meta", "defaults", "groups", "rules", "usage"}
    assert j["field_meta"]["BEAM_SIZE"] == {"kind": "int", "min": 1, "max": 20}
    assert j["field_meta"]["STREAMING_VAD_BACKEND"]["kind"] == "enum"
    assert "auto" in j["field_meta"]["STREAMING_VAD_BACKEND"]["opts"]
    # load-time model fields are NOT overridable per-identity → absent
    assert "MODEL_DEVICE" not in j["field_meta"]


def test_state_field_meta_marks_multiline_text_as_textarea(client, make_user_key):
    """The profile editor and the per-identity sub-editor rendered every 'str'
    field as a text <input>, which strips newlines: a multi-line glossary
    saved as one line and the other pairs were gone. The multi-line fields
    get the same 'textarea' kind as on the /settings per-model pane."""
    _, _, h = _admin(make_user_key)
    fm = client.get(f"{OV}/state", headers=h).json()["field_meta"]
    for name in ("TRANSLATION_GLOSSARY", "DEFAULT_PROMPT", "DEFAULT_HOTWORDS"):
        assert fm[name]["kind"] == "textarea", name
        assert fm[name]["maxlen"] > 0, name
    # Only the textarea kind is taken over: other per-model-pane kinds stay off.
    assert fm["NO_SPEECH_THRESHOLD"]["kind"] == "float"
    assert fm["DEFAULT_LANGUAGE"]["kind"] == "str"


@pytest.mark.parametrize("broken", [
    "faster_whisper_backend.admin.routes",
    "faster_whisper_backend.streaming.routes",
])
def test_overrides_and_api_keys_routers_import_without_admin_routes(
        monkeypatch, broken):
    """main.py mounts each admin router fail-soft, so a broken admin.routes
    (settings.html, translation engine, ...) must cost only /settings. The
    textarea kinds came from admin.routes' overlay, so an import failure there
    took the overrides router and (via api_keys_routes) the API-keys page
    down with it. The streaming router, imported for the What-if's stream
    values, did the same."""
    import faster_whisper_backend.admin as admin_pkg
    importlib.import_module("faster_whisper_backend.admin.api_keys_routes")
    for mod in ("overrides_routes", "api_keys_routes"):
        monkeypatch.delattr(admin_pkg, mod)
        monkeypatch.delitem(sys.modules, f"faster_whisper_backend.admin.{mod}")
    parent, _, leaf = broken.rpartition(".")
    monkeypatch.setitem(sys.modules, broken, None)
    monkeypatch.delattr(importlib.import_module(parent), leaf, raising=False)
    importlib.import_module("faster_whisper_backend.admin.api_keys_routes")
    fresh = sys.modules["faster_whisper_backend.admin.overrides_routes"]
    fm = fresh._build_field_meta()
    for name in ("TRANSLATION_GLOSSARY", "DEFAULT_PROMPT", "DEFAULT_HOTWORDS"):
        assert fm[name]["kind"] == "textarea", name


def test_state_includes_inherited_defaults(client, make_user_key, app_module):
    """The /state payload ships the live global value for every overridable field
    so the editor can render `inherits <value>` (and seed `+ override` from it).
    The source must match the /settings per-model page byte-for-byte."""
    _, _, h = _admin(make_user_key)
    j = client.get(f"{OV}/state", headers=h).json()

    defaults = j["defaults"]
    # Built by iterating field_meta → identical key set.
    assert set(defaults) == set(j["field_meta"])
    # Values come from the same serializer the per-model page uses.
    for name in ("BEAM_SIZE", "DEFAULT_LANGUAGE", "VAD_FILTER"):
        assert defaults[name] == pl_apply.resolved_value(name)
    # Every scalar field shown in the editor grid has a default, so no real row
    # can fall back to the ∅ "missing" glyph.
    grouped = {f for g in j["groups"] for sg in g["subgroups"] for f in sg["fields"]}
    assert grouped, "expected at least one editor field group"
    assert grouped <= set(defaults)


def test_state_requires_admin(client, make_user_key):
    # create the admin first (flips lockdown), then a non-admin caller
    _admin(make_user_key)
    _, raw = make_user_key("bob", is_admin=False)
    r = client.get(f"{OV}/state", headers=bearer(raw))
    assert r.status_code == 403


def test_create_profile_roundtrip_and_usage(client, make_user_key):
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de", BEAM_SIZE=8,
                  locks=["DEFAULT_LANGUAGE"])
    j = client.get(f"{OV}/state", headers=h).json()
    assert j["profiles"]["clinic-de"]["BEAM_SIZE"] == 8
    assert j["profiles"]["clinic-de"]["locks"] == ["DEFAULT_LANGUAGE"]
    assert "clinic-de" in j["usage"]


def test_delete_guard_counts_allowlist_only_reference(client, make_user_key):
    # A profile referenced ONLY in a key's requestable allowlist (not forced in
    # `profiles`) must still appear in usage, so the WebUI delete guard refuses
    # it instead of allowing a silent delete that strands a dangling allowlist
    # name — which that binding's next save would then reject.
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de")
    uid, _ = make_user_key("alice", is_admin=False)
    kid = client.get(f"{PERMS}/{uid}/keys", headers=h).json()["keys"][0]["id"]
    r = client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=h, json={
        "overrides": {}, "profiles": [], "locks": [],
        "allowed_override_profiles": ["clinic-de"]})
    assert r.status_code == 200, r.text

    usage = client.get(f"{OV}/state", headers=h).json()["usage"]["clinic-de"]
    assert kid in usage["keys"]
    assert usage["users"] == []          # not forced/allowed on any user


def test_delete_guard_refuses_null_profiles_payload(client, make_user_key):
    # {"OVERRIDE_PROFILES": null} is save_overrides' remove-the-whole-key
    # sentinel: it deletes EVERY profile. The in-use guard must treat it as
    # "all profiles removed", not skip the check because the value isn't a
    # dict — otherwise a curl POST wipes in-use profiles and their locks.
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de")
    uid, _ = make_user_key("alice", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {}, "profiles": ["clinic-de"],
                                "locks": []}})
    assert r.status_code == 200, r.text

    r = client.post(f"{OV}/state", headers=h, json={"OVERRIDE_PROFILES": None})
    assert r.status_code == 409
    assert "clinic-de" in r.json()["detail"]
    j = client.get(f"{OV}/state", headers=h).json()
    assert "clinic-de" in j["profiles"]


def test_settings_state_refuses_override_profiles(client, make_user_key):
    # The in-use guard lives on /settings/overrides/state only, so the
    # generic /settings/state save must not accept OVERRIDE_PROFILES at all:
    # a null or {} there would delete a bound profile and its locks.
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de")
    uid, _ = make_user_key("alice", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {}, "profiles": ["clinic-de"],
                                "locks": []}})
    assert r.status_code == 200, r.text

    for body in ({"OVERRIDE_PROFILES": None}, {"OVERRIDE_PROFILES": {}}):
        r = client.post("/settings/state", headers=h, json=body)
        assert r.status_code == 400, r.text
    j = client.get(f"{OV}/state", headers=h).json()
    assert "clinic-de" in j["profiles"]


def test_bad_profile_value_422(client, make_user_key):
    _, _, h = _admin(make_user_key)
    r = client.post(f"{OV}/state", headers=h,
                    json={"OVERRIDE_PROFILES": {"x": {"BEAM_SIZE": 999}}})
    assert r.status_code == 422


def test_post_rejects_foreign_field(client, make_user_key):
    _, _, h = _admin(make_user_key)
    r = client.post(f"{OV}/state", headers=h, json={"BEAM_SIZE": 5})
    assert r.status_code == 400


def test_bind_user_and_resolve(client, make_user_key):
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de", BEAM_SIZE=8,
                  locks=["DEFAULT_LANGUAGE"], TEMPERATURE="0.0")
    uid, _ = make_user_key("alice", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {"BEST_OF": 7, "TEMPERATURE": "0.0"},
                                "profiles": ["clinic-de"], "locks": ["TEMPERATURE"]}})
    assert r.status_code == 200, r.text

    rj = client.get(f"{OV}/resolve", headers=h, params={
        "user_id": uid, "model": "whisper-1",
        "sim": json.dumps({"beam_size": 12, "temperature": 0.5})}).json()
    f = rj["fields"]
    assert f["DEFAULT_LANGUAGE"]["winner_value"] == "de"
    assert f["DEFAULT_LANGUAGE"]["winner_layer"] == "user.profile:clinic-de"
    assert f["DEFAULT_LANGUAGE"]["locked"] is True
    assert f["BEAM_SIZE"]["winner_value"] == 8                      # from profile
    assert f["BEST_OF"]["winner_value"] == 7                        # user.direct
    # TEMPERATURE locked by user.direct → simulated client temp is ignored
    assert f["TEMPERATURE"]["client_sim"]["outcome"] == "ignored_locked"
    assert "clinic-de" in rj["profiles_applied"]


def test_resolve_decode_master_gate_off_reports_ignored(client, make_user_key):
    # When an identity's decode-override master gate is OFF, the live path drops
    # EVERY client decode override (resolve sets locked_client_keys = all client
    # keys). The /resolve diagnostic must agree and report a sim'd override as
    # ignored — even though the field itself carries no field-level lock.
    _, _, h = _admin(make_user_key)
    bob_uid, _ = make_user_key("bob", is_admin=False)
    kid = client.get(f"{PERMS}/{bob_uid}/keys", headers=h).json()["keys"][0]["id"]
    r = client.patch(f"{PERMS}/{bob_uid}/keys/{kid}/config", headers=h, json={
        "overrides": {}, "profiles": [], "locks": [],
        "allow_request_decode_overrides": False})
    assert r.status_code == 200, r.text

    rj = client.get(f"{OV}/resolve", headers=h, params={
        "user_id": bob_uid, "key_id": kid, "model": "whisper-1",
        "sim": json.dumps({"beam_size": 12})}).json()
    bs = rj["fields"]["BEAM_SIZE"]
    assert bs["locked"] is False                       # no field-level lock
    assert bs["client_sim"]["value"] == 12
    assert bs["client_sim"]["outcome"] == "ignored_locked"  # gate, not field-lock


def test_resolve_sim_reports_the_value_the_decode_path_uses(client, make_user_key):
    # The What-if row reports what the live decode path (_apply_decode_overrides)
    # would use, not the raw value: clamped to the client bounds (beam_size 500
    # runs as 20), unparsable values ignored ("abc").
    _, _, h = _admin(make_user_key)
    uid, _ = make_user_key("dora", is_admin=False)
    sim = {"beam_size": 500, "temperature": "abc", "best_of": "3",
           "vad_threshold": 7, "patience": 1.5, "vad_filter": "false",
           "hotwords": "x" * 3000}
    f = client.get(f"{OV}/resolve", headers=h, params={
        "user_id": uid, "model": "whisper-1", "sim": json.dumps(sim)}).json()["fields"]

    def cs(name):
        return f[name]["client_sim"]

    assert cs("BEAM_SIZE") == {"value": 20, "sent": 500, "outcome": "clamped"}
    assert cs("TEMPERATURE")["outcome"] == "ignored_invalid"
    assert cs("BEST_OF") == {"value": 3, "sent": "3", "outcome": "applied"}
    assert cs("VAD_THRESHOLD") == {"value": 1.0, "sent": 7, "outcome": "clamped"}
    assert cs("PATIENCE") == {"value": 1.5, "sent": 1.5, "outcome": "applied"}
    assert cs("VAD_FILTER") == {"value": False, "sent": "false", "outcome": "applied"}
    assert cs("DEFAULT_HOTWORDS")["outcome"] == "clamped"
    assert cs("DEFAULT_HOTWORDS")["value"] == "x" * 2048


def test_resolve_sim_reports_the_live_dictation_knobs_as_the_handshake_sets_them(
        client, make_user_key):
    # Live-dictation keys report what the streaming handshake
    # (_client_stream_values) would set: clamped, unparsable dropped, the
    # silence-gate pair fitted, and an outer a LOCKED partner cannot fit dropped.
    # Outer 99999 runs as 10000 and moves the hard break 5000 -> 11000.
    _, _, h = _admin(make_user_key)
    uid, _ = make_user_key("erin", is_admin=False)

    def sim(user, **keys):
        return client.get(f"{OV}/resolve", headers=h, params={
            "user_id": user, "model": "whisper-1",
            "sim": json.dumps(keys)}).json()["fields"]

    f = sim(uid, streaming_vad_outer_silence_ms=99999,
            streaming_vad_threshold="abc", streaming_hard_break_separator=" / ")
    assert f["STREAMING_VAD_OUTER_SILENCE_MS"]["client_sim"] == {
        "value": 10000, "sent": 99999, "outcome": "clamped"}
    assert f["STREAMING_HARD_BREAK_SILENCE_MS"]["client_sim"] == {
        "value": 11000, "sent": None, "outcome": "clamped"}  # the moved partner
    assert f["STREAMING_VAD_THRESHOLD"]["client_sim"]["outcome"] == "ignored_invalid"
    assert f["STREAMING_HARD_BREAK_SEPARATOR"]["client_sim"] == {
        "value": " / ", "sent": " / ", "outcome": "applied"}
    assert f["STREAMING_VAD_INNER_SILENCE_MS"]["client_sim"] is None  # inner 700 fits

    fit = sim(uid, streaming_vad_outer_silence_ms=2000)["STREAMING_VAD_OUTER_SILENCE_MS"]
    assert fit["client_sim"] == {"value": 2000, "sent": 2000, "outcome": "applied"}

    # A LOCKED hard break is never moved: a client outer above it is dropped,
    # reported as the partner's lock (the outer field itself is not locked).
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {"STREAMING_HARD_BREAK_SILENCE_MS": 5000},
                                "profiles": [],
                                "locks": ["STREAMING_HARD_BREAK_SILENCE_MS"]}})
    assert r.status_code == 200, r.text
    f = sim(uid, streaming_vad_outer_silence_ms=6000)
    assert f["STREAMING_VAD_OUTER_SILENCE_MS"]["client_sim"] == {
        "value": 6000, "sent": 6000, "outcome": "ignored_partner_locked"}
    assert f["STREAMING_HARD_BREAK_SILENCE_MS"]["client_sim"] is None


def test_resolve_reports_a_binding_store_fault(client, make_user_key, monkeypatch):
    # A binding-fetch fault locks every field fail-closed with no owning
    # layer; the explorer showed "locked" everywhere and gave no reason.
    from faster_whisper_backend.auth import api_keys_store
    _, _, h = _admin(make_user_key)
    uid, _ = make_user_key("carol", is_admin=False)
    params = {"user_id": uid, "model": "whisper-1"}
    rj = client.get(f"{OV}/resolve", headers=h, params=params).json()
    assert rj["binding_fault"] is False

    def _boom(_ident_id):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(api_keys_store, "get_user_config", _boom)
    r = client.get(f"{OV}/resolve", headers=h, params=params)
    assert r.status_code == 200, r.text
    rj = r.json()
    assert rj["binding_fault"] is True
    assert rj["fields"]["DEFAULT_LANGUAGE"]["locked"] is True


def test_page_explorer_names_a_binding_fault(client):
    html = client.get(OV).text
    dr = html[html.index("async function doResolve() {"):]
    dr = dr[:dr.index("\n  }\n")]
    assert "if (j.binding_fault) {" in dr
    assert "Binding store unreadable" in dr


def test_per_key_config_overrides_user(client, make_user_key):
    _, _, h = _admin(make_user_key)
    uid, _ = make_user_key("alice", is_admin=False)
    client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {"BEAM_SIZE": 8}, "profiles": [], "locks": []}})
    keys = client.get(f"{PERMS}/{uid}/keys", headers=h).json()["keys"]
    kid = keys[0]["id"]
    r = client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=h,
                     json={"overrides": {"BEAM_SIZE": 4}, "profiles": [], "locks": []})
    assert r.status_code == 200, r.text
    assert r.json()["config"]["direct"]["BEAM_SIZE"] == 4

    rj = client.get(f"{OV}/resolve", headers=h, params={
        "user_id": uid, "key_id": kid, "model": "whisper-1"}).json()
    bs = rj["fields"]["BEAM_SIZE"]
    assert bs["winner_value"] == 4 and bs["winner_layer"] == "key.direct"


def test_per_key_config_unknown_profile_400(client, make_user_key):
    _, _, h = _admin(make_user_key)
    uid, _ = make_user_key("alice", is_admin=False)
    kid = client.get(f"{PERMS}/{uid}/keys", headers=h).json()["keys"][0]["id"]
    r = client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=h,
                     json={"overrides": {}, "profiles": ["ghost"], "locks": []})
    assert r.status_code == 400


def test_rename_profile_cascades_to_user_and_key(client, make_user_key):
    from faster_whisper_backend.auth import api_keys_store
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de", BEAM_SIZE=8)
    uid, _ = make_user_key("alice", is_admin=False)
    # alice references the profile in BOTH her ordered list and her allowlist
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {}, "profiles": ["clinic-de"],
                                "locks": [],
                                "allowed_override_profiles": ["clinic-de"]}})
    assert r.status_code == 200, r.text
    kid = client.get(f"{PERMS}/{uid}/keys", headers=h).json()["keys"][0]["id"]
    r = client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=h,
                     json={"overrides": {}, "profiles": ["clinic-de"], "locks": []})
    assert r.status_code == 200, r.text

    r = client.post(f"{OV}/profiles/rename", headers=h,
                    json={"old": "clinic-de", "new": "clinic-deutsch"})
    assert r.status_code == 200, r.text
    assert r.json()["bindings_updated"] == 2          # user row + key row

    j = client.get(f"{OV}/state", headers=h).json()
    assert "clinic-de" not in j["profiles"]
    assert j["profiles"]["clinic-deutsch"]["BEAM_SIZE"] == 8   # overrides preserved

    uc = api_keys_store.get_user_config(uid)
    assert uc["profiles"] == ["clinic-deutsch"]
    assert uc["allowed_override_profiles"] == ["clinic-deutsch"]
    assert api_keys_store.get_key_config(kid)["profiles"] == ["clinic-deutsch"]

    rj = client.get(f"{OV}/resolve", headers=h,
                    params={"user_id": uid, "model": "whisper-1"}).json()
    assert rj["fields"]["DEFAULT_LANGUAGE"]["winner_layer"] \
        == "user.profile:clinic-deutsch"


def test_rename_profile_keeps_locks_while_the_cascade_commits(
        client, make_user_key, monkeypatch):
    """The cascade commits bindings to the new name BEFORE the hot-apply
    installs it in the running cfg: in that window a bound identity must
    still resolve its profile (and the profile's locks), not drop the layer."""
    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.settings import effective_config
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de",
                  locks=["DEFAULT_LANGUAGE"])
    uid, _ = make_user_key("alice", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {}, "profiles": ["clinic-de"],
                                "locks": []}})
    assert r.status_code == 200, r.text

    real = api_keys_store.rename_profile_refs
    seen = {}

    def _cascade_then_resolve(old, new):
        n = real(old, new)
        seen["locked"] = effective_config.resolve(None, user_id=uid).locked
        return n

    monkeypatch.setattr(api_keys_store, "rename_profile_refs", _cascade_then_resolve)
    r = client.post(f"{OV}/profiles/rename", headers=h,
                    json={"old": "clinic-de", "new": "clinic-deutsch"})
    assert r.status_code == 200, r.text
    assert "DEFAULT_LANGUAGE" in seen["locked"]
    # The alias is gone once the hot-apply ran.
    j = client.get(f"{OV}/state", headers=h).json()
    assert "clinic-de" not in j["profiles"]
    assert "clinic-deutsch" in j["profiles"]


def test_rename_profile_failed_cascade_keeps_the_old_name(
        client, make_user_key, monkeypatch):
    """A cascade that raises (SQLite locked / disk I/O) leaves every binding on
    `old`; the route must roll the saved rename back instead of hot-applying
    it, so the bound identity keeps its profile layer and locks."""
    import sqlite3

    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.settings import config as cfg
    from faster_whisper_backend.settings import config_store
    from faster_whisper_backend.settings import effective_config
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de",
                  locks=["DEFAULT_LANGUAGE"])
    uid, _ = make_user_key("alice", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {}, "profiles": ["clinic-de"],
                                "locks": []}})
    assert r.status_code == 200, r.text

    def _locked(old, new):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(api_keys_store, "rename_profile_refs", _locked)
    r = client.post(f"{OV}/profiles/rename", headers=h,
                    json={"old": "clinic-de", "new": "clinic-deutsch"})
    assert r.status_code == 500, r.text
    assert set(cfg.OVERRIDE_PROFILES) == {"clinic-de"}
    on_disk = config_store.load_overrides()["OVERRIDE_PROFILES"]
    assert set(on_disk) == {"clinic-de"}
    assert api_keys_store.get_user_config(uid)["profiles"] == ["clinic-de"]
    assert "DEFAULT_LANGUAGE" in effective_config.resolve(None, user_id=uid).locked


def test_profile_save_waits_for_an_in_flight_rename(client, make_user_key, monkeypatch):
    """A /state save from a page still naming `old` landed between a rename's
    save and its hot-apply (which re-reads disk): the cascade moved every
    binding to `new` while cfg reloaded a library without it, so the bound
    identity silently lost the profile layer and its locks. The two routes
    are serialized now: the stale save runs after the rename and meets the
    in-use guard."""
    import asyncio
    import threading

    from fastapi import HTTPException

    from faster_whisper_backend.admin import overrides_routes
    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.settings import config_store
    from faster_whisper_backend.settings import effective_config
    from tests.conftest import fake_request
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de",
                  locks=["DEFAULT_LANGUAGE"])
    uid, _ = make_user_key("alice", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {}, "profiles": ["clinic-de"],
                                "locks": []}})
    assert r.status_code == 200, r.text

    in_cascade, release, stale_saved = (threading.Event() for _ in range(3))
    real_cascade = api_keys_store.rename_profile_refs
    real_save = config_store.save_overrides

    def _slow_cascade(old, new):
        in_cascade.set()
        release.wait(5)
        return real_cascade(old, new)

    def _save(payload):
        out = real_save(payload)
        if "clinic-de" in (payload.get("OVERRIDE_PROFILES") or {}):
            stale_saved.set()
        return out

    monkeypatch.setattr(api_keys_store, "rename_profile_refs", _slow_cascade)
    monkeypatch.setattr(config_store, "save_overrides", _save)
    stale = {"OVERRIDE_PROFILES": {"clinic-de": {
        "DEFAULT_LANGUAGE": "de", "BEAM_SIZE": 8, "locks": ["DEFAULT_LANGUAGE"]}}}

    async def drive():
        rename = asyncio.create_task(overrides_routes.rename_profile(
            overrides_routes._RenameProfileIn(old="clinic-de", new="clinic-deutsch"),
            fake_request()))
        while not in_cascade.is_set():
            await asyncio.sleep(0.01)
        save = asyncio.create_task(overrides_routes.post_state(stale, fake_request()))
        # Unserialized, the stale save completes inside the rename's window.
        await asyncio.to_thread(stale_saved.wait, 1.0)
        release.set()
        return await asyncio.gather(rename, save, return_exceptions=True)

    renamed, saved = asyncio.run(drive())
    assert renamed.status_code == 200
    assert isinstance(saved, HTTPException) and saved.status_code == 409
    assert "DEFAULT_LANGUAGE" in effective_config.resolve(None, user_id=uid).locked


def test_rename_profile_env_pinned_409(client, make_user_key, monkeypatch):
    """An env-pinned OVERRIDE_PROFILES comes back at the next restart with
    only `old`: a cascaded rename would strand every binding on a missing
    profile (its locks gone). Refused before any save, alias or cascade."""
    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.settings import config as cfg
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", BEAM_SIZE=8)
    before = dict(cfg.OVERRIDE_PROFILES)
    monkeypatch.setenv("WHISPER_OVERRIDE_PROFILES",
                       json.dumps({"clinic-de": {"BEAM_SIZE": 8}}))
    calls = []
    monkeypatch.setattr(api_keys_store, "rename_profile_refs",
                        lambda old, new: calls.append((old, new)) or 0)
    r = client.post(f"{OV}/profiles/rename", headers=h,
                    json={"old": "clinic-de", "new": "clinic-deutsch"})
    assert r.status_code == 409, r.text
    assert "WHISPER_OVERRIDE_PROFILES" in r.json()["detail"]
    assert cfg.OVERRIDE_PROFILES == before
    assert calls == []


def test_page_rename_409_shows_the_server_detail(client):
    """The rename route has two 409s (env-pinned library, name collision);
    doRename mapped both to 'already exists', hiding the env pin."""
    html = client.get(OV).text
    body = html[html.index("async function doRename(oldName, newName) {"):]
    body = body[:body.index("\n  }\n")]
    branch = body[body.index("if (r.status === 409) {"):]
    branch = branch[:branch.index("return;")]
    assert "r.json()" in branch
    assert "setStatus(jc.detail || (" in branch


def test_state_save_env_pinned_409(client, make_user_key, monkeypatch):
    """apply_hot_changes skips an env-pinned OVERRIDE_PROFILES, so a save
    reported "saved" while the editor reloaded the env value and
    config.local.json quietly diverged. Refused like rename, nothing written."""
    from faster_whisper_backend.settings import config_store
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", BEAM_SIZE=8)
    before = config_store.load_overrides()
    monkeypatch.setattr(
        config_store, "env_pinned_fields",
        lambda: {"OVERRIDE_PROFILES": "WHISPER_OVERRIDE_PROFILES"})
    r = client.post(f"{OV}/state", headers=h, json={"OVERRIDE_PROFILES": {
        "clinic-de": {"BEAM_SIZE": 8}, "other": {"BEAM_SIZE": 3}}})
    assert r.status_code == 409, r.text
    assert "WHISPER_OVERRIDE_PROFILES" in r.json()["detail"]
    assert config_store.load_overrides() == before


def test_rename_profile_unknown_404(client, make_user_key):
    _, _, h = _admin(make_user_key)
    r = client.post(f"{OV}/profiles/rename", headers=h,
                    json={"old": "ghost", "new": "phantom"})
    assert r.status_code == 404


def test_rename_profile_collision_409(client, make_user_key):
    _, _, h = _admin(make_user_key)
    # Both profiles must coexist — the page always saves the full dict.
    client.post(f"{OV}/state", headers=h, json={"OVERRIDE_PROFILES": {
        "a-prof": {"BEAM_SIZE": 5}, "b-prof": {"BEAM_SIZE": 6}}})
    r = client.post(f"{OV}/profiles/rename", headers=h,
                    json={"old": "a-prof", "new": "b-prof"})
    assert r.status_code == 409


def test_rename_profile_bad_name_and_noop_400(client, make_user_key):
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "good", BEAM_SIZE=5)
    assert client.post(f"{OV}/profiles/rename", headers=h,
                       json={"old": "good", "new": "Bad Name!"}).status_code == 400
    assert client.post(f"{OV}/profiles/rename", headers=h,
                       json={"old": "good", "new": "good"}).status_code == 400


def test_rename_profile_old_name_case_insensitive(client, make_user_key):
    """Profile keys are stored lowercase, so a mixed-case `old` from a direct
    API caller must still match (200, not a spurious 404) and reach the binding
    cascade lowercased."""
    from faster_whisper_backend.auth import api_keys_store
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de")
    uid, _ = make_user_key("alice", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {}, "profiles": ["clinic-de"],
                                "locks": [], "allowed_override_profiles": []}})
    assert r.status_code == 200, r.text
    r = client.post(f"{OV}/profiles/rename", headers=h,
                    json={"old": "CLINIC-DE", "new": "clinic-deutsch"})
    assert r.status_code == 200, r.text
    # The lowercased `old` also reached the cascade (the binding ref matched).
    assert r.json()["bindings_updated"] == 1
    j = client.get(f"{OV}/state", headers=h).json()
    assert "clinic-de" not in j["profiles"]
    assert "clinic-deutsch" in j["profiles"]
    assert api_keys_store.get_user_config(uid)["profiles"] == ["clinic-deutsch"]


def test_page_save_rerenders_after_reloading_state(client):
    """save() replaced `profiles` via loadState(true) without render(): every
    widget still closed over the old profiles[sel], so the next edit landed
    on an orphan (dropped, or never lighting Save)."""
    html = client.get(OV).text
    body = html[html.index("async function save() {"):]
    body = body[:body.index("\n  }\n")]
    ok = body[body.index("if (fresh) adoptServerProfiles(true);"):]
    assert ok.index("adoptServerProfiles(true);") < ok.index("render();") \
        < ok.index("setStatus('saved', 'ok');")


def _page_save_body(client):
    html = client.get(OV).text
    body = html[html.index("async function save() {"):]
    return body[:body.index("\n  }\n")]


def test_page_save_409_keeps_the_working_copy(client):
    """Both 409s (in-use guard, env-pinned OVERRIDE_PROFILES) refuse the WHOLE
    save; loadState(true) then replaced `profiles`/`snapshot` with the server
    copy, silently dropping every other unsaved edit. The branch refreshes
    only the server state S and leaves the working copy dirty."""
    body = _page_save_body(client)
    branch = body[body.index("if (r.status === 409) {"):]
    branch = branch[:branch.index("return;")]
    assert "loadState(" not in branch
    assert "profiles =" not in branch and "snapshot =" not in branch
    assert "await refreshServerState();" in branch
    assert "setStatus(jc.detail || 'profile still in use', 'err');" in branch


def test_page_save_keeps_edits_made_while_in_flight(client):
    """An edit made while the POST is in flight is not in the server copy, so
    the success path must not reload over it; and one POST at a time."""
    body = _page_save_body(client)
    assert "if (saving) return;" in body
    assert "var sent = JSON.stringify(profiles);" in body
    assert "{ OVERRIDE_PROFILES: JSON.parse(sent) }" in body
    late = body[body.index("if (JSON.stringify(profiles) !== sent) {"):]
    late = late[:late.index("return;")]
    assert "loadState(" not in late and "adoptServerProfiles(" not in late
    assert "saving = false; refreshButtons();" in body[body.index("} finally {"):]
    # The edited-while-saving check runs AFTER the awaited state GET: an edit
    # typed during that second round trip is as unsaved as one during the POST.
    ok = body[body.index("if (!r.ok) { setStatus('save failed ("):]
    assert ok.index("snapshot = sent;") \
        < ok.index("var fresh = await refreshServerState();") \
        < ok.index("if (JSON.stringify(profiles) !== sent) {") \
        < ok.index("if (fresh) adoptServerProfiles(true);")
    assert "loadState(" not in ok


def test_page_explorer_drops_stale_answers(client):
    """fillKeys / doResolve had no sequencing: a slower answer for an earlier
    pick mixed keys from two users or overwrote the current waterfall."""
    html = client.get(OV).text
    fk = html[html.index("async function fillKeys() {"):]
    fk = fk[:fk.index("\n  }\n")]
    assert "var seq = ++_keysSeq;" in fk
    assert "if (seq !== _keysSeq || !r.ok) return;" in fk
    assert "if (seq !== _keysSeq) return;" in fk
    dr = html[html.index("async function doResolve() {"):]
    dr = dr[:dr.index("\n  }\n")]
    assert "var seq = ++_resolveSeq;" in dr
    assert dr.count("if (seq !== _resolveSeq) return;") == dr.count("await ")


@pytest.mark.skipif(shutil.which("node") is None, reason="node unavailable")
def test_page_commit_rename_rechecks_dirty(client, tmp_path):
    """startRename refused a dirty working copy, but the main pane stays
    editable while the rename input is open; confirming then ran the server
    rename, whose reload dropped those later edits without a warning."""
    html = client.get(OV).text

    def fn(name):
        start = html.index("  function " + name + "(")
        return html[start:html.index("\n  }\n", start) + 4]

    dirty = html[html.index("  function dirty() {"):]
    js = dirty[:dirty.index("\n") + 1] + fn("commitRename") + r"""
var NAME_RE = /^[a-z0-9][a-z0-9-]{0,31}$/;
var S = { profiles: { clinic: {} } };
var profiles = { clinic: {} };
var snapshot = JSON.stringify(profiles);
var sel = 'clinic', uiRenaming = true, renamed = [], status = null;
function setStatus(m, k) { status = k || null; }
function render() {}
function refreshButtons() {}
function doRename(a, b) { renamed.push([a, b]); }
profiles.clinic.LANGUAGE = 'de';     // edited after the rename input opened
commitRename('clinic', 'clinic-de');
var out = { dirty_renamed: renamed.length, dirty_status: status, still_open: uiRenaming };
snapshot = JSON.stringify(profiles);  // saved
commitRename('clinic', 'clinic-de');
out.clean_renamed = renamed.length;
console.log(JSON.stringify(out));
"""
    f = tmp_path / "probe.js"
    f.write_text(js, encoding="utf-8")
    p = subprocess.run([shutil.which("node"), str(f)], capture_output=True,
                       text=True, encoding="utf-8", timeout=30)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out == {"dirty_renamed": 0, "dirty_status": "err", "still_open": True,
                   "clean_renamed": 1}


@pytest.mark.skipif(shutil.which("node") is None, reason="node unavailable")
def test_page_textarea_field_keeps_its_newlines(client, tmp_path):
    """makeWidget's 'textarea' kind builds a <textarea> (a text <input> strips
    the newlines of a multi-line glossary) and hands its value back as is."""
    html = client.get(OV).text
    start = html.index("  function makeWidget(")
    js = html[start:html.index("\n  }\n", start) + 4] + r"""
var document = { createElement: function (tag) { return { tag: tag }; } };
var got = null;
var el = makeWidget('TRANSLATION_GLOSSARY', { kind: 'textarea', maxlen: 4000 },
                    'a = b\nc = d', function (v) { got = v; });
el.value += '\ne = f';
el.onchange();
console.log(JSON.stringify({ tag: el.tag, maxlen: el.maxLength, got: got }));
"""
    f = tmp_path / "probe.js"
    f.write_text(js, encoding="utf-8")
    p = subprocess.run([shutil.which("node"), str(f)], capture_output=True,
                       text=True, encoding="utf-8", timeout=30)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out == {"tag": "textarea", "maxlen": 4000, "got": "a = b\nc = d\ne = f"}


@pytest.mark.parametrize("writer", ["post_state", "rename_profile"])
def test_cancelled_profiles_save_still_finishes_under_the_lock(monkeypatch, writer):
    """A dropped connection cancels the handler inside the offloaded save,
    but the save keeps writing config.local.json on its thread: the hot-apply
    must still run, with the profiles lock held until it has."""
    import asyncio
    import threading

    from faster_whisper_backend.admin import overrides_routes
    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.settings import config as cfg
    from faster_whisper_backend.settings import config_store
    from tests.conftest import fake_request

    in_save, release = threading.Event(), threading.Event()
    steps = []

    def _save(payload):
        in_save.set()
        release.wait(5)
        steps.append("saved")
        return dict(payload)

    async def _apply(written):
        steps.append("applied")
        return {"hot_applied": [], "cold_pending": [], "env_pinned_ignored": [],
                "evicted": []}

    monkeypatch.setattr(config_store, "save_overrides", _save)
    monkeypatch.setattr(pl_apply, "apply_hot_changes", _apply)
    monkeypatch.setattr(api_keys_store, "rename_profile_refs", lambda old, new: 0)
    monkeypatch.setattr(cfg, "OVERRIDE_PROFILES", {"clinic-de": {}}, raising=False)
    call = {
        "post_state": lambda: overrides_routes.post_state(
            {"OVERRIDE_PROFILES": {"clinic-de": {"BEAM_SIZE": 8}}}, fake_request()),
        "rename_profile": lambda: overrides_routes.rename_profile(
            overrides_routes._RenameProfileIn(old="clinic-de", new="clinic-x"),
            fake_request()),
    }[writer]

    async def scenario():
        lock = overrides_routes._PROFILES_LOCK
        caller = asyncio.create_task(call())
        await asyncio.to_thread(in_save.wait, 5)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        held_after_cancel = lock.locked()
        release.set()
        # Bounded: a regression that leaves the body hanging fails, not hangs.
        await asyncio.wait_for(lock.acquire(), timeout=5)
        lock.release()
        return held_after_cancel

    assert asyncio.run(scenario()) is True
    assert steps == ["saved", "applied"]
