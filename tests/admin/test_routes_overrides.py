"""Route tests for /settings/overrides (state / resolve) + the per-user and
per-key binding endpoints on /settings/api-keys. Driven through the real app
via the conftest TestClient; no faster-whisper needed."""

import json

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
                   "setStatus(jc.detail || 'profile still in use'"):
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
    # bind the profile ONLY via the allowlist — NOT forced in `profiles`
    r = client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=h, json={
        "overrides": {}, "profiles": [], "locks": [],
        "allowed_override_profiles": ["clinic-de"]})
    assert r.status_code == 200, r.text

    usage = client.get(f"{OV}/state", headers=h).json()["usage"]["clinic-de"]
    assert kid in usage["keys"]          # allowlist-only ref now counted
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
    # the profile survived the attempt
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
    # bind alice to the profile + a direct BEST_OF override + lock TEMPERATURE
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
    # alice (user) gets BEAM_SIZE 8; her laptop key forces BEAM_SIZE 4
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


# --- profile rename (key migration + reference cascade) --------------------

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

    # The renamed profile still resolves end-to-end under its new name.
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
    # invalid characters / shape
    assert client.post(f"{OV}/profiles/rename", headers=h,
                       json={"old": "good", "new": "Bad Name!"}).status_code == 400
    # renaming to the current name is a no-op error
    assert client.post(f"{OV}/profiles/rename", headers=h,
                       json={"old": "good", "new": "good"}).status_code == 400


def test_rename_profile_old_name_case_insensitive(client, make_user_key):
    """Profile keys are always stored lowercase, so the endpoint lowercases `old`
    before both the lookup and the binding cascade. A direct API caller passing
    `old` in mixed case must still match (200, not a spurious 404) and the same
    lowercased name must flow to the reference cascade."""
    from faster_whisper_backend.auth import api_keys_store
    _, _, h = _admin(make_user_key)
    _make_profile(client, h, "clinic-de", DEFAULT_LANGUAGE="de")
    uid, _ = make_user_key("alice", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {}, "profiles": ["clinic-de"],
                                "locks": [], "allowed_override_profiles": []}})
    assert r.status_code == 200, r.text
    # `old` sent upper-case; stored key is lowercase "clinic-de".
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
