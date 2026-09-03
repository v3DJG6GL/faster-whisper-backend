"""P11: per-identity request gates + allowlist, the GET /v1/me capability
contract, the caller-filtered GET /v1/override-profiles, and the per-profile
GET /v1/override-profiles/{name} values endpoint. Driven through the real app.

Resolver-level precedence/gate semantics are covered purely (no DB) in
test_effective_config.py; here we assert the HTTP wiring and the admin binding
round-trip (the new request-gate keys survive validate_binding/_parse_binding).
"""

from tests.conftest import bearer

OV = "/settings/overrides"
PERMS = "/settings/api-keys/api/users"


def _profiles(client, h, profiles):
    r = client.post(f"{OV}/state", headers=h, json={"OVERRIDE_PROFILES": profiles})
    assert r.status_code == 200, r.text


def _key_id(client, h, uid):
    r = client.get(f"{PERMS}/{uid}/keys", headers=h)
    assert r.status_code == 200, r.text
    return r.json()["keys"][0]["id"]


def _set_key_binding(client, h, uid, kid, **binding):
    body = {"overrides": {}, "profiles": [], "locks": [], **binding}
    r = client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=h, json=body)
    assert r.status_code == 200, r.text
    return r.json()["config"]


# --- GET /v1/me -----------------------------------------------------------

def test_me_open_mode_all_allowed(client):
    j = client.get("/v1/me").json()
    assert j["can_request_override_profile"] is True
    assert j["can_request_decode_overrides"] is True
    assert j["allowed_override_profiles"] == ["*"]


def test_me_reports_vad_filter_default(client, app_module):
    # Additive convenience for the client's Skip-silence "Default" label.
    assert client.get("/v1/me").json()["vad_filter_default"] is True
    app_module.cfg.VAD_FILTER = False
    assert client.get("/v1/me").json()["vad_filter_default"] is False


def test_me_reports_stage_availability(client, app_module):
    # Additive pre-flight flags for the client's Separate-music / diarization
    # toggles — a disabled feature otherwise only soft-fails into a warning
    # after the run.
    j = client.get("/v1/me").json()
    assert j["bgm_separation_enabled"] is False
    assert j["diarization_enabled"] is False
    app_module.cfg.BGM_SEPARATION_ENABLED = True
    app_module.cfg.DIARIZATION_ENABLED = True
    j = client.get("/v1/me").json()
    assert j["bgm_separation_enabled"] is True
    assert j["diarization_enabled"] is True


def test_me_translation_flag_present_details_gated(client, app_module):
    # translation_enabled is ALWAYS present; the detail keys ride only when
    # the stage is on (yt_dlp_version shape discipline).
    j = client.get("/v1/me").json()
    assert j["translation_enabled"] is False
    for k in ("translation_models", "translation_languages",
              "translate_to_default", "llama_cpp_version"):
        assert k not in j
    app_module.cfg.TRANSLATION_ENABLED = True
    j = client.get("/v1/me").json()
    assert j["translation_enabled"] is True
    for k in ("translation_models", "translation_languages",
              "translate_to_default", "llama_cpp_version"):
        assert k in j


def test_me_translation_models_default_first_with_loaded_flags(
        client, app_module):
    from faster_whisper_backend.audio import translation
    app_module.cfg.TRANSLATION_ENABLED = True
    app_module.cfg.TRANSLATION_DEFAULT_MODEL = "org/default-GGUF:Q4"
    app_module.cfg.TRANSLATION_ALLOWED_MODELS = {
        "org/zeta-GGUF:Q4", "org/alpha-GGUF:Q4", "org/default-GGUF:Q4"}
    # A loaded model OUTSIDE the allowlist (loaded before the admin tightened
    # it) is NOT offered — every request naming it would be refused; the list
    # is the stage's own admission rule (allowlist ∪ configured default),
    # residency only feeds the loaded flags (from the module LRU).
    translation._models["org/default-GGUF:Q4"] = object()
    translation._models["org/extra-GGUF:Q4"] = object()
    j = client.get("/v1/me").json()
    assert j["translation_models"] == [
        {"id": "org/default-GGUF:Q4", "loaded": True},
        {"id": "org/alpha-GGUF:Q4", "loaded": False},
        {"id": "org/zeta-GGUF:Q4", "loaded": False},
    ]
    # Language menu: "en" first, then the sorted rest — non-empty either way.
    langs = j["translation_languages"]
    assert langs[0] == "en" and "de" in langs
    # llama_cpp_version is best-effort: a string when installed, else null —
    # never absent while the stage is enabled.
    assert "llama_cpp_version" in j


def test_me_translation_models_empty_default_still_answers(
        client, app_module):
    # No default configured: no crash, no phantom "" entry, and the language
    # list still answers via the chatml fallback family.
    app_module.cfg.TRANSLATION_ENABLED = True
    app_module.cfg.TRANSLATION_DEFAULT_MODEL = ""
    app_module.cfg.TRANSLATION_ALLOWED_MODELS = {"org/only-GGUF:Q4"}
    j = client.get("/v1/me").json()
    assert j["translation_models"] == [
        {"id": "org/only-GGUF:Q4", "loaded": False}]
    assert j["translation_languages"][0] == "en"


def test_me_translate_to_default_respects_identity_override(
        client, app_module, make_user_key):
    app_module.cfg.TRANSLATION_ENABLED = True
    app_module.cfg.TRANSLATE_TO = "en"
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"de-fr": {"TRANSLATE_TO": "de,fr-CA"}})
    uid, raw_alice = make_user_key("alice")
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {},
        "config": {"overrides": {}, "profiles": ["de-fr"], "locks": []}})
    assert r.status_code == 200, r.text
    # Alice sees HER effective default (profile layer), parsed csv → list.
    j = client.get("/v1/me", headers=bearer(raw_alice)).json()
    assert j["translate_to_default"] == ["de", "fr-CA"]
    # The admin (no binding) sees the global default.
    j = client.get("/v1/me", headers=h).json()
    assert j["translate_to_default"] == ["en"]


def test_me_stage_model_lists_with_loaded_flags(client, app_module):
    from faster_whisper_backend.audio import bgm_separation
    from faster_whisper_backend.audio import diarization
    # Always present (independent of the enabled switches) — the client's
    # model pickers pre-flight on the allowlists.
    j = client.get("/v1/me").json()
    assert {m["id"] for m in j["diarization_models"]} == \
        set(app_module.cfg.DIARIZATION_ALLOWED_MODELS)
    assert all(m["loaded"] is False for m in j["diarization_models"])
    assert {m["id"] for m in j["separation_models"]} == \
        set(app_module.cfg.BGM_SEPARATION_ALLOWED_MODELS)
    assert all(m["loaded"] is False for m in j["separation_models"])
    # Loaded = the module's cached key matches (separator caches by FILENAME).
    diar_id = app_module.cfg.DIARIZATION_ALLOWED_MODELS[0]
    sep_id = app_module.cfg.BGM_SEPARATION_ALLOWED_MODELS[0]
    diarization._pipeline_key = (diar_id, "cpu", 4)
    bgm_separation._separator_key = (
        sep_id if "." in sep_id else f"{sep_id}.onnx", "cpu")
    try:
        j = client.get("/v1/me").json()
        assert {m["id"]: m["loaded"] for m in j["diarization_models"]}[
            diar_id] is True
        assert {m["id"]: m["loaded"] for m in j["separation_models"]}[
            sep_id] is True
    finally:
        diarization._pipeline_key = None
        bgm_separation._separator_key = None


def test_me_reflects_per_key_gate(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 3}})
    uid, raw_alice = make_user_key("alice")
    kid = _key_id(client, h, uid)
    _set_key_binding(client, h, uid, kid,
                     allow_request_override_profile=False,
                     allow_request_decode_overrides=False,
                     allowed_override_profiles=["fast"])
    j = client.get("/v1/me", headers=bearer(raw_alice)).json()
    assert j["can_request_override_profile"] is False
    assert j["can_request_decode_overrides"] is False
    # gate off ⇒ no names, even though the allowlist named one
    assert j["allowed_override_profiles"] == []


def test_me_explicit_allowlist(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 3}, "slow": {"BEAM_SIZE": 12}})
    uid, raw_alice = make_user_key("alice")
    kid = _key_id(client, h, uid)
    _set_key_binding(client, h, uid, kid, allowed_override_profiles=["fast"])
    j = client.get("/v1/me", headers=bearer(raw_alice)).json()
    assert j["can_request_override_profile"] is True
    assert j["allowed_override_profiles"] == ["fast"]


def test_per_key_gate_follows_session_login(client, make_user_key):
    """A per-key restriction must still apply after the key holder logs into the
    WebUI (cookie auth), not only when the key is sent as a bearer token.
    Regression guard: the session now stamps the login key_id so per-key
    overrides/locks bind on cookie-authed requests (previously the '(session)'
    sentinel shed them)."""
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 3}})
    uid, raw_alice = make_user_key("alice")
    kid = _key_id(client, h, uid)
    _set_key_binding(client, h, uid, kid,
                     allow_request_decode_overrides=False,
                     allow_request_override_profile=False,
                     allowed_override_profiles=["fast"])
    # Baseline: as a bearer token, the restriction applies.
    jb = client.get("/v1/me", headers=bearer(raw_alice)).json()
    assert jb["can_request_decode_overrides"] is False
    assert jb["can_request_override_profile"] is False

    # Log in with the SAME key → HttpOnly session cookie (kept by the client).
    r = client.post("/auth/login", json={"key": raw_alice})
    assert r.status_code == 200, r.text
    assert r.json().get("open_mode") is False
    # Cookie-authed /v1/me (no bearer header): the per-key gate must still apply.
    jc = client.get("/v1/me").json()
    assert jc["can_request_decode_overrides"] is False
    assert jc["can_request_override_profile"] is False
    assert jc["allowed_override_profiles"] == []  # gate off ⇒ no names
    # Clear the session cookie so it can't leak into later tests on this client.
    client.cookies.clear()


# --- GET /v1/override-profiles (caller-filtered) --------------------------

def test_override_profiles_filtered_by_allowlist(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 3}, "slow": {"BEAM_SIZE": 12}})
    uid, raw_alice = make_user_key("alice")
    kid = _key_id(client, h, uid)
    _set_key_binding(client, h, uid, kid, allowed_override_profiles=["fast"])
    r = client.get("/v1/override-profiles", headers=bearer(raw_alice))
    assert r.json() == {"profiles": ["fast"]}


def test_override_profiles_excludes_non_requestable(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 3},
                          "internal": {"BEAM_SIZE": 1, "requestable": False}})
    _, raw_alice = make_user_key("alice")
    r = client.get("/v1/override-profiles", headers=bearer(raw_alice))
    assert r.json() == {"profiles": ["fast"]}     # internal hidden from clients


# --- GET /v1/override-profiles/{name} (values) ----------------------------

def test_override_profile_detail_values_and_locks(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 3, "VAD_FILTER": True,
                                   "locks": ["BEAM_SIZE"]}})
    _, raw_alice = make_user_key("alice")
    r = client.get("/v1/override-profiles/fast", headers=bearer(raw_alice))
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["name"] == "fast"
    assert j["values"] == {"beam_size": 3, "vad_filter": True}   # projected to client keys
    assert j["locked"] == ["beam_size"]


def test_override_profile_detail_exposes_prompt(client, make_user_key):
    """B5: the detail endpoint exposes the profile's DEFAULT_PROMPT SEPARATELY (not
    in `values`, which is the 19 client decode keys) so the editor can ghost it as
    the inherited 'Vocabulary / prompt'. Its lock state rides along; a prompt-less
    profile reports null/false."""
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {
        "withp": {"BEAM_SIZE": 3, "DEFAULT_PROMPT": "Medizin: Anamnese",
                  "locks": ["DEFAULT_PROMPT"]},
        "nop": {"BEAM_SIZE": 5},
    })
    _, raw_alice = make_user_key("alice")
    ah = bearer(raw_alice)
    j = client.get("/v1/override-profiles/withp", headers=ah).json()
    assert j["prompt"] == "Medizin: Anamnese"
    assert j["prompt_locked"] is True
    assert "default_prompt" not in j["values"]      # prompt is NOT a client decode key
    j2 = client.get("/v1/override-profiles/nop", headers=ah).json()
    assert j2["prompt"] is None
    assert j2["prompt_locked"] is False


def test_override_profile_detail_404_when_not_allowed(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 3},
                          "internal": {"BEAM_SIZE": 1, "requestable": False}})
    uid, raw_alice = make_user_key("alice")
    kid = _key_id(client, h, uid)
    _set_key_binding(client, h, uid, kid, allowed_override_profiles=["fast"])
    ah = bearer(raw_alice)
    assert client.get("/v1/override-profiles/internal", headers=ah).status_code == 404
    assert client.get("/v1/override-profiles/slow", headers=ah).status_code == 404  # unknown
    assert client.get("/v1/override-profiles/fast", headers=ah).status_code == 200


# --- admin binding round-trip (new keys survive validate/parse) -----------

def test_binding_roundtrip_preserves_request_gates(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 3}})
    uid, _ = make_user_key("alice")
    kid = _key_id(client, h, uid)
    stored = _set_key_binding(client, h, uid, kid,
                              allow_request_override_profile=False,
                              allow_request_decode_overrides=True,
                              allowed_override_profiles=["fast"])
    assert stored["allow_request_override_profile"] is False
    assert stored["allow_request_decode_overrides"] is True
    assert stored["allowed_override_profiles"] == ["fast"]
    # re-read via the keys listing → the stored config carries the gates
    r = client.get(f"{PERMS}/{uid}/keys", headers=h)
    cfg = r.json()["keys"][0]["config"]
    assert cfg["allow_request_override_profile"] is False
    assert cfg["allowed_override_profiles"] == ["fast"]


# --- admin per-key "apply no profiles" force ------------------------------

def test_apply_no_profiles_roundtrip_and_suppresses_user_profile(client, make_user_key):
    # End-to-end: a profile bound at the USER scope applies, until the per-KEY
    # apply_no_profiles force flips the key to plain defaults — through the real
    # PATCH → store → resolve path.
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"clinic": {"BEAM_SIZE": 7}})
    uid, _ = make_user_key("alice")
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=h, json={
        "pages": {}, "config": {"overrides": {}, "profiles": ["clinic"], "locks": []}})
    assert r.status_code == 200, r.text
    kid = _key_id(client, h, uid)

    # baseline: the user-bound profile applies for this key
    rj = client.get(f"{OV}/resolve", headers=h, params={
        "user_id": uid, "key_id": kid, "model": "whisper-1"}).json()
    assert rj["fields"]["BEAM_SIZE"]["winner_value"] == 7
    assert "clinic" in rj["profiles_applied"]

    # set the per-key force → stored + re-read carry it
    stored = _set_key_binding(client, h, uid, kid, apply_no_profiles=True)
    assert stored["apply_no_profiles"] is True
    cfg = client.get(f"{PERMS}/{uid}/keys", headers=h).json()["keys"][0]["config"]
    assert cfg["apply_no_profiles"] is True

    # resolve now ignores the user-bound profile → plain defaults
    rj = client.get(f"{OV}/resolve", headers=h, params={
        "user_id": uid, "key_id": kid, "model": "whisper-1"}).json()
    assert rj["profiles_applied"] == []
    assert rj["fields"]["BEAM_SIZE"]["winner_layer"] != "user.profile:clinic"


def test_me_stage_model_lists_seed_the_configured_model(client, app_module,
                                                        monkeypatch):
    """An empty allowlist means "the configured model only" for both stages,
    so /v1/me must publish that model — an empty list told the client's
    picker nothing was available. With an allowlist, the configured model
    leads (de-duplicated) so the picker defaults correctly."""
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_MODEL", "pyannote/x")
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ALLOWED_MODELS", [])
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_UVR_MODEL", "UVR-A")
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_ALLOWED_MODELS", [])
    j = client.get("/v1/me").json()
    assert [m["id"] for m in j["diarization_models"]] == ["pyannote/x"]
    assert [m["id"] for m in j["separation_models"]] == ["UVR-A"]

    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ALLOWED_MODELS",
                        ["pyannote/y", "pyannote/x"])
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_ALLOWED_MODELS",
                        ["UVR-B", "UVR-A"])
    j = client.get("/v1/me").json()
    assert [m["id"] for m in j["diarization_models"]] == ["pyannote/x", "pyannote/y"]
    assert [m["id"] for m in j["separation_models"]] == ["UVR-A", "UVR-B"]
