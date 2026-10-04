"""GET /v1/decode-defaults: the decode values a caller's requests get with no
decode_overrides — each client decode key with its value, source and lock, the
prompt, and the two values live dictation pins. Driven through the real app.

Resolver precedence itself is covered in test_effective_config.py; here we
assert the projection, the source categories and the request-profile layering.
"""

import pytest

from faster_whisper_backend import config_store
from tests.conftest import bearer

OV = "/settings/overrides"
PERMS = "/settings/api-keys/api/users"
URL = "/v1/decode-defaults"


@pytest.fixture(autouse=True)
def _a_default_model(app_module, monkeypatch):
    # The test config has no default model; the endpoint resolves one.
    monkeypatch.setattr(app_module.cfg, "DEFAULT_MODEL", "tiny")
    monkeypatch.setattr(app_module.cfg, "ALLOWED_MODELS", set())


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


def test_every_client_key_with_value_source_and_lock(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "BEAM_SIZE", 7)
    j = client.get(URL).json()
    assert set(j["settings"]) == set(config_store.CONFIG_TO_CLIENT_KEY.values())
    assert j["model"] == app_module.cfg.DEFAULT_MODEL
    assert j["profile_applied"] is None
    beam = j["settings"]["beam_size"]
    assert beam == {"value": 7, "source": "server", "label": "global default", "locked": False}
    for entry in j["settings"].values():
        assert set(entry) == {"value", "source", "label", "locked"}


def test_whisper_1_is_the_default_model(client, app_module):
    assert client.get(URL, params={"model": "whisper-1"}).json()["model"] == app_module.cfg.DEFAULT_MODEL


def test_per_model_value_says_model(client, app_module, monkeypatch):
    model = app_module.cfg.DEFAULT_MODEL
    monkeypatch.setattr(app_module.cfg, "MODEL_OVERRIDES",
                        {model: {"BEAM_SIZE": 3}}, raising=False)
    beam = client.get(URL, params={"model": model}).json()["settings"]["beam_size"]
    assert beam["value"] == 3
    assert beam["source"] == "model"


def test_unset_value_is_builtin(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "DEFAULT_HOTWORDS", None)
    hw = client.get(URL).json()["settings"]["hotwords"]
    assert hw["value"] is None
    assert hw["source"] == "builtin"


def test_blank_hotwords_read_as_unset(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "DEFAULT_HOTWORDS", "   ")
    assert client.get(URL).json()["settings"]["hotwords"]["value"] is None


def test_prompt_is_exposed(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "DEFAULT_PROMPT", "Medizin: Anamnese")
    p = client.get(URL).json()["prompt"]
    assert p["value"] == "Medizin: Anamnese"
    assert p["source"] == "server"
    assert p["locked"] is False
    monkeypatch.setattr(app_module.cfg, "DEFAULT_PROMPT", "")
    assert client.get(URL).json()["prompt"]["value"] is None


def test_streaming_pins(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "STREAMING_FINAL_CONDITION_ON_PREVIOUS_TEXT", False)
    monkeypatch.setattr(app_module.cfg, "STREAMING_FINAL_BEST_OF", 1)
    s = client.get(URL).json()["streaming"]
    assert s["condition_on_previous_text"]["final"] is False
    assert s["condition_on_previous_text"]["pinned"] is True
    assert isinstance(s["condition_on_previous_text"]["partial"], bool)
    assert s["best_of"] == {"value": 1}


def test_refused_model_is_400(client, app_module, monkeypatch):
    assert client.get(URL, params={"model": "../etc"}).status_code == 400
    assert client.get(URL, params={"model": "x" * 201}).status_code == 400
    monkeypatch.setattr(app_module.cfg, "ALLOWED_MODELS", {"tiny"})
    assert client.get(URL, params={"model": "small"}).status_code == 400
    assert client.get(URL, params={"model": "tiny"}).status_code == 200


def test_bound_profile_is_account_and_locks(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"studio": {"BEAM_SIZE": 4, "DEFAULT_PROMPT": "Studio",
                                     "locks": ["BEAM_SIZE", "DEFAULT_PROMPT"]}})
    uid, raw_alice = make_user_key("alice")
    kid = _key_id(client, h, uid)
    _set_key_binding(client, h, uid, kid, profiles=["studio"])
    j = client.get(URL, headers=bearer(raw_alice)).json()
    beam = j["settings"]["beam_size"]
    assert beam["value"] == 4
    assert beam["source"] == "account"
    assert "studio" in beam["label"]
    assert beam["locked"] is True
    assert j["prompt"] == {"value": "Studio", "source": "account",
                           "label": beam["label"], "locked": True}
    # "__none__" drops the bound profile: plain defaults, unlocked.
    j2 = client.get(URL, params={"override_profile": "__none__"},
                    headers=bearer(raw_alice)).json()
    assert j2["settings"]["beam_size"]["source"] == "server"
    assert j2["settings"]["beam_size"]["locked"] is False


def test_lock_without_value_keeps_the_inherited_source(client, make_user_key, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "BEAM_SIZE", 9)
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    uid, raw_alice = make_user_key("alice")
    kid = _key_id(client, h, uid)
    _set_key_binding(client, h, uid, kid, locks=["BEAM_SIZE"])
    beam = client.get(URL, headers=bearer(raw_alice)).json()["settings"]["beam_size"]
    assert beam == {"value": 9, "source": "server", "label": "global default", "locked": True}


def test_request_profile_layers_under_identity(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    _profiles(client, h, {"fast": {"BEAM_SIZE": 2, "BEST_OF": 2},
                          "internal": {"BEAM_SIZE": 1, "requestable": False}})
    uid, raw_alice = make_user_key("alice")
    kid = _key_id(client, h, uid)
    _set_key_binding(client, h, uid, kid, overrides={"BEST_OF": 6})
    ah = bearer(raw_alice)
    j = client.get(URL, params={"override_profile": "fast"}, headers=ah).json()
    assert j["profile_applied"] == "fast"
    assert j["settings"]["beam_size"]["value"] == 2
    assert j["settings"]["beam_size"]["source"] == "override_profile"
    # The key's own value beats the requested profile.
    assert j["settings"]["best_of"]["value"] == 6
    assert j["settings"]["best_of"]["source"] == "account"
    # Not requestable: ignored, not applied.
    j2 = client.get(URL, params={"override_profile": "internal"}, headers=ah).json()
    assert j2["profile_applied"] is None
    assert j2["settings"]["beam_size"]["source"] == "server"


def test_decode_gate_off_locks_every_key(client, make_user_key):
    _, raw_admin = make_user_key("admin", is_admin=True)
    h = bearer(raw_admin)
    uid, raw_alice = make_user_key("alice")
    kid = _key_id(client, h, uid)
    _set_key_binding(client, h, uid, kid, allow_request_decode_overrides=False)
    j = client.get(URL, headers=bearer(raw_alice)).json()
    assert all(e["locked"] for e in j["settings"].values())
