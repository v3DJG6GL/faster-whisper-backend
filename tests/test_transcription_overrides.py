"""Phase 3: the batch /v1/audio/transcriptions path honours per-identity
config. Asserts the kwargs the FakeModel receives + the overrides_ignored
feedback. Driven through the real app; no faster-whisper needed."""

import json

from tests.conftest import bearer

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}
OV = "/settings/overrides"
PERMS = "/settings/api-keys/api/users"


def _setup_profile(client, admin_h, name, **fields):
    r = client.post(f"{OV}/state", headers=admin_h,
                    json={"OVERRIDE_PROFILES": {name: fields}})
    assert r.status_code == 200, r.text


def test_identity_profile_applies_to_decode_kwargs(client, make_user_key, fake_model):
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    # profile: beam=8 (LOCKED), best_of=5 (unlocked)
    _setup_profile(client, admin_h, "p", BEAM_SIZE=8, BEST_OF=5, locks=["BEAM_SIZE"])
    uid, raw_alice = make_user_key("alice", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=admin_h,
                     json={"pages": {}, "config": {"overrides": {}, "profiles": ["p"], "locks": []}})
    assert r.status_code == 200, r.text

    # alice transcribes, trying to override BOTH beam_size (locked) and best_of.
    r = client.post(
        "/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
        data={"model": "whisper-1", "response_format": "verbose_json",
              "decode_overrides": json.dumps({"beam_size": 20, "best_of": 3})},
    )
    assert r.status_code == 200, r.text
    kw = fake_model.last_kwargs
    assert kw["beam_size"] == 8        # profile value; client 20 dropped (locked)
    assert kw["best_of"] == 3          # unlocked → client override wins
    assert r.json()["overrides_ignored"] == ["beam_size"]


def test_locked_language_ignores_client_param(client, make_user_key, fake_model):
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    _setup_profile(client, admin_h, "de", DEFAULT_LANGUAGE="de", locks=["DEFAULT_LANGUAGE"])
    uid, raw_alice = make_user_key("alice", is_admin=False)
    client.patch(f"{PERMS}/{uid}/permissions", headers=admin_h,
                 json={"pages": {}, "config": {"overrides": {}, "profiles": ["de"], "locks": []}})

    r = client.post(
        "/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
        data={"model": "whisper-1", "response_format": "verbose_json", "language": "fr"},
    )
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["language"] == "de"      # locked → client 'fr' ignored
    assert "language" in r.json()["overrides_ignored"]


def test_no_identity_config_is_unchanged(client, make_user_key, fake_model):
    # A user with no binding decodes with the global defaults + the client
    # override applied (no lock) and no overrides_ignored field.
    _, raw_admin = make_user_key("admin", is_admin=True)
    uid, raw_alice = make_user_key("alice", is_admin=False)
    r = client.post(
        "/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
        data={"model": "whisper-1", "response_format": "verbose_json",
              "decode_overrides": json.dumps({"beam_size": 7})},
    )
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["beam_size"] == 7        # client override applies
    assert "overrides_ignored" not in r.json()


def test_per_key_override_beats_user(client, make_user_key, fake_model):
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    uid, raw_alice = make_user_key("alice", is_admin=False)
    # user-level beam 8; alice's key forces beam 4
    client.patch(f"{PERMS}/{uid}/permissions", headers=admin_h,
                 json={"pages": {}, "config": {"overrides": {"BEAM_SIZE": 8}, "profiles": [], "locks": []}})
    kid = client.get(f"{PERMS}/{uid}/keys", headers=admin_h).json()["keys"][0]["id"]
    client.patch(f"{PERMS}/{uid}/keys/{kid}/config", headers=admin_h,
                 json={"overrides": {"BEAM_SIZE": 4}, "profiles": [], "locks": []})

    r = client.post(
        "/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
        data={"model": "whisper-1"},
    )
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["beam_size"] == 4       # key.direct wins over user.direct
