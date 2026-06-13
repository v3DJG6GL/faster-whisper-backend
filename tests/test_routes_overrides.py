"""Route tests for /settings/overrides (state / resolve) + the per-user and
per-key binding endpoints on /settings/api-keys. Driven through the real app
via the conftest TestClient; no faster-whisper needed."""

import json

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


def test_state_shape_and_field_meta(client, make_user_key):
    _, _, h = _admin(make_user_key)
    j = client.get(f"{OV}/state", headers=h).json()
    assert set(j) >= {"profiles", "field_meta", "groups", "rules", "usage"}
    assert j["field_meta"]["BEAM_SIZE"] == {"kind": "int", "min": 1, "max": 20}
    assert j["field_meta"]["STREAMING_VAD_BACKEND"]["kind"] == "enum"
    assert "auto" in j["field_meta"]["STREAMING_VAD_BACKEND"]["opts"]
    # load-time model fields are NOT overridable per-identity → absent
    assert "MODEL_DEVICE" not in j["field_meta"]


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
