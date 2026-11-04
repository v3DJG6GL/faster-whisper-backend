"""Per-request `override_profile` selection on the batch endpoint + the
`/v1/override-profiles` names endpoint. Driven through the real app; no
faster-whisper needed (FakeModel records the decode kwargs).

The resolver-level precedence/gate semantics are covered purely (no DB) in
test_effective_config.py; here we assert the HTTP wiring: the Form field reaches
the resolver, `profile_applied` is echoed, an admin binding/lock still wins over
the request profile, the gate disables it, and the names endpoint behaves.
"""


from faster_whisper_backend.settings import config as cfg
from tests._transcription_helpers import bind_profile, post_verbose_ok
from tests.conftest import bearer

OV = "/settings/overrides"
PERMS = "/settings/api-keys/api/users"


def _transcribe(client, profile, headers=None):
    return post_verbose_ok(client, headers, override_profile=profile)


def _gate_profiles(monkeypatch, allow, profiles=None):
    monkeypatch.setattr(cfg, "OVERRIDE_PROFILES", profiles or {"fast": {"BEAM_SIZE": 3}},
                        raising=False)
    monkeypatch.setattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", allow, raising=False)


def test_request_profile_applies_to_decode_kwargs(client, fake_model, monkeypatch):
    # Open mode (no keys) → synthetic admin; the request profile is the only layer.
    _gate_profiles(monkeypatch, True)
    body = _transcribe(client, "fast")
    assert fake_model.last_kwargs["beam_size"] == 3
    assert body["profile_applied"] == "fast"


def test_request_profile_gated_off_is_ignored(client, fake_model, monkeypatch):
    _gate_profiles(monkeypatch, False)
    body = _transcribe(client, "fast")
    assert fake_model.last_kwargs["beam_size"] != 3
    assert body["profile_applied"] is None


def test_unknown_request_profile_not_applied(client, fake_model, monkeypatch):
    _gate_profiles(monkeypatch, True)
    body = _transcribe(client, "nope")
    assert body["profile_applied"] is None


def test_binding_lock_beats_request_profile(client, make_user_key, fake_model):
    # An admin-bound profile that LOCKS beam_size must win over a request profile
    # that sets a different beam_size — the request profile is least-specific.
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    r = client.post(f"{OV}/state", headers=admin_h, json={"OVERRIDE_PROFILES": {
        "locked": {"BEAM_SIZE": 8, "locks": ["BEAM_SIZE"]},
        "fast": {"BEAM_SIZE": 3},
    }})
    assert r.status_code == 200, r.text
    uid, raw_alice = make_user_key("alice", is_admin=False)
    r = client.patch(f"{PERMS}/{uid}/permissions", headers=admin_h,
                     json={"pages": {}, "config": {"overrides": {}, "profiles": ["locked"], "locks": []}})
    assert r.status_code == 200, r.text

    body = _transcribe(client, "fast", bearer(raw_alice))
    assert fake_model.last_kwargs["beam_size"] == 8      # binding wins; request profile shadowed
    assert body["profile_applied"] == "fast"             # it was applied (just shadowed on beam_size)


def test_override_profiles_endpoint_lists_names(client, monkeypatch):
    _gate_profiles(monkeypatch, True,
                   {"fast": {"BEAM_SIZE": 3}, "clinic-de": {"DEFAULT_LANGUAGE": "de"}})
    r = client.get("/v1/override-profiles")
    assert r.status_code == 200, r.text
    assert r.json() == {"profiles": ["clinic-de", "fast"]}   # sorted, names only


def test_override_profiles_endpoint_empty_when_gated(client, monkeypatch):
    _gate_profiles(monkeypatch, False)
    r = client.get("/v1/override-profiles")
    assert r.status_code == 200, r.text
    assert r.json() == {"profiles": []}


def test_override_profiles_endpoint_requires_auth_when_locked(client, make_user_key, monkeypatch):
    _gate_profiles(monkeypatch, True)
    # Creating the first admin key flips the app to locked-down mode.
    _, raw_admin = make_user_key("admin", is_admin=True)
    assert client.get("/v1/override-profiles").status_code == 401   # no bearer
    _, raw_alice = make_user_key("alice", is_admin=False)            # non-admin is fine
    r = client.get("/v1/override-profiles", headers=bearer(raw_alice))
    assert r.status_code == 200, r.text
    assert r.json()["profiles"] == ["fast"]


def test_none_sentinel_suppresses_bound_profile(client, make_user_key, fake_model):
    # A profile bound to a user normally applies; sending the reserved "__none__"
    # request name suppresses it, falling back to plain server defaults.
    from faster_whisper_backend.settings.schema import NO_PROFILE_SENTINEL
    admin_h, raw_alice = bind_profile(client, make_user_key, "clinic", {"BEAM_SIZE": 7})

    # Baseline: admin has no bound profile → the plain default beam_size.
    post_verbose_ok(client, admin_h)
    default_beam = fake_model.last_kwargs["beam_size"]
    assert default_beam != 7

    # Alice with the bound profile, no override → the profile applies.
    post_verbose_ok(client, bearer(raw_alice))
    assert fake_model.last_kwargs["beam_size"] == 7

    # Alice sends "__none__" → bound profile suppressed → back to the plain default.
    body = post_verbose_ok(client, bearer(raw_alice), override_profile=NO_PROFILE_SENTINEL)
    assert fake_model.last_kwargs["beam_size"] == default_beam
    assert body["profile_applied"] is None


def test_none_sentinel_never_listed(client, monkeypatch):
    from faster_whisper_backend.settings.schema import NO_PROFILE_SENTINEL
    _gate_profiles(monkeypatch, True)
    r = client.get("/v1/override-profiles")
    assert r.status_code == 200, r.text
    assert NO_PROFILE_SENTINEL not in r.json()["profiles"]
