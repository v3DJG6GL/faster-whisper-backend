"""Helpers shared by the stage tests that drive POST /v1/audio/transcriptions
(the stub upload, a plain / verbose_json post, a profile-bound non-admin key).
Import as ``from tests._transcription_helpers import ...``."""

from tests.conftest import bearer

WAV_UPLOAD = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


def post_transcription(client, headers=None, **data):
    """POST the stub upload with form fields `data` (model defaults to whisper-1)."""
    data.setdefault("model", "whisper-1")
    return client.post("/v1/audio/transcriptions", files=WAV_UPLOAD, data=data,
                       headers=headers)


def post_verbose(client, headers=None, **data):
    data.setdefault("response_format", "verbose_json")
    return post_transcription(client, headers, **data)


def bind_profile(client, make_user_key, name, fields):
    """Create admin + non-admin 'alice' bound to the one OVERRIDE_PROFILES
    entry `name` = `fields`; return (admin_headers, alice_raw_key)."""
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    r = client.post("/settings/overrides/state", headers=admin_h,
                    json={"OVERRIDE_PROFILES": {name: fields}})
    assert r.status_code == 200, r.text
    uid, raw_alice = make_user_key("alice", is_admin=False)
    r = client.patch(
        f"/settings/api-keys/api/users/{uid}/permissions", headers=admin_h,
        json={"pages": {}, "config": {"overrides": {},
                                      "profiles": [name], "locks": []}})
    assert r.status_code == 200, r.text
    return admin_h, raw_alice
