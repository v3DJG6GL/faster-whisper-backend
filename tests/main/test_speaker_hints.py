"""Diarization speaker hints on /v1/audio/transcriptions: how the client's
num/min/max_speakers combine with inherited and LOCKED DIARIZATION_* config.
Asserts the kwargs the stubbed diarize() receives + overrides_ignored."""

from tests.conftest import bearer
from tests.main.test_transcription_overrides import (
    _FILE, PERMS, _record_diarize, _setup_profile,
)


def _as_profile_user(client, make_user_key, **fields):
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    _setup_profile(client, admin_h, "spk", **fields)
    uid, raw_alice = make_user_key("alice", is_admin=False)
    client.patch(f"{PERMS}/{uid}/permissions", headers=admin_h,
                 json={"pages": {}, "config": {"overrides": {},
                       "profiles": ["spk"], "locks": []}})
    return bearer(raw_alice)


def _post(client, headers=None, **hints):
    data = {"model": "whisper-1", "response_format": "verbose_json",
            "diarize": "true", **{k: str(v) for k, v in hints.items()}}
    r = client.post("/v1/audio/transcriptions", files=_FILE,
                    headers=headers or {}, data=data)
    assert r.status_code == 200, r.text
    return r.json().get("overrides_ignored", [])


def test_explicit_zero_num_speakers_auto_detects_instead_of_inheriting(
        client, app_module, fake_model, monkeypatch):
    calls = []
    _record_diarize(monkeypatch, calls)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_NUM_SPEAKERS", 2)
    _post(client, num_speakers=0)
    assert calls == [{"num": None, "min": None, "max": None}]


def test_explicit_zero_against_a_locked_count_is_reported_ignored(
        client, app_module, make_user_key, fake_model, monkeypatch):
    calls = []
    _record_diarize(monkeypatch, calls)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True)
    h = _as_profile_user(client, make_user_key, DIARIZATION_NUM_SPEAKERS=3,
                         locks=["DIARIZATION_NUM_SPEAKERS"])
    ignored = _post(client, h, num_speakers=0)
    assert calls == [{"num": 3, "min": None, "max": None}]
    assert "num_speakers" in ignored
