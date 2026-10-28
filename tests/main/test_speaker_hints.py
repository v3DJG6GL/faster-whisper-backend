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


def test_client_min_beats_an_inherited_lower_max(
        client, app_module, fake_model, monkeypatch):
    calls = []
    _record_diarize(monkeypatch, calls)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_MAX_SPEAKERS", 4)
    _post(client, min_speakers=6)
    assert calls == [{"num": None, "min": 6, "max": None}]


def test_client_max_beats_an_inherited_higher_min(
        client, app_module, fake_model, monkeypatch):
    calls = []
    _record_diarize(monkeypatch, calls)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_MIN_SPEAKERS", 5)
    _post(client, max_speakers=2)
    assert calls == [{"num": None, "min": None, "max": 2}]


def test_inverted_client_range_still_clamps_min_down_to_max(
        client, app_module, fake_model, monkeypatch):
    calls = []
    _record_diarize(monkeypatch, calls)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True)
    _post(client, min_speakers=6, max_speakers=3)
    assert calls == [{"num": None, "min": 3, "max": 3}]


def test_refused_client_bound_keeps_the_inherited_count(
        client, app_module, make_user_key, fake_model, monkeypatch):
    calls = []
    _record_diarize(monkeypatch, calls)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True)
    h = _as_profile_user(client, make_user_key, DIARIZATION_NUM_SPEAKERS=3,
                         DIARIZATION_MIN_SPEAKERS=2,
                         locks=["DIARIZATION_MIN_SPEAKERS"])
    ignored = _post(client, h, min_speakers=5)
    assert calls == [{"num": 3, "min": None, "max": None}]
    assert "min_speakers" in ignored


def test_client_num_outside_a_locked_min_is_refused(
        client, app_module, make_user_key, fake_model, monkeypatch):
    calls = []
    _record_diarize(monkeypatch, calls)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True)
    h = _as_profile_user(client, make_user_key, DIARIZATION_MIN_SPEAKERS=2,
                         locks=["DIARIZATION_MIN_SPEAKERS"])
    ignored = _post(client, h, num_speakers=1)
    assert calls == [{"num": None, "min": 2, "max": None}]
    assert "num_speakers" in ignored


def test_client_max_below_a_locked_min_is_refused(
        client, app_module, make_user_key, fake_model, monkeypatch):
    calls = []
    _record_diarize(monkeypatch, calls)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True)
    h = _as_profile_user(client, make_user_key, DIARIZATION_MIN_SPEAKERS=3,
                         locks=["DIARIZATION_MIN_SPEAKERS"])
    ignored = _post(client, h, max_speakers=1)
    assert calls == [{"num": None, "min": 3, "max": None}]
    assert "max_speakers" in ignored


def test_client_min_above_a_locked_max_keeps_the_inherited_count(
        client, app_module, make_user_key, fake_model, monkeypatch):
    calls = []
    _record_diarize(monkeypatch, calls)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True)
    h = _as_profile_user(client, make_user_key, DIARIZATION_NUM_SPEAKERS=3,
                         DIARIZATION_MAX_SPEAKERS=4,
                         locks=["DIARIZATION_MAX_SPEAKERS"])
    ignored = _post(client, h, min_speakers=6)
    assert calls == [{"num": 3, "min": None, "max": None}]
    assert "min_speakers" in ignored


def test_client_max_below_a_locked_min_keeps_the_inherited_count(
        client, app_module, make_user_key, fake_model, monkeypatch):
    calls = []
    _record_diarize(monkeypatch, calls)
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True)
    h = _as_profile_user(client, make_user_key, DIARIZATION_NUM_SPEAKERS=4,
                         DIARIZATION_MIN_SPEAKERS=3,
                         locks=["DIARIZATION_MIN_SPEAKERS"])
    ignored = _post(client, h, max_speakers=1)
    assert calls == [{"num": 4, "min": None, "max": None}]
    assert "max_speakers" in ignored
