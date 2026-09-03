"""The default `json` response shape on POST /v1/audio/transcriptions must
carry the same additive keys as verbose_json: an ignored (locked) client
override, the applied request profile, and the translation guard's kept
map — none of them should require verbose_json to be seen."""

from faster_whisper_backend import config as cfg
from faster_whisper_backend.audio import translation
from tests.conftest import bearer

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}
OV = "/settings/overrides"
PERMS = "/settings/api-keys/api/users"


def _post(client, headers=None, **data):
    data.setdefault("model", "whisper-1")
    data.setdefault("response_format", "json")
    return client.post("/v1/audio/transcriptions", files=_FILE, data=data,
                       headers=headers or {})


def test_plain_json_reports_locked_override_as_ignored(client, make_user_key,
                                                       fake_model):
    # Mirrors test_transcription_overrides::test_locked_task_ignores_client_param
    # with the default response_format — the /v1/audio/translations docstring
    # promises overrides_ignored: ["task"] on its own default shape.
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    r = client.post(f"{OV}/state", headers=admin_h,
                    json={"OVERRIDE_PROFILES": {"notask": {"locks": ["TASK"]}}})
    assert r.status_code == 200, r.text
    uid, raw_alice = make_user_key("alice", is_admin=False)
    client.patch(f"{PERMS}/{uid}/permissions", headers=admin_h,
                 json={"pages": {}, "config": {"overrides": {},
                                                "profiles": ["notask"],
                                                "locks": []}})
    r = _post(client, headers=bearer(raw_alice), task="translate")
    assert r.status_code == 200, r.text
    assert "task" not in fake_model.last_kwargs
    assert "task" in r.json()["overrides_ignored"]
    # Nothing locked and no profile asked for: the plain shape stays plain.
    r = _post(client, headers=bearer(raw_admin))
    assert r.status_code == 200, r.text
    assert "overrides_ignored" not in r.json()
    assert "profile_applied" not in r.json()


def test_plain_json_echoes_profile_applied(client, fake_model, monkeypatch):
    monkeypatch.setattr(cfg, "OVERRIDE_PROFILES", {"fast": {"BEAM_SIZE": 3}},
                        raising=False)
    monkeypatch.setattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", True,
                        raising=False)
    r = _post(client, override_profile="fast")
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["beam_size"] == 3
    assert r.json()["profile_applied"] == "fast"


def test_plain_json_translation_meta_names_kept_originals(client, app_module,
                                                          monkeypatch):
    # A guard fallback that kept the SOURCE text is flagged per segment only
    # in verbose_json; the shared meta carries target -> segment indices so
    # the joined-text shape can tell a kept German line from a translation.
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)

    async def _fake(segments, targets, **kwargs):
        per_seg = [{t: seg["text"] for t in targets} for seg in segments]
        return per_seg, ["segment 1: kept original — translation failed "
                         "(length ratio)"], {
            "model": "org/d:Q4", "source": "de", "mode": "fluent",
            "kept": {0: list(targets)}}
    monkeypatch.setattr(translation, "translate_segments", _fake)

    r = _post(client, translate_to="en")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "segments" not in body
    assert body["translation"]["kept"] == {"en": [0]}
    # verbose_json exposes the same map beside its per-segment markers.
    r = _post(client, translate_to="en", response_format="verbose_json")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["translation"]["kept"] == {"en": [0]}
    assert body["segments"][0]["translations_kept"] == ["en"]


def test_plain_json_translation_meta_kept_is_explicit_empty(client, app_module,
                                                            monkeypatch):
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)

    async def _fake(segments, targets, **kwargs):
        per_seg = [{t: f"XLATED-{t}" for t in targets} for _ in segments]
        return per_seg, [], {"model": "org/d:Q4", "source": "de",
                             "mode": "fluent"}
    monkeypatch.setattr(translation, "translate_segments", _fake)
    r = _post(client, translate_to="en")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["translations"] == {"en": "XLATED-en"}
    assert body["translation"]["kept"] == {}
