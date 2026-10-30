"""The default `json` response shape on POST /v1/audio/transcriptions must
carry the same additive keys as verbose_json: an ignored (locked) client
override, the applied request profile, and the translation guard's kept
map — none of them should require verbose_json to be seen."""

from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.translation import engine as translation
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


def test_joined_translations_glue_unspaced_pieces(client, app_module,
                                                  fake_model, monkeypatch):
    # A fluent zh group cut back across two cues, rejoined with " ", put a
    # space inside 商店 ("store") in the joined field and the capture row.
    from faster_whisper_backend.captures import store as captures_store
    from tests.conftest import FakeSegment
    fake_model._segments = [FakeSegment("I went to the store", 0.0, 1.0),
                            FakeSegment("to buy some bread today.", 1.0, 2.0)]
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)
    monkeypatch.setattr(app_module.cfg, "CAPTURES_RECORDING_ENABLED", True,
                        raising=False)
    pieces = {"zh": ["我今天去商", "店买一些面包。"], "en": ["Hi", "there."]}

    async def _fake(segments, targets, **kwargs):
        return [{t: pieces[t][i] for t in targets}
                for i in range(len(segments))], [], {
            "model": "org/d:Q4", "source": "en", "mode": "fluent"}
    monkeypatch.setattr(translation, "translate_segments", _fake)
    stored = {}

    def _spy_create(**kw):
        stored.update(kw)
        return "cap-test-id"
    monkeypatch.setattr(captures_store, "create_capture", _spy_create)

    want = {"zh": "我今天去商店买一些面包。", "en": "Hi there."}
    for fmt in ("json", "verbose_json"):
        r = _post(client, translate_to="zh,en", response_format=fmt)
        assert r.status_code == 200, r.text
        assert r.json()["translations"] == want
    assert stored["translations"] == want


def test_join_translations_spacing():
    j = translation.join_translations
    assert j(["你好。", "你好吗？"]) == "你好。你好吗？"
    assert j(["他说：“你好。”", "我很好"]) == "他说：“你好。”我很好"
    assert j(["こんにちは、", "元気？"]) == "こんにちは、元気？"
    # A Latin side keeps its space, and so do Thai and Korean.
    assert j(["我用", "Python 写代码"]) == "我用 Python 写代码"
    assert j(["Hello", "world.", "", " "]) == "Hello world."
    assert j(["สวัสดี", "ครับ"]) == "สวัสดี ครับ"
    assert j(["안녕하세요.", "반갑습니다."]) == "안녕하세요. 반갑습니다."
    assert j([]) == ""
