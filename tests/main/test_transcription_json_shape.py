"""The default `json` response shape on POST /v1/audio/transcriptions must
carry the same additive keys as verbose_json: an ignored (locked) client
override, the applied request profile, and the translation guard's kept
map — none of them should require verbose_json to be seen."""

from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.translation import engine as translation
from tests.conftest import FakeSegment, bearer
from tests._translation_helpers import enable_translation, stub_translate
from tests._transcription_helpers import bind_profile, post_json_ok, post_verbose_ok


def _stub_translated(monkeypatch, text_for, *warnings, source="en", **meta):
    """translate_segments answers text_for(target, index, segment) per target."""
    async def _fake(segments, targets, **kwargs):
        return [{t: text_for(t, i, seg) for t in targets}
                for i, seg in enumerate(segments)], list(warnings), {
            "model": "org/d:Q4", "source": source, "mode": "fluent", **meta}
    monkeypatch.setattr(translation, "translate_segments", _fake)


def test_plain_json_reports_locked_override_as_ignored(client, make_user_key,
                                                       fake_model):
    # Mirrors test_transcription_overrides::test_locked_task_ignores_client_param
    # with the default response_format — the /v1/audio/translations docstring
    # promises overrides_ignored: ["task"] on its own default shape.
    admin_h, raw_alice = bind_profile(client, make_user_key, "notask",
                                      {"locks": ["TASK"]})
    body = post_json_ok(client, headers=bearer(raw_alice), task="translate")
    assert "task" not in fake_model.last_kwargs
    assert "task" in body["overrides_ignored"]
    # Nothing locked and no profile asked for: the plain shape stays plain.
    body = post_json_ok(client, headers=admin_h)
    assert "overrides_ignored" not in body
    assert "profile_applied" not in body


def test_plain_json_echoes_profile_applied(client, fake_model, monkeypatch):
    monkeypatch.setattr(cfg, "OVERRIDE_PROFILES", {"fast": {"BEAM_SIZE": 3}},
                        raising=False)
    monkeypatch.setattr(cfg, "ALLOW_REQUEST_OVERRIDE_PROFILE", True,
                        raising=False)
    body = post_json_ok(client, override_profile="fast")
    assert fake_model.last_kwargs["beam_size"] == 3
    assert body["profile_applied"] == "fast"


def test_plain_json_translation_meta_names_kept_originals(client, app_module,
                                                          monkeypatch):
    # A guard fallback that kept the SOURCE text is flagged per segment only
    # in verbose_json; the shared meta carries target -> segment indices so
    # the joined-text shape can tell a kept German line from a translation.
    enable_translation(app_module, monkeypatch)
    _stub_translated(monkeypatch, lambda t, i, seg: seg["text"],
                     "segment 1: kept original — translation failed (length ratio)",
                     source="de", kept={0: ["en"]})

    body = post_json_ok(client, translate_to="en")
    assert "segments" not in body
    assert body["translation"]["kept"] == {"en": [0]}
    # verbose_json exposes the same map beside its per-segment markers.
    body = post_verbose_ok(client, translate_to="en")
    assert body["translation"]["kept"] == {"en": [0]}
    assert body["segments"][0]["translations_kept"] == ["en"]


def test_plain_json_translation_meta_kept_is_explicit_empty(client, app_module,
                                                            monkeypatch):
    enable_translation(app_module, monkeypatch)
    stub_translate(monkeypatch)
    body = post_json_ok(client, translate_to="en")
    assert body["translations"] == {"en": "hallo welt-en"}
    assert body["translation"]["kept"] == {}


def test_joined_translations_glue_unspaced_pieces(client, app_module, fake_model,
                                                  monkeypatch, created_captures):
    # A fluent zh group cut back across two cues must rejoin with no space
    # inside 商店 ("store"), in the joined field and the capture row.
    fake_model._segments = [FakeSegment("I went to the store", 0.0, 1.0),
                            FakeSegment("to buy some bread today.", 1.0, 2.0)]
    enable_translation(app_module, monkeypatch, CAPTURES_RECORDING_ENABLED=True)
    pieces = {"zh": ["我今天去商", "店买一些面包。"], "en": ["Hi", "there."]}
    _stub_translated(monkeypatch, lambda t, i, seg: pieces[t][i])

    want = {"zh": "我今天去商店买一些面包。", "en": "Hi there."}
    for fmt in ("json", "verbose_json"):
        assert post_json_ok(client, translate_to="zh,en",
                            response_format=fmt)["translations"] == want
    assert created_captures[-1]["translations"] == want


def test_joined_translations_glue_a_digit_to_han_only_for_a_zh_target(
        client, app_module, fake_model, monkeypatch, created_captures):
    # The joined field and the capture row are built per target: zh glues
    # '3' | '个', while an en target quoting the same pieces keeps its space.
    fake_model._segments = [FakeSegment("I bought 3", 0.0, 1.0),
                            FakeSegment("apples.", 1.0, 2.0)]
    enable_translation(app_module, monkeypatch, CAPTURES_RECORDING_ENABLED=True)
    pieces = ["我买了3", "个苹果。"]
    _stub_translated(monkeypatch, lambda t, i, seg: pieces[i])
    body = post_json_ok(client, translate_to="zh,en")
    want = {"zh": "我买了3个苹果。", "en": "我买了3 个苹果。"}
    assert body["translations"] == want
    assert created_captures[-1]["translations"] == want


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
