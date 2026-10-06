"""Helpers shared by the text-translation route tests and the jobs tests
that drive /v1/text/translations (stubbed translator, enable switch, request
body). Import as ``from tests._translation_helpers import ...``."""

from faster_whisper_backend.translation import engine as translation


def stub_translate(monkeypatch, calls=None):
    """Per-index-verifiable stub: segment i translates to '<text>-<target>'."""
    async def _fake(segments, targets, *, source_lang=None, model_ref=None,
                    mode="fluent", glossary="", context_segments=None,
                    progress_cb=None, cancel_check=None, download_cb=None):
        if calls is not None:
            calls.append({"segments": segments, "targets": list(targets),
                          "source_lang": source_lang, "model_ref": model_ref,
                          "mode": mode, "glossary": glossary,
                          "context_segments": context_segments})
        per_seg = [{t: f"{seg['text']}-{t}" for t in targets}
                   for seg in segments]
        return per_seg, [], {"model": (model_ref or "").strip() or "org/d:Q4",
                             "source": source_lang or "", "mode": mode}
    monkeypatch.setattr(translation, "translate_segments", _fake)


def enable_translation(app_module, monkeypatch, **cfg_fields):
    monkeypatch.setattr(app_module.cfg, "TRANSLATION_ENABLED", True,
                        raising=False)
    for name, value in cfg_fields.items():
        monkeypatch.setattr(app_module.cfg, name, value, raising=False)


def text_translation_body(**overrides):
    body = {"segments": [{"id": 7, "text": "eins", "speaker": "SPEAKER_00"},
                         {"id": 3, "text": "zwei"}],
            "targets": ["en"]}
    body.update(overrides)
    return body
