"""Anti-hallucination guards for the residual-window echo failure mode.

Whisper re-decodes the sub-second leftover after the last aligned word as its
own zero-padded window and confidently replays its text context (prompt echo /
self-repetition). Observed signature: 20+ words crammed into ~0.5 s at the
buffer tail, high avg_logprob, T=0.0, nsp possibly below NO_SPEECH_THRESHOLD.

Covers:
  * main.segment_exceeds_word_rate (SEGMENT_MAX_WORDS_PER_SEC) — unit
  * batch POST /v1/audio/transcriptions drops word-rate-anomalous segments
  * streaming_routes._trim_trailing_nonspeech (STREAMING_TAIL_TRIM_PAD_MS)
  * streaming FINAL decode's condition_on_previous_text override
    (STREAMING_FINAL_CONDITION_ON_PREVIOUS_TEXT), batch left untouched
"""

import numpy as np

from conftest import RATE, FakeSegment, FakeWord

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


def _words(text: str, start: float, end: float) -> list:
    """Evenly spread FakeWords for `text` across [start, end]."""
    toks = text.split()
    step = (end - start) / max(len(toks), 1)
    return [FakeWord(t, start + i * step, start + (i + 1) * step)
            for i, t in enumerate(toks)]


def _echo_segment() -> FakeSegment:
    """Incident-2 shape: ~22 words in 0.54 s (~40 w/s), confident stats."""
    text = ("Punkt Neue Zeile Die Unterschenkel seien stark wenn er auf dem "
            "Fuss steht Punkt Neue Zeile Die Unterschenkel seien stark wenn "
            "er auf dem Fuss steht")
    return FakeSegment(text, 13.42, 13.96, words=_words(text, 13.42, 13.96))


# ---------------------------------------------------------------------------
# segment_exceeds_word_rate
# ---------------------------------------------------------------------------

def test_word_rate_normal_speech_passes(app_module):
    seg = FakeSegment("hallo schöne welt", 0.0, 1.0,
                      words=_words("hallo schöne welt", 0.0, 1.0))
    assert app_module.segment_exceeds_word_rate(seg, 10.0) is False


def test_word_rate_echo_segment_dropped(app_module):
    assert app_module.segment_exceeds_word_rate(_echo_segment(), 10.0) is True


def test_word_rate_needs_min_words(app_module):
    # 2 words in 0.1 s is 20 w/s but below the 3-word floor -> never dropped.
    seg = FakeSegment("ja gut", 0.0, 0.1, words=_words("ja gut", 0.0, 0.1))
    assert app_module.segment_exceeds_word_rate(seg, 10.0) is False


def test_word_rate_zero_duration_with_words_dropped(app_module):
    seg = FakeSegment("eins zwei drei", 1.0, 1.0,
                      words=_words("eins zwei drei", 1.0, 1.0))
    assert app_module.segment_exceeds_word_rate(seg, 10.0) is True


def test_word_rate_disabled_by_zero(app_module):
    assert app_module.segment_exceeds_word_rate(_echo_segment(), 0.0) is False
    assert app_module.segment_exceeds_word_rate(_echo_segment(), 0) is False


def test_word_rate_falls_back_to_text_when_no_words(app_module):
    # word_timestamps off -> segment has no word list; count text tokens.
    seg = FakeSegment("eins zwei drei vier fünf sechs", 0.0, 0.3, words=[])
    assert app_module.segment_exceeds_word_rate(seg, 10.0) is True


# ---------------------------------------------------------------------------
# Batch endpoint integration
# ---------------------------------------------------------------------------

def test_batch_drops_echo_segment(client, fake_model):
    real = FakeSegment("hallo welt", 0.0, 1.0,
                       words=_words("hallo welt", 0.0, 1.0))
    fake_model._segments = [real, _echo_segment()]
    r = client.post("/v1/audio/transcriptions", files=_FILE,
                    data={"model": "whisper-1",
                          "response_format": "verbose_json"})
    assert r.status_code == 200
    body = r.json()
    assert "Unterschenkel" not in body["text"]
    assert body["text"] == "hallo welt"
    # Kept segments only, with contiguous ids.
    assert [s["id"] for s in body["segments"]] == [0]
    assert body["segments"][0]["text"] == "hallo welt"
    # Dropped segment's words don't leak either.
    assert all("Unterschenkel" not in w["word"] for w in body.get("words", []))


def test_batch_guard_disabled_keeps_everything(client, app_module, fake_model):
    app_module.cfg.SEGMENT_MAX_WORDS_PER_SEC = 0
    real = FakeSegment("hallo welt", 0.0, 1.0,
                       words=_words("hallo welt", 0.0, 1.0))
    fake_model._segments = [real, _echo_segment()]
    r = client.post("/v1/audio/transcriptions", files=_FILE,
                    data={"model": "whisper-1",
                          "response_format": "verbose_json"})
    assert r.status_code == 200
    body = r.json()
    assert "Unterschenkel" in body["text"]
    assert len(body["segments"]) == 2


def test_batch_conditioning_unchanged(client, fake_model):
    # The streaming-final override must NOT leak into batch decodes: batch
    # keeps the per-model CONDITION_ON_PREVIOUS_TEXT (default true).
    r = client.post("/v1/audio/transcriptions", files=_FILE,
                    data={"model": "whisper-1"})
    assert r.status_code == 200
    assert fake_model.last_kwargs["condition_on_previous_text"] is True


# ---------------------------------------------------------------------------
# _trim_trailing_nonspeech
# ---------------------------------------------------------------------------

def _speech_then_silence(speech_sec: float, silence_sec: float) -> np.ndarray:
    audio = np.zeros(int((speech_sec + silence_sec) * RATE), dtype=np.float32)
    audio[: int(speech_sec * RATE)] = 0.1
    return audio


def test_tail_trim_cuts_trailing_silence(fake_vad):
    import streaming_routes as sr
    audio = _speech_then_silence(1.0, 2.0)
    out = sr._trim_trailing_nonspeech(audio, pad_ms=300, threshold=0.5)
    assert out.shape[0] == RATE + (300 * RATE) // 1000


def test_tail_trim_no_speech_returns_unchanged(fake_vad):
    import streaming_routes as sr
    audio = np.zeros(2 * RATE, dtype=np.float32)
    out = sr._trim_trailing_nonspeech(audio, pad_ms=300, threshold=0.5)
    assert out.shape[0] == audio.shape[0]


def test_tail_trim_disabled_by_zero_pad(fake_vad):
    import streaming_routes as sr
    audio = _speech_then_silence(1.0, 2.0)
    out = sr._trim_trailing_nonspeech(audio, pad_ms=0, threshold=0.5)
    assert out is audio


def test_tail_trim_speech_to_the_end_untouched(fake_vad):
    import streaming_routes as sr
    audio = np.full(2 * RATE, 0.1, dtype=np.float32)
    out = sr._trim_trailing_nonspeech(audio, pad_ms=300, threshold=0.5)
    assert out.shape[0] == audio.shape[0]


def test_tail_trim_vad_unavailable_returns_unchanged(no_vad):
    import streaming_routes as sr
    audio = _speech_then_silence(1.0, 2.0)
    out = sr._trim_trailing_nonspeech(audio, pad_ms=300, threshold=0.5)
    assert out is audio


# ---------------------------------------------------------------------------
# Streaming FINAL conditioning override
# ---------------------------------------------------------------------------

def test_streaming_final_conditioning_off_by_default(app_module, fake_model):
    import streaming_routes as sr
    kw = sr._build_transcribe_kwargs(
        app_module, "some-model", final=True, prompt="",
        want_words=False, model_obj=fake_model)
    assert kw["condition_on_previous_text"] is False


def test_streaming_final_conditioning_config_on(app_module, fake_model):
    app_module.cfg.STREAMING_FINAL_CONDITION_ON_PREVIOUS_TEXT = True
    import streaming_routes as sr
    kw = sr._build_transcribe_kwargs(
        app_module, "some-model", final=True, prompt="",
        want_words=False, model_obj=fake_model)
    assert kw["condition_on_previous_text"] is True


def test_streaming_partial_conditioning_unaffected(app_module, fake_model):
    import streaming_routes as sr
    kw = sr._build_transcribe_kwargs(
        app_module, "some-model", final=False, prompt="",
        want_words=False, model_obj=fake_model)
    # Partials keep their own knob (default off) and stay vad_filter-less.
    assert kw["condition_on_previous_text"] is False
    assert kw["vad_filter"] is False


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_admin_config_accepts_new_fields():
    import config_store
    m = config_store.AdminConfig(
        SEGMENT_MAX_WORDS_PER_SEC=8.0,
        STREAMING_TAIL_TRIM_PAD_MS=500,
        STREAMING_FINAL_CONDITION_ON_PREVIOUS_TEXT=True,
    )
    assert m.SEGMENT_MAX_WORDS_PER_SEC == 8.0
    assert m.STREAMING_TAIL_TRIM_PAD_MS == 500
    assert m.STREAMING_FINAL_CONDITION_ON_PREVIOUS_TEXT is True


def test_admin_config_rejects_out_of_range():
    import pytest as _pytest
    import config_store
    with _pytest.raises(Exception):
        config_store.AdminConfig(SEGMENT_MAX_WORDS_PER_SEC=-1.0)
    with _pytest.raises(Exception):
        config_store.AdminConfig(STREAMING_TAIL_TRIM_PAD_MS=999999)


def test_defaults_present_in_config(app_module):
    assert app_module.cfg.SEGMENT_MAX_WORDS_PER_SEC == 10.0
    assert app_module.cfg.STREAMING_TAIL_TRIM_PAD_MS == 300
    assert app_module.cfg.STREAMING_FINAL_CONDITION_ON_PREVIOUS_TEXT is False
