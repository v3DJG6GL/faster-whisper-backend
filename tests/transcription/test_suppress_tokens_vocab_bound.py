"""suppress_tokens ids are bounded by the LOADED model's vocabulary, not only
by the largest Whisper vocabulary (large-v3's 51866): large-v2 / small end at
51865 and the English-only models at 51864, and CTranslate2 writes a
suppressed id into the logits without a range check."""

from types import SimpleNamespace

from faster_whisper_backend.settings import effective_config as ec
from faster_whisper_backend.transcription import models as tx_models


def _whisper(is_multilingual, num_languages):
    """A WhisperModel stand-in carrying the CTranslate2 model's attributes."""
    return SimpleNamespace(model=SimpleNamespace(is_multilingual=is_multilingual,
                                                 num_languages=num_languages))


def _assemble(model, values=None, overrides=None):
    return tx_models.assemble_transcribe_kwargs(
        None, model, language="", temperature=0.0, vad_filter=False,
        vad_parameters=None, want_word_ts=False, initial_prompt=None,
        overrides=overrides, ident=ec.Resolved(values=values or {}))


def test_model_suppress_id_max_matches_the_vocabulary_sizes():
    # large-v2 / small: 99 languages, 51865 tokens; large-v3 / turbo: 100, 51866
    assert tx_models._model_suppress_id_max(_whisper(True, 99)) == 51865
    assert tx_models._model_suppress_id_max(_whisper(True, 100)) == 51866
    assert tx_models._model_suppress_id_max(_whisper(False, 0)) == 51864
    # never looser than the global bound
    assert tx_models._model_suppress_id_max(_whisper(True, 500)) \
        == tx_models._SUPPRESS_TOKEN_ID_MAX


def test_unknown_model_shape_falls_back_to_the_global_bound():
    class _Unloaded:
        @property
        def is_multilingual(self):
            raise RuntimeError("model is unloaded")

    for model in (None, object(), SimpleNamespace(model=_Unloaded()),
                  SimpleNamespace(model=SimpleNamespace()),
                  SimpleNamespace(model=SimpleNamespace(is_multilingual=object(),
                                                        num_languages=object()))):
        assert tx_models._model_suppress_id_max(model) \
            == tx_models._SUPPRESS_TOKEN_ID_MAX, model


def test_client_override_drops_ids_past_the_loaded_vocabulary():
    ov = {"suppress_tokens": [50364, 51864, 51865]}
    # large-v2: 51865 is one past the vocabulary; timestamp ids stay
    assert _assemble(_whisper(True, 99), overrides=ov)["suppress_tokens"] \
        == [50364, 51864]
    # large-v3: 51865 is a real token
    assert _assemble(_whisper(True, 100), overrides=ov)["suppress_tokens"] \
        == [50364, 51864, 51865]
    # *.en: 51864 is past the vocabulary too
    assert _assemble(_whisper(False, 0), overrides=ov)["suppress_tokens"] == [50364]


def test_override_whose_ids_all_filter_away_keeps_the_config_value():
    kw = _assemble(_whisper(True, 99), values={"SUPPRESS_TOKENS": "-1,50256"},
                   overrides={"suppress_tokens": [51865]})
    assert kw["suppress_tokens"] == [-1, 50256]


def test_config_value_drops_ids_past_the_loaded_vocabulary():
    kw = _assemble(_whisper(True, 99), values={"SUPPRESS_TOKENS": "-1,51864,51865"})
    assert kw["suppress_tokens"] == [-1, 51864]
    # nothing left in range: "not set", never an explicit clear
    kw = _assemble(_whisper(False, 0), values={"SUPPRESS_TOKENS": "51864,51865"})
    assert "suppress_tokens" not in kw
