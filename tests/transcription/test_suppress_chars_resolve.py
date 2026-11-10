"""_resolve_suppress_chars: multi-piece chars are skipped with ONE log line
per resolve, not one per char variant (a client picks the chars)."""

import logging
from types import SimpleNamespace

import pytest

from faster_whisper_backend.transcription import models as tx_models


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(tx_models, "_suppress_chars_cache", tx_models.OrderedDict())


def _model(encode):
    return SimpleNamespace(hf_tokenizer=SimpleNamespace(
        encode=lambda v, add_special_tokens=False: SimpleNamespace(ids=encode(v))))


def _skip_records(caplog):
    return [r for r in caplog.records if "SUPPRESS_CHARS skipping" in r.getMessage()]


def test_client_multi_piece_chars_log_one_debug_line(caplog):
    model = _model(lambda v: [1, 2])
    with caplog.at_level(logging.DEBUG, logger="whisper-api"):
        assert tx_models._resolve_suppress_chars("m", model, "🙂" * 64, True) == ()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    skips = _skip_records(caplog)
    assert len(skips) == 1 and skips[0].levelno == logging.DEBUG


def test_config_multi_piece_chars_log_one_bounded_warning(caplog):
    model = _model(lambda v: [1, 2])
    chars = "".join(chr(0x1F600 + i) for i in range(64))
    with caplog.at_level(logging.DEBUG, logger="whisper-api"):
        assert tx_models._resolve_suppress_chars("m", model, chars) == ()
    warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warns) == 1 and warns[0] is _skip_records(caplog)[0]
    msg = warns[0].getMessage()
    assert "128" in msg                      # every skipped variant is counted
    assert len(msg) < 600                    # ...but only a few are listed


def test_single_piece_chars_resolve_to_the_same_ids(caplog):
    model = _model(lambda v: [ord(v[-1])] if v[-1] != "!" else [7, 8])
    with caplog.at_level(logging.DEBUG, logger="whisper-api"):
        out = tx_models._resolve_suppress_chars("m", model, ".,.? !")
    assert out == (ord(","), ord("."), ord("?"))
    assert len(_skip_records(caplog)) == 1
