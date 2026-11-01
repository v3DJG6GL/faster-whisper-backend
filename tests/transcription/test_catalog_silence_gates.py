"""GET /v1/request-default-settings reports the live-dictation silence gates a
session will actually run: fitted as a set (streaming/routes._stream_config),
not the raw resolved values the client's "Inherit · <value>" labels would
otherwise show."""

import pytest

from faster_whisper_backend.settings import effective_config
from faster_whisper_backend.streaming import routes as stream_routes

URL = "/v1/request-default-settings"
INNER, OUTER, HB = ("STREAMING_VAD_INNER_SILENCE_MS", "STREAMING_VAD_OUTER_SILENCE_MS",
                    "STREAMING_HARD_BREAK_SILENCE_MS")


@pytest.fixture(autouse=True)
def _gates(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "DEFAULT_MODEL", "tiny")
    monkeypatch.setattr(app_module.cfg, "ALLOWED_MODELS", set())
    # A pair AdminConfig would refuse on save but a hand edit / env can hold.
    for field, v in ((INNER, 700), (OUTER, 6000), (HB, 5000)):
        monkeypatch.setattr(app_module.cfg, field, v)
        monkeypatch.delenv(f"WHISPER_{field}", raising=False)


def _shown(client):
    s = client.get(URL).json()["settings"]
    return tuple(s[effective_config._CONFIG_TO_CLIENT_KEY[f]]["value"]
                 for f in (INNER, OUTER, HB))


def _session():
    sc = stream_routes._stream_config(effective_config.cfg_for)
    return sc.vad_min_silence_ms, sc.commit_silence_ms, sc.hard_break_silence_ms


def test_silence_gates_are_shown_as_the_session_fits_them(client):
    assert _shown(client) == _session() == (700, 6000, 7000)


def test_an_env_pinned_hard_break_is_shown_unmoved(client, monkeypatch):
    monkeypatch.setenv(f"WHISPER_{HB}", "5000")
    assert _shown(client) == _session() == (700, 4000, 5000)
