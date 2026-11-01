"""The live-dictation silence gates (inner < outer < hard break) are fitted as
a set at runtime (settings/effective_config.fit_silence_gates), and a gate
the admin LOCKED — by a profile lock or a WHISPER_* env var — is never the
one that moves. Driven through streaming/routes' own builders, never a
streamed socket."""

import logging

import pytest

from faster_whisper_backend.settings import effective_config
from faster_whisper_backend.streaming import routes

INNER, OUTER, HB = ("STREAMING_VAD_INNER_SILENCE_MS", "STREAMING_VAD_OUTER_SILENCE_MS",
                    "STREAMING_HARD_BREAK_SILENCE_MS")


@pytest.fixture(autouse=True)
def _global_gates(monkeypatch):
    for field, v in ((INNER, 700), (OUTER, 1200), (HB, 5000)):
        monkeypatch.setattr(effective_config.cfg, field, v)
        monkeypatch.delenv(f"WHISPER_{field}", raising=False)


def _gates(ident=None, client=None):
    sc = routes._stream_config(effective_config.cfg_for, ident, client=client)
    return sc.vad_min_silence_ms, sc.commit_silence_ms, sc.hard_break_silence_ms


def _ident(locked=(), **values):
    keys = {effective_config._CONFIG_TO_CLIENT_KEY[f] for f in locked}
    return effective_config.Resolved(values=values, locked=set(locked),
                                     locked_client_keys=frozenset(keys))


def test_a_locked_hard_break_moves_the_outer_instead():
    # A profile outer of 6000 above a hard break the admin locked at 5000:
    # the locked hard break stays, the outer comes down below it.
    assert _gates(_ident(locked=[HB], **{OUTER: 6000})) == (700, 4000, 5000)
    # Unlocked, the hard break still follows the outer as before.
    assert _gates(_ident(**{OUTER: 6000})) == (700, 6000, 7000)


def test_a_locked_inner_moves_the_outer_instead():
    assert _gates(_ident(locked=[INNER], **{INNER: 1300})) == (1300, 1350, 5000)


def test_an_env_pinned_gate_is_locked_unless_a_profile_sets_it(monkeypatch):
    monkeypatch.setenv(f"WHISPER_{HB}", "5000")
    assert _gates(_ident(**{OUTER: 6000})) == (700, 4000, 5000)
    # A profile's own hard break runs instead of the env value: no pin on it.
    assert _gates(_ident(**{OUTER: 6000, HB: 5000})) == (700, 6000, 7000)


def test_locked_gates_that_cannot_fit_are_kept_and_reported(caplog):
    with caplog.at_level(logging.WARNING, logger=routes.logger.name):
        got = _gates(_ident(locked=[INNER, OUTER], **{INNER: 1300}))
    assert got == (1300, 1200, 5000)
    assert any("do not fit and are locked" in r.getMessage() for r in caplog.records)


def test_b_client_outer_is_judged_against_the_gate_the_session_runs():
    # The admin locked the hard break at 5000; a client outer of 6000 would
    # need it moved, so the outer is dropped and the session keeps 5000 (the
    # old refit moved the "locked" hard break to 7000 anyway).
    ident = _ident(locked=[HB], **{OUTER: 6000})
    values, notes = routes._client_stream_values(
        {"streaming_vad_outer_silence_ms": 6000}, ident)
    assert values == {} and notes[0].startswith("outer 6000 ms dropped")
    assert _gates(ident, values) == (700, 4000, 5000)
    # A client outer that fits the locked hard break applies unchanged.
    values, notes = routes._client_stream_values(
        {"streaming_vad_outer_silence_ms": 3000}, ident)
    assert values == {OUTER: 3000} and notes == []
    assert _gates(ident, values) == (700, 3000, 5000)


def test_b_client_inner_follows_an_outer_moved_by_a_locked_partner():
    ident = _ident(locked=[HB], **{HB: 1500, OUTER: 6000})
    values, notes = routes._client_stream_values(
        {"streaming_vad_inner_silence_ms": 900}, ident)
    assert values == {INNER: 450} and notes == ["inner 900→450 ms (below outer 500)"]
    assert _gates(ident, values) == (450, 500, 1500)


def test_d_a_hard_break_within_a_frame_of_outer_is_fitted():
    # Silence is counted in 32 ms frames: with outer 1200 the finalize fires
    # at 1216 ms and the hard-break check sees 1248 ms one frame later, so a
    # hard break of 1240 broke after every utterance just like 1200 does.
    assert _gates(_ident(**{HB: 1240})) == (700, 1200, 2200)
    assert _gates(_ident(**{HB: 1249})) == (700, 1200, 1249)
    values, notes = routes._client_stream_values(
        {"streaming_hard_break_silence_ms": 1248}, _ident())
    assert values == {HB: 2200} and notes == ["hard break 1248→2200 ms (above outer 1200)"]
    # ...and a client outer within a frame of a locked hard break is dropped.
    values, _ = routes._client_stream_values(
        {"streaming_vad_outer_silence_ms": 1200}, _ident(locked=[HB], **{HB: 1240}))
    assert values == {}
