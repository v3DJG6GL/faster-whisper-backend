"""register_plan's early returns and failure fallback never echo another
user's plan id.

A job binds whatever plan_id register_plan hands back and drives that plan's
stage advances, so the paths that build no Plan (master switch off, every
entry statically refused, a registration that raised) must re-key a
colliding supplied id just like the Plan-building path does.
"""

from faster_whisper_backend.runtime import model_sizes
from faster_whisper_backend.runtime import preload
from tests.runtime.test_preload_registry import _enable, _fits

_ALICE_PID = "ab" * 8


def _alice_plan(monkeypatch):
    _enable(monkeypatch)
    _fits(monkeypatch, (None, "size_unknown"))
    r = preload.register_plan("alice", [("diarization", "p/a")],
                              plan_id=_ALICE_PID)
    assert r["plan_id"] == _ALICE_PID
    return preload._plans[_ALICE_PID].expires_mono


def test_all_refused_entries_do_not_echo_another_users_plan_id(monkeypatch):
    expires = _alice_plan(monkeypatch)
    _enable(monkeypatch, TRANSLATION_ENABLED=False)
    r = preload.register_plan("mallory", [("translation", "o/r:Q4")],
                              plan_id=_ALICE_PID, trigger="job")
    assert r["models"][0]["reason"] == "stage_disabled"
    assert r["expires_in_s"] == 0
    assert r["plan_id"] != _ALICE_PID
    assert preload._plans[_ALICE_PID].user_id == "alice"
    assert preload._plans[_ALICE_PID].expires_mono == expires


def test_master_switch_off_does_not_echo_another_users_plan_id(monkeypatch):
    _alice_plan(monkeypatch)
    _enable(monkeypatch, MODEL_PRELOAD_ENABLED=False)
    r = preload.register_plan("mallory", [("diarization", "p/m")],
                              plan_id=_ALICE_PID, trigger="job")
    assert r["models"][0]["reason"] == "disabled"
    assert r["plan_id"] != _ALICE_PID


def test_early_return_keeps_the_callers_own_supplied_id(monkeypatch):
    _alice_plan(monkeypatch)
    _enable(monkeypatch, TRANSLATION_ENABLED=False)
    r = preload.register_plan("alice", [("translation", "o/r:Q4")],
                              plan_id=_ALICE_PID, trigger="job")
    assert r["plan_id"] == _ALICE_PID


def test_failure_fallback_does_not_echo_another_users_plan_id(monkeypatch):
    _alice_plan(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("ledger on fire")
    monkeypatch.setattr(model_sizes, "fits", _boom)
    r = preload.register_plan("mallory", [("diarization", "p/m")],
                              plan_id=_ALICE_PID, trigger="job")
    assert r["models"][0]["state"] == "deferred"
    assert r["plan_id"] == ""
    assert preload._plans[_ALICE_PID].user_id == "alice"


def test_failure_fallback_keeps_the_callers_own_supplied_id(monkeypatch):
    _enable(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("ledger on fire")
    monkeypatch.setattr(model_sizes, "fits", _boom)
    r = preload.register_plan("bob", [("diarization", "p/b")],
                              plan_id="f" * 8)
    assert r["plan_id"] == "f" * 8
