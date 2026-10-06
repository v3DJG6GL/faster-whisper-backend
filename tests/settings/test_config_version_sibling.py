"""settings.version._bump_if_sibling_committed: the cross-worker PRAGMA probe is
throttled so config_version() — called synchronously on the event loop once
per partial-decode interval per live streaming session — cannot queue behind
api_keys_store._lock more than ~4×/s."""

import time
import types

from faster_whisper_backend.auth import api_keys_store
from faster_whisper_backend.auth import sessions_store
from faster_whisper_backend.settings import version as settings_version


def _install(monkeypatch, versions):
    """Stub api_keys_store.data_version with a call counter and a fresh
    probe state; return the (calls, clock) mutable cells."""
    calls = {"n": 0}
    clock = {"t": 1000.0}
    monkeypatch.setattr(settings_version, "_KEYS_DATA_VERSION", -1)
    monkeypatch.setattr(settings_version, "_KEYS_LAST_PROBE", 0.0)
    # Patch the version module's OWN clock seam, not the stdlib module attribute:
    # settings_version.time IS the stdlib `time`, so patching its .monotonic would freeze
    # the clock process-wide (asyncio, other stores' throttles).
    monkeypatch.setattr(settings_version, "time", types.SimpleNamespace(
        monotonic=lambda: clock["t"], time=time.time, sleep=time.sleep))

    def _dv():
        calls["n"] += 1
        return versions[min(calls["n"] - 1, len(versions) - 1)]
    monkeypatch.setattr(api_keys_store, "data_version", _dv)
    # The sessions half of the probe is held still here (see
    # tests/auth/test_session_auth.py for a sibling logout).
    monkeypatch.setattr(settings_version, "_SESSIONS_REV_GEN", -1)
    monkeypatch.setattr(sessions_store, "revocation_generation", lambda: 0)
    return calls, clock


def test_first_sample_adopted_without_bump(monkeypatch):
    calls, _ = _install(monkeypatch, [5])
    v0 = settings_version._CONFIG_VERSION
    assert settings_version.config_version() == v0
    assert calls["n"] == 1
    assert settings_version._KEYS_DATA_VERSION == 5


def test_repeated_calls_inside_window_do_not_reprobe(monkeypatch):
    calls, clock = _install(monkeypatch, [5, 6])
    settings_version.config_version()                      # adopts 5
    clock["t"] += settings_version._KEYS_PROBE_MIN_INTERVAL_S / 2
    v = settings_version.config_version()
    settings_version.config_version()
    assert calls["n"] == 1                   # throttled — no PRAGMA
    assert settings_version.config_version() == v          # and therefore no bump


def test_changed_sibling_version_bumps_after_window(monkeypatch):
    calls, clock = _install(monkeypatch, [5, 6])
    v0 = settings_version.config_version()                 # adopts 5
    clock["t"] += settings_version._KEYS_PROBE_MIN_INTERVAL_S
    v1 = settings_version.config_version()                 # re-probes, sees 6
    assert calls["n"] == 2
    assert v1 == v0 + 1
    assert settings_version._KEYS_DATA_VERSION == 6


def test_unready_store_is_never_throttled(monkeypatch):
    # data_version() returns -1 before init_db(); the throttle only kicks in
    # once a real sample has been adopted, so the store opening is not missed.
    calls, _ = _install(monkeypatch, [-1, -1, 7])
    settings_version.config_version()
    settings_version.config_version()
    settings_version.config_version()
    assert calls["n"] == 3
    assert settings_version._KEYS_DATA_VERSION == 7


def test_sibling_session_revocation_bumps_after_window(monkeypatch):
    """A logout served by a sibling worker moves sessions_store's revocation
    counter, never api_keys.db's data_version — the probe watches both."""
    calls, clock = _install(monkeypatch, [5])
    gens = iter([3, 3, 4])
    monkeypatch.setattr(sessions_store, "revocation_generation",
                        lambda: next(gens))
    v0 = settings_version.config_version()                 # adopts 5 and 3
    clock["t"] += settings_version._KEYS_PROBE_MIN_INTERVAL_S
    assert settings_version.config_version() == v0         # 3 again
    clock["t"] += settings_version._KEYS_PROBE_MIN_INTERVAL_S
    assert settings_version.config_version() == v0 + 1     # 4: sibling logout
