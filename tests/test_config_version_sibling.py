"""config_store._bump_if_sibling_committed: the cross-worker PRAGMA probe is
throttled so config_version() — called synchronously on the event loop once
per partial-decode interval per live streaming session — cannot queue behind
api_keys_store._lock more than ~4×/s."""

from faster_whisper_backend.auth import api_keys_store
from faster_whisper_backend import config_store as cs


def _install(monkeypatch, versions):
    """Stub api_keys_store.data_version with a call counter and a fresh
    probe state; return the (calls, clock) mutable cells."""
    calls = {"n": 0}
    clock = {"t": 1000.0}
    monkeypatch.setattr(cs, "_KEYS_DATA_VERSION", -1)
    monkeypatch.setattr(cs, "_KEYS_LAST_PROBE", 0.0)
    monkeypatch.setattr(cs.time, "monotonic", lambda: clock["t"])

    def _dv():
        calls["n"] += 1
        return versions[min(calls["n"] - 1, len(versions) - 1)]
    monkeypatch.setattr(api_keys_store, "data_version", _dv)
    return calls, clock


def test_first_sample_adopted_without_bump(monkeypatch):
    calls, _ = _install(monkeypatch, [5])
    v0 = cs._CONFIG_VERSION
    assert cs.config_version() == v0
    assert calls["n"] == 1
    assert cs._KEYS_DATA_VERSION == 5


def test_repeated_calls_inside_window_do_not_reprobe(monkeypatch):
    calls, clock = _install(monkeypatch, [5, 6])
    cs.config_version()                      # adopts 5
    clock["t"] += cs._KEYS_PROBE_MIN_INTERVAL_S / 2
    v = cs.config_version()
    cs.config_version()
    assert calls["n"] == 1                   # throttled — no PRAGMA
    assert cs.config_version() == v          # and therefore no bump


def test_changed_sibling_version_bumps_after_window(monkeypatch):
    calls, clock = _install(monkeypatch, [5, 6])
    v0 = cs.config_version()                 # adopts 5
    clock["t"] += cs._KEYS_PROBE_MIN_INTERVAL_S
    v1 = cs.config_version()                 # re-probes, sees 6
    assert calls["n"] == 2
    assert v1 == v0 + 1
    assert cs._KEYS_DATA_VERSION == 6


def test_unready_store_is_never_throttled(monkeypatch):
    # data_version() returns -1 before init_db(); the throttle only kicks in
    # once a real sample has been adopted, so the store opening is not missed.
    calls, _ = _install(monkeypatch, [-1, -1, 7])
    cs.config_version()
    cs.config_version()
    cs.config_version()
    assert calls["n"] == 3
    assert cs._KEYS_DATA_VERSION == 7
