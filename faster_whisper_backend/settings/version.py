"""
Config version counter: lets long-lived consumers notice persisted config edits.
"""

from __future__ import annotations

import time


# ---------------------------------------------------------------------------
# Config version — a monotonic counter bumped on every persisted change to
# settings, override profiles, or per-user/per-key bindings. Long-lived
# consumers that cache a resolved view (notably a streaming connection's
# per-identity ``ident``, resolved ONCE at handshake) poll this to know when to
# re-resolve, so admin edits take effect without forcing a reconnect. The
# counter itself is per-process; restart resets to 0, which is fine (a restart
# re-resolves too).
# ---------------------------------------------------------------------------
_CONFIG_VERSION = 0

# Last `PRAGMA data_version` observed on api_keys_store's connection. main.py
# can run uvicorn with SERVER_WORKERS > 1 and each worker holds its own
# _CONFIG_VERSION, so a binding write (revoke_user / revoke_key /
# set_key_config / set_user_permissions / rename_profile_refs) bumps the
# counter in the worker that served the request and leaves a sibling worker's
# live idents resolving the pre-edit binding for the life of the connection.
# Bindings live in the SHARED api_keys.db, so the sibling re-resolves them
# correctly once it knows to; the pragma is what tells it. -1 = not sampled
# yet (init_db() runs long after this module imports).
#
# This is the DB-backed half only. config.local.json edits are applied to the
# running `config` module by the worker that served the save and a sibling has
# no reload path for them, so signalling those here would only make it
# re-resolve values that are themselves stale.
_KEYS_DATA_VERSION: int = -1
# config_version() runs synchronously on the event loop once per partial-decode
# interval per live streaming session, and the probe is a PRAGMA under
# api_keys_store._lock — the lock every store read/write holds. Rate-limit it
# so the loop cannot queue behind an unrelated store write more than ~4×/s;
# a sibling worker's binding change is still seen within this window, far
# below the utterance cadence that consumes the counter.
_KEYS_PROBE_MIN_INTERVAL_S = 0.25
_KEYS_LAST_PROBE: float = 0.0
# Last sessions_store.revocation_generation() seen — the same probe for a
# logout served by a sibling worker. Sessions live in their own DB file, so
# api_keys.db's data_version never moves for one. -1 = not sampled yet.
_SESSIONS_REV_GEN: int = -1


def bump_config_version() -> None:
    """Increment the global config version. Call after any persisted change to
    settings / override profiles / per-user / per-key bindings."""
    global _CONFIG_VERSION
    _CONFIG_VERSION += 1


def _bump_if_sibling_committed() -> None:
    """Bump the counter when another PROCESS committed to api_keys.db since the
    last check — the cross-worker stand-in for the bump_config_version() call
    every binding writer already makes in its own process. Also bumps when a
    sibling revoked a session (sessions_store.revocation_generation), which
    is the cross-worker half of revoke_session's own bump. Mirrors
    api_keys_store._refresh_if_sibling_committed(): `PRAGMA data_version` only
    moves for commits made by a different connection, so a single-worker server
    never bumps here. The first sample is adopted silently — the store opening
    is not a sibling write. Once a sample is adopted the PRAGMA runs at most
    every _KEYS_PROBE_MIN_INTERVAL_S — see the note on that constant."""
    global _KEYS_DATA_VERSION, _KEYS_LAST_PROBE, _SESSIONS_REV_GEN
    now = time.monotonic()
    if (_KEYS_DATA_VERSION >= 0
            and now - _KEYS_LAST_PROBE < _KEYS_PROBE_MIN_INTERVAL_S):
        return
    _KEYS_LAST_PROBE = now
    from faster_whisper_backend.auth import api_keys_store   # lazy: api_keys_store imports this module
    from faster_whisper_backend.auth import sessions_store   # lazy: same cycle
    changed = False
    v = api_keys_store.data_version()
    if v != _KEYS_DATA_VERSION:
        changed = _KEYS_DATA_VERSION >= 0
        _KEYS_DATA_VERSION = v
    # A sibling's logout. -1 (store not open / read failed) is never adopted,
    # so a transient error cannot pass for a revocation on the next read.
    g = sessions_store.revocation_generation()
    if g >= 0 and g != _SESSIONS_REV_GEN:
        changed = changed or _SESSIONS_REV_GEN >= 0
        _SESSIONS_REV_GEN = g
    if changed:
        bump_config_version()


def config_version() -> int:
    """Current config version — see :func:`bump_config_version`."""
    _bump_if_sibling_committed()
    return _CONFIG_VERSION
