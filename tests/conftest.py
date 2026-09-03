"""Shared fixtures for the test suite.

The application keeps process-global state in several modules (SQLite
connections, in-memory key indexes, ring-buffer metrics, proposer caches,
the reapply worker). Tests must not leak that state between cases, so this
file provides:

  * an autouse fixture that resets the cross-cutting singletons,
  * per-store fixtures that (re)initialise each store on a fresh temp DB,
  * deterministic audio helpers (fake/real VAD, WAV writer),
  * a timezone setter (POSIX-only; skips on Windows),
  * a TestClient factory that drives the real FastAPI app with the model
    load neutralised, temp DBs, and a loopback client so the host gate
    allows admin/stats/captures/etc. routes.

Everything here only depends on the base requirements (which CI installs in
full), so all fixtures are usable on every matrix leg.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import types
import wave
from typing import Any, Callable

import numpy as np
import pytest

# The path defaults are container-first (/data, /data/db — see config.py
# _DATA_DIR). A dev box usually has no writable /data, and a few test modules
# import `main` at COLLECTION time (before any fixture monkeypatch runs), so
# root every default into a throwaway dir for the whole test session. The
# app_module fixture still pins each store to its own tmp_path per test.
import atexit
import shutil
import tempfile

if not os.environ.get("WHISPER_DATA_DIR"):
    _TEST_DATA_DIR = tempfile.mkdtemp(prefix="whisper-test-data-")
    os.environ["WHISPER_DATA_DIR"] = _TEST_DATA_DIR
    # ignore_errors: a still-open SQLite handle keeps the file locked on
    # Windows; a leftover dir there must not fail the run.
    atexit.register(shutil.rmtree, _TEST_DATA_DIR, True)

RATE = 16000


# ---------------------------------------------------------------------------
# Cross-cutting singleton reset
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset module-global mutable state that is NOT tied to a store
    connection, so unrelated tests don't observe each other's writes.

    Store connections themselves are reset by re-`init`/`init_db` in the
    per-store fixtures (each reassigns the module `_conn` global)."""
    import metrics
    import captures_merge_proposer as proposer
    import captures_reapply
    import api_keys_store

    # metrics ring buffers / counters
    metrics.req_count.clear()
    metrics.err_count.clear()
    metrics._latency.clear()
    metrics._errors_ts.clear()
    metrics.model_loads.clear()
    metrics.in_flight_transcriptions = 0

    # proposer caches
    proposer._CACHE.clear()
    proposer._TRIM_DUR_CACHE.clear()

    # reapply worker state — reset to the module's canonical idle shape.
    captures_reapply._worker = None
    captures_reapply._state = {
        "status": "idle",
        "started_ts": None,
        "finished_ts": None,
        "total": 0,
        "processed": 0,
        "captures_updated": 0,
        "groups_updated": 0,
        "error": None,
    }

    # api-key debounce cache (the index/lockdown are rebuilt by init_db)
    api_keys_store._LAST_USED_CACHE.clear()

    # central running-jobs registry
    try:
        import jobs
        jobs._reset_for_tests()
    except Exception:
        pass

    # held dictation receipts — a leftover would be swept into a later test's
    # log output, and would count against the pending cap.
    try:
        import receipt_hold
        receipt_hold._reset_for_tests()
    except Exception:
        pass

    # preload plan registry + warm leases. The warm predicate is cleared
    # separately from _reset_for_tests: one left installed in system_stats
    # would keep a later eviction test's model pinned by a plan this test
    # owned, and the failure would look like a broken evictor.
    try:
        import preload
        preload._reset_for_tests()
    except Exception:
        pass
    try:
        import system_stats as _ss
        _ss.set_warm_predicate(None)
    except Exception:
        pass

    # session-store caches (the index is rebuilt by init_db)
    try:
        import sessions_store
        sessions_store._reset_for_tests()
    except Exception:
        pass

    # shared per-identity limiters (rate_limit._ALL): one hook clears every
    # FixedWindow/InFlight bucket, so a limit tripped in one case cannot 429
    # or refuse a slot in the next. This covers reports' submit limiter too:
    # reports_routes._rate is a rate_limit.FixedWindow, which registers
    # itself in rate_limit._ALL on construction.
    try:
        import rate_limit
        rate_limit.reset_all()
    except Exception:
        pass

    # streaming session-id registry: a socket torn down without the route's
    # finally would strand an id and make the cap tests' exact-count pins
    # order-dependent.
    try:
        import streaming_routes
        streaming_routes._active_sessions.clear()
    except Exception:
        pass

    # translation model LRU (module-global cache of loaded GGUF models —
    # tests only ever put stubs in it, but they must not leak across tests).
    try:
        import translation
        translation._models.clear()
        translation._last_used.clear()
        # A leaked lease makes every later eviction test see a refusal.
        translation._active.clear()
    except Exception:
        pass

    # Stage-model singletons (pyannote / UVR) + their job leases. Tests only
    # ever put stubs in them, but a leaked pipeline/lease would change what a
    # later eviction or cache-hit test observes.
    try:
        import diarization
        diarization._pipeline = None
        diarization._pipeline_key = None
        diarization._leases.clear()
        diarization._orphans.clear()
    except Exception:
        pass
    try:
        import bgm_separation
        bgm_separation._separator = None
        bgm_separation._separator_key = None
        bgm_separation._leases.clear()
        bgm_separation._orphans.clear()
    except Exception:
        pass

    # Loaded-model registry behind /stats: stage tests register stubs in it.
    try:
        import system_stats
        system_stats._loaded_models.clear()
    except Exception:
        pass

    # Persisted measured-size ledger: drop the in-memory read cache so a test
    # that repoints PATH doesn't inherit the previous test's file contents.
    try:
        import model_sizes
        model_sizes._reset_for_tests()
    except Exception:
        pass

    # /v1/text/translations rate limiter (module-global fixed-window counter):
    # clear so a rate-limit test can't 429 a later translate test. Guarded on
    # sys.modules — don't force the heavy main import on pure-unit tests.
    _main = sys.modules.get("main")
    if _main is not None:
        try:
            _main._text_translate_rate.clear()
        except Exception:
            pass
        try:
            _main._JOB_BY_PID.clear()
        except Exception:
            pass
        try:
            _main._PLAN_BY_PID.clear()
        except Exception:
            pass
        # Whisper job leases: a leaked lease would make a later eviction test
        # see a refusal instead of the drop it asserts.
        try:
            _main._model_leases.clear()
        except Exception:
            pass

    yield


# ---------------------------------------------------------------------------
# Per-store fixtures (fresh temp DB per test)
# ---------------------------------------------------------------------------

@pytest.fixture
def captures_store_db(tmp_path):
    """captures_store initialised on a temp DB + temp audio dir."""
    import captures_store
    import captures_merge_proposer as proposer

    db = str(tmp_path / "captures.sqlite3")
    audio = str(tmp_path / "captures_audio")
    captures_store.init_db(db, audio)
    proposer._CACHE.clear()
    proposer._TRIM_DUR_CACHE.clear()
    yield captures_store
    try:
        captures_store._require_conn().close()
    except Exception:
        pass
    captures_store._conn = None
    captures_store._audio_dir = None


@pytest.fixture
def groups_store_db(captures_store_db):
    """capture_samples_store sharing the captures connection (as in main)."""
    import capture_samples_store
    capture_samples_store.init_db(
        captures_store_db._require_conn(), captures_store_db._require_audio_dir()
    )
    yield capture_samples_store
    capture_samples_store._conn = None
    capture_samples_store._groups_audio_dir = None


@pytest.fixture
def tx_store(tmp_path):
    import transcriptions_store
    transcriptions_store.init_db(str(tmp_path / "recent.sqlite3"))
    transcriptions_store._insert_counter = 0
    yield transcriptions_store
    try:
        transcriptions_store._require_conn().close()
    except Exception:
        pass
    transcriptions_store._conn = None


@pytest.fixture
def usage_store_db(tmp_path):
    import usage_store
    usage_store.init_db(str(tmp_path / "usage.sqlite3"))
    yield usage_store
    try:
        usage_store._require_conn().close()
    except Exception:
        pass
    usage_store._conn = None


@pytest.fixture
def reports_store_db(tmp_path):
    import reports_store
    reports_store.init_db(str(tmp_path / "reports.sqlite3"))
    yield reports_store
    try:
        reports_store._require_conn().close()
    except Exception:
        pass
    reports_store._conn = None


@pytest.fixture
def client_settings_store_db(tmp_path):
    import client_settings_store
    client_settings_store.init_db(str(tmp_path / "client_settings.sqlite3"))
    yield client_settings_store
    try:
        client_settings_store._require_conn().close()
    except Exception:
        pass
    client_settings_store._conn = None
    client_settings_store._DB_READY = False


@pytest.fixture
def api_keys_db(tmp_path):
    import api_keys_store
    api_keys_store.init_db(str(tmp_path / "api_keys.sqlite3"))
    api_keys_store._LAST_USED_CACHE.clear()
    yield api_keys_store
    try:
        api_keys_store._require_conn().close()
    except Exception:
        pass
    api_keys_store._conn = None
    # Closing the connection un-readies the store; leaving _DB_READY set would
    # let a later test read this test's stale lockdown cache.
    api_keys_store._reset_for_tests()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _pcm_from_spec(spec):
    """Build int16 PCM from [(kind, ms), ...] where kind is 'sil'|'speech'."""
    parts = []
    for kind, ms in spec:
        n = int(ms * RATE / 1000)
        val = 0 if kind == "sil" else 4000
        parts.append(np.full(n, val, dtype=np.int16))
    a = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)
    return a.tobytes(), len(a)


@pytest.fixture
def pcm_spec():
    """Expose the PCM builder so tests can assemble silence/speech timelines."""
    return _pcm_from_spec


@pytest.fixture
def fake_vad(monkeypatch):
    """Install a deterministic faster_whisper.vad: every contiguous non-zero
    run is one speech segment. Mirrors test_group_trim._install_fake_vad."""
    def get_speech_timestamps(audio, opts, sampling_rate=RATE):
        nz = np.abs(audio) > 1e-6
        if not nz.any():
            return []
        d = np.diff(nz.astype(np.int8))
        starts = list(np.where(d == 1)[0] + 1)
        ends = list(np.where(d == -1)[0] + 1)
        if nz[0]:
            starts = [0] + starts
        if nz[-1]:
            ends = ends + [len(audio)]
        return [{"start": int(s), "end": int(e)} for s, e in zip(starts, ends)]

    class VadOptions:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    pkg = types.ModuleType("faster_whisper")
    vad = types.ModuleType("faster_whisper.vad")
    vad.VadOptions = VadOptions
    vad.get_speech_timestamps = get_speech_timestamps
    pkg.vad = vad
    monkeypatch.setitem(sys.modules, "faster_whisper", pkg)
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    return vad


@pytest.fixture
def no_vad(monkeypatch):
    """Force the 'VAD unavailable -> identity' path."""
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    # Nulling the package alone is not enough once another test has imported
    # the submodule: `from faster_whisper.vad import ...` resolves straight
    # from the sys.modules cache. Null it too so the ImportError path fires
    # regardless of test order.
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", None)


@pytest.fixture
def wav_factory(tmp_path):
    """Return write(name_or_spec) -> path; writes a 1ch/16k/s16 WAV."""
    counter = {"n": 0}

    def _write(spec, pcm: bytes | None = None) -> str:
        if pcm is None:
            pcm, _ = _pcm_from_spec(spec)
        counter["n"] += 1
        path = str(tmp_path / f"w{counter['n']}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm)
        return path

    return _write


# ---------------------------------------------------------------------------
# Timezone control (usage_store day/week bucketing is server-local time)
# ---------------------------------------------------------------------------

@pytest.fixture
def set_tz():
    """Return set_tz(name) that pins the process timezone. POSIX-only:
    skips the test on platforms without time.tzset() (Windows)."""
    if not hasattr(time, "tzset"):
        pytest.skip("timezone control requires POSIX time.tzset()")
    original = os.environ.get("TZ")

    def _set(name: str) -> None:
        os.environ["TZ"] = name
        time.tzset()

    yield _set

    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    time.tzset()


# ---------------------------------------------------------------------------
# FastAPI app / TestClient harness (route tests)
# ---------------------------------------------------------------------------

class FakeSegment:
    def __init__(self, text, start=0.0, end=1.0, words=None):
        self.text = text
        self.start = start
        self.end = end
        self.avg_logprob = -0.1
        self.no_speech_prob = 0.01
        self.temperature = 0.0
        self.compression_ratio = 1.2
        self.words = words or []


class FakeWord:
    def __init__(self, word, start, end, probability=0.9):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class FakeInfo:
    def __init__(self, language="de", duration=1.0):
        self.language = language
        self.duration = duration
        self.language_probability = 0.99
        self.duration_after_vad = duration


class FakeModel:
    """Stand-in for faster_whisper.WhisperModel. Records the last transcribe
    kwargs (cfg_for forwarding asserts) and audio input (a tmp-file path, or
    the padded ndarray when LEADING_SILENCE_PAD_MS pre-decodes)."""

    def __init__(self, segments=None, info=None):
        self._segments = segments
        self._info = info or FakeInfo()
        self.last_kwargs: dict[str, Any] = {}
        self.last_audio: Any = None

    def transcribe(self, path, **kwargs):
        self.last_kwargs = kwargs
        self.last_audio = path
        segs = self._segments
        if segs is None:
            include_words = bool(kwargs.get("word_timestamps"))
            words = [FakeWord("hallo", 0.0, 0.5), FakeWord("welt", 0.5, 1.0)] \
                if include_words else []
            segs = [FakeSegment("hallo welt", 0.0, 1.0, words)]
        return iter(segs), self._info


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def app_module(tmp_path, monkeypatch, fake_model):
    """Import `main`, neutralise the model preload, and point every store at
    a temp DB. Yields the main module with a `_get_or_load_model` that returns
    the fake model. Importing main has only benign side effects (logging)."""
    # Point all stores at temp files BEFORE importing main / running lifespan.
    monkeypatch.setenv("WHISPER_API_KEYS_DB", str(tmp_path / "api_keys.sqlite3"))
    monkeypatch.setenv("WHISPER_SESSIONS_DB", str(tmp_path / "sessions.sqlite3"))
    monkeypatch.setenv("WHISPER_REPORTS_DB", str(tmp_path / "reports.sqlite3"))
    monkeypatch.setenv("WHISPER_RECENT_TRANSCRIPTIONS_DB", str(tmp_path / "recent.sqlite3"))
    monkeypatch.setenv("WHISPER_USAGE_DB", str(tmp_path / "usage.sqlite3"))
    monkeypatch.setenv("WHISPER_CAPTURES_DB", str(tmp_path / "captures.sqlite3"))
    monkeypatch.setenv("WHISPER_CAPTURES_DIR", str(tmp_path / "captures_audio"))
    monkeypatch.setenv("WHISPER_CLIENT_SETTINGS_DB", str(tmp_path / "client_settings.sqlite3"))
    monkeypatch.setenv("WHISPER_LOG_FILE", str(tmp_path / "whisper.log"))
    # The lifespan's url_media_store.startup_reset() rmtree's URL_MEDIA_DIR
    # unconditionally — without this it would wipe the REAL /data/url_media.
    monkeypatch.setenv("WHISPER_URL_MEDIA_DIR", str(tmp_path / "url_media"))

    import config as cfg
    # Re-apply env onto the already-imported config singleton.
    importlib.reload(cfg)
    monkeypatch.setattr(cfg, "PRELOAD_MODELS", [], raising=False)
    monkeypatch.setattr(cfg, "DEFAULT_MODEL", "", raising=False)

    # config_store.save_overrides / load_overrides persist to the REAL
    # <repo>/config.local.json by default — route tests that POST /settings/state
    # or /quick-config/state would mutate the working tree and leak override
    # state between tests. Repoint both at a per-test temp file. The path is a
    # default ARG (bound at def time), so we rewrite each function's defaults
    # in addition to the module-level constant.
    import config_store
    _tmp_overrides = str(tmp_path / "config.local.json")
    monkeypatch.setattr(config_store, "OVERRIDES_PATH", _tmp_overrides, raising=False)
    for _fn in (config_store.load_overrides, config_store.save_overrides):
        _defaults = list(_fn.__defaults__ or ())
        if _defaults:
            _defaults[-1] = _tmp_overrides
            monkeypatch.setattr(_fn, "__defaults__", tuple(_defaults), raising=False)

    # model_sizes has the same default-ARG trap: _read/_write bind path=PATH at
    # def time, so a route test that loads a model would otherwise write the
    # measured-size ledger into the REAL /data (or <repo>\data).
    import model_sizes
    _tmp_sizes = str(tmp_path / "model_sizes.json")
    monkeypatch.setattr(model_sizes, "PATH", _tmp_sizes, raising=False)
    for _fn in (model_sizes._read, model_sizes._write):
        _defaults = list(_fn.__defaults__ or ())
        if _defaults:
            _defaults[-1] = _tmp_sizes
            monkeypatch.setattr(_fn, "__defaults__", tuple(_defaults), raising=False)

    import main
    importlib.reload(main)

    async def _fake_loader(name: str, *, lease: bool = False):
        return fake_model
    monkeypatch.setattr(main, "_get_or_load_model", _fake_loader)
    # The lifespan's hard-restart TMPDIR sweep would prune the REAL system
    # tempdir (urldl-/sepsrc-/vocals- leftovers) on every TestClient startup
    # — neuter it here; test_routes_url exercises the real function against
    # a fake tempdir directly. (The URL retention dir is temp-rooted via
    # WHISPER_URL_MEDIA_DIR above for the same reason.)
    monkeypatch.setattr(main, "_reclaim_hard_restart_orphans", lambda: None)

    yield main

    # The lifespan opens five store connections on a temp DB; close them so a
    # GC'd-without-close() sqlite3.Connection doesn't emit ResourceWarning noise
    # (one per store × every route test). capture_samples_store shares the
    # captures connection, so just drop its reference.
    import api_keys_store, reports_store, transcriptions_store
    import usage_store, captures_store, capture_samples_store
    import sessions_store, client_settings_store
    for _mod in (api_keys_store, sessions_store, reports_store,
                 transcriptions_store, usage_store, captures_store,
                 client_settings_store):
        _c = getattr(_mod, "_conn", None)
        if _c is not None:
            try:
                _c.close()
            except Exception:
                pass
            _mod._conn = None
    capture_samples_store._conn = None
    # Same reset api_keys_db performs: leaving _DB_READY set over a closed
    # connection would let a later test read this test's stale lockdown
    # cache (is_locked_down() passes the _DB_READY gate, swallows the
    # RuntimeError from _require_conn(), and returns the unchanged cache).
    api_keys_store._reset_for_tests()
    client_settings_store._DB_READY = False
    sessions_store._reset_for_tests()


@pytest.fixture
def client(app_module):
    """TestClient bound to a loopback client so the host gate (which always
    allows 127.0.0.1/::1) admits the gated admin/stats/captures routes."""
    from starlette.testclient import TestClient
    with TestClient(app_module.app, client=("127.0.0.1", 12345)) as c:
        yield c


@pytest.fixture
def make_user_key(app_module):
    """Return create(username, is_admin=False, pages=None) -> (uid, raw_key).

    Creating the FIRST admin key flips the running app to LOCKED-DOWN mode
    (api_keys_store.is_locked_down() becomes True), so after the first admin
    key every request needs a valid bearer. `pages` (a {page: scope} dict)
    is applied via set_user_permissions for non-admin scope tests.

    Uses the SAME api_keys_store module instance the running app uses (the
    lifespan re-inits it onto the temp DB), so the lockdown index stays in
    sync with what the auth dependency reads."""
    import api_keys_store

    def _create(username, is_admin=False, pages=None):
        uid = api_keys_store.create_user(username, is_admin=is_admin)
        if pages:
            api_keys_store.set_user_permissions(uid, {"pages": pages})
        raw, _rec = api_keys_store.create_key(uid)
        return uid, raw

    return _create


def bearer(raw_key):
    """Authorization header dict for a raw API key."""
    return {"Authorization": f"Bearer {raw_key}"}
