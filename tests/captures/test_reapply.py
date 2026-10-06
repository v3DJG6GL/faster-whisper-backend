"""Tests for captures_reapply.start()/status() worker management.

_run() runs the whole rules pipeline over the entire captures DB, so we
never run the real worker: threading.Thread is monkeypatched to a stub
that records the target but never executes it. We assert start() is
idempotent (a second call while "running" returns state without spawning a
2nd worker) and that status() returns a dict copy of the live state.

The _run() tests at the bottom DO execute the worker body, synchronously and
against a temp captures DB, with the engine's `_postprocess_text` and
`build_ident` replaced by fakes — the start() tests alone left the whole
row loop uncovered.

The conftest autouse fixture resets _worker/_state between tests, but it
seeds _state with a different key set than the module's real schema, so each
test first restores the canonical idle state (the shape start() expects).
"""

import logging
import types

import pytest

from faster_whisper_backend.captures import reapply as captures_reapply
from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.settings import effective_config

# The module's canonical idle state (start() reads _state["status"]).
_IDLE = {
    "status": "idle",
    "started_ts": None,
    "finished_ts": None,
    "total": 0,
    "processed": 0,
    "captures_updated": 0,
    "groups_updated": 0,
    "error": None,
}


@pytest.fixture(autouse=True)
def _canonical_idle():
    """conftest reset uses a foreign key set; restore the real idle schema."""
    captures_reapply._state.clear()
    captures_reapply._state.update(_IDLE)
    captures_reapply._worker = None
    yield


class _FakeThread:
    """Records the target/name but never runs it (no real worker)."""
    created: list["_FakeThread"] = []

    def __init__(self, target=None, daemon=None, name=None, **kw):
        self.target = target
        self.daemon = daemon
        self.name = name
        self.started = False
        _FakeThread.created.append(self)

    def start(self):
        self.started = True


@pytest.fixture
def fake_thread(monkeypatch):
    _FakeThread.created = []
    monkeypatch.setattr(captures_reapply.threading, "Thread", _FakeThread)
    return _FakeThread


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------

def test_status_returns_dict_copy():
    s = captures_reapply.status()
    assert s == _IDLE
    # Mutating the returned dict must not affect module state.
    s["status"] = "tampered"
    assert captures_reapply._state["status"] == "idle"


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------

def test_start_spawns_one_worker(fake_thread):
    state = captures_reapply.start()
    assert state["status"] == "running"
    assert state["started_ts"] is not None
    assert len(fake_thread.created) == 1
    t = fake_thread.created[0]
    assert t.target is captures_reapply._run
    assert t.daemon is True
    assert t.name == "reapply-rules"
    assert t.started is True
    # The module tracks the live worker.
    assert captures_reapply._worker is t


def test_start_resets_counters(fake_thread):
    # Pre-dirty the state to prove start() resets it.
    captures_reapply._state.update({
        "status": "done", "processed": 99, "captures_updated": 5,
        "groups_updated": 3, "error": "old",
    })
    state = captures_reapply.start()
    assert state["status"] == "running"
    assert state["processed"] == 0
    assert state["captures_updated"] == 0
    assert state["groups_updated"] == 0
    assert state["error"] is None
    assert state["finished_ts"] is None


def test_start_is_idempotent_while_running(fake_thread):
    first = captures_reapply.start()
    assert first["status"] == "running"
    assert len(fake_thread.created) == 1
    worker1 = captures_reapply._worker

    # Second call while running: returns current state, no 2nd worker spawned.
    second = captures_reapply.start()
    assert second["status"] == "running"
    assert len(fake_thread.created) == 1  # still just one
    assert captures_reapply._worker is worker1


def test_start_after_done_spawns_again(fake_thread):
    captures_reapply.start()
    assert len(fake_thread.created) == 1
    # Simulate the worker finishing.
    captures_reapply._state["status"] = "done"
    captures_reapply.start()
    assert len(fake_thread.created) == 2


def test_status_reflects_running_after_start(fake_thread):
    captures_reapply.start()
    assert captures_reapply.status()["status"] == "running"


# ---------------------------------------------------------------------------
# _run() — the worker body, executed for real
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_pipeline(monkeypatch):
    """Stand-in for the rules engine: `pl_engine._postprocess_text` records its
    kwargs and tags its output. The owner-identity resolve (effective_config.
    build_ident) is stubbed too, so no key store is needed."""
    fake = types.SimpleNamespace(calls=[])

    def build_ident(who, model_id):
        return {"who": who, "model": model_id}

    def _postprocess_text(text, **kw):
        fake.calls.append(kw)
        suffix = " [training]" if kw.get("extra_excludes") else " [final]"
        return text.upper() + suffix

    monkeypatch.setattr(pl_engine, "_postprocess_text", _postprocess_text)
    monkeypatch.setattr(effective_config, "build_ident", build_ident)
    return fake


def _insert(conn, cid, *, language, raw="hello", final="hello"):
    conn.execute(
        "INSERT INTO captures (id, created_ts, model, language, audio_relpath,"
        " audio_format, raw_text, final_text, words, segments, corrections,"
        " status, user_id) VALUES (?,1.0,'m',?,'x.wav','wav',?,?,'[]','[]',"
        "'[]','new','alice')", (cid, language, raw, final))


def test_run_passes_the_row_language_and_updates_the_capture(
        captures_store_db, fake_pipeline, monkeypatch, caplog):
    """The rows are sqlite3.Row, which has no .get(): `r.get("language")`
    raised inside the per-row try, so EVERY capture was logged as skipped
    and the job still finished "done" with captures_updated == 0."""
    from faster_whisper_backend.settings import config as cfg

    cs = captures_store_db
    monkeypatch.setattr(cfg, "CAPTURES_PIPELINE_RULES_EXCLUDE", None,
                        raising=False)
    conn = cs._require_conn()
    _insert(conn, "reapply00001", language="de")
    _insert(conn, "reapply00002", language=None)

    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        captures_reapply._run()

    st = captures_reapply.status()
    assert st["status"] == "done" and st["error"] is None
    assert st["total"] == 2 and st["processed"] == 2
    assert st["captures_updated"] == 2
    assert "skipped" not in caplog.text
    assert sorted(str(c["language"]) for c in fake_pipeline.calls) == ["None", "de"]
    row = cs.get_capture("reapply00001")
    assert row["final"] == "HELLO [final]"
    assert row["text_for_training"] == "HELLO [final]"


def test_run_training_pass_gets_the_language_too(
        captures_store_db, fake_pipeline, monkeypatch, caplog):
    """Second call site: the captures-excludes training-form pass."""
    from faster_whisper_backend.settings import config as cfg

    cs = captures_store_db
    monkeypatch.setattr(cfg, "CAPTURES_PIPELINE_RULES_EXCLUDE", ["some-rule"],
                        raising=False)
    _insert(cs._require_conn(), "reapply00003", language="fr")

    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        captures_reapply._run()

    assert "skipped" not in caplog.text
    assert captures_reapply.status()["captures_updated"] == 1
    assert [(c["language"], bool(c.get("extra_excludes")))
            for c in fake_pipeline.calls] == [("fr", False), ("fr", True)]
    row = cs.get_capture("reapply00003")
    assert row["final"] == "HELLO [final]"
    assert row["text_for_training"] == "HELLO [training]"
