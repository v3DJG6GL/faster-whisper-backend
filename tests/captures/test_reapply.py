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

The conftest autouse fixture resets _worker/_state between tests through
captures_reapply._reset_for_tests, which restores the canonical idle state
(the shape start() expects); _IDLE below pins that shape.
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


def test_start_while_running_queues_one_more_pass(fake_thread):
    """quick-config auto-starts this job on every save and has no manual
    re-apply button, so a save landing mid-run must not be dropped."""
    captures_reapply.start()
    assert captures_reapply._rerun_requested is False
    captures_reapply.start()
    assert captures_reapply._rerun_requested is True
    assert len(fake_thread.created) == 1


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


def test_run_scopes_a_translate_capture_by_its_english_text(
        captures_store_db, monkeypatch):
    """A task=translate capture stores the SPOKEN language ("de") but English
    text; the live run scoped its rules by "en", so the reapply must too —
    a de-scoped rule must not rewrite final or text_for_training. The
    transcribe twin with the same language still gets the de scope."""
    from faster_whisper_backend.settings import config as cfg

    cs = captures_store_db
    monkeypatch.setattr(cfg, "CAPTURES_PIPELINE_RULES_EXCLUDE", ["some-rule"],
                        raising=False)
    monkeypatch.setattr(effective_config, "build_ident", lambda who, m: {})
    langs: list = []

    def _postprocess_text(text, **kw):
        langs.append(kw.get("language"))
        # Stand-in for a de-only rule (de-dictation-map lowercasing).
        return text.lower() if kw.get("language") == "de" else text

    monkeypatch.setattr(pl_engine, "_postprocess_text", _postprocess_text)
    conn = cs._require_conn()
    raw = "It is over. Was it good?"
    _insert(conn, "reapplytr001", language="de", raw=raw, final=raw)
    conn.execute("UPDATE captures SET task = 'translate', text_for_training = ?"
                 " WHERE id = 'reapplytr001'", (raw,))
    _insert(conn, "reapplytx001", language="de", raw=raw, final=raw)
    conn.execute("UPDATE captures SET task = 'transcribe' WHERE id = 'reapplytx001'")

    captures_reapply._run()

    row = cs.get_capture("reapplytr001")
    assert row["final"] == raw and row["text_for_training"] == raw
    assert cs.get_capture("reapplytx001")["final"] == raw.lower()
    assert sorted(langs) == ["de", "de", "en", "en"]


def test_run_skips_a_row_whose_owner_identity_fails_to_resolve(
        captures_store_db, fake_pipeline, monkeypatch, caplog):
    """The owner-identity resolve sat outside the per-row try, so one row
    whose build_ident raised ended the whole run with status "error" and the
    remaining rows were never processed."""
    from faster_whisper_backend.settings import config as cfg

    cs = captures_store_db
    monkeypatch.setattr(cfg, "CAPTURES_PIPELINE_RULES_EXCLUDE", None,
                        raising=False)
    conn = cs._require_conn()
    _insert(conn, "reapplyid001", language="de")
    _insert(conn, "reapplyid002", language="de")
    conn.execute("UPDATE captures SET user_id = 'bob' WHERE id = 'reapplyid001'")

    def build_ident(who, model_id):
        if who.get("user_id") == "bob":
            raise RuntimeError("resolve failed")
        return {"who": who, "model": model_id}

    monkeypatch.setattr(effective_config, "build_ident", build_ident)

    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        captures_reapply._run()

    st = captures_reapply.status()
    assert st["status"] == "done" and st["error"] is None
    assert st["processed"] == 2 and st["captures_updated"] == 1
    assert "reapplyi skipped" in caplog.text
    assert cs.get_capture("reapplyid001")["final"] == "hello"
    assert cs.get_capture("reapplyid002")["final"] == "HELLO [final]"


def test_run_reapplies_rules_saved_while_a_pass_was_running(
        captures_store_db, fake_pipeline, monkeypatch, fake_thread):
    """A start() during a pass used to return the running state only: the
    pass kept its ident snapshot of the OLD rules and finished "done", so the
    newer save was never applied to any capture."""
    from faster_whisper_backend.settings import config as cfg

    cs = captures_store_db
    monkeypatch.setattr(cfg, "CAPTURES_PIPELINE_RULES_EXCLUDE", None,
                        raising=False)
    conn = cs._require_conn()
    _insert(conn, "reapplyrr001", language="de")
    captures_reapply.start()
    rules = {"v": "A"}

    def _postprocess_text(text, **kw):
        out = f"{text} [{rules['v']}]"
        if rules["v"] == "A":
            # An admin saves rule set B while pass A is mid-walk.
            rules["v"] = "B"
            captures_reapply.start()
        return out

    monkeypatch.setattr(pl_engine, "_postprocess_text", _postprocess_text)
    captures_reapply._run()

    st = captures_reapply.status()
    assert st["status"] == "done" and st["error"] is None
    assert st["processed"] == 1 and st["captures_updated"] == 1
    assert captures_reapply._rerun_requested is False
    assert cs.get_capture("reapplyrr001")["final"] == "hello [B]"
    assert len(fake_thread.created) == 1


def test_rerun_keeps_the_job_totals_of_the_earlier_pass(
        captures_store_db, fake_pipeline, monkeypatch, fake_thread):
    """A queued rerun walks every row again; a row pass 1 rewrote is current
    by then, so resetting the totals per pass reported "0 updated" for a job
    that rewrote it."""
    from faster_whisper_backend.settings import config as cfg

    cs = captures_store_db
    monkeypatch.setattr(cfg, "CAPTURES_PIPELINE_RULES_EXCLUDE", None,
                        raising=False)
    _insert(cs._require_conn(), "reapplyrt001", language="de")
    captures_reapply.start()
    calls = []

    def _postprocess_text(text, **kw):
        calls.append(text)
        if len(calls) == 1:
            # A second save (same rules) lands while pass 1 is mid-walk.
            captures_reapply.start()
        return f"{text} [A]"

    monkeypatch.setattr(pl_engine, "_postprocess_text", _postprocess_text)
    captures_reapply._run()

    st = captures_reapply.status()
    assert len(calls) == 2                      # both passes walked the row
    assert st["status"] == "done" and st["error"] is None
    assert st["processed"] == 1 and st["captures_updated"] == 1
    assert cs.get_capture("reapplyrt001")["final"] == "hello [A]"
