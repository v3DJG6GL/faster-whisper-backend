"""Tests for recent_transcriptions_store: trace/timing UPSERT merge, pagination,
prune, step truncation, and the row-to-dict derivations."""

import sqlite3
import time

import pytest

from faster_whisper_backend.stats import recent_transcriptions_store as rts
from tests.conftest import _close_store


@pytest.fixture
def ts(tx_store):
    """tx_store under its short test-body name."""
    return tx_store


def _trace(ts, request_id, raw="x", final="y", **kw):
    ts.record_trace(request_id=request_id, model="m", raw=raw, final=final, **kw)


def _timing(ts, request_id, audio_s, processing_s, status="ok", words=1, **kw):
    ts.record_timing(request_id=request_id, model="m", audio_s=audio_s,
                     processing_s=processing_s, status=status, words=words, **kw)


# ---------------------------------------------------------------------------
# Recent-jobs columns (kind / stages / key_label) + migration
# ---------------------------------------------------------------------------

def test_kind_stages_key_label_roundtrip(ts):
    stages = [{"name": "translate", "secs": 1.25, "model": "org/m:Q4",
               "detail": "3 segs → en"}]
    ts.record_timing(request_id="t1", model="org/m:Q4", audio_s=None,
                     processing_s=1.25, status="ok", words=0,
                     kind="translate", stages=stages, key_label="ci-key")
    row = ts.list_recent(limit=10)[0]
    assert row["kind"] == "translate"
    assert row["key_label"] == "ci-key"
    assert row["stages"] == stages


def test_timing_without_kind_keeps_earlier_values(ts):
    _timing(ts, "k1", None, 1.0, words=0, kind="download",
            stages=[{"name": "download", "secs": 1}], key_label="lbl")
    # A later write without kind / stages / key_label must not blank them:
    # the ON CONFLICT clause COALESCEs those columns.
    _timing(ts, "k1", None, 2.0, words=0)
    row = ts.list_recent(limit=10)[0]
    assert row["kind"] == "download"
    assert row["key_label"] == "lbl"
    assert row["stages"] and row["processing_s"] == 2.0


def test_migration_adds_columns_to_old_db(tmp_path):
    """A DB created before the recent-jobs columns migrates on init_db and
    its old rows still render (kind None, stages [], key_label '')."""
    path = str(tmp_path / "old.sqlite3")
    conn = sqlite3.connect(path)
    conn.executescript("""
    CREATE TABLE recent_transcriptions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      request_id TEXT NOT NULL UNIQUE,
      created_ts REAL NOT NULL,
      user_id TEXT, username TEXT,
      model TEXT NOT NULL, language TEXT,
      status TEXT NOT NULL DEFAULT 'ok',
      audio_dur_s REAL, proc_dur_s REAL, words_count INTEGER,
      raw_text TEXT, final_text TEXT,
      steps_json TEXT, tokens_json TEXT, bigrams_json TEXT
    );
    INSERT INTO recent_transcriptions
      (request_id, created_ts, model, status, audio_dur_s, proc_dur_s, words_count)
      VALUES ('old1', 1000.0, 'm', 'ok', 2.0, 1.0, 5);
    """)
    conn.commit()
    conn.close()

    rts.init_db(path)
    try:
        cols = {r["name"] for r in rts._require_conn().execute(
            "PRAGMA table_info(recent_transcriptions)")}
        assert {"source", "kind", "stages", "key_label",
                "wait_s", "error_class", "error_stage"} <= cols
        # The legacy JSON columns were renamed in place, not duplicated.
        assert {"steps", "tokens", "bigrams"} <= cols
        assert not ({"steps_json", "tokens_json", "bigrams_json"} & cols)
        row = rts.list_recent(limit=10)[0]
        assert row["request_id"] == "old1"
        assert row["kind"] is None
        assert row["stages"] == []
        assert row["key_label"] == ""
        assert row["source"] == "file"
        assert row["wait_s"] is None and row["error_class"] is None
        assert row["error_stage"] is None
        # The measurement columns were renamed in place with their data:
        # list_recent's slow_rtf SQL reads the new names.
        assert {"audio_s", "processing_s", "words"} <= cols
        assert not ({"audio_dur_s", "proc_dur_s", "words_count"} & cols)
        assert (row["audio_s"], row["processing_s"], row["words"]) == (2.0, 1.0, 5)
        assert row["rtf"] == 2.0
        assert [r["request_id"] for r in rts.list_recent(limit=10, slow_rtf=0.4)] == ["old1"]
        assert rts.list_recent(limit=10, slow_rtf=0.6) == []
    finally:
        _close_store(rts)


# ---------------------------------------------------------------------------
# record_trace / record_timing UPSERT
# ---------------------------------------------------------------------------

def test_trace_then_timing_merges_one_row(ts):
    _trace(ts, "r1", raw="hallo welt", final="Hallo Welt",
           tokens=["Hallo", "Welt"], language="de", user_id="u")
    _timing(ts, "r1", 2.0, 1.0, words=2, user_id="u")
    assert ts.count() == 1
    row = ts.list_recent(limit=10)[0]
    assert row["raw"] == "hallo welt" and row["final"] == "Hallo Welt"
    assert row["audio_s"] == 2.0 and row["processing_s"] == 1.0
    assert row["words"] == 2
    assert row["rtf"] == 2.0  # audio/proc


def test_timing_only_inserts_minimal_row(ts):
    _timing(ts, "err1", None, 0.5, status="error", words=0)
    row = ts.list_recent(limit=10)[0]
    assert row["status"] == "error"
    # NULL text coerced to empty strings.
    assert row["raw"] == "" and row["final"] == ""
    assert row["rtf"] is None


def test_falsy_request_id_skipped(ts):
    _trace(ts, "")
    _timing(ts, "", 1.0, 1.0)
    assert ts.count() == 0


def test_trace_coalesce_preserves_user_on_timing(ts):
    _trace(ts, "r", raw="a", final="b", user_id="alice")
    # timing with user_id None must not wipe the existing user: record_timing's
    # ON CONFLICT clause never assigns user_id.
    _timing(ts, "r", 1.0, 1.0, user_id=None)
    row = ts.list_recent(limit=1)[0]
    assert row["user_id"] == "alice"


# ---------------------------------------------------------------------------
# list_recent pagination
# ---------------------------------------------------------------------------

def test_list_recent_newest_first_and_limit(ts):
    for i in range(5):
        _trace(ts, f"r{i}", created_ts=1000.0 + i)
    rows = ts.list_recent(limit=3)
    assert len(rows) == 3
    assert [r["request_id"] for r in rows] == ["r4", "r3", "r2"]


def test_list_recent_before_ts_cursor(ts):
    for i in range(5):
        _trace(ts, f"r{i}", created_ts=1000.0 + i)
    rows = ts.list_recent(before_ts=1002.0, limit=10)
    # strictly older than 1002 -> r0 (1000), r1 (1001)
    assert [r["request_id"] for r in rows] == ["r1", "r0"]


def test_list_recent_before_ts_zero_ignored(ts):
    _trace(ts, "r", created_ts=10.0)
    assert len(ts.list_recent(before_ts=0, limit=10)) == 1


def test_list_recent_user_filter(ts):
    _trace(ts, "a", user_id="u1", created_ts=1.0)
    _trace(ts, "b", user_id="u2", created_ts=2.0)
    rows = ts.list_recent(user_id_filter="u1", limit=10)
    assert [r["request_id"] for r in rows] == ["a"]


def test_list_recent_query_matches_raw_or_final(ts):
    _trace(ts, "r1", raw="patient hat Fieber", final="Patient hat Fieber", created_ts=1.0)
    _trace(ts, "r2", raw="kein treffer hier", final="Aspirin verordnet", created_ts=2.0)
    _trace(ts, "r3", raw="nichts", final="nichts", created_ts=3.0)
    # Matches raw of r1 and final of r2 (case-insensitive, ASCII).
    assert {r["request_id"] for r in ts.list_recent(query="fieber", limit=10)} == {"r1"}
    assert {r["request_id"] for r in ts.list_recent(query="ASPIRIN", limit=10)} == {"r2"}
    assert ts.list_recent(query="zzzznope", limit=10) == []


def test_list_recent_query_composes_with_user_and_cursor(ts):
    _trace(ts, "a", raw="alpha note", final="x", user_id="u1", created_ts=1.0)
    _trace(ts, "b", raw="alpha note", final="x", user_id="u2", created_ts=2.0)
    _trace(ts, "c", raw="alpha note", final="x", user_id="u1", created_ts=3.0)
    rows = ts.list_recent(query="alpha", user_id_filter="u1", before_ts=3.0, limit=10)
    assert [r["request_id"] for r in rows] == ["a"]


def test_list_recent_query_escapes_like_wildcards(ts):
    _trace(ts, "lit", raw="50% done", final="x", created_ts=1.0)
    _trace(ts, "other", raw="50 percent", final="x", created_ts=2.0)
    assert {r["request_id"] for r in ts.list_recent(query="50%", limit=10)} == {"lit"}


def test_list_recent_limit_floored_to_one(ts):
    _trace(ts, "r")
    assert len(ts.list_recent(limit=0)) == 1


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------

def test_prune_noop_when_both_zero(ts):
    _trace(ts, "r")
    assert ts.prune(max_rows=0, ttl_days=0) == 0
    assert ts.count() == 1


def test_prune_max_rows_keeps_newest(ts):
    # Use non-zero timestamps: created_ts=0.0 is falsy and falls back to now().
    for i in range(10):
        _trace(ts, f"r{i}", created_ts=1000.0 + i)
    deleted = ts.prune(max_rows=3, ttl_days=0)
    assert deleted == 7
    assert ts.count() == 3
    ids = {r["request_id"] for r in ts.list_recent(limit=10)}
    assert ids == {"r7", "r8", "r9"}


def test_prune_ttl_drops_old(ts):
    _trace(ts, "old", created_ts=time.time() - 40 * 86400)
    _trace(ts, "new")
    deleted = ts.prune(max_rows=0, ttl_days=30)
    assert deleted == 1
    assert {r["request_id"] for r in ts.list_recent(limit=10)} == {"new"}


def test_sweep_retention_applies_configured_cap_and_ttl(ts, monkeypatch):
    # The startup / hourly sweep: no insert counter involved, so a quiet
    # server still ages its rows out.
    from faster_whisper_backend.settings import config as cfg
    monkeypatch.setattr(cfg, "RECENT_TRANSCRIPTIONS_RETENTION_DAYS", 30, raising=False)
    monkeypatch.setattr(cfg, "RECENT_TRANSCRIPTIONS_MAX", 2, raising=False)
    _trace(ts, "old", created_ts=time.time() - 40 * 86400)
    for i in range(3):
        _trace(ts, f"n{i}", created_ts=time.time() - 10 + i)
    assert ts.sweep_retention() == 2
    assert {r["request_id"] for r in ts.list_recent(limit=10)} == {"n1", "n2"}


def test_clear_all(ts):
    for i in range(3):
        _trace(ts, f"r{i}")
    assert ts.clear_all() == 3
    assert ts.count() == 0


# ---------------------------------------------------------------------------
# _truncate_steps (pure)
# ---------------------------------------------------------------------------

def test_truncate_steps_drops_malformed():
    steps = [("ok", "a", "b"), "bad", ("short",), [1, 2], ("x", "y", "z")]
    out = rts._truncate_steps(steps)
    assert out == [["ok", "a", "b"], ["x", "y", "z"]]


def test_truncate_steps_front_trim(monkeypatch):
    monkeypatch.setattr(rts, "_CAP_STEPS_JSON", 80)
    steps = [(f"label{i}", "x" * 20, "y" * 20) for i in range(10)]
    out = rts._truncate_steps(steps)
    # Oldest (front) entries shed until the JSON blob fits the cap.
    assert 0 < len(out) < 10
    import json
    assert len(json.dumps(out)) <= 80
    assert out[-1][0] == "label9"


# ---------------------------------------------------------------------------
# _row_to_dict rtf guards
# ---------------------------------------------------------------------------

def test_row_to_dict_rtf_guard_proc_zero(ts):
    _timing(ts, "z", 5.0, 0.0)
    row = ts.list_recent(limit=1)[0]
    assert row["rtf"] is None


def test_schema_is_canonical_no_migration_on_fresh_db(ts):
    """A fresh DB gets kind/stages/key_label from _SCHEMA itself: the
    ALTER-TABLE migration loop adds nothing on a fresh create."""
    live = [r["name"] for r in
            ts._require_conn().execute("PRAGMA table_info(recent_transcriptions)")]
    bare = sqlite3.connect(":memory:")
    bare.executescript(ts._SCHEMA)
    schema_only = [r[1] for r in
                   bare.execute("PRAGMA table_info(recent_transcriptions)")]
    bare.close()
    assert live == schema_only
    for col in ("kind", "stages", "key_label"):
        assert col in schema_only


def _raw_insert(ts, request_id, **cols):
    conn = ts._require_conn()
    names = ["request_id", "created_ts", "model"] + list(cols)
    vals = [request_id, time.time(), "m"] + list(cols.values())
    conn.execute(
        f"INSERT INTO recent_transcriptions ({', '.join(names)}) "
        f"VALUES ({', '.join('?' * len(vals))})", vals)


def test_row_to_dict_json_columns_normalised_to_list(ts):
    """NULL, a JSON object and malformed text all read back as [] for both
    the recent-jobs `stages` column and the legacy `steps` column."""
    _raw_insert(ts, "null", stages=None, steps=None)
    _raw_insert(ts, "obj", stages='{"a": 1}', steps='{"a": 1}')
    _raw_insert(ts, "bad", stages="not json", steps="not json")
    rows = {r["request_id"]: r for r in ts.list_recent(limit=10)}
    for rid in ("null", "obj", "bad"):
        assert rows[rid]["stages"] == [], rid
        assert rows[rid]["steps"] == [], rid


def test_record_timing_keeps_wait_and_error_class(tx_store):
    """wait_s / error_class / error_stage ride the timing UPSERT and are
    COALESCEd on conflict, so a later write without them never blanks
    what an earlier one recorded."""
    _timing(tx_store, "w1", 2.0, 1.0, status="error", words=0, wait_s=3.25,
            error_class="cuda_oom", error_stage="transcribing")
    row = tx_store.list_recent(limit=1)[0]
    assert (row["wait_s"], row["error_class"], row["error_stage"]) == (
        3.25, "cuda_oom", "transcribing")
    _timing(tx_store, "w1", 2.0, 1.5, status="error", words=0)
    row = tx_store.list_recent(limit=1)[0]
    assert row["processing_s"] == 1.5
    assert (row["wait_s"], row["error_class"], row["error_stage"]) == (
        3.25, "cuda_oom", "transcribing")
    _timing(tx_store, "w2", 1.0, 1.0, words=3, wait_s=-1)
    assert tx_store.list_recent(limit=1)[0]["wait_s"] == 0.0


def test_list_recent_filters_compose_with_the_cursor(tx_store):
    """kind / status / slow_rtf narrow the page and keep the cursor walk."""
    for i in range(6):
        _timing(
            tx_store, f"r{i}", 10.0, (8.0 if i % 3 == 0 else 1.0),
            status=("error" if i == 1 else "cancelled" if i == 4 else "ok"),
            kind=("dictate" if i % 2 else "transcribe"),
            created_ts=1000.0 + i)
    ids = lambda rows: [r["request_id"] for r in rows]
    assert ids(tx_store.list_recent(limit=10, kind="dictate")) == ["r5", "r3", "r1"]
    assert ids(tx_store.list_recent(limit=10, status="failed")) == ["r4", "r1"]
    assert ids(tx_store.list_recent(limit=10, status="cancelled")) == ["r4"]
    assert ids(tx_store.list_recent(limit=10, slow_rtf=0.5)) == ["r3", "r0"]
    page = tx_store.list_recent(limit=2, kind="dictate")
    assert ids(page) == ["r5", "r3"]
    assert ids(tx_store.list_recent(limit=2, kind="dictate",
                                    before_ts=page[-1]["ts"])) == ["r1"]


def test_list_recent_kind_filter_resolves_null_kind_like_projection(ts):
    """Batch /transcribe rows are stored with kind NULL (resolved via source
    on read); kind=transcribe / kind=dictate must find them the way
    metrics.project_recent_row labels them."""
    _trace(ts, "b1", source="file", created_ts=1001.0)
    _timing(ts, "b1", 10.0, 1.0, created_ts=1001.0)
    _timing(ts, "b2", None, 0.5, status="error", words=0, created_ts=1002.0)
    _trace(ts, "l1", source="stream", created_ts=1003.0)
    _timing(ts, "l1", 3.0, 0.3, created_ts=1003.0)
    _timing(ts, "d1", 3.0, 0.3, kind="dictate", created_ts=1004.0)
    _timing(ts, "t1", None, 0.3, kind="translate", created_ts=1005.0)
    ids = lambda rows: [r["request_id"] for r in rows]
    assert set(ids(ts.list_recent(limit=10, kind="transcribe"))) == {"b1", "b2"}
    assert set(ids(ts.list_recent(limit=10, kind="dictate"))) == {"l1", "d1"}
    assert ids(ts.list_recent(limit=10, kind="translate")) == ["t1"]


def test_list_recent_kind_accepts_a_sequence(ts):
    """The /stats jobs page sends its kind chips as a set (the chosen kinds
    plus preload + download, which always show): list_recent takes it as an
    IN-style OR, each member resolved like the single-kind filter (NULL-kind
    batch rows included)."""
    _trace(ts, "b1", source="file", created_ts=1001.0)
    _timing(ts, "b1", 10.0, 1.0, created_ts=1001.0)
    _timing(ts, "dl", None, 2.0, words=0, kind="download", created_ts=1002.0)
    _timing(ts, "d1", 3.0, 0.3, kind="dictate", created_ts=1003.0)
    ids = lambda rows: [r["request_id"] for r in rows]
    assert ids(ts.list_recent(limit=10, kind=["transcribe", "download"])) == ["dl", "b1"]
    assert ids(ts.list_recent(limit=10, kind=("download",))) == ["dl"]
    assert ids(ts.list_recent(limit=10, kind=[])) == ["d1", "dl", "b1"]


def test_record_timing_carries_username_for_rows_without_a_trace(tx_store):
    """A translate row (and every error-path row) has no record_trace()
    half, so the username must land through record_timing itself —
    otherwise /stats shows '—' in USER·KEY for a job whose user is known."""
    tx_store.record_timing(request_id="tr1", model="mt", audio_s=None,
                           processing_s=0.7, status="ok", words=0,
                           user_id="u1", username="John Doe", kind="translate")
    row = tx_store.list_recent(limit=5)[0]
    assert row["request_id"] == "tr1"
    assert row["username"] == "John Doe"
    assert row["user_id"] == "u1"
    # A later timing write without a username never blanks the recorded one.
    tx_store.record_timing(request_id="tr1", model="mt", audio_s=None,
                           processing_s=0.9, status="ok", words=0, user_id="u1")
    assert tx_store.list_recent(limit=5)[0]["username"] == "John Doe"


def test_append_stage_folds_a_later_stage_into_the_row(tx_store):
    """A dictation's translation arrives on a separate request: it lands as
    a second stage on the utterance's row, adding its wall time — one row,
    two stages, like a batch job. A miss returns False so the caller can
    still record its own row."""
    tx_store.record_timing(request_id="u1", model="w", audio_s=4.0,
                           processing_s=0.9, status="ok", words=7,
                           user_id="u", kind="dictate",
                           stages=[{"name": "transcribing", "secs": 0.9}])
    ok = tx_store.append_stage("u1", {"name": "translating", "secs": 12.5,
                                      "model": "mt"}, add_processing_s=12.5)
    assert ok is True
    row = tx_store.list_recent(limit=5)[0]
    assert [s["name"] for s in row["stages"]] == ["transcribing", "translating"]
    assert row["processing_s"] == 13.4
    assert row["kind"] == "dictate"
    assert tx_store.append_stage("nope", {"name": "translating", "secs": 1}) is False


def test_failed_init_leaves_the_store_closed(tmp_path, monkeypatch):
    """A schema step that raises after the DB opened (here: an index on a
    column an old table lacks) must leave is_open() False, so metrics_snapshot
    does not re-read and re-log the broken store on every /stats frame."""
    monkeypatch.setattr(rts, "_conn", None)
    path = str(tmp_path / "broken.sqlite3")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE recent_transcriptions (id INTEGER PRIMARY KEY, request_id TEXT)")
    conn.commit()
    conn.close()
    with pytest.raises(sqlite3.OperationalError):
        rts.init_db(path)
    assert rts.is_open() is False
