"""Tests for transcriptions_store: trace/timing UPSERT merge, pagination,
prune, step truncation, and the row-to-dict derivations."""

import sqlite3

import pytest
import time


# ---------------------------------------------------------------------------
# Recent-jobs columns (kind / stages_json / key_label) + migration
# ---------------------------------------------------------------------------

def test_kind_stages_key_label_roundtrip(tx_store):
    ts = tx_store
    stages = [{"name": "translate", "secs": 1.25, "model": "org/m:Q4",
               "detail": "3 segs → en"}]
    ts.record_timing(request_id="t1", model="org/m:Q4", audio_dur_s=None,
                     proc_dur_s=1.25, status="ok", words_count=0,
                     kind="translate", stages=stages, key_label="ci-key")
    row = ts.list_recent(limit=10)[0]
    assert row["kind"] == "translate"
    assert row["key_label"] == "ci-key"
    assert row["stages"] == stages


def test_timing_without_kind_keeps_earlier_values(tx_store):
    ts = tx_store
    ts.record_timing(request_id="k1", model="m", audio_dur_s=None,
                     proc_dur_s=1.0, status="ok", words_count=0,
                     kind="download", stages=[{"name": "download", "secs": 1}],
                     key_label="lbl")
    # A later kwarg-less write (e.g. legacy caller) must not blank them.
    ts.record_timing(request_id="k1", model="m", audio_dur_s=None,
                     proc_dur_s=2.0, status="ok", words_count=0)
    row = ts.list_recent(limit=10)[0]
    assert row["kind"] == "download"
    assert row["key_label"] == "lbl"
    assert row["stages"] and row["proc_dur"] == 2.0


def test_migration_adds_columns_to_old_db(tmp_path):
    """A DB created before the recent-jobs columns migrates on init_db and
    its old rows still render (kind None, stages [], key_label '')."""
    import transcriptions_store as mod

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

    mod.init_db(path)
    try:
        cols = {r["name"] for r in mod._require_conn().execute(
            "PRAGMA table_info(recent_transcriptions)")}
        assert {"source", "kind", "stages_json", "key_label",
                "wait_s", "error_class", "error_stage"} <= cols
        row = mod.list_recent(limit=10)[0]
        assert row["request_id"] == "old1"
        assert row["kind"] is None
        assert row["stages"] == []
        assert row["key_label"] == ""
        assert row["source"] == "file"
        assert row["wait_s"] is None and row["error_class"] is None
        assert row["error_stage"] is None
    finally:
        mod._require_conn().close()
        mod._conn = None


# ---------------------------------------------------------------------------
# record_trace / record_timing UPSERT
# ---------------------------------------------------------------------------

def test_trace_then_timing_merges_one_row(tx_store):
    ts = tx_store
    ts.record_trace(request_id="r1", model="m", raw="hallo welt", final="Hallo Welt",
                    tokens=["Hallo", "Welt"], language="de", user_id="u")
    ts.record_timing(request_id="r1", model="m", audio_dur_s=2.0, proc_dur_s=1.0,
                     status="ok", words_count=2, user_id="u")
    assert ts.count() == 1
    row = ts.list_recent(limit=10)[0]
    assert row["raw"] == "hallo welt" and row["final"] == "Hallo Welt"
    assert row["audio_dur"] == 2.0 and row["proc_dur"] == 1.0
    assert row["words"] == 2
    assert row["rtf"] == 2.0  # audio/proc


def test_timing_only_inserts_minimal_row(tx_store):
    ts = tx_store
    ts.record_timing(request_id="err1", model="m", audio_dur_s=None, proc_dur_s=0.5,
                     status="error", words_count=0)
    row = ts.list_recent(limit=10)[0]
    assert row["status"] == "error"
    # NULL text coerced to empty strings.
    assert row["raw"] == "" and row["final"] == ""
    assert row["rtf"] is None  # audio_dur_s None -> no rtf


def test_falsy_request_id_skipped(tx_store):
    ts = tx_store
    ts.record_trace(request_id="", model="m", raw="x", final="y")
    ts.record_timing(request_id="", model="m", audio_dur_s=1.0, proc_dur_s=1.0,
                     status="ok", words_count=1)
    assert ts.count() == 0


def test_trace_coalesce_preserves_user_on_timing(tx_store):
    ts = tx_store
    ts.record_trace(request_id="r", model="m", raw="a", final="b", user_id="alice")
    # timing with user_id None must not wipe the existing user (COALESCE).
    ts.record_timing(request_id="r", model="m", audio_dur_s=1.0, proc_dur_s=1.0,
                     status="ok", words_count=1, user_id=None)
    row = ts.list_recent(limit=1)[0]
    assert row["user_id"] == "alice"


# ---------------------------------------------------------------------------
# list_recent pagination
# ---------------------------------------------------------------------------

def test_list_recent_newest_first_and_limit(tx_store):
    ts = tx_store
    for i in range(5):
        ts.record_trace(request_id=f"r{i}", model="m", raw="x", final="y",
                        created_ts=1000.0 + i)
    rows = ts.list_recent(limit=3)
    assert len(rows) == 3
    assert [r["request_id"] for r in rows] == ["r4", "r3", "r2"]


def test_list_recent_before_ts_cursor(tx_store):
    ts = tx_store
    for i in range(5):
        ts.record_trace(request_id=f"r{i}", model="m", raw="x", final="y",
                        created_ts=1000.0 + i)
    rows = ts.list_recent(before_ts=1002.0, limit=10)
    # strictly older than 1002 -> r0 (1000), r1 (1001)
    assert [r["request_id"] for r in rows] == ["r1", "r0"]


def test_list_recent_before_ts_zero_ignored(tx_store):
    ts = tx_store
    ts.record_trace(request_id="r", model="m", raw="x", final="y", created_ts=10.0)
    assert len(ts.list_recent(before_ts=0, limit=10)) == 1


def test_list_recent_user_filter(tx_store):
    ts = tx_store
    ts.record_trace(request_id="a", model="m", raw="x", final="y", user_id="u1",
                    created_ts=1.0)
    ts.record_trace(request_id="b", model="m", raw="x", final="y", user_id="u2",
                    created_ts=2.0)
    rows = ts.list_recent(user_id_filter="u1", limit=10)
    assert [r["request_id"] for r in rows] == ["a"]


def test_list_recent_query_matches_raw_or_final(tx_store):
    ts = tx_store
    ts.record_trace(request_id="r1", model="m", raw="patient hat Fieber",
                    final="Patient hat Fieber", created_ts=1.0)
    ts.record_trace(request_id="r2", model="m", raw="kein treffer hier",
                    final="Aspirin verordnet", created_ts=2.0)
    ts.record_trace(request_id="r3", model="m", raw="nichts", final="nichts",
                    created_ts=3.0)
    # Matches raw of r1 and final of r2 (case-insensitive, ASCII).
    assert {r["request_id"] for r in ts.list_recent(query="fieber", limit=10)} == {"r1"}
    assert {r["request_id"] for r in ts.list_recent(query="ASPIRIN", limit=10)} == {"r2"}
    assert ts.list_recent(query="zzzznope", limit=10) == []


def test_list_recent_query_composes_with_user_and_cursor(tx_store):
    ts = tx_store
    ts.record_trace(request_id="a", model="m", raw="alpha note", final="x",
                    user_id="u1", created_ts=1.0)
    ts.record_trace(request_id="b", model="m", raw="alpha note", final="x",
                    user_id="u2", created_ts=2.0)
    ts.record_trace(request_id="c", model="m", raw="alpha note", final="x",
                    user_id="u1", created_ts=3.0)
    # query AND user_id_filter AND before_ts all compose.
    rows = ts.list_recent(query="alpha", user_id_filter="u1", before_ts=3.0, limit=10)
    assert [r["request_id"] for r in rows] == ["a"]


def test_list_recent_query_escapes_like_wildcards(tx_store):
    ts = tx_store
    ts.record_trace(request_id="lit", model="m", raw="50% done", final="x",
                    created_ts=1.0)
    ts.record_trace(request_id="other", model="m", raw="50 percent", final="x",
                    created_ts=2.0)
    # A literal "%" must match only the row containing it, not act as a wildcard.
    assert {r["request_id"] for r in ts.list_recent(query="50%", limit=10)} == {"lit"}


def test_list_recent_limit_floored_to_one(tx_store):
    ts = tx_store
    ts.record_trace(request_id="r", model="m", raw="x", final="y")
    assert len(ts.list_recent(limit=0)) == 1


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------

def test_prune_noop_when_both_zero(tx_store):
    ts = tx_store
    ts.record_trace(request_id="r", model="m", raw="x", final="y")
    assert ts.prune(max_rows=0, ttl_days=0) == 0
    assert ts.count() == 1


def test_prune_max_rows_keeps_newest(tx_store):
    ts = tx_store
    # Use non-zero timestamps: created_ts=0.0 is falsy and falls back to now().
    for i in range(10):
        ts.record_trace(request_id=f"r{i}", model="m", raw="x", final="y",
                        created_ts=1000.0 + i)
    deleted = ts.prune(max_rows=3, ttl_days=0)
    assert deleted == 7
    assert ts.count() == 3
    ids = {r["request_id"] for r in ts.list_recent(limit=10)}
    assert ids == {"r7", "r8", "r9"}


def test_prune_ttl_drops_old(tx_store):
    ts = tx_store
    ts.record_trace(request_id="old", model="m", raw="x", final="y",
                    created_ts=time.time() - 40 * 86400)
    ts.record_trace(request_id="new", model="m", raw="x", final="y")
    deleted = ts.prune(max_rows=0, ttl_days=30)
    assert deleted == 1
    assert {r["request_id"] for r in ts.list_recent(limit=10)} == {"new"}


def test_clear_all(tx_store):
    ts = tx_store
    for i in range(3):
        ts.record_trace(request_id=f"r{i}", model="m", raw="x", final="y")
    assert ts.clear_all() == 3
    assert ts.count() == 0


# ---------------------------------------------------------------------------
# _truncate_steps (pure)
# ---------------------------------------------------------------------------

def test_truncate_steps_drops_malformed():
    import transcriptions_store as ts
    steps = [("ok", "a", "b"), "bad", ("short",), [1, 2], ("x", "y", "z")]
    out = ts._truncate_steps(steps)
    assert out == [["ok", "a", "b"], ["x", "y", "z"]]


def test_truncate_steps_front_trim(monkeypatch):
    import transcriptions_store as ts
    monkeypatch.setattr(ts, "_CAP_STEPS_JSON", 80)
    steps = [(f"label{i}", "x" * 20, "y" * 20) for i in range(10)]
    out = ts._truncate_steps(steps)
    # Oldest (front) entries shed until the JSON blob fits the cap.
    assert len(out) < 10
    assert out[-1][0] == "label9"  # newest preserved


# ---------------------------------------------------------------------------
# _row_to_dict rtf guards
# ---------------------------------------------------------------------------

def test_row_to_dict_rtf_guard_proc_zero(tx_store):
    ts = tx_store
    ts.record_timing(request_id="z", model="m", audio_dur_s=5.0, proc_dur_s=0.0,
                     status="ok", words_count=1)
    row = ts.list_recent(limit=1)[0]
    assert row["rtf"] is None  # proc 0 -> guarded


# ---------------------------------------------------------------------------
# _SCHEMA describes the whole table; JSON-column normalisation
# ---------------------------------------------------------------------------

def test_schema_is_canonical_no_migration_on_fresh_db(tx_store):
    """A fresh DB gets kind/stages_json/key_label from _SCHEMA itself: the
    ALTER-TABLE migration loop adds nothing on a fresh create."""
    ts = tx_store
    live = [r["name"] for r in
            ts._require_conn().execute("PRAGMA table_info(recent_transcriptions)")]
    bare = sqlite3.connect(":memory:")
    bare.executescript(ts._SCHEMA)
    schema_only = [r[1] for r in
                   bare.execute("PRAGMA table_info(recent_transcriptions)")]
    bare.close()
    assert live == schema_only
    for col in ("kind", "stages_json", "key_label"):
        assert col in schema_only


def _raw_insert(ts, request_id, **cols):
    conn = ts._require_conn()
    names = ["request_id", "created_ts", "model"] + list(cols)
    vals = [request_id, time.time(), "m"] + list(cols.values())
    conn.execute(
        f"INSERT INTO recent_transcriptions ({', '.join(names)}) "
        f"VALUES ({', '.join('?' * len(vals))})", vals)


def test_row_to_dict_json_columns_normalised_to_list(tx_store):
    """NULL, a JSON object and malformed text all read back as [] for both
    the recent-jobs `stages` column and the legacy `steps` column: the
    isinstance-list guard now covers every JSON column, not just stages."""
    ts = tx_store
    _raw_insert(ts, "null", stages_json=None, steps_json=None)
    _raw_insert(ts, "obj", stages_json='{"a": 1}', steps_json='{"a": 1}')
    _raw_insert(ts, "bad", stages_json="not json", steps_json="not json")
    rows = {r["request_id"]: r for r in ts.list_recent(limit=10)}
    for rid in ("null", "obj", "bad"):
        assert rows[rid]["stages"] == [], rid
        assert rows[rid]["steps"] == [], rid


def test_record_timing_keeps_wait_and_error_class(tx_store):
    """wait_s / error_class / error_stage ride the timing UPSERT and are
    COALESCEd on conflict, so a later write without them never blanks
    what an earlier one recorded."""
    tx_store.record_timing(request_id="w1", model="m", audio_dur_s=2.0,
                           proc_dur_s=1.0, status="error", words_count=0,
                           wait_s=3.25, error_class="cuda_oom",
                           error_stage="transcribing")
    row = tx_store.list_recent(limit=1)[0]
    assert (row["wait_s"], row["error_class"], row["error_stage"]) == (
        3.25, "cuda_oom", "transcribing")
    tx_store.record_timing(request_id="w1", model="m", audio_dur_s=2.0,
                           proc_dur_s=1.5, status="error", words_count=0)
    row = tx_store.list_recent(limit=1)[0]
    assert row["proc_dur"] == 1.5
    assert (row["wait_s"], row["error_class"]) == (3.25, "cuda_oom")
    tx_store.record_timing(request_id="w2", model="m", audio_dur_s=1.0,
                           proc_dur_s=1.0, status="ok", words_count=3, wait_s=-1)
    assert tx_store.list_recent(limit=1)[0]["wait_s"] == 0.0


def test_sys_samples_round_trip_downsample_and_prune(tx_store):
    import time
    now = int(time.time()) // 10 * 10
    rows = [{"ts": now - 60 + i * 10, "gpu_util": 10.0 * i, "gpu_mem_mb": 100.0,
             "gpu_temp": None, "cpu_pct": 5.0, "ram_pct": 50.0, "slot_busy": 0.5}
            for i in range(6)]
    assert tx_store.record_sys_samples(rows) == 6
    tx_store.record_sys_samples([{"ts": now - 60, "gpu_util": 99.0}])   # replace
    fine = tx_store.list_sys_samples(metric="gpu_util", from_ts=now - 60,
                                     to_ts=now + 1, step_s=10)
    assert fine["t"] == [now - 60 + i * 10 for i in range(6)]
    assert fine["avg"][0] == 99.0 and fine["max"][-1] == 50.0
    coarse = tx_store.list_sys_samples(metric="gpu_util", from_ts=now - 60,
                                       to_ts=now + 1, step_s=30)
    assert len(coarse["t"]) == 2
    assert coarse["max"][0] == 99.0 and coarse["avg"][1] == pytest.approx(40.0)
    # NULL samples (no NVML) are skipped, not zeroed.
    assert tx_store.list_sys_samples(metric="gpu_temp", from_ts=now - 60,
                                     to_ts=now + 1, step_s=10)["t"] == []
    with pytest.raises(ValueError):
        tx_store.list_sys_samples(metric="ts; DROP TABLE x", from_ts=0,
                                  to_ts=now, step_s=10)
    tx_store.record_sys_samples([{"ts": now - 40 * 86400, "gpu_util": 1.0}])
    assert tx_store.prune_sys_samples(30) == 1
    assert tx_store.prune_sys_samples(0) == 0
