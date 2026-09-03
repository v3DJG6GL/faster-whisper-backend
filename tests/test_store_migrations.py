"""Additive column migrations on the report and capture stores.

Both stores only ever ran `executescript(_SCHEMA)`, which is a no-op against
an existing table — so neither had a migration hook and neither could grow a
column without silently doing nothing. The hooks added for job provenance and
per-language translations are new code that runs against live databases on
every startup, so what matters here is that they are idempotent and that they
leave existing rows intact.
"""

import contextlib
import sqlite3

import pytest


def _cols(conn, table):
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


@contextlib.contextmanager
def _own_db(module):
    """Teardown for tests that call init_db()/init() on their own DB path:
    neither hook closes the previous handle, so a bare call leaks the
    connection AND leaves the module global bound to a tmp_path DB pytest
    deletes. Tests without a custom path use the conftest fixtures instead."""
    try:
        yield module
    finally:
        conn = module._conn
        module._conn = None
        if conn is not None:
            conn.close()
        if hasattr(module, "_audio_dir"):
            module._audio_dir = None


def _fake_wav_transcode(src_path, dst_path):
    import wave
    with wave.open(dst_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 100)
    return 1234


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------

def test_reports_migration_adds_columns_and_is_idempotent(tmp_path):
    import reports_store
    db = str(tmp_path / "reports.db")

    with _own_db(reports_store):
        reports_store.init_db(db)
        cols = _cols(reports_store._conn, "reports")
        assert {"language", "stages"} <= cols

        # Second init on the same file must not raise "duplicate column name".
        first_conn = reports_store._conn
        reports_store.init_db(db)
        first_conn.close()
        assert _cols(reports_store._conn, "reports") == cols


def test_reports_migration_upgrades_a_pre_existing_table(tmp_path):
    """The real case: a database created before the columns existed."""
    import reports_store
    db = str(tmp_path / "old.db")
    old = sqlite3.connect(db)
    old.executescript("""
        CREATE TABLE reports (
          id TEXT PRIMARY KEY, created_ts REAL NOT NULL, trace_ts REAL NOT NULL,
          request_id TEXT, model TEXT NOT NULL, raw TEXT NOT NULL,
          final TEXT NOT NULL, steps_json TEXT NOT NULL,
          corrections_json TEXT NOT NULL DEFAULT '[]',
          intended_text TEXT NOT NULL DEFAULT '',
          user_comment TEXT NOT NULL DEFAULT '',
          reporter_role TEXT NOT NULL,
          reporter_host TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL DEFAULT 'open',
          admin_notes TEXT NOT NULL DEFAULT '', resolved_ts REAL, user_id TEXT);
    """)
    old.execute(
        "INSERT INTO reports (id, created_ts, trace_ts, model, raw, final,"
        " steps_json, reporter_role) VALUES (?,?,?,?,?,?,?,?)",
        ("keep", 1.0, 1.0, "m", "r", "f", "[]", "user"))
    old.commit()
    old.close()

    with _own_db(reports_store):
        reports_store.init_db(db)
        cols = _cols(reports_store._conn, "reports")
        assert {"language", "stages"} <= cols
        # The JSON columns were renamed in place: the bare names exist, the
        # `_json`-suffixed ones are gone.
        assert {"steps", "corrections", "stages"} <= cols
        assert not ({"steps_json", "corrections_json", "stages_json"} & cols)
        # The pre-existing row survives and reads back with empty provenance.
        row = reports_store.get_report("keep")
        assert row is not None
        assert row["raw"] == "r"
        assert row["language"] == ""
        assert row["stages"] == []


def test_reports_round_trip_provenance(reports_store_db):
    reports_store = reports_store_db
    rid, updated = reports_store.upsert_report(
        user_id="u1", request_id="req1", trace_ts=1.0, model="large-v2",
        raw="r", final="f", steps=[], corrections=[], intended_text="",
        user_comment="the french is wrong", reporter_role="user",
        reporter_host="h", language="de",
        stages=[{"name": "translating", "secs": 2.0, "model": "HY",
                 "detail": "3 segs → en,fr"}])
    assert updated is False
    got = reports_store.get_report(rid)
    assert got["language"] == "de"
    assert got["stages"][0]["detail"] == "3 segs → en,fr"


def test_reports_resubmission_without_provenance_keeps_it(reports_store_db):
    """COALESCE, not overwrite: a client that resubmits a correction without
    re-sending the job context must not erase it."""
    reports_store = reports_store_db
    common = dict(user_id="u1", request_id="req1", trace_ts=1.0, model="m",
                  raw="r", final="f", steps=[], corrections=[],
                  intended_text="", user_comment="c", reporter_role="user",
                  reporter_host="h")
    rid, _ = reports_store.upsert_report(
        language="de", stages=[{"name": "translating"}], **common)
    rid2, updated = reports_store.upsert_report(**common)
    assert (rid2, updated) == (rid, True)
    got = reports_store.get_report(rid)
    assert got["language"] == "de"
    assert got["stages"] == [{"name": "translating"}]


# ---------------------------------------------------------------------------
# captures
# ---------------------------------------------------------------------------

def test_captures_migration_adds_columns_and_is_idempotent(tmp_path):
    import captures_store
    db = str(tmp_path / "cap.db")
    audio = str(tmp_path / "audio")

    with _own_db(captures_store):
        captures_store.init_db(db, audio)
        cols = _cols(captures_store._conn, "captures")
        assert {"translations", "translation_model",
                "translation_source", "task"} <= cols

        first_conn = captures_store._conn
        captures_store.init_db(db, audio)
        first_conn.close()
        assert _cols(captures_store._conn, "captures") == cols


def test_captures_migration_upgrades_a_pre_existing_table(tmp_path):
    import captures_store
    db = str(tmp_path / "old.db")
    audio = str(tmp_path / "audio")
    old = sqlite3.connect(db)
    old.executescript("""
        CREATE TABLE captures (
          id TEXT PRIMARY KEY, created_ts REAL NOT NULL, request_id TEXT,
          model TEXT NOT NULL, language TEXT, duration_seconds REAL,
          audio_relpath TEXT NOT NULL, audio_format TEXT NOT NULL,
          raw TEXT NOT NULL, final TEXT NOT NULL, text_for_training TEXT,
          audio_trimmed_relpath TEXT, audio_trim_lead_ms INTEGER,
          audio_trim_trail_ms INTEGER, words_json TEXT NOT NULL,
          segments_json TEXT NOT NULL DEFAULT '[]',
          corrected_text TEXT NOT NULL DEFAULT '',
          corrections_json TEXT NOT NULL DEFAULT '[]',
          admin_notes TEXT NOT NULL DEFAULT '',
          status TEXT NOT NULL DEFAULT 'new', reviewed_ts REAL,
          user_id TEXT, sample_id TEXT, sample_order INTEGER);
    """)
    old.execute(
        "INSERT INTO captures (id, created_ts, model, audio_relpath,"
        " audio_format, raw, final, words_json) VALUES (?,?,?,?,?,?,?,?)",
        ("keep", 1.0, "m", "a/b/keep.wav", "wav", "r", "f", "[]"))
    old.commit()
    old.close()

    with _own_db(captures_store):
        captures_store.init_db(db, audio)
        cols = _cols(captures_store._conn, "captures")
        # The JSON columns were renamed in place: the bare names exist, the
        # `_json`-suffixed ones are gone.
        assert {"words", "segments", "corrections", "translations"} <= cols
        assert not ({"words_json", "segments_json", "corrections_json",
                     "translations_json"} & cols)
        got = captures_store.get_capture("keep")
        assert got is not None
        assert got["words"] == []
        assert got["final"] == "f"
        # A pre-migration row reads back with no translations, not a crash.
        assert got["translations"] == {}


def test_captures_translations_round_trip_as_a_keyed_map(
        tmp_path, monkeypatch, captures_store_db):
    """The point of the column: the exporter must be able to pick ONE
    language out. Whisper's translate task targets English only, so a joined
    blob would be unusable."""
    import audio_transcode

    captures_store = captures_store_db
    monkeypatch.setattr(audio_transcode, "transcode_to_wav_16k_mono",
                        _fake_wav_transcode)
    src = tmp_path / "in.bin"
    src.write_bytes(b"junk")

    cid = captures_store.create_capture(
        audio_src_path=str(src), request_id="r1", model="large-v2",
        language="de", audio_s=1.0, raw="hallo", final="hallo",
        words=[], segments=[], task="transcribe",
        translations={"en": "hello", "fr": "salut"},
        translation_model="HY-MT", translation_source="cascade-mt")

    got = captures_store.get_capture(cid)
    assert got["translations"] == {"en": "hello", "fr": "salut"}
    assert got["task"] == "transcribe"
    assert got["translation_source"] == "cascade-mt"
    assert got["translation_model"] == "HY-MT"
    # The transcript itself stays in the source language.
    assert got["final"] == "hallo"


def test_captures_list_projection_carries_the_new_columns(
        tmp_path, monkeypatch, captures_store_db):
    """_LIST_COLUMNS is a hand-maintained projection whose own comment exists
    to stop exactly this drift: a column missing from it vanishes from
    /captures/api/list while still being present in the table."""
    import audio_transcode

    captures_store = captures_store_db
    # _LIST_COLUMNS is one comma-separated string; parse it into real column
    # names (a substring check would pass on any renamed superstring).
    cols = {c.strip() for c in captures_store._LIST_COLUMNS.split(",")}
    assert {"translations", "translation_model",
            "translation_source", "task"} <= cols

    # And end-to-end: a listed row actually carries the public keys.
    monkeypatch.setattr(audio_transcode, "transcode_to_wav_16k_mono",
                        _fake_wav_transcode)
    src = tmp_path / "in.bin"
    src.write_bytes(b"junk")
    captures_store.create_capture(
        audio_src_path=str(src), request_id="r1", model="m", language="de",
        audio_s=1.0, raw="r", final="f", words=[], segments=[],
        task="transcribe", translations={"en": "hello"},
        translation_model="HY-MT", translation_source="cascade-mt")
    listed = captures_store.list_captures()[0]
    assert listed["translations"] == {"en": "hello"}
    assert listed["translation_model"] == "HY-MT"
    assert listed["translation_source"] == "cascade-mt"
    assert listed["task"] == "transcribe"
