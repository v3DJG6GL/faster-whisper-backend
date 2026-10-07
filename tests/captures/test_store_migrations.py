"""Additive column migrations on the capture store (per-language translations,
job provenance).

The store only ever ran `executescript(_SCHEMA)`, which is a no-op against an
existing table — so it had no migration hook and could not grow a column
without silently doing nothing. The hook runs against live databases on every
startup, so what matters is that it is idempotent and leaves existing rows
intact. The report store's twin lives in tests/reports/test_reports_store_migrations.py."""

import contextlib
import sqlite3


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
# captures
# ---------------------------------------------------------------------------

def test_captures_migration_adds_columns_and_is_idempotent(tmp_path):
    from faster_whisper_backend.captures import store as captures_store
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
    from faster_whisper_backend.captures import store as captures_store
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
          user_id TEXT, sample_id TEXT, sample_order INTEGER,
          translations_json TEXT);
    """)
    old.execute(
        "INSERT INTO captures (id, created_ts, model, duration_seconds,"
        " audio_relpath, audio_format, raw, final, words_json,"
        " translations_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("keep", 1.0, "m", 2.5, "a/b/keep.wav", "wav", "r", "f", "[]",
         '{"en": "hello"}'))
    old.execute(
        "INSERT INTO captures (id, created_ts, model, audio_relpath,"
        " audio_format, raw, final, words_json) VALUES (?,?,?,?,?,?,?,?)",
        ("bare", 2.0, "m", "a/b/bare.wav", "wav", "r", "f", "[]"))
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
        # Every rename keeps its data, not just the column name.
        assert got["words"] == []
        assert got["raw"] == "r" and got["final"] == "f"
        assert got["audio_s"] == 2.5
        assert got["translations"] == {"en": "hello"}
        # A row that never had translations reads back empty, not a crash.
        assert captures_store.get_capture("bare")["translations"] == {}


def test_captures_translations_round_trip_as_a_keyed_map(
        tmp_path, monkeypatch, captures_store_db):
    """The point of the column: the exporter must be able to pick ONE
    language out. Whisper's translate task targets English only, so a joined
    blob would be unusable."""
    from faster_whisper_backend.audio import transcode as audio_transcode

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
    from faster_whisper_backend.audio import transcode as audio_transcode

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


# ---------------------------------------------------------------------------
# capture_samples
# ---------------------------------------------------------------------------

def test_capture_samples_migration_renames_the_json_columns(tmp_path):
    """A pre-cddbf13 install carries member_hashes_json / member_trims_json.
    Every read and write uses the bare names, so a rename that silently
    skipped would 500 every sample read on an existing install."""
    from faster_whisper_backend.captures import samples_store as capture_samples_store
    from faster_whisper_backend.core import store_common

    conn = store_common.open_wal_db(str(tmp_path / "old.db"))
    conn.executescript("""
        CREATE TABLE capture_samples (
          id TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_ts REAL NOT NULL,
          merged_wav_relpath TEXT NOT NULL, merged_duration_ms INTEGER NOT NULL,
          transcript TEXT NOT NULL,
          transcript_join_strategy TEXT NOT NULL DEFAULT 'space',
          member_hashes_json TEXT NOT NULL,
          inter_segment_silence_ms INTEGER NOT NULL DEFAULT 300,
          is_stale INTEGER NOT NULL DEFAULT 0,
          is_locked INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL DEFAULT 'new',
          admin_notes TEXT NOT NULL DEFAULT '', language TEXT,
          merged_lead_trim_ms INTEGER NOT NULL DEFAULT 0,
          merged_trail_trim_ms INTEGER NOT NULL DEFAULT 0,
          member_trims_json TEXT NOT NULL DEFAULT '{}');
    """)
    conn.execute(
        "INSERT INTO capture_samples (id, user_id, created_ts,"
        " merged_wav_relpath, merged_duration_ms, transcript,"
        " member_hashes_json, member_trims_json) VALUES (?,?,?,?,?,?,?,?)",
        ("sold", "u1", 1.0, "groups/so/ld/sold.wav", 5000, "t",
         '{"a":"h"}', '{"a":{"lead_ms":5}}'))
    audio = str(tmp_path / "audio")
    try:
        capture_samples_store.init_db(conn, audio)
        cols = _cols(conn, "capture_samples")
        assert {"member_hashes", "member_trims"} <= cols
        assert not ({"member_hashes_json", "member_trims_json"} & cols)
        got = capture_samples_store.get_sample("sold")
        assert got["member_hashes"] == {"a": "h"}
        assert got["member_trims"] == {"a": {"lead_ms": 5}}
        # A second startup is a no-op.
        capture_samples_store.init_db(conn, audio)
        assert _cols(conn, "capture_samples") == cols
        assert capture_samples_store.get_sample("sold")["member_hashes"] == {"a": "h"}
    finally:
        capture_samples_store._conn = None
        capture_samples_store._groups_audio_dir = None
        conn.close()
