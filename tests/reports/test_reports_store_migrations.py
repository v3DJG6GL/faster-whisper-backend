"""Additive column migrations on the report store (job provenance).

The store only ever ran `executescript(_SCHEMA)`, which is a no-op against an
existing table — so it had no migration hook and could not grow a column
without silently doing nothing. The hook runs against live databases on every
startup, so what matters is that it is idempotent and leaves existing rows
intact. The capture store's twin lives in tests/captures/test_store_migrations.py."""

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


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------

def test_reports_migration_adds_columns_and_is_idempotent(tmp_path):
    from faster_whisper_backend.reports import store as reports_store
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
    from faster_whisper_backend.reports import store as reports_store
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
