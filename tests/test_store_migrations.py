"""Additive column migrations on the report and capture stores.

Both stores only ever ran `executescript(_SCHEMA)`, which is a no-op against
an existing table — so neither had a migration hook and neither could grow a
column without silently doing nothing. The hooks added for job provenance and
per-language translations are new code that runs against live databases on
every startup, so what matters here is that they are idempotent and that they
leave existing rows intact.
"""

import sqlite3

import pytest


def _cols(conn, table):
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------

def test_reports_migration_adds_columns_and_is_idempotent(tmp_path):
    import reports_store
    db = str(tmp_path / "reports.db")

    reports_store.init_db(db)
    cols = _cols(reports_store._conn, "reports")
    assert {"language", "stages_json"} <= cols

    # Second init on the same file must not raise "duplicate column name".
    reports_store.init_db(db)
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

    reports_store.init_db(db)
    assert {"language", "stages_json"} <= _cols(reports_store._conn, "reports")
    # The pre-existing row survives and reads back with empty provenance.
    row = reports_store.get_report("keep")
    assert row is not None
    assert row["raw"] == "r"
    assert row["language"] == ""
    assert row["stages"] == []


def test_reports_round_trip_provenance(tmp_path):
    import reports_store
    reports_store.init_db(str(tmp_path / "r.db"))
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


def test_reports_resubmission_without_provenance_keeps_it(tmp_path):
    """COALESCE, not overwrite: a client that resubmits a correction without
    re-sending the job context must not erase it."""
    import reports_store
    reports_store.init_db(str(tmp_path / "r.db"))
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
