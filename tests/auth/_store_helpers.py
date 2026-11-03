"""Helpers shared by the auth store test modules (api_keys_store and
sessions_store, seen from a second uvicorn worker).

Import as ``from tests.auth._store_helpers import ...``.
"""

import importlib.util
import os
import sqlite3

from faster_whisper_backend.paths import REPO_ROOT


def load_sessions_store_copy(name, db_path):
    """An independent copy of sessions_store on `db_path`, like one uvicorn
    worker with its own connection and in-memory index."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO_ROOT, "faster_whisper_backend", "auth", "sessions_store.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.init_db(db_path)
    return mod


def count_rebuilds(store):
    """Wrap `store`'s _rebuild_index_locked; the returned list gains one entry
    per rebuild."""
    calls = []
    real = store._rebuild_index_locked

    def _counting():
        calls.append(1)
        real()

    store._rebuild_index_locked = _counting
    return calls


def sibling_write(store, *stmts):
    """Commit `stmts` ((sql,) or (sql, params)) in one transaction on a second
    connection to `store`'s DB file, the way a sibling worker would."""
    db_path = store._require_conn().execute(
        "PRAGMA database_list").fetchone()["file"]
    sibling = sqlite3.connect(db_path)
    try:
        with sibling:
            for stmt in stmts:
                sibling.execute(*stmt)
    finally:
        sibling.close()


def failing_rebuild_conn(store, select_prefix):
    """A `_require_conn` stand-in whose rebuild SELECT (sql starting with
    `select_prefix`) raises 'database is locked'; everything else runs on
    the real connection."""
    real = store._require_conn()

    class _FailingRebuild:
        def execute(self, sql, *args):
            if sql.startswith(select_prefix):
                raise sqlite3.OperationalError("database is locked")
            return real.execute(sql, *args)

    return _FailingRebuild
