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
