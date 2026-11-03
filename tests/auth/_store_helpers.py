"""Helpers shared by the auth store test modules (api_keys_store and
sessions_store, seen from a second uvicorn worker).

Import as ``from tests.auth._store_helpers import ...``.
"""

import sqlite3


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
