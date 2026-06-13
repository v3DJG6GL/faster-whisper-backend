"""Store-level tests for per-identity config bindings + the api_keys.config
migration. Pure-Python (SQLite only) — runnable on the web-only box."""

import sqlite3

import pytest

import config as cfg
import api_keys_store as ak


@pytest.fixture
def profiles(monkeypatch):
    monkeypatch.setattr(cfg, "OVERRIDE_PROFILES", {
        "clinic-de": {"DEFAULT_LANGUAGE": "de", "BEAM_SIZE": 8,
                      "locks": ["DEFAULT_LANGUAGE"]},
        "fast": {"BEAM_SIZE": 1},
    }, raising=False)
    return cfg.OVERRIDE_PROFILES


def test_new_key_has_empty_binding(api_keys_db):
    uid = api_keys_db.create_user("alice", is_admin=False)
    _, rec = api_keys_db.create_key(uid, label="laptop")
    assert rec["config"] == {"direct": {}, "profiles": []}
    assert api_keys_db.get_key_config(rec["id"]) == {"direct": {}, "profiles": []}


def test_set_and_get_key_config(api_keys_db, profiles):
    uid = api_keys_db.create_user("alice", is_admin=False)
    _, rec = api_keys_db.create_key(uid)
    kid = rec["id"]
    binding = api_keys_db.set_key_config(uid, kid, {
        "overrides": {"BEAM_SIZE": 12}, "profiles": ["fast"], "locks": ["BEAM_SIZE"]})
    assert binding == {"direct": {"BEAM_SIZE": 12, "locks": ["BEAM_SIZE"]},
                       "profiles": ["fast"]}
    assert api_keys_db.get_key_config(kid) == binding


def test_set_key_config_rejects_unknown_profile(api_keys_db, profiles):
    uid = api_keys_db.create_user("alice", is_admin=False)
    _, rec = api_keys_db.create_key(uid)
    with pytest.raises(ValueError, match="unknown profile"):
        api_keys_db.set_key_config(uid, rec["id"], {"profiles": ["nope"]})


def test_set_key_config_rejects_bad_field(api_keys_db, profiles):
    uid = api_keys_db.create_user("alice", is_admin=False)
    _, rec = api_keys_db.create_key(uid)
    with pytest.raises(ValueError):
        api_keys_db.set_key_config(uid, rec["id"], {"overrides": {"BEAM_SIZE": 999}})


def test_set_key_config_wrong_user_404(api_keys_db, profiles):
    uid = api_keys_db.create_user("alice", is_admin=False)
    other = api_keys_db.create_user("bob", is_admin=False)
    _, rec = api_keys_db.create_key(uid)
    with pytest.raises(ValueError, match="not found"):
        api_keys_db.set_key_config(other, rec["id"], {"overrides": {"BEAM_SIZE": 5}})


def test_user_config_via_permissions(api_keys_db, profiles):
    uid = api_keys_db.create_user("alice", is_admin=False)
    api_keys_db.set_user_permissions(uid, {"pages": {}, "config": {
        "overrides": {"BEST_OF": 7}, "profiles": ["clinic-de"], "locks": []}})
    assert api_keys_db.get_user_config(uid) == {
        "direct": {"BEST_OF": 7}, "profiles": ["clinic-de"]}


def test_user_config_preserved_when_absent(api_keys_db, profiles):
    uid = api_keys_db.create_user("alice", is_admin=False)
    api_keys_db.set_user_permissions(uid, {"pages": {}, "config": {
        "overrides": {"BEST_OF": 7}, "profiles": ["clinic-de"]}})
    # A later save that only touches pages must not wipe the config.
    api_keys_db.set_user_permissions(uid, {"pages": {"quick_config": "own"}})
    assert api_keys_db.get_user_config(uid)["profiles"] == ["clinic-de"]


def test_user_config_cleared_by_empty(api_keys_db, profiles):
    uid = api_keys_db.create_user("alice", is_admin=False)
    api_keys_db.set_user_permissions(uid, {"pages": {}, "config": {
        "overrides": {"BEST_OF": 7}, "profiles": ["clinic-de"]}})
    api_keys_db.set_user_permissions(uid, {"pages": {}, "config": {
        "overrides": {}, "profiles": []}})
    assert api_keys_db.get_user_config(uid) == {"direct": {}, "profiles": []}


def test_config_column_migration_on_legacy_db(tmp_path):
    """A DB created before the feature (no api_keys.config column) gains the
    column on init_db, and existing rows read back with the empty binding."""
    db = str(tmp_path / "legacy.sqlite3")
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE users (id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE,"
        " is_admin INTEGER NOT NULL DEFAULT 0, created_ts REAL NOT NULL,"
        " revoked_ts REAL, permissions TEXT NOT NULL DEFAULT '{}');"
        "CREATE TABLE api_keys (id TEXT PRIMARY KEY, user_id TEXT NOT NULL,"
        " key_hash TEXT NOT NULL UNIQUE, key_prefix TEXT NOT NULL,"
        " key_last4 TEXT NOT NULL, label TEXT NOT NULL DEFAULT '',"
        " created_ts REAL NOT NULL, revoked_ts REAL, last_used_ts REAL);"
    )
    conn.execute("INSERT INTO users VALUES ('u1','alice',0,1.0,NULL,'{}')")
    conn.execute(
        "INSERT INTO api_keys VALUES"
        " ('k1','u1','h1','wk_aaaa','bbbb','laptop',1.0,NULL,NULL)")
    conn.commit()
    conn.close()

    try:
        ak.init_db(db)              # runs the additive migration
        cols = {r["name"] for r in ak._require_conn().execute(
            "PRAGMA table_info(api_keys)").fetchall()}
        assert "config" in cols
        assert ak.get_key("k1")["config"] == {"direct": {}, "profiles": []}
        # idempotent: a second init must not error
        ak.init_db(db)
    finally:
        try:
            ak._require_conn().close()
        except Exception:
            pass
        ak._conn = None
        ak._KEY_INDEX = {}
        ak._IS_LOCKED_DOWN = False
