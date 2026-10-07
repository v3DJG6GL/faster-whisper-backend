"""Tests for api_keys_store — the security-sensitive identity layer.

Covers key hashing/generation, the O(1) lookup + last-used debounce, user/key
CRUD, the atomic last-admin guard (including a threaded concurrency test),
lockdown state transitions, and the permission model.
"""

import logging
import sqlite3
import threading

import pytest

from faster_whisper_backend.auth import api_keys_store


# ---------------------------------------------------------------------------
# Hashing / generation (pure)
# ---------------------------------------------------------------------------

def test_hash_key_deterministic_hex():
    from faster_whisper_backend.auth import api_keys_store as ak
    h1 = ak.hash_key("wk_abc")
    h2 = ak.hash_key("wk_abc")
    assert h1 == h2 and len(h1) == 64
    int(h1, 16)  # valid hex
    assert ak.hash_key("wk_abc") != ak.hash_key("wk_abd")


def test_hash_key_unicode_safe():
    from faster_whisper_backend.auth import api_keys_store as ak
    assert len(ak.hash_key("schlüssel-Ω")) == 64


def test_generate_raw_key_shape_and_uniqueness():
    from faster_whisper_backend.auth import api_keys_store as ak
    keys = {ak.generate_raw_key() for _ in range(200)}
    assert len(keys) == 200  # all unique
    for k in list(keys)[:5]:
        assert k.startswith(ak.KEY_PREFIX)
        assert len(k) == len(ak.KEY_PREFIX) + 43


def test_split_display_parts():
    from faster_whisper_backend.auth import api_keys_store as ak
    prefix, last4 = ak._split_display_parts("wk_abcdef1234567890wxyz")
    assert prefix == "wk_abcde" and last4 == "wxyz"


# ---------------------------------------------------------------------------
# create_user
# ---------------------------------------------------------------------------

def test_create_user_and_duplicate(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("alice", is_admin=False)
    assert ak.get_user(uid)["username"] == "alice"
    with pytest.raises(ValueError):
        ak.create_user("alice", is_admin=False)  # duplicate


def test_create_user_blank_and_too_long(api_keys_db):
    ak = api_keys_db
    with pytest.raises(ValueError):
        ak.create_user("   ", is_admin=False)
    with pytest.raises(ValueError):
        ak.create_user("x" * 129, is_admin=False)


def test_create_user_default_perms(api_keys_db):
    ak = api_keys_db
    nonadmin = ak.create_user("bob", is_admin=False)
    admin = ak.create_user("root", is_admin=True)
    assert ak.get_user_permissions(nonadmin)["pages"]["quick_config"] == "own"
    assert ak.get_user_permissions(admin) == {}  # admins bypass policy


# ---------------------------------------------------------------------------
# Lockdown transitions
# ---------------------------------------------------------------------------

def test_lockdown_transitions(api_keys_db):
    ak = api_keys_db
    assert ak.is_locked_down() is False                 # fresh DB -> open
    uid = ak.create_user("root", is_admin=True)
    assert ak.is_locked_down() is False                 # admin user, no key yet
    _, rec = ak.create_key(uid)
    assert ak.is_locked_down() is True                  # active admin key -> locked
    # Add a second admin key, then revoke one -> still locked.
    _, rec2 = ak.create_key(uid)
    ak.revoke_key(rec["id"])
    assert ak.is_locked_down() is True
    # Revoking the last admin key is blocked by the guard, so stays locked.
    with pytest.raises(ak.LastAdminError):
        ak.revoke_key(rec2["id"])


# ---------------------------------------------------------------------------
# lookup_by_raw_key + debounce
# ---------------------------------------------------------------------------

def test_lookup_hit_miss_and_falsy(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    raw, rec = ak.create_key(uid, label="my-laptop")
    got = ak.lookup_by_raw_key(raw)
    assert got["user_id"] == uid and got["key_id"] == rec["id"]
    assert got["is_admin"] is False
    assert got["key_label"] == "my-laptop"   # cached for the per-request log block
    assert ak.lookup_by_raw_key("wk_nope") is None
    assert ak.lookup_by_raw_key("") is None


def test_last_used_debounce(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    raw, rec = ak.create_key(uid)
    ak.lookup_by_raw_key(raw)
    t1 = ak.get_key(rec["id"])["last_used_ts"]
    assert t1 is not None
    # Second lookup within the 60s window must NOT write again.
    ak.lookup_by_raw_key(raw)
    assert ak.get_key(rec["id"])["last_used_ts"] == t1


def test_last_used_by_user_batched_max(api_keys_db):
    ak = api_keys_db
    conn = ak._require_conn()
    # User A: two keys — the newer use wins the MAX.
    a = ak.create_user("a", is_admin=False)
    _, ka1 = ak.create_key(a)
    _, ka2 = ak.create_key(a)
    conn.execute("UPDATE api_keys SET last_used_ts=? WHERE id=?", (100.0, ka1["id"]))
    conn.execute("UPDATE api_keys SET last_used_ts=? WHERE id=?", (250.0, ka2["id"]))
    # User B: a key that was never used -> absent (NULL last_used_ts).
    b = ak.create_user("b", is_admin=False)
    ak.create_key(b)
    # User C: a more-recently-used key that gets revoked is excluded; only the
    # remaining active key's use counts (matches the "N active keys" framing).
    c = ak.create_user("c", is_admin=False)
    _, kc1 = ak.create_key(c)
    _, kc2 = ak.create_key(c)
    conn.execute("UPDATE api_keys SET last_used_ts=? WHERE id=?", (300.0, kc1["id"]))
    conn.execute("UPDATE api_keys SET last_used_ts=? WHERE id=?", (500.0, kc2["id"]))
    ak.revoke_key(kc2["id"])

    m = ak.last_used_by_user()
    assert m[a] == 250.0
    assert b not in m            # never used -> absent, caller renders "—"
    assert m[c] == 300.0         # 500.0 belonged to a now-revoked key


def test_rename_profile_refs_cascades_lists_allowlist_and_wildcard(api_keys_db):
    import json
    ak = api_keys_db
    conn = ak._require_conn()
    # User binding: 'old' appears in BOTH the ordered profiles list and the
    # allowlist; an unrelated profile and the '*' wildcard must survive untouched.
    u = ak.create_user("u", is_admin=False)
    conn.execute("UPDATE users SET permissions=? WHERE id=?", (json.dumps({
        "pages": {}, "config": {"direct": {}, "profiles": ["old", "keep"],
                                "allowed_override_profiles": ["old", "*"]}}), u))
    # Two key bindings — one active, one revoked (migrated all the same so no
    # dangling references survive anywhere).
    _, k = ak.create_key(u)
    conn.execute("UPDATE api_keys SET config=? WHERE id=?",
                 (json.dumps({"direct": {}, "profiles": ["old"]}), k["id"]))
    _, kr = ak.create_key(u)
    conn.execute("UPDATE api_keys SET config=? WHERE id=?",
                 (json.dumps({"direct": {}, "profiles": ["old"]}), kr["id"]))
    ak.revoke_key(kr["id"])

    assert ak.rename_profile_refs("old", "new") == 3   # user row + 2 key rows

    uc = ak.get_user_config(u)
    assert uc["profiles"] == ["new", "keep"]
    assert uc["allowed_override_profiles"] == ["new", "*"]   # wildcard preserved
    assert ak.get_key_config(k["id"])["profiles"] == ["new"]
    # The revoked key is filtered by get_key_config, so read it raw to confirm
    # its stored binding migrated too.
    raw = conn.execute("SELECT config FROM api_keys WHERE id=?",
                        (kr["id"],)).fetchone()["config"]
    assert json.loads(raw)["profiles"] == ["new"]

    # No-op rename (same name) and an absent source name touch nothing.
    assert ak.rename_profile_refs("new", "new") == 0
    assert ak.rename_profile_refs("ghost", "x") == 0


def test_rename_profile_refs_is_all_or_nothing(api_keys_db, monkeypatch):
    """The overrides route rolls the profile rename back when the cascade
    raises; on the autocommit connection a mid-scan error used to leave the
    rows already rewritten pointing at a profile name that no longer exists."""
    import json
    ak = api_keys_db
    conn = ak._require_conn()
    u = ak.create_user("u", is_admin=False)
    conn.execute("UPDATE users SET permissions=? WHERE id=?", (json.dumps({
        "pages": {}, "config": {"direct": {}, "profiles": ["clinic-de"]}}), u))
    _, k = ak.create_key(u)
    conn.execute("UPDATE api_keys SET config=? WHERE id=?",
                 (json.dumps({"direct": {}, "profiles": ["clinic-de"]}), k["id"]))
    real = ak._rewrite_profile_in_binding
    calls = []

    def _flaky(binding, old, new):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("disk full")
        return real(binding, old, new)
    monkeypatch.setattr(ak, "_rewrite_profile_in_binding", _flaky)
    with pytest.raises(RuntimeError):
        ak.rename_profile_refs("clinic-de", "clinic-deutsch")
    assert len(calls) == 2           # the user row WAS rewritten before the error
    assert ak.get_user_config(u)["profiles"] == ["clinic-de"]    # ...and rolled back
    assert ak.get_key_config(k["id"])["profiles"] == ["clinic-de"]
    assert not conn.in_transaction
    # The connection is usable again (no transaction left open).
    monkeypatch.setattr(ak, "_rewrite_profile_in_binding", real)
    assert ak.rename_profile_refs("clinic-de", "clinic-deutsch") == 2


def test_rename_profile_refs_keeps_the_real_error_after_sqlite_rolled_back(
        api_keys_db, monkeypatch):
    """SQLite rolls the transaction back by itself on SQLITE_FULL / IOERR; a
    bare ROLLBACK then raised "no transaction is active" and replaced the
    real error in the overrides route's 500 detail."""
    import json
    ak = api_keys_db
    conn = ak._require_conn()
    u = ak.create_user("u", is_admin=False)
    conn.execute("UPDATE users SET permissions=? WHERE id=?", (json.dumps({
        "pages": {}, "config": {"direct": {}, "profiles": ["clinic-de"]}}), u))

    def _full(binding, old, new):
        conn.execute("ROLLBACK")      # what SQLite does on its own
        raise sqlite3.OperationalError("database or disk is full")
    monkeypatch.setattr(ak, "_rewrite_profile_in_binding", _full)
    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        ak.rename_profile_refs("clinic-de", "clinic-deutsch")
    assert not conn.in_transaction
    assert ak.get_user_config(u)["profiles"] == ["clinic-de"]


# ---------------------------------------------------------------------------
# create_key validation
# ---------------------------------------------------------------------------

def test_create_key_user_missing(api_keys_db):
    ak = api_keys_db
    with pytest.raises(ValueError):
        ak.create_key("nonexistent")


def test_create_key_label_too_long(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    with pytest.raises(ValueError):
        ak.create_key(uid, label="x" * 129)


def test_create_key_returns_prefixed_raw(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    raw, rec = ak.create_key(uid, label="laptop")
    assert raw.startswith("wk_")
    assert rec["label"] == "laptop" and rec["revoked_ts"] is None


# ---------------------------------------------------------------------------
# Last-admin guard
# ---------------------------------------------------------------------------

def test_revoke_last_admin_key_blocked(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("root", is_admin=True)
    _, rec = ak.create_key(uid)
    with pytest.raises(ak.LastAdminError):
        ak.revoke_key(rec["id"])


def test_revoke_admin_key_ok_when_second_exists(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("root", is_admin=True)
    _, r1 = ak.create_key(uid)
    _, r2 = ak.create_key(uid)
    ak.revoke_key(r1["id"])  # second admin key remains -> allowed
    assert ak.get_key(r1["id"])["revoked_ts"] is not None


def test_revoke_last_admin_user_blocked(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("root", is_admin=True)
    ak.create_key(uid)
    with pytest.raises(ak.LastAdminError):
        ak.revoke_user(uid)


def test_revoke_nonadmin_unguarded(api_keys_db):
    ak = api_keys_db
    admin = ak.create_user("root", is_admin=True)
    ak.create_key(admin)
    u = ak.create_user("bob", is_admin=False)
    ak.create_key(u)
    ak.revoke_user(u)  # no guard for non-admins
    assert ak.get_user(u)["revoked_ts"] is not None


def test_concurrent_reads_are_thread_safe(api_keys_db):
    # Regression: get_user() & friends read the single shared sqlite3
    # connection. Without serializing every read under _lock, concurrent
    # auth lookups from FastAPI threadpool workers raced on one connection
    # object → `sqlite3.InterfaceError: bad parameter or other API misuse`
    # and torn rows (e.g. created_ts read back as None). Hammer the readers
    # from many threads, interleaved with a writer, and assert clean results.
    ak = api_keys_db
    uid = ak.create_user("root", is_admin=True)
    raw, _ = ak.create_key(uid)
    other = ak.create_user("alice", is_admin=False)

    errors: list[BaseException] = []
    barrier = threading.Barrier(16)

    def reader():
        barrier.wait()
        try:
            for _ in range(80):
                rec = ak.get_user_record(uid)
                assert rec is not None and rec["user_id"] == uid
                u = ak.get_user(uid)
                assert isinstance(u["created_ts"], float)   # not None (torn read)
                ak.lookup_by_raw_key(raw)                    # bearer hot path
                ak.list_users()
                ak.list_keys(uid)
                ak.active_key_counts()
                ak.get_user_permissions(other)
                ak.get_usernames([uid, other, None])
        except BaseException as e:  # noqa: BLE001 — capture for the assert
            errors.append(e)

    def writer():
        barrier.wait()
        try:
            for i in range(40):
                ak.set_user_permissions(other, {"pages": {"captures": "own"}})
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(15)]
    threads.append(threading.Thread(target=writer))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent access raised: {errors[:3]}"


def test_concurrent_revoke_of_two_admin_keys_keeps_one(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("root", is_admin=True)
    _, r1 = ak.create_key(uid)
    _, r2 = ak.create_key(uid)
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def worker(kid):
        barrier.wait()
        try:
            ak.revoke_key(kid)
        except ak.LastAdminError as e:
            errors.append(e)

    t1 = threading.Thread(target=worker, args=(r1["id"],))
    t2 = threading.Thread(target=worker, args=(r2["id"],))
    t1.start(); t2.start(); t1.join(); t2.join()
    # Exactly one revoke is refused by the atomic guard -> one admin key left.
    assert len(errors) == 1
    assert ak.active_key_counts().get(uid, 0) == 1
    assert ak.is_locked_down() is True


# ---------------------------------------------------------------------------
# set_user_permissions
# ---------------------------------------------------------------------------

def test_set_permissions_validates_page_and_scope(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    with pytest.raises(ValueError):
        ak.set_user_permissions(uid, {"pages": {"nosuchpage": "all"}})
    with pytest.raises(ValueError):
        ak.set_user_permissions(uid, {"pages": {"captures": "sideways"}})


def test_set_permissions_access_only_rejects_own(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    # logs is access-only: none|all, not own.
    with pytest.raises(ValueError):
        ak.set_user_permissions(uid, {"pages": {"logs": "own"}})
    ak.set_user_permissions(uid, {"pages": {"logs": "all"}})


def test_stats_accepts_own_scope(api_keys_db):
    """stats left ACCESS_ONLY_PAGES in v2: "own" is a legal stored value."""
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    ak.set_user_permissions(uid, {"pages": {"stats": "own"}})
    assert ak.get_user_permissions(uid)["pages"]["stats"] == "own"
    assert "stats" in ak.SCOPED_PAGES
    assert "stats" not in ak.ACCESS_ONLY_PAGES


def test_binding_saves_keep_a_stored_slug_a_rules_edit_removed(
        api_keys_db, monkeypatch):
    """The binding drawer sends the whole overrides blob back: a rule slug the
    stored user / key binding already names (removed from the rules since)
    must not 422 an edit of another field, while a newly added unknown slug
    still does."""
    from faster_whisper_backend.settings import config_store
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    _raw, rec = ak.create_key(uid)
    slugs = {"x", "known"}
    monkeypatch.setattr(config_store, "_canonical_rule_slugs", lambda: set(slugs))
    body = {"overrides": {"PIPELINE_RULES_EXCLUDE": ["x"]}}
    ak.set_user_permissions(uid, {"config": body})
    ak.set_key_config(uid, rec["id"], body)
    slugs.discard("x")                                   # rule "x" deleted
    edit = {"overrides": {"PIPELINE_RULES_EXCLUDE": ["x"], "BEAM_SIZE": 3}}
    ak.set_user_permissions(uid, {"config": edit})
    ak.set_key_config(uid, rec["id"], edit)
    assert ak.get_user_config(uid)["direct"]["BEAM_SIZE"] == 3
    assert ak.get_key_config(rec["id"])["direct"]["BEAM_SIZE"] == 3
    added = {"overrides": {"PIPELINE_RULES_EXCLUDE": ["x", "typo"]}}
    with pytest.raises(ValueError, match="typo"):
        ak.set_user_permissions(uid, {"config": added})
    with pytest.raises(ValueError, match="typo"):
        ak.set_key_config(uid, rec["id"], added)


def test_scoped_and_access_only_pages_are_disjoint():
    # /settings/api-keys reports both sets; a page in both would be offered
    # "own" that set_user_permissions then rejects.
    assert not (api_keys_store.SCOPED_PAGES & api_keys_store.ACCESS_ONLY_PAGES)
    assert api_keys_store.SCOPED_PAGES | api_keys_store.ACCESS_ONLY_PAGES == set(
        api_keys_store.PAGES)


def test_default_nonadmin_perms_stats_own(api_keys_db):
    """A freshly created non-admin gets stats="own" (logs stays "none")."""
    ak = api_keys_db
    uid = ak.create_user("fresh", is_admin=False)
    pages = ak.get_user_permissions(uid)["pages"]
    assert pages["stats"] == "own"
    assert pages["logs"] == "none"


def test_existing_stats_all_survives_merge(api_keys_db):
    """Widening the allowed scopes must not touch rows that already hold
    "all": a later edit to another page merges around it."""
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    ak.set_user_permissions(uid, {"pages": {"stats": "all"}})
    ak.set_user_permissions(uid, {"pages": {"captures": "none"}})
    assert ak.get_user_permissions(uid)["pages"]["stats"] == "all"


def test_set_permissions_merge_preserves_untouched(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    ak.set_user_permissions(uid, {"pages": {"captures": "all"}})
    # Patch only reports; captures must survive the merge.
    clean = ak.set_user_permissions(uid, {"pages": {"reports": "all"}})
    assert clean["pages"]["captures"] == "all"
    assert clean["pages"]["reports"] == "all"


def test_set_permissions_normalises_tags(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("u", is_admin=False)
    clean = ak.set_user_permissions(uid, {"pages": {}, "quick_config_tags": ["B", "a", "a"]})
    assert clean["quick_config_tags"] == ["a", "b"]


def test_set_permissions_revoked_user_raises(api_keys_db):
    ak = api_keys_db
    admin = ak.create_user("root", is_admin=True)
    ak.create_key(admin)
    u = ak.create_user("bob", is_admin=False)
    ak.create_key(u)
    ak.revoke_user(u)
    with pytest.raises(ValueError):
        ak.set_user_permissions(u, {"pages": {"captures": "all"}})


# ---------------------------------------------------------------------------
# username / sentinel helpers
# ---------------------------------------------------------------------------

def test_username_helpers(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("alice", is_admin=False)
    assert ak.get_username(uid) == "alice"
    assert ak.get_username(None) is None
    assert ak.get_username("(open-mode)") is None
    assert ak.get_username("missing") is None
    batch = ak.get_usernames([uid, "(open-mode)", None, "missing"])
    assert batch[uid] == "alice" and batch["missing"] is None
    assert "(open-mode)" not in batch


def test_get_user_permissions_sentinel(api_keys_db):
    ak = api_keys_db
    assert ak.get_user_permissions("(open-mode)") == {}
    assert ak.get_user_permissions("missing") == {}


def test_open_mode_user_is_admin():
    from faster_whisper_backend.auth import api_keys_store as ak
    assert ak.OPEN_MODE_USER["is_admin"] is True


# ---------------------------------------------------------------------------
# update_key_label (rename)
# ---------------------------------------------------------------------------

def test_update_key_label_renames(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("renamer", is_admin=False)
    raw, rec = ak.create_key(uid, label="old")
    out = ak.update_key_label(rec["id"], "  fresh  ")
    assert out is not None and out["label"] == "fresh"  # trimmed
    rows = ak.list_keys(uid)
    assert any(k["id"] == rec["id"] and k["label"] == "fresh" for k in rows)
    # The relabel must refresh the cached auth-index label too (the log block
    # reads it from there), not just the DB row.
    assert ak.lookup_by_raw_key(raw)["key_label"] == "fresh"


def test_update_key_label_validates(api_keys_db):
    ak = api_keys_db
    uid = ak.create_user("badlabel", is_admin=False)
    _, rec = ak.create_key(uid, label="ok")
    for bad in ("", "   ", "x" * 129):
        with pytest.raises(ValueError):
            ak.update_key_label(rec["id"], bad)


def test_update_key_label_missing_returns_none(api_keys_db):
    ak = api_keys_db
    assert ak.update_key_label("does-not-exist", "x") is None


# ---------------------------------------------------------------------------
# WHISPER_BOOTSTRAP_ADMIN_KEY ingest (api_keys_store.bootstrap_admin_from_env)
# ---------------------------------------------------------------------------

_BOOTSTRAP_KEY = "bootstrap-key-with-enough-entropy-1234"


def test_bootstrap_admin_from_env_is_idempotent(api_keys_db):
    api_keys_store.bootstrap_admin_from_env(_BOOTSTRAP_KEY)
    assert api_keys_db.is_locked_down() is True
    # Re-running with the same live key is a no-op, not a duplicate insert.
    api_keys_store.bootstrap_admin_from_env(_BOOTSTRAP_KEY)
    h = api_keys_db.hash_key(_BOOTSTRAP_KEY)
    rows = api_keys_db._require_conn().execute(
        "SELECT COUNT(*) FROM api_keys WHERE key_hash = ?", (h,)).fetchone()[0]
    assert rows == 1


def test_bootstrap_admin_from_env_accepts_a_key_a_sibling_worker_registered(
        api_keys_db, tmp_path, caplog):
    # Every uvicorn worker runs the bootstrap at boot. A sibling that commits
    # the user + key after this worker's init_db built its index must not make
    # this worker refuse to start over a key that is in fact registered.
    import time
    import uuid
    db_path = api_keys_db._require_conn().execute(
        "PRAGMA database_list").fetchone()["file"]
    h = api_keys_db.hash_key(_BOOTSTRAP_KEY)
    sibling = sqlite3.connect(db_path)
    try:
        uid = uuid.uuid4().hex
        with sibling:
            sibling.execute(
                "INSERT INTO users (id, username, is_admin, created_ts,"
                " revoked_ts, permissions) VALUES (?,?,1,?,NULL,'{}')",
                (uid, "bootstrap-admin", time.time()))
            sibling.execute(
                "INSERT INTO api_keys (id, user_id, key_hash, key_prefix,"
                " key_last4, label, created_ts, revoked_ts, last_used_ts)"
                " VALUES (?,?,?,?,?,?,?,NULL,NULL)",
                (uuid.uuid4().hex, uid, h, "bootstra", "1234",
                 "bootstrap (env)", time.time()))
    finally:
        sibling.close()
    assert api_keys_db._KEY_INDEX.get(h) is None   # this worker's stale index

    caplog.set_level(logging.DEBUG, logger=api_keys_store.logger.name)
    api_keys_store.bootstrap_admin_from_env(_BOOTSTRAP_KEY)

    rows = api_keys_db._require_conn().execute(
        "SELECT COUNT(*) FROM api_keys WHERE key_hash = ?", (h,)).fetchone()[0]
    assert rows == 1
    assert api_keys_db.is_locked_down() is True
    # The sibling refresh before the live-hash check is what found the key:
    # without it the INSERT's IntegrityError fallback lands on the same rows.
    assert "bootstrap key already present" in caplog.text
    assert "registered by a sibling" not in caplog.text


def test_failed_reinit_clears_db_ready(api_keys_db, tmp_path):
    """A second init_db that raises partway (corrupt DB, read-only remount)
    must leave _DB_READY False so the stale _KEY_INDEX/_IS_LOCKED_DOWN caches
    fail closed instead of being trusted. sessions_store.init_db does the same
    (test_failed_reinit_fails_session_lookup_closed)."""
    assert api_keys_db._DB_READY is True
    uid = api_keys_db.create_user("admin", is_admin=True)
    raw, _rec = api_keys_db.create_key(uid)
    assert api_keys_db.lookup_by_raw_key(raw) is not None
    old = api_keys_db._conn
    garbage = tmp_path / "garbage.sqlite3"
    garbage.write_bytes(b"this is not a sqlite database")
    with pytest.raises(sqlite3.DatabaseError):   # "file is not a database"
        api_keys_db.init_db(str(garbage))
    assert api_keys_db._DB_READY is False
    # The previous DB's keys no longer authenticate from the stale index.
    assert api_keys_db.lookup_by_raw_key(raw) is None
    assert api_keys_db.is_locked_down() is True
    # Re-init must close the previous connection instead of leaking it (plus
    # its WAL/-shm handles) on every re-init.
    with pytest.raises(sqlite3.ProgrammingError):
        old.execute("SELECT 1")


def test_failed_reinit_fails_cookie_path_lookups_closed(api_keys_db, tmp_path,
                                                       monkeypatch):
    """The cookie path (get_user_record / touch_key_if_active) must fail
    closed during a failed re-init exactly like lookup_by_raw_key, instead of
    reading whatever DB _conn points at (or 500ing once it is None)."""
    uid = api_keys_db.create_user("admin", is_admin=True)
    _raw, rec = api_keys_db.create_key(uid)
    assert api_keys_db.get_user_record(uid) is not None
    assert api_keys_db.touch_key_if_active(rec["id"]) is True

    def _boom(conn):
        raise sqlite3.OperationalError("migration failed")
    monkeypatch.setattr(api_keys_db, "_ensure_columns", _boom)
    with pytest.raises(sqlite3.OperationalError):
        # Same file: _conn points at a DB that still holds the user and key.
        api_keys_db.init_db(str(tmp_path / "api_keys.sqlite3"))
    assert api_keys_db.get_user_record(uid) is None
    assert api_keys_db.touch_key_if_active(rec["id"]) is False


def test_parse_binding_migrates_tightened_direct_values():
    """A binding stored before the bundle validators tightened (guard words 1,
    a non-Whisper DEFAULT_LANGUAGE) must not 422 every later edit of the key
    or user on a field the admin never touched."""
    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.settings import config_store
    b = api_keys_store._parse_binding(
        {"direct": {"DEFAULT_LANGUAGE": "jp", "SEGMENT_HEAD_ECHO_MIN_WORDS": 1,
                    "locks": ["DEFAULT_LANGUAGE"]}, "profiles": []})
    assert b["direct"] == {"SEGMENT_HEAD_ECHO_MIN_WORDS": 0, "locks": []}
    direct = dict(b["direct"])
    locks = direct.pop("locks")
    config_store.validate_binding({"overrides": direct, "locks": locks})


def test_parse_binding_migrates_renamed_direct_keys_and_locks():
    # Bindings are stored as JSON and bypass config_store's key migration;
    # a pre-rename direct blob must come back under the current names so the
    # admin's next re-save is accepted instead of "Extra inputs are not
    # permitted".
    from faster_whisper_backend.auth import api_keys_store
    raw = '{"direct": {"SEGMENT_MAX_WORDS_PER_SEC": 3.0, "BEAM_SIZE": 2,' \
          ' "locks": ["SEGMENT_MAX_WORDS_PER_SEC", "BEAM_SIZE"]}, "profiles": []}'
    b = api_keys_store._parse_binding(raw)
    assert b["direct"] == {"SEGMENT_MAX_WORDS_PER_S": 3.0, "BEAM_SIZE": 2,
                           "locks": ["SEGMENT_MAX_WORDS_PER_S", "BEAM_SIZE"]}


def test_parse_binding_migrates_renamed_rule_slugs():
    # A binding stored before a RENAMED_RULES rename still names the old slug;
    # it must come back under the current one, or re-saving the binding
    # unchanged fails validate_binding's unknown-slug check.
    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.settings import config_store
    inc = api_keys_store._parse_binding(
        '{"direct": {"PIPELINE_RULES_INCLUDE": ["dictation-map"]}, "profiles": []}')
    assert inc["direct"]["PIPELINE_RULES_INCLUDE"] == ["de-dictation-map"]
    b = api_keys_store._parse_binding(
        '{"direct": {"PIPELINE_RULES_EXCLUDE": ["dictation-map"]}, "profiles": []}')
    assert b["direct"]["PIPELINE_RULES_EXCLUDE"] == ["de-dictation-map"]
    stored = config_store.validate_binding(
        {"overrides": {k: v for k, v in b["direct"].items() if k != "locks"},
         "profiles": b["profiles"]})
    assert stored["direct"]["PIPELINE_RULES_EXCLUDE"] == ["de-dictation-map"]
