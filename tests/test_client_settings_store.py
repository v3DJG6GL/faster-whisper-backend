"""Unit tests for client_settings_store — CAS semantics, caps, isolation."""

import pytest


def test_get_empty_returns_none(client_settings_store_db):
    store = client_settings_store_db
    assert store.get("u1") is None


def test_create_then_get(client_settings_store_db):
    store = client_settings_store_db
    ok, state = store.put("u1", {"theme": "dark"}, 0, device="laptop")
    assert ok is True
    assert state["version"] == 1
    assert state["blob"] == {"theme": "dark"}
    assert state["device"] == "laptop"
    got = store.get("u1")
    assert got["version"] == 1
    assert got["blob"] == {"theme": "dark"}


def test_double_create_conflicts(client_settings_store_db):
    store = client_settings_store_db
    ok, _ = store.put("u1", {"a": 1}, 0)
    assert ok
    ok, current = store.put("u1", {"b": 2}, 0)
    assert ok is False
    # 409 body state = what's actually stored, so the loser can merge.
    assert current["version"] == 1
    assert current["blob"] == {"a": 1}


def test_update_bumps_version_chain(client_settings_store_db):
    store = client_settings_store_db
    store.put("u1", {"n": 1}, 0)
    ok, state = store.put("u1", {"n": 2}, 1)
    assert ok and state["version"] == 2
    ok, state = store.put("u1", {"n": 3}, 2)
    assert ok and state["version"] == 3
    assert store.get("u1")["blob"] == {"n": 3}


def test_stale_version_conflicts_with_current(client_settings_store_db):
    store = client_settings_store_db
    store.put("u1", {"n": 1}, 0)
    store.put("u1", {"n": 2}, 1)  # now at version 2
    ok, current = store.put("u1", {"n": 99}, 1)  # stale base
    assert ok is False
    assert current["version"] == 2
    assert current["blob"] == {"n": 2}


def test_oversize_blob_rejected(client_settings_store_db):
    store = client_settings_store_db
    big = {"x": "a" * (store._CAP_BLOB + 100)}
    with pytest.raises(ValueError):
        store.put("u1", big, 0)
    assert store.get("u1") is None  # nothing landed


def test_device_truncated(client_settings_store_db):
    store = client_settings_store_db
    ok, state = store.put("u1", {}, 0, device="d" * 1000)
    assert ok
    assert len(state["device"]) == store._CAP_DEVICE


def test_updated_at_moves_forward(client_settings_store_db):
    store = client_settings_store_db
    store.put("u1", {"n": 1}, 0)
    t1 = store.get("u1")["updated_at"]
    store.put("u1", {"n": 2}, 1)
    t2 = store.get("u1")["updated_at"]
    assert t2 >= t1


def test_delete(client_settings_store_db):
    store = client_settings_store_db
    store.put("u1", {"n": 1}, 0)
    assert store.delete("u1") is True
    assert store.get("u1") is None
    assert store.delete("u1") is False
    # After delete, a device holding the old version conflicts (row gone).
    ok, current = store.put("u1", {"n": 2}, 1)
    assert ok is False
    assert current is None
    # Re-create from zero works.
    ok, state = store.put("u1", {"n": 2}, 0)
    assert ok and state["version"] == 1


def test_users_isolated(client_settings_store_db):
    store = client_settings_store_db
    store.put("u1", {"who": "one"}, 0)
    store.put("u2", {"who": "two"}, 0)
    assert store.get("u1")["blob"] == {"who": "one"}
    assert store.get("u2")["blob"] == {"who": "two"}
    store.delete("u1")
    assert store.get("u1") is None
    assert store.get("u2")["blob"] == {"who": "two"}


def test_profile_column_ready(client_settings_store_db):
    """v1 clients always use profile='' — prove the schema already keys
    (user_id, profile) independently so named sets can land additively."""
    store = client_settings_store_db
    store.put("u1", {"set": "default"}, 0)
    ok, state = store.put("u1", {"set": "alt"}, 0, profile="work")
    assert ok and state["version"] == 1
    assert store.get("u1")["blob"] == {"set": "default"}
    assert store.get("u1", "work")["blob"] == {"set": "alt"}
    store.delete("u1", "work")
    assert store.get("u1", "work") is None
    assert store.get("u1")["blob"] == {"set": "default"}


def test_list_meta_empty(client_settings_store_db):
    assert client_settings_store_db.list_meta() == []


def test_list_meta_rows_without_blob(client_settings_store_db):
    """The admin listing must carry metadata only — never blob contents —
    and `bytes` must be the stored UTF-8 size, not a character count."""
    store = client_settings_store_db
    store.put("u1", {"a": "ä"}, 0, device="laptop")   # 'ä' = 2 UTF-8 bytes
    store.put("u2", {"b": 2}, 0, profile="work")
    rows = store.list_meta()
    assert len(rows) == 2
    by_key = {(r["user_id"], r["profile"]): r for r in rows}
    r1 = by_key[("u1", "")]
    assert r1["version"] == 1
    assert r1["device"] == "laptop"
    assert r1["updated_at"] is not None
    assert r1["bytes"] == len('{"a":"ä"}'.encode("utf-8"))
    assert "blob" not in r1
    assert by_key[("u2", "work")]["profile"] == "work"


def test_force_put_creates_then_bumps(client_settings_store_db):
    """The WebUI import path: no base_version, always lands, always bumps —
    so a device holding the old version conflicts on its next CAS push."""
    store = client_settings_store_db
    state = store.force_put("u1", {"n": 1}, device="WebUI import")
    assert state["version"] == 1
    assert state["blob"] == {"n": 1}
    assert state["device"] == "WebUI import"

    store.put("u1", {"n": 2}, 1)                      # device write → v2
    state = store.force_put("u1", {"n": 99})          # admin import → v3
    assert state["version"] == 3
    assert state["blob"] == {"n": 99}

    # The device's stale CAS push now conflicts, surfacing the import.
    ok, current = store.put("u1", {"n": 3}, 2)
    assert ok is False
    assert current["version"] == 3
    assert current["blob"] == {"n": 99}


def test_force_put_oversize_rejected(client_settings_store_db):
    store = client_settings_store_db
    store.put("u1", {"n": 1}, 0)
    big = {"x": "a" * (store._CAP_BLOB + 100)}
    with pytest.raises(ValueError):
        store.force_put("u1", big)
    assert store.get("u1")["blob"] == {"n": 1}        # nothing clobbered


def test_force_put_truncates_device(client_settings_store_db):
    store = client_settings_store_db
    state = store.force_put("u1", {}, device="d" * 500)
    assert state["device"] == "d" * store._CAP_DEVICE
