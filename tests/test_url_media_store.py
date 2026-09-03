"""Unit tests for url_media_store.py (retention of URL-downloaded audio)."""

from __future__ import annotations

import os

import pytest

import url_media_store as ums


@pytest.fixture(autouse=True)
def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setattr(ums.cfg, "URL_MEDIA_DIR", str(tmp_path / "url_media"),
                        raising=False)
    monkeypatch.setattr(ums.cfg, "URL_MEDIA_TTL_S", 3600, raising=False)
    monkeypatch.setattr(ums.cfg, "URL_MEDIA_MAX_BYTES", 10_000, raising=False)
    ums.startup_reset()
    yield
    ums._REG.clear()


def _make_src(tmp_path, name="dl.m4a", size=100):
    p = tmp_path / name
    p.write_bytes(b"x" * size)
    return str(p)


def test_register_resolve_roundtrip(tmp_path):
    mid = ums.register(_make_src(tmp_path), user_id="u1")
    assert mid and len(mid) == 32
    resolved = ums.resolve(mid, user_id="u1")
    assert resolved is not None
    path, ext = resolved
    assert ext == "m4a" and os.path.isfile(path)
    # the source was MOVED, not copied
    assert not os.path.exists(str(tmp_path / "dl.m4a"))


def test_unknown_ext_becomes_bin(tmp_path):
    mid = ums.register(_make_src(tmp_path, name="dl.weird"), user_id=None)
    assert ums.resolve(mid, user_id=None)[1] == "bin"


def test_owner_mismatch_hidden(tmp_path):
    mid = ums.register(_make_src(tmp_path), user_id="owner")
    assert ums.resolve(mid, user_id="someone-else") is None
    # open-mode symmetry: a None on either side allows
    assert ums.resolve(mid, user_id=None) is not None


def test_ttl_expiry(tmp_path, monkeypatch):
    mid = ums.register(_make_src(tmp_path), user_id=None)
    entry_path = ums._REG[mid]["path"]
    monkeypatch.setattr(ums.cfg, "URL_MEDIA_TTL_S", 0, raising=False)
    assert ums.resolve(mid, user_id=None) is None
    assert not os.path.exists(entry_path)  # expiry deletes the file


def test_sweep_ttl_and_unknown_id(tmp_path, monkeypatch):
    mid = ums.register(_make_src(tmp_path), user_id=None)
    monkeypatch.setattr(ums.cfg, "URL_MEDIA_TTL_S", 0, raising=False)
    ums.sweep()
    assert mid not in ums._REG
    assert ums.resolve("f" * 32, user_id=None) is None


def test_lru_eviction_over_byte_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(ums.cfg, "URL_MEDIA_MAX_BYTES", 250, raising=False)
    first = ums.register(_make_src(tmp_path, "a.m4a", size=100), user_id=None)
    second = ums.register(_make_src(tmp_path, "b.m4a", size=100), user_id=None)
    third = ums.register(_make_src(tmp_path, "c.m4a", size=100), user_id=None)
    # 300 bytes > 250 cap → the oldest goes
    assert ums.resolve(first, user_id=None) is None
    assert ums.resolve(second, user_id=None) is not None
    assert ums.resolve(third, user_id=None) is not None


def test_register_oversized_file_returns_none_and_keeps_older(tmp_path,
                                                              monkeypatch):
    monkeypatch.setattr(ums.cfg, "URL_MEDIA_MAX_BYTES", 50, raising=False)
    first = ums.register(_make_src(tmp_path, "a.m4a", size=30), user_id=None)
    assert first is not None
    # 100 bytes alone busts the 50-byte cap: no dead id may be advertised,
    # and the older retained file must survive.
    assert ums.register(_make_src(tmp_path, "b.m4a", size=100),
                        user_id=None) is None
    assert ums.resolve(first, user_id=None) is not None
    assert set(ums._REG) == {first}
    # the oversized file itself was unlinked, not left behind
    assert sorted(os.listdir(ums._dir())) == [
        os.path.basename(ums._REG[first]["path"])]


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")
def test_register_creates_dir_0700(tmp_path):
    # Runtime enable: register() may be the FIRST thing to create the dir
    # (no lifespan startup_reset ran) — it must still end up 0700.
    import shutil
    shutil.rmtree(ums._dir())
    mid = ums.register(_make_src(tmp_path), user_id=None)
    assert mid is not None
    assert os.stat(ums._dir()).st_mode & 0o777 == 0o700


def test_sweep_reclaims_file_whose_unlink_failed(tmp_path, monkeypatch):
    mid = ums.register(_make_src(tmp_path), user_id=None)
    retained = ums._REG[mid]["path"]

    def _no_unlink(path):
        raise PermissionError("file busy (streaming FileResponse)")

    with monkeypatch.context() as m:
        m.setattr(ums.os, "unlink", _no_unlink)
        ums._drop(mid)
    assert mid not in ums._REG and os.path.exists(retained)
    # too fresh for the orphan scan yet — a concurrent register()'s move
    # must never be mistaken for an orphan
    ums.sweep()
    assert os.path.exists(retained)
    import time
    old = time.time() - 3600
    os.utime(retained, (old, old))
    ums.sweep()
    assert not os.path.exists(retained)


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")
def test_pipeline_copy_fallback_is_0600(tmp_path, monkeypatch):
    src = _make_src(tmp_path, size=64)

    def _no_link(a, b, **kw):
        raise OSError("cross-device link")

    monkeypatch.setattr(ums.os, "link", _no_link)
    copy = ums.make_pipeline_copy(src)
    assert copy is not None
    with open(copy, "rb") as f:
        assert f.read() == b"x" * 64
    assert os.stat(copy).st_mode & 0o777 == 0o600
    os.unlink(copy)


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes only")
def test_pipeline_copy_link_path_is_0600(tmp_path, monkeypatch):
    # The os.link fast path inherits the source's mode: a 0644 download must
    # still end up 0600 in the shared tempdir.
    src = _make_src(tmp_path, size=64)
    os.chmod(src, 0o644)
    monkeypatch.setattr(ums.tempfile, "gettempdir", lambda: str(tmp_path))
    copy = ums.make_pipeline_copy(src)
    assert copy is not None
    assert os.stat(copy).st_nlink == 2  # really the link path, not the copy
    assert os.stat(copy).st_mode & 0o777 == 0o600
    os.unlink(copy)


def test_register_refreshes_mtime_for_orphan_guard(tmp_path):
    # The move preserves the download's mtime; the orphan scan must see the
    # PLACEMENT time, or a slow cross-device move lands already "old".
    import time
    src = _make_src(tmp_path)
    old = time.time() - 3600
    os.utime(src, (old, old))
    mid = ums.register(src, user_id=None)
    retained = ums._REG[mid]["path"]
    assert time.time() - os.path.getmtime(retained) < 60
    ums.sweep()
    assert os.path.exists(retained) and mid in ums._REG


def test_startup_reset_wipes(tmp_path):
    mid = ums.register(_make_src(tmp_path), user_id=None)
    retained = ums._REG[mid]["path"]
    ums.startup_reset()
    assert ums._REG == {}
    assert not os.path.exists(retained)
    assert os.path.isdir(ums._dir())


def test_pipeline_copy_is_independent(tmp_path):
    src = _make_src(tmp_path, size=64)
    copy = ums.make_pipeline_copy(src)
    assert copy and os.path.getsize(copy) == 64
    mid = ums.register(src, user_id=None)  # moves the original away
    # unlinking the pipeline copy must not touch the retained file
    os.unlink(copy)
    assert ums.resolve(mid, user_id=None) is not None


def test_expires_at_unix(tmp_path):
    mid = ums.register(_make_src(tmp_path), user_id=None)
    exp = ums.expires_at_unix(mid)
    import time
    assert exp is not None and exp > time.time()
    assert ums.expires_at_unix("f" * 32) is None
