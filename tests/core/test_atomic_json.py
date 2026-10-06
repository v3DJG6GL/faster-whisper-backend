"""core/atomic_json.atomic_write_json: tempfile + fsync + os.replace, with the
Windows sharing-violation retry loop. The save lock around it is exercised
through config_store.save_overrides in tests/settings/test_config_store.py."""

import json
import os

import pytest

from faster_whisper_backend.core import atomic_json


def test_atomic_write_unicode(tmp_path):
    p = str(tmp_path / "u.json")
    atomic_json.atomic_write_json({"k": "Müller"}, p, sort_keys=True, tmp_prefix=".t")
    assert json.loads(open(p, encoding="utf-8").read())["k"] == "Müller"
    # ensure_ascii=False keeps the literal char on disk.
    assert "Müller" in open(p, encoding="utf-8").read()


def test_atomic_write_retries_then_succeeds(tmp_path, monkeypatch):
    p = str(tmp_path / "r.json")
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("AV lock")
        return real_replace(src, dst)

    monkeypatch.setattr(atomic_json.os, "replace", flaky)
    monkeypatch.setattr(atomic_json.time, "sleep", lambda *_: None)
    atomic_json.atomic_write_json({"ok": 1}, p, sort_keys=True, tmp_prefix=".t")
    assert calls["n"] == 3
    assert json.loads(open(p, encoding="utf-8").read()) == {"ok": 1}


def test_atomic_write_gives_up_after_retries(tmp_path, monkeypatch):
    p = str(tmp_path / "x.json")

    def always_fail(src, dst):
        raise PermissionError("locked")

    monkeypatch.setattr(atomic_json.os, "replace", always_fail)
    monkeypatch.setattr(atomic_json.time, "sleep", lambda *_: None)
    with pytest.raises(PermissionError):
        atomic_json.atomic_write_json({"ok": 1}, p, sort_keys=True, tmp_prefix=".t")
    # The temp file is cleaned up in finally; only the (untouched) dir remains.
    leftovers = [f for f in os.listdir(tmp_path) if f != "x.json"]
    assert leftovers == []
