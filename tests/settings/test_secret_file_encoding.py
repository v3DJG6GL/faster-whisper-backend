"""Secret *_FILE indirection copes with what Windows editors and shells write.

Windows PowerShell 5.1 `"key" > file` writes UTF-16 with a BOM, which a strict
UTF-8 read turned into an import-time UnicodeDecodeError (the server did not
start); `Set-Content -Encoding UTF8` writes a UTF-8 BOM, which .strip() keeps,
so the bootstrap admin key never matched what the operator typed.
"""

import importlib

import pytest

from faster_whisper_backend.settings import config

_VAR = "WHISPER_BOOTSTRAP_ADMIN_KEY"


def _reload_with_secret_file(monkeypatch, path):
    # delenv records the var, so undo() also clears the value the prepass
    # writes straight into os.environ.
    monkeypatch.delenv(_VAR, raising=False)
    monkeypatch.setenv(_VAR + "_FILE", str(path))
    importlib.reload(config)


@pytest.mark.parametrize("data", [
    "sk-from-file\r\n".encode("utf-16"),          # BOM + native byte order
    b"\xfe\xff" + "sk-from-file".encode("utf-16-be"),
    b"\xef\xbb\xbfsk-from-file\n",                 # UTF-8 with BOM
])
def test_bom_files_yield_the_bare_secret(monkeypatch, tmp_path, data):
    secret = tmp_path / "key"
    secret.write_bytes(data)
    try:
        _reload_with_secret_file(monkeypatch, secret)
        assert config.BOOTSTRAP_ADMIN_KEY == "sk-from-file"
    finally:
        monkeypatch.undo()
        importlib.reload(config)


@pytest.mark.parametrize("data", [
    b"\xffsk-not-utf8",                            # undecodable
    b"sk\x00from-file",                            # NUL: os.environ refuses it
])
def test_unusable_file_warns_instead_of_failing_import(monkeypatch, tmp_path, data):
    secret = tmp_path / "key"
    secret.write_bytes(data)
    try:
        _reload_with_secret_file(monkeypatch, secret)
        assert any(m.startswith(_VAR + "_FILE=") and "could not be read" in m
                   for m in config._ENV_WARNINGS), config._ENV_WARNINGS
        assert "sk" not in (config.BOOTSTRAP_ADMIN_KEY or "")
    finally:
        monkeypatch.undo()
        importlib.reload(config)
