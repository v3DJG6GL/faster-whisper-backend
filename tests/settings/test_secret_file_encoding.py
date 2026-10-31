"""Secret *_FILE indirection and the .env loader cope with what Windows editors
and shells write.

Windows PowerShell 5.1 `"key" > file` writes UTF-16 with a BOM, which a strict
UTF-8 read turned into an import-time UnicodeDecodeError (the server did not
start); `Set-Content -Encoding UTF8` writes a UTF-8 BOM, which .strip() keeps,
so the bootstrap admin key never matched what the operator typed.
"""

import importlib
import os

import pytest

from faster_whisper_backend.settings import config

_VAR = "WHISPER_BOOTSTRAP_ADMIN_KEY"


def _reload_with_secret_file(monkeypatch, path):
    # setenv (not delenv: that records nothing for an absent var) so undo()
    # also clears the value the prepass writes straight into os.environ; ""
    # is falsy, so the prepass still reads the file.
    monkeypatch.setenv(_VAR, "")
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


def test_helper_leaves_no_secret_behind(tmp_path):
    secret = tmp_path / "key"
    secret.write_bytes(b"sk-from-file\n")
    before = os.environ.get(_VAR)
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv(_VAR, raising=False)
        try:
            with pytest.MonkeyPatch.context() as inner:
                _reload_with_secret_file(inner, secret)
                assert config.BOOTSTRAP_ADMIN_KEY == "sk-from-file"
            assert _VAR not in os.environ
        finally:
            importlib.reload(config)
    assert os.environ.get(_VAR) == before


@pytest.mark.parametrize("data, applied", [
    # PowerShell 5.1 `>`: UTF-16 with a BOM is decoded and applied.
    ("WHISPER_BEAM_SIZE=3\r\nWHISPER_DEFAULT_PROMPT=Grüße\r\n".encode("utf-16"),
     True),
    # An "ANSI" (cp1252) save is not guessed at: skipped with a warning.
    ("WHISPER_BEAM_SIZE=3\nWHISPER_DEFAULT_PROMPT=Grüße\n".encode("cp1252"),
     False),
], ids=["utf16-bom", "cp1252"])
def test_a_non_utf8_dotenv_does_not_fail_import(tmp_path, data, applied):
    """dotenv reads .env as strict UTF-8: either file crashed `import config`
    with a UnicodeDecodeError naming neither .env nor the fix."""
    import json
    import subprocess
    import sys
    from faster_whisper_backend.paths import REPO_ROOT
    (tmp_path / ".env").write_bytes(data)
    # Under `python -c` dotenv searches from the cwd, so tmp_path's .env is
    # the one loaded; no inherited WHISPER_* may shadow its keys.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("WHISPER_") and k != "PYTHON_DOTENV_DISABLED"}
    env.update(PYTHONPATH=str(REPO_ROOT), WHISPER_DATA_DIR=str(tmp_path / "data"))
    out = subprocess.run(
        [sys.executable, "-c",
         "import json; from faster_whisper_backend.settings import config as c;"
         "print(json.dumps([c.BEAM_SIZE, c.DEFAULT_PROMPT, c._ENV_WARNINGS]))"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    beam, prompt, warnings = json.loads(out.stdout.strip().splitlines()[-1])
    skipped = [w for w in warnings if "re-save it as UTF-8" in w]
    if applied:
        assert (beam, prompt, skipped) == (3, "Grüße", [])
    else:
        assert beam != 3 and prompt != "Grüße"
        assert len(skipped) == 1 and str(tmp_path / ".env") in skipped[0]
