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


def _import_config_from(tmp_path, data, expr="[c.BEAM_SIZE, c.DEFAULT_PROMPT]",
                        **env_vars):
    """Import config in a fresh interpreter whose .env holds `data`; print
    `expr` (over the module `c`) as JSON."""
    import subprocess
    import sys
    from faster_whisper_backend.paths import REPO_ROOT
    (tmp_path / ".env").write_bytes(data)
    # Under `python -c` dotenv searches from the cwd, so tmp_path's .env is
    # the one loaded; no inherited WHISPER_* may shadow its keys.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("WHISPER_") and k != "PYTHON_DOTENV_DISABLED"}
    env.update(PYTHONPATH=str(REPO_ROOT), WHISPER_DATA_DIR=str(tmp_path / "data"),
               **env_vars)
    return subprocess.run(
        [sys.executable, "-c",
         "import json; from faster_whisper_backend.settings import config as c;"
         f"print(json.dumps({expr}))"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)


_DOTENV = "WHISPER_BEAM_SIZE=3\r\nWHISPER_DEFAULT_PROMPT=Grüße\r\n"


@pytest.mark.parametrize("data", [
    _DOTENV.encode("utf-16"),       # PowerShell 5.1 `>`: UTF-16 with a BOM
    _DOTENV.encode("utf-32"),       # `Out-File -Encoding utf32`: FF FE 00 00
    b"\xef\xbb\xbf" + _DOTENV.encode(),  # UTF-8 BOM: not glued to the 1st key
], ids=["utf16-bom", "utf32-bom", "utf8-bom"])
def test_a_bom_dotenv_is_decoded_and_applied(tmp_path, data):
    """dotenv reads .env as strict UTF-8: a UTF-16 file crashed `import
    config` with a UnicodeDecodeError naming neither .env nor the fix, and a
    UTF-32 one (whose FF FE prefix looks like UTF-16) with 'embedded null
    byte'."""
    import json
    out = _import_config_from(tmp_path, data)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == [3, "Grüße"]


@pytest.mark.parametrize("data", [
    # An "ANSI" (cp1252) save is not guessed at.
    _DOTENV.encode("cp1252"),
    # PowerShell 5.1 `>>` appends BOM-less UTF-16LE to a UTF-8 file: valid
    # UTF-8 with NULs, which os.environ refuses after earlier keys were set.
    b"WHISPER_BEAM_SIZE=3\n" + "WHISPER_DEFAULT_PROMPT=x\r\n".encode("utf-16-le"),
    b"WHISPER_BEAM_SIZE=3\nWHISPER_DEFAULT_PROMPT=a\x00b\n",
], ids=["cp1252", "utf16le-append", "utf8-nul"])
def test_an_unusable_dotenv_stops_import_with_the_fix(tmp_path, data):
    """Skipping the file started the server on the default DATA_DIR with
    empty stores and every security pin of .env dropped; a NUL crashed with
    a bare 'embedded null byte'. Now the import stops naming .env and the
    fix."""
    out = _import_config_from(tmp_path, data)
    assert out.returncode != 0
    assert f"{tmp_path / '.env'} is not valid UTF-8 text" in out.stderr, out.stderr
    assert "re-save it as UTF-8" in out.stderr
    assert "embedded null byte" not in out.stderr


def test_env_from_dotenv_holds_exactly_the_names_dotenv_set(tmp_path):
    """The self-restart drops these from the exec'd environment so the new
    process reads .env again; a real env var that .env also names stays out
    (it is the environment's), an alias copied from an old name .env set is
    in (it would otherwise outlive that line's removal)."""
    import json
    out = _import_config_from(
        tmp_path,
        b"WHISPER_BEAM_SIZE=3\nWHISPER_DEFAULT_PROMPT=from-dotenv\n"
        b"WHISPER_USE_AUTH_TOKEN=hf_old\n",
        expr="[c.DEFAULT_PROMPT, sorted(c._ENV_FROM_DOTENV)]",
        WHISPER_DEFAULT_PROMPT="real")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == [
        "real", ["WHISPER_BEAM_SIZE", "WHISPER_HF_TOKEN", "WHISPER_USE_AUTH_TOKEN"]]


def test_env_from_dotenv_holds_an_alias_copied_onto_a_blank_new_name(tmp_path):
    """alias_env copies an old .env name onto a new name the environment holds
    blank (a templated compose line). Left out, the copy outlived the .env
    line's edit or removal on Restart."""
    import json
    out = _import_config_from(
        tmp_path, b"WHISPER_SESSION_TTL_SECONDS=1234\n",
        expr="[c.SESSION_TTL_S, sorted(c._ENV_FROM_DOTENV)]",
        WHISPER_SESSION_TTL_S="")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == [
        1234, ["WHISPER_SESSION_TTL_S", "WHISPER_SESSION_TTL_SECONDS"]]


def test_a_crlf_dotenv_multiline_value_keeps_no_carriage_return(tmp_path):
    """The decoded text went to dotenv without the newline translation of a
    text-mode open, so a quoted multi-line value from a Windows editor kept
    its "\\r"."""
    import json
    out = _import_config_from(
        tmp_path, b'WHISPER_DEFAULT_PROMPT="multi\r\nline"\r\n',
        expr="c.DEFAULT_PROMPT")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == "multi\nline"
