"""build_info version resolution: env override > git describe > "unknown"."""

import importlib
import subprocess

import pytest

from faster_whisper_backend import build_info


@pytest.fixture(autouse=True)
def _keep_process_identity():
    """importlib.reload re-mints BOOT_ID / STARTED_AT / STARTED_UTC, but
    transcription/catalog_routes imports BOOT_ID by value at import time and
    serves it on /v1/models (main.py logs it at startup) — a reload here
    would leave them diverged for the rest of the session. Reinstate the
    originals after each test's reloads."""
    saved = (build_info.BOOT_ID, build_info.STARTED_AT, build_info.STARTED_UTC)
    yield
    build_info.BOOT_ID, build_info.STARTED_AT, build_info.STARTED_UTC = saved


def test_resolves_to_nonempty_string():
    # In any environment (CI, checkout, tarball) the constant must be a
    # usable display string — the exact value depends on the build.
    assert isinstance(build_info.APP_VERSION, str) and build_info.APP_VERSION
    assert build_info.SERVER_NAME == "faster-whisper-backend"


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("WHISPER_BUILD_VERSION", "v9.9.9-test")
    try:
        assert importlib.reload(build_info).APP_VERSION == "v9.9.9-test"
    finally:
        # Restore the ORIGINAL env first (a baked-in WHISPER_BUILD_VERSION
        # must come back), then reload so build_info matches it again.
        monkeypatch.undo()
        importlib.reload(build_info)


def test_no_git_falls_back_to_unknown(monkeypatch):
    monkeypatch.delenv("WHISPER_BUILD_VERSION", raising=False)

    def _no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", _no_git)
    try:
        assert importlib.reload(build_info).APP_VERSION == "unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(build_info)


# --- engine_versions: optional llama-cpp-python part ----------------------

def _stub_pkg_version(monkeypatch, llama):
    """Route importlib.metadata.version through a stub: known core packages
    answer fixed versions; llama-cpp-python answers `llama` (or raises when
    None, like an uninstalled package)."""
    import importlib.metadata as md

    def _version(name):
        if name == "llama-cpp-python":
            if llama is None:
                raise md.PackageNotFoundError(name)
            return llama
        return {"faster-whisper": "1.2.1", "ctranslate2": "4.6.1"}.get(name, "?")

    monkeypatch.setattr(md, "version", _version)


def test_engine_versions_appends_llama_cpp_when_installed(monkeypatch):
    _stub_pkg_version(monkeypatch, "0.3.99")
    s = build_info.engine_versions()
    assert s.endswith(" · llama-cpp-python 0.3.99")
    assert "faster-whisper 1.2.1" in s


def test_engine_versions_omits_llama_cpp_when_absent(monkeypatch):
    # Deliberately skipped optional install ⇒ the part is ABSENT, never a
    # "llama-cpp-python ?" placeholder.
    _stub_pkg_version(monkeypatch, None)
    s = build_info.engine_versions()
    assert "llama-cpp-python" not in s
    assert "CTranslate2 4.6.1" in s


def test_reload_does_not_leak_a_new_boot_id(monkeypatch):
    # Bind the by-value copies BEFORE the reload re-mints the id, so the next
    # test checks real survival even when this file runs alone.
    from faster_whisper_backend import main
    from faster_whisper_backend.transcription import catalog_routes
    before = build_info.BOOT_ID
    assert catalog_routes.BOOT_ID == before
    assert main.BOOT_ID == before
    monkeypatch.setenv("WHISPER_BUILD_VERSION", "v0.0.0-reload")
    try:
        importlib.reload(build_info)
    finally:
        monkeypatch.undo()
        importlib.reload(build_info)
    # A reload mints a fresh id inside the test; the autouse fixture puts
    # the process one back afterwards (checked by the next test).
    assert build_info.BOOT_ID != before


def test_process_identity_survives_the_reload_tests():
    # Runs after the reload tests above (file order): the identity the
    # /v1/models route and main.py bound by value must still be build_info's.
    from faster_whisper_backend import main
    from faster_whisper_backend.transcription import catalog_routes
    assert catalog_routes.BOOT_ID == build_info.BOOT_ID
    assert main.BOOT_ID == build_info.BOOT_ID
