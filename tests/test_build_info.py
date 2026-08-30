"""build_info version resolution: env override > git describe > "unknown"."""

import importlib
import subprocess

import build_info


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
        monkeypatch.delenv("WHISPER_BUILD_VERSION")
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
