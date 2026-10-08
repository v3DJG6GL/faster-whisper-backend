"""runtime.hf_cache: the one hub-cache precedence downloads and the
model-size lookup share."""
import os

from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.runtime import hf_cache
from faster_whisper_backend.runtime import model_sizes


def test_hf_home_wins_then_download_root_then_hub_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", "/hfhome")
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", str(tmp_path), raising=False)
    assert hf_cache.hub_cache_dir() == os.path.join("/hfhome", "hub")
    monkeypatch.delenv("HF_HOME")
    assert hf_cache.hub_cache_dir() == os.path.join(str(tmp_path), "hf", "hub")
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", "  ", raising=False)
    assert hf_cache.hub_cache_dir() is None


def test_lookup_follows_hf_hub_cache_when_downloads_use_the_hub_default(monkeypatch):
    # Downloads pass cache_dir=None here, so the hub writes to its own
    # HF_HUB_CACHE (frozen at its import: HF_HUB_CACHE / HF_HOME /
    # XDG_CACHE_HOME); the size lookup must look there too.
    from huggingface_hub import constants
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", None, raising=False)
    monkeypatch.setattr(constants, "HF_HUB_CACHE", "/custom/hub")
    assert hf_cache.hub_lookup_dir() == "/custom/hub"
    assert model_sizes._model_path("gguf:org/repo:Q4") == os.path.join(
        "/custom/hub", "models--org--repo")


def test_hf_hub_cache_wins_over_hf_home(monkeypatch, tmp_path):
    # The hub ranks HF_HUB_CACHE above HF_HOME/hub, and whisper's own
    # hub-default download follows it: the GGUFs, pyannote and the size
    # lookup must land in the same dir, not re-download under HF_HOME.
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "a"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "b"))
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", str(tmp_path / "c"),
                        raising=False)
    assert hf_cache.hub_cache_dir() == str(tmp_path / "a")
    assert hf_cache.hub_lookup_dir() == str(tmp_path / "a")
    # "~" expanded like huggingface_hub.constants does.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", "~/hub")
    assert hf_cache.hub_cache_dir() == os.path.join(str(tmp_path), "hub")


def test_legacy_huggingface_hub_cache_alias_wins_over_hf_home(monkeypatch,
                                                              tmp_path):
    # huggingface_hub still honours the deprecated HUGGINGFACE_HUB_CACHE
    # above HF_HOME/hub; following HF_HOME instead would split whisper's
    # download from the GGUF/pyannote cache and the size lookup.
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "x"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "y"))
    assert hf_cache.hub_cache_dir() == str(tmp_path / "x")
    assert hf_cache.hub_lookup_dir() == str(tmp_path / "x")


def test_no_hub_cache_var_is_inherited_from_the_shell():
    # conftest's autouse scrub: every variable hub_cache_dir() ranks above
    # HF_HOME / DOWNLOAD_ROOT must be gone inside a test, or an exported one
    # silently overrides each tmp-dir test. A new alias belongs in the scrub.
    assert "HF_HUB_CACHE" not in os.environ
    assert "HUGGINGFACE_HUB_CACHE" not in os.environ
