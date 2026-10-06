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
