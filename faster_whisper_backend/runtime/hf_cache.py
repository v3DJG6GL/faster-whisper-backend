"""Where Hugging Face hub downloads land — one precedence for every caller.

huggingface_hub freezes HF_HUB_CACHE from the environment at import time
(faster_whisper imports the hub at startup), so a later HF_HOME setdefault
cannot redirect a download: the translation GGUFs and the pyannote pipeline
pass `hub_cache_dir()` EXPLICITLY as their cache_dir, and runtime.model_sizes
looks for the same files through `hub_lookup_dir()`.
"""
from __future__ import annotations

import os


def hub_cache_dir() -> "str | None":
    """The cache_dir to pass to a hub download: a set HF_HUB_CACHE wins (the
    hub ranks it above HF_HOME too, so whisper's own hub-default download
    lands there), else HF_HOME/hub, else <DOWNLOAD_ROOT>/hf/hub, else None
    (the hub's own default).

    Expanded the way the hub expands them (python-dotenv and systemd
    Environment= leave "~" alone): HF_HUB_CACHE and HF_HOME like
    huggingface_hub.constants (vars, then "~"), DOWNLOAD_ROOT like an
    explicit cache_dir ("~" only). Unexpanded, the download lands under the
    home dir while the lookup checks a cwd-relative "~/..." that never
    exists."""
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        return os.path.expandvars(os.path.expanduser(hub_cache))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(
            os.path.expandvars(os.path.expanduser(hf_home)), "hub")
    from faster_whisper_backend.settings import config as cfg
    download_root = (getattr(cfg, "DOWNLOAD_ROOT", None) or "").strip()
    if download_root:
        return os.path.join(os.path.expanduser(download_root), "hf", "hub")
    return None


def hub_lookup_dir() -> str:
    """Where a download made with `hub_cache_dir()` actually sits: that dir,
    or — when it is None — the hub's own default, read from the hub itself
    (HF_HUB_CACHE as frozen at its import, XDG_CACHE_HOME included); the
    env/home chain only stands in when huggingface_hub is not installed."""
    explicit = hub_cache_dir()
    if explicit:
        return explicit
    try:
        from huggingface_hub import constants
        return constants.HF_HUB_CACHE
    except ImportError:
        return (os.environ.get("HF_HUB_CACHE")
                or os.path.join(os.path.expanduser("~/.cache/huggingface"), "hub"))
