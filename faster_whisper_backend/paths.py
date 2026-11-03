"""Where the repo root is — ONE definition.

The package lives one level below the checkout; config.json, static/,
ytdlp_plugins/, ffmpeg/bin and the Windows data/ + models/ defaults all live
AT the checkout; modules read REPO_ROOT instead of their own dirname(__file__).

os.path.abspath, not Path.resolve(): symlinks are deliberately not followed,
so a symlinked checkout keeps resolving the same paths.
"""
import os

REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_data_dir() -> str:
    """WHISPER_DATA_DIR, else <repo>\\data on Windows ("/data" is drive-relative
    there), else /data — the root config._DATA_DIR and the runtime ledgers share.
    A function, not a constant: tests reload settings.config under a changed
    env, and this module is not reloaded."""
    return (os.environ.get("WHISPER_DATA_DIR") or "").strip() or (
        os.path.join(REPO_ROOT, "data") if os.name == "nt" else "/data")
