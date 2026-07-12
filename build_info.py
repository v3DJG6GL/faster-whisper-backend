"""Build + runtime identity — the version string clients and the WebUI display.

Version resolution order (first hit wins):
  1. WHISPER_BUILD_VERSION env — baked into container images by CI
     (docker build --build-arg BUILD_VERSION=$(git describe ...); the image
     has no .git to describe, see .dockerignore).
  2. `git describe --tags --always --dirty` on a bare-metal checkout
     (Linux/Windows) — anchors to the newest v* tag and bumps automatically
     with every commit, e.g. "v0.1.0-3-g1a2b3c4".
  3. "unknown" — tarball without .git, or no git on PATH.

Resolved once at import: the value cannot change while the process runs, and
/v1/models must not fork a git subprocess per request. The module also owns
the rest of the per-process identity (BOOT_ID, start time) shown by the
WebUI's version surfaces — header tag, hub build line, settings card.
"""

import os
import platform
import subprocess
import time
import uuid

SERVER_NAME = "faster-whisper-backend"

# Per-process token, regenerated on every interpreter start. Served on
# /v1/models so clients (and the WebUI's restart flow) can detect the new
# process even if their polling missed the brief "service down" window.
BOOT_ID = uuid.uuid4().hex

# ≈ process start — this module is imported at the top of main.py.
STARTED_AT = time.time()
STARTED_UTC = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(STARTED_AT))


def _resolve() -> str:
    env = (os.environ.get("WHISPER_BUILD_VERSION") or "").strip()
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


APP_VERSION = _resolve()

# The release part alone ("v0.1.0" out of "v0.1.0-3-g1a2b3c4") for compact UI
# chips; non-describe strings (bare sha, "dev", "unknown") pass through whole.
VERSION_SHORT = (
    APP_VERSION.split("-")[0]
    if APP_VERSION.startswith("v") and APP_VERSION[1:2].isdigit()
    else APP_VERSION
)


def runs_as() -> str:
    """"docker" when running inside a container, else "bare-metal"."""
    return "docker" if os.path.exists("/.dockerenv") else "bare-metal"


def uptime_str(now: float | None = None) -> str:
    """Compact uptime since process start: "2 d 4 h" / "3 h 12 m" / "5 m" / "12 s"."""
    s = int(max(0.0, (time.time() if now is None else now) - STARTED_AT))
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    if d:
        return f"{d} d {h} h"
    if h:
        return f"{h} h {m} m"
    if m:
        return f"{m} m"
    return f"{sec} s"


def engine_versions() -> str:
    """Display string "python 3.14.6 · faster-whisper 1.2.1 · CTranslate2 4.6.1".
    Each part is best-effort ("?") — never raises."""
    try:
        from importlib.metadata import version as _pkg_version
    except Exception:  # pragma: no cover — stdlib since 3.8
        _pkg_version = None

    def _pkg(name: str) -> str:
        if _pkg_version is None:
            return "?"
        try:
            return _pkg_version(name)
        except Exception:
            return "?"

    return (
        f"python {platform.python_version()}"
        f" · faster-whisper {_pkg('faster-whisper')}"
        f" · CTranslate2 {_pkg('ctranslate2')}"
    )
