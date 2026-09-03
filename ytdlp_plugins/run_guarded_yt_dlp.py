"""Entry point for the yt-dlp download subprocess: guard first, then yt-dlp.

faster_whisper_backend/url/download.py download() runs `python <this file> <yt-dlp args…>` instead of
`python -m yt_dlp` for two reasons, both about failing CLOSED:

  1. yt-dlp's plugin loader SWALLOWS a plugin's import error — it prints a
     traceback and carries on with an unguarded network stack, which is the
     one outcome this feature must never have. Importing the guard here, by
     path, makes a broken guard an immediate non-zero exit instead. (The CLI
     is still invoked with `--no-plugin-dirs --plugin-dirs <this directory>`,
     so the guard is ALSO installed through the official channel and an
     operator's stray plugins are cleared first; install() is idempotent.)
  2. `python <script>` puts THIS directory on sys.path — not the repo root.
     The repo root would let its own directories (``static/``, a bind-mounted
     ``secrets/``, …) shadow stdlib modules for ~2000 third-party extractors.
     The guard reaches net_policy.py by path instead; see fwb_ssrf_guard.py.

Everything after the script path is passed to yt-dlp verbatim, so
build_download_argv's argument order (and the trailing `-- <url>`) is
unchanged.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_GUARD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fwb_ssrf_guard", "yt_dlp_plugins", "extractor", "fwb_ssrf_guard.py")

# Marker the parent greps for; url_download.classify_error maps it to the
# client-safe "the site could not be reached from the server".
_MARKER = "fwb-ssrf-guard"


def _install_guard():
    spec = importlib.util.spec_from_file_location(
        "fwb_ssrf_guard_boot", _GUARD_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"guard module not found at {_GUARD_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fwb_ssrf_guard_boot"] = module
    spec.loader.exec_module(module)  # registers the handler as a side effect
    return module


def main() -> int:
    try:
        guard = _install_guard()
        if not guard.is_installed():
            raise RuntimeError("the guard handler did not register")
    except Exception as e:  # noqa: BLE001 — any failure here is fail-closed
        try:
            import yt_dlp.version
            version = yt_dlp.version.__version__
        except Exception:  # noqa: BLE001
            version = "unknown"
        sys.stderr.write(
            f"ERROR: {_MARKER}: refusing to run unguarded — could not install "
            f"the SSRF guard for yt-dlp {version}: {type(e).__name__}: {e}\n")
        return 78  # EX_CONFIG
    import yt_dlp
    return yt_dlp.main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
