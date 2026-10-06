"""Package-internal page templates and shared CSS/JS widgets, stored as files.

Each sub-package keeps its big HTML pages / inline-script widgets in a
``templates/`` directory next to the module that serves them, and that module
reads them ONCE at import into the same module-level constant it always had
(``_STATS_VIEWER_HTML = templates.load(__file__, "stats.html")``). Keeping the
constant names matters: tests pin them (and their values), and
``web_common.render_page`` caches by the template string's value, so the text
must be loaded once and never re-read per request. Any import-time
post-processing (``.replace("__ASSET_V__", ...)``, ``{{STAGE_TOKENS}}``,
``__MAP_CAP__``) stays in Python, applied to the loaded text.

These files deliberately do NOT live under ``static/``: that tree is served
raw and publicly, while these are un-substituted server-side templates.

Files are read as UTF-8 with ``newline=""`` so the bytes on disk are exactly
the string the module used to hold; a ``\\r`` (a CRLF checkout, e.g. a Windows
``core.autocrlf`` clone) would change every hash and the served page, so it
fails the import loudly instead.
"""
from __future__ import annotations

import os


def load(module_file: str, name: str) -> str:
    """Return the text of ``<dir of module_file>/templates/<name>``."""
    path = os.path.join(os.path.dirname(os.path.abspath(module_file)), "templates", name)
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    assert "\r" not in text, f"{path} has CR line endings; templates must be LF-only"
    return text
