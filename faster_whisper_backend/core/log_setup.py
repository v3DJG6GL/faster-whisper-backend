"""Process logging setup: stderr (the console handler, its threshold set by
CONSOLE_LOG_LEVEL) + a rotating, owner-only log file (no colors, UTC
timestamps) + the in-memory severity ring behind the nav pills and /stats.

main.py calls install() once per import, before the heavy imports, so the
first lines any later module logs already land in the file. install() drops
every root handler first, so a re-run (uvicorn auto-reload, the test suite's
importlib.reload(main)) never stacks handlers. The settings save path calls
apply_console_log_level() when CONSOLE_LOG_LEVEL changes.
"""
import logging
import logging.handlers
import os
import re
import sys
import time

from faster_whisper_backend.settings import config as cfg
from faster_whisper_backend.core import store_common
# Tail WARNING+ records into an in-memory ring used by the nav-row severity
# pills and the /stats page. Does no I/O — append to a deque and return.
from faster_whisper_backend.core.web_common import SeverityCounter

# =============================================================================
# Logging setup: stderr (with colors when TTY) + rotating file (no colors)
# =============================================================================
# Log path and rotation policy come from settings/config.py / WHISPER_LOG_FILE.
# The file copy strips ANSI escape codes so it stays grep-friendly and the
# /logs web viewer can re-color via CSS based on content.
# An uncreatable log dir (e.g. the container-first /data default on a
# bare-metal box without WHISPER_DATA_DIR) must not kill the import — the
# server degrades to stderr-only logging, the standard container posture.

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


class _StripAnsiFormatter(logging.Formatter):
    # Emit file timestamps in UTC (ISO-8601 with a trailing 'Z'). The log file
    # is then unambiguous regardless of the server's timezone; the /logs web
    # viewer converts each line to the reader's local time (like every other
    # timestamp surface). gmtime is a class attribute so it applies to asctime.
    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        return _ANSI_ESCAPE_RE.sub("", super().format(record))


class _SecureRotatingFileHandler(logging.handlers.RotatingFileHandler):
    # Every request block written here carries RAW WHISPER / FINAL transcript
    # text, the upload filename, the username and the key label — the same
    # plaintext dictation store_common.secure_db_file() keeps at 0600 in the
    # SQLite stores. The handler would otherwise create the file at the process
    # umask (0644 typical), so tighten it on open and after every rollover.
    def _open(self):  # type: ignore[override]
        stream = super()._open()
        store_common.secure_file(self.baseFilename)
        return stream


def _secure_log_dir(path: str, created: bool) -> None:
    # Only a directory WE created is ours to lock down. LOG_FILE is an
    # operator-supplied path, so a pre-existing directory (e.g. /var/log on a
    # root-run install) must not be re-permissioned to 0700 under the server —
    # that locks every other user and daemon out of it. The log file itself is
    # still tightened unconditionally by _SecureRotatingFileHandler.
    if created:
        store_common.secure_dir(path)


_root = logging.getLogger()
# Rebound by every install(); read through the module (log_setup._console_handler),
# never via `from ... import`, or a re-install leaves the caller on a stale handler.
_console_handler: logging.StreamHandler = logging.StreamHandler(sys.stderr)
_file_handler: "_SecureRotatingFileHandler | None" = None
_log_dir_ok = True


def install() -> None:
    """Configure the root logger. Idempotent: every root handler a previous
    install() (or a basicConfig) added is removed and closed first."""
    global _console_handler, _file_handler, _log_dir_ok
    _log_dir_ok = True
    _log_dir_new = not os.path.isdir(os.path.dirname(cfg.LOG_FILE) or ".")
    try:
        os.makedirs(os.path.dirname(cfg.LOG_FILE) or ".", exist_ok=True)
    except OSError as _log_exc:
        _log_dir_ok = False
        print(
            f"WARNING: cannot create log directory for {cfg.LOG_FILE!r} ({_log_exc}) "
            "— file logging disabled, logging to stderr only. Set WHISPER_LOG_FILE "
            "or WHISPER_DATA_DIR to a writable location.",
            file=sys.stderr,
        )

    _root.setLevel(logging.INFO)
    # Remove any handlers a previous import (or basicConfig) added so we don't
    # double-log on auto-reload.
    for _h in list(_root.handlers):
        _root.removeHandler(_h)
        _h.close()

    _console_handler = logging.StreamHandler(sys.stderr)
    _console_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    _root.addHandler(_console_handler)

    if _log_dir_ok:
        _secure_log_dir(os.path.dirname(cfg.LOG_FILE), _log_dir_new)
        _file_handler = _SecureRotatingFileHandler(
            cfg.LOG_FILE, maxBytes=cfg.LOG_MAX_BYTES, backupCount=cfg.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        _file_handler.setFormatter(_StripAnsiFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",   # UTC (converter=gmtime); viewer localizes
        ))
        # Pinned so the file (and /logs) reads the same whatever the console
        # level is — CONSOLE_LOG_LEVEL=debug lowers the ROOT level, see below.
        _file_handler.setLevel(logging.INFO)
        _root.addHandler(_file_handler)

    _root.addHandler(SeverityCounter())

    apply_console_log_level(getattr(cfg, "CONSOLE_LOG_LEVEL", "warning"))
    if _console_handler.level > logging.INFO:
        # A quiet `docker logs` should not look like a dead server. handle()
        # skips the level check, and the record goes to the console ONLY — not
        # the file, not the severity pills.
        _console_handler.handle(logging.LogRecord(
            "whisper-api", logging.INFO, __file__, 0,
            "console log level is %s: INFO lines (including each transcription's "
            "log block) %s. Change CONSOLE_LOG_LEVEL under Settings > Logging.",
            (logging.getLevelName(_console_handler.level),
             f"go only to {cfg.LOG_FILE} and the /logs page" if _log_dir_ok
             else "are not written anywhere (the log file is unavailable)"),
            None))


def apply_console_log_level(name: object) -> None:
    """Point the stderr handler (docker logs / journald) at CONSOLE_LOG_LEVEL.

    The root level follows it down to DEBUG only — never up — so INFO keeps
    reaching the file handler (pinned at INFO above) at every setting. Called
    at install() and again by the settings save path when the field changes."""
    lvl = getattr(logging, str(name or "").strip().upper(), None)
    if not isinstance(lvl, int):
        lvl = logging.WARNING
    _console_handler.setLevel(lvl)
    _root.setLevel(min(logging.INFO, lvl))
