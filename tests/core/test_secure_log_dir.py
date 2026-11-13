"""log_setup._secure_log_dir: only a log directory the service created itself is
tightened to owner-only; an unusable log location degrades to stderr."""

import os
import sys

import pytest

from faster_whisper_backend.core import store_common


def test_secure_log_dir_only_tightens_a_directory_we_created(monkeypatch):
    # LOG_FILE is operator-chosen: chmod-ing a pre-existing /var/log to 0700
    # would lock every other daemon out of it, so only a freshly created
    # directory is ours to secure.
    from faster_whisper_backend.core import log_setup
    seen = []
    monkeypatch.setattr(store_common, "secure_dir", seen.append)
    log_setup._secure_log_dir("/some/dir", created=False)
    assert seen == []
    log_setup._secure_log_dir("/some/dir", created=True)
    assert seen == ["/some/dir"]


@pytest.mark.skipif(sys.platform == "win32" or not hasattr(os, "geteuid")
                    or os.geteuid() == 0,
                    reason="needs POSIX permissions and a non-root user")
def test_unwritable_existing_log_dir_degrades_to_stderr(monkeypatch, tmp_path,
                                                        capsys):
    """makedirs(exist_ok=True) succeeds on an existing read-only directory, so
    the handler's open was what raised — PermissionError out of install(),
    and main.py failed to import."""
    import logging

    from faster_whisper_backend.core import log_setup

    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        # A private logger stands in for the root one, and the module globals
        # install() rebinds are restored by monkeypatch.
        monkeypatch.setattr(log_setup, "_root",
                            logging.getLogger("test-log-setup-isolated"))
        for name in ("_console_handler", "_file_handler", "_log_dir_ok"):
            monkeypatch.setattr(log_setup, name, getattr(log_setup, name))
        monkeypatch.setattr(log_setup.cfg, "LOG_FILE", str(ro / "whisper.log"),
                            raising=False)
        log_setup.install()
        assert log_setup._log_dir_ok is False
        assert log_setup._file_handler is None
        assert "cannot open log file" in capsys.readouterr().err
    finally:
        ro.chmod(0o755)
        for h in list(log_setup._root.handlers):
            log_setup._root.removeHandler(h)
            h.close()


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX file modes")
def test_log_file_is_never_wider_than_0600_at_create(monkeypatch, tmp_path):
    """The file carries transcript text, so it is born owner-only — at start
    and at every rollover — instead of at the umask until the chmod lands."""
    import logging
    import stat

    from faster_whisper_backend.core import log_setup

    seen = []
    real_secure_file = store_common.secure_file

    def record_then_secure(path):
        # The mode the stdlib open left behind, before the chmod narrows it.
        seen.append(stat.S_IMODE(os.stat(path).st_mode))
        real_secure_file(path)

    monkeypatch.setattr(log_setup.store_common, "secure_file", record_then_secure)
    old_umask = os.umask(0o022)
    handler = None
    try:
        handler = log_setup._SecureRotatingFileHandler(
            str(tmp_path / "w.log"), maxBytes=200, backupCount=2, encoding="utf-8")
        for _ in range(12):
            handler.emit(logging.LogRecord(
                "t", logging.INFO, __file__, 0, "x" * 60, None, None))
    finally:
        os.umask(old_umask)
        if handler is not None:
            handler.close()
    assert len(seen) >= 3, seen          # the first open + at least two rollovers
    assert set(seen) == {0o600}, [oct(m) for m in seen]
    assert stat.S_IMODE(os.stat(tmp_path / "w.log").st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX file modes")
def test_log_file_left_wider_by_an_older_build_is_still_narrowed(tmp_path):
    import stat

    from faster_whisper_backend.core import log_setup

    path = tmp_path / "old.log"
    path.write_text("earlier line\n", encoding="utf-8")
    path.chmod(0o644)
    handler = log_setup._SecureRotatingFileHandler(
        str(path), maxBytes=10_000, backupCount=1, encoding="utf-8")
    handler.close()
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert path.read_text(encoding="utf-8") == "earlier line\n"
