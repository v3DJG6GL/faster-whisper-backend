"""CONSOLE_LOG_LEVEL: the stderr handler (docker logs / journald) gets its own
threshold while the log file — and so /logs — always keeps INFO."""

import io
import logging

import pytest


@pytest.fixture
def levels(app_module):
    # apply_console_log_level moves the ROOT level too; put it back so later
    # tests (caplog) see the import-time state.
    root = logging.getLogger()
    saved = (app_module._console_handler.level, root.level)
    yield app_module
    app_module._console_handler.setLevel(saved[0])
    root.setLevel(saved[1])


def test_default_keeps_info_off_the_console_but_in_the_file(levels):
    m = levels
    assert m.cfg.CONSOLE_LOG_LEVEL == "warning"
    assert m._console_handler.level == logging.WARNING
    assert logging.getLogger().level == logging.INFO
    assert m._file_handler.level == logging.INFO


def test_debug_lowers_root_but_not_the_file(levels):
    m = levels
    m.apply_console_log_level("debug")
    assert m._console_handler.level == logging.DEBUG
    assert logging.getLogger().level == logging.DEBUG
    assert m._file_handler.level == logging.INFO


@pytest.mark.parametrize("name", ["verbose", "", None])
def test_unknown_level_falls_back_to_warning(levels, name):
    levels.apply_console_log_level(name)
    assert levels._console_handler.level == logging.WARNING
    assert logging.getLogger().level == logging.INFO


def test_receipt_level_record_reaches_file_not_console(levels, monkeypatch):
    m = levels
    out = io.StringIO()
    monkeypatch.setattr(m._console_handler, "stream", out)
    written = []
    monkeypatch.setattr(m._file_handler, "emit", written.append)
    logging.getLogger("whisper-api").info("RAW WHISPER secret words")
    assert "secret words" not in out.getvalue()
    assert any("secret words" in r.getMessage() for r in written)


def test_settings_save_applies_live(levels, client):
    m = levels
    r = client.post("/settings/state", json={"CONSOLE_LOG_LEVEL": "error"})
    assert r.status_code == 200
    assert "CONSOLE_LOG_LEVEL" in r.json()["saved"]
    assert m._console_handler.level == logging.ERROR
    r = client.post("/settings/state", json={"CONSOLE_LOG_LEVEL": "info"})
    assert r.status_code == 200
    assert m._console_handler.level == logging.INFO
