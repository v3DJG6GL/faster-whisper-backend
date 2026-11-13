"""Link checks in a helper process (media/link_helper.py).

A pasted link is matched against every yt-dlp site pattern, and yt-dlp scans
what its site answers: regex work on text the caller or the site wrote, which
holds the GIL and cannot be stopped in-process. Once the app has started the
helpers, that work runs in a process of its own and is killed at the check's
deadline. The rest of the suite runs the same checks in-process
(tests/conftest.py turns the helpers off); the tests here turn them on.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from asyncio import run as _run

import pytest

from faster_whisper_backend.media import download as udl
from faster_whisper_backend.media import link_helper as lh
from tests.media.test_download import _stand_in_yt_dlp

pytest.importorskip("yt_dlp")

# CiscoWebex's `(?:[^#]*&)*RCID=` backtracks on this shape, twice as long per
# `a=b&`: 0.6 s at 24 of them, 5 s at 27, about 40 s at 30 (no match at the
# end, so the URL is "Generic"). Run in-process, the match holds the GIL for
# all of it.
_SLOW_URL = "https://x.webex.com/x/ldr.php?RCID=" + "a=b&" * 30 + "!"

# The time limit of a check whose helper is expected to ANSWER: a helper
# starts in about 0.6 s on an idle box, and the suite runs on every core.
_ANSWER_S = 60.0

# A stand-in helper: speaks the transport, then does `{body}` with the
# request in `request` and the connection in `conn`.
_FAKE_BOOT = (
    "import os, sys, time\n"
    "from multiprocessing import connection as c\n"
    "conn = (c.PipeConnection if os.name == 'nt' else c.Connection)(int(sys.argv[2]))\n"
    "{before}\n"
    "conn.send_bytes(b'ready')\n"
    "request = conn.recv_bytes()\n"
    "{body}\n")


@pytest.fixture
def start(monkeypatch):
    """start(boot=None): turn the helpers on for this test (a stand-in
    helper program when `boot` is given) and return the running pool."""
    monkeypatch.setattr(lh, "ISOLATE", True)

    def _start(boot: "str | None" = None, **kw):
        if boot is not None:
            monkeypatch.setattr(lh, "_BOOT", boot)
        lh.start(**kw)
        return lh._HELPERS
    try:
        yield _start
    finally:
        pool = lh._HELPERS
        lh.reset()
        assert pool is None or pool.pids() == []


class _Ticker:
    """The longest time a 10 ms sleeper in this process went without
    running: what every other thread of the server would have waited."""

    def __enter__(self):
        self.longest = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()
        return self

    def _tick(self):
        last = time.monotonic()
        while not self._stop.wait(0.01):
            now = time.monotonic()
            self.longest = max(self.longest, now - last)
            last = now

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


# --- the point: a check that does not end is stopped, and stalls nothing ---

def test_a_link_whose_pattern_match_does_not_end_is_stopped_at_the_deadline(
        start, monkeypatch, caplog):
    # Offline even if a faster machine finished the match in time: "Generic"
    # is then refused by policy, nothing is fetched.
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", False, raising=False)
    pool = start()
    checker = pool.spare_pid()
    assert checker is not None
    t0 = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="whisper-api"), _Ticker() as ticker:
        with pytest.raises(udl.UrlTimeoutError, match="too long"):
            _run(udl.probe(_SLOW_URL, timeout=2.0))
    assert time.monotonic() - t0 < 8.0
    # In-process the sleeper would have stood still for the whole match.
    assert ticker.longest < 1.0, ticker.longest
    # The helper that ran the check is gone; only the next spare lives.
    assert checker not in pool.pids()
    assert pool.pids() == [pool.spare_pid()]
    assert "stopped at its time limit" in caplog.text
    assert "x.webex.com" in caplog.text and "RCID" not in caplog.text


def test_a_download_start_whose_match_does_not_end_is_stopped(start, monkeypatch):
    monkeypatch.setattr(udl, "_POLICY_TIMEOUT_S", 1.5)
    pool = start()
    with _Ticker() as ticker:
        with pytest.raises(udl.UrlTimeoutError, match="could not be checked in time"):
            _run(udl._extractor_policy_for(_SLOW_URL))
    assert ticker.longest < 1.0, ticker.longest
    assert pool.pids() == [pool.spare_pid()]


# --- same answers as in-process ----------------------------------------------

def test_the_helper_answers_as_the_in_process_match_under_the_live_settings(
        start, monkeypatch):
    """The settings a helper judges under are this process's, sent with the
    request: the allowlist below exists nowhere on disk."""
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    expected = udl._extractor_policy(url)
    monkeypatch.setattr(udl, "_POLICY_TIMEOUT_S", _ANSWER_S)
    pool = start()
    first = pool.spare_pid()
    assert _run(udl._extractor_policy_for(url)) == expected
    # One helper per check: the spare was used up and replaced.
    assert pool.spare_pid() not in (None, first)

    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", ["Youtube"], raising=False)
    pinned = udl._extractor_policy(url)
    assert pinned != expected and pinned[0]
    assert _run(udl._extractor_policy_for(url)) == pinned
    assert _run(udl._extractor_policy_for("https://example.com/a.mp4")) == \
        udl._extractor_policy("https://example.com/a.mp4")


def test_a_policy_refusal_in_the_helper_reaches_the_caller_unchanged(
        start, monkeypatch):
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", ["Vimeo"], raising=False)
    start()
    with pytest.raises(udl.UrlPolicyError, match="isn't on the server's allowed list"):
        _run(udl.probe("https://www.youtube.com/watch?v=dQw4w9WgXcQ", timeout=_ANSWER_S))
    with pytest.raises(udl.UrlPolicyError, match="isn't the site it names"):
        _run(udl.probe("https://help.senate.gov.evil.example/x", timeout=_ANSWER_S))


def test_extraction_in_the_helper_runs_behind_the_guard_and_is_logged_here(
        start, monkeypatch, caplog):
    """The helper installs the SSRF guard itself; what yt-dlp raised comes
    back to be logged (masked) and classified by the server."""
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", True, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    start()
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        with pytest.raises(udl.UrlDownloadError) as err:
            _run(udl.probe("http://127.0.0.1:9/a.mp3?token=s3cret", timeout=_ANSWER_S))
    assert str(err.value) == udl._UNREACHABLE_MSG
    assert "probe failed for host 127.0.0.1" in caplog.text
    assert udl.GUARD_MARKER in caplog.text
    assert "s3cret" not in caplog.text


def test_what_the_check_logs_in_the_helper_is_relayed_to_the_server_log(
        start, monkeypatch, caplog):
    """The direct-media probe's own line (why a link was not media) is
    written inside the helper, which has no log."""
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    start()
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        with pytest.raises(udl.UrlDownloadError, match="neither a supported site"):
            _run(udl.probe("http://127.0.0.1:9/a.mp3", timeout=_ANSWER_S))
    assert "SSRF guard active" not in caplog.text


# --- fail closed -------------------------------------------------------------

def test_after_shutdown_a_link_is_refused_not_checked_in_process(start, monkeypatch):
    start()
    lh.shutdown()
    assert lh.active() and lh._HELPERS.pids() == []

    async def _never(*a, **k):
        raise AssertionError("the check ran in the server process")
    monkeypatch.setattr(udl, "_probe_info", _never)
    monkeypatch.setattr(udl, "match_extractor", _never)
    with pytest.raises(udl.UrlDownloadError, match="could not be checked on the server"):
        _run(udl.probe("https://example.com/a.mp3", timeout=5.0))
    with pytest.raises(udl.UrlDownloadError, match="could not be checked on the server"):
        _run(udl._extractor_policy_for("https://example.com/a.mp3"))


def test_a_helper_that_cannot_be_started_refuses_the_link(start, monkeypatch, caplog):
    start(spare=False)

    def _no_process(*a, **k):
        raise OSError("out of processes")
    monkeypatch.setattr(lh.subprocess, "Popen", _no_process)
    with caplog.at_level(logging.WARNING, logger="whisper-api"):
        with pytest.raises(udl.UrlDownloadError, match="could not be checked on the server"):
            _run(udl.probe("https://example.com/a.mp3", timeout=5.0))
    assert "could not start a link check helper" in caplog.text


def test_the_suite_runs_link_checks_in_process():
    """tests/conftest.py's switch: start() (every TestClient lifespan calls
    it) starts nothing."""
    assert lh.ISOLATE is False
    lh.start()
    assert not lh.active()


# --- the transport ------------------------------------------------------------

def test_a_helper_that_does_not_answer_is_killed_at_the_timeout(start):
    pool = start(_FAKE_BOOT.format(before="", body="time.sleep(600)"))
    t0 = time.monotonic()
    with pytest.raises(lh.HelperTimeout):
        lh.run(b"x", 1.0)
    assert time.monotonic() - t0 < 6.0
    assert pool.pids() == [pool.spare_pid()]


def test_a_helper_that_never_gets_ready_is_killed_at_the_timeout(start):
    pool = start(_FAKE_BOOT.format(before="time.sleep(600)", body=""))
    with pytest.raises(lh.HelperTimeout):
        lh.run(b"x", 1.0)
    assert pool.pids() == [pool.spare_pid()]


def test_a_helper_that_dies_is_a_failure_with_its_exit_code(start, monkeypatch):
    pool = start(_FAKE_BOOT.format(before="", body="os._exit(3)"))
    with pytest.raises(lh.HelperFailed, match="exit code 3"):
        lh.run(b"x", _ANSWER_S)
    # A spare that died while idle is not handed out: the check starts its own.
    monkeypatch.setattr(lh, "_BOOT", _FAKE_BOOT.format(
        before="", body="conn.send_bytes(b'+' + request[8:])"))
    spare = pool._spare
    spare.proc.kill()
    spare.proc.wait(10)
    assert lh.run(b"x", _ANSWER_S) == b"x"
    assert spare.proc.pid not in pool.pids()


def test_a_reply_over_the_cap_is_not_read(start, monkeypatch):
    monkeypatch.setattr(lh, "_REPLY_MAX", 1000)
    start(_FAKE_BOOT.format(before="", body="conn.send_bytes(b'+' + b'x' * 5000)"))
    with pytest.raises(lh.HelperFailed, match="without an answer"):
        lh.run(b"x", _ANSWER_S)


def test_the_request_and_its_time_limit_reach_the_helper(start):
    start(_FAKE_BOOT.format(
        before="", body="import struct\nconn.send_bytes(b'+' + request[8:] + "
                        "str(struct.unpack('!d', request[:8])[0]).encode())"))
    assert lh.run(b"hello ", 12.5) == b"hello 12.5"


def test_a_helper_that_cannot_run_the_check_says_why(start):
    """The real helper, given a request that is no request."""
    start()
    with pytest.raises(lh.HelperFailed, match="could not run the check: JSONDecodeError"):
        lh.run(b"not json", _ANSWER_S)


def test_run_without_started_helpers_is_a_failure():
    assert not lh.active()
    with pytest.raises(lh.HelperFailed, match="never started"):
        lh.run(b"x", 1.0)


# --- the helper's side, run in this process -------------------------------------

def _request(op, url, **settings):
    return json.dumps({"op": op, "url": url, "left": 5.0, "cfg": settings}).encode()


def test_helper_handle_answers_a_probe_with_key_info_and_handoff_target(monkeypatch):
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    monkeypatch.setattr(udl.cfg, "MEDIA_MAX_BYTES", 5, raising=False)
    captured = _stand_in_yt_dlp(monkeypatch, "Youtube", {
        "title": "t", "extractor_key": "Generic",
        "webpage_url": "https://third.example/page"})
    reply = json.loads(udl.helper_handle(_request(
        "probe", "https://www.youtube.com/watch?v=x",
        URL_SOCKET_TIMEOUT_S=7, MEDIA_MAX_BYTES=99)))
    assert reply["ok"] == ["Youtube", {
        "title": "t", "extractor_key": "Generic",
        "webpage_url": "https://third.example/page"}, "https://third.example/page"]
    # The sent link settings were applied; nothing else can be set this way.
    assert captured["socket_timeout"] == 7.0
    assert udl.cfg.MEDIA_MAX_BYTES == 5
    assert udl._helper_result("probe", reply)[0] == "Youtube"


def test_helper_handle_names_a_refusal_and_keeps_what_was_logged(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="whisper-api")  # helper_warm's, in a helper
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    _stand_in_yt_dlp(monkeypatch, "Generic", {})

    def _not_media(url, timeout, **kw):
        udl.logger.info("[url-dl] direct-media probe failed for host %s", "x")
        return False
    monkeypatch.setattr(udl, "_direct_media_probe_sync", _not_media)
    before = list(udl.logger.handlers)
    reply = json.loads(udl.helper_handle(_request("probe", "https://x/page.html")))
    assert udl.logger.handlers == before
    assert reply["refused"][0] == "UrlDownloadError"
    assert "neither a supported site" in reply["refused"][1]
    assert reply["log"] == [[logging.INFO,
                             "[url-dl] direct-media probe failed for host x"]]
    with pytest.raises(udl.UrlDownloadError, match="neither a supported site"):
        udl._helper_result("probe", reply)


def test_helper_handle_hands_back_the_text_of_a_failed_extraction(monkeypatch):
    def _boom(url):
        raise RuntimeError("ERROR: [generic] " + "x" * 20_000 + " (caused by <HTTPError 404: Not Found>)")
    _stand_in_yt_dlp(monkeypatch, "Youtube", _boom)
    reply = json.loads(udl.helper_handle(_request("probe", "https://www.youtube.com/watch?v=x")))
    text = reply["extract_failed"]
    assert len(text) == 2 * udl._MASK_WINDOW + 1
    assert text.startswith("ERROR: [generic]") and text.endswith("Not Found>)")
    with pytest.raises(udl._ExtractFailed):
        udl._helper_result("probe", reply)


def test_helper_handle_answers_the_download_policy(monkeypatch):
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    monkeypatch.setattr(udl, "match_extractor", lambda u: "Generic")
    reply = json.loads(udl.helper_handle(_request("policy", "https://x/a.mp3")))
    assert udl._helper_result("policy", reply) == (
        ["generic"], ["direct", "protocol^=m3u8", "protocol=f4m"])


@pytest.mark.parametrize("op, reply", [
    ("probe", {}),
    ("probe", {"ok": ["Youtube", {}]}),
    ("probe", {"ok": [1, {}, None]}),
    ("probe", {"ok": ["Youtube", [], None]}),
    ("probe", {"ok": ["Youtube", {}, 5]}),
    ("policy", {"ok": [None]}),
    ("policy", {"ok": [["a"], "b"]}),
    ("policy", {"ok": [[1], None]}),
    ("probe", {"refused": "UrlPolicyError"}),
    ("probe", {"refused": ["UrlPolicyError", 5]}),
    ("probe", {"extract_failed": None}),
])
def test_a_reply_of_another_shape_is_a_failed_helper(op, reply):
    with pytest.raises(lh.HelperFailed, match="unexpected shape"):
        udl._helper_result(op, reply)


def test_a_refusal_from_a_helper_is_one_of_ours_and_short():
    with pytest.raises(udl.UrlDownloadError) as err:
        udl._helper_result("probe", {"refused": ["SystemExit", "x" * 5000]})
    assert type(err.value) is udl.UrlDownloadError
    assert len(str(err.value)) == udl._HELPER_REFUSAL_MAX
    with pytest.raises(udl.UrlTimeoutError):
        udl._helper_result("probe", {"refused": ["UrlTimeoutError", "slow"]})


def test_relayed_helper_lines_are_screened_and_capped(caplog):
    lines = [[logging.INFO, "ok line"], [logging.WARNING, "forged\n2026 ERROR line"],
             [55, "odd level"], "not a line", [logging.ERROR, 7],
             [logging.ERROR, "y" * 5000]]
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        udl._relay_helper_log({"log": lines + [[logging.INFO, "more"]] * 50})
        udl._relay_helper_log({"log": "x"})
    messages = [r.getMessage() for r in caplog.records]
    assert messages[:3] == ["ok line", "forged?2026 ERROR line",
                            "y" * udl._HELPER_LOG_LINE_MAX]
    assert len(messages) == 3 + (udl._HELPER_LOG_MAX - len(lines))


def test_the_helper_is_sent_the_link_settings_only(monkeypatch):
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", ["Youtube"], raising=False)
    sent = udl._helper_cfg()
    assert sent["URL_ALLOWED_EXTRACTORS"] == ["Youtube"]
    assert "URL_ALLOW_GENERIC" in sent and "URL_SOCKET_TIMEOUT_S" in sent
    assert all(name.startswith("URL_") for name in sent)
    json.dumps(sent)


# --- the app ---------------------------------------------------------------------

@pytest.mark.parametrize("enabled", [True, False])
def test_the_lifespan_starts_and_stops_the_link_helpers(app_module, monkeypatch, enabled):
    from starlette.testclient import TestClient

    events: list = []
    monkeypatch.setattr(lh, "start", lambda **k: events.append(("start", k)))
    monkeypatch.setattr(lh, "shutdown", lambda: events.append("shutdown"))
    monkeypatch.setattr(app_module.cfg, "URL_DOWNLOAD_ENABLED", enabled, raising=False)
    with TestClient(app_module.app, client=("127.0.0.1", 12345)):
        assert events == [("start", {"spare": enabled})]
        events.clear()
    assert events == ["shutdown"]
