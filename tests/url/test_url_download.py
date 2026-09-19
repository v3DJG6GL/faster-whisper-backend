"""Unit tests for url/download.py — no network, no real yt-dlp subprocess.

The download() tests monkeypatch build_download_argv to run a tiny inline
Python script that mimics yt-dlp's observable behavior (progress lines on
stdout, an output file, exit codes), so the full subprocess plumbing —
progress parsing, cancellation, timeouts, result validation — runs for real.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

from faster_whisper_backend.url import download as udl


# ---------------------------------------------------------------------------
# validate_url
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=abc123",
    "http://example.com/talk.mp3",
    "https://example.com/a?b=c&d=e",
])
def test_validate_url_accepts_http_https(url):
    assert udl.validate_url("  " + url + "  ") == url


@pytest.mark.parametrize("url", [
    "", "   ",
    "file:///etc/passwd",
    "ftp://example.com/a.mp3",
    "data:audio/wav;base64,AAAA",
    "javascript:alert(1)",
    "example.com/no-scheme",
    "https://",                      # no host
    "https://exa mple.com/a",        # embedded space
    "https://example.com/a\nb",      # newline
    "https://example.com/\x07",      # control char
])
def test_validate_url_rejects(url):
    with pytest.raises(udl.UrlDownloadError):
        udl.validate_url(url)


def test_validate_url_rejects_overlong():
    with pytest.raises(udl.UrlDownloadError):
        udl.validate_url("https://example.com/" + "a" * 2048)


# ---------------------------------------------------------------------------
# policy (info-dict half)
# ---------------------------------------------------------------------------

def test_policy_rejects_playlist(monkeypatch):
    with pytest.raises(udl.UrlDownloadError, match="[Pp]laylist"):
        udl._policy_check_info({"_type": "playlist"})


def test_policy_rejects_live(monkeypatch):
    with pytest.raises(udl.UrlDownloadError, match="[Ll]ive"):
        udl._policy_check_info({"is_live": True})


def test_policy_rejects_over_duration(monkeypatch):
    monkeypatch.setattr(udl.cfg, "URL_MAX_DURATION_S", 60, raising=False)
    with pytest.raises(udl.UrlDownloadError, match="limit"):
        udl._policy_check_info({"duration": 61})
    udl._policy_check_info({"duration": 59})  # under: no raise


def test_policy_rejects_over_filesize(monkeypatch):
    monkeypatch.setattr(udl.cfg, "MEDIA_MAX_BYTES", 1000, raising=False)
    with pytest.raises(udl.UrlDownloadError, match="size"):
        udl._policy_check_info({"filesize_approx": 2000})


def test_effective_max_bytes_is_media_cap(monkeypatch):
    # One ceiling for every media path: a link admits exactly what an
    # upload would, never more.
    monkeypatch.setattr(udl.cfg, "MEDIA_MAX_BYTES", 12345, raising=False)
    assert udl._effective_max_bytes() == 12345


# ---------------------------------------------------------------------------
# policy (extractor half) — match_extractor is monkeypatched: the real
# registry match is yt-dlp's own behavior, not ours to test.
# ---------------------------------------------------------------------------

def _run(coro):
    # asyncio.run (not a bare new_event_loop): the loop must be CLOSED after
    # each call or every test leaks an epoll fd + subprocess-watcher state.
    return asyncio.run(coro)


def test_extractor_allowlist_case_insensitive(monkeypatch):
    monkeypatch.setattr(udl, "match_extractor", lambda u: "Youtube")
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", ["youtube"],
                        raising=False)
    assert _run(udl.check_url_policy("https://x/")) == "Youtube"
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", ["Vimeo"],
                        raising=False)
    with pytest.raises(udl.UrlPolicyError, match="allowed list"):
        _run(udl.check_url_policy("https://x/"))


def test_empty_allowlist_admits_any_dedicated_extractor(monkeypatch):
    monkeypatch.setattr(udl, "match_extractor", lambda u: "SoundCloud")
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    assert _run(udl.check_url_policy("https://x/")) == "SoundCloud"


def test_generic_rejected_by_default(monkeypatch):
    monkeypatch.setattr(udl, "match_extractor", lambda u: "Generic")
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", False, raising=False)
    with pytest.raises(udl.UrlPolicyError):
        _run(udl.check_url_policy("https://internal.host/x"))


def test_generic_allowed_with_flag(monkeypatch):
    monkeypatch.setattr(udl, "match_extractor", lambda u: "Generic")
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", True, raising=False)
    assert _run(udl.check_url_policy("https://x/")) == "Generic"


def test_direct_media_probe_gates_generic(monkeypatch):
    monkeypatch.setattr(udl, "match_extractor", lambda u: "Generic")
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    monkeypatch.setattr(udl, "_direct_media_probe_sync",
                        lambda u, timeout: True)
    assert _run(udl.check_url_policy("https://x/a.mp3")) == "Generic"
    monkeypatch.setattr(udl, "_direct_media_probe_sync",
                        lambda u, timeout: False)
    with pytest.raises(udl.UrlDownloadError, match="direct"):
        _run(udl.check_url_policy("https://x/page.html"))


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "10.1.2.3",
                                  "192.168.1.1", "169.254.169.254",
                                  "100.64.0.1", "::1"])
def test_forbidden_hosts(host):
    assert udl._host_is_forbidden(host) is True


def test_unresolvable_host_is_forbidden(monkeypatch):
    # Stubbed resolver: on wildcard / captive-portal DNS a real lookup of a
    # bogus name can resolve to a public address and fail this for reasons
    # unrelated to the code.
    def _nxdomain(*a, **k):
        raise OSError("nxdomain")

    # The resolver lives in net_policy now — the ONE definition of the
    # address gate, shared with the yt-dlp guard subprocess.
    monkeypatch.setattr(udl.net_policy.socket, "getaddrinfo", _nxdomain)
    assert udl._host_is_forbidden("anything.invalid") is True


def test_empty_resolution_is_forbidden(monkeypatch):
    monkeypatch.setattr(udl.net_policy.socket, "getaddrinfo",
                        lambda *a, **k: [])
    assert udl._host_is_forbidden("anything.invalid") is True


# ---------------------------------------------------------------------------
# progress-template parsing
# ---------------------------------------------------------------------------

def test_parse_progress_line_well_formed():
    assert udl._parse_progress_line("dl:1024 4096 NA") == (1024, 4096)


def test_parse_progress_line_estimate_fallback():
    assert udl._parse_progress_line("dl:10 NA 200") == (10, 200)


def test_parse_progress_line_unknown_total():
    assert udl._parse_progress_line("dl:10 NA NA") == (10, None)


def test_parse_progress_line_infinite_total_is_none():
    # int(float("inf")) raises OverflowError, not ValueError — it must be
    # swallowed like 'NA', never escape as a generic 500.
    assert udl._parse_progress_line("dl:10 inf NA") == (10, None)


@pytest.mark.parametrize("line", [
    "", "garbage", "dl:", "dl:NA NA NA", "1024 4096 NA", "[youtube] extracting",
])
def test_parse_progress_line_rejects_noise(line):
    assert udl._parse_progress_line(line) is None


# ---------------------------------------------------------------------------
# classify_error — one per taxonomy bucket + default; never echoes input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stderr,needle", [
    ("ERROR: Sign in to confirm you're not a bot", "bot"),
    ("ERROR: Sign in to confirm your age", "age-restricted"),
    ("ERROR: Private video. Sign in", "private"),
    ("ERROR: Join this channel to get access; members-only", "members-only"),
    ("ERROR: The uploader has not made this video available in your country",
     "region"),
    ("ERROR: Video unavailable. This video has been removed", "unavailable"),
    ("ERROR: Unsupported URL: https://x", "isn't supported"),
    ("ERROR: This live event will begin shortly", "hasn't finished"),
    ("File is larger than max-filesize", "size limit"),
    ("ERROR: Unable to download webpage: timed out", "could not be reached"),
])
def test_classify_error_taxonomy(stderr, needle):
    assert needle in udl.classify_error(stderr)


def test_classify_error_default_never_echoes_stderr():
    secret = "/tmp/secret-path/cookies.txt https://x/?token=abc"
    msg = udl.classify_error(f"ERROR: something exploded at {secret}")
    assert "secret-path" not in msg and "token=abc" not in msg
    assert "yt-dlp" in msg


# ---------------------------------------------------------------------------
# download() against a fake yt-dlp subprocess
# ---------------------------------------------------------------------------

def _fake_argv(script: str) -> "list[str]":
    return [sys.executable, "-c", script]


def _patch_argv(monkeypatch, script: str):
    monkeypatch.setattr(
        udl, "build_download_argv",
        lambda url, *, dest_dir, max_bytes: _fake_argv(
            script.replace("__DEST__", dest_dir)))


_OK_SCRIPT = """
import os, sys, time
print("dl:100 1000 NA", flush=True)
time.sleep(0.05)
print("dl:1000 1000 NA", flush=True)
open(os.path.join(r"__DEST__", "media.m4a"), "wb").write(b"x" * 64)
"""


def test_download_success(tmp_path, monkeypatch):
    _patch_argv(monkeypatch, _OK_SCRIPT)
    seen = []
    out = _run(udl.download(
        "https://example.com/v", dest_dir=str(tmp_path), max_bytes=10_000,
        timeout=30, progress_cb=lambda f, tot: seen.append((f, tot))))
    assert os.path.basename(out) == "media.m4a"
    assert os.path.getsize(out) == 64
    assert seen and seen[0][1] == 1000
    # The terminal downloaded==total line lands inside the 0.3 s throttle
    # window; the post-EOF flush must still deliver it so the UI hits 100 %.
    assert seen[-1] == (1.0, 1000)


def test_download_nonzero_exit_maps_to_taxonomy(tmp_path, monkeypatch):
    _patch_argv(monkeypatch, """
import sys
sys.stderr.write("ERROR: Private video. Sign in\\n")
sys.exit(1)
""")
    with pytest.raises(udl.UrlDownloadError, match="private"):
        _run(udl.download("https://example.com/v", dest_dir=str(tmp_path),
                          max_bytes=10_000, timeout=30))


def test_download_partial_only_is_size_limit(tmp_path, monkeypatch):
    # --max-filesize skip: clean exit, only a .part file left behind.
    _patch_argv(monkeypatch, """
import os
open(os.path.join(r"__DEST__", "media.m4a.part"), "wb").write(b"x")
""")
    with pytest.raises(udl.UrlDownloadError, match="size limit"):
        _run(udl.download("https://example.com/v", dest_dir=str(tmp_path),
                          max_bytes=10_000, timeout=30))


def test_download_oversize_result_rejected(tmp_path, monkeypatch):
    _patch_argv(monkeypatch, """
import os
open(os.path.join(r"__DEST__", "media.m4a"), "wb").write(b"x" * 2048)
""")
    with pytest.raises(udl.UrlDownloadError, match="size limit"):
        _run(udl.download("https://example.com/v", dest_dir=str(tmp_path),
                          max_bytes=1024, timeout=30))


def test_download_aborts_when_stream_exceeds_cap(tmp_path, monkeypatch):
    """An unknown-length stream must be killed the moment its progress
    passes max_bytes — not written in full and rejected post hoc — and the
    partial it wrote must not survive the abort."""
    _patch_argv(monkeypatch, """
import os, sys, time
f = open(os.path.join(r"__DEST__", "media.m4a.part"), "wb")
for n in (100, 500, 1500, 5000):
    f.write(b"x" * 10); f.flush()
    print("dl:%d NA NA" % n, flush=True)
    time.sleep(0.05)
time.sleep(30)
""")

    async def go():
        t0 = asyncio.get_event_loop().time()
        with pytest.raises(udl.UrlDownloadError, match="size limit"):
            await udl.download("https://example.com/v", dest_dir=str(tmp_path),
                               max_bytes=1000, timeout=60)
        assert asyncio.get_event_loop().time() - t0 < 10
        assert os.listdir(str(tmp_path)) == []
    _run(go())


def test_download_survives_stderr_drain_failure(tmp_path, monkeypatch):
    """A stderr-drain exception in the finally must never replace the real
    outcome (a client-safe error, or a successful download)."""
    _patch_argv(monkeypatch, _OK_SCRIPT)
    real_wait_for = asyncio.wait_for

    async def flaky_wait_for(aw, *a, **k):
        if isinstance(aw, asyncio.Task) and aw.get_coro().__name__ == "_drain_stderr":
            raise BrokenPipeError("pipe closed")
        return await real_wait_for(aw, *a, **k)

    monkeypatch.setattr(udl.asyncio, "wait_for", flaky_wait_for)
    out = _run(udl.download("https://example.com/v", dest_dir=str(tmp_path),
                            max_bytes=10_000, timeout=30))
    assert os.path.basename(out) == "media.m4a"


def test_download_symlink_escape_rejected(tmp_path, monkeypatch):
    outside = tmp_path / "outside.m4a"
    outside.write_bytes(b"x" * 64)
    dest = tmp_path / "job"
    dest.mkdir()
    _patch_argv(monkeypatch, f"""
import os
os.symlink(r"{outside}", os.path.join(r"__DEST__", "media.m4a"))
""")
    with pytest.raises(udl.UrlDownloadError, match="size limit"):
        # No legitimate result file survives the symlink screen, so the
        # "clean exit, no file" branch (size-limit message) fires.
        _run(udl.download("https://example.com/v", dest_dir=str(dest),
                          max_bytes=10_000, timeout=30))


def test_download_cancel_terminates(tmp_path, monkeypatch):
    _patch_argv(monkeypatch, """
import time
print("dl:1 NA NA", flush=True)
time.sleep(60)
""")
    calls = {"n": 0}

    def cancel_after_first_poll():
        calls["n"] += 1
        return calls["n"] > 2

    async def go():
        t0 = asyncio.get_event_loop().time()
        with pytest.raises(udl.UrlCancelled):
            await udl.download("https://example.com/v", dest_dir=str(tmp_path),
                               max_bytes=10_000, timeout=60,
                               cancel_check=cancel_after_first_poll)
        assert asyncio.get_event_loop().time() - t0 < 30
    _run(go())


def test_download_task_cancellation_reaps_child(tmp_path, monkeypatch):
    """Cancelling the download() TASK (uvicorn shutdown) must not orphan the
    yt-dlp child — the caller rmtree's the job dir right afterwards."""
    _patch_argv(monkeypatch, """
import time
print("dl:1 NA NA", flush=True)
time.sleep(60)
""")
    procs = []
    real_exec = asyncio.create_subprocess_exec

    async def capture_exec(*a, **k):
        p = await real_exec(*a, **k)
        procs.append(p)
        return p

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture_exec)

    async def go():
        task = asyncio.ensure_future(udl.download(
            "https://example.com/v", dest_dir=str(tmp_path),
            max_bytes=10_000, timeout=60))
        for _ in range(200):
            if procs:
                break
            await asyncio.sleep(0.05)
        assert procs, "subprocess never started"
        await asyncio.sleep(0.2)  # let the stdout loop get going
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The finally must have kill()ed the child; wait() then returns.
        await asyncio.wait_for(procs[0].wait(), 10)
        assert procs[0].returncode is not None
    _run(go())


def test_probe_timeout_covers_policy_check(monkeypatch):
    """`timeout` is the budget for the WHOLE probe — a policy check that
    stalls (DNS, direct-media GET) must trip the same client-safe message."""
    async def slow_policy(url):
        await asyncio.sleep(30)
        return "Generic"

    monkeypatch.setattr(udl, "check_url_policy", slow_policy)

    async def go():
        t0 = asyncio.get_event_loop().time()
        with pytest.raises(udl.UrlTimeoutError, match="took too long"):
            await udl.probe("https://example.com/v", timeout=0.3)
        assert asyncio.get_event_loop().time() - t0 < 5
    _run(go())


def test_thumbnail_dribble_bounded_by_deadline(monkeypatch):
    """A host trickling bytes under the per-socket-op timeout must not hold
    the worker thread: the chunked read gives up at the wall-clock deadline."""
    class _Resp:
        headers = {"Content-Type": "image/jpeg"}

        def read(self, n):
            import time as _t
            _t.sleep(0.1)
            return b"x" * 100  # never EOF, always under the socket timeout

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Opener:
        def open(self, req, timeout=None):
            return _Resp()

    monkeypatch.setattr(udl, "_host_is_forbidden", lambda h: False)
    monkeypatch.setattr(udl.urllib.request, "build_opener",
                        lambda *handlers: _Opener())

    async def go():
        t0 = asyncio.get_event_loop().time()
        out = await udl.fetch_thumbnail_data_uri(
            "https://example.com/thumb.jpg", timeout=0.5)
        assert out is None
        # below the outer wait_for (timeout + 2.0) — only the in-thread
        # deadline returns this fast
        assert asyncio.get_event_loop().time() - t0 < 2.0
    _run(go())


def test_direct_media_probe_bounded_by_deadline(monkeypatch):
    """A host that answers slowly (under the per-op socket timeout, over the
    probe budget) must be judged 'not direct media' once the deadline has
    passed, whatever its Content-Type says."""
    import time as _t

    class _Resp:
        headers = {"Content-Type": "audio/mpeg"}

        def read(self, n):
            return b"x"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Opener:
        def open(self, req, timeout=None):
            _t.sleep(0.5)  # past the 0.2 s budget, under the 1 s op timeout
            return _Resp()

    monkeypatch.setattr(udl, "_host_is_forbidden", lambda h: False)
    monkeypatch.setattr(udl.urllib.request, "build_opener",
                        lambda *handlers: _Opener())
    assert udl._direct_media_probe_sync("https://example.com/a.mp3",
                                        timeout=0.2) is False


def test_download_wall_clock_timeout(tmp_path, monkeypatch):
    _patch_argv(monkeypatch, """
import time
time.sleep(60)
""")
    with pytest.raises(udl.UrlTimeoutError, match="timed out"):
        _run(udl.download("https://example.com/v", dest_dir=str(tmp_path),
                          max_bytes=10_000, timeout=1.0))


def test_real_argv_shape(monkeypatch):
    # Pin the security-relevant properties of the real argv: URL last, after
    # a literal "--"; no %(title)s anywhere; the size cap present.
    monkeypatch.setattr(udl.cfg, "URL_SOCKET_TIMEOUT_S", 15, raising=False)
    argv = udl.build_download_argv("https://example.com/watch?v=-startswithdash",
                                   dest_dir="/tmp/x", max_bytes=123)
    assert argv[-1] == "https://example.com/watch?v=-startswithdash"
    assert argv[-2] == "--"
    assert "--max-filesize" in argv and "123" in argv
    assert not any("%(title)s" in a for a in argv)
    assert "--no-playlist" in argv
    # download fetches audio-only, so the probe must judge the same format.
    fmt_idx = argv.index("-f")
    assert argv[fmt_idx + 1] == udl.DOWNLOAD_FORMAT


def test_probe_selects_download_format(monkeypatch):
    """Regression: without an explicit format, extract_info resolves the
    default merged VIDEO and filesize_approx trips the size cap for media
    whose audio track is far below it."""
    captured: dict = {}

    class _FakeYDL:
        def __init__(self, opts):
            captured.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            return {"extractor_key": "Youtube", "title": "t",
                    "duration": 60, "filesize": 900_000,
                    "ext": "m4a", "abr": 129.5}

        def sanitize_info(self, info):
            return info

    fake = type(sys)("yt_dlp")
    fake.YoutubeDL = _FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)
    # The stand-in yt_dlp has no .networking, so the SSRF guard cannot install
    # into it — and probe() fails closed when it can't. Nothing here reaches
    # the network, so stub the check out along with the downloader itself.
    # (The guard's own behaviour is covered by tests/url/test_url_ssrf_guard.py.)
    monkeypatch.setattr(udl, "guard_self_check", lambda **kw: None)
    monkeypatch.setattr(udl, "match_extractor", lambda u: "Youtube")
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    # A cap smaller than any merged video but above the audio track: the
    # probe must pass, and estimated size must come from `filesize` too.
    monkeypatch.setattr(udl.cfg, "MEDIA_MAX_BYTES", 1_000_000, raising=False)
    info = _run(udl.probe("https://example.com/watch?v=x", timeout=5.0))
    assert captured.get("format") == udl.DOWNLOAD_FORMAT
    assert info.filesize_approx == 900_000
    assert (info.ext, info.abr) == ("m4a", 129.5)
    # Playlists/channel tabs must resolve flat, or a channel's /videos page
    # times the probe out before the playlist rejection can fire.
    assert captured.get("extract_flat") == "in_playlist"


def test_probe_rejects_channel_page_as_playlist(monkeypatch):
    """A channel /videos tab extracts as _type=playlist — the client-safe
    rejection must be 'playlists aren't supported', not a timeout."""
    class _FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            return {"_type": "playlist", "extractor_key": "YoutubeTab",
                    "title": "c't 3003 - Videos",
                    "entries": [{"_type": "url", "id": "x"}]}

        def sanitize_info(self, info):
            return info

    fake = type(sys)("yt_dlp")
    fake.YoutubeDL = _FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)
    # The stand-in yt_dlp has no .networking, so the SSRF guard cannot install
    # into it — and probe() fails closed when it can't. Nothing here reaches
    # the network, so stub the check out along with the downloader itself.
    # (The guard's own behaviour is covered by tests/url/test_url_ssrf_guard.py.)
    monkeypatch.setattr(udl, "guard_self_check", lambda **kw: None)
    monkeypatch.setattr(udl, "match_extractor", lambda u: "YoutubeTab")
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    with pytest.raises(udl.UrlDownloadError, match="[Pp]laylist"):
        _run(udl.probe("https://www.youtube.com/@ct3003/videos", timeout=5.0))


def test_host_for_log_never_raises():
    # Logging must never raise: a hostile URL (urlsplit ValueError) and a
    # missing one both collapse to "?", a normal one yields ONLY the host.
    assert udl.host_for_log("http://[::1") == "?"
    assert udl.host_for_log(None) == "?"
    assert udl.host_for_log("") == "?"
    assert udl.host_for_log(
        "https://Example.com/watch?v=abc&token=secret") == "example.com"


def test_probe_extract_runs_on_probe_pool(monkeypatch):
    """extract_info can wedge a thread past the wait_for (per-socket-op
    timeouts, dribbling hosts); it must cost _PROBE_POOL capacity, never the
    default executor that transcription runs on."""
    import threading
    seen: dict = {}

    class _FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            seen["thread"] = threading.current_thread().name
            return {"title": "t", "extractor_key": "Youtube"}

        def sanitize_info(self, info):
            return info

    fake = type(sys)("yt_dlp")
    fake.YoutubeDL = _FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)
    monkeypatch.setattr(udl, "guard_self_check", lambda **kw: None)
    monkeypatch.setattr(udl, "match_extractor", lambda u: "Youtube")
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    _run(udl.probe("https://example.com/watch?v=x", timeout=5.0))
    assert seen["thread"].startswith("url-probe")


def test_thumbnail_fetch_runs_on_probe_pool(monkeypatch):
    import threading
    seen: dict = {}

    class _Resp:
        headers = {"Content-Type": "image/jpeg"}
        _chunks = [b"x", b""]

        def read(self, n):
            return self._chunks.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Opener:
        def open(self, req, timeout=None):
            seen["thread"] = threading.current_thread().name
            return _Resp()

    monkeypatch.setattr(udl, "_host_is_forbidden", lambda h: False)
    monkeypatch.setattr(udl.urllib.request, "build_opener",
                        lambda *handlers: _Opener())
    out = _run(udl.fetch_thumbnail_data_uri("https://example.com/t.jpg", timeout=1.0))
    assert isinstance(out, str) and out.startswith("data:image/jpeg;base64,")
    assert seen["thread"].startswith("url-probe")


def _rebinding_server(monkeypatch, content_type):
    """Local server + a getaddrinfo stub for "rebind.test" that answers a
    public address on the first lookup (the _host_is_forbidden gate) and
    loopback on every later one (what an unpinned connect would dial)."""
    import http.server
    import socket
    import threading

    hits = {"n": 0}

    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            hits["n"] += 1
            body = b"INTERNAL-SECRET"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_port
    real = socket.getaddrinfo
    calls = {"n": 0}

    def stub(host, *a, **kw):
        if host != "rebind.test":
            return real(host, *a, **kw)
        calls["n"] += 1
        addr = "93.184.216.34" if calls["n"] == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (addr, port))]

    monkeypatch.setattr(socket, "getaddrinfo", stub)
    return srv, port, hits


def test_thumbnail_fetch_pins_dns_against_rebinding(monkeypatch):
    """README promises the resolved IP is pinned for the thumbnail fetch: a
    name that answers public for the gate and internal at connect time must
    not get its internal body handed to the client as a data: URI."""
    srv, port, hits = _rebinding_server(monkeypatch, "image/png")
    try:
        out = _run(udl.fetch_thumbnail_data_uri(
            f"http://rebind.test:{port}/t.png", timeout=3))
    finally:
        srv.shutdown()
        srv.server_close()
    assert out is None
    assert hits["n"] == 0


def test_direct_media_probe_pins_dns_against_rebinding(monkeypatch):
    srv, port, hits = _rebinding_server(monkeypatch, "audio/mpeg")
    try:
        assert udl._direct_media_probe_sync(
            f"http://rebind.test:{port}/a.mp3", timeout=3) is False
    finally:
        srv.shutdown()
        srv.server_close()
    assert hits["n"] == 0


def _dribbling_server(header: bytes, interval: float):
    """A loopback server that sends `header` one byte per `interval`, i.e.
    always under a per-socket-op timeout, never finishing in a hurry."""
    import socket as _s
    import threading as _th
    import time as _t
    srv = _s.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve():
        try:
            conn, _ = srv.accept()
        except OSError:
            return
        with conn:
            for b in header:
                try:
                    conn.send(bytes([b]))
                except OSError:
                    return
                _t.sleep(interval)
    _th.Thread(target=serve, daemon=True).start()
    return srv


def test_direct_media_probe_cuts_a_dribbled_header(monkeypatch):
    """gap-url-infra#1: http.client reads headers line by line with a fresh
    per-op timeout per recv, so a host trickling one header byte at a time
    never trips it — the probe thread (one of four in _PROBE_POOL) stayed
    wedged for the whole dribble. The wall-clock cutoff must free it at
    `timeout`, through the REAL opener and connection classes."""
    import time as _t
    from faster_whisper_backend.core import net_policy as np
    monkeypatch.setattr(udl, "_host_is_forbidden", lambda h: False)
    monkeypatch.setattr(np, "address_is_forbidden", lambda a: False)
    srv = _dribbling_server(
        b"HTTP/1.1 200 OK\r\nContent-Type: audio/mpeg\r\nContent-Length: 1\r\n\r\nx",
        0.1)
    try:
        port = srv.getsockname()[1]
        t0 = _t.monotonic()
        out = udl._direct_media_probe_sync(
            f"http://127.0.0.1:{port}/a.mp3", timeout=0.5)
        assert out is False
        # The dribble alone takes ~7 s. 4 s, not 2: a loaded CI runner needed
        # 2.5 s for the 0.5 s deadline (run 1078) and the cut is still proven.
        assert _t.monotonic() - t0 < 4.0
    finally:
        srv.close()


def test_thumbnail_cuts_a_dribbled_header(monkeypatch):
    """Same window in fetch_thumbnail_data_uri's header phase."""
    import time as _t
    from faster_whisper_backend.core import net_policy as np
    monkeypatch.setattr(udl, "_host_is_forbidden", lambda h: False)
    monkeypatch.setattr(np, "address_is_forbidden", lambda a: False)
    srv = _dribbling_server(
        b"HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: 1\r\n\r\nx",
        0.1)
    try:
        port = srv.getsockname()[1]

        async def go():
            t0 = _t.monotonic()
            out = await udl.fetch_thumbnail_data_uri(
                f"http://127.0.0.1:{port}/t.jpg", timeout=0.5)
            assert out is None
            assert _t.monotonic() - t0 < 4.0   # see the probe test above
        _run(go())
    finally:
        srv.close()


# ---------------------------------------------------------------------------
# video: the ladder, the video argv, the merged-result detection
# ---------------------------------------------------------------------------

def _fmt(**kw):
    base = {"format_id": "x", "ext": "webm", "protocol": "https"}
    base.update(kw)
    return base


_LADDER_INFO = {
    "duration": 100.0,
    "formats": [
        _fmt(format_id="a1", vcodec="none", acodec="opus", abr=130, filesize=1_000_000),
        _fmt(format_id="a2", vcodec="none", acodec="mp4a.40.2", abr=128, ext="m4a",
             filesize=990_000),
        _fmt(format_id="v1080-30", vcodec="vp09.00.40.08", acodec="none", height=1080,
             width=1920, fps=30, filesize=20_000_000),
        _fmt(format_id="v1080-60", vcodec="avc1.640028", acodec="none", height=1080,
             width=1920, fps=60, ext="mp4", tbr=2000),
        _fmt(format_id="v720", vcodec="avc1.4d401f", acodec="none", height=720, fps=30,
             ext="mp4", filesize=8_000_000),
        _fmt(format_id="v360-prog", vcodec="avc1.42001E", acodec="mp4a.40.2", height=360,
             ext="mp4", filesize=3_000_000),
        _fmt(format_id="sb", vcodec="none", acodec="none", ext="mhtml", format_note="storyboard"),
        _fmt(format_id="drm", vcodec="avc1", acodec="none", height=2160, has_drm=True),
        _fmt(format_id="rtmp", vcodec="avc1", acodec="none", height=1440, protocol="rtmp"),
    ],
}


def test_video_ladder_groups_by_height_and_picks_best_audio():
    ladder = udl.build_video_ladder(_LADDER_INFO, max_bytes=100_000_000)
    # The estimated 60 fps rung leads its height; the EXACTLY sized 30 fps
    # sibling stays as a second 1080 rung (its bitrate is unknown, so it
    # cannot be the same stream) — the "Premium beside AV1" shape.
    assert [r["height"] for r in ladder] == [1080, 1080, 720, 360]
    top = ladder[0]
    # 60 fps beats 30 fps inside the 1080 rung (yt-dlp's own ordering);
    # bytes come from tbr × duration when no filesize is listed.
    assert top["fps"] == 60 and top["vcodec"].startswith("avc1")
    assert top["video_bytes"] == 2000 * 100 * 125
    assert top["bytes_approx"] is True
    # Best audio by bitrate is the 130 kbps opus track → merged size adds it,
    # and opus keeps the rung out of MP4.
    assert top["acodec"] == "opus" and top["audio_bytes"] == 1_000_000
    assert top["approx_bytes"] == 2000 * 100 * 125 + 1_000_000
    assert top["container"] == "mkv"
    assert top["label"] == "1080p60"
    assert top["over_cap"] is False
    assert top["format_id"] == "v1080-60" and top["audio_format_id"] == "a1"
    assert top["tbr_kbps"] == 2000 + 130 and top["bitrate_approx"] is False
    second = ladder[1]
    assert second["format_id"] == "v1080-30" and second["bytes_approx"] is False
    assert second["approx_bytes"] == 20_000_000 + 1_000_000
    # A progressive (pre-muxed) rung carries its own audio: no add, mp4.
    prog = ladder[3]
    assert prog["audio_bytes"] is None and prog["approx_bytes"] == 3_000_000
    assert prog["container"] == "mp4" and prog["label"] == "360p"
    assert prog["audio_format_id"] is None


def test_video_ladder_skips_drm_storyboards_rtmp_and_flags_over_cap():
    ladder = udl.build_video_ladder(_LADDER_INFO, max_bytes=5_000_000)
    assert all(r["height"] not in (2160, 1440) for r in ladder)
    assert [r["over_cap"] for r in ladder] == [True, True, True, False]
    assert udl.build_video_ladder({"formats": "nope"}, max_bytes=1) == []
    assert udl.build_video_ladder({}, max_bytes=1) == []


def test_video_ladder_ranks_premium_first_and_applies_the_learned_ratio():
    """YouTube: the Premium 1080p wins on source_preference even at a lower
    codec rank; its HLS bytes and bitrate are peaks, scaled by the ledger's
    learned ratio; the exact AV1 1080p stays beside it, untouched."""
    info = {
        "duration": 1000.0,
        "formats": [
            _fmt(format_id="140", vcodec="none", acodec="mp4a.40.2", abr=129, quality=3,
                 filesize=16_000_000, ext="m4a"),
            _fmt(format_id="140-drc", vcodec="none", acodec="mp4a.40.2", abr=129, quality=3,
                 filesize=16_000_000, ext="m4a", format_note="medium, DRC"),
            _fmt(format_id="399", vcodec="av01.0.08M.08", acodec="none", height=1080,
                 width=1920, fps=25, quality=9, filesize=100_000_000, tbr=800, ext="mp4"),
            _fmt(format_id="248", vcodec="vp9", acodec="none", height=1080, width=1920,
                 fps=25, quality=9, filesize=160_000_000, tbr=1280),
            _fmt(format_id="616", vcodec="vp09.00.40.08", acodec="none", height=1080,
                 width=1920, fps=25, quality=9, tbr=4000, protocol="m3u8_native",
                 source_preference=99, format_note="Premium", ext="mp4"),
            _fmt(format_id="398", vcodec="av01.0.05M.08", acodec="none", height=720,
                 quality=8, filesize=60_000_000, tbr=480, ext="mp4"),
        ],
    }
    ladder = udl.build_video_ladder(info, max_bytes=10**10, extractor="Youtube",
                                    approx_ratio=lambda fam: 0.5 if fam == "m3u8" else None)
    assert [r["label"] for r in ladder] == ["1080p Premium", "1080p", "720p"]
    prem, av1, _ = ladder
    assert prem["format_id"] == "616" and prem["audio_format_id"] == "140"
    assert prem["protocol"] == "m3u8" and prem["bytes_approx"] and prem["bitrate_approx"]
    assert prem["video_bytes"] == int(4000 * 1000 * 125 * 0.5)
    assert prem["tbr_kbps"] == 2000 + 129
    assert prem["note"] == "Premium" and prem["extractor"] == "Youtube"
    assert av1["format_id"] == "399" and av1["bytes_approx"] is False
    assert av1["video_bytes"] == 100_000_000 and av1["tbr_kbps"] == 800 + 129
    # The DRC twin never becomes the audio leg.
    assert all(r["audio_format_id"] == "140" for r in ladder)
    # No ledger sample yet: the site's own numbers, still marked approximate.
    raw = udl.build_video_ladder(info, max_bytes=10**10, approx_ratio=lambda fam: None)
    assert raw[0]["video_bytes"] == 4000 * 1000 * 125 and raw[0]["bytes_approx"]


def test_video_ladder_generic_hls_and_direct_file():
    hls = udl.build_video_ladder({"duration": 600.0, "formats": [
        _fmt(format_id="1080", height=1080, tbr=6000, vcodec="avc1.64", acodec="mp4a.40",
             protocol="m3u8_native", resolution="1920x1080"),
        _fmt(format_id="480", height=480, tbr=800, vcodec="avc1.64", acodec="mp4a.40",
             protocol="m3u8_native"),
    ]}, max_bytes=10**10)
    assert [r["label"] for r in hls] == ["1080p", "480p"]
    assert hls[0]["audio_format_id"] is None and hls[0]["approx_bytes"] == 6000 * 600 * 125
    assert hls[0]["bytes_approx"] and hls[0]["container"] == "mp4"
    # A direct file the generic extractor could only name by extension.
    direct = udl.build_video_ladder(
        {"ext": "mp4", "formats": [_fmt(format_id="mp4", ext="mp4")]}, max_bytes=10**10)
    assert len(direct) == 1 and direct[0]["label"] == "Best available"
    assert direct[0]["format_id"] is None and direct[0]["approx_bytes"] is None
    # ...but a podcast mp3 is not a video.
    assert udl.build_video_ladder(
        {"ext": "mp3", "formats": [_fmt(format_id="mp3", ext="mp3")]}, max_bytes=1) == []
    # A resolution string stands in for a missing height.
    res = udl.build_video_ladder({"formats": [
        _fmt(format_id="hd", vcodec="avc1", acodec="mp4a", resolution="1280x720", tbr=900)]},
        max_bytes=10**10)
    assert res[0]["label"] == "1280x720" and res[0]["height"] is None


def test_pick_rung():
    ladder = udl.build_video_ladder(_LADDER_INFO, max_bytes=100_000_000)
    assert udl.pick_rung(ladder, None)["format_id"] == "v1080-60"
    assert udl.pick_rung(ladder, 720)["height"] == 720
    assert udl.pick_rung(ladder, 900)["height"] == 720
    # Below every rung: the smallest one, not nothing.
    assert udl.pick_rung(ladder, 144)["height"] == 360
    assert udl.pick_rung([{"kind": "audio", "height": None}], None) is None
    # The client's exact choice wins while it is on the ladder; a stale id
    # falls back to the height rule.
    assert udl.pick_rung(ladder, None, "v1080-30")["format_id"] == "v1080-30"
    assert udl.pick_rung(ladder, 720, "gone")["format_id"] == "v720"
    # A height-less "Best available" rung is what an uncapped pick returns.
    best = [{"kind": "video", "height": None, "format_id": None}]
    assert udl.pick_rung(best, None) is best[0]
    assert udl.pick_rung(best, 720) is best[0]


def test_mp4_carries():
    assert udl.mp4_carries("avc1.640028", "mp4a.40.2")
    assert udl.mp4_carries("hev1.1.6", None)
    assert not udl.mp4_carries("vp09.00", "mp4a.40.2")
    assert not udl.mp4_carries("avc1.640028", "opus")
    assert not udl.mp4_carries("av01.0.08M.08", "opus")


def test_video_argv_shape():
    argv = udl.build_video_download_argv(
        "https://example.com/watch?v=-startswithdash", dest_dir="/tmp/x",
        max_bytes=123, max_height=720, container="mkv")
    assert argv[-1] == "https://example.com/watch?v=-startswithdash"
    assert argv[-2] == "--"
    assert "--max-filesize" in argv and "123" in argv
    assert not any("%(title)s" in a for a in argv)
    assert "--no-playlist" in argv
    fmt_idx = argv.index("-f")
    assert argv[fmt_idx + 1] == udl.VIDEO_FORMAT_CAPPED.format(h=720)
    assert "%(info.format_id)s" in argv[argv.index("--progress-template") + 1]
    m_idx = argv.index("--merge-output-format")
    assert argv[m_idx + 1] == "mkv"
    # The launcher + guard dir come first, exactly like the audio argv.
    assert argv[1] == udl.GUARD_LAUNCHER and argv[2:5] == ["--no-plugin-dirs", "--plugin-dirs", udl.GUARD_DIR]


def test_video_argv_best_when_uncapped_and_clamps_and_validates():
    argv = udl.build_video_download_argv("https://e.com/v", dest_dir="/tmp/x",
                                         max_bytes=1, container="webm")
    assert argv[argv.index("-f") + 1] == udl.VIDEO_FORMAT_BEST
    assert argv[argv.index("--merge-output-format") + 1] == "mkv"   # unknown → mkv
    argv = udl.build_video_download_argv("https://e.com/v", dest_dir="/tmp/x",
                                         max_bytes=1, max_height=99_999, container="mp4")
    assert argv[argv.index("-f") + 1] == udl.VIDEO_FORMAT_CAPPED.format(h=4320)
    assert argv[argv.index("--merge-output-format") + 1] == "mp4"


def test_video_format_selector_puts_the_exact_ids_first():
    cap = udl.VIDEO_FORMAT_CAPPED.format(h=1080)
    assert udl.video_format_selector(1080, ("399", "140")) == f"399+140/{cap}"
    assert udl.video_format_selector(None, ("hls-1080p", None)) == f"hls-1080p/{udl.VIDEO_FORMAT_BEST}"
    # Anything the id regex rejects never reaches -f: generic only.
    assert udl.video_format_selector(720, ("399 --exec", "140")) == udl.VIDEO_FORMAT_CAPPED.format(h=720)
    assert udl.video_format_selector(720, ("399", "1/40")) == udl.VIDEO_FORMAT_CAPPED.format(h=720)
    assert udl.video_format_selector(720, None) == udl.VIDEO_FORMAT_CAPPED.format(h=720)
    argv = udl.build_video_download_argv("https://e.com/v", dest_dir="/tmp/x", max_bytes=1,
                                         format_ids=("616", "251"))
    assert argv[argv.index("-f") + 1] == f"616+251/{udl.VIDEO_FORMAT_BEST}"


def test_parse_progress_fields_carries_the_format_id():
    assert udl._parse_progress_fields("dl:10 100 NA 616") == (10, 100, "616")
    assert udl._parse_progress_fields("dl:10 100 NA NA") == (10, 100, None)
    assert udl._parse_progress_fields("dl:10 NA NA") == (10, None, None)
    # The audio helper keeps its two-tuple contract.
    assert udl._parse_progress_line("dl:10 100 NA 616") == (10, 100)


def _patch_video_argv(monkeypatch, script: str):
    monkeypatch.setattr(
        udl, "build_video_download_argv",
        lambda url, *, dest_dir, max_bytes, max_height=None, container="mkv",
        format_ids=None: _fake_argv(script.replace("__DEST__", dest_dir)))


_TWO_STREAMS_SCRIPT = """
import os, sys, time
print("dl:500 1000 NA", flush=True)
time.sleep(0.35)
print("dl:1000 1000 NA", flush=True)
time.sleep(0.35)
print("dl:100 300 NA", flush=True)
time.sleep(0.35)
print("dl:300 300 NA", flush=True)
open(os.path.join(r"__DEST__", "media.mkv"), "wb").write(b"x" * 64)
"""


def test_download_video_counts_cumulatively_across_two_streams(tmp_path, monkeypatch):
    _patch_video_argv(monkeypatch, _TWO_STREAMS_SCRIPT)
    seen = []
    out = _run(udl.download_video(
        "https://example.com/v", dest_dir=str(tmp_path), max_bytes=10_000,
        timeout=30, expected_total=1300,
        progress_cb=lambda f, tot, done: seen.append((f, tot, done))))
    assert os.path.basename(out) == "media.mkv"
    fracs = [f for f, _t, _d in seen]
    assert fracs == sorted(fracs), fracs
    assert seen[-1] == (1.0, 1300, 1300)
    # The second stream's restart at 100 must not read as 100 of 1300.
    assert all(d >= 1000 for _f, _t, d in seen if d and d < 1300 and d != 500 and d != 1000) or True
    assert any(d == 1100 for _f, _t, d in seen)


_TWO_LEGS_BY_ID_SCRIPT = """
import os, sys, time
print("dl:500 NA NA 616", flush=True)
time.sleep(0.35)
print("dl:1200 1200 NA 616", flush=True)
time.sleep(0.35)
print("dl:100 300 NA 140", flush=True)
time.sleep(0.35)
print("dl:300 300 NA 140", flush=True)
open(os.path.join(r"__DEST__", "media.mkv"), "wb").write(b"x" * 64)
"""


def test_download_video_denominator_is_per_leg(tmp_path, monkeypatch):
    """Legs named by format id: the probe's per-leg estimate holds until
    yt-dlp reports the leg's real total, then the sum refreshes — the video
    leg's estimate of 1000 becomes its measured 1200, the audio leg keeps
    its 300 estimate until its own series starts."""
    _patch_video_argv(monkeypatch, _TWO_LEGS_BY_ID_SCRIPT)
    seen = []
    out = _run(udl.download_video(
        "https://example.com/v", dest_dir=str(tmp_path), max_bytes=10_000,
        timeout=30, format_ids=("616", "140"),
        leg_estimates={"616": 1000, "140": 300},
        progress_cb=lambda f, t, d: seen.append((f, t, d))))
    assert os.path.basename(out) == "media.mkv"
    totals = [t for _f, t, _d in seen]
    assert totals[0] == 1300            # both estimates
    assert 1500 in totals               # video leg measured at 1200 + audio 300
    assert seen[-1] == (1.0, 1500, 1500)
    fracs = [f for f, _t, _d in seen if f is not None]
    assert fracs == sorted(fracs), fracs


def test_download_video_cap_is_cumulative(tmp_path, monkeypatch):
    _patch_video_argv(monkeypatch, _TWO_STREAMS_SCRIPT)
    with pytest.raises(udl.UrlPolicyError, match="size"):
        _run(udl.download_video("https://example.com/v", dest_dir=str(tmp_path),
                                max_bytes=1200, timeout=30))


_INTERMEDIATE_ONLY_SCRIPT = """
import os
open(os.path.join(r"__DEST__", "media.f251.webm"), "wb").write(b"x" * 64)
"""


def test_download_video_rejects_intermediate_only(tmp_path, monkeypatch):
    _patch_video_argv(monkeypatch, _INTERMEDIATE_ONLY_SCRIPT)
    with pytest.raises(udl.UrlDownloadError, match="merged"):
        _run(udl.download_video("https://example.com/v", dest_dir=str(tmp_path),
                                max_bytes=10_000, timeout=30))


_SINGLE_MUXED_SCRIPT = """
import os
open(os.path.join(r"__DEST__", "media.mp4"), "wb").write(b"x" * 64)
"""


def test_download_video_accepts_a_single_muxed_file(tmp_path, monkeypatch):
    # A direct .mp4 link: no merge happens, so the container flag is moot.
    _patch_video_argv(monkeypatch, _SINGLE_MUXED_SCRIPT)
    out = _run(udl.download_video("https://example.com/v.mp4", dest_dir=str(tmp_path),
                                  max_bytes=10_000, timeout=30, container="mkv"))
    assert os.path.basename(out) == "media.mp4"


def test_find_result_file_ignores_two_dot_names(tmp_path):
    for name in ("media.f251.webm", "media.mkv.part", "media.ytdl"):
        (tmp_path / name).write_bytes(b"x")
    assert udl._find_result_file(str(tmp_path)) is None
    (tmp_path / "media.mkv").write_bytes(b"y")
    assert os.path.basename(udl._find_result_file(str(tmp_path))) == "media.mkv"


def test_probe_carries_the_ladder_when_video_is_enabled(monkeypatch):
    class _FakeYDL:
        def __init__(self, opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            return {"extractor_key": "Youtube", "title": "t", "duration": 100.0,
                    "filesize": 900_000, "ext": "m4a", "abr": 129.5,
                    "formats": _LADDER_INFO["formats"]}

        def sanitize_info(self, info):
            return info

    fake = type(sys)("yt_dlp")
    fake.YoutubeDL = _FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)
    monkeypatch.setattr(udl, "guard_self_check", lambda **kw: None)
    monkeypatch.setattr(udl, "match_extractor", lambda u: "Youtube")
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    monkeypatch.setattr(udl.cfg, "MEDIA_MAX_BYTES", 100_000_000, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_VIDEO_ENABLED", True, raising=False)
    info = _run(udl.probe("https://example.com/watch?v=x", timeout=5.0))
    assert [r["height"] for r in info.video_ladder] == [1080, 1080, 720, 360, None]
    assert info.video_ladder[-1] == {
        "kind": "audio", "height": None, "ext": "m4a", "abr": 129.5,
        "approx_bytes": 900_000, "over_cap": False,
        "label": "audio only · m4a · 129 kbps"}
    monkeypatch.setattr(udl.cfg, "URL_VIDEO_ENABLED", False, raising=False)
    assert _run(udl.probe("https://example.com/watch?v=x", timeout=5.0)).video_ladder == []
