"""Unit tests for url_download.py — no network, no real yt-dlp subprocess.

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

import url_download as udl


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
    monkeypatch.setattr(udl.cfg, "URL_MAX_DURATION_SEC", 60, raising=False)
    with pytest.raises(udl.UrlDownloadError, match="limit"):
        udl._policy_check_info({"duration": 61})
    udl._policy_check_info({"duration": 59})  # under: no raise


def test_policy_rejects_over_filesize(monkeypatch):
    monkeypatch.setattr(udl.cfg, "URL_MAX_BYTES", 1000, raising=False)
    with pytest.raises(udl.UrlDownloadError, match="size"):
        udl._policy_check_info({"filesize_approx": 2000})


def test_effective_max_bytes_inherits_upload_cap(monkeypatch):
    monkeypatch.setattr(udl.cfg, "URL_MAX_BYTES", 0, raising=False)
    monkeypatch.setattr(udl.cfg, "MAX_UPLOAD_BYTES", 12345, raising=False)
    assert udl._effective_max_bytes() == 12345
    monkeypatch.setattr(udl.cfg, "URL_MAX_BYTES", 99, raising=False)
    assert udl._effective_max_bytes() == 99


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
    with pytest.raises(udl.UrlDownloadError, match="allowed list"):
        _run(udl.check_url_policy("https://x/"))


def test_empty_allowlist_admits_any_dedicated_extractor(monkeypatch):
    monkeypatch.setattr(udl, "match_extractor", lambda u: "SoundCloud")
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    assert _run(udl.check_url_policy("https://x/")) == "SoundCloud"


def test_generic_rejected_by_default(monkeypatch):
    monkeypatch.setattr(udl, "match_extractor", lambda u: "Generic")
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", False, raising=False)
    with pytest.raises(udl.UrlDownloadError):
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
        with pytest.raises(udl.UrlDownloadError, match="took too long"):
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
        assert asyncio.get_event_loop().time() - t0 < 5
    _run(go())


def test_direct_media_probe_bounded_by_deadline(monkeypatch):
    """A host that answers slowly (under the per-op socket timeout, over the
    probe budget) must not pin the probe thread: give up at the deadline."""
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
    t0 = _t.monotonic()
    assert udl._direct_media_probe_sync("https://example.com/a.mp3",
                                        timeout=0.2) is False
    assert _t.monotonic() - t0 < 2


def test_download_wall_clock_timeout(tmp_path, monkeypatch):
    _patch_argv(monkeypatch, """
import time
time.sleep(60)
""")
    with pytest.raises(udl.UrlDownloadError, match="timed out"):
        _run(udl.download("https://example.com/v", dest_dir=str(tmp_path),
                          max_bytes=10_000, timeout=1.0))


def test_real_argv_shape(monkeypatch):
    # Pin the security-relevant properties of the real argv: URL last, after
    # a literal "--"; no %(title)s anywhere; the size cap present.
    monkeypatch.setattr(udl.cfg, "URL_SOCKET_TIMEOUT_SEC", 15, raising=False)
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
    # (The guard's own behaviour is covered by tests/test_url_ssrf_guard.py.)
    monkeypatch.setattr(udl, "guard_self_check", lambda **kw: None)
    monkeypatch.setattr(udl, "match_extractor", lambda u: "Youtube")
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    # A cap smaller than any merged video but above the audio track: the
    # probe must pass, and estimated size must come from `filesize` too.
    monkeypatch.setattr(udl.cfg, "URL_MAX_BYTES", 1_000_000, raising=False)
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
    # (The guard's own behaviour is covered by tests/test_url_ssrf_guard.py.)
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
