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
    return asyncio.new_event_loop().run_until_complete(coro)


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


def test_unresolvable_host_is_forbidden():
    assert udl._host_is_forbidden("definitely-not-a-real-host.invalid") is True


# ---------------------------------------------------------------------------
# progress-template parsing
# ---------------------------------------------------------------------------

def test_parse_progress_line_well_formed():
    assert udl._parse_progress_line("dl:1024 4096 NA") == (1024, 4096)


def test_parse_progress_line_estimate_fallback():
    assert udl._parse_progress_line("dl:10 NA 200") == (10, 200)


def test_parse_progress_line_unknown_total():
    assert udl._parse_progress_line("dl:10 NA NA") == (10, None)


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
