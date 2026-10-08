"""Unit tests for media/download.py — no network, no real yt-dlp subprocess.

The download() tests monkeypatch build_download_argv to run a tiny inline
Python script that mimics yt-dlp's observable behavior (progress lines on
stdout, an output file, exit codes), so the full subprocess plumbing —
progress parsing, cancellation, timeouts, result validation — runs for real.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest

from faster_whisper_backend.media import download as udl


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


def test_validate_url_rejects_a_host_idna_cannot_encode():
    # An empty or >63-char label makes getaddrinfo raise UnicodeError
    # downstream (a 500 from url-preview); refuse it here as a client error.
    for url in ("https://a..com/x.mp3", "https://" + "a" * 64 + ".com/x.mp3"):
        with pytest.raises(udl.UrlDownloadError, match="host name is invalid"):
            udl.validate_url(url)
    assert udl.validate_url("https://bücher.de/x.mp3") == "https://bücher.de/x.mp3"


def test_validate_url_rejects_a_lone_surrogate():
    """A JSON body can carry "\\ud800" in the path or query: it could never
    reach the download child's argv (UnicodeEncodeError → a 500 after the
    probe), so it is a client error up front."""
    for url in ("https://example.com/a\ud800",
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ&x=\udfff"):
        with pytest.raises(udl.UrlDownloadError, match="invalid characters"):
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
    # Some extractors set only live_status.
    with pytest.raises(udl.UrlDownloadError, match="[Ll]ive"):
        udl._policy_check_info({"live_status": "is_live"})


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


def _stand_in_yt_dlp(monkeypatch, matched, info):
    """probe() with the offline match saying `matched` and a stand-in yt-dlp
    whose extract_info returns `info` (a dict, or a callable of the URL that
    returns or raises). Returns the YoutubeDL opts the probe built.

    The stand-in has no .networking, so the SSRF guard cannot install into
    it — and probe() fails closed when it can't. Nothing here reaches the
    network, so the check is stubbed out along with the downloader itself
    (the guard's own behaviour is covered by test_ssrf_guard.py)."""
    captured: dict = {}

    class _FakeYDL:
        def __init__(self, opts):
            captured.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            return info(url) if callable(info) else info

        def sanitize_info(self, info):
            return info

    fake = type(sys)("yt_dlp")
    fake.YoutubeDL = _FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake)
    monkeypatch.setattr(udl, "guard_self_check", lambda **kw: None)
    monkeypatch.setattr(udl, "match_extractor", lambda u: matched)
    monkeypatch.setattr(udl, "_direct_media_probe_sync",
                        lambda u, timeout, **kw: True)
    return captured


def _probe_with_extractor(monkeypatch, matched, ran):
    """probe() with the offline match saying `matched` and a stand-in
    yt-dlp whose extraction ended in the extractor `ran`."""
    return _stand_in_yt_dlp(monkeypatch, matched,
                            {"title": "t", "extractor_key": ran})


def test_probe_refuses_a_direct_media_link_handed_to_a_site_extractor(monkeypatch):
    """A Generic URL admitted only as direct media (the probe's distinctive
    request saw audio/*) must not come back from a site extractor GenericIE
    delegated to — that is the site policy bypassed, not a media file."""
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    captured = _probe_with_extractor(monkeypatch, "Generic", "Youtube")
    with pytest.raises(udl.UrlPolicyError):
        _run(udl.probe("https://x/a.mp3", timeout=5.0))
    # ...and the extraction itself was pinned to GenericIE.
    assert captured["allowed_extractors"] == ["generic"]


def test_probe_refuses_a_delegation_off_the_allowlist(monkeypatch):
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", ["Youtube"], raising=False)
    _probe_with_extractor(monkeypatch, "Youtube", "Vimeo")
    # The stand-in yt_dlp has no extractor registry to map the allowlist on.
    monkeypatch.setattr(udl, "pinned_extractors", lambda key: ["youtube"])
    # "leads to a site": the delegation gate's wording, not the offline
    # gate's ("isn't on the server's allowed list") — a regression there
    # must not keep this test green.
    with pytest.raises(udl.UrlPolicyError, match="leads to a site"):
        _run(udl.probe("https://x/watch", timeout=5.0))


def test_probe_refuses_a_site_extractor_handing_off_to_generic(monkeypatch):
    """Default config (no allowlist, so no pin): a site extractor that
    url_result()s a scraped embed URL lands on GenericIE, which scrapes any
    page — the widening URL_ALLOW_GENERIC=off forbids. A hand-off target
    that passed the direct-media Content-Type check is what
    URL_ALLOW_DIRECT_MEDIA admits — yt-dlp's "direct" flag alone is not."""
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    with pytest.raises(udl.UrlPolicyError, match="leads to a site"):
        udl._policy_check_extractor("Youtube", {"extractor_key": "Generic"})
    with pytest.raises(udl.UrlPolicyError, match="leads to a site"):
        udl._policy_check_extractor(
            "Youtube", {"extractor_key": "Generic", "direct": True})
    udl._policy_check_extractor(
        "Youtube", {"extractor_key": "Generic"}, handoff_is_media=True)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", False, raising=False)
    with pytest.raises(udl.UrlPolicyError, match="leads to a site"):
        udl._policy_check_extractor(
            "Youtube", {"extractor_key": "Generic"}, handoff_is_media=True)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", True, raising=False)
    udl._policy_check_extractor("Youtube", {"extractor_key": "Generic"})


def _handoff_policy(monkeypatch):
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)


def test_probe_admits_an_hls_manifest_handed_off_to_generic(monkeypatch):
    """GenericIE turns an audio/mpegurl manifest into m3u8 formats and never
    sets "direct" — yet the same URL pasted directly passes check_url_policy.
    The hand-off is judged by that same Content-Type check, on the manifest
    GenericIE fetched. The info is url_transparent-shaped: yt-dlp overlays
    the outer site result, so webpage_url is the SITE page (the probed
    URL), not the manifest."""
    _handoff_policy(monkeypatch)
    fmt = {"format_id": "hls-1", "protocol": "m3u8_native",
           "url": "https://cdn.example/live/variant.m3u8",
           "manifest_url": "https://cdn.example/live.m3u8"}
    _stand_in_yt_dlp(monkeypatch, "Youtube", {
        "title": "t", "extractor_key": "Generic",
        "webpage_url": "https://x/watch",
        **fmt, "formats": [fmt]})
    probed: "list[str]" = []

    def _media(u, timeout, accept_hls=False):
        probed.append(u)
        return accept_hls
    monkeypatch.setattr(udl, "_direct_media_probe_sync", _media)
    assert _run(udl.probe("https://x/watch", timeout=5.0)).extractor_key == "Generic"
    assert probed == ["https://cdn.example/live.m3u8"]


def test_probe_bounds_the_hand_off_check_by_the_probe_deadline(monkeypatch):
    """The hand-off re-check (DNS outside the socket cutoff, a busy probe
    pool) spends from the probe's one wall-clock budget like the steps
    before it: "took too long", not a request running past the timeout."""
    import threading
    _handoff_policy(monkeypatch)
    _stand_in_yt_dlp(monkeypatch, "Youtube", {
        "title": "t", "extractor_key": "Generic",
        "webpage_url": "https://cdn.example/live.m3u8",
        "protocol": "m3u8_native"})
    release = threading.Event()

    def _slow(u, timeout, **kw):
        release.wait(10)
        return True
    monkeypatch.setattr(udl, "_direct_media_probe_sync", _slow)
    t0 = time.monotonic()
    try:
        with pytest.raises(udl.UrlTimeoutError, match="too long"):
            _run(udl.probe("https://x/watch", timeout=0.5))
        assert time.monotonic() - t0 < 3.0
    finally:
        release.set()


def test_generic_handoff_target_picks_the_manifest(monkeypatch):
    _handoff_policy(monkeypatch)
    page = "https://site.example/watch"
    # A plain `url` hand-off with no manifest_url: the URL GenericIE got.
    assert udl._generic_handoff_target("Youtube", {
        "extractor_key": "Generic", "webpage_url": page}, page) == page
    # Formats naming two different manifests: nothing to judge, refuse.
    assert udl._generic_handoff_target("Youtube", {
        "extractor_key": "Generic", "webpage_url": page, "formats": [
            {"manifest_url": "https://a.example/1.m3u8"},
            {"manifest_url": "https://b.example/2.m3u8"}]}, page) is None
    # url_transparent: webpage_url is the site's canonical page (one the
    # matched extractor claims), so the manifest is what GenericIE fetched.
    yt = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert udl._generic_handoff_target("Youtube", {
        "extractor_key": "Generic", "webpage_url": yt,
        "formats": [{"manifest_url": "https://cdn.example/x.m3u8"}]},
        "https://youtu.be/dQw4w9WgXcQ") == "https://cdn.example/x.m3u8"


def test_generic_handoff_target_judges_a_scraped_page_by_the_page(
        monkeypatch):
    """A plain hand-off to an HTML page GenericIE scraped (a <video> pointing
    at HLS): its formats name the embedded manifest, which answers mpegurl,
    but the URL GenericIE was handed is the page — text/html, refused."""
    _handoff_policy(monkeypatch)
    assert udl._generic_handoff_target("Youtube", {
        "extractor_key": "Generic",
        "webpage_url": "https://evil.example/page.html",
        "formats": [{"manifest_url": "https://cdn.example/x.m3u8"}]},
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ") == (
        "https://evil.example/page.html")


def test_probe_refuses_a_direct_flagged_non_media_hand_off(monkeypatch):
    """GenericIE flags ANY non-HTML body "direct" (octet-stream, JSON…);
    the offline gate would refuse that URL, so the hand-off is refused
    too."""
    _handoff_policy(monkeypatch)
    _stand_in_yt_dlp(monkeypatch, "Youtube", {
        "title": "t", "extractor_key": "Generic", "direct": True,
        "url": "https://cdn.example/blob.bin",
        "webpage_url": "https://x/watch"})
    probed: "list[str]" = []

    def _octet_stream(u, timeout, **kw):
        probed.append(u)
        return False
    monkeypatch.setattr(udl, "_direct_media_probe_sync", _octet_stream)
    with pytest.raises(udl.UrlPolicyError, match="leads to a site"):
        _run(udl.probe("https://x/watch", timeout=5.0))
    # A "direct" result is judged by the URL GenericIE served as-is.
    assert probed == ["https://cdn.example/blob.bin"]


def test_download_subprocess_cannot_hand_off_to_a_scraped_generic_page(
        monkeypatch):
    """The subprocess re-extracts from scratch, so the probe's hand-off gate
    never sees ITS extraction: with URL_ALLOW_GENERIC off a site key's argv
    carries match filters that admit only the site's own result or a media
    file / manifest GenericIE served itself."""
    import yt_dlp
    _handoff_policy(monkeypatch)
    filters = udl.handoff_match_filters("Youtube")
    assert udl.handoff_match_filters("Generic") is None
    for build in (udl.build_download_argv, udl.build_video_download_argv):
        argv = build("https://e.com/a", dest_dir="/tmp/x", max_bytes=1,
                     match_filters=filters)
        assert argv.index("--match-filters") < argv.index("--")
        match = yt_dlp.parse_options(argv[2:-2]).ydl_opts["match_filter"]
        assert match({"extractor_key": "Youtube", "protocol": "https"}) is None
        assert match({"extractor_key": "Generic", "protocol": "https",
                      "direct": True}) is None
        assert match({"extractor_key": "Generic",
                      "protocol": "m3u8_native"}) is None
        assert "does not pass filter" in match(
            {"extractor_key": "Generic", "protocol": "https"})
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", False, raising=False)
    assert udl.handoff_match_filters("Youtube") == ["extractor_key!=Generic"]
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", True, raising=False)
    assert udl.handoff_match_filters("Youtube") is None


def test_probe_reports_a_pinned_out_hand_off_as_a_policy_refusal(monkeypatch):
    """The REAL yt-dlp, pinned by allowed_extractors, never returns a
    delegated info dict: it raises "No suitable extractor (X) found". That
    is the policy at work, not a broken yt-dlp — the client must not be
    told to update the downloader."""
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)

    def _pinned_out(url):
        raise Exception("ERROR: No suitable extractor (Youtube) found for "
                        "URL https://www.youtube.com/watch?v=x")
    _stand_in_yt_dlp(monkeypatch, "Generic", _pinned_out)
    with pytest.raises(udl.UrlPolicyError, match="leads to a site"):
        _run(udl.probe("https://x/a.mp3", timeout=5.0))


def test_classify_error_maps_no_suitable_extractor_to_the_policy():
    raw = "ERROR: No suitable extractor (Youtube) found for URL https://x/a"
    msg = udl.classify_error(raw)
    assert msg == "this link leads to a site the server's URL policy doesn't allow"
    assert isinstance(udl.classified_error(raw), udl.UrlPolicyError)
    assert type(udl.classified_error("ERROR: Private video")) is udl.UrlDownloadError


def test_probe_admits_the_extractor_it_matched(monkeypatch):
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    _probe_with_extractor(monkeypatch, "Generic", "Generic")
    assert _run(udl.probe("https://x/a.mp3", timeout=5.0)).extractor_key == "Generic"


def test_pinned_extractors_follow_the_site_policy(monkeypatch):
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    assert udl.pinned_extractors("Generic") == ["generic"]
    assert udl.pinned_extractors("Youtube") is None
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", ["youtube"], raising=False)
    assert udl.pinned_extractors("Youtube") == ["youtube"]
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", True, raising=False)
    assert sorted(udl.pinned_extractors("Generic")) == ["generic", "youtube"]


def test_download_argv_pins_the_extractors_and_ignores_config():
    for argv in (
            udl.build_download_argv("https://e.com/a", dest_dir="/tmp/x",
                                    max_bytes=1, extractors=["generic"]),
            udl.build_video_download_argv("https://e.com/a", dest_dir="/tmp/x",
                                          max_bytes=1, extractors=["generic"])):
        i = argv.index("--use-extractors")
        assert argv[i + 1] == "generic"
        assert i < argv.index("--")
        assert "--ignore-config" in argv
        # protocol "m3u8" formats go to the native HlsFD (through the
        # guarded handler), not to the FFmpegFD the guard refuses.
        j = argv.index("--downloader")
        assert argv[j + 1] == "m3u8:native"
        assert j < argv.index("--")
    assert "--use-extractors" not in udl.build_download_argv(
        "https://e.com/a", dest_dir="/tmp/x", max_bytes=1)


def test_probe_duration_drops_non_finite_values():
    """A NaN duration from a broken site payload passes the max-duration
    check (nan > limit is False) and must not reach the language check."""
    assert udl._finite_duration(None) is None
    assert udl._finite_duration(float("nan")) is None
    assert udl._finite_duration("inf") is None
    assert udl._finite_duration(0) is None
    assert udl._finite_duration("61.5") == 61.5


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
    assert udl._parse_progress_fields("dl:1024 4096 NA")[:2] == (1024, 4096)


def test_parse_progress_line_estimate_fallback():
    assert udl._parse_progress_fields("dl:10 NA 200")[:2] == (10, 200)


def test_parse_progress_line_unknown_total():
    assert udl._parse_progress_fields("dl:10 NA NA")[:2] == (10, None)


def test_parse_progress_line_infinite_total_is_none():
    # int(float("inf")) raises OverflowError, not ValueError — it must be
    # swallowed like 'NA', never escape as a generic 500.
    assert udl._parse_progress_fields("dl:10 inf NA")[:2] == (10, None)


@pytest.mark.parametrize("line", [
    "", "garbage", "dl:", "dl:NA NA NA", "1024 4096 NA", "[youtube] extracting",
])
def test_parse_progress_line_rejects_noise(line):
    assert udl._parse_progress_fields(line) is None


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
        lambda url, *, dest_dir, max_bytes, extractors=None,
        match_filters=None: _fake_argv(
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


def test_download_failure_logs_the_end_of_a_long_stderr(tmp_path, monkeypatch, caplog):
    """The fatal "ERROR: ..." line is the LAST thing yt-dlp writes: a long
    stderr must be logged by its end, not its head."""
    _patch_argv(monkeypatch, """
import sys
sys.stderr.write("WARNING: " + "w" * 400 + "\\n")
sys.stderr.write("ERROR: fwb-tail-marker\\n")
sys.exit(1)
""")
    with caplog.at_level("WARNING", logger="whisper-api"):
        with pytest.raises(udl.UrlDownloadError):
            _run(udl.download("https://example.com/v", dest_dir=str(tmp_path),
                              max_bytes=10_000, timeout=30))
    rec = [r.getMessage() for r in caplog.records if "yt-dlp exited" in r.getMessage()]
    assert rec and "ERROR: fwb-tail-marker" in rec[0]


def test_download_partial_only_is_size_limit(tmp_path, monkeypatch):
    # --max-filesize skip: clean exit, only a .part file left behind.
    _patch_argv(monkeypatch, """
import os
open(os.path.join(r"__DEST__", "media.m4a.part"), "wb").write(b"x")
""")
    with pytest.raises(udl.UrlDownloadError, match="size limit"):
        _run(udl.download("https://example.com/v", dest_dir=str(tmp_path),
                          max_bytes=10_000, timeout=30))


def test_download_filtered_out_run_is_the_policy_refusal(tmp_path, monkeypatch):
    # handoff_match_filters at work: clean exit, nothing fetched — the site
    # policy's refusal, not the size cap.
    _patch_argv(monkeypatch, """
print("[download] t does not pass filter (extractor_key!=Generic), skipping ..",
      flush=True)
""")
    with pytest.raises(udl.UrlPolicyError, match="leads to a site"):
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
    captured = _stand_in_yt_dlp(monkeypatch, "Youtube", {
        "extractor_key": "Youtube", "title": "t",
        "duration": 60, "filesize": 900_000,
        "ext": "m4a", "abr": 129.5, "language": "de",
        "subtitles": {"de": [{"url": "https://e.test/s.vtt?pot=T"}]}})
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    # A cap smaller than any merged video but above the audio track: the
    # probe must pass, and estimated size must come from `filesize` too.
    monkeypatch.setattr(udl.cfg, "MEDIA_MAX_BYTES", 1_000_000, raising=False)
    info = _run(udl.probe("https://example.com/watch?v=x", timeout=5.0))
    assert captured.get("format") == udl.DOWNLOAD_FORMAT
    assert info.filesize_approx == 900_000
    assert (info.ext, info.abr) == ("m4a", 129.5)
    # The site's language and tracks ride along; the URL stays server-side.
    assert info.language == "de"
    assert [t["id"] for t in info.subtitle_tracks] == ["m-de"]
    assert "pot=T" not in repr(info)
    # A progressive file: the language check downloads it whole.
    assert info.segmented is None
    # Playlists/channel tabs must resolve flat, or a channel's /videos page
    # times the probe out before the playlist rejection can fire.
    assert captured.get("extract_flat") == "in_playlist"


def test_probe_rejects_channel_page_as_playlist(monkeypatch):
    """A channel /videos tab extracts as _type=playlist — the client-safe
    rejection must be 'playlists aren't supported', not a timeout."""
    _stand_in_yt_dlp(monkeypatch, "YoutubeTab", {
        "_type": "playlist", "extractor_key": "YoutubeTab",
        "title": "c't 3003 - Videos",
        "entries": [{"_type": "url", "id": "x"}]})
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

    def _extract(url):
        seen["thread"] = threading.current_thread().name
        return {"title": "t", "extractor_key": "Youtube"}
    _stand_in_yt_dlp(monkeypatch, "Youtube", _extract)
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


def _fake_opener(monkeypatch, body: bytes, ctype: str = "text/vtt",
                 length: "int | None" = None):
    """Swap the guarded opener for one that serves `body` in 1 KB chunks
    (`length`: a Content-Length header, which may claim more)."""
    class _Resp:
        headers = {"Content-Type": ctype,
                   **({"Content-Length": str(length)} if length else {})}

        def __init__(self):
            self._chunks = [body[i:i + 1024] for i in range(0, len(body), 1024)]

        def read(self, n):
            return self._chunks.pop(0) if self._chunks else b""

        def geturl(self):
            return "https://edge.e.com/b"      # where the redirects ended

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


def test_capped_get_caps_filters_and_refuses_hosts(monkeypatch):
    _fake_opener(monkeypatch, b"x" * 5000)
    assert udl._capped_get("https://e.com/a", max_bytes=5000, timeout=1.0) == (
        "text/vtt", b"x" * 5000)
    assert udl._capped_get("https://e.com/a", max_bytes=5000, timeout=1.0,
                           want_url=True)[2] == "https://edge.e.com/b"
    with pytest.raises(udl.UrlDownloadError, match="size limit"):
        udl._capped_get("https://e.com/a", max_bytes=4999, timeout=1.0)
    with pytest.raises(udl.UrlDownloadError, match="unexpected file type"):
        udl._capped_get("https://e.com/a", max_bytes=9999, timeout=1.0,
                        accept=lambda c: c.startswith("image/"))
    with pytest.raises(udl.UrlDownloadError, match="could not be reached"):
        udl._capped_get("file:///etc/passwd", max_bytes=10, timeout=1.0)
    monkeypatch.setattr(udl, "_host_is_forbidden", lambda h: True)
    with pytest.raises(udl.UrlDownloadError, match="could not be reached"):
        udl._capped_get("https://e.com/a", max_bytes=10, timeout=1.0)


def test_capped_get_refuses_a_short_or_cut_body(monkeypatch):
    """A body cut short — fewer bytes than Content-Length, or a read ended
    by the wall-clock cutoff's socket shutdown (http.client answers b"",
    not IncompleteRead) — is an error, never a complete file."""
    _fake_opener(monkeypatch, b"x" * 3000, length=5000)
    with pytest.raises(udl.UrlDownloadError, match="incomplete"):
        udl._capped_get("https://e.com/a", max_bytes=9999, timeout=1.0)
    _fake_opener(monkeypatch, b"x" * 5000, length=5000)
    assert udl._capped_get("https://e.com/a", max_bytes=9999,
                           timeout=1.0)[1] == b"x" * 5000

    class _Fired(udl._WallClockCutoff):
        def __enter__(self):
            super().__enter__()
            self.fired = True
            return self
    monkeypatch.setattr(udl, "_WallClockCutoff", _Fired)
    _fake_opener(monkeypatch, b"x" * 3000)
    with pytest.raises(udl.UrlTimeoutError):
        udl._capped_get("https://e.com/a", max_bytes=9999, timeout=1.0)


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


def _no_proxy(monkeypatch):
    """net_policy honours http(s)_proxy: a shell that exports one would make
    urllib dial the proxy instead of the loopback server under test."""
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY",
                "no_proxy", "NO_PROXY"):
        monkeypatch.delenv(var, raising=False)


def _dribbling_server(header: bytes, interval: float):
    """A loopback server that sends `header` one byte per `interval`, i.e.
    always under a per-socket-op timeout, never finishing in a hurry.
    Returns (socket, accepted): the Event proves the client got as far as
    the dribble — a fast refusal before connecting would otherwise pass
    for the cut."""
    import socket as _s
    import threading as _th
    import time as _t
    srv = _s.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    accepted = _th.Event()

    def serve():
        try:
            conn, _ = srv.accept()
        except OSError:
            return
        accepted.set()
        with conn:
            for b in header:
                try:
                    conn.send(bytes([b]))
                except OSError:
                    return
                _t.sleep(interval)
    _th.Thread(target=serve, daemon=True).start()
    return srv, accepted


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
    _no_proxy(monkeypatch)
    srv, accepted = _dribbling_server(
        b"HTTP/1.1 200 OK\r\nContent-Type: audio/mpeg\r\nContent-Length: 1\r\n\r\nx",
        0.1)
    try:
        port = srv.getsockname()[1]
        t0 = _t.monotonic()
        out = udl._direct_media_probe_sync(
            f"http://127.0.0.1:{port}/a.mp3", timeout=0.5)
        assert out is False
        assert accepted.is_set()
        # The dribble alone takes ~7 s. 4 s, not 2: a loaded CI runner needed
        # 2.5 s for the 0.5 s deadline (run 1078) and the cut is still proven.
        assert _t.monotonic() - t0 < 4.0
    finally:
        srv.close()


def test_thumbnail_cuts_a_dribbled_header(monkeypatch):
    """Same window in fetch_thumbnail_data_uri's header phase. Driven on the
    sync layer: capped_get's outer wait_for gives up at timeout + 2 s and
    soft-fails to None, which would pass this bound with the cut gone while
    the pool thread stays wedged for the whole ~7 s dribble."""
    import http.client
    import time as _t
    from faster_whisper_backend.core import net_policy as np
    monkeypatch.setattr(udl, "_host_is_forbidden", lambda h: False)
    monkeypatch.setattr(np, "address_is_forbidden", lambda a: False)
    _no_proxy(monkeypatch)
    srv, accepted = _dribbling_server(
        b"HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: 1\r\n\r\nx",
        0.1)
    try:
        port = srv.getsockname()[1]
        t0 = _t.monotonic()
        # The cut surfaces as http.client's BadStatusLine / a socket error;
        # anything else (a TypeError, a host-gate refusal) never reached it.
        with pytest.raises((http.client.HTTPException, OSError)):
            udl._capped_get(f"http://127.0.0.1:{port}/t.jpg", max_bytes=512_000,
                            timeout=0.5, accept=lambda c: c.startswith("image/"))
        assert accepted.is_set()
        assert _t.monotonic() - t0 < 4.0   # see the probe test above
    finally:
        srv.close()


def test_direct_media_probe_admits_an_hls_manifest_only_for_a_hand_off(
        monkeypatch):
    """CDNs answer an HLS manifest as application/vnd.apple.mpegurl, which
    no audio/ video/ prefix matches: the hand-off verdict (accept_hls) must
    take it, while a pasted link to a bare manifest stays refused."""
    import http.server
    import threading
    from faster_whisper_backend.core import net_policy as np
    monkeypatch.setattr(udl, "_host_is_forbidden", lambda h: False)
    monkeypatch.setattr(np, "address_is_forbidden", lambda a: False)
    _no_proxy(monkeypatch)

    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"#EXTM3U\n"
            self.send_response(200)
            self.send_header("Content-Type",
                             "application/vnd.apple.mpegurl; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/live.m3u8"
    try:
        assert udl._direct_media_probe_sync(
            url, timeout=3, accept_hls=True) is True
        assert udl._direct_media_probe_sync(url, timeout=3) is False
    finally:
        srv.shutdown()
        srv.server_close()


def test_cutoff_shuts_a_socket_added_after_it_fired():
    """A socket whose connect() returns after the timer fired (a dribbled TLS
    handshake, a redirect hop dialled past the cutoff) must be shut at once,
    not parked in the list cut() already emptied."""
    import socket as _s
    a, b = _s.socketpair()
    try:
        a.settimeout(2.0)
        cutoff = udl._WallClockCutoff(10.0)
        cutoff.cut()
        cutoff.add(a)
        assert a.recv(1) == b""
    finally:
        a.close()
        b.close()


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


def test_video_ladder_audio_leg_prefers_the_original_language_over_bitrate():
    """yt-dlp's `ba` ranks language_preference ahead of bitrate: a louder dub
    (or the audio-description track) must not become the kept video's
    soundtrack while the transcript follows the original."""
    vid = _fmt(format_id="616", vcodec="avc1", acodec="none", height=1080,
               ext="mp4", filesize=20_000_000)
    dub = _fmt(format_id="251-1", vcodec="none", acodec="opus", abr=140,
               language_preference=-1, filesize=1_000_000)
    orig = _fmt(format_id="251-0", vcodec="none", acodec="opus", abr=130,
                language_preference=10, filesize=1_000_000)
    desc = _fmt(format_id="251-2", vcodec="none", acodec="opus", abr=160,
                language_preference=-10, filesize=1_000_000)
    for formats in ([dub, orig, desc, vid], [orig, dub, desc, vid]):
        ladder = udl.build_video_ladder({"duration": 100.0, "formats": formats},
                                        max_bytes=100_000_000)
        assert ladder[0]["audio_format_id"] == "251-0"
    # A tie on language falls to bitrate, as before.
    a = _fmt(format_id="a-lo", vcodec="none", acodec="opus", abr=96,
             language_preference=10)
    b = _fmt(format_id="a-hi", vcodec="none", acodec="opus", abr=128,
             language_preference=10)
    ladder = udl.build_video_ladder({"duration": 100.0, "formats": [b, a, vid]},
                                    max_bytes=100_000_000)
    assert ladder[0]["audio_format_id"] == "a-hi"


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
    # The unscaled estimate rides along, independent of the learned ratio:
    # it is what the recorder divides the real size by.
    assert prem["raw_approx_bytes"] == 4000 * 1000 * 125 + 16_000_000
    assert raw[0]["raw_approx_bytes"] == prem["raw_approx_bytes"]
    assert prem["approx_bytes"] == int(4000 * 1000 * 125 * 0.5) + 16_000_000
    assert av1["raw_approx_bytes"] == av1["approx_bytes"] == 100_000_000 + 16_000_000


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


def test_video_ladder_video_only_rung_without_an_audio_candidate_still_merges():
    """An HLS master playlist: the video variants are explicitly video-only
    and the audio rendition carries no acodec, so it never becomes a
    candidate. The rung must still ask for a merge ("ba"), or the exact id
    alone downloads a soundless file."""
    ladder = udl.build_video_ladder({"duration": 600.0, "formats": [
        _fmt(format_id="hls-audio", vcodec="none", protocol="m3u8_native", ext="mp4"),
        _fmt(format_id="hls-3000", height=1080, tbr=3000, vcodec="avc1.64", acodec="none",
             protocol="m3u8_native"),
    ]}, max_bytes=10**10)
    assert ladder[0]["format_id"] == "hls-3000" and ladder[0]["audio_format_id"] == "ba"
    assert udl.video_format_selector(1080, ("hls-3000", "ba")).startswith("hls-3000+ba/")
    # Unknown acodec (possibly muxed) stays a single-file rung.
    unknown = udl.build_video_ladder({"duration": 600.0, "formats": [
        _fmt(format_id="v", height=720, tbr=900, vcodec="avc1.64",
             protocol="m3u8_native")]}, max_bytes=10**10)
    assert unknown[0]["audio_format_id"] is None


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


def _patch_video_argv(monkeypatch, script: str):
    # The scripts' middle lines are asserted on: with the 0.3 s throttle on,
    # a reader scheduled 50 ms late on a loaded runner swallowed one.
    monkeypatch.setattr(udl, "_PROGRESS_EMIT_MIN_S", 0.0)
    monkeypatch.setattr(
        udl, "build_video_download_argv",
        lambda url, *, dest_dir, max_bytes, max_height=None, container="mkv",
        format_ids=None, extractors=None, match_filters=None: _fake_argv(
            script.replace("__DEST__", dest_dir)))


_TWO_STREAMS_SCRIPT = """
import os, sys
print("dl:500 1000 NA", flush=True)
print("dl:1000 1000 NA", flush=True)
print("dl:100 300 NA", flush=True)
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
    # The second stream's restart at 100 must not read as 100 of 1300
    # (deterministic: _patch_video_argv turns the emit throttle off).
    assert [d for _f, _t, d in seen] == [500, 1000, 1100, 1300], seen


_TWO_LEGS_BY_ID_SCRIPT = """
import os, sys
print("dl:500 NA NA 616", flush=True)
print("dl:1200 1200 NA 616", flush=True)
print("dl:100 300 NA 140", flush=True)
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


_UNPRICED_AUDIO_LEG_SCRIPT = """
import os, sys
print("dl:500 1000 NA 616", flush=True)
print("dl:1000 1000 NA 616", flush=True)
print("dl:100 NA NA 140", flush=True)
print("dl:300 300 NA 140", flush=True)
open(os.path.join(r"__DEST__", "media.mkv"), "wb").write(b"x" * 64)
"""


def test_download_video_never_reads_100_percent_while_a_leg_is_pending(tmp_path, monkeypatch):
    # The probe priced the video leg only: the bar must not sit at 100 %
    # through the whole audio leg.
    _patch_video_argv(monkeypatch, _UNPRICED_AUDIO_LEG_SCRIPT)
    seen = []
    _run(udl.download_video(
        "https://example.com/v", dest_dir=str(tmp_path), max_bytes=10_000,
        timeout=30, format_ids=("616", "140"), leg_estimates={"616": 1000},
        expected_total=1000, progress_cb=lambda f, t, d: seen.append((f, t, d))))
    fracs = [f for f, _t, _d in seen]
    assert len(seen) == 4 and all(f < 1.0 for f in fracs[:-1]), seen
    assert fracs == sorted(fracs), fracs
    assert seen[-1] == (1.0, 1300, 1300)


_DROP_INSIDE_ONE_LEG_SCRIPT = """
import os, sys
print("dl:500 1000 NA 616", flush=True)
print("dl:200 1000 NA 616", flush=True)
print("dl:1000 1000 NA 616", flush=True)
open(os.path.join(r"__DEST__", "media.mkv"), "wb").write(b"x" * 64)
"""


def test_download_video_counter_reset_inside_one_leg_is_not_a_new_file(tmp_path, monkeypatch):
    _patch_video_argv(monkeypatch, _DROP_INSIDE_ONE_LEG_SCRIPT)
    seen = []
    _run(udl.download_video(
        "https://example.com/v", dest_dir=str(tmp_path), max_bytes=1200,
        timeout=30, format_ids=("616", None),
        progress_cb=lambda f, t, d: seen.append((f, t, d))))
    assert seen[-1] == (1.0, 1000, 1000)


_AUDIO_RESTART_SCRIPT = """
import os, sys, time
print("dl:600 1000 NA", flush=True)
time.sleep(0.35)
print("dl:200 1000 NA", flush=True)
time.sleep(0.35)
print("dl:1000 1000 NA", flush=True)
open(os.path.join(r"__DEST__", "media.m4a"), "wb").write(b"x" * 64)
"""


def test_download_audio_counter_restart_is_not_double_counted(tmp_path, monkeypatch):
    # The audio selector fetches ONE file: a retry that does not resume
    # restarts the counter, and must not read as a finished first file
    # (600 + 1000 would trip the 1200 cap on a 1000-byte download).
    _patch_argv(monkeypatch, _AUDIO_RESTART_SCRIPT)
    seen = []
    out = _run(udl.download(
        "https://example.com/v", dest_dir=str(tmp_path), max_bytes=1200,
        timeout=30, progress_cb=lambda f, tot: seen.append((f, tot))))
    assert os.path.basename(out) == "media.m4a"
    assert seen[-1] == (1.0, 1000)
    assert all(tot == 1000 for _f, tot in seen)


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


@pytest.mark.parametrize("name", ["media.MP3", "media.MP4", "media.unknown_video"])
def test_find_result_file_accepts_the_extension_yt_dlp_kept(tmp_path, name):
    # yt-dlp keeps the URL's own case and falls back to `unknown_video`; a
    # finished download must not read as "the cap bit, no file".
    (tmp_path / name).write_bytes(b"y")
    assert os.path.basename(udl._find_result_file(str(tmp_path))) == name
    assert os.path.basename(udl._find_video_result(str(tmp_path), "mkv")) == name
    # The one-dot rule still holds beside it.
    (tmp_path / "media.f251.WEBM").write_bytes(b"x")
    with pytest.raises(udl.UrlDownloadError, match="merged"):
        udl._find_video_result(str(tmp_path), "mkv")


_UPPERCASE_EXT_SCRIPT = """
import os
open(os.path.join(r"__DEST__", "media.MP3"), "wb").write(b"x" * 64)
"""


def test_download_returns_an_uppercase_extension_file(tmp_path, monkeypatch):
    _patch_argv(monkeypatch, _UPPERCASE_EXT_SCRIPT)
    out = _run(udl.download("https://example.com/AUDIO.MP3", dest_dir=str(tmp_path),
                            max_bytes=10_000, timeout=30))
    assert os.path.basename(out) == "media.MP3"


def test_probe_carries_the_ladder_when_video_is_enabled(monkeypatch):
    _stand_in_yt_dlp(monkeypatch, "Youtube", {
        "extractor_key": "Youtube", "title": "t", "duration": 100.0,
        "filesize": 900_000, "ext": "m4a", "abr": 129.5,
        "formats": _LADDER_INFO["formats"]})
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
