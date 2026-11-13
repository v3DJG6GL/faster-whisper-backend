"""The yt-dlp SSRF guard (ytdlp_plugins/) — the finding it closes, live.

url_download's own probes already refused private / loopback / link-local /
metadata addresses on every hop, but yt-dlp fetched with its own opener: a
host that looks public to the probe could 302 `extract_info` and the download
subprocess into an internal address, and nothing checked that hop.

The redirect and happy-path tests run the real thing against two loopback
HTTP servers:

  PUBLIC   127.0.0.2  — stands in for the attacker's public host. It answers
                        the direct-media probe (User-Agent
                        "faster-whisper-backend") and any /direct… path with
                        200 audio/mpeg, and redirects everything else — i.e.
                        yt-dlp — to INTERNAL.
  INTERNAL 127.0.0.1  — stands in for 169.254.169.254 / a LAN service. Every
                        request it receives is a guard failure.

In those tests THE ONLY PATCH is the repro's: the literal 127.0.0.2 is treated
as a public address. In-process that is a monkeypatch of
net_policy.address_is_forbidden; for the download subprocess it is a copy of
the guard tree next to a patched copy of net_policy.py (the guard loads
net_policy relative to its own location, so the copy is what the child
enforces). 127.0.0.1 is judged by the real, unpatched policy, and yt-dlp itself
is never patched.
"""

from __future__ import annotations

import asyncio
from asyncio import run as _run
import gzip
import http.client
import http.server
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import types
from unittest import mock
import urllib.request
import zlib

import pytest
import yt_dlp
from yt_dlp.downloader import external, hls, rtmp
from yt_dlp.networking.exceptions import RequestError

from faster_whisper_backend.core import net_policy
from faster_whisper_backend.media import download as udl
from tests.conftest import read_repo_text
from tests.media._server_helpers import _QuietHandler, _stopping
from tests.media.test_download import _black_hole_dns, _fetch

SECRET = b"INTERNAL-SECRET-" * 64
PUBLIC_BODY = b"\xff\xfb\x90\x44" + b"\x00" * 4092  # plausible MPEG audio head

from faster_whisper_backend.paths import REPO_ROOT


class _Internal(_QuietHandler):
    hits: "list[str]" = []

    def do_GET(self):
        type(self).hits.append(f"{self.command} {self.path}")
        self._ok_headers("audio/mpeg", len(SECRET))
        self.wfile.write(SECRET)

    do_HEAD = do_GET


class _Public(_QuietHandler):
    """200 audio/mpeg for the direct-media probe, a redirect for everyone
    else — exactly the shape that slipped past the pre-guard code."""

    redirect_to = ""

    def do_GET(self):
        ua = self.headers.get("User-Agent") or ""
        if ua.startswith("faster-whisper-backend") or self.path.startswith("/direct"):
            self._ok_headers("audio/mpeg", len(PUBLIC_BODY))
            if self.command == "GET":
                self.wfile.write(PUBLIC_BODY)
            return
        self.send_response(302)
        self.send_header("Location", type(self).redirect_to)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_HEAD = do_GET


def _serve(handler, host):
    srv = http.server.ThreadingHTTPServer((host, 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _inproc_guard():
    """The guard module guard_self_check loaded in-process."""
    udl.guard_self_check(force=True)
    return sys.modules["fwb_ssrf_guard_inproc"]


def _reset_budget():
    """A fresh extraction: the manifests one thread parses spend from one
    segment / variant budget until probe() resets it (observe_dials)."""
    guard = sys.modules.get("fwb_ssrf_guard_inproc")
    if guard is not None:
        guard.reset_budget()


@pytest.fixture(autouse=True)
def _fresh_manifest_budget():
    _reset_budget()


@pytest.fixture(name="public")
def servers():
    """The public URL base, with a clean INTERNAL hit log."""
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.2", 0))
    except OSError:  # pragma: no cover — non-Linux loopback aliasing
        pytest.skip("this platform does not alias 127.0.0.2")
    internal = _serve(_Internal, "127.0.0.1")
    _Internal.hits = []
    _Public.redirect_to = f"http://127.0.0.1:{internal.server_port}/secret.mp3"
    public = _serve(_Public, "127.0.0.2")
    with _stopping(internal), _stopping(public):
        yield f"http://127.0.0.2:{public.server_port}"


@pytest.fixture
def public_is_public(monkeypatch):
    """THE only patch: 127.0.0.2 counts as a public address, in-process."""
    real = net_policy.address_is_forbidden
    monkeypatch.setattr(
        net_policy, "address_is_forbidden",
        lambda addr: False if addr == "127.0.0.2" else real(addr))


@pytest.fixture
def guard_tree(tmp_path, monkeypatch):
    """A copy of the guard tree whose net_policy.py calls 127.0.0.2 public.

    The guard resolves net_policy relative to its own file, so patching the
    child's policy means copying the tree — no test hook in shipped code."""
    root = tmp_path / "root"
    root.mkdir()
    shutil.copytree(os.path.join(REPO_ROOT, "ytdlp_plugins"),
                    root / "ytdlp_plugins")
    policy_dir = root / "faster_whisper_backend" / "core"
    policy_dir.mkdir(parents=True)
    (policy_dir / "net_policy.py").write_text(
        read_repo_text("faster_whisper_backend", "core", "net_policy.py")
        + '\n_real = address_is_forbidden\n'
          'def address_is_forbidden(addr):\n'
          '    return False if addr == "127.0.0.2" else _real(addr)\n',
        encoding="utf-8")
    guard_dir = str(root / "ytdlp_plugins")
    monkeypatch.setattr(udl, "GUARD_DIR", guard_dir)
    monkeypatch.setattr(udl, "GUARD_LAUNCHER",
                        os.path.join(guard_dir, "run_guarded_yt_dlp.py"))
    monkeypatch.setattr(udl, "GUARD_MODULE",
                        os.path.join(guard_dir, "fwb_ssrf_guard",
                                     "yt_dlp_plugins", "extractor",
                                     "fwb_ssrf_guard.py"))
    return guard_dir


def test_guard_registers_in_process():
    udl.guard_self_check(force=True)
    from yt_dlp.networking.common import _REQUEST_HANDLERS

    assert _REQUEST_HANDLERS["FwbSsrfGuard"].RH_NAME == "fwb-guarded-urllib"
    # The point is not that ours exists but that nothing UNGUARDED is left to
    # fall back to when ours declines a request (data:/ftp:, impersonation).
    for superseded in ("Urllib", "Requests", "CurlCFFI", "Websockets"):
        assert superseded not in _REQUEST_HANDLERS
    assert set(_REQUEST_HANDLERS) == {"FwbSsrfGuard"}


def test_unknown_handler_is_retired_and_fails_closed(monkeypatch):
    # A handler a future yt-dlp registers under a NEW key must not survive as
    # an unguarded way out: is_installed() fails closed until install() runs.
    udl.guard_self_check(force=True)
    from yt_dlp.networking.common import _REQUEST_HANDLERS
    guard = udl._guard_module

    class Dummy:
        RH_KEY = "FwbTestRH"
    monkeypatch.setitem(_REQUEST_HANDLERS, "FwbTestRH", Dummy)
    assert guard.is_installed() is False
    guard.install()
    assert "FwbTestRH" not in _REQUEST_HANDLERS
    assert guard.is_installed() is True


def test_guard_prefers_over_every_builtin():
    udl.guard_self_check(force=True)

    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        director = ydl._request_director
        chosen = sorted(
            director.handlers.values(),
            key=lambda rh: sum(p(rh, None) for p in director.preferences),
            reverse=True)[0]
    assert chosen.RH_KEY == "FwbSsrfGuard"


def test_guard_refuses_ffmpeg_delegation_of_an_aes128_manifest(monkeypatch, tmp_path):
    """Without pycryptodomex, HlsFD hands an AES-128 manifest to FFmpegFD,
    and ffmpeg fetches the key URI and the segments itself — outside the
    RequestHandler registry, so a public manifest pointing them at
    169.254.169.254 or a LAN host was a blind SSRF. The guard must refuse the
    hand-off (with the marker) before any external program is spawned."""
    udl.guard_self_check(force=True)

    manifest = (
        "#EXTM3U\n#EXT-X-TARGETDURATION:4\n"
        '#EXT-X-KEY:METHOD=AES-128,URI="http://127.0.0.1/k"\n'
        "#EXTINF:4,\nhttp://127.0.0.1/seg0.ts\n#EXT-X-ENDLIST\n")

    class _Resp:
        url = "https://public.example/a.m3u8"

        def read(self):
            return manifest.encode()

    spawned = []
    monkeypatch.setattr(hls.Cryptodome, "AES", None)
    monkeypatch.setattr(external.FFmpegFD, "available",
                        classmethod(lambda cls, path=None: True))
    monkeypatch.setattr(external.FFmpegFD, "_call_downloader",
                        lambda self, *a: spawned.append(a) or 0)
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        monkeypatch.setattr(ydl, "urlopen", lambda req: _Resp())
        fd = hls.HlsFD(ydl, ydl.params)
        with pytest.raises(yt_dlp.utils.DownloadError, match=udl.GUARD_MARKER):
            fd.real_download(str(tmp_path / "media.mp4"), {
                "url": "https://public.example/a.m3u8", "id": "x",
                "ext": "mp4", "http_headers": {}})
    assert spawned == []


def test_guard_is_not_installed_without_the_downloader_refusal(monkeypatch):
    guard = _inproc_guard()
    assert guard.is_installed()
    for cls in (external.ExternalFD, rtmp.RtmpFD):
        with monkeypatch.context() as m:
            m.setattr(cls, "real_download", lambda self, f, i: True)
            assert not guard.is_installed()
    assert guard.is_installed()


def test_https_through_a_connect_proxy_verifies_the_target_not_the_proxy():
    """With an http(s)_proxy in the environment every HTTPS fetch tunnels via
    CONNECT; SNI / certificate verification must then name the URL's host
    (_tunnel_host), not the proxy the socket was dialled to."""
    guard = _inproc_guard()
    seen: dict = {}

    class FakeCtx:
        def wrap_socket(self, sock, server_hostname=None):
            seen["sni"] = server_hostname
            return sock

    conn = guard._PinnedHTTPSConnection("proxy.example", 8080, context=FakeCtx())
    conn.set_tunnel("target.example", 443)
    with mock.patch.object(guard, "_connect_pinned", return_value=object()), \
            mock.patch.object(conn, "_tunnel"):
        conn.connect()
    assert seen["sni"] == "target.example"

    # Direct path unchanged: the URL's host is conn.host itself.
    direct = guard._PinnedHTTPSConnection("target.example", 443, context=FakeCtx())
    with mock.patch.object(guard, "_connect_pinned", return_value=object()):
        direct.connect()
    assert seen["sni"] == "target.example"


def test_net_policy_resolve_pinned_refuses_forbidden_answer(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **kw: [(0, 0, 0, "", ("169.254.169.254", 80))])
    with pytest.raises(OSError):
        net_policy.resolve_pinned("meta.example", 80)
    public = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: list(public))
    assert net_policy.resolve_pinned("public.example", 443) == public


def test_guard_resolve_pinned_fails_closed_on_a_host_idna_cannot_encode():
    """`a..com` (empty label) or a 64-char label makes getaddrinfo raise
    UnicodeError, not OSError; inside the yt-dlp child it must still end as
    the guard's RequestError (fail closed), never an unhandled error."""
    guard = _inproc_guard()
    for host in ("a..com", "a" * 64 + ".com"):
        with pytest.raises(RequestError, match=guard.MARKER):
            guard._resolve_pinned(host, 443)


def _run_launcher(plugin_flags, tmp_path):
    """The launcher with build_download_argv's plugin flags, run far enough
    to load plugins (`--version` exits inside option parsing, before yt-dlp
    ever loads one) but with no network: a data: URL, which the guard has
    no handler for."""
    argv = udl.build_download_argv("https://example.com/a", dest_dir=".",
                                   max_bytes=1)
    return subprocess.run(
        argv[:5] + plugin_flags
        + ["--ignore-config", "-v", "--simulate", "--", "data:,"],
        capture_output=True, text=True, timeout=60, cwd=str(tmp_path))


def test_launcher_starts_yt_dlp_with_the_guard_flags(tmp_path):
    """The launcher must not fail closed on a healthy tree, and the guard
    must ALSO load through yt-dlp's official --plugin-dirs channel."""
    proc = _run_launcher([], tmp_path)
    assert udl.GUARD_UNAVAILABLE_MARKER not in proc.stderr
    assert "Error while importing module" not in proc.stderr
    plugin_dirs = [ln for ln in proc.stderr.splitlines()
                   if "Plugin directories:" in ln]
    assert plugin_dirs and "fwb_ssrf_guard" in plugin_dirs[0], proc.stderr


def test_launcher_check_sees_a_broken_plugin_import(tmp_path):
    """Negative control for the test above: a plugin whose import raises
    must show up in the same run, or the assertion there proves nothing."""
    broken = tmp_path / "broken" / "pkg" / "yt_dlp_plugins" / "extractor"
    broken.mkdir(parents=True)
    (broken / "fwb_test_broken.py").write_text("raise ImportError('boom')\n")
    proc = _run_launcher(["--plugin-dirs", str(tmp_path / "broken")], tmp_path)
    assert "Error while importing module" in proc.stderr, proc.stderr


def _ydl_opts(argv):
    """The YoutubeDL params the download subprocess would build from
    `argv` (interpreter and launcher dropped, and the trailing `-- url`)."""
    assert argv[-2] == "--"
    return yt_dlp.parse_options(argv[2:-2]).ydl_opts


def test_pinned_download_flags_parse_on_the_pinned_yt_dlp():
    """--ignore-config, --downloader and --use-extractors (with a
    regex-escaped IE name) must be accepted by the pinned yt-dlp, the
    extractor regexes must compile and the -f selector must parse, or every
    link download fails. Done in-process: yt-dlp only validates the values
    when it builds a YoutubeDL, which `--version` never reaches."""
    import re

    from yt_dlp.downloader import get_suitable_downloader
    from yt_dlp.downloader.hls import HlsFD
    for build in (udl.build_download_argv, udl.build_video_download_argv):
        argv = build("https://example.com/a", dest_dir=".", max_bytes=1,
                     extractors=["generic", re.escape("youtube:tab")])
        opts = _ydl_opts(argv)
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.build_format_selector(opts["format"])
        # protocol "m3u8" goes to the native HlsFD (guarded handler), not to
        # the FFmpegFD the guard refuses.
        assert get_suitable_downloader(
            {"protocol": "m3u8", "url": "https://e.test/a.m3u8"}, opts) is HlsFD
    # ...and the check can fail.
    bad = _ydl_opts(udl.build_download_argv(
        "https://example.com/a", dest_dir=".", max_bytes=1,
        extractors=["[unclosed("]))
    with pytest.raises(ValueError, match="allowed_extractors"):
        yt_dlp.YoutubeDL(bad)


def test_probe_refuses_redirect_into_loopback(public, public_is_public,
                                              monkeypatch):
    udl.guard_self_check(force=True)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    with pytest.raises(udl.UrlDownloadError) as ei:
        _run(udl.probe(f"{public}/x.mp3", timeout=20))
    # Client-safe wording, and never the address the guard actually saw.
    assert "could not be reached" in str(ei.value)
    assert "127.0.0.1" not in str(ei.value)
    assert _Internal.hits == []


def test_download_refuses_redirect_into_loopback(tmp_path, public,
                                                 public_is_public, guard_tree):
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(udl.UrlDownloadError) as ei:
        _fetch(udl.download, dest, url=f"{public}/x.mp3", max_bytes=10_000_000, timeout=60)
    assert "could not be reached" in str(ei.value)
    assert "127.0.0.1" not in str(ei.value)
    assert _Internal.hits == []
    assert os.listdir(dest) == []


def test_download_allows_a_public_looking_target(tmp_path, public,
                                                 public_is_public, guard_tree):
    dest = tmp_path / "dest"
    dest.mkdir()
    out = _fetch(udl.download, dest, url=f"{public}/direct.mp3", max_bytes=10_000_000, timeout=60)
    assert os.path.getsize(out) == len(PUBLIC_BODY)
    assert _Internal.hits == []


class _Dribbler(_QuietHandler):
    """200 audio/mpeg for the direct-media probe at once; yt-dlp's page GET
    gets its status line and then one header byte every 0.3 s — never
    tripping a per-op socket timeout, never finishing."""

    def do_GET(self):
        ua = self.headers.get("User-Agent") or ""
        if ua.startswith("faster-whisper-backend"):
            self._ok_headers("audio/mpeg", len(PUBLIC_BODY))
            self.wfile.write(PUBLIC_BODY)
            return
        import time as _t
        try:
            self.wfile.write(b"HTTP/1.1 200 OK\r\n")
            self.wfile.flush()
            for _ in range(200):
                self.wfile.write(b"X")
                self.wfile.flush()
                _t.sleep(0.3)
        except OSError:
            pass


def test_probe_extraction_is_cut_at_its_wall_clock_deadline(
        public_is_public, monkeypatch):
    """socket_timeout is per op, so a host dribbling yt-dlp's headers held
    the _PROBE_POOL worker long after the probe answered "took too long";
    four such links wedged every worker (and every user's url-preview).
    The guard's dial observer shuts the extraction's sockets at the
    deadline, so the worker comes back."""
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.2", 0))
    except OSError:  # pragma: no cover — non-Linux loopback aliasing
        pytest.skip("this platform does not alias 127.0.0.2")
    srv = _serve(_Dribbler, "127.0.0.2")
    udl.guard_self_check(force=True)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    url = f"http://127.0.0.2:{srv.server_port}/a.mp3"

    async def _four():
        return await asyncio.gather(
            *(udl.probe(url, timeout=3) for _ in range(4)),
            return_exceptions=True)
    with _stopping(srv):
        results = _run(_four())
        assert all(isinstance(r, udl.UrlTimeoutError) for r in results), results
        # Every worker is free again shortly after the deadline.
        futs = [udl._PROBE_POOL.submit(lambda: 1) for _ in range(4)]
        assert [f.result(timeout=5) for f in futs] == [1] * 4


@pytest.mark.parametrize("target", [
    "data:audio/mpeg;base64,//uQxAAAAAAAAAAAAAAAAAAAAAAA",
    "ftp://127.0.0.1:1/secret.mp3",
    "file:///etc/passwd",
])
def test_probe_refuses_non_http_redirect(target, public, public_is_public,
                                         monkeypatch, caplog):
    _Public.redirect_to = target
    udl.guard_self_check(force=True)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    with caplog.at_level("WARNING", logger="whisper-api"):
        with pytest.raises(udl.UrlDownloadError) as ei:
            _run(udl.probe(f"{public}/x.mp3", timeout=20))
    assert "could not be reached" in str(ei.value)
    # The client wording is the same for an unreachable host, so it cannot
    # tell the guard's refusal apart: check the raw error in the log.
    failed = [r.getMessage() for r in caplog.records
              if "[url-dl] probe failed" in r.getMessage()]
    assert failed
    if target.startswith("ftp:"):
        # Without the guard's scheme refusal this would read "connection
        # refused" (nothing listens on port 1).
        assert udl.GUARD_MARKER in failed[-1]
    # data: and file: never reach the guard: urllib's own redirect handler
    # refuses a non-http/ftp Location first ("HTTP Error 302"). Those cases
    # pin that the hop is refused at all, whichever layer does it.


@pytest.fixture
def broken_guard(monkeypatch):
    monkeypatch.setattr(udl, "_guard_ok", False)
    monkeypatch.setattr(udl, "GUARD_MODULE",
                        os.path.join(REPO_ROOT, "no-such-guard.py"))


def test_probe_refuses_when_the_guard_cannot_install(broken_guard, caplog,
                                                     monkeypatch):
    async def _policy(url, deadline=None):
        return "Generic"

    monkeypatch.setattr(udl, "check_url_policy", _policy)
    with caplog.at_level("ERROR"):
        with pytest.raises(udl.UrlDownloadError, match="unavailable"):
            _run(udl.probe("https://example.com/v", timeout=5))
    assert any("REFUSING link downloads" in r.message for r in caplog.records)


def test_download_refuses_when_the_guard_cannot_install(tmp_path, broken_guard,
                                                        caplog):
    with caplog.at_level("ERROR"):
        with pytest.raises(udl.UrlDownloadError, match="unavailable"):
            _fetch(udl.download, tmp_path, max_bytes=1000, timeout=5)
    assert any("REFUSING link downloads" in r.message for r in caplog.records)
    assert os.listdir(tmp_path) == []


def test_guard_failure_is_not_sticky(monkeypatch):
    """The next ORDINARY call (probe()/download() never pass force) retries
    once the tree is back — no "failed once, refuse until forced" latch."""
    real_module = udl.GUARD_MODULE
    monkeypatch.setattr(udl, "_guard_ok", False)
    monkeypatch.setattr(udl, "GUARD_MODULE",
                        os.path.join(REPO_ROOT, "no-such-guard.py"))
    with pytest.raises(udl.UrlDownloadError):
        udl.guard_self_check()
    assert udl._guard_ok is False
    monkeypatch.setattr(udl, "GUARD_MODULE", real_module)  # the tree is back
    udl.guard_self_check()
    assert udl._guard_ok is True


def test_url_download_uses_net_policy_directly():
    assert udl._host_is_forbidden is net_policy.host_is_forbidden
    assert udl._CGNAT_NET is net_policy.CGNAT_NET


def test_guard_uses_the_same_net_policy_module():
    guard = _inproc_guard()
    assert guard.net_policy is net_policy
    assert guard.MARKER == udl.GUARD_MARKER


def _load_launcher():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fwb_launcher_probe",
                                                  udl.GUARD_LAUNCHER)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)   # main() only runs under __main__
    return launcher


def test_launcher_marker_matches_the_parent():
    """The launcher's fail-closed line carries its own copy of the marker;
    a drift would turn a refused run into an unclassified error."""
    launcher = _load_launcher()
    assert launcher._MARKER == udl.GUARD_MARKER
    assert launcher._UNAVAILABLE_MARKER == udl.GUARD_UNAVAILABLE_MARKER


def test_launcher_fail_closed_line_reads_as_unavailable(monkeypatch, capsys):
    """A child that cannot install the guard refuses to run: the client must
    hear what guard_self_check says for the same condition, not that the
    site could not be reached."""
    launcher = _load_launcher()
    monkeypatch.setattr(launcher, "_GUARD_PATH",
                        os.path.join(REPO_ROOT, "no-such-guard.py"))
    assert launcher.main() == 78
    line = capsys.readouterr().err
    assert udl.GUARD_UNAVAILABLE_MARKER in line
    assert (udl.classify_error(line)
            == "link downloads are unavailable on this server")


def test_guard_copy_agrees_with_net_policy_address_by_address(guard_tree,
                                                              tmp_path):
    """The subprocess's guard loads net_policy BY PATH (the repo root is not
    on its sys.path); prove the file it reaches through its OWN resolver is
    the one beside it, verdict for verdict, so the two halves can never
    drift."""
    child = subprocess.run(
        [sys.executable, "-I", "-c",
         "import importlib.util,sys,json\n"
         "spec=importlib.util.spec_from_file_location('g', sys.argv[1])\n"
         "g=importlib.util.module_from_spec(spec); spec.loader.exec_module(g)\n"
         "addrs=json.loads(sys.argv[2])\n"
         "print(json.dumps({'file': g.net_policy.__file__, 'verdicts':"
         " [g.net_policy.address_is_forbidden(a) for a in addrs]}))",
         udl.GUARD_MODULE, json.dumps(_ADDRESS_CORPUS)],
        capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
    assert child.returncode == 0, child.stderr
    out = json.loads(child.stdout)
    assert os.path.realpath(out["file"]) == os.path.realpath(os.path.join(
        os.path.dirname(guard_tree), "faster_whisper_backend", "core",
        "net_policy.py"))
    ours = [net_policy.address_is_forbidden(a) for a in _ADDRESS_CORPUS]
    for addr, mine, theirs in zip(_ADDRESS_CORPUS, ours, out["verdicts"]):
        if addr == "127.0.0.2":
            # The copy's one deliberate difference (see guard_tree) — and
            # the proof that the child read the copy, not the repo's file.
            assert theirs is False
            continue
        assert mine == theirs, addr


_ADDRESS_CORPUS = [
    "127.0.0.1", "127.0.0.2", "10.1.2.3", "172.16.0.1", "172.32.0.1",
    "192.168.1.1", "169.254.169.254", "100.64.0.1", "100.128.0.1",
    "0.0.0.0", "224.0.0.1", "240.0.0.1", "8.8.8.8", "1.1.1.1",
    "::1", "fc00::1", "fe80::1", "::ffff:127.0.0.1", "::ffff:8.8.8.8",
    "2606:4700:4700::1111", "not-an-address",
]


@pytest.mark.parametrize("addr", _ADDRESS_CORPUS)
def test_host_gate_agrees_with_the_address_gate(addr, monkeypatch):
    """_host_is_forbidden is only a resolver in front of the ONE predicate."""
    monkeypatch.setattr(net_policy.socket, "getaddrinfo",
                        lambda *a, **k: [(0, 0, 0, "", (addr, 0))])
    assert (net_policy.host_is_forbidden("whatever")
            is net_policy.address_is_forbidden(addr))


def test_classify_error_maps_a_guard_refusal_without_leaking():
    raw = (f"ERROR: {udl.GUARD_MARKER}: evil.example resolves to the "
           f"forbidden address 169.254.169.254 — refusing to fetch")
    msg = udl.classify_error(raw)
    assert msg == "the site could not be reached from the server"
    assert "169.254" not in msg and "evil.example" not in msg


def test_classify_error_tells_an_external_downloader_refusal_apart():
    """The site WAS reached: a format only an external program could fetch
    is refused by the guard, which is no reason to retry."""
    guard = _inproc_guard()
    assert guard.EXTERNAL_FD_MARKER == udl.GUARD_EXTERNAL_FD_MARKER
    with pytest.raises(Exception) as ei:
        guard._refuse_external_fd(external.FFmpegFD.__new__(external.FFmpegFD),
                                  "f", {})
    assert (udl.classify_error(f"ERROR: {ei.value}")
            == "this media needs a downloader the server does not allow")


class _FakeSock:
    """Records the address the pinned connect dials; never touches the wire."""
    dialled: list = []

    def __init__(self, *a):
        pass

    def settimeout(self, t):
        pass

    def bind(self, a):
        pass

    def connect(self, addr):
        _FakeSock.dialled.append(addr)

    def setsockopt(self, *a):
        pass

    def close(self):
        pass


@pytest.fixture
def private_proxy(monkeypatch):
    """Every name resolves to 10.0.0.5 (a private proxy) and sockets are fake."""
    _FakeSock.dialled = []
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port))])
    monkeypatch.setattr(socket, "socket", _FakeSock)
    return _FakeSock


def test_net_policy_trusts_the_proxy_hop_but_not_a_direct_private_host(private_proxy):
    """With http(s)_proxy set the socket goes to the proxy, which may sit on a
    private address (a corporate squid, a docker sidecar). That hop is the
    operator's choice: dial it. A DIRECT connection to the same address is
    still refused — the target policy is unchanged."""
    direct = net_policy.PinnedHTTPConnection("proxy.example", 3128)
    with pytest.raises(OSError):
        direct.connect()
    assert private_proxy.dialled == []

    proxied = net_policy.proxied_conn_factory(
        net_policy.PinnedHTTPConnection, True)("proxy.example:3128")
    proxied.connect()
    assert private_proxy.dialled == [("10.0.0.5", 3128)]

    # HTTPS through CONNECT: the tunnel host marks the proxy hop on its own,
    # and SNI still names the target.
    seen = {}

    class FakeCtx:
        def wrap_socket(self, sock, server_hostname=None):
            seen["sni"] = server_hostname
            return sock

    tunnel = net_policy.PinnedHTTPSConnection("proxy.example", 3128, context=FakeCtx())
    tunnel.set_tunnel("target.example", 443)
    with mock.patch.object(tunnel, "_tunnel"):
        tunnel.connect()
    assert private_proxy.dialled[-1] == ("10.0.0.5", 3128)
    assert seen["sni"] == "target.example"


def test_guard_trusts_the_proxy_hop_but_not_a_direct_private_host(private_proxy):
    """Same contract in yt-dlp's opener (the guard copy of the pin)."""
    guard = _inproc_guard()

    direct = guard._PinnedHTTPConnection("proxy.example", 3128)
    with pytest.raises(RequestError, match="forbidden address"):
        direct.connect()
    assert private_proxy.dialled == []

    proxied = guard._PinnedHTTPConnection("proxy.example", 3128)
    proxied.via_proxy = True
    proxied.connect()
    assert private_proxy.dialled == [("10.0.0.5", 3128)]


def test_pinned_handlers_stamp_via_proxy_from_the_request(monkeypatch):
    """urllib's do_open only hands the host to the connection class, so the
    handlers are where 'this request was rewritten to go via a proxy' is
    known; both openers must carry it onto the connection."""
    import urllib.request
    built = {}

    def fake_do_open(self, http_class, req, **kw):
        built["conn"] = http_class("proxy.example:3128", timeout=1)
        return None

    monkeypatch.setattr(urllib.request.AbstractHTTPHandler, "do_open", fake_do_open)
    plain = urllib.request.Request("http://target.example/x")
    udl._PinnedHTTPHandler().http_open(plain)
    assert built["conn"].via_proxy is False

    proxied = urllib.request.Request("http://target.example/x")
    proxied.set_proxy("proxy.example:3128", "http")
    udl._PinnedHTTPHandler().http_open(proxied)
    assert built["conn"].via_proxy is True

    guard = _inproc_guard()
    handler = guard._GuardedHTTPHandler(context=None, source_address=None)
    handler.http_open(plain)
    assert built["conn"].via_proxy is False
    handler.http_open(proxied)
    assert built["conn"].via_proxy is True


# A public IP literal: no DNS, and every test below either refuses before a
# socket opens or reaches it only through the loopback proxy listener.
_PUBLIC_LITERAL = "93.184.216.34"


class _ProxyListener(_QuietHandler):
    """Stands in for a LAN service an extractor names as a 'proxy'. Every
    request it receives through a page-chosen proxy is a guard failure."""

    hits: "list[str]" = []
    body = b'{"PresentationDetailsList": [], "CurrentFolder": {"Name": "x"}}'

    def do_GET(self):
        type(self).hits.append(f"{self.command} {self.path}")
        self._ok_headers("application/json", len(self.body))
        self.wfile.write(self.body)

    do_POST = do_GET


@pytest.fixture
def proxy_listener(monkeypatch):
    """A 127.0.0.1 listener with a clean hit log; no proxy from the env."""
    for var in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.upper(), raising=False)
    srv = _serve(_ProxyListener, "127.0.0.1")
    _ProxyListener.hits = []
    with _stopping(srv):
        yield f"http://127.0.0.1:{srv.server_port}"


def test_guard_refuses_a_per_request_proxy_header(proxy_listener):
    """Ytdl-Request-Proxy becomes request.proxies, and the proxy hop is
    trusted — so a header must not be able to name the proxy."""
    from yt_dlp.networking.common import Request
    guard = _inproc_guard()
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        with pytest.raises(RequestError, match=guard.MARKER):
            ydl.urlopen(Request(f"http://{_PUBLIC_LITERAL}/x",
                                headers={"Ytdl-Request-Proxy": proxy_listener}))
    assert _ProxyListener.hits == []


def test_guard_refuses_mediasite_catalog_header_injection(proxy_listener,
                                                          monkeypatch):
    """MediasiteCatalog copies a header NAME and VALUE from the page onto its
    JSON POST; a page naming Ytdl-Request-Proxy must not reach the LAN."""
    from yt_dlp.extractor.mediasite import MediasiteCatalogIE
    guard = _inproc_guard()
    page = (f'AntiForgeryToken: "{proxy_listener}", '
            'AntiForgeryHeaderName: "Ytdl-Request-Proxy"')
    monkeypatch.setattr(MediasiteCatalogIE, "_download_webpage",
                        lambda self, *a, **kw: page)
    url = (f"http://{_PUBLIC_LITERAL}/Mediasite/Catalog/Full/"
           "631f9e48530d454381549f955d08c75e21")
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        # A bare RequestError, like every other guard refusal: extractors
        # only rewrap HTTPError / TransportError.
        with pytest.raises(RequestError, match=guard.MARKER):
            ydl.extract_info(url, download=False, ie_key="MediasiteCatalog",
                             process=False)
    assert _ProxyListener.hits == []


def test_guard_still_uses_the_operators_proxy(proxy_listener):
    """The operator's proxy ('proxy' param / http(s)_proxy env) is the RH's
    own and is dialled even on a private address — also when a request names
    it again per request (yt-dlp's PO-token helper does exactly that)."""
    from yt_dlp.networking.common import Request
    _inproc_guard()
    target = f"http://{_PUBLIC_LITERAL}/x"
    with yt_dlp.YoutubeDL({"quiet": True, "proxy": proxy_listener}) as ydl:
        assert ydl.urlopen(Request(target)).status == 200
        assert ydl.urlopen(Request(target, proxies={"all": proxy_listener})
                           ).status == 200
    assert _ProxyListener.hits == [f"GET {target}"] * 2


def test_guard_still_uses_the_proxy_from_the_env(proxy_listener, monkeypatch):
    from yt_dlp.networking.common import Request
    _inproc_guard()
    monkeypatch.setenv("http_proxy", proxy_listener)
    target = f"http://{_PUBLIC_LITERAL}/x"
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        assert ydl.urlopen(Request(target)).status == 200
    assert _ProxyListener.hits == [f"GET {target}"]


def test_a_per_request_no_proxy_is_still_gated_as_a_direct_dial(proxy_listener):
    """{'all': None} names no proxy: nothing to trust, so it is not refused as
    an override — the direct dial meets the address policy instead."""
    from yt_dlp.networking.common import Request
    _inproc_guard()
    with yt_dlp.YoutubeDL({"quiet": True, "proxy": proxy_listener}) as ydl:
        with pytest.raises(RequestError, match="forbidden address"):
            ydl.urlopen(Request(f"{proxy_listener}/x", proxies={"all": None}))
    assert _ProxyListener.hits == []


# ---- Content-Encoding: yt-dlp inflates a compressed response whole, in memory,
# before anyone reads it (in-process that is the server's own probe), so the
# guard's handler bounds what a body may expand to.

class _Encoded(_QuietHandler):
    """Answers every request (through the operator proxy) with `body` sent
    under `encoding`."""

    body = b""
    encoding = "gzip"

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Encoding", type(self).encoding)
        self.send_header("Content-Length", str(len(type(self).body)))
        self.end_headers()
        self.wfile.write(type(self).body)


def _registered_guard():
    """The globals of the guard load whose handler yt-dlp actually uses: the
    file loads more than once (a forced self-check reloads it under the same
    module name) and only the first load registers (install())."""
    udl.guard_self_check(force=True)
    from yt_dlp.networking.common import _REQUEST_HANDLERS
    return _REQUEST_HANDLERS["FwbSsrfGuard"]._send.__globals__


@pytest.fixture
def encoded_listener(monkeypatch):
    for var in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.upper(), raising=False)
    srv = _serve(_Encoded, "127.0.0.1")
    with _stopping(srv):
        yield f"http://127.0.0.1:{srv.server_port}"


def _raw_deflate(data):
    c = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return c.compress(data) + c.flush()


_ENCODERS = {
    "gzip": gzip.compress,
    "deflate": _raw_deflate,
    "deflate-zlib": zlib.compress,  # the zlib-wrapped fallback yt-dlp accepts
}


def _fetch_encoded(listener, encoding, body):
    from yt_dlp.networking.common import Request
    _Encoded.encoding = encoding.split("-")[0]
    _Encoded.body = _ENCODERS[encoding](body)
    with yt_dlp.YoutubeDL({"quiet": True, "proxy": listener}) as ydl:
        return ydl.urlopen(Request(f"http://{_PUBLIC_LITERAL}/page")).read()


@pytest.mark.parametrize("encoding", sorted(_ENCODERS))
def test_guard_refuses_a_compression_bomb(encoded_listener, encoding,
                                          monkeypatch):
    guard = _registered_guard()
    monkeypatch.setitem(guard, "_MAX_DECODED_BYTES", 1 << 20)
    with pytest.raises(RequestError, match=guard["MARKER"]):
        _fetch_encoded(encoded_listener, encoding, b"\0" * (4 << 20))


@pytest.mark.parametrize("encoding", sorted(_ENCODERS))
def test_guard_decodes_a_compressed_page_unchanged(encoded_listener, encoding,
                                                   monkeypatch):
    guard = _registered_guard()
    monkeypatch.setitem(guard, "_MAX_DECODED_BYTES", 1 << 20)
    page = os.urandom(1000) + b"<html>" * 50_000  # ~300 KB, under the cap
    assert _fetch_encoded(encoded_listener, encoding, page) == page
    exact = b"a" * (1 << 20)  # exactly the cap still decodes
    assert _fetch_encoded(encoded_listener, encoding, exact) == exact


def _outcome(fn, data):
    try:
        return ("ok", fn(data))
    except zlib.error as e:
        return ("zlib.error", str(e))


_GZ = gzip.compress(b"hello " * 1000)
_RAW = _raw_deflate(b"hello " * 1000)
_ZLIB = zlib.compress(b"hello " * 1000)


@pytest.mark.parametrize("method, data", [
    ("gz", b""),
    ("gz", _GZ),
    ("gz", _GZ + b"trailing junk"),
    ("gz", _GZ + gzip.compress(b"second member")),  # only ONE member decodes
    ("gz", _GZ[:-3]),                               # truncated
    ("gz", _GZ[:10] + b"\xff" * 40),                # corrupt
    ("gz", _RAW),                                   # not gzip at all
    ("deflate", b""),
    ("deflate", _RAW),
    ("deflate", _RAW + b"junk"),
    ("deflate", _ZLIB),                             # zlib-wrapped fallback
    ("deflate", _ZLIB + b"junk"),
    ("deflate", _RAW[:-4]),
    ("deflate", _ZLIB[:-4]),
    ("deflate", b"\xff" * 40),
])
def test_bounded_decoders_match_yt_dlp_under_the_cap(method, data):
    """Every input under the cap decodes to the same bytes, or fails with the
    same zlib.error (yt-dlp maps that to a TransportError), as yt-dlp's own."""
    from yt_dlp.networking import _urllib
    guard = _registered_guard()
    ours = getattr(guard["_GuardedHTTPHandler"], method)
    theirs = getattr(_urllib.HTTPHandler, method)
    assert _outcome(ours, data) == _outcome(theirs, data)


class _FakeBrotliError(Exception):
    pass


def _fake_brotli(output, *, finished=True, limit_supported=True):
    class Decompressor:
        def process(self, data, **kw):
            if kw and not limit_supported:
                raise TypeError("process() takes no keyword arguments")
            return output

        def is_finished(self):
            return finished

    return types.SimpleNamespace(Decompressor=Decompressor,
                                 error=_FakeBrotliError,
                                 decompress=lambda data: output)


def test_bounded_brotli(monkeypatch):
    from yt_dlp.networking import _urllib
    guard = _registered_guard()
    monkeypatch.setitem(guard, "_MAX_DECODED_BYTES", 100)
    br = guard["_GuardedHTTPHandler"].brotli
    monkeypatch.setattr(_urllib, "brotli", _fake_brotli(b"x" * 100))
    assert br(b"compressed") == b"x" * 100
    assert br(b"") == b""
    monkeypatch.setattr(_urllib, "brotli", _fake_brotli(b"x" * 101))
    with pytest.raises(RequestError, match=guard["MARKER"]):
        br(b"compressed")
    # brotli.decompress raises on a truncated stream; so does the bound one.
    monkeypatch.setattr(_urllib, "brotli",
                        _fake_brotli(b"x", finished=False))
    with pytest.raises(_FakeBrotliError):
        br(b"compressed")
    # A brotli module without an output limit (brotlicffi, Brotli < 1.2)
    # is refused rather than inflated unbounded.
    monkeypatch.setattr(_urllib, "brotli",
                        _fake_brotli(b"x", limit_supported=False))
    with pytest.raises(RequestError, match=guard["MARKER"]):
        br(b"compressed")


# ---- Stacked Content-Encodings: yt-dlp decodes one layer per comma token, each
# bounded on its own, so the number of layers is bounded too.

def _fetch_stacked(listener, encoding, body):
    from yt_dlp.networking.common import Request
    _Encoded.encoding = encoding
    _Encoded.body = body
    with yt_dlp.YoutubeDL({"quiet": True, "proxy": listener}) as ydl:
        return ydl.urlopen(Request(f"http://{_PUBLIC_LITERAL}/page")).read()


def test_guard_refuses_stacked_content_codings(encoded_listener):
    guard = _registered_guard()
    page = b"<html>" * 1000
    body = gzip.compress(gzip.compress(gzip.compress(page)))
    with pytest.raises(RequestError, match=guard["MARKER"]):
        _fetch_stacked(encoded_listener, "gzip, gzip, gzip", body)
    with pytest.raises(RequestError, match=guard["MARKER"]):
        _fetch_stacked(encoded_listener, "gzip,deflate,br",
                       gzip.compress(_raw_deflate(page)))


def test_guard_decodes_two_stacked_gzip_codings(encoded_listener):
    """A double-gzipping proxy is the one multi-coding case seen in practice."""
    _registered_guard()
    page = b"<html>" * 1000
    assert _fetch_stacked(encoded_listener, "gzip, gzip",
                          gzip.compress(gzip.compress(page))) == page
    # An unknown token is not decoded by yt-dlp, so it does not count.
    assert _fetch_stacked(encoded_listener, "identity, gzip, gzip",
                          gzip.compress(gzip.compress(page))) == page


def test_guarded_handler_binds_https_response_to_its_own_override():
    handler = _registered_guard()["_GuardedHTTPHandler"]
    assert handler.https_response is handler.http_response
    from yt_dlp.networking import _urllib
    assert handler.http_response is not _urllib.HTTPHandler.http_response


# ---- A whole-body read (read() with no size: yt-dlp's webpage / manifest / API
# reads and its decoders' raw read) is bounded; sized reads (HttpFD's
# read(block_size)) are not touched.

class _Plain(_QuietHandler):
    """`body` as text/html, with or without a Content-Length."""

    body = b""
    with_length = True

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        if type(self).with_length:
            self.send_header("Content-Length", str(len(type(self).body)))
        self.end_headers()
        self.wfile.write(type(self).body)


@pytest.fixture
def plain_listener(monkeypatch):
    for var in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.upper(), raising=False)
    srv = _serve(_Plain, "127.0.0.1")
    with _stopping(srv):
        yield f"http://127.0.0.1:{srv.server_port}"


def _read_plain(listener, body, *, with_length, block=None):
    from yt_dlp.networking.common import Request
    _Plain.body = body
    _Plain.with_length = with_length
    with yt_dlp.YoutubeDL({"quiet": True, "proxy": listener}) as ydl:
        resp = ydl.urlopen(Request(f"http://{_PUBLIC_LITERAL}/page"))
        if block is None:
            return resp.read()
        parts = []
        while chunk := resp.read(block):
            parts.append(chunk)
        return b"".join(parts)


@pytest.mark.parametrize("with_length", [True, False])
def test_guard_refuses_a_whole_body_read_past_the_cap(plain_listener,
                                                      with_length, monkeypatch):
    guard = _registered_guard()
    monkeypatch.setitem(guard, "_MAX_DECODED_BYTES", 1 << 20)
    with pytest.raises(RequestError, match=guard["MARKER"]):
        _read_plain(plain_listener, b"x" * (4 << 20), with_length=with_length)


@pytest.mark.parametrize("with_length", [True, False])
def test_guard_passes_whole_bodies_under_the_cap_and_sized_reads(
        plain_listener, with_length, monkeypatch):
    guard = _registered_guard()
    monkeypatch.setitem(guard, "_MAX_DECODED_BYTES", 1 << 20)
    page = os.urandom(1000) + b"<html>" * 50_000
    assert _read_plain(plain_listener, page, with_length=with_length) == page
    exact = b"a" * (1 << 20)
    assert _read_plain(plain_listener, exact, with_length=with_length) == exact
    media = os.urandom(4 << 20)  # a media fetch reads in blocks: never capped
    assert _read_plain(plain_listener, media, with_length=with_length,
                       block=32768) == media



# ---- The in-process extraction's dial loop spends from the probe's deadline:
# the dial observer only sees a socket once connect returns, so a name with
# many black-holed answers held the worker for N x socket_timeout.

def test_guard_connect_pinned_stops_at_the_observed_deadline(monkeypatch):
    import time
    guard = _inproc_guard()
    attempts = _black_hole_dns(monkeypatch, 5)
    conn = types.SimpleNamespace(host="blackhole.example", port=80, timeout=1.0,
                                 source_address=None)
    t0 = time.monotonic()
    guard.observe_dials(lambda sock: None, deadline=t0 + 0.5)
    try:
        with pytest.raises(TimeoutError):
            guard._connect_pinned(conn)
    finally:
        guard.observe_dials(None)
    assert attempts and time.monotonic() - t0 < 0.9
    # Stopping the observer drops the deadline: the next dial is as before.
    assert getattr(sys.modules["fwb_ssrf_guard_dials"].tls, "deadline") is None


def test_probe_extraction_dial_loop_honours_the_wall_clock(monkeypatch):
    import time
    attempts = _black_hole_dns(monkeypatch, 8)
    udl.guard_self_check(force=True)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_SOCKET_TIMEOUT_S", 1.0, raising=False)

    async def _generic(url, *, deadline=None):
        return "Generic"
    monkeypatch.setattr(udl, "check_url_policy", _generic)
    t0 = time.monotonic()
    with pytest.raises(udl.UrlDownloadError):
        _run(udl.probe("http://blackhole.example/a.mp3", timeout=2.0))
    # The worker thread outlives the abandoned await: give it time to dial
    # on (it did, for 8 x 1 s), then check it stopped at the deadline.
    time.sleep(max(0.0, t0 + 4.5 - time.monotonic()))
    assert attempts
    assert max(attempts) - t0 < 2.5, [round(a - t0, 2) for a in attempts]


# ---- DASH: yt-dlp expands a SegmentTemplate / SegmentTimeline into one dict
# per segment, in process, with no cap — a ~300-byte manifest can ask for
# billions. The guard counts the segments before the expansion.

_MPD_NS = "urn:mpeg:dash:schema:mpd:2011"


def _mpd(body, *, duration="PT14400S"):
    dur = f' mediaPresentationDuration="{duration}"' if duration else ""
    return (f'<?xml version="1.0"?><MPD xmlns="{_MPD_NS}" type="static"{dur}>'
            f'{body}</MPD>')


def _template(**attrs):
    a = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    return f'<SegmentTemplate media="s$Number$.m4s" {a}/>'


def _adaptation(inner, reps=1):
    return ('<Period><AdaptationSet mimeType="audio/mp4">' + inner
            + "".join(f'<Representation id="r{i}" bandwidth="{i + 1}" '
                      'codecs="mp4a.40.2"/>' for i in range(reps))
            + "</AdaptationSet></Period>")


def _parse_mpd(xml):
    from yt_dlp.extractor.common import InfoExtractor
    from yt_dlp.utils import compat_etree_fromstring
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        ie = InfoExtractor(ydl)
        return ie._parse_mpd_formats_and_subtitles(
            compat_etree_fromstring(xml.encode()),
            mpd_base_url="http://cdn.example/", mpd_url="http://cdn.example/m.mpd")


@pytest.mark.parametrize("xml", [
    # 14 400 s at 1 µs per segment: 1.44e10 fragment dicts.
    _mpd(_adaptation(_template(timescale="1000000", duration="1"))),
    # A SegmentTimeline repeat count, likewise unbounded by the manifest size.
    _mpd(_adaptation('<SegmentTemplate media="s$Time$.m4s" timescale="1">'
                     '<SegmentTimeline><S t="0" d="1" r="99999999999"/>'
                     '</SegmentTimeline></SegmentTemplate>')),
    # Many representations each expanding a modest template.
    _mpd(_adaptation(_template(timescale="1", duration="1"), reps=100),
         duration="PT30000S"),
    # No duration on the MPD: the first representation's timeline sets the
    # period's length, which the next one's template then expands.
    _mpd('<Period><AdaptationSet mimeType="audio/mp4">'
         '<Representation id="a" bandwidth="1" codecs="mp4a.40.2">'
         '<SegmentTemplate media="a$Time$.m4s" timescale="1"><SegmentTimeline>'
         '<S t="0" d="100000000000"/></SegmentTimeline></SegmentTemplate>'
         '</Representation>'
         '<Representation id="b" bandwidth="1" codecs="mp4a.40.2">'
         '<SegmentTemplate media="b$Number$.m4s" timescale="1" duration="1"/>'
         '</Representation></AdaptationSet></Period>', duration=None),
])
def test_guard_refuses_a_manifest_listing_too_many_segments(xml):
    import time
    t0 = time.monotonic()
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many segments"):
        _parse_mpd(xml)
    assert time.monotonic() - t0 < 2.0


def test_guard_parses_an_ordinary_long_manifest_unchanged(monkeypatch):
    """A 4 h stream at 2 s segments (7 200) across a few representations, and
    a timeline, expand exactly as yt-dlp does on its own."""
    xml = _mpd(_adaptation(_template(timescale="1000", duration="2000"), reps=5))
    fmts, _subs = _parse_mpd(xml)
    assert len(fmts) == 5
    assert all(len(f["fragments"]) == 7200 for f in fmts)
    timeline = _mpd(_adaptation(
        '<SegmentTemplate media="s$Time$.m4s" timescale="1000"><SegmentTimeline>'
        '<S t="0" d="2000" r="7199"/></SegmentTimeline></SegmentTemplate>'))
    fmts, _subs = _parse_mpd(timeline)
    assert len(fmts[0]["fragments"]) == 7200
    # Right at the cap still parses, one past it is refused (the cap is
    # lowered so the at-cap manifest stays cheap to expand), in an
    # extraction of its own.
    assert _registered_guard()["_MAX_MPD_SEGMENTS"] >= 216_000
    _reset_budget()
    cap = 50_000
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_SEGMENTS", cap)
    at_cap = _mpd(_adaptation(_template(timescale="1", duration="1")),
                  duration=f"PT{cap}S")
    fmts, _subs = _parse_mpd(at_cap)
    assert len(fmts[0]["fragments"]) == cap
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many segments"):
        _parse_mpd(_mpd(_adaptation(_template(timescale="1", duration="1")),
                        duration=f"PT{cap + 1}S"))


def test_guard_is_not_installed_without_the_mpd_bound(monkeypatch):
    from yt_dlp.extractor.common import InfoExtractor
    guard = _inproc_guard()
    assert guard.is_installed()
    with monkeypatch.context() as m:
        m.setattr(InfoExtractor, "_parse_mpd_periods",
                  lambda self, *a, **kw: iter(()))
        assert not guard.is_installed()
    with monkeypatch.context() as m:  # a yt-dlp that renamed the method
        m.delattr(InfoExtractor, "_parse_mpd_periods")
        assert not guard.is_installed()
        guard.install()  # nothing to bound: still fails closed
        assert not guard.is_installed()
    assert guard.is_installed()


def test_guard_counts_segments_under_a_root_yt_dlp_does_not_namespace(monkeypatch):
    """yt-dlp takes the namespace only from a root named MPD (any case) and
    otherwise looks for unqualified Period elements — _extract_mpd_periods
    hands it whatever XML root the URL served. The count must look where the
    expansion looks, or a namespaced <x:Root> expands uncounted."""
    cap = 50
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_SEGMENTS", cap)
    body = _adaptation(_template(timescale="1", duration="1"))
    for root in ("x:Root", "x:MPDX", "x:Period"):
        xml = (f'<?xml version="1.0"?><{root} xmlns:x="urn:x" '
               f'mediaPresentationDuration="PT{cap + 1}S">{body}</{root}>')
        with pytest.raises(yt_dlp.utils.ExtractorError, match="too many segments"):
            _parse_mpd(xml)
    # A lower-case mpd root is still yt-dlp's namespaced case.
    xml = (f'<?xml version="1.0"?><mpd xmlns="{_MPD_NS}" '
           f'mediaPresentationDuration="PT{cap + 1}S">{body}</mpd>')
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many segments"):
        _parse_mpd(xml)


def test_guard_refuses_an_mpd_listing_too_many_representations(monkeypatch):
    """Every Representation is a format yt-dlp parses and processes, so a
    manifest of one-segment representations stays under the segment budget
    yet costs seconds per thousand. They are counted across periods."""
    cap = 40
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_REPRESENTATIONS", cap)
    plain = '<BaseURL>a.m4a</BaseURL>'
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_mpd(_mpd(_adaptation(plain, reps=cap + 1)))
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_mpd(_mpd(_adaptation(plain, reps=cap // 2 + 1) * 2))
    formats, _ = _parse_mpd(_mpd(_adaptation(plain, reps=cap // 2) * 2))
    assert formats


def test_guard_counts_empty_periods_and_adaptation_sets(monkeypatch):
    """A Period or an AdaptationSet without a Representation yields no format
    and yet is walked, by the count and by yt-dlp: 1.9 million empty Periods
    (59 MB decoded) took 30 s of CPU. Both are held to the variant cap."""
    cap = 100
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_REPRESENTATIONS", cap)
    _reset_budget()
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_mpd(_mpd("<Period/>" * (cap + 1)))
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_mpd(_mpd("<Period>" + "<AdaptationSet/>" * (cap + 1) + "</Period>"))
    # Counted across periods, like the representations.
    half = "<Period>" + "<AdaptationSet/>" * (cap // 2 + 1) + "</Period>"
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_mpd(_mpd(half * 2))
    # A refused manifest spends nothing, and one at the cap parses.
    assert _registered_guard()["_DIALS"].tls.variants == 0
    formats, _ = _parse_mpd(_mpd(_adaptation("<BaseURL>a.m4a</BaseURL>") * cap))
    assert formats
    assert _registered_guard()["_DIALS"].tls.variants == cap


def _mpd_beside(reps=50, *, root="", period="", in_set="", set_attrs=""):
    """`reps` Representations with other content beside them at each level."""
    return _mpd(
        root + "<Period>" + period
        + f'<AdaptationSet mimeType="audio/mp4"{set_attrs}>' + in_set
        + "<BaseURL>a.m4a</BaseURL>"
        + "".join(f'<Representation id="r{i}" bandwidth="{i + 1}" '
                  'codecs="mp4a.40.2"/>' for i in range(reps))
        + "</AdaptationSet></Period>")


@pytest.mark.parametrize("beside", [
    {"in_set": "<a/>" * 200},
    {"period": "<a/>" * 200},
    {"root": "<a/>" * 200},
    {"set_attrs": "".join(f' a{i}=""' for i in range(200))},
], ids=["in the set", "under the period", "under the root", "set attributes"])
def test_guard_charges_what_yt_dlp_rewalks_per_representation(monkeypatch, beside):
    """For every Representation yt-dlp searches the children of the MPD, of
    the Period and (twice) of the AdaptationSet and copies the set's
    attributes: 1 000 Representations beside 1.9 million empty elements
    (7.6 MB) took 36 s of CPU with every count within its cap."""
    guard = _registered_guard()
    assert guard["_MAX_MANIFEST_WALK"] == 2 ** 30
    monkeypatch.setitem(guard, "_MAX_MANIFEST_WALK", 10_000)
    _reset_budget()
    with pytest.raises(yt_dlp.utils.ExtractorError, match="manifest is too large"):
        _parse_mpd(_mpd_beside(**beside))
    # A refused manifest spends nothing; the same Representations with
    # nothing beside them parse, and spend from the extraction's budget.
    assert guard["_DIALS"].tls.walk == 0
    formats, _ = _parse_mpd(_mpd_beside())
    assert len(formats) == 50
    # Per Representation: the root's and the Period's one child, the set's
    # 51 children and one attribute twice, its own three attributes.
    spent = guard["_DIALS"].tls.walk
    assert spent == 50 * (1 + 1 + 2 * 51 + 2 * 1 + 3)
    # One extraction parses many manifests: each within the cap is not enough.
    with pytest.raises(yt_dlp.utils.ExtractorError, match="manifest is too large"):
        _parse_mpd(_mpd_beside())
    assert guard["_DIALS"].tls.walk == spent
    _reset_budget()
    assert guard["_DIALS"].tls.walk == 0
    assert len(_parse_mpd(_mpd_beside())[0]) == 50


def test_guard_admits_the_widest_manifests_the_variant_cap_allows():
    """What yt-dlp re-walks is bounded on top of the variant cap, not in its
    place: all 20 000 Representations in one AdaptationSet, or one each in
    20 000 Periods, stay inside the bound (sized here, not parsed)."""
    guard = _registered_guard()
    variants = guard["_MAX_MPD_REPRESENTATIONS"]
    # The charge of the test above, at the variant cap.
    one_set = variants * (1 + 1 + 2 * (variants + 1) + 2 * 1 + 3)
    one_each = variants * (variants + 1 + 2 * (1 + 1) + 2 * 1 + 3)
    assert max(one_set, one_each) <= guard["_MAX_MANIFEST_WALK"]
    # Smooth Streaming, per QualityLevel: the root's children and its
    # StreamIndex's. Every level in one StreamIndex, or a StreamIndex each,
    # with one chunk.
    assert variants * (1 + variants + 1) <= guard["_MAX_MANIFEST_WALK"]
    assert variants * (variants + 1 + 1) <= guard["_MAX_MANIFEST_WALK"]


# ---- Smooth Streaming: yt-dlp expands every <c r="N"> into N fragment dicts
# for each QualityLevel of the stream, in process and with no cap. The guard
# counts them against the DASH budget before the expansion.

def _ism(streams, *, live=False):
    flag = ' IsLive="TRUE"' if live else ""
    return ('<?xml version="1.0"?><SmoothStreamingMedia MajorVersion="2" '
            f'MinorVersion="0" Duration="72000000000" TimeScale="10000000"{flag}>'
            f'{streams}</SmoothStreamingMedia>')


def _ism_audio(chunks, *, levels=1, kind="audio"):
    return (f'<StreamIndex Type="{kind}" Name="audio" Language="eng" '
            'Url="QualityLevels({bitrate})/Fragments(audio={start time})">'
            + "".join(f'<QualityLevel Index="{i}" Bitrate="{128000 + i}" '
                      'FourCC="AACL" SamplingRate="48000" Channels="2" '
                      'BitsPerSample="16" PacketSize="4" AudioTag="255" '
                      'CodecPrivateData="1190"/>' for i in range(levels))
            + chunks + "</StreamIndex>")


def _parse_ism(xml):
    from yt_dlp.extractor.common import InfoExtractor
    from yt_dlp.utils import compat_etree_fromstring
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        ie = InfoExtractor(ydl)
        return ie._parse_ism_formats_and_subtitles(
            compat_etree_fromstring(xml.encode()), "http://cdn.example/x.ism/Manifest")


@pytest.mark.parametrize("xml", [
    # One chunk repeated ten billion times.
    _ism(_ism_audio('<c t="0" d="20000000" r="10000000000"/>')),
    # The repeat is expanded once per QualityLevel: 4 x 300 000.
    _ism(_ism_audio('<c t="0" d="20000000" r="300000"/>', levels=4)),
], ids=["r=1e10", "4x300000"])
def test_guard_refuses_an_ism_manifest_listing_too_many_fragments(xml):
    import time
    t0 = time.monotonic()
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many segments"):
        _parse_ism(xml)
    assert time.monotonic() - t0 < 2.0


def test_guard_parses_an_ordinary_ism_manifest_unchanged(monkeypatch):
    """A 2 h VOD at 2 s chunks over two qualities expands exactly as yt-dlp
    does on its own; right at the cap still parses, one past it is refused."""
    fmts, _subs = _parse_ism(_ism(_ism_audio(
        '<c t="0" d="20000000" r="3599"/><c d="20000000"/>', levels=2)))
    assert len(fmts) == 2
    assert all(len(f["fragments"]) == 3600 for f in fmts)
    # A live manifest is left to yt-dlp, which returns nothing for it.
    assert _parse_ism(_ism(_ism_audio('<c t="0" d="1" r="10000000000"/>'),
                           live=True)) == ([], {})
    _reset_budget()  # the at-cap manifest is an extraction of its own
    cap = 30_000
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_SEGMENTS", cap)
    fmts, _subs = _parse_ism(_ism(_ism_audio(
        f'<c t="0" d="1" r="{cap // 3}"/>', levels=3)))
    assert sum(len(f["fragments"]) for f in fmts) == cap
    for over in (_ism_audio(f'<c t="0" d="1" r="{cap // 3}"/><c d="1"/>', levels=3),
                 _ism_audio(f'<c t="0" d="1" r="{cap}"/>')
                 + _ism_audio('<c t="0" d="1"/>', kind="text")):
        with pytest.raises(yt_dlp.utils.ExtractorError, match="too many segments"):
            _parse_ism(_ism(over))


# ---- Fragment addresses: yt-dlp builds every fragment's URL as a fresh copy
# of the manifest's template, so a short manifest with a huge format width, a
# long template or a long value substituted into it asks for gigabytes while
# its segment count stays tiny. The guard sizes those strings before the
# expansion, without building them.

def _ism_url(url, chunks, *, bitrate="128000"):
    return (f'<StreamIndex Type="audio" Name="audio" Language="eng" Url="{url}">'
            f'<QualityLevel Index="0" Bitrate="{bitrate}" FourCC="AACL" '
            'SamplingRate="48000" Channels="2" BitsPerSample="16" PacketSize="4" '
            'AudioTag="255" CodecPrivateData="1190"/>' + chunks + "</StreamIndex>")


def _live_template(media, initialization=None):
    init = f' initialization="{initialization}"' if initialization else ""
    return (f'<SegmentTemplate media="{media}"{init} timescale="1" '
            'duration="1"/>')


@pytest.mark.parametrize("parse,xml", [
    # A 300-byte MPD: fifty segments of ten million characters each.
    (_parse_mpd, _mpd(_adaptation(
        '<SegmentTemplate media="x$Number%010000000d$" timescale="1" '
        'duration="1"/>'), duration="PT50S")),
    # The same width in @initialization, built once per representation.
    (_parse_mpd, _mpd(_adaptation(
        '<SegmentTemplate media="s$Number$.m4s" '
        'initialization="i$Bandwidth%010000000d$" timescale="1" duration="1"/>',
        reps=30), duration="PT1S")),
    # $RepresentationID$ repeated next to a long @id: one 100 MB template.
    (_parse_mpd, _mpd(
        '<Period><AdaptationSet mimeType="audio/mp4"><SegmentTemplate media="'
        + "$RepresentationID$" * 2000 + '$Number$" timescale="1" duration="1"/>'
        f'<Representation id="{"i" * 50000}" bandwidth="1" codecs="mp4a.40.2"/>'
        "</AdaptationSet></Period>", duration="PT3S")),
    # A width spelled across the template and the @id it substitutes.
    (_parse_mpd, _mpd(
        '<Period><AdaptationSet mimeType="audio/mp4"><SegmentTemplate '
        'media="$$RepresentationID$$" timescale="1" duration="1"/>'
        '<Representation id="Number%010000000d" bandwidth="1" '
        'codecs="mp4a.40.2"/></AdaptationSet></Period>', duration="PT50S")),
    # No width at all: a 4 000-digit @startNumber printed a thousand times.
    (_parse_mpd, _mpd(_adaptation(
        '<SegmentTemplate media="' + "$Number$" * 1000 + '" timescale="1" '
        f'duration="1" startNumber="{"9" * 4000}"/>'), duration="PT100S")),
    # A one-megabyte MPD-level BaseURL joined afresh for each representation.
    (_parse_mpd, _mpd(f'<BaseURL>{"a" * 1_000_000}</BaseURL>' + _adaptation(
        "<BaseURL>b</BaseURL>", reps=300))),
    # A one-megabyte StreamIndex@Url copied for each of 400 fragments.
    (_parse_ism, _ism(_ism_url("a" * 1_000_000 + "/{start time}",
                               '<c t="0" d="1" r="400"/>'))),
    # {bitrate} repeated, each replaced by a 4 001-digit Bitrate.
    (_parse_ism, _ism(_ism_url("{bitrate}" * 5000 + "/{start time}",
                               '<c t="0" d="1" r="20"/>',
                               bitrate="0" * 4000 + "1"))),
    # {start time} repeated, each replaced by a 4 000-digit chunk time.
    (_parse_ism, _ism(_ism_url("{start time}" * 1000,
                               f'<c t="{"9" * 4000}" d="1" r="100"/>'))),
    # yt-dlp keeps every `%` after an odd `$` live, whatever the run names:
    # a width in a run that is no field prints the whole values dict.
    (_parse_mpd, _mpd(_adaptation(_live_template("s$%2000000s$Number$.m4s")),
                      duration="PT3S")),
    (_parse_mpd, _mpd(_adaptation(_live_template("$x%2000000s$$Number$")),
                      duration="PT3S")),
    # ... also in a run nothing closes,
    (_parse_mpd, _mpd(_adaptation(_live_template(
        "s$Number$.m4s", initialization="i$%2000000s")), duration="PT3S")),
    # ... as a precision, on a conversion that names its own key,
    (_parse_mpd, _mpd(_adaptation(_live_template("$%(Number).2000000d$")),
                      duration="PT3S")),
    # ... and in a run that reads as a field whose opening `$` the field
    # before it took (yt-dlp substitutes $Bandwidth$ alone in @initialization).
    (_parse_mpd, _mpd(_adaptation(_live_template(
        "a$b$Number$Number%(Number)2000000d$")), duration="PT3S")),
    (_parse_mpd, _mpd(_adaptation(_live_template(
        "s$Number$.m4s", initialization="a$b$Number$Number%2000000s$")),
        duration="PT3S")),
    # More fields, or more conversions inside one, than any real template.
    (_parse_mpd, _mpd(_adaptation(_live_template("$Number$" * 65)),
                      duration="PT3S")),
    (_parse_mpd, _mpd(_adaptation(_live_template(
        "$Number%5d" + "%(Number)64d" * 65 + "$")), duration="PT3S")),
    # More numbers in one conversion's text than the width check walks.
    (_parse_mpd, _mpd(_adaptation(_live_template(
        "$Number%05d" + "a1" * 65 + "$")), duration="PT3S")),
], ids=["mpd-width", "mpd-init-width", "mpd-representation-id",
        "mpd-width-across-id", "mpd-start-number", "mpd-base-url", "ism-long-url",
        "ism-bitrate", "ism-start-time", "mpd-width-outside-field",
        "mpd-width-unknown-name", "mpd-init-width-unterminated",
        "mpd-keyed-precision", "mpd-width-after-a-shifted-field",
        "mpd-init-width-after-a-shifted-field", "mpd-field-count",
        "mpd-conversion-count", "mpd-conversion-numbers"])
def test_guard_refuses_absurd_fragment_url_bytes(parse, xml):
    import time
    t0 = time.monotonic()
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
        parse(xml)
    assert time.monotonic() - t0 < 2.0


_BIG = "1" + "0" * 24  # 25 digits: past UINT64's 20
_YT_NS = "http://youtube.com/yt/2012/10/10"
_PLAIN_ISM_URL = "QualityLevels(128000)/Fragments(audio=0)"
_TIMELINE = ('<SegmentTemplate media="s$Number$.m4s" timescale="{timescale}" '
             'startNumber="{start}"><SegmentTimeline><S t="{t}" d="{d}" r="{r}"/>'
             '<S d="1"/></SegmentTimeline></SegmentTemplate>')


def _timeline(t="0", d="1", r="2", timescale="1", start="1"):
    return _mpd(_adaptation(_TIMELINE.format(t=t, d=d, r=r, timescale=timescale,
                                             start=start)))


def _ism_numbers(t="0", d="1", r="2", bitrate="128000", url=_PLAIN_ISM_URL):
    return _ism(_ism_url(url, f'<c t="{t}" d="{d}" r="{r}"/>', bitrate=bitrate))


def _mpd_content_length(value):
    """A Representation whose own BaseURL carries YouTube's size attribute."""
    return _mpd('<Period><AdaptationSet mimeType="audio/mp4"><Representation '
                'id="r0" bandwidth="1" codecs="mp4a.40.2">'
                f'<BaseURL xmlns:yt="{_YT_NS}" yt:contentLength="{value}">a.m4a'
                "</BaseURL></Representation></AdaptationSet></Period>")


def _ism_level_number(name, value=_BIG):
    """_ism_numbers() with the QualityLevel's `name` set to `value`."""
    xml = _ism_numbers()
    if f' {name}="' in xml:
        return re.sub(rf' {name}="[^"]*"', f' {name}="{value}"', xml)
    return xml.replace("<QualityLevel ", f'<QualityLevel {name}="{value}" ')


@pytest.mark.parametrize("parse,xml", [
    (_parse_ism, _ism_numbers().replace('Duration="72000000000"', f'Duration="{_BIG}"')),
    (_parse_ism, _ism_numbers().replace('TimeScale="10000000"', f'TimeScale="{_BIG}"')),
    (_parse_ism, _ism_numbers().replace("<StreamIndex ", f'<StreamIndex TimeScale="{_BIG}" ')),
    (_parse_ism, _ism_numbers(t=_BIG)),
    (_parse_ism, _ism_numbers(d=_BIG)),
    (_parse_ism, _ism_numbers(r="-" + "9" * 24)),
    (_parse_ism, _ism_numbers(bitrate=_BIG)),
    (_parse_mpd, _timeline(t=_BIG)),
    (_parse_mpd, _timeline(d=_BIG)),
    (_parse_mpd, _timeline(r="-" + "9" * 24)),
    (_parse_mpd, _timeline(timescale=_BIG)),
    (_parse_mpd, _timeline(start=_BIG)),
    (_parse_mpd, _timeline().replace('bandwidth="1"', f'bandwidth="{_BIG}"')),
    (_parse_mpd, _timeline().replace('bandwidth="1"', f'bandwidth="1" width="{_BIG}"')),
    (_parse_mpd, _timeline().replace('bandwidth="1"', f'bandwidth="1" height="{_BIG}"')),
    (_parse_mpd, _timeline().replace(
        'bandwidth="1"', f'bandwidth="1" audioSamplingRate="{_BIG}"')),
    (_parse_mpd, _timeline().replace(
        'bandwidth="1"', f'bandwidth="1" frameRate="{_BIG}"')),
    # yt-dlp overlays the Representation on its AdaptationSet's attributes.
    (_parse_mpd, _timeline().replace("<AdaptationSet ", f'<AdaptationSet bandwidth="{_BIG}" ')
     .replace(' bandwidth="1"', "")),
    (_parse_mpd, _timeline().replace("<AdaptationSet ", f'<AdaptationSet height="{_BIG}" ')),
    (_parse_mpd, _timeline().replace("<AdaptationSet ", f'<AdaptationSet frameRate="{_BIG}" ')),
    (_parse_mpd, _mpd_content_length(_BIG)),
    *((_parse_ism, _ism_level_number(name))
      for name in ("MaxWidth", "Width", "MaxHeight", "Height", "SamplingRate",
                   "Channels", "BitsPerSample", "NALUnitLengthField")),
], ids=["ism-duration", "ism-timescale", "ism-stream-timescale", "ism-t", "ism-d",
        "ism-r", "ism-bitrate", "mpd-t", "mpd-d", "mpd-r", "mpd-timescale",
        "mpd-start-number", "mpd-bandwidth", "mpd-width", "mpd-height",
        "mpd-audio-sampling-rate", "mpd-frame-rate", "mpd-set-bandwidth",
        "mpd-set-height", "mpd-set-frame-rate", "mpd-content-length",
        "ism-max-width", "ism-width", "ism-max-height", "ism-height",
        "ism-sampling-rate", "ism-channels", "ism-bits-per-sample",
        "ism-nal-unit-length-field"])
def test_guard_refuses_an_overlong_manifest_number(parse, xml):
    """yt-dlp does big-integer arithmetic and a str() per fragment on these
    numbers; past UINT64's 20 digits that costs ~1 ms a fragment at 4 000
    digits, so a short manifest buys minutes of CPU."""
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
        parse(xml)


def test_guard_refuses_a_long_bandwidth_before_its_per_fragment_expansion():
    """$Bandwidth$ puts the number into every fragment's address: at 4 000
    digits, 60 000 fragments cost ~18 s of CPU and ~350 MB before."""
    import time
    xml = _mpd(_adaptation(
        '<SegmentTemplate media="$Bandwidth$" timescale="1"><SegmentTimeline>'
        '<S t="0" d="1" r="59999"/></SegmentTimeline></SegmentTemplate>'
    ).replace('bandwidth="1"', 'bandwidth="1' + "0" * 3999 + '"'))
    t0 = time.monotonic()
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
        _parse_mpd(xml)
    assert time.monotonic() - t0 < 1.0


def test_guard_admits_manifest_numbers_at_uint64_width():
    """A live timeline's times in a fine timescale reach 17-20 digits."""
    top = "18446744073709551615"
    fmts, _subs = _parse_ism(_ism_numbers(t=f" {top} ", d=top, bitrate=top))
    assert len(fmts[0]["fragments"]) == 2
    _reset_budget()
    fmts, _subs = _parse_mpd(_timeline(t=top, d=top, timescale=top, start=top))
    assert len(fmts[0]["fragments"]) == 4
    _reset_budget()
    fmts, _subs = _parse_mpd(_timeline().replace(
        'bandwidth="1"',
        f'bandwidth="{top}" width="3840" height="2160" '
        'audioSamplingRate="44100 48000" frameRate="30000/1001"'))
    assert (fmts[0]["tbr"], fmts[0]["width"], fmts[0]["height"]) == (
        int(top) / 1000, 3840, 2160)
    assert fmts[0]["fps"] is None  # yt-dlp reads only a whole-number rate
    _reset_budget()
    fmts, _subs = _parse_mpd(_mpd_content_length(top))
    assert fmts[0]["filesize"] == int(top)
    _reset_budget()
    fmts, _subs = _parse_ism(_ism_level_number("SamplingRate", top))
    assert fmts[0]["asr"] == int(top)


def test_guard_parses_real_template_widths_unchanged():
    """Real templates (zero-padded widths, a signed query, the
    representation's own @id) expand exactly as yt-dlp does on its own, and a
    12 h ten-quality VOD's addresses fit the budget with room to spare."""
    query = "?" + "&".join(f"k{i}=" + "v" * 20 for i in range(12))
    assert len(query) >= 280
    tmpl = ('<SegmentTemplate media="seg-$RepresentationID$-$Number%09d$.m4s'
            + query.replace("&", "&amp;") + '" initialization='
            '"init-$RepresentationID$-$Bandwidth%05d$.mp4" timescale="1000" '
            'duration="2000"/>')
    fmts, _subs = _parse_mpd(_mpd(_adaptation(tmpl, reps=5)))
    assert len(fmts) == 5
    for i, f in enumerate(fmts):
        frags = f["fragments"]
        assert len(frags) == 7201  # the initialization segment, then 7 200
        assert frags[0]["path"] == f"init-r{i}-{i + 1:05d}.mp4"
        assert frags[1]["path"] == f"seg-r{i}-000000001.m4s{query}"
        assert frags[-1]["path"] == f"seg-r{i}-000007200.m4s{query}"
    # 216 000 segments of a ~200-byte address: about 43 MB, sized by the
    # guard alone (the expansion itself is the at-cap test's business).
    from yt_dlp.utils import compat_etree_fromstring
    url = "https://cdn.example/vod/" + "p" * 150 + "/s$Number%05d$.m4s"
    vod = _mpd(_adaptation(f'<SegmentTemplate media="{url}" timescale="1000" '
                           'duration="2000"/>', reps=10), duration="PT43200S")
    guard = _registered_guard()
    guard["_refuse_oversized_mpd"](compat_etree_fromstring(vod.encode()))
    assert 0 < guard["_DIALS"].tls.url_bytes < guard["_MAX_FRAGMENT_URL_BYTES"]


def test_guard_leaves_percent_escapes_outside_a_field_alone():
    """A `%` outside a `$...$` run (the escapes of a signed query) is text to
    yt-dlp, whatever digits follow it: it expands byte for byte."""
    fmts, _subs = _parse_mpd(_mpd(_adaptation(_live_template(
        "seg-$Number%05d$.m4s?sig=%3D1234567&amp;p=a%2Fb&amp;w=%2000000s")),
        duration="PT3S"))
    assert [f["path"] for f in fmts[0]["fragments"]] == [
        f"seg-{i:05d}.m4s?sig=%3D1234567&p=a%2Fb&w=%2000000s" for i in (1, 2, 3)]


@pytest.mark.parametrize("media,initialization", [
    ("$Number$" * 64, None),
    ("$RepresentationID$/$Number%05d$_$Bandwidth$.m4s", "$Bandwidth%05d$.mp4"),
    ("p$$q$Number$", None),
    ("s$%64s$Number$.m4s", None),
    ("$x%60r$$Number$", None),
    ("$%(Number)64d%(Number)64d$", None),
    ("$Number%5d%(Number)64d%(Number)64d$", None),
    ("$Number%s$Number$", None),
    ("a$b$Number$Number%(Number)64d$", None),
    ("s$Number$.m4s", "a$b$Number$Number%64s$"),
], ids=["64-fields", "real", "escaped-dollar", "keyless", "unknown-name",
        "keyed", "conversions-in-a-field", "field-before-a-bare-field",
        "shifted-field", "init-shifted-field"])
def test_guard_charges_every_live_template_conversion(media, initialization):
    """An admitted template, however odd, is charged at least the bytes
    yt-dlp builds from it: a live `%` conversion prints up to its width, the
    values dict or a value."""
    fmts, _subs = _parse_mpd(_mpd(_adaptation(
        _live_template(media, initialization)), duration="PT3S"))
    frags = fmts[0]["fragments"]
    assert len(frags) == (4 if initialization else 3)
    built = sum(len(f.get("path") or f.get("url")) for f in frags)
    assert built <= _registered_guard()["_DIALS"].tls.url_bytes


@pytest.mark.parametrize("media,initialization", [
    ("s$Number%0$100000d.m4s", None),
    ("s$Number%.$100000d.m4s", None),
    ("s$Number%5$.100000d.m4s", None),
    ("s$Number%5d%(Number$)0100000d.m4s", None),
    ("s$Number%5d%(Number)$0100000d.m4s", None),
    ("s$Number$.m4s", "i$Bandwidth%0$100000d"),
], ids=["flag", "dot", "width", "open-key", "closed-key", "initialization"])
def test_guard_refuses_a_conversion_the_text_after_the_field_completes(
        media, initialization):
    """yt-dlp drops a field's closing `$`, so a conversion the field leaves
    unfinished takes its width from the text after it: a 330-byte MPD with
    `$Number%0$10000000d` built 10 MB per fragment for a charge of ~30."""
    guard = _registered_guard()
    _reset_budget()
    with pytest.raises(yt_dlp.utils.ExtractorError,
                       match="segment addresses are too long"):
        _parse_mpd(_mpd(_adaptation(
            _live_template(media, initialization)), duration="PT3S"))
    assert guard["_DIALS"].tls.url_bytes == 0


def test_guard_admits_every_finished_conversion_of_a_field():
    """A field whose conversion ends in its type character before the closing
    `$` (the standard's `%0[width]d`, and the other printf types yt-dlp also
    prints) builds what it always did."""
    for spec, shown in (("", "7"), ("%05d", "00007"), ("%d", "7"),
                        ("%05u", "00007"), ("%x", "7"), ("%05X", "00007")):
        _reset_budget()
        fmts, _subs = _parse_mpd(_mpd(_adaptation(_live_template(
            f"seg-$Number{spec}$.m4s", f"init-$Bandwidth{spec}$.mp4")),
            duration="PT7S"))
        frags = fmts[0]["fragments"]
        assert frags[0]["path"] == "init-" + shown.replace("7", "1") + ".mp4"
        assert frags[-1]["path"] == f"seg-{shown}.m4s"


def test_guard_charges_real_template_fields_as_before():
    """yt-dlp's own fields cost their padded width or the value's digits,
    nothing more: a real manifest spends what it always did."""
    guard = _registered_guard()
    for template, extra in (("seg-$Number%05d$.m4s", 5),
                            ("$RepresentationID$/$Time$-$Bandwidth%09d$.m4s", 4 + 9),
                            ("a?b=%2F$$c$Number$", 4)):
        length = len(template.replace("$RepresentationID$", "r0"))
        assert guard["_template_bytes"](template, "r0", 4) == length + extra


@pytest.mark.parametrize("media", [
    "$Number$" * 7500,
    "$Number%" + "1a" * 30000 + "$",
], ids=["fields", "numbers-in-a-conversion"])
def test_guard_refuses_a_dense_template_on_the_first_representation(media):
    """A 60 KiB template of 7 500 fields, or of one field followed by 30 000
    numbers, cost the count itself 2-7 ms per Representation: ten seconds
    and more for 5 000 of them."""
    import time
    from yt_dlp.utils import compat_etree_fromstring
    xml = _mpd('<Period><AdaptationSet mimeType="video/mp4"><SegmentTemplate '
               f'media="{media}" timescale="1" duration="1"/>'
               + "".join(f'<Representation id="r{i}" bandwidth="{i + 1}"/>'
                         for i in range(5000))
               + "</AdaptationSet></Period>", duration="PT1S")
    doc = compat_etree_fromstring(xml.encode())
    refuse = _registered_guard()["_refuse_oversized_mpd"]
    t0 = time.process_time()
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
        refuse(doc)
    assert time.process_time() - t0 < 1.0


def test_guard_charges_fragment_url_bytes_across_manifests(monkeypatch):
    """Like the counts, the address bytes of every manifest one extraction
    parses come from one budget; a refused manifest spends none of it."""
    monkeypatch.setitem(_registered_guard(), "_MAX_FRAGMENT_URL_BYTES", 100_000)
    url = "a" * 990 + "/{start time}"
    sixty = _ism(_ism_url(url, '<c t="0" d="1" r="60"/>'))
    assert len(_parse_ism(sixty)[0][0]["fragments"]) == 60
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
        _parse_ism(sixty)
    thirty = _mpd(_adaptation('<SegmentTemplate media="' + "b" * 990
                              + '$Number$" timescale="1" duration="1"/>'),
                  duration="PT30S")
    assert len(_parse_mpd(thirty)[0][0]["fragments"]) == 30
    _inproc_guard().observe_dials(None)
    assert len(_parse_ism(sixty)[0][0]["fragments"]) == 60


def _parse_mpd_at(xml, base, *, positional):
    """_parse_mpd with the manifest's own base URL `base`, handed over the way
    _extract_mpd_formats does (positionally) or as a keyword."""
    from yt_dlp.extractor.common import InfoExtractor
    from yt_dlp.utils import compat_etree_fromstring
    udl.guard_self_check(force=True)
    doc = compat_etree_fromstring(xml.encode())
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        ie = InfoExtractor(ydl)
        if positional:
            return ie._merge_mpd_periods(
                ie._parse_mpd_periods(doc, None, base, base + "m.mpd"))
        return ie._parse_mpd_formats_and_subtitles(
            doc, mpd_base_url=base, mpd_url=base + "m.mpd")


@pytest.mark.parametrize("positional", [True, False], ids=["positional", "keyword"])
def test_guard_charges_the_manifest_url_for_each_representation(
        monkeypatch, positional):
    """yt-dlp prepends the manifest's own post-redirect URL (a Location of up
    to ~64 KB) to every representation's relative BaseURL: 300 bare
    representations under a 10 KB URL ask for 3 MB of base URLs."""
    guard = _registered_guard()
    xml = _mpd(_adaptation("<BaseURL>a</BaseURL>", reps=300))
    with monkeypatch.context() as m:
        m.setitem(guard, "_MAX_FRAGMENT_URL_BYTES", 1_000_000)
        with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
            _parse_mpd_at(xml, "http://cdn.example/" + "p" * 10_000 + "/",
                          positional=positional)
        # A real CDN URL leaves the same manifest as yt-dlp builds it.
        fmts, _subs = _parse_mpd_at(xml, "http://cdn.example/vod/",
                                    positional=positional)
        assert len(fmts) == 300
        assert fmts[0]["url"] == "http://cdn.example/vod/a"
    # The real budget: 20 000 representations under a 2 KB signed URL fit.
    _reset_budget()
    from yt_dlp.utils import compat_etree_fromstring
    big = _mpd("<Period>" + ('<AdaptationSet mimeType="audio/mp4">'
                             + '<Representation bandwidth="1"/>' * 1000
                             + "</AdaptationSet>") * 20 + "</Period>")
    guard["_refuse_oversized_mpd"](compat_etree_fromstring(big.encode()), 2049)
    assert guard["_DIALS"].tls.url_bytes < guard["_MAX_FRAGMENT_URL_BYTES"]


def _parse_ism_at(xml, ism_url, *, positional):
    """_parse_ism with the manifest's own URL `ism_url`, handed over the way
    GenericIE does (positionally) or as a keyword."""
    from yt_dlp.extractor.common import InfoExtractor
    from yt_dlp.utils import compat_etree_fromstring
    udl.guard_self_check(force=True)
    doc = compat_etree_fromstring(xml.encode())
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        ie = InfoExtractor(ydl)
        if positional:
            return ie._parse_ism_formats_and_subtitles(doc, ism_url)
        return ie._parse_ism_formats_and_subtitles(doc, ism_url=ism_url)


@pytest.mark.parametrize("positional", [True, False], ids=["positional", "keyword"])
def test_guard_charges_the_manifest_url_for_each_ism_fragment(
        monkeypatch, positional):
    """yt-dlp urljoins the manifest's own post-redirect URL (a Location of up
    to ~64 KB) onto the stream's Url, and every fragment copies the result:
    300 fragments under a 10 KB URL ask for 3 MB of addresses."""
    import time
    xml = _ism(_ism_url("a{start time}", '<c t="0" d="1" r="300"/>'))
    with monkeypatch.context() as m:
        m.setitem(_registered_guard(), "_MAX_FRAGMENT_URL_BYTES", 1_000_000)
        with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
            _parse_ism_at(xml, "http://cdn.example/" + "p" * 10_000 + "/x.ism/Manifest",
                          positional=positional)
        # A real CDN URL leaves the same manifest as yt-dlp builds it.
        fmts, _subs = _parse_ism_at(xml, "http://cdn.example/vod/x.ism/Manifest",
                                    positional=positional)
        assert len(fmts[0]["fragments"]) == 300
        assert fmts[0]["fragments"][0]["url"] == "http://cdn.example/vod/x.ism/a0"
    # The real budget: 20 000 fragments under a 64 KB redirect target are
    # refused before any is built; 1 000 under a 2 KB signed URL fit.
    _reset_budget()
    t0 = time.monotonic()
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
        _parse_ism_at(_ism(_ism_url("a{start time}", '<c t="0" d="1" r="20000"/>')),
                      "http://cdn.example/" + "p" * 65_000 + "/x.ism/Manifest",
                      positional=positional)
    assert time.monotonic() - t0 < 2.0
    fmts, _subs = _parse_ism_at(
        _ism(_ism_url("a{start time}", '<c t="0" d="1" r="1000"/>')),
        "http://cdn.example/" + "p" * 2_000 + "/x.ism/Manifest",
        positional=positional)
    assert len(fmts[0]["fragments"]) == 1000


def test_guard_is_not_installed_without_the_ism_bound(monkeypatch):
    from yt_dlp.extractor.common import InfoExtractor
    guard = _inproc_guard()
    assert guard.is_installed()
    with monkeypatch.context() as m:
        m.setattr(InfoExtractor, "_parse_ism_formats_and_subtitles",
                  lambda self, *a, **kw: ([], {}))
        assert not guard.is_installed()
    with monkeypatch.context() as m:  # a yt-dlp that renamed the method
        m.delattr(InfoExtractor, "_parse_ism_formats_and_subtitles")
        assert not guard.is_installed()
        guard.install()  # nothing to bound: still fails closed
        assert not guard.is_installed()
    assert guard.is_installed()


@pytest.mark.parametrize("over", [
    lambda cap: _ism_audio("", levels=cap + 1),
    lambda cap: _ism_audio('<c t="0" d="1"/>', levels=cap + 1),
    lambda cap: (_ism_audio('<c t="0" d="1"/>', levels=cap // 2 + 1)
                 + _ism_audio('<c t="0" d="1"/>', levels=cap // 2, kind="text")),
], ids=["no-chunk", "one-chunk", "audio+text"])
def test_guard_refuses_an_ism_manifest_listing_too_many_variants(monkeypatch, over):
    """Every QualityLevel is a format (or subtitle track) of its own, chunks
    or not, and yt-dlp re-walks the stream's chunks for each one: a manifest
    of one-chunk levels stays under the segment budget yet costs seconds per
    thousand. They are counted across StreamIndexes, like DASH variants."""
    cap = 40
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_REPRESENTATIONS", cap)
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_ism(_ism(over(cap)))
    fmts, _subs = _parse_ism(_ism(_ism_audio('<c t="0" d="1"/>', levels=cap)))
    assert len(fmts) == cap


@pytest.mark.parametrize("beside", [
    lambda: _ism(_ism_audio("<a/>" * 300 + '<c t="0" d="1"/>', levels=50)),
    lambda: _ism("<a/>" * 300 + _ism_audio('<c t="0" d="1"/>', levels=50)),
], ids=["in the stream", "under the root"])
def test_guard_charges_what_yt_dlp_rewalks_per_quality_level(monkeypatch, beside):
    """For every QualityLevel yt-dlp searches the children of its StreamIndex
    (for the chunks) and of the manifest (for a Protection element): 300
    levels under a root of 1.9 million empty elements took 4 s of CPU with
    every count within its cap."""
    guard = _registered_guard()
    monkeypatch.setitem(guard, "_MAX_MANIFEST_WALK", 10_000)
    _reset_budget()
    with pytest.raises(yt_dlp.utils.ExtractorError, match="manifest is too large"):
        _parse_ism(beside())
    assert guard["_DIALS"].tls.walk == 0
    plain = _ism(_ism_audio('<c t="0" d="1"/>', levels=50))
    assert len(_parse_ism(plain)[0]) == 50
    # Per level: the root's one child and the stream's 51.
    assert guard["_DIALS"].tls.walk == 50 * (1 + 51)
    # DASH and Smooth Streaming manifests of one extraction share the budget.
    with pytest.raises(yt_dlp.utils.ExtractorError, match="manifest is too large"):
        for _ in range(4):
            _parse_ism(plain)


def test_guard_charges_an_ism_chunk_that_builds_no_fragment(monkeypatch):
    """A <c r="-1"> expands to nothing, but yt-dlp still reads it once for
    every QualityLevel of its stream: levels x chunks is the work, so each
    chunk costs at least one per level."""
    cap = 1000
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_SEGMENTS", cap)
    empty = '<c t="0" d="1" r="-1"/>'
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many segments"):
        _parse_ism(_ism(_ism_audio(empty * (cap // 10 + 1), levels=10)))
    fmts, _subs = _parse_ism(_ism(_ism_audio(empty * (cap // 10), levels=10)))
    assert len(fmts) == 10 and all(f["fragments"] == [] for f in fmts)


# ---- HLS: a master playlist becomes one format per variant URI line (and
# per EXT-X-MEDIA rendition), and every format is then sorted, selected and
# sanitized in pure Python — ~0.5 ms each. The guard counts them before the
# parse.

def _parse_m3u8(doc):
    from yt_dlp.extractor.common import InfoExtractor
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        ie = InfoExtractor(ydl)
        return ie._parse_m3u8_formats_and_subtitles(
            doc, "http://cdn.example/master.m3u8", ext="mp4")


def _hls_cap():
    return _registered_guard()["_MAX_M3U8_FORMATS"]


_AUDIO_MEDIA = '#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="a",NAME="n{i}",URI="a{i}.m3u8"'


@pytest.mark.parametrize("make", [
    lambda n: "#EXTM3U\n" + "u\n" * n,  # a bare URI line is a variant too
    lambda n: "#EXTM3U\n" + "#EXT-X-STREAM-INF:BANDWIDTH=1\nv.m3u8\n" * n,
    lambda n: "#EXTM3U\n" + "".join(_AUDIO_MEDIA.format(i=i) + "\n" for i in range(n)),
    lambda n: ("#EXTM3U\n" + "".join(_AUDIO_MEDIA.format(i=i) + "\n" for i in range(n // 2))
               + "v.m3u8\n" * (n - n // 2)),
    # PROGRESSIVE-URI adds a second (http) format per variant.
    lambda n: "#EXTM3U\n" + ('#EXT-X-STREAM-INF:BANDWIDTH=1,PROGRESSIVE-URI="p.mp4"\n'
                             "v.m3u8\n") * (n // 2 + 1),
    # Every line break str.splitlines() honours, as yt-dlp splits.
    lambda n: "#EXTM3U\r" + "u\r" * n,
    lambda n: "#EXTM3U\u2028" + "u\u2028" * n,
], ids=["bare", "stream-inf", "media", "mixed", "progressive", "cr", "u2028"])
def test_guard_refuses_an_hls_master_listing_too_many_formats(make):
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_m3u8(make(_hls_cap() + 1))


def test_guard_parses_an_ordinary_hls_master_unchanged():
    master = ("#EXTM3U\n"
              '#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="aud",NAME="en",LANGUAGE="en",'
              'DEFAULT=YES,URI="en.m3u8"\n'
              '#EXT-X-MEDIA:TYPE=SUBTITLES,GROUP-ID="sub",NAME="de",LANGUAGE="de",'
              'URI="de.m3u8"\n'
              + "".join(f'#EXT-X-STREAM-INF:BANDWIDTH={b}000,RESOLUTION={w}x{h},'
                        'CODECS="avc1.4d401f,mp4a.40.2",AUDIO="aud"\n'
                        f"v{h}.m3u8\n"
                        for b, w, h in ((800, 640, 360), (2400, 1280, 720),
                                        (5000, 1920, 1080))))
    fmts, subs = _parse_m3u8(master)
    assert [f["url"].rsplit("/", 1)[1] for f in fmts] == [
        "en.m3u8", "v360.m3u8", "v720.m3u8", "v1080.m3u8"]
    assert list(subs) == ["de"]
    cap = _hls_cap()
    assert cap >= 1000
    assert len(_parse_m3u8("#EXTM3U\n# comment\n\n" + "u\n" * cap)[0]) == cap
    # A media playlist is returned whole as one format, however many
    # segments it lists.
    media = ("#EXTM3U\n#EXT-X-TARGETDURATION:2\n"
             + "#EXTINF:2,\ns.ts\n" * (cap * 3) + "#EXT-X-ENDLIST\n")
    assert len(_parse_m3u8(media)[0]) == 1


def test_guard_charges_hls_master_variants_across_manifests(monkeypatch):
    """Each master within its own cap is not enough: a page or a set-level
    F4M can hand one extraction many of them. Their variants spend from the
    budget DASH and Smooth Streaming spend from; a refused master spends
    nothing, and the next extraction starts afresh."""
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_REPRESENTATIONS", 10)
    six = "#EXTM3U\n" + "u\n" * 6
    assert len(_parse_m3u8(six)[0]) == 6
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_m3u8(six)
    assert len(_parse_m3u8("#EXTM3U\n" + "u\n" * 4)[0]) == 4
    # DASH representations share the budget too.
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_mpd(_mpd(_adaptation("<BaseURL>a.m4a</BaseURL>", reps=1)))
    # A media playlist is one format whatever it lists, and spends nothing.
    media = "#EXTM3U\n#EXT-X-TARGETDURATION:2\n" + "#EXTINF:2,\ns.ts\n" * 50
    assert len(_parse_m3u8(media)[0]) == 1
    _inproc_guard().observe_dials(None)
    assert len(_parse_m3u8(six)[0]) == 6


def _parse_m3u8_at(doc, m3u8_url, *, positional):
    """_parse_m3u8 with the master's own URL `m3u8_url`, positionally (as
    _extract_m3u8_formats_and_subtitles hands it over) or as a keyword."""
    from yt_dlp.extractor.common import InfoExtractor
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        ie = InfoExtractor(ydl)
        if positional:
            return ie._parse_m3u8_formats_and_subtitles(doc, m3u8_url, "mp4")
        return ie._parse_m3u8_formats_and_subtitles(doc, m3u8_url=m3u8_url, ext="mp4")


@pytest.mark.parametrize("positional", [True, False], ids=["positional", "keyword"])
def test_guard_charges_hls_master_addresses(monkeypatch, positional):
    """yt-dlp builds every variant's address as a fresh urljoin(master URL,
    line), so a ~64 KB redirect target costs 2 000 x 64 KB per master."""
    master = "#EXTM3U\n" + "".join(f"v{i}.m3u8\n" for i in range(400))
    long_url = "http://cdn.example/" + "p" * 1000 + "/master.m3u8"
    with monkeypatch.context() as m:
        m.setitem(_registered_guard(), "_MAX_FRAGMENT_URL_BYTES", 1_000_000)
        assert len(_parse_m3u8_at(master, long_url, positional=positional)[0]) == 400
        assert len(_parse_m3u8_at(master, long_url, positional=positional)[0]) == 400
        with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
            _parse_m3u8_at(master, long_url, positional=positional)


def test_guard_admits_real_hls_masters_and_hds_sets():
    """Sixteen masters of 300 variants and renditions under a 300-byte signed
    URL, plus a set-level HDS manifest naming a few bitrates, spend a small
    part of one extraction's budget."""
    query = "?" + "&".join(f"k{i}=" + "v" * 20 for i in range(12))
    url = "https://cdn.example/vod/master.m3u8" + query
    assert len(url) >= 300
    master = ("#EXTM3U\n"
              + "".join(f'#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="a",NAME="n{i}",'
                        f'URI="a{i}.m3u8"\n' for i in range(50))
              + "".join(f'#EXT-X-STREAM-INF:BANDWIDTH={i + 1},AUDIO="a"\n'
                        f"v{i}.m3u8{query}\n" for i in range(250)))
    for _ in range(16):
        assert len(_parse_m3u8_at(master, url, positional=True)[0]) == 300
    fmts = _parse_f4m(_f4m(f"<baseURL>{url.rsplit('/', 1)[0]}</baseURL>"
                           + _f4m_media(5), bootstrap=False))
    assert len(fmts) == 5
    guard = _registered_guard()
    assert guard["_DIALS"].tls.variants == 16 * 300
    assert 0 < guard["_DIALS"].tls.url_bytes < guard["_MAX_FRAGMENT_URL_BYTES"] // 50


def test_guard_is_not_installed_without_the_hls_bound(monkeypatch):
    from yt_dlp.extractor.common import InfoExtractor
    guard = _inproc_guard()
    assert guard.is_installed()
    with monkeypatch.context() as m:
        m.setattr(InfoExtractor, "_parse_m3u8_formats_and_subtitles",
                  lambda self, *a, **kw: ([], {}))
        assert not guard.is_installed()
    with monkeypatch.context() as m:  # a yt-dlp that renamed the method
        m.delattr(InfoExtractor, "_parse_m3u8_formats_and_subtitles")
        assert not guard.is_installed()
        guard.install()  # nothing to bound: still fails closed
        assert not guard.is_installed()
    assert guard.is_installed()


@pytest.mark.parametrize("brk", ["\n", "\r", "\r\n", " "],
                         ids=["lf", "cr", "crlf", "u2028"])
def test_guard_refuses_an_hls_master_of_too_many_lines(monkeypatch, brk):
    """yt-dlp splits a master into one str per line, twice, before it looks
    at a single one: 64 MiB of comment lines lists no format at all yet costs
    seconds and over a gigabyte. Lines are counted where str.splitlines()
    splits, a CRLF once."""
    cap = 1000
    monkeypatch.setitem(_registered_guard(), "_MAX_M3U8_MASTER_LINES", cap)
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many lines"):
        _parse_m3u8("#EXTM3U" + brk + ("#a" + brk) * cap)
    assert _parse_m3u8("#EXTM3U" + brk + ("#a" + brk) * (cap - 1)) == ([], {})


def test_guard_refuses_a_large_comment_only_hls_master_quickly():
    import time
    guard = _registered_guard()
    # Far above a master at the format cap (two lines per variant plus a
    # few tags), so no playlist the format bound admits is refused here.
    assert guard["_MAX_M3U8_MASTER_LINES"] >= 50 * guard["_MAX_M3U8_FORMATS"]
    doc = "#EXTM3U\n" + "#a\n" * (5 * 1024 * 1024 // 3)
    t0 = time.monotonic()
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many lines"):
        _parse_m3u8(doc)
    assert time.monotonic() - t0 < 1.0


# yt-dlp runs backtracking regexes over single lines of a master, each
# quadratic in the line and holding the GIL throughout: the attribute parse
# of an EXT-X-MEDIA / EXT-X-STREAM-INF line, the size search in its
# RESOLUTION, an `audio...=` search over each variant's joined address, and
# the DRM search from every key tag to the end of its line.
_SLOW_PLAYLISTS = {
    "resolution": lambda: ("#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1,RESOLUTION="
                           + "1" * 20_000 + "\nv.m3u8\n"),
    "stream-inf": lambda: "#EXTM3U\n#EXT-X-STREAM-INF:" + "A" * 20_000 + "\nv.m3u8\n",
    "media": lambda: ("#EXTM3U\n#EXT-X-MEDIA:" + "A=" * 10_000 + '"\n'
                      "#EXT-X-STREAM-INF:BANDWIDTH=1\nv.m3u8\n"),
    # No line is long and none lists a format: the squares add up.
    "many-short": lambda: ("#EXTM3U\n"
                           + ("#EXT-X-STREAM-INF:" + "A" * 238 + "\n") * 60_000),
    "audio-uri": lambda: ("#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\n"
                          + "audio" * 20_000 + "\n"),
    "key-master": lambda: ("#EXTM3U\n#" + "#EXT-X-KEY:" * 3000 + "\n"
                           "#EXT-X-STREAM-INF:BANDWIDTH=1\nv.m3u8\n"),
    # The DRM search runs before yt-dlp tells a media playlist apart.
    "key-media": lambda: (_HLS_HEAD + "#EXT-X-KEY:" * 3000 + "\na.ts\n"),
}


@pytest.mark.parametrize("shape", sorted(_SLOW_PLAYLISTS))
def test_guard_refuses_hls_lines_that_cost_quadratic_regex_time(shape):
    import time
    doc = _SLOW_PLAYLISTS[shape]()
    _parse_m3u8("#EXTM3U\n")  # the guard is loaded: not part of the timing
    t0 = time.monotonic()
    with pytest.raises(yt_dlp.utils.ExtractorError,
                       match="line in the playlist is too long"):
        _parse_m3u8(doc)
    assert time.monotonic() - t0 < 0.5


def test_guard_charges_the_audio_scan_against_the_master_address():
    """The `audio` search runs over urljoin(master URL, line): twenty
    ordinary variants under a redirect-sized address made of the word cost
    over a minute."""
    import time
    master = "#EXTM3U\n" + "#EXT-X-STREAM-INF:BANDWIDTH=1\nv.m3u8\n" * 20
    url = "http://cdn.example/" + "audio" * 12_000 + "/master.m3u8"
    _parse_m3u8("#EXTM3U\n")
    t0 = time.monotonic()
    with pytest.raises(yt_dlp.utils.ExtractorError,
                       match="line in the playlist is too long"):
        _parse_m3u8_at(master, url, positional=True)
    assert time.monotonic() - t0 < 0.5
    # The same redirect-sized address without the word costs nothing extra.
    _reset_budget()
    plain = "http://cdn.example/" + "p" * 60_000 + "/master.m3u8"
    assert len(_parse_m3u8_at(master, plain, positional=True)[0]) == 20


def test_guard_admits_real_hls_lines():
    """A master of 100 renditions and 200 variants with a 1.5 kB token on
    every line under a 2 kB address spends about half the line budget, and a
    media playlist that rotates its key on each of 20 000 segments (one tag a
    line) is not taken for the many-keys-on-one-line shape."""
    guard = _registered_guard()
    token = "?token=" + "x" * 1500
    master = ("#EXTM3U\n" + "".join(
        f'#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",LANGUAGE="l{i}",NAME="Audio {i}",'
        f'AUTOSELECT=YES,DEFAULT=NO,CHANNELS="2",'
        f'URI="https://cdn.example/audio/a{i}/audio_{i}.m3u8{token}"\n'
        for i in range(100)) + "".join(
        f'#EXT-X-STREAM-INF:BANDWIDTH={i + 1}000,AVERAGE-BANDWIDTH={i + 1}000,'
        'CODECS="avc1.640028,mp4a.40.2",RESOLUTION=1920x1080,FRAME-RATE=25.000,'
        'AUDIO="audio",CLOSED-CAPTIONS=NONE\n'
        f"https://cdn.example/video/audio-video/v{i}.m3u8{token}\n"
        for i in range(200)))
    url = "https://cdn.example/audio/master.m3u8?" + "t" * 2000
    assert len(_parse_m3u8_at(master, url, positional=True)[0]) == 300
    assert 0 < guard["_DIALS"].tls.line_work < guard["_MAX_M3U8_LINE_WORK"] * 0.6
    media = _HLS_HEAD + "".join(
        f'#EXT-X-KEY:METHOD=AES-128,URI="https://k.example/key{i}",IV=0x{i:032x}\n'
        f"#EXTINF:6,\ns{i}.ts\n" for i in range(20_000))
    assert media.count("KEY:") * len(media) > guard["_MAX_M3U8_KEY_WORK"]
    assert len(_parse_m3u8(media)[0]) == 1
    # A tag inside a line is text to yt-dlp, whatever follows it.
    _reset_budget()
    note = "## not a tag: #EXT-X-STREAM-INF:" + "A" * 5000 + "\n"
    assert len(_parse_m3u8("#EXTM3U\n" + note + "v.m3u8\n")[0]) == 1
    assert guard["_DIALS"].tls.line_work < 1000


def test_guard_charges_hls_line_work_across_manifests(monkeypatch):
    """Like the counts, the line work of every master one extraction parses
    comes from one budget; a refused master spends none of it."""
    guard = _registered_guard()
    master = ("#EXTM3U\n" + ("#EXT-X-STREAM-INF:BANDWIDTH=1," + "A" * 71
                             + "\nv.m3u8\n") * 10)
    assert len(_parse_m3u8(master)[0]) == 10
    spent = guard["_DIALS"].tls.line_work
    assert spent >= 10 * 100 * 100
    monkeypatch.setitem(guard, "_MAX_M3U8_LINE_WORK", 2 * spent - 1)
    with pytest.raises(yt_dlp.utils.ExtractorError,
                       match="line in the playlist is too long"):
        _parse_m3u8(master)
    assert guard["_DIALS"].tls.line_work == spent
    _inproc_guard().observe_dials(None)
    assert len(_parse_m3u8(master)[0]) == 10


# ---- HLS media playlists: the download child's HlsFD turns every segment
# line into a fragment dict before it fetches any, bounded only by the 64 MiB
# body cap (~33 M lines, ~12 GB). The guard counts them, against the DASH
# segment budget, when HlsFD checks the manifest it is about to expand.

def _hls_download(monkeypatch, tmp_path, doc, *,
                  url="https://public.example/a.m3u8", params=None):
    """HlsFD.real_download on `doc` (None: HlsFD fetches `url` itself) with
    the fragment fetch stubbed out: the fragment counts it would have
    fetched, or the refusal."""
    udl.guard_self_check(force=True)
    built = []
    monkeypatch.setattr(
        hls.HlsFD, "download_and_append_fragments",
        lambda self, ctx, fragments, info_dict, **kw: built.append(len(fragments)) or True)
    info = {"url": url, "id": "x", "ext": "mp4", "http_headers": {}}
    if doc is not None:
        info["hls_media_playlist_data"] = doc
    with yt_dlp.YoutubeDL({"quiet": True, **(params or {})}) as ydl:
        fd = hls.HlsFD(ydl, ydl.params)
        fd.real_download(str(tmp_path / "media.mp4"), info)
    return built


_HLS_HEAD = "#EXTM3U\n#EXT-X-TARGETDURATION:1\n"
_HLS_MAP = '#EXT-X-MAP:URI="init.mp4"\n'


@pytest.mark.parametrize("make", [
    lambda n: _HLS_HEAD + "a\n" * n,
    lambda n: _HLS_HEAD + _HLS_MAP + "a\n" * (n - 1),  # the init segment too
    lambda n: _HLS_HEAD + " \ta\n" * n,  # HlsFD strips each line first
    lambda n: _HLS_HEAD.replace("\n", "\r") + "a\r" * n,
    lambda n: _HLS_HEAD + "a " * n,
], ids=["bare", "map", "indented", "cr", "u2028"])
def test_guard_refuses_an_hls_media_playlist_listing_too_many_segments(
        monkeypatch, tmp_path, make):
    cap = 1000
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_SEGMENTS", cap)
    built = []
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many segments"):
        built = _hls_download(monkeypatch, tmp_path, make(cap + 1))
    assert built == []


def test_guard_downloads_an_ordinary_hls_media_playlist_unchanged(
        monkeypatch, tmp_path):
    """Comments and blank lines are no fragments; right at the cap every
    segment is still built."""
    assert _registered_guard()["_MAX_MPD_SEGMENTS"] >= 86_400  # 24 h at 1 s
    cap = 1000
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_SEGMENTS", cap)
    doc = (_HLS_HEAD + _HLS_MAP + "#EXTINF:1,\n a.ts\n\n # note\n" * (cap - 1)
           + "#EXT-X-ENDLIST\n")
    assert _hls_download(monkeypatch, tmp_path, doc) == [cap]


# HlsFD joins every segment line (and every #EXT-X-KEY URI) onto the
# playlist's own URL before it fetches anything, so a 2-byte line costs that
# URL's length: ~64 KB through a redirect, more from a master's variant line.

_LONG_HLS_URL = "https://public.example/" + "p" * 10_000 + "/a.m3u8"


@pytest.mark.parametrize("doc", [
    _HLS_HEAD + "s\n" * 200,
    # 60 segments alone fit; the key URI joined before each one does not.
    _HLS_HEAD + '#EXT-X-KEY:METHOD=AES-128,URI="k"\ns\n' * 60,
], ids=["segments", "keys"])
def test_guard_refuses_hls_segment_addresses_past_the_url_budget(
        monkeypatch, tmp_path, doc):
    monkeypatch.setitem(_registered_guard(), "_MAX_FRAGMENT_URL_BYTES", 1_000_000)
    built = []
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
        built = _hls_download(monkeypatch, tmp_path, doc, url=_LONG_HLS_URL)
    assert built == []


def test_guard_downloads_hls_playlists_under_the_url_budget_unchanged(
        monkeypatch, tmp_path):
    """A 12 h VOD at 2 s segments under a 300-byte signed URL takes the
    line-count fast path; a playlist whose every line joined would not fit
    but whose segments do is counted, and still built."""
    url = "https://cdn.example/vod/" + "q" * 270 + "/a.m3u8"
    vod = _HLS_HEAD + "#EXTINF:2,\nseg.ts\n" * 21_600 + "#EXT-X-ENDLIST\n"
    assert _hls_download(monkeypatch, tmp_path, vod, url=url) == [21_600]
    monkeypatch.setitem(_registered_guard(), "_MAX_FRAGMENT_URL_BYTES", 1_000_000)
    doc = _HLS_HEAD + "#EXTINF:1,\n#a\n\ns\n" * 90 + "#EXT-X-ENDLIST\n"
    assert _hls_download(monkeypatch, tmp_path, doc, url=_LONG_HLS_URL) == [90]


class _RedirectedPlaylist(_QuietHandler):
    """A short playlist URL that redirects to a 20 KB one serving 100
    segment lines (fetched through the operator proxy)."""

    long_path = "/" + "r" * 20_000 + "/a.m3u8"

    def do_GET(self):
        if not self.path.endswith(type(self).long_path):
            self.send_response(302)
            self.send_header("Location",
                             f"http://{_PUBLIC_LITERAL}{type(self).long_path}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = (_HLS_HEAD + "s\n" * 100).encode()
        self._ok_headers("application/vnd.apple.mpegurl", len(body))
        self.wfile.write(body)


def test_guard_charges_hls_lines_against_the_redirected_playlist_url(
        monkeypatch, tmp_path):
    """HlsFD joins onto the URL its own fetch ended at, which only the
    response knows: the 2 MB of addresses are refused though the info dict
    names a short URL."""
    for var in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.upper(), raising=False)
    monkeypatch.setitem(_registered_guard(), "_MAX_FRAGMENT_URL_BYTES", 1_000_000)
    srv = _serve(_RedirectedPlaylist, "127.0.0.1")
    with _stopping(srv):
        proxy = {"proxy": f"http://127.0.0.1:{srv.server_port}"}
        short = f"http://{_PUBLIC_LITERAL}/a.m3u8"
        built = []
        with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
            built = _hls_download(monkeypatch, tmp_path, None, url=short,
                                  params=proxy)
        assert built == []
        # The same playlist behind an ordinary redirect is built.
        monkeypatch.setattr(_RedirectedPlaylist, "long_path", "/vod/a.m3u8")
        assert _hls_download(monkeypatch, tmp_path, None, url=short,
                             params=proxy) == [100]


@pytest.mark.parametrize("line", [
    "#EXT-X-MEDIA-SEQUENCE:" + "9" * 4299,
    "#EXT-X-MEDIA-SEQUENCE:" + "9" * 21,
    " \t#EXT-X-MEDIA-SEQUENCE:" + "9" * 21,
    "#EXT-X-BYTERANGE:1@" + "9" * 4299,
    "#EXT-X-BYTERANGE:" + "9" * 21 + "@0",
    "#EXT-X-BYTERANGE:" + "9" * 21,
], ids=["sequence", "sequence-21", "indented", "offset", "length", "bare-length"])
def test_guard_refuses_an_hls_media_playlist_with_an_overlong_number(
        monkeypatch, tmp_path, line):
    """HlsFD int()s the media sequence and both sides of a byte range and
    keeps a number built from them per fragment: one 4 299-digit number makes
    every fragment ~1.8 kB heavier."""
    built = []
    with pytest.raises(yt_dlp.utils.ExtractorError,
                       match="number in the stream manifest is too long"):
        built = _hls_download(monkeypatch, tmp_path,
                              _HLS_HEAD + line + "\na\n" * 3)
    assert built == []


@pytest.mark.parametrize("tag", ["#EXT-X-BYTERANGE: ", "#EXT-X-BYTERANGE:@"])
def test_guard_searches_a_line_of_hls_number_tags_once(tag):
    """The search for an overlong number looks from each tag to the next
    "#", not to the end of its line: 8 000 tags on one line took 13 s when
    every one of them scanned the rest."""
    import time
    refuse = _registered_guard()["_refuse_oversized_hls_media"]
    doc = _HLS_HEAD + tag * 8000 + "\na.ts\n"
    t0 = time.process_time()
    refuse(doc, 100)
    assert time.process_time() - t0 < 0.5


def test_guard_downloads_hls_playlists_with_64_bit_numbers_unchanged(
        monkeypatch, tmp_path):
    """Numbers at UINT64's 20 digits pass, as do the long things next to them
    that are no number: a segment name on the line after a byte range, a
    title, a date's fraction."""
    big = "9" * 20
    doc = (_HLS_HEAD + f"#EXT-X-MEDIA-SEQUENCE:{big} \r\n"
           + '#EXT-X-MAP:URI="init.mp4",BYTERANGE="720@0"\n'
           + f"#EXT-X-PROGRAM-DATE-TIME:2026-01-01T00:00:00.{'0' * 30}Z\n"
           + f"#EXTINF:6,{'t' * 500}\n#EXT-X-BYTERANGE:{big}@{big}\n"
           + "https://cdn.example/" + "s" * 300 + ".mp4\n"
           + "#EXTINF:6,\n#EXT-X-BYTERANGE: 1048576 @ 720\nb.mp4\n")
    assert _hls_download(monkeypatch, tmp_path, doc) == [3]


def test_guard_is_not_installed_without_the_hls_fragment_bound(monkeypatch):
    guard = _inproc_guard()
    assert guard.is_installed()
    with monkeypatch.context() as m:
        m.setattr(hls.HlsFD, "can_download",
                  classmethod(lambda cls, manifest, info_dict, allow=False: True))
        assert not guard.is_installed()
    with monkeypatch.context() as m:  # a yt-dlp that renamed the method
        m.delattr(hls.HlsFD, "can_download")
        assert not guard.is_installed()
        guard.install()  # nothing to bound: still fails closed
        assert not guard.is_installed()
    assert guard.is_installed()


# ---- One extraction, one budget: GenericIE parses one MPD per Video.js
# `src` into a single info dict, so a per-manifest budget let N manifests
# expand N x 1 M segments. The manifests one extraction parses spend from one
# running total per thread, which observe_dials() (probe()'s start and end)
# resets.

def test_guard_charges_segments_across_manifests_in_one_extraction(monkeypatch):
    cap = 1000
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_SEGMENTS", cap)
    six_hundred = _mpd(_adaptation(_template(timescale="1", duration="1")),
                       duration="PT600S")
    assert len(_parse_mpd(six_hundred)[0][0]["fragments"]) == 600
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many segments"):
        _parse_mpd(six_hundred)
    # A refused manifest spends nothing: the rest of the budget still parses.
    four_hundred = _mpd(_adaptation(_template(timescale="1", duration="1")),
                        duration="PT400S")
    assert len(_parse_mpd(four_hundred)[0][0]["fragments"]) == 400
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many segments"):
        _parse_ism(_ism(_ism_audio('<c t="0" d="1"/>')))
    # probe() calls observe_dials() as its extraction starts and ends: the
    # pool thread's next extraction starts from a full budget.
    guard = _inproc_guard()
    guard.observe_dials(None)
    assert len(_parse_mpd(six_hundred)[0][0]["fragments"]) == 600
    guard.observe_dials(lambda sock: None, deadline=None)
    try:
        assert len(_parse_mpd(six_hundred)[0][0]["fragments"]) == 600
    finally:
        guard.observe_dials(None)


def test_guard_charges_ism_fragments_across_manifests_in_one_extraction(
        monkeypatch):
    cap = 1000
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_SEGMENTS", cap)
    six_hundred = _ism(_ism_audio('<c t="0" d="1" r="600"/>'))
    assert len(_parse_ism(six_hundred)[0][0]["fragments"]) == 600
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many segments"):
        _parse_ism(six_hundred)
    _inproc_guard().observe_dials(None)
    assert len(_parse_ism(six_hundred)[0][0]["fragments"]) == 600


def test_guard_charges_variants_across_manifests_in_one_extraction(monkeypatch):
    cap = 10
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_REPRESENTATIONS", cap)
    six = _mpd(_adaptation('<BaseURL>a.m4a</BaseURL>', reps=6))
    assert len(_parse_mpd(six)[0]) == 6
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_mpd(six)
    # DASH representations and Smooth Streaming levels share the budget.
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_ism(_ism(_ism_audio('<c t="0" d="1"/>', levels=6)))
    _inproc_guard().observe_dials(None)
    assert len(_parse_ism(_ism(_ism_audio('<c t="0" d="1"/>', levels=6)))[0]) == 6


# ---- HDS (F4M): _parse_f4m_formats makes one format per <media> node, and a
# set-level manifest (no bootstrapInfo) fetches every node that names another
# .f4m / .m3u8 — recursively. Each format then costs ~0.5 ms of format pass.
# The guard counts nodes and nested references before the parse, bounds the
# nesting depth and the formats nested children add up to.

_F4M_NS = "http://ns.adobe.com/f4m/1.0"
_F4M_BOOT = '<bootstrapInfo profile="named" id="b">AAAA</bootstrapInfo>'


def _f4m(media, *, bootstrap=True):
    return (f'<?xml version="1.0"?><manifest xmlns="{_F4M_NS}"><id>x</id>'
            f'{_F4M_BOOT if bootstrap else ""}{media}</manifest>')


def _f4m_media(n, url="v{i}"):
    return "".join(f'<media url="{url.format(i=i)}" bitrate="{i + 1}" '
                   'bootstrapInfoId="b"/>' for i in range(n))


def _parse_f4m(xml, **stubs):
    """InfoExtractor._parse_f4m_formats on `xml`, with the named methods of
    the extractor replaced by `stubs` (the nested-manifest fetches)."""
    from yt_dlp.extractor.common import InfoExtractor
    from yt_dlp.utils import compat_etree_fromstring
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        ie = InfoExtractor(ydl)
        for name, fn in stubs.items():
            setattr(ie, name, types.MethodType(fn, ie))
        return ie._parse_f4m_formats(compat_etree_fromstring(xml.encode()),
                                     "http://cdn.example/m.f4m", "x")


def test_guard_refuses_an_f4m_manifest_listing_too_many_formats():
    import time
    t0 = time.monotonic()
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_f4m(_f4m(_f4m_media(_hls_cap() + 1)))
    assert time.monotonic() - t0 < 1.0


def test_guard_refuses_an_f4m_set_manifest_nesting_too_many_manifests():
    fetched = []

    def _m3u8(self, url, video_id, *a, **kw):
        fetched.append(url)
        return [{"url": url, "format_id": url}]
    nested = _registered_guard()["_MAX_F4M_NESTED"]
    assert nested >= 8
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_f4m(_f4m(_f4m_media(nested + 1, url="c{i}.m3u8"), bootstrap=False),
                   _extract_m3u8_formats=_m3u8)
    assert fetched == []
    # Under the reference cap, the formats the children add up to are
    # bounded before the format pass sees them.
    cap = _hls_cap()

    def _big(self, url, video_id, *a, **kw):
        return [{"url": f"{url}/{i}", "format_id": f"{url}{i}"}
                for i in range(cap // 2 + 1)]
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_f4m(_f4m(_f4m_media(2, url="c{i}.m3u8"), bootstrap=False),
                   _extract_m3u8_formats=_big)


def test_guard_refuses_a_self_nesting_f4m_manifest():
    from yt_dlp.utils import compat_etree_fromstring
    set_level = _f4m(_f4m_media(1, url="again.f4m"), bootstrap=False)
    depth = []

    def _again(self, url, video_id, *a, **kw):
        depth.append(url)
        if len(depth) >= 5:  # yt-dlp itself recurses until RecursionError
            return []
        return self._parse_f4m_formats(compat_etree_fromstring(set_level.encode()),
                                       url, video_id)
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_f4m(set_level, _extract_f4m_formats=_again)
    assert len(depth) == _registered_guard()["_MAX_F4M_DEPTH"]


def test_guard_counts_a_nested_reference_in_either_f4m_attribute():
    # yt-dlp reads @href only in a 2.0 manifest and @url otherwise, so a 1.0
    # node whose href names a plain file still fetches the manifest in url.
    fetched = []

    def _child(self, url, video_id, *a, **kw):
        fetched.append(url)
        return []
    nested = _registered_guard()["_MAX_F4M_NESTED"]
    media = "".join(f'<media href="a{i}.flv" url="n{i}.f4m" bitrate="{i + 1}"/>'
                    for i in range(nested + 1))
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_f4m(_f4m(media, bootstrap=False), _extract_f4m_formats=_child)
    assert fetched == []


_JOINED_NESTED = 17  # one past _MAX_F4M_NESTED


@pytest.mark.parametrize("manifest_url,media", [
    ("http://cdn.example/m.f4m", "<baseURL>http://a.example/b.f4m?</baseURL>"
     + _f4m_media(_JOINED_NESTED, url="x{i}")),
    ("http://cdn.example/m.f4m", "<baseURL>http://a.example/b.f4m</baseURL>"
     + _f4m_media(_JOINED_NESTED, url="/")),
    ("http://cdn.example/m.f4m", "<baseURL>http://a.example/b.m3u8?</baseURL>"
     + _f4m_media(_JOINED_NESTED, url="x{i}")),
    ("http://a.example/p.f4m?x/set", _f4m_media(_JOINED_NESTED, url="x{i}")),
    # A trailing run of slashes longer than any fixed window: the head keeps
    # ".f4m" only far behind its end, and every "/" node after it is nested.
    ("http://cdn.example/m.f4m", _f4m_media(1, url="x.f4m" + "/" * 100)
     + _f4m_media(_JOINED_NESTED - 1, url="/")),
], ids=["query-in-base", "slash-under-base", "m3u8-query-in-base",
        "query-in-dir", "long-slash-run"])
def test_guard_counts_f4m_nested_on_the_joined_address(manifest_url, media):
    """yt-dlp decides a nested fetch on the address it joins, not on the
    node's own attribute: a "?" in the head hides the node's url from
    determine_ext, and a "/" node under ".../b.f4m" ends in "b.f4m//"."""
    from yt_dlp.extractor.common import InfoExtractor
    from yt_dlp.utils import compat_etree_fromstring
    assert _registered_guard()["_MAX_F4M_NESTED"] == _JOINED_NESTED - 1
    fetched = []

    def _child(self, url, video_id, *a, **kw):
        fetched.append(url)
        return []
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        ie = InfoExtractor(ydl)
        ie._extract_f4m_formats = types.MethodType(_child, ie)
        ie._extract_m3u8_formats = types.MethodType(_child, ie)
        with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
            ie._parse_f4m_formats(
                compat_etree_fromstring(_f4m(media, bootstrap=False).encode()),
                manifest_url, "x")
    assert fetched == []


def test_guard_admits_an_f4m_set_manifest_of_a_few_streams():
    fetched = []

    def _child(self, url, video_id, *a, **kw):
        fetched.append(url)
        return [{"url": url, "format_id": url}]
    fmts = _parse_f4m(_f4m(_f4m_media(3, url="stream_{i}.f4m"), bootstrap=False),
                      _extract_f4m_formats=_child)
    assert fetched == [f"http://cdn.example/stream_{i}.f4m" for i in range(3)]
    assert len(fmts) == 3


def test_guard_bounds_nested_f4m_references_per_extraction():
    from yt_dlp.utils import compat_etree_fromstring
    nested = _registered_guard()["_MAX_F4M_NESTED"]
    half = nested // 2 + 1
    f4m_fetched, m3u8_fetched = [], []

    def _m3u8(self, url, video_id, *a, **kw):
        m3u8_fetched.append(url)
        return []

    def _child(self, url, video_id, *a, **kw):
        f4m_fetched.append(url)
        child = _f4m(_f4m_media(nested, url="c{i}.m3u8"), bootstrap=False)
        return self._parse_f4m_formats(compat_etree_fromstring(child.encode()),
                                       url, video_id)
    # Each manifest within the cap, nested ones included, is not enough: one
    # extraction fetches at most the cap's worth of references in all.
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_f4m(_f4m(_f4m_media(nested, url="s{i}.f4m"), bootstrap=False),
                   _extract_f4m_formats=_child, _extract_m3u8_formats=_m3u8)
    assert m3u8_fetched == [] and len(f4m_fetched) <= nested
    # Two manifests of one extraction share the budget...
    m3u8_fetched.clear()
    _reset_budget()
    _parse_f4m(_f4m(_f4m_media(half, url="a{i}.m3u8"), bootstrap=False),
               _extract_m3u8_formats=_m3u8)
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many variants"):
        _parse_f4m(_f4m(_f4m_media(half, url="b{i}.m3u8"), bootstrap=False),
                   _extract_m3u8_formats=_m3u8)
    assert len(m3u8_fetched) == half
    # ...and the next extraction starts from a full one.
    _reset_budget()
    _parse_f4m(_f4m(_f4m_media(half, url="b{i}.m3u8"), bootstrap=False),
               _extract_m3u8_formats=_m3u8)
    assert len(m3u8_fetched) == 2 * half


def test_guard_parses_an_ordinary_f4m_manifest_unchanged():
    from yt_dlp.utils import compat_etree_fromstring
    fmts = _parse_f4m(_f4m(_f4m_media(3)))
    assert [(f["format_id"], f["protocol"], f["ext"]) for f in fmts] == [
        ("1", "f4m", "flv"), ("2", "f4m", "flv"), ("3", "f4m", "flv")]
    # A set-level manifest naming stream-level ones: one level of nesting.
    stream = _f4m(_f4m_media(1))

    def _child(self, url, video_id, *a, **kw):
        return self._parse_f4m_formats(compat_etree_fromstring(stream.encode()),
                                       url, video_id)
    fmts = _parse_f4m(_f4m(_f4m_media(2, url="s{i}.f4m"), bootstrap=False),
                      _extract_f4m_formats=_child)
    assert [f["url"] for f in fmts] == ["http://cdn.example/s0.f4m",
                                        "http://cdn.example/s1.f4m"]
    assert len(_parse_f4m(_f4m(_f4m_media(_hls_cap())))) == _hls_cap()


@pytest.mark.parametrize("positional", [True, False], ids=["positional", "keyword"])
def test_guard_charges_f4m_base_url_per_node(monkeypatch, positional):
    """A set-level manifest's node addresses are each a fresh copy of its
    <baseURL> (else the manifest URL's directory) + "/" + the node's url:
    200 nodes under a 10 KB base ask for 2 MB."""
    from yt_dlp.extractor.common import InfoExtractor
    from yt_dlp.utils import compat_etree_fromstring

    def parse(xml, manifest_url):
        udl.guard_self_check(force=True)
        doc = compat_etree_fromstring(xml.encode())
        with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
            ie = InfoExtractor(ydl)
            if positional:
                return ie._parse_f4m_formats(doc, manifest_url, "x")
            return ie._parse_f4m_formats(doc, manifest_url=manifest_url, video_id="x")
    long_base = "http://cdn.example/" + "p" * 10_000
    short_url = "http://cdn.example/m.f4m"
    monkeypatch.setitem(_registered_guard(), "_MAX_FRAGMENT_URL_BYTES", 1_000_000)
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
        parse(_f4m(f"<baseURL>{long_base}</baseURL>" + _f4m_media(200),
                   bootstrap=False), short_url)
    # With no <baseURL>, the manifest URL's own directory is the base.
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
        parse(_f4m(_f4m_media(200), bootstrap=False), long_base + "/m.f4m")
    fmts = parse(_f4m("<baseURL>http://cdn.example/vod</baseURL>" + _f4m_media(200),
                      bootstrap=False), short_url)
    assert len(fmts) == 200
    assert fmts[0]["url"] == "http://cdn.example/vod/v0"
    # A stream-level manifest's formats share the manifest URL: no charge.
    assert len(parse(_f4m(_f4m_media(200)), long_base + "/m.f4m")) == 200


def test_guard_charges_chained_f4m_set_addresses(monkeypatch):
    """With no <baseURL>, yt-dlp joins each relative node onto the directory
    of the PREVIOUS node's address, so `d{i}/v` urls chain and grow: N nodes
    of L bytes build ~N^2 * L / 2 (2000 x 500 bytes: ~1 GB from a 1 MB
    document)."""
    media = "".join(f'<media url="{"A" * 500}{i}/v" bitrate="{i + 1}"/>'
                    for i in range(_hls_cap()))
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
        _parse_f4m(_f4m(media, bootstrap=False))
    _reset_budget()
    monkeypatch.setitem(_registered_guard(), "_MAX_FRAGMENT_URL_BYTES", 10_000_000)
    media = "".join(f'<media url="{"A" * 40}{i}/v" bitrate="{i + 1}"/>'
                    for i in range(1000))  # ~23 MB of addresses, 50 KB of XML
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
        _parse_f4m(_f4m(media, bootstrap=False))


_F4M2_NS = "http://ns.adobe.com/f4m/2.0"
_LONG_DIR_QUERY = "http://cdn.example/a/m.f4m?sig=x/y/z"


@pytest.mark.parametrize("manifest_url,xml", [
    ("http://cdn.example/m.f4m", _f4m(
        "".join(f'<media url="https://o{i}.example/p/v{i}.flv" bitrate="{i + 1}"/>'
                for i in range(5)), bootstrap=False)),
    (_LONG_DIR_QUERY, _f4m(_f4m_media(5), bootstrap=False)),
    ("http://cdn.example/a/m.f4m", _f4m(_f4m_media(6, url="d{i}/e/v"),
                                        bootstrap=False)),
    ("http://cdn.example/a/m.f4m", _f4m(
        '<media url="x/y" bitrate="1"/><media url="http://o.example/q/r" bitrate="2"/>'
        '<media url="s/" bitrate="3"/><media url="t" bitrate="4"/>'
        '<media url="u/v/w" bitrate="5"/><media url="https://o.example" bitrate="6"/>'
        '<media url="z" bitrate="7"/>', bootstrap=False)),
    ("http://cdn.example/a/m.f4m", _f4m(
        "<baseURL>  http://b.example/vod  </baseURL>"
        + _f4m_media(4, url="d{i}/v"), bootstrap=False)),
    ("http://cdn.example/a/m.f4m", _f4m(
        "<baseURL>  </baseURL>" + _f4m_media(4, url="d{i}/v"), bootstrap=False)),
    ("http://cdn.example/a/m.f4m",
     f'<?xml version="1.0"?><manifest xmlns="{_F4M2_NS}"><id>x</id>'
     '<media href="h0/v" url="u0" bitrate="1"/><media url="u1/w" bitrate="2"/>'
     '<media href="" url="u2" bitrate="3"/><media href="h3/v" bitrate="4"/>'
     '</manifest>'),
    ("http://cdn.example/a/m.f4m", _f4m(
        '<media url="http://o.example/enc/v" drmAdditionalHeaderId="k" bitrate="1"/>'
        '<media url="e1/v" bitrate="2"/><media url="" bitrate="3"/>'
        '<media url="e3/v" drmAdditionalHeaderSetId="k" bitrate="4"/>'
        '<media url="e4/v" bitrate="5"/>', bootstrap=False)),
    ("http://cdn.example/a/m.f4m", _f4m(
        '<media url="n0/c.f4m" bitrate="1"/><media url="n1/c.m3u8" bitrate="2"/>'
        '<media url="n2/v" bitrate="3"/>', bootstrap=False)),
    ("m.f4m", _f4m(_f4m_media(3, url="d{i}/v"), bootstrap=False)),
    ("", _f4m(_f4m_media(3, url="d{i}/v"), bootstrap=False)),
], ids=["absolute", "flat-query-slash", "chain", "mixed", "baseurl",
        "empty-baseurl", "f4m2-href", "encrypted-empty", "nested",
        "no-slash-url", "empty-url"])
def test_guard_charges_f4m_set_addresses_as_yt_dlp_builds_them(manifest_url, xml):
    """The set-level charge is sized arithmetically, never built: it must
    equal the bytes of the addresses yt-dlp then builds, node by node."""
    from yt_dlp.extractor.common import InfoExtractor
    from yt_dlp.utils import compat_etree_fromstring
    nested = []

    def _child(self, url, video_id, *a, **kw):
        nested.append(url)
        return []
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        ie = InfoExtractor(ydl)
        ie._extract_f4m_formats = types.MethodType(_child, ie)
        ie._extract_m3u8_formats = types.MethodType(_child, ie)
        dials = _registered_guard()["_DIALS"]
        dials.tls.url_bytes = 0
        fmts = ie._parse_f4m_formats(compat_etree_fromstring(xml.encode()),
                                     manifest_url, "x")
    built = sum(len(f["url"]) for f in fmts) + sum(len(u) for u in nested)
    assert built > 0
    assert dials.tls.url_bytes == built


def test_guard_is_not_installed_without_the_f4m_bound(monkeypatch):
    from yt_dlp.extractor.common import InfoExtractor
    guard = _inproc_guard()
    assert guard.is_installed()
    with monkeypatch.context() as m:
        m.setattr(InfoExtractor, "_parse_f4m_formats",
                  lambda self, *a, **kw: [])
        assert not guard.is_installed()
    with monkeypatch.context() as m:  # a yt-dlp that renamed the method
        m.delattr(InfoExtractor, "_parse_f4m_formats")
        assert not guard.is_installed()
        guard.install()  # nothing to bound: still fails closed
        assert not guard.is_installed()
    assert guard.is_installed()


# ---- HDS in the download child: F4mFD's build_fragments_list turns every
# segment-run count of the bootstrap (a uint32) into that many fragment
# tuples before it fetches any — a 106-byte bootstrap asks for billions. The
# guard sums the counts against the segment budget first.

def _boot(runs, *, live=False):
    return {"segments": [{"segment_run": runs}],
            "fragments": [{"fragments": [{"first": 5, "ts": 0, "duration": 1,
                                          "discontinuity_indicator": None}]}],
            "live": live}


@pytest.mark.parametrize("runs", [
    [(1, 1001)],
    [(1, 600), (2, 600)],
    [(1, 4294967294)],
    [(1, 4294967295)],  # only a live stream clamps it
], ids=["one-run", "two-runs", "uint32", "uint32-max-not-live"])
def test_guard_refuses_an_f4m_bootstrap_listing_too_many_fragments(
        monkeypatch, runs):
    import time
    from yt_dlp.downloader import f4m
    udl.guard_self_check(force=True)
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_SEGMENTS", 1000)
    # F4mFD looks the function up in its module at call time.
    assert f4m.F4mFD.real_download.__globals__ is vars(f4m)
    t0 = time.monotonic()
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many segments"):
        f4m.build_fragments_list(_boot(runs))
    assert time.monotonic() - t0 < 1.0


def test_guard_builds_an_ordinary_f4m_bootstrap_unchanged(monkeypatch):
    from yt_dlp.downloader import f4m
    udl.guard_self_check(force=True)
    assert f4m.build_fragments_list(_boot([(1, 3), (2, 2)])) == [
        (1, 5), (1, 6), (1, 7), (2, 8), (2, 9)]
    # A live stream keeps the last two, its uint32-max run counts as two.
    assert f4m.build_fragments_list(_boot([(1, 3), (2, 4294967295)], live=True)) == [
        (2, 8), (2, 9)]
    assert f4m.build_fragments_list(_boot([(1, 4294967295)], live=True)) == [
        (1, 5), (1, 6)]
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_SEGMENTS", 1000)
    assert len(f4m.build_fragments_list(_boot([(1, 400), (2, 600)]))) == 1000


def test_guard_is_not_installed_without_the_f4m_fragment_bound(monkeypatch):
    from yt_dlp.downloader import f4m
    guard = _inproc_guard()
    assert guard.is_installed()
    for name in ("build_fragments_list", "read_bootstrap_info"):
        with monkeypatch.context() as m:
            m.setattr(f4m, name, lambda arg: [])
            assert not guard.is_installed()
        with monkeypatch.context() as m:  # a yt-dlp that renamed the function
            m.delattr(f4m, name)
            assert not guard.is_installed()
            guard.install()  # nothing to bound: still fails closed
            assert not guard.is_installed()
    assert guard.is_installed()


# ---- HDS bootstrap bytes: FlvReader builds one dict per 16-byte afrt entry
# and one tuple per 8-byte asrt run from uint32 counts, so a bootstrap of tens
# of MB costs about 1 GB before build_fragments_list sees a count. Real HDS
# bootstraps are hundreds of bytes (live) to a few hundred KB (long VOD).

def _abst(entries, *, runs=((1, 1),), live=False):
    """An HDS bootstrap (abst box) with one asrt of `runs` and one afrt of
    (first, ts, duration) `entries`."""
    import struct

    def box(kind, data):
        return struct.pack("!I", 8 + len(data)) + kind + data
    asrt = box(b"asrt", b"\0" * 5 + struct.pack("!I", len(runs))
               + b"".join(struct.pack("!II", *run) for run in runs))
    afrt = box(b"afrt", b"\0" * 4 + struct.pack("!I", 1000) + b"\0"
               + struct.pack("!I", len(entries))
               + b"".join(struct.pack("!IQI", *e) for e in entries))
    return box(b"abst", b"\0" * 8 + (b"\x20" if live else b"\0")
               + struct.pack("!I", 1000) + b"\0" * 16 + b"\0\0\0\0\0"
               + b"\1" + asrt + b"\1" + afrt)


def test_guard_refuses_an_oversized_f4m_bootstrap():
    import time
    from yt_dlp.downloader import f4m
    udl.guard_self_check(force=True)
    # F4mFD looks the function up in its module at call time.
    assert f4m.F4mFD.real_download.__globals__ is vars(f4m)
    boot = _abst([(i + 1, i * 4000, 4000) for i in range(140_000)])
    assert len(boot) > 2 * 1024 * 1024
    t0 = time.monotonic()
    for data in (boot, bytearray(boot), memoryview(boot)):
        with pytest.raises(yt_dlp.utils.ExtractorError, match="too many segments"):
            f4m.read_bootstrap_info(data)
    assert time.monotonic() - t0 < 1.0


def test_guard_reads_an_ordinary_f4m_bootstrap_unchanged(monkeypatch):
    from yt_dlp.downloader import f4m
    udl.guard_self_check(force=True)
    boot = _abst([(1, 0, 4000), (2, 4000, 4000), (3, 8000, 4000)],
                 runs=((1, 3),), live=True)
    info = f4m.read_bootstrap_info(boot)
    assert info == f4m.FlvReader(boot).read_bootstrap_info()
    assert info["live"] is True
    assert info["segments"] == [{"segment_run": [(1, 3)]}]
    assert [f["first"] for f in info["fragments"][0]["fragments"]] == [1, 2, 3]
    # A 12 h VOD at 4 s fragments, one afrt entry each: ~173 KB.
    vod = _abst([(i + 1, i * 4000, 4000) for i in range(10_800)],
                runs=((1, 10_800),))
    info = f4m.read_bootstrap_info(vod)
    assert len(info["fragments"][0]["fragments"]) == 10_800
    assert len(f4m.build_fragments_list(info)) == 10_800
    # Right at the cap still parses; one byte past it is refused.
    monkeypatch.setitem(f4m.read_bootstrap_info.__globals__,
                        "_MAX_F4M_BOOTSTRAP_BYTES", len(vod))
    assert f4m.read_bootstrap_info(vod) == info
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too many segments"):
        f4m.read_bootstrap_info(vod + b"\0")


# ---- SMIL: _parse_smil_formats_and_subtitles makes one format per distinct
# src (video / audio / media / imagestream), joins each relative one onto the
# page's <meta base> as a fresh copy, and fetches every src naming an HLS /
# HDS / DASH / Smooth Streaming manifest. The guard counts the srcs, sizes the
# addresses and counts the nested manifests before the parse.

_SMIL_NS = "http://www.w3.org/2005/SMIL21/Language"


def _smil(body, *, base=None, ns=None):
    head = f'<head><meta base="{base}"/></head>' if base else ""
    xmlns = f' xmlns="{ns}"' if ns else ""
    return f"<smil{xmlns}>{head}<body><switch>{body}</switch></body></smil>"


def _parse_smil(xml, smil_url="http://cdn.example/v.smil", **stubs):
    """InfoExtractor._parse_smil on `xml`, with the named methods of the
    extractor replaced by `stubs` (the nested-manifest fetches)."""
    from yt_dlp.extractor.common import InfoExtractor
    from yt_dlp.utils import compat_etree_fromstring
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        ie = InfoExtractor(ydl)
        for name, fn in stubs.items():
            setattr(ie, name, types.MethodType(fn, ie))
        return ie._parse_smil(compat_etree_fromstring(xml.encode()), smil_url, "x")


def test_guard_charges_smil_addresses_before_building_them(monkeypatch):
    """Every relative src is urljoined onto <meta base> as a fresh copy:
    1 000 srcs under a 1 MB base build ~1 GB from a ~1 MB document."""
    monkeypatch.setitem(_registered_guard(), "_MAX_FRAGMENT_URL_BYTES", 10_000_000)
    base = "http://cdn.example/" + "p" * 100_000
    body = "".join(f'<video src="v{i}.mp4"/>' for i in range(200))  # ~20 MB
    with pytest.raises(yt_dlp.utils.ExtractorError, match="too long"):
        _parse_smil(_smil(body, base=base))


def test_guard_refuses_a_smil_listing_too_many_formats():
    cap = _hls_cap()
    rtmp = "".join(f'<video src="mp4:v{i}.mp4"/>' for i in range(cap + 1))
    with pytest.raises(yt_dlp.utils.ExtractorError, match=f"more than {cap}"):
        _parse_smil(_smil(rtmp, base="rtmp://cdn.example/app"))
    # yt-dlp shares one set of srcs between media and storyboards.
    _reset_budget()
    half = cap // 2
    mixed = ("".join(f'<video src="v{i}.mp4"/>' for i in range(half))
             + "".join(f'<imagestream src="i{i}.jpg"/>' for i in range(cap + 1 - half)))
    with pytest.raises(yt_dlp.utils.ExtractorError, match=f"more than {cap}"):
        _parse_smil(_smil(mixed))
    # A src listed twice is one format.
    _reset_budget()
    twice = "".join(f'<video src="v{i % cap}.mp4"/>' for i in range(cap + 1))
    assert len(_parse_smil(_smil(twice))["formats"]) == cap


@pytest.mark.parametrize("make", [
    lambda i: f'<video src="c{i}.m3u8"/>',
    lambda i: f'<video src="c{i}.ts" proto="m3u8"/>',
    lambda i: f'<video src="p/c-{i}" ext="mpd"/>',
    lambda i: f'<audio src="c{i}.f4m"/>',
    lambda i: f'<media src="c{i}.ism/Manifest" ext="mp4"/>',
], ids=["m3u8", "proto", "ext-attr", "f4m", "ism"])
def test_guard_counts_smil_nested_manifests(make):
    fetched = []

    def _child(self, url, *a, **kw):
        fetched.append(url)
        return [], {}

    def _f4m_child(self, url, *a, **kw):
        fetched.append(url)
        return []
    nested = _registered_guard()["_MAX_F4M_NESTED"]
    stubs = {"_extract_m3u8_formats_and_subtitles": _child,
             "_extract_mpd_formats_and_subtitles": _child,
             "_extract_ism_formats_and_subtitles": _child,
             "_extract_f4m_formats": _f4m_child}
    with pytest.raises(yt_dlp.utils.ExtractorError, match=f"more than {nested}"):
        _parse_smil(_smil("".join(make(i) for i in range(nested + 1))), **stubs)
    assert fetched == []
    # Within the cap every one is still fetched.
    _reset_budget()
    _parse_smil(_smil("".join(make(i) for i in range(nested))), **stubs)
    assert len(fetched) == nested


def test_guard_counts_a_smil_base_that_names_an_ism_manifest():
    fetched = []

    def _child(self, url, *a, **kw):
        fetched.append(url)
        return [], {}
    nested = _registered_guard()["_MAX_F4M_NESTED"]
    body = "".join(f'<video src="../Manifest?i={i}" ext="mp4"/>' for i in range(nested + 1))
    with pytest.raises(yt_dlp.utils.ExtractorError, match=f"more than {nested}"):
        _parse_smil(_smil(body, base="http://cdn.example/s.ism/z"),
                    _extract_ism_formats_and_subtitles=_child)
    assert fetched == []


def test_guard_admits_a_real_smil():
    real = _smil("".join(
        f'<video src="clip_{b}.mp4" system-bitrate="{b}000" width="640" '
        'height="360"/>' for b in (400, 800, 1200, 2000)),
        base="http://cdn.example/vod", ns=_SMIL_NS)
    info = _parse_smil(real)
    assert [f["format_id"] for f in info["formats"]] == [
        "http-400", "http-800", "http-1200", "http-2000"]
    assert info["formats"][0]["url"] == "http://cdn.example/vod/clip_400.mp4"
    dials = _registered_guard()["_DIALS"]
    assert dials.tls.variants == 4 and 0 < dials.tls.url_bytes < 1000
    rtmp = _smil('<video src="mp4:a.mp4" system-bitrate="500000"/>'
                 '<video src="mp4:b.mp4" system-bitrate="900000"/>',
                 base="rtmp://cdn.example/app")
    assert [(f["url"], f["play_path"]) for f in _parse_smil(rtmp)["formats"]] == [
        ("rtmp://cdn.example/app", "mp4:a.mp4"), ("rtmp://cdn.example/app", "mp4:b.mp4")]



def test_guard_refuses_a_smil_with_too_many_textstreams():
    """_parse_smil_subtitles dedups the textstream srcs with a list, which is
    quadratic in them."""
    import time
    cap = _hls_cap()
    t0 = time.monotonic()
    with pytest.raises(yt_dlp.utils.ExtractorError, match=f"more than {cap}"):
        _parse_smil(_smil("".join(f'<textstream src="s{i}.vtt"/>'
                                  for i in range(cap + 1))))
    assert time.monotonic() - t0 < 1.0
    # A repeated src is one track, as yt-dlp keeps it.
    twice = "".join(f'<textstream src="s{i % cap}.vtt"/>' for i in range(cap + 5))
    assert len(_parse_smil(_smil('<video src="v.mp4"/>' + twice))["subtitles"]["en"]) == cap
    real = _smil('<video src="v.mp4"/>' + "".join(
        f'<textstream src="http://cdn.example/{lang}.vtt" systemLanguage="{lang}"/>'
        for lang in ("en", "de", "fr")))
    assert _parse_smil(real)["subtitles"] == {
        lang: [{"url": f"http://cdn.example/{lang}.vtt", "ext": "vtt"}]
        for lang in ("en", "de", "fr")}


def test_guard_refuses_a_smil_whose_duplicate_textstreams_would_scan_quadratically():
    """Every repeated textstream src is a scan of the whole list of distinct
    ones, so the cap of distinct srcs alone does not bound the dedup."""
    cap = _hls_cap()
    distinct = "".join(f'<textstream src="s{i}.vtt"/>' for i in range(cap))
    copies = f'<textstream src="s{cap - 1}.vtt"/>' * 5_000
    with pytest.raises(yt_dlp.utils.ExtractorError, match=f"more than {cap}"):
        _parse_smil(_smil('<video src="v.mp4"/>' + distinct + copies))


def test_guard_charges_smil_textstream_dedup_by_length(monkeypatch):
    """The dedup's comparisons cost the length of the srcs they compare (only
    equal-length strings are compared byte by byte), so few but long srcs of
    one length stay under the comparison count and still scan for seconds."""
    def body(prefix):
        distinct = "".join(f'<textstream src="{prefix}{i:06d}.vtt"/>' for i in range(200))
        copies = f'<textstream src="{prefix}{199:06d}.vtt"/>' * 300
        return _smil('<video src="v.mp4"/>' + distinct + copies)
    # ~160 MB compared: under the real budget, admitted as yt-dlp keeps it.
    long = body("s" * 1990)
    assert len(_parse_smil(long)["subtitles"]["en"]) == 200
    _reset_budget()
    monkeypatch.setitem(_registered_guard(), "_MAX_SMIL_SUB_SCAN_BYTES", 10_000_000)
    with pytest.raises(yt_dlp.utils.ExtractorError, match=f"more than {_hls_cap()}"):
        _parse_smil(long)
    # Short srcs of the same count stay well under even the lowered budget.
    _reset_budget()
    assert len(_parse_smil(body("s"))["subtitles"]["en"]) == 200


def test_guard_bounds_smil_parsed_positionally():
    """MediasetIE overrides the parse and hands every argument to yt-dlp's
    positionally."""
    from yt_dlp.extractor.common import InfoExtractor
    from yt_dlp.utils import compat_etree_fromstring
    cap = _hls_cap()
    doc = compat_etree_fromstring(_smil(
        "".join(f'<video src="v{i}.mp4"/>' for i in range(cap + 1)),
        ns=_SMIL_NS).encode())
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        ie = InfoExtractor(ydl)
        with pytest.raises(yt_dlp.utils.ExtractorError, match=f"more than {cap}"):
            ie._parse_smil_formats_and_subtitles(
                doc, "http://cdn.example/v.smil", "x", _SMIL_NS, None, None)


def test_guard_is_not_installed_without_the_smil_bound(monkeypatch):
    from yt_dlp.extractor.common import InfoExtractor
    guard = _inproc_guard()
    assert guard.is_installed()
    with monkeypatch.context() as m:
        m.setattr(InfoExtractor, "_parse_smil_formats_and_subtitles",
                  lambda self, *a, **kw: ([], {}))
        assert not guard.is_installed()
    with monkeypatch.context() as m:  # a yt-dlp that renamed the method
        m.delattr(InfoExtractor, "_parse_smil_formats_and_subtitles")
        assert not guard.is_installed()
        guard.install()  # nothing to bound: still fails closed
        assert not guard.is_installed()
    assert guard.is_installed()


# ---- Every builder's formats (HTML5 <source> tags, JW Player, XSPF, ...)
# then go through YoutubeDL's format pass at ~0.5 ms each. The formats one
# extraction hands it are counted, across its entries, before each pass.

def _videos(n_entries, n_formats):
    return {"_type": "playlist", "id": "p", "title": "p", "extractor": "generic",
            "extractor_key": "Generic", "webpage_url": "http://cdn.example/p",
            "entries": [{
                "id": f"e{e}", "title": f"e{e}", "extractor": "generic",
                "extractor_key": "Generic", "webpage_url": "http://cdn.example/p",
                "formats": [{"url": f"http://cdn.example/{e}/{i}.mp4",
                             "format_id": f"f{i}", "ext": "mp4"}
                            for i in range(n_formats)]}
                for e in range(n_entries)]}


def test_guard_caps_formats_across_one_extraction(monkeypatch):
    guard = _registered_guard()
    assert guard["_MAX_EXTRACTION_FORMATS"] == guard["_MAX_MPD_REPRESENTATIONS"]
    monkeypatch.setitem(guard, "_MAX_EXTRACTION_FORMATS", 50)
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        # yt-dlp reports a failed entry as a DownloadError.
        with pytest.raises(yt_dlp.utils.DownloadError, match="more than 50"):
            ydl.process_ie_result(_videos(2, 26), download=False)
        # The first entry's 26 formats passed; only the second was refused.
        assert _registered_guard()["_DIALS"].tls.processed_formats == 26
    # At the cap it passes, and the next extraction starts afresh.
    _reset_budget()
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        res = ydl.process_ie_result(_videos(2, 25), download=False)
        assert [len(e["formats"]) for e in res["entries"]] == [25, 25]



def _url_only_videos(n_entries):
    """Entries a scrape builds from bare media links: a url, no formats."""
    return {"_type": "playlist", "id": "p", "title": "p", "extractor": "generic",
            "extractor_key": "Generic", "webpage_url": "http://cdn.example/p",
            "entries": [{
                "id": f"e{e}", "title": f"e{e}", "extractor": "generic",
                "extractor_key": "Generic", "webpage_url": "http://cdn.example/p",
                "url": f"http://cdn.example/{e}.mp4"} for e in range(n_entries)]}


def test_guard_counts_url_only_entries_in_the_format_pass(monkeypatch):
    """yt-dlp makes a url-only entry its own single format, and each entry
    costs a format pass all the same."""
    monkeypatch.setitem(_registered_guard(), "_MAX_EXTRACTION_FORMATS", 50)
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        with pytest.raises(yt_dlp.utils.DownloadError, match="more than 50"):
            ydl.process_ie_result(_url_only_videos(51), download=False)
    _reset_budget()
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        res = ydl.process_ie_result(_url_only_videos(50), download=False)
        assert len(res["entries"]) == 50
    _reset_budget()
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        ydl.process_ie_result(_url_only_videos(1)["entries"][0], download=False)
    assert _registered_guard()["_DIALS"].tls.processed_formats == 1


def _labelled(kind, chars, n):
    """`n` formats that all take their name from one `chars`-long attribute
    (an AdaptationSet @id, a StreamIndex Name, the NAME of the rendition
    group every variant names), through the real parser and format pass."""
    from yt_dlp.extractor.common import InfoExtractor
    from yt_dlp.utils import compat_etree_fromstring
    name = "i" * chars
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        ie = InfoExtractor(ydl)
        if kind == "mpd":
            fmts, _ = ie._parse_mpd_formats_and_subtitles(compat_etree_fromstring(_mpd(
                f'<Period><AdaptationSet mimeType="audio/mp4" id="{name}">' + "".join(
                    f'<Representation bandwidth="{i + 1}" codecs="mp4a.40.2">'
                    f"<BaseURL>a{i}.m4a</BaseURL></Representation>" for i in range(n))
                + "</AdaptationSet></Period>").encode()),
                mpd_base_url="http://cdn.example/", mpd_url="http://cdn.example/m.mpd")
        elif kind == "ism":
            fmts, _ = ie._parse_ism_formats_and_subtitles(compat_etree_fromstring(_ism(
                _ism_audio('<c t="0" d="20000000"/>', levels=n).replace(
                    'Name="audio"', f'Name="{name}"')).encode()),
                "http://cdn.example/x.ism/Manifest")
        else:
            fmts, _ = ie._parse_m3u8_formats_and_subtitles(
                f'#EXTM3U\n#EXT-X-MEDIA:TYPE=VIDEO,GROUP-ID="g",NAME="{name}"\n'
                + "".join(f'#EXT-X-STREAM-INF:BANDWIDTH={i + 1},VIDEO="g"\nv{i}.m3u8\n'
                          for i in range(n)),
                "http://cdn.example/master.m3u8", ext="mp4")
        return ydl.process_ie_result({
            "id": "x", "title": "t", "formats": fmts, "extractor": "generic",
            "extractor_key": "Generic", "webpage_url": "http://cdn.example/m"},
            download=False)["formats"]


@pytest.mark.parametrize("kind", ["mpd", "ism", "hls"])
def test_guard_caps_the_labels_one_extraction_copies_per_format(kind):
    """The format pass rebuilds every format's id and label, ~5 copies of
    text the manifest states once: 3 000 characters x 1 500 formats are ~22 MB
    of copies unguarded. The refusal comes before any of them."""
    import tracemalloc
    guard = _registered_guard()
    assert guard["_MAX_FORMAT_LABEL_CHARS"] >= 20_000 * 200
    tracemalloc.start()
    try:
        with pytest.raises(yt_dlp.utils.ExtractorError,
                           match="format names are too long"):
            _labelled(kind, 3000, 1500)
        assert tracemalloc.get_traced_memory()[1] < 16 * 1024 * 1024
    finally:
        tracemalloc.stop()
    _reset_budget()
    assert len(_labelled(kind, 60, 300)) == 300


def _plain_formats(n, **first):
    """`n` direct-link formats as a scrape builds them; `first` overrides the
    first one's fields."""
    formats = [{"url": f"http://cdn.example/{i}.mp4", "format_id": f"f{i}",
                "ext": "mp4", "protocol": "https"} for i in range(n)]
    formats[0].update(first)
    return [{k: v for k, v in f.items() if v is not None} for f in formats]


@pytest.mark.parametrize("first", [
    {"format_id": "i" * 10_000},
    {"language": "l" * 10_000},
    {"vcodec": "avc1." + "c" * 10_000},
    {"width": 10 ** 4_000, "height": 10 ** 4_000},
    # Fields the pass derives from the address when a format has none.
    {"ext": None, "url": "http://cdn.example/a." + "x" * 10_000},
    {"protocol": None, "url": "x" * 10_000 + "://cdn.example/a.mp4"},
], ids=["format-id", "language", "codec", "resolution", "url-ext", "url-scheme"])
def test_guard_caps_the_widest_label_times_the_formats(first):
    """yt-dlp renders its formats table on every pass and pads each row to
    the widest cell of every column: ONE 200 000-character id among 2 000
    formats peaked at 1.2 GB. The budget is rows x widest, not the sum."""
    guard = _registered_guard()
    info = {"id": "x", "title": "t", "extractor": "generic",
            "extractor_key": "Generic", "webpage_url": "http://cdn.example/m",
            "formats": _plain_formats(1000, **first)}
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        with pytest.raises(yt_dlp.utils.ExtractorError,
                           match="format names are too long"):
            ydl.process_ie_result(info, download=False)
    assert guard["_DIALS"].tls.label_chars == 0
    # The same thousand formats without the long one pass.
    info["formats"] = _plain_formats(1000)
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        assert len(ydl.process_ie_result(info, download=False)["formats"]) == 1000


def test_guard_admits_the_labels_of_real_formats_at_the_variant_cap():
    """20 000 formats with ordinary ids, codecs, notes and signed addresses
    spend about a third of the label budget."""
    guard = _registered_guard()
    formats = [{
        "format_id": f"video=4500000-{i:05d}", "ext": "mp4",
        "container": "mp4_dash", "width": 3840, "height": 2160,
        "vcodec": "hvc1.2.4.L153.B0", "acodec": "mp4a.40.2", "language": "en-US",
        "format_note": "DASH video", "protocol": "http_dash_segments",
        "url": f"https://cdn.example/v/{i}/seg.m4s?sig=" + "s" * 300,
    } for i in range(guard["_MAX_EXTRACTION_FORMATS"])]
    chars = guard["_format_label_chars"](formats)
    assert 0 < chars < guard["_MAX_FORMAT_LABEL_CHARS"] * 0.6
    # A format with no ext / protocol of its own gets them from its address.
    bare = [{"url": "https://cdn.example/v/a.webm?sig=" + "s" * 300}]
    assert guard["_format_label_chars"](bare) == len("webm") + len("https")


_SCTE214 = "urn:scte:dash:scte214-extensions"


@pytest.mark.parametrize("attrs", [
    'mimeType="audio/{big}" codecs="mp4a.40.2"',
    'mimeType="audio/mp4" codecs="mp4a.{big}"',
    f'mimeType="audio/mp4" codecs="mp4a.40.2" xmlns:s="{_SCTE214}" '
    's:supplementalCodecs="{big}"',
], ids=["mime-type", "codecs", "supplemental-codecs"])
def test_guard_charges_the_text_a_dash_parse_copies_per_representation(attrs):
    """yt-dlp derives every Representation's ext, container, codecs and note
    afresh from attributes its AdaptationSet states once: 100 000 characters
    x 2 000 Representations were 200-470 MB before the format pass saw any."""
    import tracemalloc
    xml = _mpd('<Period><AdaptationSet ' + attrs.replace("{big}", "x" * 3000)
               + ">" + "".join(
                   f'<Representation id="r{i}" bandwidth="{i + 1}">'
                   f"<BaseURL>a{i}.m4a</BaseURL></Representation>"
                   for i in range(1500))
               + "</AdaptationSet></Period>")
    udl.guard_self_check(force=True)
    tracemalloc.start()
    try:
        with pytest.raises(yt_dlp.utils.ExtractorError,
                           match="format names are too long"):
            _parse_mpd(xml)
        assert tracemalloc.get_traced_memory()[1] < 16 * 1024 * 1024
    finally:
        tracemalloc.stop()


def test_guard_admits_the_text_of_real_dash_representations_at_the_cap():
    from yt_dlp.utils import compat_etree_fromstring
    guard = _registered_guard()
    xml = _mpd('<Period><AdaptationSet mimeType="video/mp4" '
               'codecs="hvc1.2.4.L153.B0" lang="en-US">' + "".join(
                   f'<Representation id="video=4500000-{i:05d}" '
                   f'bandwidth="{i + 1}"/>'
                   for i in range(guard["_MAX_MPD_REPRESENTATIONS"]))
               + "</AdaptationSet></Period>")
    guard["_refuse_oversized_mpd"](compat_etree_fromstring(xml.encode()))
    assert guard["_DIALS"].tls.variants == guard["_MAX_MPD_REPRESENTATIONS"]


def test_guard_charges_labels_across_one_extraction(monkeypatch):
    """A playlist's entries each run the format pass: their labels come from
    one budget, and a refused entry spends none of it."""
    guard = _registered_guard()
    monkeypatch.setitem(guard, "_MAX_FORMAT_LABEL_CHARS", 1000)
    info = {"_type": "playlist", "id": "p", "title": "p", "extractor": "generic",
            "extractor_key": "Generic", "webpage_url": "http://cdn.example/p",
            "entries": [
                {"id": f"v{e}", "title": "t", "extractor": "generic",
                 "extractor_key": "Generic", "webpage_url": "http://cdn.example/v",
                 "formats": [{"url": f"http://cdn.example/{e}-{i}.mp4",
                              "format_id": "i" * 60, "format_note": "n" * 60}
                             for i in range(5)]}
                for e in range(2)]}
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        # yt-dlp reports a failed entry as a DownloadError.
        with pytest.raises(yt_dlp.utils.DownloadError,
                           match="format names are too long"):
            ydl.process_ie_result(info, download=False)
    # The first entry's five formats: id, note, and the ext and protocol
    # their addresses spell.
    assert guard["_DIALS"].tls.label_chars == 5 * (60 + 60 + len("mp4") + len("http"))
    assert guard["_DIALS"].tls.processed_formats == 5


def test_guard_does_not_charge_the_format_pass_to_the_manifest_budget(monkeypatch):
    """DASH and HLS formats are charged once, where their manifest is parsed:
    the format pass keeps its own count."""
    monkeypatch.setitem(_registered_guard(), "_MAX_MPD_REPRESENTATIONS", 30)
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        ydl.process_ie_result(_videos(1, 40), download=False)
    assert _registered_guard()["_DIALS"].tls.variants == 0


def _site_cookies(ydl, lines):
    """The jar after a response from cdn.example carrying these Set-Cookie
    header lines, stored as the stock cookie handler stores them."""
    headers = http.client.HTTPMessage()
    for line in lines:
        headers["Set-Cookie"] = line
    ydl.cookiejar.extract_cookies(
        types.SimpleNamespace(info=lambda: headers),
        urllib.request.Request("http://cdn.example/m"))


_COOKIE_INFO = {"id": "x", "title": "t", "extractor": "generic",
                "extractor_key": "Generic", "webpage_url": "http://cdn.example/m"}


def test_guard_sizes_the_cookies_every_format_copies(monkeypatch):
    """The format pass writes all the site's matching cookies into EVERY
    format as a string of its own: 1.4 MB of Set-Cookie on one response and
    a 1 000-variant master peaked at 1.4 GB. Formats x the jar are sized
    before the pass, from one budget per extraction."""
    guard = _registered_guard()
    _reset_budget()
    monkeypatch.setitem(guard, "_MAX_FORMAT_COOKIE_CHARS", 100_000)
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        _site_cookies(ydl, [f"c{i}={'v' * 1000}; Path=/" for i in range(10)])
        jar = guard["_jar_chars"](ydl)
        with pytest.raises(yt_dlp.utils.ExtractorError,
                           match="cookies are too large"):
            ydl.process_ie_result(
                {**_COOKIE_INFO, "formats": _plain_formats(20)}, download=False)
        # A refused pass spends nothing, and fewer formats still pass.
        assert guard["_DIALS"].tls.cookie_chars == 0
        assert guard["_DIALS"].tls.processed_formats == 0
        res = ydl.process_ie_result(
            {**_COOKIE_INFO, "formats": _plain_formats(9)}, download=False)
        assert guard["_DIALS"].tls.cookie_chars == 9 * jar
        copied = [len(f["cookies"]) for f in res["formats"]]
        assert len(copied) == 9 and 10 * 1000 < max(copied) <= jar
        # A second entry of the same extraction spends from the same budget.
        with pytest.raises(yt_dlp.utils.ExtractorError,
                           match="cookies are too large"):
            ydl.process_ie_result(
                {**_COOKIE_INFO, "formats": _plain_formats(1)}, download=False)
    _reset_budget()
    assert guard["_DIALS"].tls.cookie_chars == 0


@pytest.mark.parametrize("cookie", [
    "c{i}=" + "v" * 1000,
    # yt-dlp writes a value as a quoted string, four characters for each one
    # it escapes.
    "c{i}=" + "," * 1000,
    "c{i}=" + "\xff" * 1000,
    "c{i}=a" + '"' * 1000,
    "c{i}",
    "c{i}=v; Max-Age=" + "9" * 300,
    "c{i}=v; Expires=Wed, 09 Jun 2100 10:18:14 GMT; Secure",
    "c{i}=v; Domain=.cdn.example; Path=/" + "p" * 1000,
], ids=["plain", "commas", "latin-1", "quotes", "no-value", "max-age",
        "expires", "path"])
def test_guard_sizes_a_cookie_no_smaller_than_a_format_copies_it(cookie):
    """Whatever a Set-Cookie line states, the size of the jar is at least
    the text each format gets from it."""
    guard = _registered_guard()
    _reset_budget()
    formats = [{"url": "https://cdn.example/" + "p" * 1000 + f"/{i}.mp4",
                "format_id": f"f{i}", "ext": "mp4"} for i in range(3)]
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        _site_cookies(ydl, [cookie.replace("{i}", str(i)) for i in range(10)])
        assert len(ydl.cookiejar) == 10
        jar = guard["_jar_chars"](ydl)
        res = ydl.process_ie_result({**_COOKIE_INFO, "formats": formats},
                                    download=False)
    copied = [len(f["cookies"]) for f in res["formats"]]
    assert 0 < max(copied) <= jar <= max(copied) + 10 * 64
    assert guard["_DIALS"].tls.cookie_chars == 3 * jar


def test_guard_admits_the_cookies_of_real_sites():
    """A few kB of cookies and a hundred formats are a sliver of the budget,
    a site that sets none spends nothing, and the cap leaves a 3 kB jar room
    for every format one extraction may hand the pass."""
    guard = _registered_guard()
    assert (guard["_MAX_FORMAT_COOKIE_CHARS"]
            >= guard["_MAX_EXTRACTION_FORMATS"] * 3000)
    _reset_budget()
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        assert guard["_jar_chars"](ydl) == 0
        ydl.process_ie_result(
            {**_COOKIE_INFO, "formats": _plain_formats(100)}, download=False)
        assert guard["_DIALS"].tls.cookie_chars == 0
        _site_cookies(ydl, [f"c{i}={'v' * 250}; Path=/; Secure; Max-Age=86400"
                            for i in range(8)])
        res = ydl.process_ie_result(
            {**_COOKIE_INFO, "formats": _plain_formats(100)}, download=False)
    assert len(res["formats"]) == 100
    assert (0 < guard["_DIALS"].tls.cookie_chars
            < guard["_MAX_FORMAT_COOKIE_CHARS"] // 200)


def test_guard_is_not_installed_without_the_format_pass_bound(monkeypatch):
    from yt_dlp.YoutubeDL import YoutubeDL
    guard = _inproc_guard()
    assert guard.is_installed()
    with monkeypatch.context() as m:
        m.setattr(YoutubeDL, "process_video_result",
                  lambda self, info_dict, download=True: info_dict)
        assert not guard.is_installed()
    with monkeypatch.context() as m:  # a yt-dlp that renamed the method
        m.delattr(YoutubeDL, "process_video_result")
        assert not guard.is_installed()
        guard.install()  # nothing to bound: still fails closed
        assert not guard.is_installed()
    assert guard.is_installed()


# ---- The tables yt-dlp draws. It renders the formats, thumbnails, subtitles
# and automatic-captions tables at every print point of an entry, asked for
# or not, and pads each row to the widest cell of every column: one long
# cell costs its length per row.

def _listed(**lists):
    return {"id": "x", "title": "t", "extractor": "generic",
            "extractor_key": "Generic", "webpage_url": "http://cdn.example/m",
            "formats": _plain_formats(3), **lists}


def _subtitles(n, long=0, formats=("vtt",), name=None):
    subs = {f"l{i}": [{"url": f"http://cdn.example/{i}.{ext}", "ext": ext,
                       **({"name": name} if name else {})} for ext in formats]
            for i in range(n)}
    if long:
        subs["x" * long] = subs.pop("l0")
    return subs


def _thumbnails(n, long=0):
    return [{"url": f"http://cdn.example/{i}.jpg" + "?" + "x" * (long if i == 0 else 0),
             "width": 320, "height": 180} for i in range(n)]


@pytest.mark.parametrize("lists", [
    lambda long: {"subtitles": _subtitles(1000, long)},
    lambda long: {"automatic_captions": _subtitles(1000, long)},
    lambda long: {"thumbnails": _thumbnails(1000, long)},
], ids=["subtitles", "automatic-captions", "thumbnails"])
def test_guard_caps_the_widest_cell_times_the_rows_of_every_table(lists):
    """ONE 200 000-character subtitle language among 2 000 (a 485 KiB DASH
    manifest) peaked at 1.2 GB: every row of the table is padded to it."""
    guard = _registered_guard()
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        with pytest.raises(yt_dlp.utils.ExtractorError,
                           match="format names are too long"):
            ydl.process_ie_result(_listed(**lists(10_000)), download=False)
    # A refused table spends nothing: only the tables before it were drawn.
    assert guard["_DIALS"].tls.table_chars < 10_000
    _reset_budget()
    # The same thousand rows without the long one pass.
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        res = ydl.process_ie_result(_listed(**lists(0)), download=False)
    assert len(res["formats"]) == 3
    assert 0 < guard["_DIALS"].tls.table_chars < guard["_MAX_TABLE_CHARS"] // 16


def test_guard_caps_a_long_subtitle_language_of_a_dash_manifest():
    """The measured shape, through the real parser: text AdaptationSets, one
    with a long @lang."""
    from yt_dlp.extractor.common import InfoExtractor
    from yt_dlp.utils import compat_etree_fromstring
    xml = _mpd(
        '<Period><AdaptationSet mimeType="audio/mp4" codecs="mp4a.40.2">'
        '<Representation id="a" bandwidth="1"><BaseURL>a.m4a</BaseURL>'
        "</Representation></AdaptationSet>" + "".join(
            f'<AdaptationSet mimeType="text/vtt" lang="{"x" * 10_000 if i == 0 else f"l{i}"}">'
            f'<Representation id="t{i}" bandwidth="1"><BaseURL>s{i}.vtt</BaseURL>'
            "</Representation></AdaptationSet>" for i in range(1000))
        + "</Period>")
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        fmts, subs = InfoExtractor(ydl)._parse_mpd_formats_and_subtitles(
            compat_etree_fromstring(xml.encode()),
            mpd_base_url="http://cdn.example/", mpd_url="http://cdn.example/m.mpd")
        assert len(subs) == 1000
        with pytest.raises(yt_dlp.utils.ExtractorError,
                           match="format names are too long"):
            ydl.process_ie_result(
                {**_listed(), "formats": fmts, "subtitles": subs}, download=False)


def test_guard_sizes_a_table_before_it_is_drawn():
    """Rows x the sum of the widest cell of each column, whatever the cells
    are: the renderer is never reached with a table past the budget."""
    guard = _registered_guard()
    udl.guard_self_check(force=True)
    render = sys.modules["yt_dlp.YoutubeDL"].render_table
    assert guard["_padded_chars"]([(1, 5, 0), (3, 2, 0), (2, 2, 7)]) == 3 * (3 + 5 + 7)
    assert guard["_padded_chars"]([]) == 0
    assert guard["_cell_chars"]("abc") == 3
    assert guard["_cell_chars"](None) == len("None")
    assert guard["_cell_chars"](1.5) == 3
    # A number is sized without the conversion.
    assert len(str(10 ** 4000)) <= guard["_cell_chars"](10 ** 4000) < 4100
    assert render(["A", "B"], [["a", "1"], ["bb", "22"]]) == "A  B\na  1\nbb 22"
    assert guard["_DIALS"].tls.table_chars == 2 * (2 + 2)
    cap = guard["_MAX_FORMAT_LABEL_CHARS"]
    with mock.patch.dict(guard, {"_MAX_FORMAT_LABEL_CHARS": 100}):
        with pytest.raises(yt_dlp.utils.ExtractorError,
                           match="format names are too long"):
            render(["A", "B"], [["a" * 49, "b"], ["a", "b" * 2]])
        assert render(["A", "B"], [["a" * 48, "b"], ["a", "b" * 2]])
        # Rows of any iterable are sized all the same.
        with pytest.raises(yt_dlp.utils.ExtractorError):
            render(["A"], (["a" * 51] for _ in range(2)))
    assert guard["_MAX_FORMAT_LABEL_CHARS"] == cap


def test_guard_charges_every_tab_a_table_cell_expands():
    """yt-dlp's renderer replaces EVERY tab of a cell with the column's spare
    width (its own cells hold one, a right-align): two rows of 5 000
    characters, one of them all tabs, drew 25 million for a charge of 10 002."""
    guard = _registered_guard()
    udl.guard_self_check(force=True)
    _reset_budget()
    render = sys.modules["yt_dlp.YoutubeDL"].render_table
    with pytest.raises(yt_dlp.utils.ExtractorError,
                       match="format names are too long"):
        render(["Language", "Name"], [["x" * 5000, "a"], ["\t" * 5000, "b"]])
    assert guard["_DIALS"].tls.table_chars == 0
    with mock.patch.dict(guard, {"_MAX_FORMAT_LABEL_CHARS": 1000}):
        # The header's width counts: a cell of tabs alone takes it per tab.
        with pytest.raises(yt_dlp.utils.ExtractorError,
                           match="format names are too long"):
            render(["H" * 100], [["\t\t"] for _ in range(20)])
        # An admitted table is charged its padded rows plus the widest cell
        # of the column for every tab after a cell's first: no less than its
        # rows are drawn.
        rows = [["a\tb\tc", "x" * 9], ["\t\t\t", 12345], ["wide" * 5, "\t\ty"]]
        drawn = render(["A", "B"], rows).splitlines()
        assert drawn[1] == "a" + " " * 17 + "b" + " " * 17 + "c xxxxxxxxx"
        assert sum(map(len, drawn[1:])) == 47 + 66 + 38
        assert guard["_DIALS"].tls.table_chars == 3 * (20 + 9) + 3 * 20 + 9
    _reset_budget()
    # yt-dlp's own cells hold one tab at most, which is the padding already
    # charged: an ordinary table costs rows x the widest of each column.
    own = [["137", "\t30", "~\t12.34MiB", "\t"] for _ in range(100)]
    assert render(["ID", "FPS", "\tFILESIZE", "\tTBR"], own)
    assert guard["_DIALS"].tls.table_chars == 100 * (3 + 3 + 10 + 1)
    assert guard["_tab_chars"](["ID", "FPS", "\tFILESIZE", "\tTBR"], own) == 0


def test_guard_caps_the_tabs_of_a_subtitle_language_of_a_dash_manifest():
    """The measured shape, through the real parser: a character reference
    keeps a tab in an attribute, and 40 text AdaptationSets whose @lang is
    2 000 of them beside one of 2 000 letters (a 328 kB manifest) drew 160
    million characters, twice. Half the rows and half the width here."""
    from yt_dlp.extractor.common import InfoExtractor
    from yt_dlp.utils import compat_etree_fromstring
    langs = ["x" * 1000] + [f"b{i}" + "&#9;" * 1000 for i in range(20)]
    xml = _mpd(
        '<Period><AdaptationSet mimeType="audio/mp4" codecs="mp4a.40.2">'
        '<Representation id="a" bandwidth="1"><BaseURL>a.m4a</BaseURL>'
        "</Representation></AdaptationSet>" + "".join(
            f'<AdaptationSet mimeType="text/vtt" lang="{lang}">'
            f'<Representation id="t{i}" bandwidth="1"><BaseURL>s{i}.vtt</BaseURL>'
            "</Representation></AdaptationSet>" for i, lang in enumerate(langs))
        + "</Period>")
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        fmts, subs = InfoExtractor(ydl)._parse_mpd_formats_and_subtitles(
            compat_etree_fromstring(xml.encode()),
            mpd_base_url="http://cdn.example/", mpd_url="http://cdn.example/m.mpd")
        assert sum(lang.count("\t") for lang in subs) == 20 * 1000
        with pytest.raises(yt_dlp.utils.ExtractorError,
                           match="format names are too long"):
            ydl.process_ie_result(
                {**_listed(), "formats": fmts, "subtitles": subs}, download=False)


def test_guard_charges_table_draws_across_one_extraction(monkeypatch):
    """Each entry draws its tables afresh at every print point: what one
    extraction draws comes from one budget, which a new extraction resets."""
    guard = _registered_guard()
    udl.guard_self_check(force=True)
    render = sys.modules["yt_dlp.YoutubeDL"].render_table
    monkeypatch.setitem(guard, "_MAX_TABLE_CHARS", 1000)
    rows = [["a" * 30, "b" * 10] for _ in range(10)]
    render(["A", "B"], rows)
    render(["A", "B"], rows)
    with pytest.raises(yt_dlp.utils.ExtractorError,
                       match="format names are too long"):
        render(["A", "B"], rows)
    assert guard["_DIALS"].tls.table_chars == 800
    _reset_budget()
    assert guard["_DIALS"].tls.table_chars == 0
    assert render(["A", "B"], rows)


def test_guard_admits_the_tables_of_real_lists(monkeypatch):
    """200 subtitle languages in 5 formats, as many automatic captions and
    60 thumbnails are a sliver of one table's budget, and 20 000 ordinary
    formats drawn at every print point of a download fit the extraction's."""
    guard = _registered_guard()
    names = dict(formats=("vtt", "ttml", "srv3", "srv2", "json3"),
                 name="English (auto-generated)")
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"quiet": True, "simulate": True}) as ydl:
        res = ydl.process_ie_result(_listed(
            subtitles=_subtitles(200, **names),
            automatic_captions=_subtitles(200, **names),
            thumbnails=_thumbnails(60)), download=False)
        assert len(res["subtitles"]) == 200 and len(res["thumbnails"]) == 60
        assert guard["_DIALS"].tls.table_chars < guard["_MAX_FORMAT_LABEL_CHARS"] // 20
        _reset_budget()
        # The widths of the formats table from a twentieth of the rows:
        # every row is as wide as the next.
        rows = guard["_MAX_EXTRACTION_FORMATS"] // 20
        sized = []
        monkeypatch.setattr(
            sys.modules["yt_dlp.YoutubeDL"], "render_table",
            lambda header, data, **kw: sized.append(guard["_padded_chars"](
                map(guard["_row_chars"], data)) * 20))
        ydl.render_formats_table({"formats": [{
            "format_id": f"video=4500000-{i:05d}", "ext": "mp4",
            "container": "mp4_dash", "width": 3840, "height": 2160, "fps": 59.94,
            "vcodec": "hvc1.2.4.L153.B0", "acodec": "mp4a.40.2", "language": "en-US",
            "format_note": "DASH video", "protocol": "http_dash_segments",
            "tbr": 4500.0, "vbr": 4400.0, "abr": 128.0, "asr": 48000,
            "filesize": 123456789, "audio_channels": 2, "dynamic_range": "HDR10",
            "url": f"https://cdn.example/v/{i}/seg.m4s?sig=" + "s" * 300,
        } for i in range(rows)]})
    assert 0 < sized[0] < guard["_MAX_FORMAT_LABEL_CHARS"]
    # pre_process, after_filter, video, before_dl, post_process, after_move,
    # after_video, playlist: no entry is drawn at more points.
    assert 8 * sized[0] < guard["_MAX_TABLE_CHARS"]


def test_guard_is_not_installed_without_the_table_bound(monkeypatch):
    from yt_dlp.utils import render_table
    module = sys.modules["yt_dlp.YoutubeDL"]
    guard = _inproc_guard()
    assert guard.is_installed()
    assert module.render_table is not render_table
    with monkeypatch.context() as m:
        m.setattr(module, "render_table", render_table)
        assert not guard.is_installed()
    with monkeypatch.context() as m:  # a yt-dlp that renamed the renderer
        m.delattr(module, "render_table")
        assert not guard.is_installed()
        guard.install()  # nothing to bound: still fails closed
        assert not guard.is_installed()
    for name in ("render_formats_table", "render_thumbnails_table",
                 "render_subtitles_table"):
        with monkeypatch.context() as m:  # a list drawn some other way
            m.setattr(module.YoutubeDL, name, lambda self, *a: None)
            assert not guard.is_installed()
        with monkeypatch.context() as m:
            m.delattr(module.YoutubeDL, name)
            assert not guard.is_installed()
    assert guard.is_installed()


# ---- What a page builds before anything counts it: GenericIE parses every
# HTML page it scrapes as XML first, and reads every ld+json block with
# json.loads — up to the 64 MiB body cap, ~1.6 GB of tree or objects. The
# guard counts elements / arrays and objects in C before either parse.

_ETREE_IMPORTERS = (
    "yt_dlp.compat", "yt_dlp.utils", "yt_dlp.utils._utils",
    "yt_dlp.extractor.common", "yt_dlp.extractor.generic",
    "yt_dlp.extractor.facebook", "yt_dlp.extractor.odnoklassniki",
    "yt_dlp.extractor.faz", "yt_dlp.extractor.brightcove",
    "yt_dlp.downloader.f4m")


@pytest.mark.parametrize("module", _ETREE_IMPORTERS)
def test_guard_counts_xml_elements_before_building_the_tree(monkeypatch, module):
    import importlib
    import xml.etree.ElementTree as ET
    udl.guard_self_check(force=True)
    parse = importlib.import_module(module).compat_etree_fromstring
    assert _registered_guard()["_MAX_XML_ELEMENTS"] >= 2 * 1_000_000
    cap = 1000
    monkeypatch.setitem(_registered_guard(), "_MAX_XML_ELEMENTS", cap)
    over = '<?xml version="1.0"?><x>' + "<a/>" * cap + "</x>"
    for doc in (over, over.encode()):
        with pytest.raises(ET.ParseError, match="too many elements"):
            parse(doc)
    at_cap = '<?xml version="1.0"?><x>' + "<a/>" * (cap - 2) + "</x>"
    assert len(parse(at_cap)) == len(parse(at_cap.encode())) == cap - 2


@pytest.mark.parametrize("module", _ETREE_IMPORTERS)
def test_guard_counts_xml_attributes_before_building_the_tree(monkeypatch, module):
    import importlib
    import xml.etree.ElementTree as ET
    udl.guard_self_check(force=True)
    parse = importlib.import_module(module).compat_etree_fromstring
    cap = 1000
    monkeypatch.setitem(_registered_guard(), "_MAX_XML_ELEMENTS", cap)
    # One start tag holding millions of attributes builds its whole dict.
    over = "<a " + " ".join(f'a{i}=""' for i in range(2 * cap)) + "/>"
    for doc in (over, over.encode()):
        with pytest.raises(ET.ParseError, match="too many elements"):
            parse(doc)
    # A DASH SegmentList at the segment cap (half the element cap), two
    # attributes per segment, still parses.
    segs = cap // 2
    mpd = _mpd('<Period><AdaptationSet mimeType="audio/mp4"><Representation '
               'id="r" bandwidth="1"><SegmentList duration="1">'
               + "".join(f'<SegmentURL media="s{i}.m4s" mediaRange="{i}-{i + 9}"/>'
                         for i in range(segs))
               + "</SegmentList></Representation></AdaptationSet></Period>")
    assert len(mpd) > cap
    for doc in (mpd, mpd.encode()):
        assert len(parse(doc).findall(f".//{{{_MPD_NS}}}SegmentURL")) == segs


@pytest.mark.parametrize("module", _ETREE_IMPORTERS)
def test_guard_refuses_xml_entity_declarations(module):
    import importlib
    import xml.etree.ElementTree as ET
    udl.guard_self_check(force=True)
    parse = importlib.import_module(module).compat_etree_fromstring
    # expat builds an entity's markup once per reference, where no count of
    # the raw text sees it: ~100 KB here would build ~2.1 M elements.
    bomb = ('<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "' + "<b/>" * 70
            + '">]><r>' + "&a;" * 30_000 + "</r>")
    for doc in (bomb, bomb.encode(), bytearray(bomb.encode()),
                bomb.encode("utf-16"), bomb.encode("utf-16-le"),
                bomb.encode("utf-16-be")):
        with pytest.raises(ET.ParseError, match="declares entities"):
            parse(doc)
    # A DOCTYPE without entities (an RSS 0.91 feed) still parses.
    rss = ('<?xml version="1.0"?><!DOCTYPE rss PUBLIC '
           '"-//Netscape Communications//DTD RSS 0.91//EN" '
           '"http://my.netscape.com/publish/formats/rss-0.91.dtd">'
           '<rss version="0.91"><channel/></rss>')
    assert parse(rss).tag == parse(rss.encode()).tag == "rss"


@pytest.mark.parametrize("module", _ETREE_IMPORTERS)
def test_guard_refuses_xml_attribute_defaults(module):
    import importlib
    import xml.etree.ElementTree as ET
    udl.guard_self_check(force=True)
    parse = importlib.import_module(module).compat_etree_fromstring
    # expat gives every matching element a fresh copy of an <!ATTLIST
    # default, where no count of the raw text sees it: a 120 KB document of
    # 5 000 elements under a 100 000-character default held 534 MB.
    elements = "<r>" + "<e/>" * 50 + "</r>"
    doc = ('<?xml version="1.0"?><!DOCTYPE r [<!ATTLIST e a CDATA "'
           + "x" * 1000 + '">]>' + elements)
    for spelling in (doc, doc.encode(), bytearray(doc.encode()),
                     doc.encode("utf-16"), doc.encode("utf-16-le"),
                     doc.encode("utf-16-be")):
        with pytest.raises(ET.ParseError, match="attribute defaults"):
            parse(spelling)
    # The same elements with no declaration still parse.
    assert len(parse(elements)) == len(parse(elements.encode())) == 50


_NS_CAP = 256  # the guard's _MAX_XML_NAMESPACE_CHARS, pinned below
_NS_NAMES = 20


def _ns_doc(spelling, chars):
    """A document of _NS_NAMES distinct names under a `chars`-character
    namespace, and the root tag it parses to."""
    uri = "u" * chars
    elements = "".join(f"<e{i}/>" for i in range(_NS_NAMES))
    plain = f'<r xmlns="{uri}">{elements}</r>'
    doc = {
        "str": plain,
        "utf-8": plain.encode(),
        "bytearray": bytearray(plain.encode()),
        "utf-16": ('<?xml version="1.0" encoding="utf-16"?>' + plain).encode("utf-16"),
        "utf-16-le": plain.encode("utf-16-le"),
        "prefixed-elements": f'<r xmlns:p="{uri}">'
                             + "".join(f"<p:e{i}/>" for i in range(_NS_NAMES)) + "</r>",
        "prefixed-attributes": f'<r xmlns:p="{uri}">'
                               + "".join(f'<b p:a{i}="1"/>' for i in range(_NS_NAMES))
                               + "</r>",
        "single-quoted": f"<r xmlns  =\n '{uri}'>{elements}</r>",
        "character-references": '<r xmlns="' + "&#117;" * chars + f'">{elements}</r>',
        "nested": f'<r><x xmlns="{uri}">{elements}</x><y/></r>',
        "second-prefix": f'<r xmlns:a="urn:a" xmlns:b="{uri}">{elements}</r>',
    }[spelling]
    root = "r" if spelling in ("prefixed-elements", "prefixed-attributes", "nested",
                               "second-prefix") else f"{{{uri}}}r"
    return doc, root


@pytest.mark.parametrize("spelling", [
    "str", "utf-8", "bytearray", "utf-16", "utf-16-le", "prefixed-elements",
    "prefixed-attributes", "single-quoted", "character-references", "nested",
    "second-prefix"])
def test_guard_refuses_an_overlong_xml_namespace(spelling):
    """The tree holds a namespace twice per DISTINCT name under it (element
    or prefixed attribute), where no count of the raw text sees it: a 123 KB
    document of 3 000 names under a 100 000-character namespace held 613 MB.
    The parser reports every declaration however it is spelled."""
    import xml.etree.ElementTree as ET
    import yt_dlp.compat
    # What a document at the element cap can still hold is names x 2 x this.
    assert _registered_guard().get("_MAX_XML_NAMESPACE_CHARS", _NS_CAP) == _NS_CAP
    parse = yt_dlp.compat.compat_etree_fromstring
    over, _root = _ns_doc(spelling, _NS_CAP + 1)
    with pytest.raises(ET.ParseError, match="overlong namespace"):
        parse(over)
    at_cap, root = _ns_doc(spelling, _NS_CAP)
    tree = parse(at_cap)
    assert tree.tag == root and len(tree) == (2 if spelling == "nested" else _NS_NAMES)


def _tree_shape(element):
    return (element.tag, sorted(element.attrib.items()), element.text, element.tail,
            [_tree_shape(child) for child in element])


@pytest.mark.parametrize("doc,root", [
    ('<?xml version="1.0"?><MPD xmlns="urn:mpeg:dash:schema:mpd:2011" '
     'xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:cenc="urn:mpeg:cenc:2013" '
     'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
     'xmlns:yt="http://youtube.com/yt/2012/10/10" '
     'xsi:schemaLocation="urn:mpeg:dash:schema:mpd:2011 DASH-MPD.xsd" type="static">'
     '<Period xlink:href="p.xml" xml:lang="en"><AdaptationSet mimeType="audio/mp4">'
     '<ContentProtection schemeIdUri="urn:mpeg:dash:mp4protection:2011" '
     'cenc:default_KID="0"/><Representation id="a" bandwidth="1">'
     '<BaseURL yt:contentLength="12">a.m4a</BaseURL>\n</Representation>'
     "</AdaptationSet></Period></MPD>", "{urn:mpeg:dash:schema:mpd:2011}MPD"),
    ('<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" '
     'xmlns:media="http://search.yahoo.com/mrss/" '
     'xmlns:atom="http://www.w3.org/2005/Atom"><channel><atom:link href="x"/>'
     '<item><title>t</title><media:content url="x"/>'
     "<itunes:duration>1:02</itunes:duration></item></channel></rss>", "rss"),
    ('<smil xmlns="http://www.w3.org/2001/SMIL20/Language"><head>'
     '<meta base="http://cdn.example/"/></head><body><switch>'
     '<video src="a.mp4" system-bitrate="1"/></switch></body></smil>',
     "{http://www.w3.org/2001/SMIL20/Language}smil"),
    ('<SmoothStreamingMedia MajorVersion="2" MinorVersion="2" Duration="1">'
     '<StreamIndex Type="audio" Url="Q({bitrate})/F(a={start time})">'
     '<QualityLevel Bitrate="1" FourCC="AACL"/><c d="1"/></StreamIndex>'
     "</SmoothStreamingMedia>", "SmoothStreamingMedia"),
    ('<manifest xmlns="http://ns.adobe.com/f4m/1.0"><media url="a" bitrate="1"/>'
     "</manifest>", "{http://ns.adobe.com/f4m/1.0}manifest"),
    # A namespace redeclared, and undeclared, further down.
    ('<r xmlns="urn:a"><x xmlns="urn:b"><y/></x><z xmlns=""><w/></z>'
     '<p:q xmlns:p="urn:c" p:a="1"/></r>', "{urn:a}r"),
], ids=["mpd", "rss", "smil", "ism", "f4m", "redeclared"])
def test_guard_admits_real_namespaces(doc, root):
    """The namespace check only watches declarations: the tree yt-dlp gets
    (qualified tags and attributes, text, tails) is what the parser builds
    with no guard at all."""
    import xml.etree.ElementTree as ET
    import yt_dlp.compat
    udl.guard_self_check(force=True)
    assert getattr(yt_dlp.compat._TreeBuilder, "start_ns", None) is not None
    plain = _tree_shape(ET.fromstring(doc))
    assert plain[0] == root
    for spelling in (doc, doc.encode(),
                     doc.replace('<?xml version="1.0"?>', "").encode("utf-16")):
        assert _tree_shape(yt_dlp.compat.compat_etree_fromstring(spelling)) == plain


def test_guard_is_not_installed_without_the_xml_bound(monkeypatch):
    import yt_dlp.compat
    import yt_dlp.extractor.generic as generic
    guard = _inproc_guard()
    assert guard.is_installed()
    with monkeypatch.context() as m:
        # Any loaded module holding an unbounded parse fails closed...
        m.setattr(generic, "compat_etree_fromstring", lambda text: None)
        assert not guard.is_installed()
        guard.install()  # ...until install() bounds it too
        assert guard.is_installed()
    with monkeypatch.context() as m:  # a yt-dlp that dropped the function
        m.delitem(vars(yt_dlp.compat), "compat_etree_fromstring")
        assert not guard.is_installed()
        guard.install()
        assert not guard.is_installed()
    assert guard.is_installed()


def test_guard_is_not_installed_without_the_namespace_bound(monkeypatch):
    import xml.etree.ElementTree as ET
    import yt_dlp.compat
    guard = _inproc_guard()
    assert guard.is_installed()
    with monkeypatch.context() as m:
        m.delattr(yt_dlp.compat._TreeBuilder, "start_ns")
        assert not guard.is_installed()
        guard.install()  # hooks the builder again
        assert guard.is_installed()
    with monkeypatch.context() as m:  # a yt-dlp whose builder watches them itself
        m.setattr(yt_dlp.compat._TreeBuilder, "start_ns",
                  lambda self, prefix, uri: None)
        assert not guard.is_installed()
        guard.install()  # not replaced: still fails closed
        assert not guard.is_installed()
    with monkeypatch.context() as m:  # a yt-dlp that renamed the builder
        m.delitem(vars(yt_dlp.compat), "_TreeBuilder")
        assert not guard.is_installed()
        guard.install()  # nothing to hook: still fails closed
        assert not guard.is_installed()
    with monkeypatch.context() as m:  # a parse that builds with something else
        m.setitem(vars(yt_dlp.compat), "compat_etree_fromstring",
                  lambda text: ET.XML(text))
        guard.install()  # bounded like any other, yet the hook is not on its path
        assert not guard.is_installed()
    assert guard.is_installed()


def _parse_json(doc, *args, **kwargs):
    """InfoExtractor._parse_json on `doc`: (result or raised error, warnings)."""
    from yt_dlp.extractor.common import InfoExtractor
    warnings = []
    logger = types.SimpleNamespace(debug=lambda m: None, info=lambda m: None,
                                   warning=warnings.append, error=lambda m: None)
    udl.guard_self_check(force=True)
    with yt_dlp.YoutubeDL({"logger": logger}) as ydl:
        try:
            out = InfoExtractor(ydl)._parse_json(doc, "x", *args, **kwargs)
        except yt_dlp.utils.ExtractorError as e:
            out = e
    return out, warnings


def test_guard_counts_json_containers_before_parsing(monkeypatch):
    assert _registered_guard()["_MAX_JSON_CONTAINERS"] >= 2 * 1_000_000
    cap = 1000
    monkeypatch.setitem(_registered_guard(), "_MAX_JSON_CONTAINERS", cap)
    over = "[" + "[]," * (cap - 1) + "{}]"
    at_cap = "[" + "[]," * (cap - 2) + "{}]"
    for doc in (over, over.encode()):
        err, warnings = _parse_json(doc)
        assert isinstance(err, yt_dlp.utils.ExtractorError)
        assert "Failed to parse JSON" in str(err) and "too many" in str(err)
    # The refusal follows _parse_json's own contract for JSON it cannot
    # parse: fatal raises, otherwise the same warning (none for
    # errnote=False) and None.
    why = f"the JSON holds too many arrays, objects or items (more than {cap})"
    for args, kwargs, note in (
            ((), {"fatal": False}, "Failed to parse JSON"),
            ((None, False, "boom"), {}, "boom"),
            ((), {"fatal": False, "errnote": False}, None)):
        broken, broken_warnings = _parse_json("[", *args, **kwargs)
        out, warnings = _parse_json(over, *args, **kwargs)
        assert broken is out is None
        assert warnings == [w.split(f"{note}: ")[0] + f"{note}: {why}"
                            for w in broken_warnings]
        assert len(warnings) == (1 if note else 0)
    assert _parse_json(at_cap)[0] == [[]] * (cap - 2) + [{}]
    assert _parse_json(at_cap.encode(), fatal=False)[0] == [[]] * (cap - 2) + [{}]


def test_guard_counts_json_scalars_before_parsing(monkeypatch):
    # A flat array of scalars is one container, yet every item costs a
    # Python object: the items (a ',' each after the first) are charged
    # against twice the container cap.
    cap = 1000
    monkeypatch.setitem(_registered_guard(), "_MAX_JSON_CONTAINERS", cap)
    over = "[" + ",".join(["-9"] * (2 * cap + 2)) + "]"  # 1 + 2*cap + 1
    at_cap = "[" + ",".join(["-9"] * 2 * cap) + "]"      # 1 + 2*cap - 1
    members = "{" + ",".join(f'"a{i}":1' for i in range(2 * cap + 1)) + "}"
    # An object member builds a key and a value, so its ':' is charged too:
    # cap + 1 members, or cap/2 + 200 one-member objects, pass the comma
    # count alone yet build as many objects as the scalar array above.
    flat = "{" + ",".join(f'"a{i}":1' for i in range(cap + 1)) + "}"
    singles = "[" + ",".join('{"k%d":0}' % i for i in range(cap // 2 + 200)) + "]"
    for doc in (over, over.encode(), members, members.encode(),
                flat, flat.encode(), singles, singles.encode()):
        err, _warnings = _parse_json(doc)
        assert isinstance(err, yt_dlp.utils.ExtractorError)
        assert "Failed to parse JSON" in str(err) and "too many" in str(err)
    assert _parse_json(at_cap)[0] == [-9] * 2 * cap
    assert _parse_json(at_cap.encode(), fatal=False)[0] == [-9] * 2 * cap
    # 1 + (cap - 2) + (cap - 1): an object just under the cap still parses.
    under = "{" + ",".join(f'"a{i}":1' for i in range(cap - 1)) + "}"
    assert len(_parse_json(under)[0]) == cap - 1


def test_guard_still_parses_a_page_sized_json_document():
    # A page answer of a few MB with nested objects (the shape of
    # ytInitialData) stays far below both ceilings.
    item = ('{"videoRenderer":{"videoId":"abcdefghijk","title":{"runs":'
            '[{"text":"some title, with a comma"}]},"lengthText":"3:14",'
            '"thumbnails":[{"url":"https://i.example/x.jpg","width":120}]}}')
    doc = '{"contents":[' + ",".join([item] * 20_000) + "]}"
    assert len(doc) > 3_000_000
    out, warnings = _parse_json(doc)
    assert not isinstance(out, Exception) and warnings == []
    assert len(out["contents"]) == 20_000


def test_guard_is_not_installed_without_the_json_bound(monkeypatch):
    from yt_dlp.extractor.common import InfoExtractor
    guard = _inproc_guard()
    assert guard.is_installed()
    with monkeypatch.context() as m:
        m.setattr(InfoExtractor, "_parse_json", lambda self, *a, **kw: None)
        assert not guard.is_installed()
    with monkeypatch.context() as m:  # a yt-dlp that renamed the method
        m.delattr(InfoExtractor, "_parse_json")
        assert not guard.is_installed()
        guard.install()  # nothing to bound: still fails closed
        assert not guard.is_installed()
    assert guard.is_installed()


class _HtmlForYtDlp(_QuietHandler):
    """audio/mpeg for the direct-media check, an HTML page for yt-dlp's own
    GET of the same URL: the shape that hands GenericIE a page to scrape
    with the shipped settings (direct media on, generic pages off)."""

    page = b""

    def do_GET(self):
        if (self.headers.get("User-Agent") or "").startswith("faster-whisper-backend"):
            self._ok_headers("audio/mpeg", 1)
            self.wfile.write(b"\xff")
            return
        self._ok_headers("text/html", len(type(self).page))
        self.wfile.write(type(self).page)


def test_probe_counts_a_scraped_page_before_parsing_it(monkeypatch,
                                                       public_is_public):
    import xml.etree.ElementTree as ET
    import yt_dlp.compat
    from yt_dlp.utils._utils import LenientJSONDecoder
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.2", 0))
    except OSError:  # pragma: no cover — non-Linux loopback aliasing
        pytest.skip("this platform does not alias 127.0.0.2")
    cap = 1000
    udl.guard_self_check(force=True)
    monkeypatch.setitem(_registered_guard(), "_MAX_XML_ELEMENTS", cap)
    monkeypatch.setitem(_registered_guard(), "_MAX_JSON_CONTAINERS", cap)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    ld_json = "[" + "[]," * (2 * cap) + "[]]"
    _HtmlForYtDlp.page = (
        "<html><head><title>t</title>"
        f'<script type="application/ld+json">{ld_json}</script></head>'
        "<body>" + "<a/>" * (2 * cap) + "</body></html>").encode()
    trees, decoded = [], []
    monkeypatch.setattr(yt_dlp.compat, "etree", types.SimpleNamespace(
        XML=lambda text, parser=None: trees.append(len(text)) or ET.XML(text, parser),
        XMLParser=ET.XMLParser))
    real_decode = LenientJSONDecoder.decode
    monkeypatch.setattr(LenientJSONDecoder, "decode",
                        lambda self, s: decoded.append(len(s)) or real_decode(self, s))
    srv = _serve(_HtmlForYtDlp, "127.0.0.2")
    with _stopping(srv), pytest.raises(udl.UrlDownloadError, match="isn't supported"):
        _run(udl.probe(f"http://127.0.0.2:{srv.server_port}/stream", timeout=20))
    # Neither the page nor its ld+json block was handed to a parser.
    assert trees == []
    assert all(n < len(ld_json) for n in decoded), decoded


def test_probe_prunes_a_request_handler_registered_by_plugin_loading(
        tmp_path, monkeypatch):
    """YoutubeDL() loads yt-dlp's default plugin dirs (every sys.path entry,
    the user's config dir) in-process, and a RequestHandler a plugin
    registers there would sit behind the guard and take whatever the guard
    declines. The probe must build its director from the guard alone, as the
    download child (--no-plugin-dirs) does."""
    from yt_dlp.networking.common import _REQUEST_HANDLERS
    plugin = tmp_path / "yt_dlp_plugins" / "extractor"
    plugin.mkdir(parents=True)
    (plugin / "fwb_test_stray.py").write_text(
        "from yt_dlp.networking._urllib import UrllibRH\n"
        "from yt_dlp.networking.common import register_rh\n\n\n"
        "@register_rh\n"
        "class FwbTestStrayRH(UrllibRH):\n"
        "    RH_NAME = 'fwb-test-stray'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(yt_dlp.globals.all_plugins_loaded, "value", False)
    seen = {}

    def _extract_info(self, url, download=False, **kw):
        seen["handlers"] = sorted(
            h.RH_NAME for h in self._request_director.handlers.values())
        return {"title": "t", "extractor_key": "Youtube"}

    async def _policy(url, *, deadline=None):
        return "Youtube"
    monkeypatch.setattr(yt_dlp.YoutubeDL, "extract_info", _extract_info)
    monkeypatch.setattr(udl, "check_url_policy", _policy)
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    try:
        _run(udl.probe("https://example.com/watch?v=x", timeout=10.0))
    finally:
        _REQUEST_HANDLERS.pop("FwbTestStray", None)
        for name in [n for n in sys.modules if n.endswith(".fwb_test_stray")]:
            sys.modules.pop(name)
    assert seen["handlers"] == ["fwb-guarded-urllib"]


def test_probe_fails_closed_when_plugin_loading_leaves_the_guard_broken(
        monkeypatch):
    udl.guard_self_check(force=True)
    monkeypatch.setattr(udl._guard_module, "is_installed", lambda: False)
    called = []

    async def _policy(url, *, deadline=None):
        return "Youtube"
    monkeypatch.setattr(yt_dlp.YoutubeDL, "extract_info",
                        lambda self, *a, **kw: called.append(a))
    monkeypatch.setattr(udl, "check_url_policy", _policy)
    monkeypatch.setattr(udl.cfg, "URL_ALLOWED_EXTRACTORS", [], raising=False)
    with pytest.raises(udl.UrlDownloadError,
                       match="link downloads are unavailable on this server"):
        _run(udl.probe("https://example.com/watch?v=x", timeout=10.0))
    assert called == []


# --- ffmpeg / ffprobe on a downloaded file ----------------------------------
# yt-dlp's postprocessors spawn ffmpeg / ffprobe on the downloaded file, whose
# bytes the site chose (an HLS download is its segments concatenated). A file
# that is really a DASH manifest opens ffmpeg's dash demuxer, which dials the
# BaseURL: the guard puts -format_whitelist before every input.

def _whitelist_after(argv, at):
    assert argv[at:at + 2][0] == "-format_whitelist", argv
    return set(argv[at + 1].split(","))


def _captured_ffmpeg_argvs(monkeypatch, tmp_path):
    """Every argv the ffmpeg postprocessor module spawns, with the spawn
    refused (OSError) before anything runs."""
    from yt_dlp.postprocessor import ffmpeg as ffmpeg_pp
    from yt_dlp.utils import Popen as base_popen
    _inproc_guard()
    argvs = []

    def _record(self, args, *a, **kw):
        argvs.append(list(args))
        raise OSError("spawn refused by the test")
    monkeypatch.setattr(base_popen, "__init__", _record)
    pp = ffmpeg_pp.FFmpegPostProcessor(None)
    # Pretend both binaries are present; nothing is ever executed.
    pp._paths = {"ffmpeg": "/opt/ff/ffmpeg-custom", "ffprobe": "/opt/ff/ffprobe"}
    pp.__dict__.update(basename="ffmpeg", probe_basename="ffprobe",
                       _version="8.0", _probe_version="8.0", _features={})
    a, b = tmp_path / "v.mp4", tmp_path / "a.webm"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    with pytest.raises(OSError):
        pp.run_ffmpeg_multiple_files([str(a), str(b)], str(tmp_path / "o.mkv"),
                                     ["-c", "copy", "-map", "0:v:0", "-map", "1:a:0"])
    with pytest.raises(OSError):
        pp.get_metadata_object(str(a))
    assert pp.get_audio_codec(str(a)) is None  # ffprobe -show_streams
    pp.__dict__["probe_basename"] = None       # no ffprobe: ffmpeg -i
    assert pp.get_audio_codec(str(a)) is None
    return argvs


def test_guard_whitelists_the_format_of_every_ffmpeg_postprocessor_input(
        monkeypatch, tmp_path):
    merge, metadata, codec_probe, codec_ffmpeg = _captured_ffmpeg_argvs(
        monkeypatch, tmp_path)
    for argv in (merge, codec_ffmpeg):
        inputs = [i for i, arg in enumerate(argv) if arg == "-i"]
        assert inputs, argv
        assert argv.count("-format_whitelist") == len(inputs)
        for i in inputs:
            allowed = _whitelist_after(argv, i - 4)
    assert len([a for a in merge if a == "-i"]) == 2
    for argv in (metadata, codec_probe):  # ffprobe: one input, options first
        assert argv[0] == "/opt/ff/ffprobe"
        assert _whitelist_after(argv, 3) == allowed
        assert argv[-1].startswith("file:")
    assert {"mov", "mp4", "m4a", "matroska", "webm", "ogg", "mpegts", "flv",
            "mp3", "aac", "flac", "wav"} <= allowed
    # The demuxers that open other URLs or files never make the list.
    assert not allowed & {"dash", "hls", "concat", "webm_dash_manifest",
                          "sdp", "rtsp", "rtp", "image2", "imf", "tee"}


def test_guard_caps_what_every_ffmpeg_postprocessor_spawn_opens(
        monkeypatch, tmp_path):
    """The file is the site's bytes, and the mov demuxer builds its index for
    every sample a track DECLARES: a 4.6 KB file made `ffmpeg -i`, ffprobe
    and the HLS fixup each peak at 620 MB, forty tracks at 770 MB. Every
    spawn carries the server's own allocation cap and stream limit."""
    from faster_whisper_backend.audio import ffmpeg as audio_ffmpeg
    guard = _inproc_guard()
    # The guard cannot import the package in the download child, so the two
    # numbers are repeated there.
    assert guard._FFMPEG_MAX_ALLOC == audio_ffmpeg._LIBAV_MAX_ALLOC
    assert guard._FFMPEG_MAX_STREAMS == audio_ffmpeg.PYAV_MAX_STREAMS
    alloc = ["-max_alloc", str(audio_ffmpeg._LIBAV_MAX_ALLOC)]
    streams = ["-max_streams", str(audio_ffmpeg.PYAV_MAX_STREAMS)]
    merge, metadata, codec_probe, codec_ffmpeg = _captured_ffmpeg_argvs(
        monkeypatch, tmp_path)
    for argv in (merge, metadata, codec_probe, codec_ffmpeg):
        assert argv[1:3] == alloc, argv  # a global option: once, first
        assert argv.count("-max_alloc") == 1
    for argv in (merge, codec_ffmpeg):  # an input option: before every -i
        inputs = [i for i, arg in enumerate(argv) if arg == "-i"]
        assert argv.count("-max_streams") == len(inputs) > 0
        assert all(argv[i - 2:i] == streams for i in inputs), argv
    for argv in (metadata, codec_probe):  # ffprobe: ahead of its one input
        assert argv.count("-max_streams") == 1
        assert argv[5:7] == streams, argv
    # A command with no input (a version or feature listing) gets no input
    # option.
    assert guard._whitelisted_ffmpeg_argv(["/opt/ff/ffmpeg", "-bsfs"]) == [
        "/opt/ff/ffmpeg", *alloc, "-bsfs"]


def _peak_kb_once_open(argv, fifo):
    """The peak resident kB (VmHWM) of an ffmpeg at the moment it has opened
    its first input, or None when it refused that input for lack of memory.
    Its second input is `fifo`, which it opens next and then waits on, alive,
    while its /proc status is read: the number is the ffmpeg's own (a peak
    taken from wait4 is never below what the forking process held)."""
    import time
    os.mkfifo(fifo)
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    writer = None
    try:
        deadline = time.monotonic() + 60
        while proc.poll() is None and time.monotonic() < deadline:
            try:  # ENXIO until ffmpeg has the FIFO open for reading
                writer = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError:
                time.sleep(0.005)
        if writer is None:
            proc.kill()
            assert b"Cannot allocate memory" in proc.stderr.read(), argv
            return None
        with open(f"/proc/{proc.pid}/status", encoding="ascii") as f:
            return int(re.search(r"VmHWM:\s+(\d+) kB", f.read()).group(1))
    finally:
        if writer is not None:
            os.close(writer)  # end of file: ffmpeg gives the input up
        proc.communicate(timeout=60)


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="reads the child's peak RSS from /proc")
@pytest.mark.parametrize("which", ["default", "imageio"])
def test_ffmpeg_postprocessors_bound_a_declared_sample_count(which, tmp_path):
    """The real binaries honour the cap: the 16 million samples a 4.6 KB
    file declares cost the merger's ffmpeg 400 to 600 MB of index at its
    first input without it. With it the file opens without that index or is
    refused."""
    from tests.audio.test_transcode import _mov_declaring_samples
    exe = _ffmpeg_binary(which)
    guard = _inproc_guard()
    src = _mov_declaring_samples(str(tmp_path / "declared.mov"), 16_000_000)
    assert os.path.getsize(src) < 5000
    fifo = str(tmp_path / "second-input")
    argv = guard._whitelisted_ffmpeg_argv(
        [exe, "-y", "-i", "file:" + src, "-i", "file:" + fifo])
    peak_kb = _peak_kb_once_open(argv, fifo)
    assert peak_kb is None or peak_kb < 300 * 1024


def test_guard_is_not_installed_without_the_ffmpeg_input_whitelist(monkeypatch):
    from yt_dlp.postprocessor import ffmpeg as ffmpeg_pp
    from yt_dlp.utils import Popen as base_popen
    guard = _inproc_guard()
    assert guard.is_installed()
    with monkeypatch.context() as m:
        m.setattr(ffmpeg_pp, "Popen", base_popen)
        assert not guard.is_installed()
    # A yt-dlp whose spawner stops going through the module's Popen.
    for name in ("real_run_ffmpeg", "get_audio_codec", "get_metadata_object"):
        with monkeypatch.context() as m:
            m.setattr(ffmpeg_pp.FFmpegPostProcessor, name,
                      lambda self, *a, **kw: subprocess.run(a))
            assert not guard.is_installed()
    with monkeypatch.context() as m:  # a yt-dlp without the module-level name
        m.delattr(ffmpeg_pp, "Popen")
        guard.install()  # nothing to wrap: still fails closed
        assert not guard.is_installed()
    assert guard.is_installed()


def test_guard_refuses_a_non_argv_ffmpeg_command():
    guard = _inproc_guard()
    with pytest.raises(ValueError, match="fwb-ssrf-guard"):
        guard._whitelisted_ffmpeg_argv("ffmpeg -i file:x")


def _ffmpeg_binary(which: str) -> str:
    if which == "default":
        from faster_whisper_backend.audio.ffmpeg import ffmpeg_exe
        return ffmpeg_exe()
    try:
        import imageio_ffmpeg  # noqa: PLC0415 — optional bundled binary
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        pytest.skip("imageio-ffmpeg binary unavailable")


@pytest.mark.parametrize("which", ["default", "imageio"])
def test_ffmpeg_postprocessors_never_follow_a_downloaded_dash_manifest(
        which, tmp_path):
    """ffmpeg 7.0.2 (the imageio-ffmpeg fallback) opens an MPD saved as
    media.mp4 with its dash demuxer and dials the BaseURL, from the HLS
    fixup, the merger and the codec probe alike."""
    from yt_dlp.postprocessor.ffmpeg import (
        FFmpegFixupM3u8PP, FFmpegMergerPP, FFmpegPostProcessor,
    )
    exe = _ffmpeg_binary(which)
    _inproc_guard()
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    srv.settimeout(0.1)
    port = srv.getsockname()[1]
    hits = []
    stop = threading.Event()

    def accept():
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except (socket.timeout, OSError):
                continue
            hits.append(1)
            conn.close()

    th = threading.Thread(target=accept, daemon=True)
    th.start()
    mpd = (
        '<?xml version="1.0"?>\n'
        '<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" type="static" '
        'mediaPresentationDuration="PT1S" minBufferTime="PT1S" '
        'profiles="urn:mpeg:dash:profile:isoff-on-demand:2011">\n'
        f'<BaseURL>http://127.0.0.1:{port}/</BaseURL>\n'
        '<Period><AdaptationSet mimeType="audio/mp4">'
        '<Representation id="a" bandwidth="64000" codecs="mp4a.40.2">'
        '<BaseURL>seg.mp4</BaseURL></Representation></AdaptationSet></Period>'
        '</MPD>\n')

    def fresh(name):
        path = tmp_path / name
        path.write_text(mpd, encoding="utf-8")
        return str(path)
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                               "ffmpeg_location": exe}) as ydl:
            try:
                FFmpegFixupM3u8PP(ydl).run({"ext": "mp4", "protocol": "m3u8_native",
                                            "filepath": fresh("media.mp4")})
            except yt_dlp.utils.PostProcessingError:
                pass  # the file is not media: failing is the right outcome
            v, a = fresh("v.mp4"), fresh("a.mp4")
            with pytest.raises(yt_dlp.utils.PostProcessingError):
                FFmpegMergerPP(ydl).run({
                    "filepath": str(tmp_path / "out.mkv"), "ext": "mkv",
                    "__files_to_merge": [v, a],
                    "requested_formats": [
                        {"acodec": "none", "vcodec": "h264", "protocol": "https",
                         "filepath": v},
                        {"acodec": "aac", "vcodec": "none",
                         "protocol": "m3u8_native", "filepath": a}]})
            assert FFmpegPostProcessor(ydl).get_audio_codec(fresh("c.mp4")) is None
    finally:
        stop.set()
        th.join(timeout=2)
        srv.close()
    assert hits == []


def _encode(exe, out, *args):
    r = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y", *args,
                        str(out)], capture_output=True, timeout=60)
    if r.returncode != 0 or not out.exists():
        pytest.skip(f"{out.name} encode unavailable")


@pytest.mark.parametrize("which", ["default", "imageio"])
def test_ffmpeg_postprocessors_still_fix_and_merge_a_real_download(
        which, tmp_path):
    """The containers a download yields still open: an HLS download's
    MPEG-TS saved as mp4 is remuxed, a video and an audio format merge. The
    imageio-ffmpeg fallback takes the guard's options too (it ships no
    ffprobe, so its codec probe is `ffmpeg -i`); its 7.0.2 build crashes on
    an MPEG-TS input whatever the options, so it only merges here."""
    from yt_dlp.postprocessor.ffmpeg import (
        FFmpegFixupM3u8PP, FFmpegMergerPP, FFmpegPostProcessor,
    )
    exe = _ffmpeg_binary(which)
    _inproc_guard()
    tone = ["-f", "lavfi", "-i", "sine=frequency=440:duration=1"]
    media = tmp_path / "media.mp4"
    if which == "default":
        _encode(exe, media, *tone, "-c:a", "aac", "-f", "mpegts")
    v, a = tmp_path / "v.mp4", tmp_path / "a.webm"
    _encode(exe, v, "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
            "-c:v", "mpeg4")
    _encode(exe, a, *tone, "-c:a", "libopus")
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                           "ffmpeg_location": exe}) as ydl:
        pp = FFmpegPostProcessor(ydl)
        if which == "default":
            assert pp.get_audio_codec(str(media)) == "aac"
            FFmpegFixupM3u8PP(ydl).run({"ext": "mp4", "protocol": "m3u8_native",
                                        "filepath": str(media)})
            assert media.read_bytes()[4:8] == b"ftyp"  # remuxed out of MPEG-TS
        out = tmp_path / "out.mkv"
        FFmpegMergerPP(ydl).run({
            "filepath": str(out), "ext": "mkv",
            "__files_to_merge": [str(v), str(a)],
            "requested_formats": [
                {"acodec": "none", "vcodec": "mpeg4", "protocol": "https",
                 "filepath": str(v)},
                {"acodec": "opus", "vcodec": "none", "protocol": "https",
                 "filepath": str(a)}]})
        assert pp.get_audio_codec(str(out)) == "opus"
    assert out.read_bytes()[:4] == b"\x1aE\xdf\xa3"  # EBML: Matroska
