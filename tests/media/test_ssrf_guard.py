"""The yt-dlp SSRF guard (ytdlp_plugins/) — the finding it closes, live.

url_download's own probes already refused private / loopback / link-local /
metadata addresses on every hop, but yt-dlp fetched with its own opener: a
host that looks public to the probe could 302 `extract_info` and the download
subprocess into an internal address, and nothing checked that hop.

These tests run the real thing against two loopback HTTP servers:

  PUBLIC   127.0.0.2  — stands in for the attacker's public host. It answers
                        the direct-media probe (User-Agent
                        "faster-whisper-backend") with 200 audio/mpeg and
                        redirects everyone else — i.e. yt-dlp — to INTERNAL.
  INTERNAL 127.0.0.1  — stands in for 169.254.169.254 / a LAN service. Every
                        request it receives is a guard failure.

THE ONLY PATCH is the repro's: the literal 127.0.0.2 is treated as a public
address. In-process that is a monkeypatch of net_policy.address_is_forbidden;
for the download subprocess it is a copy of the guard tree next to a patched
copy of net_policy.py (the guard loads net_policy relative to its own
location, so the copy is what the child enforces). 127.0.0.1 is judged by the
real, unpatched policy, and yt-dlp itself is never patched.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from unittest import mock

import pytest

from faster_whisper_backend.core import net_policy
from faster_whisper_backend.media import download as udl

SECRET = b"INTERNAL-SECRET-" * 64
PUBLIC_BODY = b"\xff\xfb\x90\x44" + b"\x00" * 4092  # plausible MPEG audio head

from faster_whisper_backend.paths import REPO_ROOT


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# the two servers
# ---------------------------------------------------------------------------

class _Internal(http.server.BaseHTTPRequestHandler):
    hits: "list[str]" = []

    def do_GET(self):
        type(self).hits.append(f"{self.command} {self.path}")
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Content-Length", str(len(SECRET)))
        self.end_headers()
        self.wfile.write(SECRET)

    do_HEAD = do_GET

    def log_message(self, *a):
        pass


class _Public(http.server.BaseHTTPRequestHandler):
    """200 audio/mpeg for the direct-media probe, a redirect for everyone
    else — exactly the shape that slipped past the pre-guard code."""

    redirect_to = ""

    def do_GET(self):
        ua = self.headers.get("User-Agent") or ""
        if ua.startswith("faster-whisper-backend") or self.path.startswith("/direct"):
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(PUBLIC_BODY)))
            self.end_headers()
            if self.command == "GET":
                self.wfile.write(PUBLIC_BODY)
            return
        self.send_response(302)
        self.send_header("Location", type(self).redirect_to)
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_HEAD = do_GET

    def log_message(self, *a):
        pass


def _serve(handler, host):
    srv = http.server.ThreadingHTTPServer((host, 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture
def servers():
    """(public_url_base, internal_url) with a clean INTERNAL hit log."""
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.2", 0))
    except OSError:  # pragma: no cover — non-Linux loopback aliasing
        pytest.skip("this platform does not alias 127.0.0.2")
    internal = _serve(_Internal, "127.0.0.1")
    _Internal.hits = []
    _Public.redirect_to = f"http://127.0.0.1:{internal.server_port}/secret.mp3"
    public = _serve(_Public, "127.0.0.2")
    try:
        yield (f"http://127.0.0.2:{public.server_port}",
               f"http://127.0.0.1:{internal.server_port}/secret.mp3")
    finally:
        public.shutdown()
        internal.shutdown()
        public.server_close()
        internal.server_close()


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
        (open(os.path.join(REPO_ROOT, "faster_whisper_backend", "core", "net_policy.py"),
              encoding="utf-8").read())
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


# ---------------------------------------------------------------------------
# (a) registration
# ---------------------------------------------------------------------------

def test_guard_registers_in_process():
    udl.guard_self_check(force=True)
    from yt_dlp.networking.common import _REQUEST_HANDLERS

    assert _REQUEST_HANDLERS["FwbSsrfGuard"].RH_NAME == "fwb-guarded-urllib"
    # The point is not that ours exists but that nothing UNGUARDED is left to
    # fall back to when ours declines a request (data:/ftp:, impersonation).
    for superseded in ("Urllib", "Requests", "CurlCFFI", "Websockets"):
        assert superseded not in _REQUEST_HANDLERS


def test_guard_prefers_over_every_builtin():
    udl.guard_self_check(force=True)
    import yt_dlp

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
    import yt_dlp
    from yt_dlp.downloader import external, hls

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
    udl.guard_self_check(force=True)
    guard = sys.modules["fwb_ssrf_guard_inproc"]
    from yt_dlp.downloader import external, rtmp
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
    udl.guard_self_check(force=True)
    guard = sys.modules["fwb_ssrf_guard_inproc"]
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
    udl.guard_self_check(force=True)
    guard = sys.modules["fwb_ssrf_guard_inproc"]
    from yt_dlp.networking.exceptions import RequestError
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
    assert udl.GUARD_UNAVAILABLE_MARKER not in proc.stderr  # no fail-closed
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
    import yt_dlp
    assert argv[-2] == "--"
    return yt_dlp.parse_options(argv[2:-2]).ydl_opts


def test_pinned_download_flags_parse_on_the_pinned_yt_dlp():
    """--ignore-config, --downloader and --use-extractors (with a
    regex-escaped IE name) must be accepted by the pinned yt-dlp, the
    extractor regexes must compile and the -f selector must parse, or every
    link download fails. Done in-process: yt-dlp only validates the values
    when it builds a YoutubeDL, which `--version` never reaches."""
    import re

    import yt_dlp
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


# ---------------------------------------------------------------------------
# (b) the finding: a redirect into an internal address, both fetch paths
# ---------------------------------------------------------------------------

def test_probe_refuses_redirect_into_loopback(servers, public_is_public,
                                              monkeypatch):
    public, _ = servers
    udl.guard_self_check(force=True)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_GENERIC", False, raising=False)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    with pytest.raises(udl.UrlDownloadError) as ei:
        _run(udl.probe(f"{public}/x.mp3", timeout=20))
    # Client-safe wording, and never the address the guard actually saw.
    assert "could not be reached" in str(ei.value)
    assert "127.0.0.1" not in str(ei.value)
    assert _Internal.hits == []


def test_download_refuses_redirect_into_loopback(tmp_path, servers,
                                                 public_is_public, guard_tree):
    public, _ = servers
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(udl.UrlDownloadError) as ei:
        _run(udl.download(f"{public}/x.mp3", dest_dir=str(dest),
                          max_bytes=10_000_000, timeout=60))
    assert "could not be reached" in str(ei.value)
    assert "127.0.0.1" not in str(ei.value)
    assert _Internal.hits == []
    assert os.listdir(dest) == []


# ---------------------------------------------------------------------------
# (c) the happy path still works through the guarded handler
# ---------------------------------------------------------------------------

def test_download_allows_a_public_looking_target(tmp_path, servers,
                                                 public_is_public, guard_tree):
    public, _ = servers
    dest = tmp_path / "dest"
    dest.mkdir()
    out = _run(udl.download(f"{public}/direct.mp3", dest_dir=str(dest),
                            max_bytes=10_000_000, timeout=60))
    assert os.path.getsize(out) == len(PUBLIC_BODY)
    assert _Internal.hits == []


class _Dribbler(http.server.BaseHTTPRequestHandler):
    """200 audio/mpeg for the direct-media probe at once; yt-dlp's page GET
    gets its status line and then one header byte every 0.3 s — never
    tripping a per-op socket timeout, never finishing."""

    def do_GET(self):
        ua = self.headers.get("User-Agent") or ""
        if ua.startswith("faster-whisper-backend"):
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(PUBLIC_BODY)))
            self.end_headers()
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

    def log_message(self, *a):
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
    try:
        results = _run(_four())
        assert all(isinstance(r, udl.UrlTimeoutError) for r in results), results
        # Every worker is free again shortly after the deadline.
        futs = [udl._PROBE_POOL.submit(lambda: 1) for _ in range(4)]
        assert [f.result(timeout=5) for f in futs] == [1] * 4
    finally:
        srv.shutdown()
        srv.server_close()


# ---------------------------------------------------------------------------
# (d) no data: / ftp: escape hatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", [
    "data:audio/mpeg;base64,//uQxAAAAAAAAAAAAAAAAAAAAAAA",
    "ftp://127.0.0.1:1/secret.mp3",
    "file:///etc/passwd",
])
def test_probe_refuses_non_http_redirect(target, servers, public_is_public,
                                         monkeypatch, caplog):
    public, _ = servers
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


# ---------------------------------------------------------------------------
# (e) fail closed
# ---------------------------------------------------------------------------

@pytest.fixture
def broken_guard(monkeypatch):
    monkeypatch.setattr(udl, "_guard_ok", False)
    monkeypatch.setattr(udl, "GUARD_MODULE",
                        os.path.join(REPO_ROOT, "no-such-guard.py"))


def test_probe_refuses_when_the_guard_cannot_install(broken_guard, caplog,
                                                     monkeypatch):
    async def _policy(url):
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
            _run(udl.download("https://example.com/v", dest_dir=str(tmp_path),
                              max_bytes=1000, timeout=5))
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


# ---------------------------------------------------------------------------
# (f) one definition of the address policy
# ---------------------------------------------------------------------------

def test_url_download_uses_net_policy_directly():
    assert udl._host_is_forbidden is net_policy.host_is_forbidden
    assert udl._CGNAT_NET is net_policy.CGNAT_NET


def test_guard_uses_the_same_net_policy_module():
    udl.guard_self_check(force=True)
    guard = sys.modules["fwb_ssrf_guard_inproc"]
    # Same file, and in-process literally the same module object.
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
    udl.guard_self_check(force=True)
    guard = sys.modules["fwb_ssrf_guard_inproc"]
    assert guard.EXTERNAL_FD_MARKER == udl.GUARD_EXTERNAL_FD_MARKER
    from yt_dlp.downloader import external
    with pytest.raises(Exception) as ei:
        guard._refuse_external_fd(external.FFmpegFD.__new__(external.FFmpegFD),
                                  "f", {})
    assert (udl.classify_error(f"ERROR: {ei.value}")
            == "this media needs a downloader the server does not allow")


# --- an operator-configured proxy is a trusted hop -------------------------

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
    # and SNI still names the target (the earlier proxy test's invariant).
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
    udl.guard_self_check(force=True)
    guard = sys.modules["fwb_ssrf_guard_inproc"]
    from yt_dlp.networking.exceptions import RequestError

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

    udl.guard_self_check(force=True)
    guard = sys.modules["fwb_ssrf_guard_inproc"]
    handler = guard._GuardedHTTPHandler(context=None, source_address=None)
    handler.http_open(plain)
    assert built["conn"].via_proxy is False
    handler.http_open(proxied)
    assert built["conn"].via_proxy is True
