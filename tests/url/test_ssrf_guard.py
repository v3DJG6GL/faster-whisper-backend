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
import os
import shutil
import socket
import subprocess
import sys
import threading
from unittest import mock

import pytest

from faster_whisper_backend.core import net_policy
from faster_whisper_backend.url import download as udl

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
    for superseded in ("Urllib", "Requests", "CurlCFFI"):
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


def test_launcher_starts_yt_dlp_with_the_guard_flags():
    """The exact flags build_download_argv uses must be valid on the pinned
    yt-dlp, and the launcher must not fail closed on a healthy tree."""
    argv = udl.build_download_argv("https://example.com/a", dest_dir=".",
                                   max_bytes=1)
    proc = subprocess.run(
        argv[:5] + ["--version"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert udl.GUARD_MARKER not in proc.stderr  # no fail-closed, no traceback
    assert "Error while importing module" not in proc.stderr


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


# ---------------------------------------------------------------------------
# (d) no data: / ftp: escape hatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", [
    "data:audio/mpeg;base64,//uQxAAAAAAAAAAAAAAAAAAAAAAA",
    "ftp://127.0.0.1:1/secret.mp3",
    "file:///etc/passwd",
])
def test_probe_refuses_non_http_redirect(target, servers, public_is_public,
                                         monkeypatch):
    public, _ = servers
    _Public.redirect_to = target
    udl.guard_self_check(force=True)
    monkeypatch.setattr(udl.cfg, "URL_ALLOW_DIRECT_MEDIA", True, raising=False)
    with pytest.raises(udl.UrlDownloadError) as ei:
        _run(udl.probe(f"{public}/x.mp3", timeout=20))
    assert "could not be reached" in str(ei.value)


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
    monkeypatch.setattr(udl, "_guard_ok", False)
    monkeypatch.setattr(udl, "GUARD_MODULE",
                        os.path.join(REPO_ROOT, "no-such-guard.py"))
    with pytest.raises(udl.UrlDownloadError):
        udl.guard_self_check()
    monkeypatch.undo()  # the tree is back
    udl.guard_self_check(force=True)


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


def test_guard_copy_agrees_with_net_policy_address_by_address(guard_tree):
    """The subprocess loads net_policy BY PATH; prove the file it reaches is
    the repo's, verdict for verdict, so the two halves can never drift."""
    child = subprocess.run(
        [sys.executable, "-c",
         "import importlib.util,sys,json\n"
         f"spec=importlib.util.spec_from_file_location('np', {os.path.join(os.path.dirname(guard_tree), 'faster_whisper_backend', 'core', 'net_policy.py')!r})\n"
         "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
         "addrs=json.loads(sys.argv[1])\n"
         "print(json.dumps([m.address_is_forbidden(a) for a in addrs]))",
         __import__("json").dumps(_ADDRESS_CORPUS)],
        capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr
    import json
    child_verdicts = json.loads(child.stdout)
    ours = [net_policy.address_is_forbidden(a) for a in _ADDRESS_CORPUS]
    # 127.0.0.2 is the copy's one deliberate difference (see guard_tree).
    for addr, mine, theirs in zip(_ADDRESS_CORPUS, ours, child_verdicts):
        if addr == "127.0.0.2":
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
