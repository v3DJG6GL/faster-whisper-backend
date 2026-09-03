"""SSRF guard for every fetch yt-dlp makes, in-process and in the subprocess.

WHY: url_download's own probes refuse private / loopback / link-local /
metadata addresses on every hop, but the two fetches that actually move bytes
— probe()'s `extract_info` and download()'s `python -m yt_dlp` — used yt-dlp's
own opener, which follows redirects and re-resolves DNS with no policy at all.
A "public" host could therefore 302 the downloader into 169.254.169.254 or a
LAN service, and the guarded probe never saw that hop.

WHAT this module installs (import side effect, idempotent — see install()):

  * a RequestHandler that outranks every built-in one (preference 1000 beats
    curl_cffi/requests/urllib), so it serves every http(s) request yt-dlp
    makes; and
  * the removal of the built-in http(s) handlers from yt-dlp's registry, so
    nothing can fall through to an unguarded opener when ours declines a
    request (a `data:`/`ftp:`/`file:` URL, or an extractor asking for TLS
    impersonation). Websocket handlers are left alone — they speak ws(s),
    which this download path never uses for media bytes.

The handler enforces, with url_download's policy (net_policy — ONE
definition, see below):

  (a) hop 0 AND every redirect hop are validated: scheme must be http(s) and
      the host must not resolve to a forbidden address;
  (b) DNS is pinned: the name is resolved once, the socket dials the resolved
      literal, and Host / SNI / certificate validation keep the ORIGINAL
      hostname — so a second, attacker-timed answer cannot land us somewhere
      the policy already refused (DNS rebinding);
  (c) only http and https exist. No data:, ftp:, file: — as a target or as a
      redirect destination.

Every error message carries the marker `fwb-ssrf-guard`, which
url_download.classify_error maps to the client-safe "the site could not be
reached from the server". The offending host/address appears only in the
message that the server logs at WARNING (yt-dlp stderr tail / probe failure),
never in what a client is handed.

Layout note: this file lives at the yt-dlp plugin path
`<plugin dir>/<pkg>/yt_dlp_plugins/extractor/<name>.py`, so
`--plugin-dirs <repo>/ytdlp_plugins` loads it. Registering a RequestHandler
from an "extractor" plugin module is an import side effect, not an official
plugin type — it is the only hook yt-dlp offers a packaged plugin, and it is
why ytdlp_plugins/run_guarded_yt_dlp.py also imports this file directly:
yt-dlp's plugin loader SWALLOWS import errors (prints a traceback and carries
on unguarded), which is precisely the failure this guard must not survive.
"""
from __future__ import annotations

import http.client
import importlib.util
import os
import socket
import sys
import urllib.parse
import urllib.request

from yt_dlp.networking import _urllib
from yt_dlp.networking.common import (
    _REQUEST_HANDLERS, register_preference, register_rh,
)
from yt_dlp.networking.exceptions import RequestError

# The string url_download.classify_error keys on. Changing it means changing
# the taxonomy entry there too.
MARKER = "fwb-ssrf-guard"

GUARD_RH_KEY = "FwbSsrfGuard"
RH_NAME = "fwb-guarded-urllib"

# Built-in handlers that can speak http(s) (or data/ftp/file). They are
# unregistered by install() so ours is the only way out of the process.
_SUPERSEDED_RH_KEYS = ("Urllib", "Requests", "CurlCFFI")


# ── the shared policy ───────────────────────────────────────────────────────
# net_policy.py is the single definition of "which addresses we refuse". It is
# imported normally when the repo root is on sys.path (the in-process probe),
# and loaded BY PATH otherwise (the download subprocess, whose sys.path
# deliberately excludes the repo root — repo-root directories would shadow
# stdlib modules for yt-dlp's extractors). Either way there is one file.
def _load_net_policy():
    try:
        from faster_whisper_backend.core import net_policy  # noqa: PLC0415 — optional fast path
        return net_policy
    except ImportError:
        pass
    # <repo>/ytdlp_plugins/fwb_ssrf_guard/yt_dlp_plugins/extractor/<this file>
    #   parents:            4          3              2          1      0
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             *([os.pardir] * 4)))
    path = os.path.join(repo_root, "faster_whisper_backend", "core", "net_policy.py")
    spec = importlib.util.spec_from_file_location("fwb_net_policy", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{MARKER}: net_policy.py not found at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("fwb_net_policy", module)
    spec.loader.exec_module(module)
    return module


net_policy = _load_net_policy()


def _resolve_pinned(host: str, port: int):
    """Resolve `host` ONCE and return the candidate sockaddrs, having refused
    the name outright if ANY of its answers is a forbidden address.

    Returning the resolved list (rather than re-resolving at connect time) is
    the pin: the socket dials one of exactly these addresses."""
    if not host:
        raise RequestError(f"{MARKER}: request has no host")
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise RequestError(f"{MARKER}: {host} did not resolve ({e})") from None
    if not infos:
        raise RequestError(f"{MARKER}: {host} did not resolve")
    for info in infos:
        if net_policy.address_is_forbidden(info[4][0]):
            raise RequestError(
                f"{MARKER}: {host} resolves to the forbidden address "
                f"{info[4][0]} — refusing to fetch")
    return infos


def _check_url(url: str) -> None:
    """Scheme + host gate for one hop (hop 0 or a redirect target)."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        raise RequestError(
            f"{MARKER}: refusing a non-http(s) URL (scheme {parts.scheme!r})")
    _resolve_pinned(parts.hostname or "", parts.port or 0)


# ── pinned connections ──────────────────────────────────────────────────────
# Only connect() is overridden: everything else (the Host header, the request
# line, chunking) still sees the original hostname, which is what makes
# name-based virtual hosting and certificate validation keep working.

def _connect_pinned(conn) -> "socket.socket":
    infos = _resolve_pinned(conn.host, conn.port)
    last = None
    for family, socktype, proto, _canon, sockaddr in infos:
        sock = socket.socket(family, socktype, proto)
        try:
            if conn.timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:  # type: ignore[attr-defined]
                sock.settimeout(conn.timeout)
            if conn.source_address:
                sock.bind(conn.source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as e:
            sock.close()
            last = e
    raise last if last is not None else OSError("connection failed")


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = _connect_pinned(self)
        if self._tunnel_host:
            self._tunnel()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        sock = _connect_pinned(self)
        if self._tunnel_host:
            self.sock = sock
            self._tunnel()
            sock = self.sock
        # server_hostname = the NAME, never the pinned literal: SNI and cert
        # verification must still be about the host the URL named.
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _GuardedHTTPHandler(_urllib.HTTPHandler):
    """yt-dlp's HTTPHandler with the connection classes swapped for pinned
    ones. A SOCKS proxy is refused rather than silently unguarded: the
    connection would then be to the proxy and the real target resolved on the
    far side, where this policy cannot reach."""

    def _make_conn_class(self, base, req):  # type: ignore[override]
        if req.headers.pop("Ytdl-socks-proxy", None):
            raise RequestError(
                f"{MARKER}: a SOCKS proxy would bypass the address policy")
        return base

    def http_open(self, req):
        conn_class = self._make_conn_class(_PinnedHTTPConnection, req)
        return self.do_open(
            lambda *a, **kw: _urllib._create_http_connection(
                conn_class, self._source_address, *a, **kw), req)

    def https_open(self, req):
        conn_class = self._make_conn_class(_PinnedHTTPSConnection, req)
        return self.do_open(
            lambda *a, **kw: _urllib._create_http_connection(
                conn_class, self._source_address, *a, context=self._context,
                **kw), req)


class _GuardedRedirectHandler(_urllib.RedirectHandler):
    """Re-run the full gate on every redirect hop.

    A refusal is raised as RequestError, NOT as urllib.error.HTTPError: yt-dlp
    rewrites an HTTPError carrying a response body into its own HTTPError
    built from the response, which would throw our message (and its marker)
    away and leave the caller with a bare "HTTP Error 302". RequestError
    travels out of opener.open() untouched."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            _check_url(newurl)
        except RequestError:
            # urllib only drains/closes fp once redirect_request RETURNS, so
            # the abort path has to close it here.
            try:
                fp.close()
            except Exception:  # noqa: BLE001 — never mask the refusal
                pass
            raise
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class GuardedUrllibRH(_urllib.UrllibRH):
    """The only handler left that can reach the network."""

    RH_KEY = GUARD_RH_KEY
    RH_NAME = RH_NAME
    # Deliberately narrower than UrllibRH's ('http', 'https', 'data', 'ftp'):
    # the download path only ever needs http(s), and the omitted schemes are
    # each a way to read something that is not a public web resource.
    _SUPPORTED_URL_SCHEMES = ("http", "https")

    def _create_instance(self, proxies, cookiejar, legacy_ssl_support=None):
        # A copy of UrllibRH's opener with the pinned HTTP handler and the
        # guarded redirect handler, and WITHOUT DataHandler / FTPHandler /
        # UnknownHandler / FileHandler.
        opener = urllib.request.OpenerDirector()
        for handler in (
            _urllib.ProxyHandler(proxies),
            _GuardedHTTPHandler(
                debuglevel=int(bool(self.verbose)),
                context=self._make_sslcontext(
                    legacy_ssl_support=legacy_ssl_support),
                source_address=self.source_address),
            _urllib.HTTPCookieProcessor(cookiejar),
            _urllib.HTTPDefaultErrorHandler(),
            _urllib.HTTPErrorProcessor(),
            _GuardedRedirectHandler(),
        ):
            opener.add_handler(handler)
        # Same reason as upstream: drop urllib's default User-Agent so it
        # cannot apply where our own handler didn't run.
        opener.addheaders = []
        return opener

    def _send(self, request):
        _check_url(request.url)  # hop 0 — redirects are gated above
        return super()._send(request)


def install() -> None:
    """Register the guard and retire the handlers it supersedes.

    Idempotent, and keyed on GUARD_RH_KEY rather than on this module's own
    class object: the same file legitimately loads twice under two module
    names (directly, from run_guarded_yt_dlp.py / guard_self_check, and again
    as `yt_dlp_plugins.extractor.fwb_ssrf_guard` when --plugin-dirs points
    here), which yields two distinct-but-equivalent classes. Re-registering
    would trip register_rh's assert, and yt-dlp's plugin loader would swallow
    that into a stderr traceback."""
    if GUARD_RH_KEY not in _REQUEST_HANDLERS:
        register_rh(GuardedUrllibRH)
        register_preference(GuardedUrllibRH)(_prefer_guard)
    for key in _SUPERSEDED_RH_KEYS:
        _REQUEST_HANDLERS.pop(key, None)


def _prefer_guard(rh, request):
    # Beats requests (100), urllib (0) and curl_cffi (-100) for every request.
    return 1000


def is_installed() -> bool:
    """True when the guard is the only http(s)-capable handler registered."""
    rh = _REQUEST_HANDLERS.get(GUARD_RH_KEY)
    return (rh is not None and getattr(rh, "RH_NAME", None) == RH_NAME
            and not any(k in _REQUEST_HANDLERS for k in _SUPERSEDED_RH_KEYS))


install()
