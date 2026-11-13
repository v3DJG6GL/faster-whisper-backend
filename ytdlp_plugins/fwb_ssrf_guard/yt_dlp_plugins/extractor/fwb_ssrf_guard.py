"""SSRF guard for every fetch yt-dlp makes, in-process and in the subprocess.

WHY: faster_whisper_backend/media/download.py's own probes refuse private / loopback / link-local /
metadata addresses on every hop, but the two fetches that actually move bytes
— probe()'s `extract_info` and download()'s yt-dlp subprocess — used yt-dlp's
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
    impersonation) — and of the Websockets handler, which dials ws(s) with
    no address policy, DNS pin or redirect gate. Extraction does open
    websockets (FC2 live, niconico live), but every format that needs one
    ends in FFmpegFD, which is refused below anyway; and
  * a refusal in every downloader that hands the fetch to another program
    (ExternalFD — ffmpeg, aria2c, curl, wget… — and RtmpFD's rtmpdump). Those
    open URLs themselves, outside the RequestHandler registry: HlsFD alone
    delegates an AES-128 manifest to ffmpeg when pycryptodomex is missing, and
    ffmpeg then fetches the key URI and every segment with no address policy.
    Merging and the fixups still run (FFmpeg postprocessors on local files);
    and
  * a format whitelist on every input of every ffmpeg / ffprobe those
    postprocessors spawn: the file's bytes are the site's (an HLS download
    is its segments concatenated verbatim), and a file that is really a DASH
    manifest or an HLS playlist would otherwise open ffmpeg's dash / hls
    demuxer, which fetches the URLs inside it (see _bound_ffmpeg_inputs);
    and an allocation cap and a stream limit on the same spawns, as a small
    file can declare gigabytes of index.

The handler enforces, with download.py's policy (net_policy — ONE
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
download.classify_error maps to the client-safe "the site could not be
reached from the server" — except the external-downloader refusal, whose
longer EXTERNAL_FD_MARKER maps to "this media needs a downloader the server
does not allow". The offending host/address appears only in the
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
import inspect
import math
import os
import re
import socket
import sys
import threading
import types
import urllib.parse
import urllib.request
import xml.etree.ElementTree
import zlib

from yt_dlp.networking import _urllib
from yt_dlp.networking.common import (
    _REQUEST_HANDLERS, register_preference, register_rh,
)
from yt_dlp.networking.exceptions import RequestError

# The string faster_whisper_backend/media/download.py classify_error keys on
# (as download.GUARD_MARKER). Changing it means changing the taxonomy entry
# there too.
MARKER = "fwb-ssrf-guard"
# The external-downloader refusal's own needle (download.GUARD_EXTERNAL_FD_MARKER):
# the site WAS reached, so it must not read as "could not be reached".
EXTERNAL_FD_MARKER = MARKER + "-external-fd"

GUARD_RH_KEY = "FwbSsrfGuard"
RH_NAME = "fwb-guarded-urllib"

# Built-in handlers that can speak http(s) (or data/ftp/file), and ws(s).
# Documentation of what install() retires today: it unregisters EVERY handler
# other than ours (an allowlist), so a handler a future yt-dlp adds under a new
# key cannot become an unguarded way out of the process.
_SUPERSEDED_RH_KEYS = ("Urllib", "Requests", "CurlCFFI", "Websockets")


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
    # dirname(__file__) = <repo>/ytdlp_plugins/fwb_ssrf_guard/yt_dlp_plugins/extractor,
    # so four pardirs reach <repo>.
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


def _resolve_pinned(host: str, port: int, *, trusted: bool = False):
    """Resolve `host` ONCE and return the candidate sockaddrs, having refused
    the name outright if ANY of its answers is a forbidden address.

    Returning the resolved list (rather than re-resolving at connect time) is
    the pin: the socket dials one of exactly these addresses.

    `trusted=True` is for the operator's own http(s)_proxy: the socket goes
    to the proxy, which may legitimately sit on a private address, so the
    address policy is not applied to it. The URL's target is still gated by
    name on hop 0 and every redirect (_check_url); the pin cannot reach past
    a proxy because the proxy does the final resolve."""
    if not host:
        raise RequestError(f"{MARKER}: request has no host")
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError) as e:  # `a..com`: IDNA can't encode it
        raise RequestError(f"{MARKER}: {host} did not resolve ({e})") from None
    if not infos:
        raise RequestError(f"{MARKER}: {host} did not resolve")
    if trusted:
        return infos
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
    try:
        port = parts.port or 0
    except ValueError:  # a redirect / extractor URL never passed validate_url
        raise RequestError(f"{MARKER}: invalid port") from None
    _resolve_pinned(parts.hostname or "", port)


def _dials_a_proxy(conn) -> bool:
    # HTTPS CONNECT tunnel (stdlib sets _tunnel_host) or plain-HTTP proxying
    # (via_proxy stamped by _GuardedHTTPHandler): conn.host is the proxy.
    return bool(getattr(conn, "via_proxy", False)
                or getattr(conn, "_tunnel_host", None))


# download.probe() bounds its in-process extract_info by WALL CLOCK: it hands
# observe_dials() a callback that records every socket this thread dials, and
# shuts them all at its deadline (a per-op socket timeout cannot stop a host
# that dribbles one header byte per op). Kept in sys.modules, not in this
# module: the file loads more than once per process and only the FIRST
# load's handler is registered (see install()), so every load must share it.
_DIALS = sys.modules.setdefault("fwb_ssrf_guard_dials",
                                types.ModuleType("fwb_ssrf_guard_dials"))
if not hasattr(_DIALS, "tls"):
    _DIALS.tls = threading.local()


def observe_dials(observer, *, deadline: "float | None" = None) -> None:
    """Report every connection this thread opens from now on to
    ``observer(sock)`` (the TLS-wrapped socket for https); None stops.

    `deadline` (time.monotonic()) bounds each dial loop too: the observer
    only sees a socket once its connect returns, and a name with many
    answers that drop the SYN would otherwise spend the full socket timeout
    on each of them."""
    _DIALS.tls.observer = observer
    _DIALS.tls.deadline = deadline if observer is not None else None
    # probe() calls this as its extraction starts and again as it ends, so
    # a pool thread's next extraction starts from a full manifest budget.
    reset_budget()


def reset_budget() -> None:
    """Start this thread's manifest budget (see _charge) afresh."""
    _DIALS.tls.segments = 0
    _DIALS.tls.variants = 0
    _DIALS.tls.f4m_nested = 0
    _DIALS.tls.url_bytes = 0
    _DIALS.tls.processed_formats = 0
    _DIALS.tls.line_work = 0
    _DIALS.tls.label_chars = 0
    _DIALS.tls.cookie_chars = 0
    _DIALS.tls.table_chars = 0
    _DIALS.tls.walk = 0


def _observe(sock) -> None:
    observer = getattr(_DIALS.tls, "observer", None)
    if observer is not None:
        observer(sock)


def _connect_pinned(conn) -> "socket.socket":
    infos = _resolve_pinned(conn.host, conn.port, trusted=_dials_a_proxy(conn))
    deadline = getattr(_DIALS.tls, "deadline", None)
    last = None
    for family, socktype, proto, _canon, sockaddr in infos:
        timeout = net_policy.dial_timeout(conn.timeout, deadline)
        sock = socket.socket(family, socktype, proto)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:  # type: ignore[attr-defined]
                sock.settimeout(timeout)
            if conn.source_address:
                sock.bind(conn.source_address)
            sock.connect(sockaddr)
            if timeout is not conn.timeout:  # reads keep the per-op timeout
                sock.settimeout(net_policy.undeadlined_timeout(conn.timeout))
        except OSError as e:
            sock.close()
            last = e
            continue
        try:  # stdlib's connect() sets this too; the request write must not Nagle
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        return sock
    raise last if last is not None else OSError("connection failed")


class _PinnedHTTPConnection(http.client.HTTPConnection):
    via_proxy = False

    def connect(self):
        self.sock = _connect_pinned(self)
        _observe(self.sock)
        if self._tunnel_host:
            self._tunnel()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    via_proxy = False

    def connect(self):
        sock = _connect_pinned(self)
        if self._tunnel_host:
            self.sock = sock
            self._tunnel()
            sock = self.sock
        # server_hostname = the NAME, never the pinned literal: SNI and cert
        # verification must still be about the host the URL named. Behind a
        # CONNECT proxy self.host is the proxy; the URL's host is _tunnel_host
        # (same rule as stdlib HTTPSConnection.connect).
        self.sock = self._context.wrap_socket(
            sock, server_hostname=self._tunnel_host or self.host)
        _observe(self.sock)


# The most a Content-Encoding may expand a response body to. yt-dlp inflates
# a compressed response whole, in memory, before anything reads it — in
# process that is the server's own probe — so unbounded, ~1 MB on the wire
# becomes ~1 GiB. Extractor pages, JSON APIs, player JS and manifests are a
# few MB (a long live DASH manifest some tens), and media fetches send
# Accept-Encoding: identity, so 64 MiB leaves every legitimate body as is.
# The same bound applies to a body read whole (_bound_whole_reads).
_MAX_DECODED_BYTES = 64 * 1024 * 1024
# The most Content-Encoding layers a response may stack. yt-dlp decodes one
# layer per comma token, each bounded on its own above, so without a layer
# count a few hundred KB can buy minutes of CPU. Servers send one coding; a
# double-gzipping proxy sends two.
_MAX_CODINGS = 2
_DECODED_CODINGS = ("gzip", "deflate", "br")
# Chunk size of a bounded whole-body read (see _bound_whole_reads).
_WHOLE_READ_CHUNK = 1024 * 1024


def _expands_past_the_cap() -> RequestError:
    return RequestError(
        f"{MARKER}: a compressed response expands past "
        f"{_MAX_DECODED_BYTES} bytes — refusing to decode it")


def _body_past_the_cap() -> RequestError:
    return RequestError(
        f"{MARKER}: a response body runs past {_MAX_DECODED_BYTES} bytes "
        "— refusing to read it whole")


def _bound_whole_reads(resp) -> None:
    """Make a size-less resp.read() stop past _MAX_DECODED_BYTES.

    yt-dlp reads webpages, manifests, API answers and the raw body its
    decoders get with a bare read(), whole, in memory — so a host that once
    answered the direct-media probe with media and then serves an endless
    page grows the reader without limit. Sized reads (HttpFD / FragmentFD
    read(block_size) for media) pass straight through. A body whose
    Content-Length fits is read by the original call, so its errors
    (IncompleteRead on a short body) stay exactly as they were."""
    read = resp.read

    def bounded_read(amt=None):
        if amt is not None and amt >= 0:
            return read(amt)
        cap = _MAX_DECODED_BYTES
        length = getattr(resp, "length", None)
        if length is not None and not getattr(resp, "chunked", False):
            if length > cap:
                resp.close()
                raise _body_past_the_cap()
            return read()
        parts, total = [], 0
        while chunk := read(min(_WHOLE_READ_CHUNK, cap + 1 - total)):
            parts.append(chunk)
            total += len(chunk)
            if total > cap:
                resp.close()
                raise _body_past_the_cap()
        return b"".join(parts)

    resp.read = bounded_read


def _inflate(data: bytes, wbits: int) -> bytes:
    """zlib.decompress(data, wbits) that stops past the cap. Like the one-shot
    call it decodes ONE stream (trailing bytes are ignored) and raises
    zlib.error on a truncated one."""
    d = zlib.decompressobj(wbits)
    out = d.decompress(data, _MAX_DECODED_BYTES + 1)
    if len(out) > _MAX_DECODED_BYTES:
        raise _expands_past_the_cap()
    if not d.eof:  # under the cap, so all input was consumed: truncated
        raise zlib.error(
            "Error -5 while decompressing data: incomplete or truncated stream")
    return out


def _unbrotli(data: bytes) -> bytes:
    """brotli.decompress(data) that stops past the cap (Brotli >= 1.2's
    output_buffer_limit). A module without that limit is refused, not
    inflated unbounded."""
    brotli = _urllib.brotli
    d = brotli.Decompressor()
    try:
        out = d.process(data, output_buffer_limit=_MAX_DECODED_BYTES + 1)
    except TypeError:  # brotlicffi / Brotli < 1.2: no output limit
        raise RequestError(
            f"{MARKER}: the installed brotli module cannot bound its output "
            "— refusing to decode") from None
    if len(out) > _MAX_DECODED_BYTES:
        raise _expands_past_the_cap()
    if not d.is_finished():  # brotli.decompress fails a truncated stream too
        raise brotli.error("brotli: decoder failed")
    return out


class _GuardedHTTPHandler(_urllib.HTTPHandler):
    """yt-dlp's HTTPHandler with the connection classes swapped for pinned
    ones, the Content-Encoding decoders and whole-body reads bounded
    (_MAX_DECODED_BYTES) and the number of codings capped (_MAX_CODINGS). A
    SOCKS proxy is refused rather than silently unguarded: the connection
    would then be to the proxy and the real target resolved on the far side,
    where this policy cannot reach."""

    # The inherited http_response calls these for gzip / deflate / br; the
    # empty-body and deflate's zlib-wrapped fallback match yt-dlp's own.
    @staticmethod
    def gz(data):
        if not data:
            return data
        return _inflate(data, zlib.MAX_WBITS | 16)

    @staticmethod
    def deflate(data):
        if not data:
            return data
        try:
            return _inflate(data, -zlib.MAX_WBITS)
        except zlib.error:
            return _inflate(data, zlib.MAX_WBITS)

    @staticmethod
    def brotli(data):
        if not data:
            return data
        return _unbrotli(data)

    def http_response(self, req, resp):  # type: ignore[override]
        # The same first-header lookup the inherited decoder loop reads, so
        # the count and the decoding cannot disagree.
        codings = [e.strip() for e in
                   resp.headers.get("Content-encoding", "").split(",")]
        if sum(c in _DECODED_CODINGS for c in codings) > _MAX_CODINGS:
            resp.close()
            raise RequestError(
                f"{MARKER}: a response stacks more than {_MAX_CODINGS} "
                "content codings — refusing to decode it")
        _bound_whole_reads(resp)
        return super().http_response(req, resp)

    # The parent's `https_response = http_response` binds the PARENT's
    # function, so the override has to be re-bound here.
    https_response = http_response

    def _make_conn_class(self, base, req):  # type: ignore[override]
        if req.headers.pop("Ytdl-socks-proxy", None):
            raise RequestError(
                f"{MARKER}: a SOCKS proxy would bypass the address policy")
        return base

    def _build(self, conn_class, req, *a, **kw):
        hc = _urllib._create_http_connection(
            conn_class, self._source_address, *a, **kw)
        # ProxyHandler rewrote req to point at the proxy: the address policy
        # must trust that hop (the target was gated by name in _check_url).
        hc.via_proxy = req.has_proxy()
        return hc

    def http_open(self, req):
        conn_class = self._make_conn_class(_PinnedHTTPConnection, req)
        return self.do_open(
            lambda *a, **kw: self._build(conn_class, req, *a, **kw), req)

    def https_open(self, req):
        conn_class = self._make_conn_class(_PinnedHTTPSConnection, req)
        return self.do_open(
            lambda *a, **kw: self._build(conn_class, req, *a,
                                         context=self._context, **kw), req)


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
        self._refuse_foreign_proxy(request)
        _check_url(request.url)  # hop 0 — redirects are gated above
        response = super()._send(request)
        # HlsFD joins every segment line onto the URL its playlist ended at
        # after redirects, which only the response knows (see
        # _refuse_oversized_hls_media).
        _DIALS.tls.response_url_len = len(getattr(response, "url", None) or "")
        return response

    def _refuse_foreign_proxy(self, request):
        """Only the operator's proxies (self.proxies: 'proxy' param / env)
        may carry a request. The proxy hop skips the address policy, and a
        per-request proxy can come from page data: YoutubeDL.urlopen turns a
        `Ytdl-Request-Proxy` header into request.proxies (clean_proxies), and
        MediasiteCatalog copies a header name AND value off the fetched page.
        A per-request map that names the operator's own proxy again (yt-dlp's
        PO-token helper does) or none at all (`None` = dial direct, which the
        address policy gates) trusts nothing new and passes."""
        operator = {url for key, url in self.proxies.items() if key != "no"}
        for key, url in (request.proxies or {}).items():
            if key != "no" and url is not None and url not in operator:
                raise RequestError(
                    f"{MARKER}: a per-request proxy would bypass the address "
                    "policy")


# Sentinel on the replacement function, not on this module's own object: the
# file loads twice under two module names (see install()), and the second
# load must recognise the first one's patch.
_FD_SENTINEL = "_fwb_ssrf_refused"


def _refuse_external_fd(self, filename, info_dict):
    from yt_dlp.utils import DownloadError
    raise DownloadError(
        f"{EXTERNAL_FD_MARKER}: refusing {type(self).__name__}: an external "
        "downloader fetches outside the address policy")


setattr(_refuse_external_fd, _FD_SENTINEL, True)


def _refusing_fd_classes():
    from yt_dlp.downloader import external, rtmp
    # FFmpegFD inherits ExternalFD.real_download; it is listed so that a
    # yt-dlp release giving it its own real_download fails is_installed().
    return (external.ExternalFD, external.FFmpegFD, rtmp.RtmpFD)


def _refuse_external_downloaders() -> None:
    from yt_dlp.downloader import external, rtmp
    for cls in (external.ExternalFD, rtmp.RtmpFD):
        if not getattr(cls.real_download, _FD_SENTINEL, False):
            cls.real_download = _refuse_external_fd


# DASH: yt-dlp's InfoExtractor._parse_mpd_periods expands every
# SegmentTemplate ($Number$: ceil(period length / segment duration)) and
# SegmentTimeline (1 + S@r per entry) into one fragment dict per segment, in
# process and with no cap, so a ~300-byte manifest can ask for billions (the
# probe worker grows by ~260 bytes per segment until the OOM killer stops it).
# The manifest's segments are counted, across every representation, before
# the expansion. A 4 h stream at 2 s segments is 7 200 per representation, a
# 12 h one with ten representations 216 000; the cap (~260 MB of fragment
# dicts at most) leaves long multi-quality VODs well inside it. Smooth
# Streaming (_parse_ism_formats_and_subtitles: <c r="N"> expanded once per
# QualityLevel), an HLS media playlist (HlsFD: one fragment per segment
# line) and an HDS bootstrap (F4mFD) count against the same cap. The DASH,
# Smooth Streaming, HLS master and set-level HDS manifests one extraction
# parses also spend from one running total (_charge), as every manifest within
# its own cap is not enough.
_MAX_MPD_SEGMENTS = 1_000_000
_MPD_SENTINEL = "_fwb_mpd_bounded"
# Each Representation also becomes its own format (periods merge only when
# every field but the fragments matches), and yt-dlp's parse + format pass
# cost grows faster than linearly in their number: 20 000 one-segment
# representations take ~20 s of CPU, a 64 MiB manifest hours. A 12 h stream
# split into 720 ad periods of ten representations each is 7 200. A Smooth
# Streaming QualityLevel is a variant too, from the same budget. A manifest's
# Periods and its AdaptationSets are each held to the same number: one
# without a Representation is no format and is walked all the same, here and
# by yt-dlp (1.9 million empty Periods, 59 MB decoded, took 30 s of CPU).
_MAX_MPD_REPRESENTATIONS = 20_000
# For every Representation yt-dlp also searches the children of the MPD, of
# its Period and (twice) of its AdaptationSet for a BaseURL / ContentProtection
# and copies the set's attributes; for every Smooth Streaming QualityLevel the
# children of its StreamIndex (the chunks) and of the manifest (a Protection
# element). Elements that are none of those are searched all the same, so the
# work is variants x siblings while every count above stays within its cap:
# 1 000 Representations beside 1.9 million empty elements (7.6 MB) took 36 s
# of CPU. Those searches are sized from the child counts (~9 ns a child) and
# spend from one total per extraction: ~10 s at the cap, the order of what the
# variant cap admits on its own (all 20 000 Representations in one
# AdaptationSet is 8e8). A 12 h stream of 720 ad Periods x 10 Representations
# is ~5e6.
_MAX_MANIFEST_WALK = 2 ** 30
# yt-dlp builds every fragment's address as a fresh copy of the manifest's
# template, so its cost is the template's length (with every substitution),
# not the ~260 bytes the segment cap assumes: a 300-byte MPD asking for
# `$Number%010000000d$` builds 10 MB per segment, a 1 MB StreamIndex@Url
# 1 MB per fragment. The addresses one extraction would build are sized
# arithmetically (never built) and spend from one running total, which a
# 12 h ten-quality VOD (216 000 x ~200 bytes, ~43 MB) leaves far inside.
_MAX_FRAGMENT_URL_BYTES = 256 * 1024 * 1024
# One DASH template after its $RepresentationID$ substitutions: yt-dlp walks
# it a character at a time and copies it several times per representation
# even when it expands to no segment. Real templates, signed queries
# included, stay within a few kilobytes.
_MAX_TEMPLATE_BYTES = 64 * 1024
# The width / precision in `$Number%05d$`: real manifests pad to 5..9 digits.
_MAX_TEMPLATE_WIDTH = 64
# The fields, and the `%` conversions inside `$` runs, of one template: real
# ones have one to three. Counting a 60 KiB template of 7 500 fields took
# ~2 ms, once per representation.
_MAX_TEMPLATE_FIELDS = 64
# yt-dlp's two substitutions: `$Number$` -> `%(Number)d` and
# `$Number%05d$` -> `%(Number)05d` (Time and Bandwidth likewise).
_TEMPLATE_FIELD = re.compile(r"\$(?:Number|Bandwidth|Time)(?:%([^$]+))?\$")
# The text between two `$` that _TEMPLATE_FIELD reads as a field.
_TEMPLATE_FIELD_RUN = re.compile(r"(?:Number|Bandwidth|Time)(?:%[^$]+)?")
_DIGIT_RUN = re.compile(r"\d+")
# What Python's % reads after a conversion's "%" and before its type
# character: a mapping key, flags, a width, a precision, a length modifier.
_CONVERSION_HEAD = re.compile(
    r"(?:\([^)]*\)?)?[-#0 +]*(?:[0-9]+|\*)?(?:\.(?:[0-9]+|\*)?)?[hlL]?")
_ISM_BITRATE = re.compile(r"{[Bb]itrate}")
_ISM_START_TIME = re.compile(r"{start[ _]time}")


def _too_many_segments():
    from yt_dlp.utils import ExtractorError
    return ExtractorError(
        f"the stream lists too many segments (more than {_MAX_MPD_SEGMENTS})",
        expected=True)


def _too_many_variants(cap: int):
    from yt_dlp.utils import ExtractorError
    return ExtractorError(
        f"the stream lists too many variants (more than {cap})", expected=True)


def _too_much_url():
    from yt_dlp.utils import ExtractorError
    return ExtractorError("the stream's segment addresses are too long",
                          expected=True)


def _too_much_walk():
    from yt_dlp.utils import ExtractorError
    return ExtractorError("the stream manifest is too large", expected=True)


def _charge(segments: int, variants: int, url_bytes: int = 0,
            walk: int = 0) -> None:
    """Spend a manifest's counts (the bytes of the fragment addresses it
    builds, and the children yt-dlp re-walks for its variants) from the
    budget of the extraction running on this thread. One
    extraction can parse many manifests into one result (GenericIE parses an
    MPD per Video.js `src`), so each manifest within its own cap is not
    enough. A refused manifest spends nothing. probe()
    resets the budget per extraction (observe_dials); the download child
    extracts once per process."""
    total = getattr(_DIALS.tls, "segments", 0) + segments
    if total > _MAX_MPD_SEGMENTS:
        raise _too_many_segments()
    kinds = getattr(_DIALS.tls, "variants", 0) + variants
    if kinds > _MAX_MPD_REPRESENTATIONS:
        raise _too_many_variants(_MAX_MPD_REPRESENTATIONS)
    spent = getattr(_DIALS.tls, "url_bytes", 0) + url_bytes
    if spent > _MAX_FRAGMENT_URL_BYTES:
        raise _too_much_url()
    walked = getattr(_DIALS.tls, "walk", 0) + walk
    if walked > _MAX_MANIFEST_WALK:
        raise _too_much_walk()
    _DIALS.tls.segments = total
    _DIALS.tls.variants = kinds
    _DIALS.tls.url_bytes = spent
    _DIALS.tls.walk = walked


def _number(value, kind=float):
    """`value` as yt-dlp parses it, or None where yt-dlp's own parse raises
    on it (it then fails on its own, after the count)."""
    try:
        return kind(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _digits(value: int) -> int:
    """An upper bound on len(str(value)), sign included, without the
    conversion (Python refuses str() past 4 300 digits anyway)."""
    return abs(value).bit_length() * 30103 // 100000 + 2


# Smooth Streaming's times, durations, timescales and bitrates and DASH's
# S@t/@d/@r, @timescale and @startNumber are 64-bit at most: 20 digits.
# yt-dlp parses them and does big-integer arithmetic and a str() per fragment
# on what it builds from them, ~1 ms a fragment at 4 000 digits. DASH's
# @bandwidth goes into every fragment's address through $Bandwidth$; a
# Representation's @width / @height / @audioSamplingRate / @frameRate (a
# whole number reaches int(), a fraction like 30000/1001 does not) and a
# QualityLevel's sizes, sample rate, channels, bit depth and NAL length field
# are int()-ed once per format, a cost on top of the variant cap. No real one
# comes near 20.
_MAX_NUMBER_CHARS = 20


def _refuse_long_number(value) -> None:
    if isinstance(value, str) and len(value.strip()) > _MAX_NUMBER_CHARS:
        from yt_dlp.utils import ExtractorError
        raise ExtractorError("a number in the stream manifest is too long",
                             expected=True)


def _mpd_segment_info(element, parent: dict, q) -> dict:
    """The counting half of yt-dlp's extract_multisegment_info: a level's
    SegmentList / SegmentTemplate overrides what it inherits."""
    info = dict(parent)
    source = element.find(q("SegmentList"))
    is_list = source is not None
    if not is_list:
        source = element.find(q("SegmentTemplate"))
    if source is None:
        return info
    timeline = source.find(q("SegmentTimeline"))
    entries = timeline.findall(q("S")) if timeline is not None else []
    if entries:
        for e in entries:
            for name in ("t", "d", "r"):
                _refuse_long_number(e.get(name))
        info["s"] = [(_number(e.get("d"), int) or 0,
                      max(_number(e.get("r", 0), int) or 0, 0)) for e in entries]
        # The largest $Time$ the timeline can reach: a start, plus every
        # segment's duration after it.
        info["s_time"] = (
            max(abs(_number(e.get("t", 0), int) or 0) for e in entries)
            + sum(abs(d) * (1 + r) for d, r in info["s"]))
    start = source.get("startNumber")
    _refuse_long_number(start)
    if start:
        info["start"] = _number(start, int)
    timescale = source.get("timescale")
    _refuse_long_number(timescale)
    if timescale:
        info["timescale"] = _number(timescale, int)
    duration = source.get("duration")
    if duration:
        info["duration"] = _number(duration)
    if is_list:
        urls = source.findall(q("SegmentURL"))
        if urls:
            info["urls"] = len(urls)
    else:
        # Kept as text: their length is what every fragment address costs.
        if source.get("media"):
            info["media"] = source.get("media")
        if source.get("initialization"):
            info["initialization"] = source.get("initialization")
    return info


def _representation_segments(info: dict, period_duration):
    """(segments yt-dlp would build for one representation, their total
    seconds or None) — the same branches as _parse_mpd_periods."""
    timescale = info.get("timescale") or 0
    if "urls" not in info and info.get("media") and "s" not in info:
        if not info.get("duration") or not timescale or not period_duration:
            return 0, None
        seconds = info["duration"] / timescale
        n = period_duration / seconds
        if not math.isfinite(n):
            return _MAX_MPD_SEGMENTS + 1, None
        n = max(math.ceil(n), 0)
        return n, n * seconds
    if "s" in info:
        n = sum(1 + r for _d, r in info["s"])
        total = sum(d * (1 + r) for d, r in info["s"])
        return n, (total / timescale if timescale else None)
    if "urls" in info:
        n = info["urls"]
        seconds = (info["duration"] / timescale
                   if info.get("duration") and timescale else None)
        return n, (n * seconds if seconds else None)
    return 0, None


def _spec_width(spec: str) -> int:
    """The widest width / precision a conversion's text can spell: any of its
    numbers. Refuses one past _MAX_TEMPLATE_WIDTH, and a text of more numbers
    than _MAX_TEMPLATE_FIELDS (a real conversion has one): walking the 30 000
    a 60 KiB template can hold took ~7 ms, once per representation."""
    width = 0
    for seen, match in enumerate(_DIGIT_RUN.finditer(spec)):
        run = match.group().lstrip("0")
        if (seen == _MAX_TEMPLATE_FIELDS or len(run) > 3
                or int(run or 0) > _MAX_TEMPLATE_WIDTH):
            raise _too_much_url()
        width = max(width, int(run or 0))
    return width


def _template_bytes(template: str, rep_id, value_digits: int) -> int:
    """The length of one address yt-dlp builds from a DASH template for one
    representation, each field printing its padded width or `value_digits`
    characters, whichever is more. Refuses a template that is too long, has
    too many fields, pads one too wide or leaves a conversion unfinished."""
    uses = template.count("$RepresentationID$") if rep_id is not None else 0
    length = len(template) + uses * (len(rep_id or "") - len("$RepresentationID$"))
    if length > _MAX_TEMPLATE_BYTES:
        raise _too_much_url()
    if uses:  # an @id can complete a field the template only starts
        template = template.replace("$RepresentationID$", rep_id)
    runs = template.split("$")
    if len(runs) > 2 * _MAX_TEMPLATE_FIELDS + 1:
        raise _too_much_url()
    for spec in _TEMPLATE_FIELD.findall(template):
        length += max(_spec_width(spec) if spec else 0, value_digits)
    # yt-dlp doubles a `%` only outside `$...$`: every `$` flips "inside", and
    # a `%` inside stays a live conversion whatever its run names (`$%9s$`,
    # an unknown name, a run nothing closes, a second `%` in a field). A
    # keyless one prints the whole values dict, a keyed one (`%(Number)9d`)
    # a value, each padded to its width.
    inside = [i for i in range(1, len(runs), 2) if "%" in runs[i]]
    if not inside:
        return length
    if sum(runs[i].count("%") for i in inside) > _MAX_TEMPLATE_FIELDS:
        raise _too_much_url()
    # yt-dlp's substitution drops a field's closing `$`, so a conversion
    # still open at the end of its run (no type character yet, or a mapping
    # key nothing closed) is completed by the text AFTER it, which nothing
    # here sizes: `$Number%0$10000000d` prints ten million characters.
    for i in inside:
        tail = runs[i].rpartition("%")[2]
        if _CONVERSION_HEAD.match(tail).end() == len(tail):
            raise _too_much_url()
    # A field standing outside (after a stray `$`) takes the next run's
    # opening `$`: that run can read as a field without being one, here or to
    # yt-dlp, so nothing in such a template counts as sized above.
    shifted = any(_TEMPLATE_FIELD_RUN.fullmatch(run) for run in runs[2:-1:2])
    for i in inside:
        specs = runs[i].split("%")[1:]
        if (not shifted and i < len(runs) - 1
                and _TEMPLATE_FIELD_RUN.fullmatch(runs[i])):
            specs = specs[1:]  # the field's own conversion, sized above
        for spec in specs:
            # 48 + 3 values: {'Time': …, 'Bandwidth': …, 'Number': …}.
            length += max(_spec_width(spec), 48 + 3 * value_digits)
    return length


def _representation_url_bytes(info: dict, n: int, rep_id, bandwidth) -> int:
    """The bytes of the fragment and initialization addresses yt-dlp builds
    for one representation of `n` segments."""
    digits = max(_digits(abs(info.get("start") or 1) + n),
                 _digits(info.get("s_time") or 0), len(bandwidth or ""))
    spent = 0
    if info.get("media") and "urls" not in info:
        # The template is prepared even when it expands to no segment.
        spent += max(n, 1) * _template_bytes(info["media"], rep_id, digits)
    if info.get("initialization"):
        spent += _template_bytes(info["initialization"], rep_id, digits)
    return spent


# A Representation's attributes yt-dlp copies into its format, and how often.
_MPD_LABEL_ATTRS = (
    ("mimeType", 2), ("codecs", 1),
    ("{urn:scte:dash:scte214-extensions}supplementalCodecs", 1))


def _representation_attr(representation, adaptation, name):
    """yt-dlp reads a Representation's attributes from its AdaptationSet's
    overlaid with its own."""
    if name in representation.attrib:
        return representation.get(name)
    return adaptation.get(name)


def _refuse_oversized_mpd(mpd_doc, mpd_base_len: int = 0) -> None:
    """`mpd_base_len`: the length of the manifest's own (post-redirect) base
    URL, which yt-dlp prepends afresh to each representation's relative base
    URL (the redirect hop can make it ~64 KB). It is charged for every
    representation, also one whose chain ends at an absolute BaseURL: an
    over-count of a few hundred bytes each."""
    from yt_dlp.utils import parse_duration
    # yt-dlp's own derivation: only a root named MPD (any case) lends its
    # namespace; under any other root it expands unqualified Periods, so
    # that is where the count has to look too.
    match = re.match(r"(?i)^{([^}]+)?}MPD$", getattr(mpd_doc, "tag", "") or "")
    ns = (match.group(1) or "") if match else ""

    def q(name):
        return f"{{{ns}}}{name}" if ns else name

    # yt-dlp joins each representation's base URL afresh from the BaseURL of
    # every level above it (stopping at an absolute one; counted whole here).
    def base_url(element):
        found = element.find(q("BaseURL"))
        return len(found.text or "") if found is not None else 0
    mpd_duration = parse_duration(mpd_doc.get("mediaPresentationDuration"))
    count = representations = url_bytes = label_chars = walk = 0
    mpd_base = base_url(mpd_doc)
    # len(element) is its number of children.
    root_children = len(mpd_doc)
    # Counted before the walk: each Period costs it a duration parse and a
    # segment-info lookup of its own.
    periods = mpd_doc.findall(q("Period"))
    if len(periods) > _MAX_MPD_REPRESENTATIONS:
        raise _too_many_variants(_MAX_MPD_REPRESENTATIONS)
    adaptation_sets = 0
    for period in periods:
        period_duration = (parse_duration(period.get("duration"))
                           or mpd_duration)
        period_info = _mpd_segment_info(period, {"timescale": 1}, q)
        period_base = mpd_base + base_url(period)
        above_set = root_children + len(period)
        adaptations = period.findall(q("AdaptationSet"))
        adaptation_sets += len(adaptations)
        if adaptation_sets > _MAX_MPD_REPRESENTATIONS:
            raise _too_many_variants(_MAX_MPD_REPRESENTATIONS)
        for adaptation in adaptations:
            set_info = _mpd_segment_info(adaptation, period_info, q)
            set_base = period_base + base_url(adaptation)
            # What yt-dlp walks above each Representation: the root and the
            # Period once, the set twice, and a copy of the set's attributes
            # (which costs more than a search of as many children).
            above = above_set + 2 * (len(adaptation) + len(adaptation.attrib))
            for representation in adaptation.findall(q("Representation")):
                representations += 1
                if representations > _MAX_MPD_REPRESENTATIONS:
                    raise _too_many_variants(_MAX_MPD_REPRESENTATIONS)
                walk += above + len(representation) + len(representation.attrib)
                if walk > _MAX_MANIFEST_WALK:
                    raise _too_much_walk()
                info = _mpd_segment_info(representation, set_info, q)
                for name in ("bandwidth", "width", "height",
                             "audioSamplingRate", "frameRate"):
                    _refuse_long_number(
                        _representation_attr(representation, adaptation, name))
                n, seconds = _representation_segments(info, period_duration)
                count += n
                if count > _MAX_MPD_SEGMENTS:
                    raise _too_many_segments()
                rep_id = _representation_attr(representation, adaptation, "id")
                # yt-dlp derives each format's ext and container from the
                # @mimeType, its codecs and note from @codecs and
                # supplementalCodecs and (under an mpd_id) its id from @id:
                # strings of its own per Representation, at parse time,
                # before the format pass's count.
                label_chars += len(rep_id or "") + sum(
                    weight * len(_representation_attr(
                        representation, adaptation, name) or "")
                    for name, weight in _MPD_LABEL_ATTRS)
                if label_chars > _MAX_FORMAT_LABEL_CHARS:
                    raise _labels_too_long()
                url_bytes += mpd_base_len + set_base + base_url(representation)
                url_bytes += _representation_url_bytes(
                    info, n, rep_id,
                    _representation_attr(representation, adaptation, "bandwidth"))
                if url_bytes > _MAX_FRAGMENT_URL_BYTES:
                    raise _too_much_url()
                # yt-dlp's own fallback: a period with no length takes the
                # first expanded representation's, which the next ones use.
                if not period_duration and n and seconds:
                    period_duration = seconds
    _charge(count, representations, url_bytes, walk)


def _bounded_mpd_periods(original):
    def _parse_mpd_periods(self, mpd_doc, *args, **kwargs):
        # yt-dlp's signature: (mpd_doc, mpd_id=None, mpd_base_url='', ...).
        mpd_base_url = (args[1] if len(args) > 1
                        else kwargs.get("mpd_base_url"))
        # +1: the "/" yt-dlp appends to a base that lacks one.
        _refuse_oversized_mpd(
            mpd_doc, len(mpd_base_url) + 1 if isinstance(mpd_base_url, str) else 0)
        return original(self, mpd_doc, *args, **kwargs)
    setattr(_parse_mpd_periods, _MPD_SENTINEL, True)
    return _parse_mpd_periods


def _bound_mpd_expansion() -> None:
    from yt_dlp.extractor.common import InfoExtractor
    current = getattr(InfoExtractor, "_parse_mpd_periods", None)
    # A yt-dlp without the method gets nothing to wrap, and is_installed()
    # then fails closed rather than trusting an expansion it cannot see.
    if callable(current) and not getattr(current, _MPD_SENTINEL, False):
        InfoExtractor._parse_mpd_periods = _bounded_mpd_periods(current)


def _mpd_expansion_is_bounded() -> bool:
    from yt_dlp.extractor.common import InfoExtractor
    return getattr(getattr(InfoExtractor, "_parse_mpd_periods", None),
                   _MPD_SENTINEL, False)


_ISM_SENTINEL = "_fwb_ism_bounded"


def _refuse_oversized_ism(ism_doc, ism_base_len: int = 0) -> None:
    """`ism_base_len`: the length of the manifest's own (post-redirect) URL,
    which yt-dlp urljoins onto each stream's Url and every fragment then
    copies (the redirect hop can make it ~64 KB). It is charged for every
    fragment, also under an absolute Url: an over-count of a few hundred
    bytes each."""
    from yt_dlp.utils import int_or_none
    if ism_doc.get("IsLive") == "TRUE":
        return  # yt-dlp returns no formats for it, before any expansion
    _refuse_long_number(ism_doc.get("Duration"))
    _refuse_long_number(ism_doc.get("TimeScale"))
    count = variants = url_bytes = label_chars = walk = 0
    root_children = len(ism_doc)
    for stream in ism_doc.findall("StreamIndex"):
        if stream.get("Type") not in ("video", "audio", "text"):
            continue
        _refuse_long_number(stream.get("TimeScale"))
        # yt-dlp joins the stream's Name into a format_id of its own for
        # every QualityLevel, at parse time, before the format pass's count.
        label_chars += (len(stream.get("Name") or "")
                        * len(stream.findall("QualityLevel")))
        if label_chars > _MAX_FORMAT_LABEL_CHARS:
            raise _labels_too_long()
        # Every QualityLevel counts, also one whose codec yt-dlp skips: an
        # over-count far below the cap, and a yt-dlp that learns the codec
        # stays bounded.
        tracks = len(stream.findall("QualityLevel"))
        # Each level is a format (or subtitle track) even with no chunk, and
        # yt-dlp re-walks the stream's chunks for every one of them.
        variants += tracks
        if variants > _MAX_MPD_REPRESENTATIONS:
            raise _too_many_variants(_MAX_MPD_REPRESENTATIONS)
        if not tracks:
            continue
        # For each level yt-dlp searches the stream's children for its chunks
        # and the manifest's for a Protection element.
        walk += tracks * (root_children + len(stream))
        if walk > _MAX_MANIFEST_WALK:
            raise _too_much_walk()
        for level in stream.findall("QualityLevel"):
            for name in ("Bitrate", "MaxWidth", "Width", "MaxHeight", "Height",
                         "SamplingRate", "Channels", "BitsPerSample",
                         "NALUnitLengthField"):
                _refuse_long_number(level.get(name))
        chunks = stream.findall("c")
        for c in chunks:
            for name in ("t", "d", "r"):
                _refuse_long_number(c.get(name))
        # yt-dlp: range(int_or_none(r) or 1) per chunk, so r <= -1 builds no
        # fragment, but the chunk is still read once per level: it costs one.
        per_track = sum(max(int_or_none(c.get("r")) or 1, 1) for c in chunks)
        count += tracks * per_track
        if count > _MAX_MPD_SEGMENTS:
            raise _too_many_segments()
        # Each fragment's address is the stream's Url with {bitrate} set to
        # the level's Bitrate and {start time} to the chunk's time, which
        # starts at a chunk's @t and grows by each @d (a missing @d is a
        # quotient: a float, at most 24 characters).
        pattern = stream.get("Url") or ""
        bitrates = len(_ISM_BITRATE.findall(pattern))
        starts = len(_ISM_START_TIME.findall(pattern))
        time_digits = max(_digits(
            max((abs(int_or_none(c.get("t")) or 0) for c in chunks), default=0)
            + sum(abs(int_or_none(c.get("d")) or 0)
                  * max(int_or_none(c.get("r")) or 1, 1) for c in chunks)), 24)
        for level in stream.findall("QualityLevel"):
            length = (ism_base_len + len(pattern) + starts * time_digits + bitrates
                      * max(len(level.get("Bitrate") or "") - len("{bitrate}"), 0))
            # The level's pattern is built even with no chunk to expand.
            url_bytes += max(per_track, 1) * length
            if url_bytes > _MAX_FRAGMENT_URL_BYTES:
                raise _too_much_url()
    _charge(count, variants, url_bytes, walk)


def _bounded_ism(original):
    def _parse_ism_formats_and_subtitles(self, ism_doc, *args, **kwargs):
        # yt-dlp's signature: (ism_doc, ism_url, ism_id=None).
        ism_url = args[0] if args else kwargs.get("ism_url")
        # +1: the "/" urljoin adds to a base with no path.
        _refuse_oversized_ism(
            ism_doc, len(ism_url) + 1 if isinstance(ism_url, str) else 0)
        return original(self, ism_doc, *args, **kwargs)
    setattr(_parse_ism_formats_and_subtitles, _ISM_SENTINEL, True)
    return _parse_ism_formats_and_subtitles


def _bound_ism_expansion() -> None:
    from yt_dlp.extractor.common import InfoExtractor
    current = getattr(InfoExtractor, "_parse_ism_formats_and_subtitles", None)
    # As with DASH: nothing to wrap leaves is_installed() failing closed.
    if callable(current) and not getattr(current, _ISM_SENTINEL, False):
        InfoExtractor._parse_ism_formats_and_subtitles = _bounded_ism(current)


def _ism_expansion_is_bounded() -> bool:
    from yt_dlp.extractor.common import InfoExtractor
    return getattr(getattr(InfoExtractor, "_parse_ism_formats_and_subtitles", None),
                   _ISM_SENTINEL, False)


# HLS: _parse_m3u8_formats_and_subtitles turns a master playlist into one
# format per URI line (two with PROGRESSIVE-URI) and one per EXT-X-MEDIA
# rendition, and YoutubeDL then sorts, selects and sanitizes every one of them
# in pure Python at ~0.5 ms a format: a 200 kB playlist of 100 000 bare URI
# lines cost 52 s of CPU and 360 MB, and the 64 MiB body bound admits ~30 M.
# Real masters list tens of variants, a few hundred with many audio and
# subtitle renditions; at the cap the format pass stays under a second.
# A media playlist (it carries #EXT-X-TARGETDURATION) is returned whole as a
# single format and is not counted here; HlsFD's expansion of its segments is
# (see _bound_hls_fragments).
_MAX_M3U8_FORMATS = 2_000
# yt-dlp also splits a master into one str per line, twice, before it looks
# at any of them, so a master of nothing but comments lists no format yet
# costs ~0.1 s and ~50 MB per million lines (64 MiB of "#a" lines: ~6 s and
# over a gigabyte). A master at the format cap is two lines per variant plus
# a few tags, far below this.
_MAX_M3U8_MASTER_LINES = 200_000
_M3U8_SENTINEL = "_fwb_m3u8_bounded"
# One match per line yt-dlp turns into a format, with lines split where
# str.splitlines() splits: an EXT-X-MEDIA rendition, a STREAM-INF carrying a
# PROGRESSIVE-URI, and any line that is neither blank nor starts with "#"
# (leading whitespace included). Matching in C keeps a playlist of comments
# from costing a Python step per line.
_M3U8_BREAKS = r"\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
_M3U8_BREAK_CHARS = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"
_M3U8_BREAK = re.compile(f"[{_M3U8_BREAKS}]")
_M3U8_FORMAT_LINE = re.compile(
    rf"(?<![^{_M3U8_BREAKS}])(?:#EXT-X-MEDIA:"
    rf"|#EXT-X-STREAM-INF:[^{_M3U8_BREAKS}]*?PROGRESSIVE-URI"
    rf"|[^\s#]|[^\S{_M3U8_BREAKS}]+\S)")


# yt-dlp also runs backtracking regexes over single lines, each quadratic in
# the line: parse_m3u8_attributes over every #EXT-X-MEDIA / #EXT-X-STREAM-INF
# line and a width-x-height search over its RESOLUTION (~5e-9 s x length^2,
# the GIL held throughout: one 30 kB line ~5 s, a 1 MB line hours), and an
# `audio.*?=` search over every variant's joined address (occurrences of
# "audio" x its length: 20 variants under a 60 kB redirect address of
# "audio" took 74 s). The work is counted before the parse, per extraction:
# length^2 for an attribute line, (line + master address) x ("audio" + 1) for
# a variant line. At the cap one line holds the GIL ~0.1 s and a whole
# extraction spends ~3 s; sixteen masters of 300 variants under 300-byte
# signed addresses spend a sixth of it, a master of 100 renditions and 200
# variants with 1.5 kB tokens on every line about half.
_MAX_M3U8_LINE_CHARS = 4096
_MAX_M3U8_LINE_WORK = 2 ** 29
# The tag wherever it stands, through the end of its line: a literal lead
# keeps the search of a 64 MiB document at memchr speed, and the caller
# drops a match that does not start its line.
_M3U8_ATTR_LINE = re.compile(
    rf"#EXT-X-(?:MEDIA|STREAM-INF):[^{_M3U8_BREAKS}]*")
# HlsFD._has_drm (run on master and media playlists alike) scans from every
# "#EXT-X-KEY:" / "#EXT-X-SESSION-KEY:" to the end of its line, three times:
# quadratic only when one line (to yt-dlp's `.`: up to a "\n") holds the tag
# more than once, which no playlist does. Checked only when tags x length
# could cost over ~0.2 s.
_MAX_M3U8_KEY_WORK = 2 ** 24
_M3U8_KEY_TWICE = re.compile(
    r"#EXT-X-(?:SESSION-)?KEY:[^\n]*#EXT-X-(?:SESSION-)?KEY:")


def _line_too_long():
    from yt_dlp.utils import ExtractorError
    return ExtractorError("a line in the playlist is too long", expected=True)


def _line_breaks(doc: str) -> int:
    """How many lines str.splitlines() would end in `doc`, counted in C."""
    return sum(doc.count(b) for b in _M3U8_BREAK_CHARS) - doc.count("\r\n")


def _refuse_oversized_m3u8(m3u8_doc, url_len: int = 0, url_audio: int = 0) -> None:
    """`url_len`: the length of the master's own (post-redirect) URL, which
    yt-dlp urljoins each variant and rendition line onto; `url_audio`: how
    often that URL says "audio"."""
    from yt_dlp.utils import ExtractorError
    if not isinstance(m3u8_doc, str):
        return
    if (m3u8_doc.count("KEY:") * len(m3u8_doc) > _MAX_M3U8_KEY_WORK
            and _M3U8_KEY_TWICE.search(m3u8_doc)):
        raise _line_too_long()
    if "#EXT-X-TARGETDURATION" in m3u8_doc:
        return
    if _line_breaks(m3u8_doc) > _MAX_M3U8_MASTER_LINES:
        raise ExtractorError(
            "the playlist has too many lines (more than "
            f"{_MAX_M3U8_MASTER_LINES})", expected=True)
    # Before the format count: at most one step per line the cap above let
    # through, and it stops at the budget.
    work = getattr(_DIALS.tls, "line_work", 0)
    for match in _M3U8_ATTR_LINE.finditer(m3u8_doc):
        start = match.start()
        if start and m3u8_doc[start - 1] not in _M3U8_BREAK_CHARS:
            continue  # inside a line: text to yt-dlp
        length = match.end() - start
        if length > _MAX_M3U8_LINE_CHARS:
            raise _line_too_long()
        work += length * length
        if work > _MAX_M3U8_LINE_WORK:
            raise _line_too_long()
    count = 0
    for count, match in enumerate(_M3U8_FORMAT_LINE.finditer(m3u8_doc), 1):
        if count > _MAX_M3U8_FORMATS:
            raise ExtractorError(
                f"the stream lists too many variants (more than {_MAX_M3U8_FORMATS})",
                expected=True)
        start = match.start()
        if m3u8_doc[start] == "#":
            continue  # a rendition or PROGRESSIVE-URI tag, sized above
        # A variant line, through its end: the `audio` search over its
        # joined address.
        end = _M3U8_BREAK.search(m3u8_doc, match.end())
        end = end.start() if end else len(m3u8_doc)
        scan = ((end - start + url_len)
                * (m3u8_doc.count("audio", start, end) + url_audio + 1))
        work += scan
        if scan > _MAX_M3U8_LINE_CHARS ** 2 or work > _MAX_M3U8_LINE_WORK:
            raise _line_too_long()
    # A page (or a set-level F4M) can hand one extraction many masters, each
    # within the cap: every variant is a format from the extraction's budget,
    # like a DASH representation, and its address a fresh urljoin(master URL,
    # line) (an absolute line a slice of the document instead).
    _charge(0, count, len(m3u8_doc) + count * (url_len + 1))
    _DIALS.tls.line_work = work


def _bounded_m3u8(original):
    def _parse_m3u8_formats_and_subtitles(self, m3u8_doc, *args, **kwargs):
        # yt-dlp's signature: (m3u8_doc, m3u8_url=None, ...).
        url = args[0] if args else kwargs.get("m3u8_url")
        if not isinstance(url, str):
            url = ""
        _refuse_oversized_m3u8(m3u8_doc, len(url), url.count("audio"))
        return original(self, m3u8_doc, *args, **kwargs)
    setattr(_parse_m3u8_formats_and_subtitles, _M3U8_SENTINEL, True)
    return _parse_m3u8_formats_and_subtitles


def _bound_m3u8_formats() -> None:
    from yt_dlp.extractor.common import InfoExtractor
    current = getattr(InfoExtractor, "_parse_m3u8_formats_and_subtitles", None)
    # As with DASH: nothing to wrap leaves is_installed() failing closed.
    if callable(current) and not getattr(current, _M3U8_SENTINEL, False):
        InfoExtractor._parse_m3u8_formats_and_subtitles = _bounded_m3u8(current)


def _m3u8_formats_are_bounded() -> bool:
    from yt_dlp.extractor.common import InfoExtractor
    return getattr(getattr(InfoExtractor, "_parse_m3u8_formats_and_subtitles", None),
                   _M3U8_SENTINEL, False)


# HLS media playlist, in the download child: HlsFD.real_download turns every
# segment line into a fragment dict (and splits the playlist into lines twice
# first) before it fetches anything, so the 64 MiB body bound admits ~33 M
# fragments, ~12 GB. It hands the whole playlist to can_download before
# either pass, which is where the segments are counted, against the DASH
# budget: one per line HlsFD keeps as a fragment (stripped, not blank, not a
# "#" tag) and one per #EXT-X-MAP initialization segment. That over-counts a
# playlist whose ad or discontinuity filter skips some, never under-counts.
_HLS_SENTINEL = "_fwb_hls_bounded"
_HLS_FRAGMENT_LINE = re.compile(
    rf"(?<![^{_M3U8_BREAKS}])[^\S{_M3U8_BREAKS}]*(?:[^\s#]|#EXT-X-MAP)")


# HlsFD also builds each fragment's address as urljoin(playlist URL, line)
# before it fetches anything, and the URL of the key an #EXT-X-KEY line names
# likewise (the fragments after it keep that key's dict), so one 2-byte line
# costs the playlist URL's length: up to ~64 KB through a redirect, more when
# a master's variant line names it. Those addresses must fit what the
# extraction left of DASH's budget (_MAX_FRAGMENT_URL_BYTES); a 12 h playlist
# of 21 600 segments with a 300-byte URL costs ~6.5 MB.
_HLS_URL_LINE = re.compile(
    rf"(?<![^{_M3U8_BREAKS}])[^\S{_M3U8_BREAKS}]*"
    r"(?:[^\s#]|#EXT-X-MAP|#EXT-X-KEY)")


# HlsFD int()s #EXT-X-MEDIA-SEQUENCE and both sides of #EXT-X-BYTERANGE's
# `n@o`, then keeps a number of its own per fragment (the sequence + 1, the
# running range end): one 4 299-digit number makes every fragment ~1.8 kB
# heavier, 2.2 GB instead of 0.4 GB at the segment cap. Both are 64-bit at
# most (_MAX_NUMBER_CHARS). Matched wherever the tag stands, not only at a
# line start: a literal prefix keeps the scan of a real playlist at memchr
# speed, and the tag mid-line before a 21-character token is no playlist.
# The skip to a later token stops at a "#" (int() takes none, so no number
# HlsFD reads lies past one): a line of many tags is then searched once, not
# once per tag.
_HLS_LONG_NUMBER = re.compile(
    rf"#EXT-X-(?:MEDIA-SEQUENCE|BYTERANGE)[^{_M3U8_BREAKS}]"
    rf"(?:[^#{_M3U8_BREAKS}]*?(?:@|[^\S{_M3U8_BREAKS}]))?"
    rf"[^@\s]{{{_MAX_NUMBER_CHARS + 1}}}")


def _refuse_oversized_hls_media(manifest, url_len: int = 0) -> None:
    """`url_len`: the length of the URL HlsFD joins the playlist's lines
    onto."""
    if not isinstance(manifest, str):
        return
    if _HLS_LONG_NUMBER.search(manifest):
        from yt_dlp.utils import ExtractorError
        raise ExtractorError("a number in the stream manifest is too long",
                             expected=True)
    breaks = _line_breaks(manifest)
    # A playlist with no more lines than the cap cannot list more fragments:
    # every real one pays only the C-level line count.
    if breaks >= _MAX_MPD_SEGMENTS:
        for count, _match in enumerate(_HLS_FRAGMENT_LINE.finditer(manifest), 1):
            if count > _MAX_MPD_SEGMENTS:
                raise _too_many_segments()
    # Checked against what the extraction has spent, never spent itself:
    # HlsFD drops its fragment list once the download ends, and a merged
    # video + audio download runs it twice. Every line joined is the cheap
    # upper bound; only past the budget are the joined lines counted, in C
    # (subn's copy is never larger than the playlist itself).
    spent = getattr(_DIALS.tls, "url_bytes", 0) + len(manifest)
    if spent + (breaks + 1) * url_len <= _MAX_FRAGMENT_URL_BYTES:
        return
    if spent + _HLS_URL_LINE.subn("", manifest)[1] * url_len > _MAX_FRAGMENT_URL_BYTES:
        raise _too_much_url()


def _bounded_hls_can_download(original):
    def can_download(cls, manifest, *args, **kwargs):
        # HlsFD joins onto info_dict['url'] for a playlist it was handed
        # (hls_media_playlist_data), else onto the URL its own fetch, just
        # before this call, ended at.
        info_dict = args[0] if args else kwargs.get("info_dict")
        if not isinstance(info_dict, dict):
            info_dict = {}
        url = info_dict.get("url")
        url_len = len(url) if isinstance(url, str) else 0
        if not info_dict.get("hls_media_playlist_data"):
            url_len = max(url_len, getattr(_DIALS.tls, "response_url_len", 0))
        _refuse_oversized_hls_media(manifest, url_len)
        return original(cls, manifest, *args, **kwargs)
    setattr(can_download, _HLS_SENTINEL, True)
    return can_download


def _bound_hls_fragments() -> None:
    from yt_dlp.downloader.hls import HlsFD
    current = vars(HlsFD).get("can_download")
    original = getattr(current, "__func__", None)
    # Only HlsFD's own classmethod is wrapped: anything else leaves
    # is_installed() failing closed, as with DASH.
    if (isinstance(current, classmethod) and callable(original)
            and not getattr(original, _HLS_SENTINEL, False)):
        HlsFD.can_download = classmethod(_bounded_hls_can_download(original))


def _hls_fragments_are_bounded() -> bool:
    from yt_dlp.downloader.hls import HlsFD
    return getattr(getattr(HlsFD, "can_download", None), _HLS_SENTINEL, False)


# HDS (F4M): _parse_f4m_formats turns every <media> node into a format, with
# the same ~0.5 ms format pass per format as an HLS master, so its nodes are
# counted against the HLS format cap before the parse. A set-level manifest
# (one with no bootstrapInfo) also fetches every node that names another
# .f4m / .m3u8 and parses it in turn, an .f4m child recursively: the nested
# references, the nesting depth and the formats the children add up to are
# bounded too; the references from the budget of the whole extraction, as each
# child within the cap could otherwise name the cap's worth again. Real HDS
# lists a handful of bitrates, and a set-level manifest a few stream-level
# ones one level down.
_F4M_SENTINEL = "_fwb_f4m_bounded"
_F4M_NAMESPACES = ("{http://ns.adobe.com/f4m/1.0}", "{http://ns.adobe.com/f4m/2.0}")
_MAX_F4M_NESTED = 16
_MAX_F4M_DEPTH = 2


def _refuse_oversized_f4m(manifest, manifest_url: str = "") -> None:
    if not isinstance(manifest, xml.etree.ElementTree.Element):
        return  # yt-dlp returns or fails on its own, before any node
    # yt-dlp's choice: the 1.0 nodes, else the 2.0 ones.
    nodes = manifest.findall(_F4M_NAMESPACES[0] + "media")
    v2 = not nodes
    nodes = nodes or manifest.findall(_F4M_NAMESPACES[1] + "media")
    if len(nodes) > _MAX_M3U8_FORMATS:
        raise _too_many_variants(_MAX_M3U8_FORMATS)
    if any(manifest.find(ns + "bootstrapInfo") is not None
           for ns in _F4M_NAMESPACES):
        return  # a stream-level manifest refers to nothing outside it
    from yt_dlp.downloader.f4m import get_base_url, remove_encrypted_media
    from yt_dlp.utils import determine_ext
    # Each node's address is a fresh <baseURL> + "/" + its url or, without
    # one, the directory of the PREVIOUS node's address + "/" + its url
    # (yt-dlp reassigns manifest_url in its loop), so relative urls holding
    # a "/" chain and grow: 2000 nodes of 500 bytes build ~1 GB. Sized by
    # length only, in yt-dlp's order over the nodes it keeps, never built.
    # A nested child's own variants are charged where its master is parsed,
    # so none is counted here.
    # yt-dlp fetches a node as a nested manifest when determine_ext of that
    # joined address says .f4m / .m3u8, and the head decides it as often as
    # the node does (".../b.f4m?" + "x", ".../b.f4m" + "/"), so the address's
    # extension is followed along in the same loop as its length.
    base = get_base_url(manifest)
    base_head = _f4m_head(base, determine_ext) if base else None
    dir_head = _f4m_head(manifest_url[:max(manifest_url.rfind("/"), 0)],
                         determine_ext)
    dir_len = max(manifest_url.rfind("/"), 0)
    url_bytes, joined_nested = 0, set()
    for node in remove_encrypted_media(nodes):
        media = node.get("href") if v2 else None
        if media is None:
            media = node.get("url")
        if not media:
            continue  # skipped before the reassignment
        cut = media.rfind("/")
        if media.startswith(("http://", "https://")):
            url_bytes += len(media)
            dir_len = cut
            ext = determine_ext(media)
            dir_head = _f4m_head(media[:cut], determine_ext)
        else:
            head = len(base) if base else dir_len
            url_bytes += head + 1 + len(media)
            dir_len = head + 1 + cut if cut >= 0 else head
            ext = _f4m_joined_ext(base_head or dir_head, media, determine_ext)
            if not base_head and cut >= 0:
                dir_head = _f4m_head_join(dir_head, media[:cut], determine_ext)
        if ext in ("f4m", "m3u8"):
            joined_nested.add(id(node))
    # yt-dlp reads @href in a 2.0 manifest only and @url otherwise: a node
    # naming a manifest in either counts too, which can only overcount.
    nested = sum(1 for node in nodes
                 if id(node) in joined_nested
                 or any(determine_ext(node.get(attr) or "") in ("f4m", "m3u8")
                        for attr in ("href", "url")))
    # The download child never resets the budget: it extracts once.
    total = getattr(_DIALS.tls, "f4m_nested", 0) + nested
    if total > _MAX_F4M_NESTED:
        raise _too_many_variants(_MAX_F4M_NESTED)
    _charge(0, 0, url_bytes)
    _DIALS.tls.f4m_nested = total


# determine_ext reads an address up to its first "?", takes the text after
# the last "." there, and answers "f4m" / "m3u8" only for exactly that text,
# trailing slashes allowed; it needs a "." anywhere in the address. A head
# (the base, or the directory a node is joined onto) is therefore kept as
# three facts, never as the growing text: whether it holds a "?", whether it
# holds a ".", and the manifest extension its text before the "?" ends in
# ("" for none). Appending "/" + more keeps that extension only when the
# more is slashes alone.
def _f4m_tail_ext(guess, determine_ext) -> str:
    ext = determine_ext("." + guess)
    return ext if ext in ("f4m", "m3u8") else ""


def _f4m_head(text, determine_ext):
    pre, query, _ = text.partition("?")
    return (bool(query), "." in text,
            _f4m_tail_ext(pre.rpartition(".")[2], determine_ext))


def _f4m_head_join(head, more, determine_ext):
    """The facts of head + "/" + more."""
    query, dot, tail = head
    dot = dot or "." in more
    if query:
        return query, dot, tail
    pre, query, _ = more.partition("?")
    guess = pre.rpartition(".")[2] if "." in pre else f"{tail}/{pre}"
    return bool(query), dot, _f4m_tail_ext(guess, determine_ext)


def _f4m_joined_ext(head, media, determine_ext) -> str:
    """determine_ext(head + "/" + media), from the head's facts."""
    query, dot, tail = _f4m_head_join(head, media, determine_ext)
    return tail if dot else ""


def _bounded_f4m(original):
    def _parse_f4m_formats(self, manifest, *args, **kwargs):
        depth = getattr(_DIALS.tls, "f4m_depth", 0)
        if depth >= _MAX_F4M_DEPTH:
            raise _too_many_variants(_MAX_F4M_NESTED)
        # yt-dlp's signature: (manifest, manifest_url, video_id, ...).
        url = args[0] if args else kwargs.get("manifest_url")
        _refuse_oversized_f4m(manifest, url if isinstance(url, str) else "")
        _DIALS.tls.f4m_depth = depth + 1
        try:
            formats = original(self, manifest, *args, **kwargs)
        finally:
            _DIALS.tls.f4m_depth = depth
        # The nested children's formats, summed before the format pass.
        if isinstance(formats, list) and len(formats) > _MAX_M3U8_FORMATS:
            raise _too_many_variants(_MAX_M3U8_FORMATS)
        return formats
    setattr(_parse_f4m_formats, _F4M_SENTINEL, True)
    return _parse_f4m_formats


def _bound_f4m_formats() -> None:
    from yt_dlp.extractor.common import InfoExtractor
    current = getattr(InfoExtractor, "_parse_f4m_formats", None)
    # As with DASH: nothing to wrap leaves is_installed() failing closed.
    if callable(current) and not getattr(current, _F4M_SENTINEL, False):
        InfoExtractor._parse_f4m_formats = _bounded_f4m(current)


def _f4m_formats_are_bounded() -> bool:
    from yt_dlp.extractor.common import InfoExtractor
    return getattr(getattr(InfoExtractor, "_parse_f4m_formats", None),
                   _F4M_SENTINEL, False)


# HDS in the download child: F4mFD's build_fragments_list (a module function
# it looks up at call time) turns every segment-run count of the bootstrap, a
# uint32, into that many fragment tuples before it fetches any, so a 106-byte
# bootstrap asks for billions (~107 bytes each). The counts are summed against
# the segment cap first, a live stream's 4294967295 as the two yt-dlp keeps.
# Real HDS VODs are hundreds to tens of thousands of fragments.
_F4M_FRAGMENTS_SENTINEL = "_fwb_f4m_fragments_bounded"


def _bounded_build_fragments_list(original):
    def build_fragments_list(boot_info):
        try:
            runs = boot_info["segments"][0]["segment_run"]
            live = boot_info["live"]
        except (KeyError, IndexError, TypeError):
            runs, live = (), False  # yt-dlp raises on its own
        total = 0
        for _segment, count in runs:
            total += 2 if count == 4294967295 and live else max(count, 0)
            if total > _MAX_MPD_SEGMENTS:
                raise _too_many_segments()
        return original(boot_info)
    setattr(build_fragments_list, _F4M_FRAGMENTS_SENTINEL, True)
    return build_fragments_list


# Before that count exists, F4mFD's read_bootstrap_info (also looked up at
# call time, for a fetched and for an inline base64 bootstrap) builds one dict
# per afrt entry (16 bytes at least) and one tuple per asrt run (8 bytes) from
# uint32 counts: tens of MB of bootstrap cost about 1 GB. Capping the bytes
# bounds both tables before they are built. Live bootstraps are hundreds of
# bytes; a 12 h VOD with an afrt entry per 4 s fragment is ~173 KB.
_MAX_F4M_BOOTSTRAP_BYTES = 2 * 1024 * 1024
_F4M_BOOTSTRAP_SENTINEL = "_fwb_f4m_bootstrap_bounded"


def _bounded_read_bootstrap_info(original):
    def read_bootstrap_info(bootstrap_bytes):
        if (isinstance(bootstrap_bytes, (bytes, bytearray, memoryview))
                and memoryview(bootstrap_bytes).nbytes > _MAX_F4M_BOOTSTRAP_BYTES):
            raise _too_many_segments()
        return original(bootstrap_bytes)
    setattr(read_bootstrap_info, _F4M_BOOTSTRAP_SENTINEL, True)
    return read_bootstrap_info


def _bound_f4m_fragments() -> None:
    from yt_dlp.downloader import f4m
    current = getattr(f4m, "build_fragments_list", None)
    # As with DASH: nothing to wrap leaves is_installed() failing closed.
    if callable(current) and not getattr(current, _F4M_FRAGMENTS_SENTINEL, False):
        f4m.build_fragments_list = _bounded_build_fragments_list(current)
    current = getattr(f4m, "read_bootstrap_info", None)
    if callable(current) and not getattr(current, _F4M_BOOTSTRAP_SENTINEL, False):
        f4m.read_bootstrap_info = _bounded_read_bootstrap_info(current)


def _f4m_fragments_are_bounded() -> bool:
    from yt_dlp.downloader import f4m
    return (getattr(getattr(f4m, "build_fragments_list", None),
                    _F4M_FRAGMENTS_SENTINEL, False)
            and getattr(getattr(f4m, "read_bootstrap_info", None),
                        _F4M_BOOTSTRAP_SENTINEL, False))


# SMIL: _parse_smil_formats_and_subtitles (GenericIE sends any page whose XML
# root is <smil> there) makes one format per distinct src of the <video>,
# <audio>, <media> and <imagestream> nodes, joins each relative one onto the
# <meta base> as a fresh copy (1 000 srcs under a 1 MB base: ~1 GB), and
# fetches and parses every src naming an HLS / HDS / DASH / Smooth Streaming
# manifest; every distinct <textstream> src becomes a subtitle track, found
# distinct by a list scan. The srcs and the textstream srcs are counted
# against the HLS format cap, the addresses sized against the extraction's
# budget, and the nested manifests counted with HDS's, all before the parse.
# Real SMIL lists a handful of bitrates and subtitle tracks under a short
# base.
_SMIL_SENTINEL = "_fwb_smil_bounded"
_SMIL_PATHS = (".//video", ".//audio", ".//media")
_SMIL_NESTED_EXTS = ("m3u8", "f4m", "mpd")
_ISM_MANIFEST = re.compile(r"\.ism/[Mm]anifest")
# The textstream list scan costs ~7 ns a comparison: twice what the format
# cap of distinct srcs costs on its own stays well under 0.1 s.
_MAX_SMIL_SUB_SCANS = 2 * _MAX_M3U8_FORMATS ** 2
# A comparison reads the srcs byte by byte only when their lengths match, so
# few long srcs of one length cost what the count does not see: ~8.7 GB/s,
# so this is ~0.03 s (~0.12 s for 4-byte characters). Real SMIL compares a
# few KB; 2 000 distinct 50-character srcs come to ~100 MB.
_MAX_SMIL_SUB_SCAN_BYTES = 256 * 1024 * 1024


def _refuse_oversized_smil(ie, smil, smil_url, namespace) -> None:
    if not isinstance(smil, xml.etree.ElementTree.Element):
        return  # yt-dlp fails on its own, before any node
    from yt_dlp.utils import determine_ext
    base = smil_url if isinstance(smil_url, str) else ""
    for meta in smil.findall(ie._xpath_ns("./head/meta", namespace)):
        b = meta.get("base") or meta.get("httpBase")
        if b:
            base = b
            break
    paths = [(path, False) for path in _SMIL_PATHS] + [(".//imagestream", True)]
    # yt-dlp keeps the first node per src, media before storyboards.
    srcs = {}
    for path, image in paths:
        for node in smil.findall(ie._xpath_ns(path, namespace)):
            src = node.get("src")
            if not src or src in srcs:
                continue
            srcs[src] = None if image else node
            if len(srcs) > _MAX_M3U8_FORMATS:
                raise _too_many_variants(_MAX_M3U8_FORMATS)
    # The subtitle parse dedups the textstream srcs with a list scan, and a
    # repeated src scans it again, so what is bounded is the comparisons that
    # scan makes: at most the distinct srcs seen so far, per textstream, and
    # the bytes of those that share its length.
    subs, scanned, compared, same_len = set(), 0, 0, {}
    for node in smil.findall(ie._xpath_ns(".//textstream", namespace)):
        src = node.get("src")
        if not src:
            continue
        scanned += len(subs)
        if scanned > _MAX_SMIL_SUB_SCANS:
            raise _too_many_variants(_MAX_M3U8_FORMATS)
        n = len(src)
        compared += same_len.get(n, 0) * n
        if compared > _MAX_SMIL_SUB_SCAN_BYTES:
            raise _too_many_variants(_MAX_M3U8_FORMATS)
        if src not in subs:
            subs.add(src)
            same_len[n] = same_len.get(n, 0) + 1
            if len(subs) > _MAX_M3U8_FORMATS:
                raise _too_many_variants(_MAX_M3U8_FORMATS)
    # A relative src is a fresh urljoin(base + "/", src), at most that long
    # (an .f4m one gains a short query); an absolute one, an rtmp play_path
    # or a storyboard is a slice of the document, counted all the same.
    url_bytes = sum(len(src) if src.startswith("http") else len(base) + 1 + len(src)
                    for src in srcs)
    if getattr(_DIALS.tls, "url_bytes", 0) + url_bytes > _MAX_FRAGMENT_URL_BYTES:
        raise _too_much_url()
    # Which srcs yt-dlp fetches as a manifest, decided as it decides: by the
    # address it joins (a base under ".ism/Manifest" makes every relative
    # src one), so each is joined once here; the budget check above bounds
    # what that copies. An rtmp src and a storyboard are never fetched.
    nested, joined_base = 0, f"{base}/"
    for src, node in srcs.items():
        if node is None:
            continue  # a storyboard
        streamer = node.get("streamer") or base
        proto = node.get("proto")
        if proto == "rtmp" or streamer.startswith("rtmp"):
            continue
        src_url = (src if src.startswith("http")
                   else urllib.parse.urljoin(joined_base, src)).strip()
        if (proto == "m3u8"
                or (determine_ext(src, default_ext=None) or node.get("ext"))
                in _SMIL_NESTED_EXTS
                or _ISM_MANIFEST.search(src_url)):
            nested += 1
    total = getattr(_DIALS.tls, "f4m_nested", 0) + nested
    if total > _MAX_F4M_NESTED:
        raise _too_many_variants(_MAX_F4M_NESTED)
    _charge(0, len(srcs), url_bytes)
    _DIALS.tls.f4m_nested = total


def _bounded_smil(original):
    def _parse_smil_formats_and_subtitles(self, smil, *args, **kwargs):
        # yt-dlp's signature, which MediasetIE passes on positionally:
        # (smil, smil_url, video_id, namespace=None, f4m_params=None,
        #  transform_rtmp_url=None).
        url = args[0] if args else kwargs.get("smil_url")
        namespace = args[2] if len(args) > 2 else kwargs.get("namespace")
        _refuse_oversized_smil(self, smil, url, namespace)
        return original(self, smil, *args, **kwargs)
    setattr(_parse_smil_formats_and_subtitles, _SMIL_SENTINEL, True)
    return _parse_smil_formats_and_subtitles


def _bound_smil_formats() -> None:
    from yt_dlp.extractor.common import InfoExtractor
    current = getattr(InfoExtractor, "_parse_smil_formats_and_subtitles", None)
    # As with DASH: nothing to wrap leaves is_installed() failing closed.
    if callable(current) and not getattr(current, _SMIL_SENTINEL, False):
        InfoExtractor._parse_smil_formats_and_subtitles = _bounded_smil(current)


def _smil_formats_are_bounded() -> bool:
    from yt_dlp.extractor.common import InfoExtractor
    return getattr(getattr(InfoExtractor, "_parse_smil_formats_and_subtitles", None),
                   _SMIL_SENTINEL, False)


# Every format any builder made (HTML5 <source> tags, JW Player, XSPF, and
# the manifests above) then goes through YoutubeDL.process_video_result's
# sort / select / sanitize pass at ~0.5 ms each, once per entry of the
# result. The formats one extraction hands that pass are counted, across its
# entries, before each pass: ~10 s of CPU at the cap. A count of its own:
# the manifest formats were already charged where they were parsed, and a
# DASH stream of up to _MAX_MPD_REPRESENTATIONS must still pass.
_MAX_EXTRACTION_FORMATS = _MAX_MPD_REPRESENTATIONS
# The pass also rebuilds every format's label (`<format_id>-<n>` when ids
# collide, `format` = "<format_id> - <resolution> (<format_note>)") and
# renders the formats table several times, each row padded to the widest
# cell of every column: text a manifest states once (an AdaptationSet @id or
# @lang every Representation inherits, a rendition NAME every variant of its
# group takes, a StreamIndex Name), or states for ONE format, costs its
# length per format. A 290 kB MPD with a 100 000-character id and 2 000
# Representations peaked at 1 GB, and so did one such id among 2 000 short
# ones. What one extraction hands the pass is sized first, as rows x the
# widest of each label: 20 000 formats with 200 characters of labels fit,
# and at the cap a table is ~4 MB.
_MAX_FORMAT_LABEL_CHARS = 4 * 1024 * 1024
# The text cells of yt-dlp's formats table.
_FORMAT_LABELS = ("format_id", "format_note", "format", "ext", "container",
                  "resolution", "dynamic_range", "audio_channels", "protocol",
                  "vcodec", "acodec", "language")
# yt-dlp's determine_ext: what follows the last "." of an address, if it is
# letters and digits.
_URL_EXT = re.compile(r"[A-Za-z0-9]+$")
_FORMAT_PASS_SENTINEL = "_fwb_format_pass_bounded"


def _labels_too_long():
    from yt_dlp.utils import ExtractorError
    return ExtractorError("the stream's format names are too long",
                          expected=True)


def _padded_chars(rows) -> int:
    """The characters of a table whose rows are each padded to the widest
    cell of every column: its rows x the sum of those widths (never less
    than the cells' sum). `rows` holds each row's cell widths."""
    widest: "list[int]" = []
    count = 0
    for row in rows:
        count += 1
        for pos, chars in enumerate(row):
            if pos == len(widest):
                widest.append(chars)
            elif chars > widest[pos]:
                widest[pos] = chars
    return count * sum(widest)


def _format_label_widths(fmt: dict) -> "list[int]":
    widths = {key: len(value) if isinstance(value := fmt.get(key), str) else 0
              for key in _FORMAT_LABELS}
    # What the pass fills in for a format that states none: the resolution
    # from its sizes, the ext and the protocol from its address (yt-dlp's
    # determine_ext and the URL's scheme).
    chars = sum(_digits(fmt[key]) for key in ("width", "height")
                if isinstance(fmt.get(key), int))
    widths["resolution"] = max(widths["resolution"], chars)
    url = fmt.get("url")
    if isinstance(url, str):
        if not fmt.get("ext"):
            ext = url.partition("?")[0].rpartition(".")[2]
            if _URL_EXT.match(ext):
                widths["ext"] = max(widths["ext"], len(ext))
        if not fmt.get("protocol"):
            widths["protocol"] = max(widths["protocol"], url.find(":"))
    return list(widths.values())


def _format_label_chars(formats) -> int:
    """The label text the pass and the rows of the formats table copy per
    format, before any table is drawn: sized as that table, rows x the
    widest label of each kind."""
    if not isinstance(formats, (list, tuple)):
        return 0  # yt-dlp fails on its own
    return _padded_chars(_format_label_widths(fmt) for fmt in formats
                         if isinstance(fmt, dict))


# The pass also gives EVERY format a `cookies` string of its own, holding all
# the cookies the site has set that match the format's address
# (_calc_headers(load_cookies=True)): 1.4 MB of Set-Cookie on one response
# and a 1 000-variant master peaked at 1.4 GB. Sized first as formats x the
# jar, from a budget of its own: a real jar x 20 000 formats would eat the
# label budget, and at 3 kB still fits this one.
_MAX_FORMAT_COOKIE_CHARS = 64 * 1024 * 1024


def _cookies_too_large():
    from yt_dlp.utils import ExtractorError
    return ExtractorError("the site's cookies are too large", expected=True)


def _jar_chars(ydl) -> int:
    """The `cookies` text the pass builds for ONE format from everything the
    site has set so far, sized a cookie at a time and never joined: per
    cookie `name=value; Domain=; Path=; Secure; Expires=; Version=`, at most
    64 characters around its own. yt-dlp writes the value as a quoted string
    (four characters for each one it escapes), so that is what is measured;
    the expiry is a number of up to ~300 digits. A cookie that does not
    match a given format is counted all the same."""
    jar = getattr(ydl, "cookiejar", None)
    if jar is None:
        return 0  # yt-dlp fails on its own
    from yt_dlp.cookies import LenientSimpleCookie
    encode = LenientSimpleCookie().value_encode
    chars = 0
    for cookie in jar:
        chars += 64 + len(encode(cookie.value)[1]) + sum(
            _cell_chars(part) for part in (
                cookie.name, cookie.domain, cookie.path, cookie.expires,
                cookie.version) if part)
    return chars


def _bounded_process_video_result(original):
    def process_video_result(self, info_dict, *args, **kwargs):
        formats = info_dict.get("formats") if isinstance(info_dict, dict) else None
        # Each call is a format pass, and yt-dlp makes an entry with a url
        # and no formats its own single format.
        try:
            count = max(len(formats) if formats is not None else 0, 1)
        except TypeError:
            count = 1  # yt-dlp fails on its own
        total = getattr(_DIALS.tls, "processed_formats", 0) + count
        if total > _MAX_EXTRACTION_FORMATS:
            raise _too_many_variants(_MAX_EXTRACTION_FORMATS)
        chars = (getattr(_DIALS.tls, "label_chars", 0)
                 + _format_label_chars(formats))
        if chars > _MAX_FORMAT_LABEL_CHARS:
            raise _labels_too_long()
        cookie_chars = (getattr(_DIALS.tls, "cookie_chars", 0)
                        + count * _jar_chars(self))
        if cookie_chars > _MAX_FORMAT_COOKIE_CHARS:
            raise _cookies_too_large()
        _DIALS.tls.processed_formats = total
        _DIALS.tls.label_chars = chars
        _DIALS.tls.cookie_chars = cookie_chars
        return original(self, info_dict, *args, **kwargs)
    setattr(process_video_result, _FORMAT_PASS_SENTINEL, True)
    return process_video_result


def _bound_format_pass() -> None:
    from yt_dlp.YoutubeDL import YoutubeDL
    current = getattr(YoutubeDL, "process_video_result", None)
    # As with DASH: nothing to wrap leaves is_installed() failing closed.
    if callable(current) and not getattr(current, _FORMAT_PASS_SENTINEL, False):
        YoutubeDL.process_video_result = _bounded_process_video_result(current)


def _format_pass_is_bounded() -> bool:
    from yt_dlp.YoutubeDL import YoutubeDL
    return getattr(getattr(YoutubeDL, "process_video_result", None),
                   _FORMAT_PASS_SENTINEL, False)


# yt-dlp draws four tables (formats, thumbnails, subtitles, automatic
# captions) at every print point of an entry, whether or not a listing was
# asked for, and every one through the `render_table` its YoutubeDL module
# imports: each row is padded to the widest cell of every column, so ONE
# long cell costs its length per row. One 200 000-character subtitle language
# among 2 000 (a 485 kB DASH manifest) peaked at 1.2 GB. Every table is sized
# there, before it is drawn, against the label budget; real subtitle and
# thumbnail lists are a few kB.
# An entry draws the same tables again at each of its print points (eight at
# most), so a draw cannot spend from the label budget itself: 20 000 ordinary
# formats are over half of it once. What one extraction draws has a budget
# of its own, sixteen full tables.
_MAX_TABLE_CHARS = 16 * _MAX_FORMAT_LABEL_CHARS
_TABLE_SENTINEL = "_fwb_table_bounded"
# The YoutubeDL methods that draw a table.
_TABLE_DRAWERS = ("render_formats_table", "render_thumbnails_table",
                  "render_subtitles_table")


def _cell_chars(cell) -> int:
    if isinstance(cell, str):
        return len(cell)
    if isinstance(cell, int):
        return _digits(cell)  # no conversion: see _digits
    return len(str(cell))


def _row_chars(row) -> "list[int]":
    return [_cell_chars(cell) for cell in row]


def _tab_chars(header_row, rows) -> int:
    """What render_table draws on top of the padded rows: it replaces EVERY
    tab of a cell with the column's width less the cell's own, so a cell of
    t tabs draws up to t x that width. Each tab after a cell's first costs
    the column's widest cell (the header's included); the first is the
    padding itself."""
    widest = [_cell_chars(cell) for cell in header_row]
    tabs = [0] * len(widest)
    for row in rows:
        for pos, cell in enumerate(row):
            if pos == len(widest):
                widest.append(0)
                tabs.append(0)
            if isinstance(cell, int):
                chars = _digits(cell)
            else:
                if not isinstance(cell, str):
                    cell = str(cell)
                chars = len(cell)
                more = cell.count("\t") - 1
                if more > 0:
                    tabs[pos] += more
            if chars > widest[pos]:
                widest[pos] = chars
    return sum(t * w for t, w in zip(tabs, widest) if t)


def _bounded_render_table(original):
    def render_table(header_row, data, *args, **kwargs):
        rows = data if isinstance(data, (list, tuple)) else list(data)
        chars = _padded_chars(map(_row_chars, rows)) + _tab_chars(header_row, rows)
        total = getattr(_DIALS.tls, "table_chars", 0) + chars
        if chars > _MAX_FORMAT_LABEL_CHARS or total > _MAX_TABLE_CHARS:
            raise _labels_too_long()
        _DIALS.tls.table_chars = total
        return original(header_row, rows, *args, **kwargs)
    setattr(render_table, _TABLE_SENTINEL, True)
    return render_table


def _ydl_module():
    # `yt_dlp.YoutubeDL` the attribute is the class; the module is this.
    import importlib
    return importlib.import_module("yt_dlp.YoutubeDL")


def _bound_tables() -> None:
    module = _ydl_module()
    current = vars(module).get("render_table")
    # As with DASH: nothing to wrap leaves is_installed() failing closed.
    if callable(current) and not getattr(current, _TABLE_SENTINEL, False):
        module.render_table = _bounded_render_table(current)


def _tables_are_bounded() -> bool:
    module = _ydl_module()
    if not getattr(vars(module).get("render_table"), _TABLE_SENTINEL, False):
        return False
    # A yt-dlp whose lists stop looking up the module's `render_table`
    # would bypass the swap: each must still name it, as a global of that
    # module.
    for name in _TABLE_DRAWERS:
        fn = getattr(module.YoutubeDL, name, None)
        code = getattr(fn, "__code__", None)
        if (code is None or "render_table" not in code.co_names
                or getattr(fn, "__globals__", None) is not vars(module)):
            return False
    return True


# What a page builds before any count above sees it. GenericIE parses every
# page it scrapes as XML first (an RSS / SMIL / MPD check) and every ld+json
# block with json.loads, and both run on up to the 64 MiB body cap: ~1.6 GB of
# element tree or of lists and dicts, seconds of CPU on the probe thread. The
# elements (start tags) and the JSON arrays and objects are counted in C
# first, and so are the attributes (every one needs an `=`, so their count
# can only overcount), against twice the element cap. JSON scalars cost an
# object each just the same, and every array item and object member after
# the first needs a `,`; an object member also needs a `:` and builds a key
# and a value. Commas and colons are charged against twice the container
# cap too (those inside strings only overcount). expat builds an
# internal entity's markup once per reference, where no count of the raw text
# sees it, so a document that declares entities is refused outright; none of
# the manifest formats yt-dlp parses declares any. An attribute default
# (<!ATTLIST) is the same amplifier: expat gives every matching element a
# fresh copy of it (a 120 KB document of 5 000 elements under a
# 100 000-character default held 534 MB), so a document that declares one is
# refused too. No other declaration puts content into the tree. With both
# refused, a document no longer than the cap cannot exceed it and pays
# nothing. Two million elements is twice a DASH manifest at the segment cap
# (its attributes fit the doubled cap), and real pages and API answers hold
# far fewer arrays and objects.
_MAX_XML_ELEMENTS = 2_000_000
_MAX_JSON_CONTAINERS = 2_000_000
_ETREE_SENTINEL = "_fwb_etree_bounded"
_ETREE_NAME = "compat_etree_fromstring"
_JSON_SENTINEL = "_fwb_json_bounded"


_DTD_DECLS = ("<!ENTITY", "<!ATTLIST")
# expat also reads UTF-16 (either byte order), where a declaration's bytes
# differ; literal productions, they have no other spelling.
_DTD_DECL_BYTES = tuple(decl.encode(encoding) for decl in _DTD_DECLS
                        for encoding in ("utf-8", "utf-16-le", "utf-16-be"))


def _refuse_oversized_xml(text) -> None:
    if isinstance(text, str):
        declares = any(decl in text for decl in _DTD_DECLS)
        lt, close, eq = "<", "</", "="
    elif isinstance(text, (bytes, bytearray)):
        declares = any(decl in text for decl in _DTD_DECL_BYTES)
        lt, close, eq = b"<", b"</", b"="
    else:
        return
    # What a malformed document raises: every caller handles it (GenericIE
    # goes on scraping the page as HTML).
    if declares:
        raise xml.etree.ElementTree.ParseError(
            "the document declares entities or attribute defaults")
    if len(text) <= _MAX_XML_ELEMENTS:
        return
    elements = text.count(lt) - text.count(close)
    if (elements > _MAX_XML_ELEMENTS
            or elements + text.count(eq) > 2 * _MAX_XML_ELEMENTS):
        raise xml.etree.ElementTree.ParseError(
            f"the document has too many elements (more than {_MAX_XML_ELEMENTS})")


def _bounded_etree(original):
    def compat_etree_fromstring(text):
        _refuse_oversized_xml(text)
        return original(text)
    setattr(compat_etree_fromstring, _ETREE_SENTINEL, True)
    return compat_etree_fromstring


def _etree_bindings():
    """(module, function) for every loaded yt-dlp module that holds its own
    reference to compat_etree_fromstring (`from ..compat import ...`)."""
    for name, module in dict(sys.modules).items():
        if name.partition(".")[0] not in ("yt_dlp", "yt_dlp_plugins"):
            continue
        fn = getattr(module, "__dict__", {}).get(_ETREE_NAME)
        if fn is not None:
            yield module, fn


def _bound_etree_parses() -> None:
    import yt_dlp.compat
    # The compat module first: a module imported later binds the bounded
    # function. Every module already holding a reference is rebound.
    wrapped = {}
    for module, fn in [(yt_dlp.compat, vars(yt_dlp.compat).get(_ETREE_NAME)),
                       *_etree_bindings()]:
        if callable(fn) and not getattr(fn, _ETREE_SENTINEL, False):
            if id(fn) not in wrapped:
                wrapped[id(fn)] = _bounded_etree(fn)
            setattr(module, _ETREE_NAME, wrapped[id(fn)])


def _etree_parses_are_bounded() -> bool:
    import yt_dlp.compat
    # A yt-dlp without the compat function fails closed, as with DASH.
    return (getattr(vars(yt_dlp.compat).get(_ETREE_NAME), _ETREE_SENTINEL, False)
            and all(getattr(fn, _ETREE_SENTINEL, False)
                    for _module, fn in _etree_bindings()))


def _json_oversized(doc) -> bool:
    if isinstance(doc, str):
        containers = doc.count("[") + doc.count("{")
        items = doc.count(",") + doc.count(":")
    else:
        containers = doc.count(b"[") + doc.count(b"{")
        items = doc.count(b",") + doc.count(b":")
    return (containers > _MAX_JSON_CONTAINERS
            or containers + items > 2 * _MAX_JSON_CONTAINERS)


def _bounded_parse_json(original):
    signature = inspect.signature(original)

    def _parse_json(self, json_string, *args, **kwargs):
        if (isinstance(json_string, (str, bytes, bytearray))
                and len(json_string) > _MAX_JSON_CONTAINERS
                and _json_oversized(json_string)):
            # Refused as yt-dlp refuses JSON it cannot parse: fatal raises,
            # otherwise a warning (unless errnote is False) and None.
            from yt_dlp.utils import ExtractorError
            call = signature.bind(self, json_string, *args, **kwargs)
            call.apply_defaults()
            fatal = call.arguments.get("fatal", True)
            errnote = call.arguments.get("errnote")
            video_id = call.arguments.get("video_id")
            error = ValueError("the JSON holds too many arrays, objects or items "
                               f"(more than {_MAX_JSON_CONTAINERS})")
            errnote = "Failed to parse JSON" if errnote is None else errnote
            if fatal:
                raise ExtractorError(errnote, cause=error, video_id=video_id)
            if errnote:
                self.report_warning(f"{errnote}: {error}", video_id=video_id)
            return None
        return original(self, json_string, *args, **kwargs)
    setattr(_parse_json, _JSON_SENTINEL, True)
    return _parse_json


def _bound_json_parses() -> None:
    from yt_dlp.extractor.common import InfoExtractor
    current = getattr(InfoExtractor, "_parse_json", None)
    # As with DASH: nothing to wrap leaves is_installed() failing closed.
    if callable(current) and not getattr(current, _JSON_SENTINEL, False):
        InfoExtractor._parse_json = _bounded_parse_json(current)


def _json_parses_are_bounded() -> bool:
    from yt_dlp.extractor.common import InfoExtractor
    return getattr(getattr(InfoExtractor, "_parse_json", None),
                   _JSON_SENTINEL, False)


# The download child's FFmpeg postprocessors — the fixups yt-dlp runs by
# default (FixupM3u8 on every HLS download into mp4 / m4a) and a video's
# merger — spawn ffmpeg and ffprobe on the downloaded file, also to read its
# codec and format (get_audio_codec, get_metadata_object). Its bytes are the
# site's, and ffmpeg picks the demuxer from them: a DASH manifest or an HLS
# playlist opens the dash / hls demuxer, which fetches the URLs it names
# (ffmpeg 7 does even under -protocol_whitelist file), and concat opens the
# files it lists. `-format_whitelist` before each input limits ffmpeg to the
# containers a download yields; one that probes as anything else fails to
# open, as a file ffmpeg cannot read fails. The CLI cannot set it: yt-dlp's
# postprocessor args reach real_run_ffmpeg only per postprocessor, and the
# two probes take none. Every spawn in yt_dlp.postprocessor.ffmpeg goes
# through that module's own `Popen` name, which is swapped for a subclass
# that rewrites the argv. FFmpegConcatPP / split / remove-chapters force the
# concat demuxer and would fail under it; the download never enables them.
_FFMPEG_FORMAT_WHITELIST = ",".join((
    "mov,mp4,m4a,3gp,3g2,mj2",       # one demuxer: the ISO BMFF family
    "matroska,webm",                  # one demuxer: mkv / webm
    "ogg", "mpegts", "flv", "live_flv",
    # HLS packed audio, and direct audio links.
    "mp3", "aac", "ac3", "eac3", "flac", "wav",
))
# The same file also decides what those spawns allocate: the mov demuxer
# builds its index for every sample a track DECLARES, so a 4.6 KB file made
# `ffmpeg -i`, ffprobe and the HLS fixup each peak at 620 MB, and forty
# tracks at 770 MB (ffmpeg's own limits: INT_MAX per allocation, 1000
# streams). Every spawn gets the cap and the stream limit the server's own
# opens of a client file have: faster_whisper_backend/audio/ffmpeg.py
# _LIBAV_MAX_ALLOC and PYAV_MAX_STREAMS, repeated here because the download
# child cannot import the package (a test pins them equal). An input past
# the stream limit fails to open, as a file ffmpeg cannot read fails. A
# download has one or two streams per file, and the server's own open of the
# result holds it to the same two numbers afterwards.
_FFMPEG_MAX_ALLOC = 256 * 1024 * 1024
_FFMPEG_MAX_STREAMS = 20
_FFMPEG_SENTINEL = "_fwb_ffmpeg_inputs_bounded"
# The FFmpegPostProcessor methods that spawn ffmpeg / ffprobe on a file.
_FFMPEG_SPAWNERS = ("real_run_ffmpeg", "get_audio_codec", "get_metadata_object")


def _whitelisted_ffmpeg_argv(args):
    """`args` with `-max_alloc` first (a global option) and with
    `-format_whitelist` and `-max_streams` before every ffmpeg input (`-i`),
    or ahead of everything else for ffprobe, whose single input is its last
    argument."""
    if not isinstance(args, (list, tuple)) or not args:
        # Every spawn there is an argv list; anything else cannot be
        # rewritten and must not run unrestricted.
        raise ValueError(f"{MARKER}: refusing an ffmpeg command that is not an argv list")
    flag = ["-format_whitelist", _FFMPEG_FORMAT_WHITELIST,
            "-max_streams", str(_FFMPEG_MAX_STREAMS)]
    exe, *rest = args
    argv = [exe, "-max_alloc", str(_FFMPEG_MAX_ALLOC)]
    if "ffprobe" in os.path.basename(os.fsdecode(exe)).lower():
        return [*argv, *flag, *rest]
    for arg in rest:
        if arg == "-i":
            argv += flag
        argv.append(arg)
    return argv


def _whitelisting_popen(base):
    class Popen(base):
        def __init__(self, args, *a, **kw):
            super().__init__(_whitelisted_ffmpeg_argv(args), *a, **kw)
    setattr(Popen, _FFMPEG_SENTINEL, True)
    return Popen


def _bound_ffmpeg_inputs() -> None:
    from yt_dlp.postprocessor import ffmpeg
    current = vars(ffmpeg).get("Popen")
    # As with DASH: nothing to wrap leaves is_installed() failing closed.
    if isinstance(current, type) and not getattr(current, _FFMPEG_SENTINEL, False):
        ffmpeg.Popen = _whitelisting_popen(current)


def _ffmpeg_inputs_are_bounded() -> bool:
    from yt_dlp.postprocessor import ffmpeg
    if not getattr(vars(ffmpeg).get("Popen"), _FFMPEG_SENTINEL, False):
        return False
    # A yt-dlp whose spawners stop looking up the module's `Popen` would
    # bypass the swap: each must still name it, as a global of that module.
    for name in _FFMPEG_SPAWNERS:
        fn = getattr(ffmpeg.FFmpegPostProcessor, name, None)
        code = getattr(fn, "__code__", None)
        if (code is None or "Popen" not in code.co_names
                or getattr(fn, "__globals__", None) is not vars(ffmpeg)):
            return False
    return True


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
    for key in [k for k in _REQUEST_HANDLERS if k != GUARD_RH_KEY]:
        _REQUEST_HANDLERS.pop(key, None)
    _refuse_external_downloaders()
    _bound_mpd_expansion()
    _bound_ism_expansion()
    _bound_m3u8_formats()
    _bound_hls_fragments()
    _bound_f4m_formats()
    _bound_f4m_fragments()
    _bound_smil_formats()
    _bound_format_pass()
    _bound_tables()
    _bound_etree_parses()
    _bound_json_parses()
    _bound_ffmpeg_inputs()


def _prefer_guard(rh, request):
    # Beats requests (100), urllib (0) and curl_cffi (-100) for every request.
    return 1000


def is_installed() -> bool:
    """True when the guard is the only request handler registered (an
    allowlist: an unknown handler key fails closed), no downloader can hand
    a fetch to an external program, a DASH or Smooth Streaming manifest's
    segments are counted before yt-dlp expands them, an HLS master's
    variants and lines before it builds their formats, an HLS media
    playlist's segments before HlsFD expands them, an HDS manifest's nodes
    and nested manifests before it builds their formats, an HDS bootstrap's
    bytes before F4mFD parses it and its fragments before F4mFD expands them, a SMIL document's srcs, addresses
    and nested manifests before it builds their formats, the formats of one
    extraction before each format pass, every table of formats, thumbnails
    or subtitles before it is drawn, an XML document's elements and
    a JSON document's arrays and objects before either is parsed, and every
    ffmpeg / ffprobe a postprocessor spawns carries the format whitelist,
    the allocation cap and the stream limit."""
    rh = _REQUEST_HANDLERS.get(GUARD_RH_KEY)
    return (rh is not None and getattr(rh, "RH_NAME", None) == RH_NAME
            and set(_REQUEST_HANDLERS) == {GUARD_RH_KEY}
            and all(getattr(cls.real_download, _FD_SENTINEL, False)
                    for cls in _refusing_fd_classes())
            and _mpd_expansion_is_bounded()
            and _ism_expansion_is_bounded()
            and _m3u8_formats_are_bounded()
            and _hls_fragments_are_bounded()
            and _f4m_formats_are_bounded()
            and _f4m_fragments_are_bounded()
            and _smil_formats_are_bounded()
            and _format_pass_is_bounded()
            and _tables_are_bounded()
            and _etree_parses_are_bounded()
            and _json_parses_are_bounded()
            and _ffmpeg_inputs_are_bounded())


install()
