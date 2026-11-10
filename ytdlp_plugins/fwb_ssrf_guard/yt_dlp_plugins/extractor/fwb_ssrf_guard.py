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
    Merging is unaffected (FFmpegMergerPP is a postprocessor on local files).

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
        return super()._send(request)

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
# line) and an HDS bootstrap (F4mFD) count against the same cap. The DASH and
# Smooth Streaming manifests one extraction parses also spend from one running
# total (_charge), as every manifest within its own cap is not enough.
_MAX_MPD_SEGMENTS = 1_000_000
_MPD_SENTINEL = "_fwb_mpd_bounded"
# Each Representation also becomes its own format (periods merge only when
# every field but the fragments matches), and yt-dlp's parse + format pass
# cost grows faster than linearly in their number: 20 000 one-segment
# representations take ~20 s of CPU, a 64 MiB manifest hours. A 12 h stream
# split into 720 ad periods of ten representations each is 7 200. A Smooth
# Streaming QualityLevel is a variant too, from the same budget.
_MAX_MPD_REPRESENTATIONS = 20_000


def _too_many_segments():
    from yt_dlp.utils import ExtractorError
    return ExtractorError(
        f"the stream lists too many segments (more than {_MAX_MPD_SEGMENTS})",
        expected=True)


def _too_many_variants(cap: int):
    from yt_dlp.utils import ExtractorError
    return ExtractorError(
        f"the stream lists too many variants (more than {cap})", expected=True)


def _charge(segments: int, variants: int) -> None:
    """Spend a manifest's counts from the budget of the extraction running
    on this thread. One extraction can parse many manifests into one result
    (GenericIE parses an MPD per Video.js `src`), so each manifest within
    its own cap is not enough. A refused manifest spends nothing. probe()
    resets the budget per extraction (observe_dials); the download child
    extracts once per process."""
    total = getattr(_DIALS.tls, "segments", 0) + segments
    if total > _MAX_MPD_SEGMENTS:
        raise _too_many_segments()
    kinds = getattr(_DIALS.tls, "variants", 0) + variants
    if kinds > _MAX_MPD_REPRESENTATIONS:
        raise _too_many_variants(_MAX_MPD_REPRESENTATIONS)
    _DIALS.tls.segments = total
    _DIALS.tls.variants = kinds


def _number(value, kind=float):
    """`value` as yt-dlp parses it, or None where yt-dlp's own parse raises
    on it (it then fails on its own, after the count)."""
    try:
        return kind(value)
    except (TypeError, ValueError, OverflowError):
        return None


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
        info["s"] = [(_number(e.get("d"), int) or 0,
                      max(_number(e.get("r", 0), int) or 0, 0)) for e in entries]
    timescale = source.get("timescale")
    if timescale:
        info["timescale"] = _number(timescale, int)
    duration = source.get("duration")
    if duration:
        info["duration"] = _number(duration)
    if is_list:
        urls = source.findall(q("SegmentURL"))
        if urls:
            info["urls"] = len(urls)
    elif source.get("media"):
        info["media"] = True
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


def _refuse_oversized_mpd(mpd_doc) -> None:
    from yt_dlp.utils import parse_duration
    # yt-dlp's own derivation: only a root named MPD (any case) lends its
    # namespace; under any other root it expands unqualified Periods, so
    # that is where the count has to look too.
    match = re.match(r"(?i)^{([^}]+)?}MPD$", getattr(mpd_doc, "tag", "") or "")
    ns = (match.group(1) or "") if match else ""

    def q(name):
        return f"{{{ns}}}{name}" if ns else name
    mpd_duration = parse_duration(mpd_doc.get("mediaPresentationDuration"))
    count = representations = 0
    for period in mpd_doc.findall(q("Period")):
        period_duration = (parse_duration(period.get("duration"))
                           or mpd_duration)
        period_info = _mpd_segment_info(period, {"timescale": 1}, q)
        for adaptation in period.findall(q("AdaptationSet")):
            set_info = _mpd_segment_info(adaptation, period_info, q)
            for representation in adaptation.findall(q("Representation")):
                representations += 1
                if representations > _MAX_MPD_REPRESENTATIONS:
                    raise _too_many_variants(_MAX_MPD_REPRESENTATIONS)
                info = _mpd_segment_info(representation, set_info, q)
                n, seconds = _representation_segments(info, period_duration)
                count += n
                if count > _MAX_MPD_SEGMENTS:
                    raise _too_many_segments()
                # yt-dlp's own fallback: a period with no length takes the
                # first expanded representation's, which the next ones use.
                if not period_duration and n and seconds:
                    period_duration = seconds
    _charge(count, representations)


def _bounded_mpd_periods(original):
    def _parse_mpd_periods(self, mpd_doc, *args, **kwargs):
        _refuse_oversized_mpd(mpd_doc)
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


def _refuse_oversized_ism(ism_doc) -> None:
    from yt_dlp.utils import int_or_none
    if ism_doc.get("IsLive") == "TRUE":
        return  # yt-dlp returns no formats for it, before any expansion
    count = variants = 0
    for stream in ism_doc.findall("StreamIndex"):
        if stream.get("Type") not in ("video", "audio", "text"):
            continue
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
        # yt-dlp: range(int_or_none(r) or 1) per chunk, so r <= -1 builds no
        # fragment, but the chunk is still read once per level: it costs one.
        per_track = sum(max(int_or_none(c.get("r")) or 1, 1)
                        for c in stream.findall("c"))
        count += tracks * per_track
        if count > _MAX_MPD_SEGMENTS:
            raise _too_many_segments()
    _charge(count, variants)


def _bounded_ism(original):
    def _parse_ism_formats_and_subtitles(self, ism_doc, *args, **kwargs):
        _refuse_oversized_ism(ism_doc)
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
_M3U8_FORMAT_LINE = re.compile(
    rf"(?<![^{_M3U8_BREAKS}])(?:#EXT-X-MEDIA:"
    rf"|#EXT-X-STREAM-INF:[^{_M3U8_BREAKS}]*?PROGRESSIVE-URI"
    rf"|[^\s#]|[^\S{_M3U8_BREAKS}]+\S)")


def _line_breaks(doc: str) -> int:
    """How many lines str.splitlines() would end in `doc`, counted in C."""
    return (sum(doc.count(b) for b in "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029")
            - doc.count("\r\n"))


def _refuse_oversized_m3u8(m3u8_doc) -> None:
    from yt_dlp.utils import ExtractorError
    if not isinstance(m3u8_doc, str) or "#EXT-X-TARGETDURATION" in m3u8_doc:
        return
    if _line_breaks(m3u8_doc) > _MAX_M3U8_MASTER_LINES:
        raise ExtractorError(
            "the playlist has too many lines (more than "
            f"{_MAX_M3U8_MASTER_LINES})", expected=True)
    for count, _match in enumerate(_M3U8_FORMAT_LINE.finditer(m3u8_doc), 1):
        if count > _MAX_M3U8_FORMATS:
            raise ExtractorError(
                f"the stream lists too many variants (more than {_MAX_M3U8_FORMATS})",
                expected=True)


def _bounded_m3u8(original):
    def _parse_m3u8_formats_and_subtitles(self, m3u8_doc, *args, **kwargs):
        _refuse_oversized_m3u8(m3u8_doc)
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


def _refuse_oversized_hls_media(manifest) -> None:
    # A playlist with no more lines than the cap cannot list more fragments:
    # every real one pays only the C-level line count.
    if not isinstance(manifest, str) or _line_breaks(manifest) < _MAX_MPD_SEGMENTS:
        return
    for count, _match in enumerate(_HLS_FRAGMENT_LINE.finditer(manifest), 1):
        if count > _MAX_MPD_SEGMENTS:
            raise _too_many_segments()


def _bounded_hls_can_download(original):
    def can_download(cls, manifest, *args, **kwargs):
        _refuse_oversized_hls_media(manifest)
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
# bounded too. Real HDS lists a handful of bitrates, and a set-level manifest
# a few stream-level ones one level down.
_F4M_SENTINEL = "_fwb_f4m_bounded"
_F4M_NAMESPACES = ("{http://ns.adobe.com/f4m/1.0}", "{http://ns.adobe.com/f4m/2.0}")
_MAX_F4M_NESTED = 16
_MAX_F4M_DEPTH = 2


def _refuse_oversized_f4m(manifest) -> None:
    if not isinstance(manifest, xml.etree.ElementTree.Element):
        return  # yt-dlp returns or fails on its own, before any node
    # yt-dlp's choice: the 1.0 nodes, else the 2.0 ones.
    nodes = (manifest.findall(_F4M_NAMESPACES[0] + "media")
             or manifest.findall(_F4M_NAMESPACES[1] + "media"))
    if len(nodes) > _MAX_M3U8_FORMATS:
        raise _too_many_variants(_MAX_M3U8_FORMATS)
    if any(manifest.find(ns + "bootstrapInfo") is not None
           for ns in _F4M_NAMESPACES):
        return  # a stream-level manifest refers to nothing outside it
    from yt_dlp.utils import determine_ext
    nested = sum(1 for node in nodes
                 if determine_ext(node.get("href") or node.get("url") or "")
                 in ("f4m", "m3u8"))
    if nested > _MAX_F4M_NESTED:
        raise _too_many_variants(_MAX_F4M_NESTED)


def _bounded_f4m(original):
    def _parse_f4m_formats(self, manifest, *args, **kwargs):
        depth = getattr(_DIALS.tls, "f4m_depth", 0)
        if depth >= _MAX_F4M_DEPTH:
            raise _too_many_variants(_MAX_F4M_NESTED)
        _refuse_oversized_f4m(manifest)
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


def _bound_f4m_fragments() -> None:
    from yt_dlp.downloader import f4m
    current = getattr(f4m, "build_fragments_list", None)
    # As with DASH: nothing to wrap leaves is_installed() failing closed.
    if callable(current) and not getattr(current, _F4M_FRAGMENTS_SENTINEL, False):
        f4m.build_fragments_list = _bounded_build_fragments_list(current)


def _f4m_fragments_are_bounded() -> bool:
    from yt_dlp.downloader import f4m
    return getattr(getattr(f4m, "build_fragments_list", None),
                   _F4M_FRAGMENTS_SENTINEL, False)


# What a page builds before any count above sees it. GenericIE parses every
# page it scrapes as XML first (an RSS / SMIL / MPD check) and every ld+json
# block with json.loads, and both run on up to the 64 MiB body cap: ~1.6 GB of
# element tree or of lists and dicts, seconds of CPU on the probe thread. The
# elements (start tags) and the JSON arrays and objects are counted in C
# first; a document no longer than the cap cannot exceed it and pays nothing.
# Two million elements is twice a DASH manifest at the segment cap, and real
# pages and API answers hold far fewer arrays and objects.
_MAX_XML_ELEMENTS = 2_000_000
_MAX_JSON_CONTAINERS = 2_000_000
_ETREE_SENTINEL = "_fwb_etree_bounded"
_ETREE_NAME = "compat_etree_fromstring"
_JSON_SENTINEL = "_fwb_json_bounded"


def _refuse_oversized_xml(text) -> None:
    if not isinstance(text, (str, bytes, bytearray)) or len(text) <= _MAX_XML_ELEMENTS:
        return
    lt, close = ("<", "</") if isinstance(text, str) else (b"<", b"</")
    if text.count(lt) - text.count(close) > _MAX_XML_ELEMENTS:
        # What a malformed document raises: every caller handles it (GenericIE
        # goes on scraping the page as HTML).
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


def _json_containers(doc) -> int:
    if isinstance(doc, str):
        return doc.count("[") + doc.count("{")
    return doc.count(b"[") + doc.count(b"{")


def _bounded_parse_json(original):
    signature = inspect.signature(original)

    def _parse_json(self, json_string, *args, **kwargs):
        if (isinstance(json_string, (str, bytes, bytearray))
                and len(json_string) > _MAX_JSON_CONTAINERS
                and _json_containers(json_string) > _MAX_JSON_CONTAINERS):
            # Refused as yt-dlp refuses JSON it cannot parse: fatal raises,
            # otherwise a warning (unless errnote is False) and None.
            from yt_dlp.utils import ExtractorError
            call = signature.bind(self, json_string, *args, **kwargs)
            call.apply_defaults()
            fatal = call.arguments.get("fatal", True)
            errnote = call.arguments.get("errnote")
            video_id = call.arguments.get("video_id")
            error = ValueError("the JSON holds too many arrays and objects "
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
    _bound_etree_parses()
    _bound_json_parses()


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
    fragments before F4mFD expands them, and an XML document's elements and
    a JSON document's arrays and objects before either is parsed."""
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
            and _etree_parses_are_bounded()
            and _json_parses_are_bounded())


install()
