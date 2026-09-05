"""Which outbound addresses this server refuses to fetch from — ONE definition.

Why this is its own module: the SSRF policy has to be enforced in two places
that cannot reach each other through a normal import.

  * url/download.py enforces it IN-PROCESS — the direct-media probe, the
    thumbnail fetch and their shared redirect handler.
  * ytdlp_plugins/ enforces it INSIDE `python -m yt_dlp`, a separate process
    that must not have the repo root on its sys.path (repo-root directories
    such as ``static/`` or a bind-mounted ``secrets/`` would shadow stdlib
    modules for yt-dlp's ~2000 extractors). That guard therefore loads THIS
    FILE BY PATH, computed from the plugin's own location.

Two copies of a range list drift apart on the first review; one module that
both sides load cannot. Everything here is stdlib-only for exactly that
reason — the path-loading side has nothing else available.

The predicate is deliberately allow-nothing-unknown: a name that does not
resolve, or that resolves to anything we cannot parse, counts as forbidden.
"""
from __future__ import annotations

import http.client
import ipaddress
import socket

# Carrier-grade NAT (RFC 6598). `ipaddress` has no property for it, yet it is
# exactly as internal as RFC1918 from a hosted backend's point of view.
CGNAT_NET = ipaddress.ip_network("100.64.0.0/10")


def address_is_forbidden(addr: str) -> bool:
    """THE policy: True when this literal address is one we never fetch from.

    Covers loopback (127/8, ::1), RFC1918 (10/8, 172.16/12, 192.168/16), ULA
    (fc00::/7), link-local (169.254/16 — cloud metadata — and fe80::/10),
    CGNAT (100.64/10), multicast, reserved and the unspecified address.
    An IPv4-mapped IPv6 literal (::ffff:127.0.0.1) is judged as its IPv4 half,
    so the mapping can't be used to smuggle an internal target past the gate.
    """
    try:
        ip = ipaddress.ip_address(addr.split("%", 1)[0])  # strip zone id
    except ValueError:
        return True
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
        return True
    return isinstance(ip, ipaddress.IPv4Address) and ip in CGNAT_NET


def host_is_forbidden(host: str) -> bool:
    """True when `host` resolves to ANY address we refuse to fetch from.

    Any answer being forbidden condemns the whole name: a dual-stack host
    with one public and one internal record must not be reachable by letting
    the client pick which one the connect happens to use. Resolution failure
    counts as forbidden too (we can't vouch for what we can't look up)."""
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError:
        return True
    if not infos:
        return True
    return any(address_is_forbidden(info[4][0]) for info in infos)


def resolve_pinned(host: str, port: int, *, trusted: bool = False) -> list:
    """Resolve ONCE; the returned sockaddrs ARE the pin.

    Refuses the whole name (OSError) when it does not resolve or when ANY
    answer is a forbidden address — same verdict as host_is_forbidden, but
    the caller dials one of exactly these addresses instead of letting
    http.client re-resolve (a rebinding name could answer differently).

    `trusted=True` skips the address policy (still resolves once, still
    refuses an unresolvable name): it is for the operator's own HTTP proxy,
    which is where the socket goes when http(s)_proxy is set. The URL's real
    target is still policy-gated by name on every hop (host_is_forbidden /
    the guard's _check_url); only the pin cannot reach past a proxy, because
    the proxy does the final resolve."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        infos = []
    if not infos or (not trusted
                     and any(address_is_forbidden(i[4][0]) for i in infos)):
        raise OSError(f"{host}: forbidden or unresolvable address")
    return infos


def dials_a_proxy(conn) -> bool:
    """True when conn.host is an operator-configured proxy, not the URL's
    host: an HTTPS CONNECT tunnel (stdlib sets _tunnel_host) or a plain-HTTP
    proxied request (flagged via_proxy by the handler that built conn)."""
    return bool(getattr(conn, "via_proxy", False)
                or getattr(conn, "_tunnel_host", None))


def proxied_conn_factory(conn_class, via_proxy: bool):
    """http_class stand-in for urllib's do_open() that stamps `via_proxy` on
    the connection it builds (do_open only ever passes the host)."""
    def make(host, **kw):
        conn = conn_class(host, **kw)
        conn.via_proxy = via_proxy
        return conn
    return make


def connect_pinned(conn) -> socket.socket:
    """http.client-compatible connect: dial the pinned answers for
    conn.host:conn.port, honouring conn.timeout / conn.source_address."""
    last = None
    infos = resolve_pinned(conn.host, conn.port, trusted=dials_a_proxy(conn))
    for family, socktype, proto, _canon, sockaddr in infos:
        sock = socket.socket(family, socktype, proto)
        try:
            if conn.timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:  # type: ignore[attr-defined]
                sock.settimeout(conn.timeout)
            if conn.source_address:
                sock.bind(conn.source_address)
            sock.connect(sockaddr)
        except OSError as e:
            sock.close()
            last = e
            continue
        try:  # what stdlib's HTTPConnection.connect does after connecting
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        return sock
    raise last if last is not None else OSError("connection failed")


# Only connect() is overridden: the Host header, request line and certificate
# validation still see the NAME the URL carried.
class PinnedHTTPConnection(http.client.HTTPConnection):
    via_proxy = False  # set by proxied_conn_factory for plain-HTTP proxying

    def connect(self):
        self.sock = connect_pinned(self)
        if self._tunnel_host:
            self._tunnel()


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    via_proxy = False

    def connect(self):
        sock = connect_pinned(self)
        if self._tunnel_host:
            self.sock = sock
            self._tunnel()
            sock = self.sock
        # SNI / cert verification on the NAME, never the pinned literal;
        # behind a CONNECT proxy the URL's host is _tunnel_host.
        self.sock = self._context.wrap_socket(
            sock, server_hostname=self._tunnel_host or self.host)
