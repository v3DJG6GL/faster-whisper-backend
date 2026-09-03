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
