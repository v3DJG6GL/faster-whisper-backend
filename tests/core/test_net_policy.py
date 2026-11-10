"""net_policy.address_is_forbidden: the literal-address half of the yt-dlp
SSRF policy (the end-to-end guard lives in tests/media/test_ssrf_guard.py)."""

import socket
import time
import types

import pytest

from faster_whisper_backend.core import net_policy


def test_internal_ranges_are_forbidden():
    for addr in ("127.0.0.1", "10.1.2.3", "169.254.169.254", "100.64.0.1",
                 "::1", "fc00::1", "fe80::1%eth0", "::ffff:127.0.0.1", "::"):
        assert net_policy.address_is_forbidden(addr), addr


def test_deprecated_ipv6_site_local_is_forbidden():
    # fec0::/10 is none of private/reserved/link-local to ipaddress, yet a
    # network that still routes it reaches internal hosts through it.
    assert net_policy.address_is_forbidden("fec0::1")
    assert net_policy.address_is_forbidden("feff::1")


def test_public_addresses_pass():
    for addr in ("8.8.8.8", "2606:4700:4700::1111"):
        assert not net_policy.address_is_forbidden(addr), addr


def test_a_host_idna_cannot_encode_fails_closed_not_raises():
    """getaddrinfo raises UnicodeError (not OSError) for an empty label, a
    label over 63 characters or a lone surrogate; that must count as
    forbidden / unresolvable instead of escaping as a 500."""
    for host in ("a..com", "a" * 64 + ".com", "\udcff.com"):
        assert net_policy.host_is_forbidden(host), host
        with pytest.raises(OSError):
            net_policy.resolve_pinned(host, 443)


def test_nat64_is_judged_by_its_embedded_ipv4():
    """A DNS64 resolver answers 64:ff9b::<ipv4> for every IPv4-only site;
    the public ones must pass, the internal ones must not."""
    for addr in ("64:ff9b::808:808", "64:ff9b::8.8.8.8"):
        assert not net_policy.address_is_forbidden(addr), addr
    for addr in ("64:ff9b::7f00:1", "64:ff9b::a00:1", "64:ff9b::a9fe:a9fe",
                 "64:ff9b::6440:1"):
        assert net_policy.address_is_forbidden(addr), addr


def _black_holed(monkeypatch, n):
    """`n` public-looking answers for every name, each dropping the SYN: a
    connect sleeps its whole timeout, then times out. Returns the attempt log."""
    answers = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (f"203.0.113.{i}", 80))
               for i in range(1, n + 1)]
    monkeypatch.setattr(net_policy, "address_is_forbidden", lambda a: False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: list(answers))
    attempts = []

    def connect(self, addr):
        attempts.append(self.gettimeout())
        time.sleep(self.gettimeout() or 0)
        raise socket.timeout("timed out")
    monkeypatch.setattr(socket.socket, "connect", connect)
    return attempts


def _conn(timeout):
    return types.SimpleNamespace(host="blackhole.example", port=80,
                                 timeout=timeout, source_address=None)


def test_connect_pinned_stops_at_its_deadline(monkeypatch):
    """Every getaddrinfo answer used to get the full per-op timeout, so a
    name with many black-holed records held the dialling thread for
    N x timeout, far past the caller's wall-clock budget."""
    attempts = _black_holed(monkeypatch, 5)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        net_policy.connect_pinned(_conn(1.0), deadline=t0 + 0.5)
    assert time.monotonic() - t0 < 0.9
    assert attempts and max(attempts) <= 0.5


def test_connect_pinned_without_a_deadline_dials_as_before(monkeypatch):
    attempts = _black_holed(monkeypatch, 3)
    with pytest.raises(TimeoutError):
        net_policy.connect_pinned(_conn(0.05))
    assert attempts == [0.05] * 3


def test_connect_pinned_keeps_the_conns_timeout_on_the_dialled_socket(monkeypatch):
    """The deadline only clamps the connect: the returned socket's reads keep
    conn.timeout, exactly as without one."""
    monkeypatch.setattr(net_policy, "address_is_forbidden", lambda a: False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.1", 80))])
    seen = []
    monkeypatch.setattr(socket.socket, "connect",
                        lambda self, addr: seen.append(self.gettimeout()))
    sock = net_policy.connect_pinned(_conn(5.0), deadline=time.monotonic() + 1.0)
    try:
        assert seen and seen[0] <= 1.0
        assert sock.gettimeout() == 5.0
    finally:
        sock.close()
    with pytest.raises(TimeoutError):  # nothing left: no dial at all
        net_policy.connect_pinned(_conn(5.0), deadline=time.monotonic() - 1)
    assert len(seen) == 1


def test_connect_pinned_restores_the_default_timeout_after_a_clamped_dial(monkeypatch):
    monkeypatch.setattr(net_policy, "address_is_forbidden", lambda a: False)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **kw: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("203.0.113.1", 80))])
    monkeypatch.setattr(socket.socket, "connect", lambda self, addr: None)
    conn = _conn(socket._GLOBAL_DEFAULT_TIMEOUT)
    sock = net_policy.connect_pinned(conn, deadline=time.monotonic() + 1.0)
    try:
        assert sock.gettimeout() == socket.getdefaulttimeout()
    finally:
        sock.close()
