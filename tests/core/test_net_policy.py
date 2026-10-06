"""net_policy.address_is_forbidden: the literal-address half of the yt-dlp
SSRF policy (the end-to-end guard lives in tests/media/test_ssrf_guard.py)."""

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
