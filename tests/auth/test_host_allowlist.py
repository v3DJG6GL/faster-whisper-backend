"""hosts.host_in_allowlist — the client-IP check behind the admin and user
WebUI host gates."""

import types

from faster_whisper_backend.auth import hosts


def _req(host):
    return types.SimpleNamespace(client=types.SimpleNamespace(host=host))


def test_raw_ipv4_mapped_entry_matches_the_folded_client():
    # The env vars (WHISPER_ADMIN_WEBUI_ALLOWED_HOSTS / ..._USER_...) skip the
    # schema's mapped->IPv4 rewrite, so "::ffff:192.168.1.5" copied from the
    # "::" bind's log reached _build_networks as an IPv6 network, which the
    # IPv4-folded client never matched -> 403.
    assert hosts.host_in_allowlist(_req("::ffff:192.168.1.5"), ["::ffff:192.168.1.5"])
    assert hosts.host_in_allowlist(_req("192.168.1.5"), ["::ffff:192.168.1.5"])
    assert hosts.host_in_allowlist(_req("::ffff:10.0.0.77"), ["::ffff:10.0.0.0/120"])
    assert not hosts.host_in_allowlist(_req("10.0.1.77"), ["::ffff:10.0.0.0/120"])
    # A plain IPv6 entry is unchanged.
    assert hosts.host_in_allowlist(_req("fd00::5"), ["fd00::/8"])
    assert not hosts.host_in_allowlist(_req("192.168.1.5"), ["fd00::/8"])
