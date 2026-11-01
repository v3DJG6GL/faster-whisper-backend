"""The SESSION_COOKIE_SECURE help text puts the silent login failure on the
side that has it.

It read "turn ON when serving over HTTPS ..., else login silently fails",
which says OFF breaks HTTPS. OFF works over both (a non-Secure samesite=lax
cookie is accepted over HTTPS); ON breaks login over plain HTTP, because the
browser drops a Secure Set-Cookie from an insecure origin.
"""

from faster_whisper_backend.settings.descriptions import FIELD_DESCRIPTIONS


def test_help_names_on_over_plain_http_as_the_failing_case():
    text = FIELD_DESCRIPTIONS["SESSION_COOKIE_SECURE"]
    assert "else login silently fails" not in text
    assert "with ON, logging in over plain HTTP silently fails" in text
    assert "OFF works over both" in text
