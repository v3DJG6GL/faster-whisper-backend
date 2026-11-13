"""The ADMIN_WEBUI_ALLOWED_HOSTS help text says what the list does in open
mode.

It read "an admin API key is also required", without condition. Before the
first admin key exists there is no key to require: every host on this list is
handed the synthetic admin (auth.dependencies.open_mode_host_ok), so widening
the list first and creating the key later opens the admin pages to that
network.
"""

from faster_whisper_backend.settings.descriptions import FIELD_DESCRIPTIONS


def test_help_says_a_listed_host_is_admin_without_a_key_in_open_mode():
    text = FIELD_DESCRIPTIONS["ADMIN_WEBUI_ALLOWED_HOSTS"]
    assert "an admin API key is also required" not in text
    assert "Once an admin API key exists, that key is required as well" in text
    assert "open mode" in text
    assert "admin without a key" in text
    assert "create the admin key before you widen the list" in text
    # What the old text said and still holds.
    assert "/settings, /settings/api-keys, /docs" in text
    assert "Loopback (127.0.0.1, ::1) is always implicitly allowed" in text
    assert "default is loopback only" in text
