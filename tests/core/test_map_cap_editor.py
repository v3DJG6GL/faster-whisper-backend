"""The cb:map editor's "n / cap" readout needs window.__mme on EVERY page
that embeds the shared rule editor, or it shows a bare count and the add
button never disables. /quick-config sets it from its API payload
(map_max_entries); RULE_EDITOR_JS bakes a schema-derived default for the
rest (/settings).
"""

from faster_whisper_backend.core import web_common
from tests.conftest import schema_map_cap


def test_settings_ships_map_entry_cap(client):
    cap = schema_map_cap()
    line = ("window.__mme = (typeof window.__mme === 'number') "
            f"? window.__mme : {cap};")
    assert line in web_common.RULE_EDITOR_JS
    assert line in client.get("/settings").text
