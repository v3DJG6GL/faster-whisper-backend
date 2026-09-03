"""The cb:map editor's "n / cap" readout needs window.__mme on EVERY page
that embeds the shared rule editor. /quick-config sets it from its API
payload (map_max_entries); /settings previously got nothing and showed a
bare count with a never-disabled add button, so RULE_EDITOR_JS now bakes a
schema-derived default into the editor itself.
"""

from faster_whisper_backend import config_store
from faster_whisper_backend.core import web_common


def test_settings_ships_map_entry_cap(client):
    cap = next(m.max_length
               for m in config_store.MapRule.model_fields["map"].metadata
               if getattr(m, "max_length", None) is not None)
    line = ("window.__mme = (typeof window.__mme === 'number') "
            f"? window.__mme : {cap};")
    assert line in web_common.RULE_EDITOR_JS
    assert line in client.get("/settings").text
