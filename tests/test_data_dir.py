"""Resolution tests for the container-first path layout.

config.py resolves WHISPER_DATA_DIR / WHISPER_DB_DIR at import time, so each
case runs `import config` in a fresh subprocess with a controlled env instead
of monkeypatching the already-imported module.

Layout under test (see config.py _DATA_DIR/_DB_DIR):
  WHISPER_<X>_DB  >  WHISPER_DB_DIR  >  WHISPER_DATA_DIR/db  >  /data/db
and for non-DB paths the same minus the DB_DIR level.
"""

import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PROBE = (
    "import json, config, config_store; "
    "print(json.dumps({"
    "'api': config.API_KEYS_DB, "
    "'cs': config.CLIENT_SETTINGS_DB, "
    "'log': config.LOG_FILE, "
    "'captures_dir': config.CAPTURES_DIR, "
    "'overrides': config_store.OVERRIDES_PATH}))"
)


def _resolve(extra_env):
    """Import config in a clean subprocess (no inherited WHISPER_*) and return
    the resolved paths."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("WHISPER_")}
    env.update(extra_env)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=_REPO, env=env, capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


# The resolver normpaths, so on Windows the same logic yields \-separated
# paths — compare against the platform's own normalization, not raw POSIX.
_n = os.path.normpath


def test_defaults_are_container_first():
    got = _resolve({})
    assert got["api"] == _n("/data/db/api_keys.local.sqlite3")
    assert got["cs"] == _n("/data/db/client_settings.local.sqlite3")
    assert got["log"] == _n("/data/logs/whisper.log")
    assert got["captures_dir"] == _n("/data/captures")
    assert got["overrides"] == _n("/data/config.local.json")


def test_data_dir_moves_everything():
    got = _resolve({"WHISPER_DATA_DIR": "/srv/whisper"})
    assert got["api"] == _n("/srv/whisper/db/api_keys.local.sqlite3")
    assert got["cs"] == _n("/srv/whisper/db/client_settings.local.sqlite3")
    assert got["log"] == _n("/srv/whisper/logs/whisper.log")
    assert got["captures_dir"] == _n("/srv/whisper/captures")
    assert got["overrides"] == _n("/srv/whisper/config.local.json")


def test_db_dir_moves_only_the_stores():
    got = _resolve({"WHISPER_DATA_DIR": "/srv/whisper", "WHISPER_DB_DIR": "/ssd/db"})
    assert got["api"] == _n("/ssd/db/api_keys.local.sqlite3")
    assert got["cs"] == _n("/ssd/db/client_settings.local.sqlite3")
    # Non-DB paths stay under the data dir.
    assert got["log"] == _n("/srv/whisper/logs/whisper.log")
    assert got["overrides"] == _n("/srv/whisper/config.local.json")


def test_individual_env_beats_the_dir_knobs():
    got = _resolve({
        "WHISPER_DATA_DIR": "/srv/whisper",
        "WHISPER_DB_DIR": "/ssd/db",
        "WHISPER_API_KEYS_DB": "/elsewhere/keys.sqlite3",
        "WHISPER_CONFIG_LOCAL": "/etc/whisper/config.local.json",
    })
    # Explicit per-path envs pass through VERBATIM (no normpath) — the
    # operator wrote them; the sibling still resolves via the dir knobs.
    assert got["api"] == "/elsewhere/keys.sqlite3"
    assert got["cs"] == _n("/ssd/db/client_settings.local.sqlite3")
    assert got["overrides"] == "/etc/whisper/config.local.json"
