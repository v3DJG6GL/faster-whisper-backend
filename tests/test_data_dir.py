"""Resolution tests for the data-layout path defaults.

config.py resolves WHISPER_DATA_DIR / WHISPER_DB_DIR / WHISPER_MODELS_DIR at
import time, so each case runs `import config` in a fresh subprocess with a
controlled env instead of monkeypatching the already-imported module.

Layout under test (see config.py _DATA_DIR/_DB_DIR/_MODELS_DIR):
  WHISPER_<X>_DB  >  WHISPER_DB_DIR  >  WHISPER_DATA_DIR/db  >  platform default
and for non-DB paths the same minus the DB_DIR level. Platform defaults are
container-first on Linux (/data, /data/db, /models) and in-checkout on Windows
(repo dir, SQLite stores flat in the repo root, <repo>/models).
"""

import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WIN = os.name == "nt"

_PROBE = (
    "import json, config, config_store; "
    "print(json.dumps({"
    "'api': config.API_KEYS_DB, "
    "'cs': config.CLIENT_SETTINGS_DB, "
    "'log': config.LOG_FILE, "
    "'captures_dir': config.CAPTURES_DIR, "
    "'dl': config.DOWNLOAD_ROOT, "
    "'conv': config.CONVERTED_MODELS_DIR, "
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


def _j(*parts):
    return os.path.join(*parts)


def test_defaults_land_on_the_platform_layout():
    got = _resolve({})
    if _WIN:
        # Bare metal by definition (the images are Linux): the historical
        # in-checkout layout, SQLite stores FLAT in the repo root (no db/).
        assert got["api"] == _j(_REPO, "api_keys.local.sqlite3")
        assert got["cs"] == _j(_REPO, "client_settings.local.sqlite3")
        assert got["log"] == _j(_REPO, "logs", "whisper.log")
        assert got["captures_dir"] == _j(_REPO, "captures")
        assert got["overrides"] == _j(_REPO, "config.local.json")
        assert got["dl"] == _j(_REPO, "models")
        assert got["conv"] == _j(_REPO, "models", "converted")
    else:
        assert got["api"] == _n("/data/db/api_keys.local.sqlite3")
        assert got["cs"] == _n("/data/db/client_settings.local.sqlite3")
        assert got["log"] == _n("/data/logs/whisper.log")
        assert got["captures_dir"] == _n("/data/captures")
        assert got["overrides"] == _n("/data/config.local.json")
        assert got["dl"] == _n("/models")
        assert got["conv"] == _n("/models/converted")


def test_data_dir_moves_everything():
    got = _resolve({"WHISPER_DATA_DIR": "/srv/whisper"})
    root = _n("/srv/whisper")
    # Windows keeps the stores flat at the data root; Linux nests them in db/.
    db_root = root if _WIN else _j(root, "db")
    assert got["api"] == _j(db_root, "api_keys.local.sqlite3")
    assert got["cs"] == _j(db_root, "client_settings.local.sqlite3")
    assert got["log"] == _j(root, "logs", "whisper.log")
    assert got["captures_dir"] == _j(root, "captures")
    assert got["overrides"] == _j(root, "config.local.json")
    # Models follow the data root only on Windows; on Linux they are their own
    # mount (/models) and move via WHISPER_MODELS_DIR alone.
    assert got["dl"] == (_j(root, "models") if _WIN else _n("/models"))


def test_db_dir_moves_only_the_stores():
    got = _resolve({"WHISPER_DATA_DIR": "/srv/whisper", "WHISPER_DB_DIR": "/ssd/db"})
    assert got["api"] == _n("/ssd/db/api_keys.local.sqlite3")
    assert got["cs"] == _n("/ssd/db/client_settings.local.sqlite3")
    # Non-DB paths stay under the data dir.
    assert got["log"] == _n("/srv/whisper/logs/whisper.log")
    assert got["overrides"] == _n("/srv/whisper/config.local.json")


def test_models_dir_moves_only_the_models():
    got = _resolve({"WHISPER_MODELS_DIR": "/nvme/models"})
    assert got["dl"] == _n("/nvme/models")
    assert got["conv"] == _n("/nvme/models/converted")
    # The stores stay on the platform default.
    assert got["api"] == (
        _j(_REPO, "api_keys.local.sqlite3") if _WIN
        else _n("/data/db/api_keys.local.sqlite3")
    )


def test_individual_env_beats_the_dir_knobs():
    got = _resolve({
        "WHISPER_DATA_DIR": "/srv/whisper",
        "WHISPER_DB_DIR": "/ssd/db",
        "WHISPER_MODELS_DIR": "/nvme/models",
        "WHISPER_API_KEYS_DB": "/elsewhere/keys.sqlite3",
        "WHISPER_DOWNLOAD_ROOT": "/elsewhere/models",
        "WHISPER_CONFIG_LOCAL": "/etc/whisper/config.local.json",
    })
    # Explicit per-path envs pass through VERBATIM (no normpath) — the
    # operator wrote them; the sibling still resolves via the dir knobs.
    assert got["api"] == "/elsewhere/keys.sqlite3"
    assert got["cs"] == _n("/ssd/db/client_settings.local.sqlite3")
    assert got["dl"] == "/elsewhere/models"
    assert got["conv"] == _n("/nvme/models/converted")
    assert got["overrides"] == "/etc/whisper/config.local.json"
