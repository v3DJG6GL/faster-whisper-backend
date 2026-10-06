"""State reset for the modules carved out of main.py.

`importlib.reload(main)` in the app_module fixture used to re-create every
engine container, lock and cache per test. Once that state lives in its own
module the reload no longer reaches it, so each such module owns a
`_reset_for_tests()` registered in conftest._RESET_HOOKS. These tests pin that
every module-level container / lock / semaphore / lru_cache in those modules is
covered by its hook (true constants are allowlisted with the reason), and that
main.py no longer defines any name that moved out of it (a stale twin in main
would silently stop receiving the tests' patches).
"""

import ast
import asyncio
import collections
import functools
import importlib
import threading

import pytest

from faster_whisper_backend.core.loop_lock import LoopLock
from tests.conftest import _RESET_HOOKS

_CONTAINERS = (dict, set, list, collections.deque)   # OrderedDict is a dict
_LOCKS = (asyncio.Lock, asyncio.Semaphore, LoopLock,
          type(threading.Lock()), type(threading.RLock()))

# module -> {name: why the hook leaves it alone}. Everything else stateful must
# be reset by the module's hook.
_NOT_RESET = {
    "faster_whisper_backend.pipeline.engine": {
        "_COMPILED_RULES": "derived from cfg.PIPELINE_RULES — rebuilt by "
                           "rebuild_caches(), which app_module calls",
    },
    "faster_whisper_backend.transcription.receipt": {
        "_KWARG_TO_CFG": "constant kwarg → config-field map",
    },
    "faster_whisper_backend.transcription.guards": {},
    "faster_whisper_backend.transcription.models": {
        "_CT2_QUANTIZATIONS": "constant",
        "_DECODE_INT_BOUNDS": "constant clamp table",
        "_DECODE_FLOAT_BOUNDS": "constant clamp table",
        "_DECODE_STR_CAPS": "constant clamp table",
    },
    "faster_whisper_backend.transcription.progress": {},
    "faster_whisper_backend.translation.gating": {},
    "faster_whisper_backend.captures.samples": {
        "_JOIN_STR": "constant join-strategy → separator map",
    },
    "faster_whisper_backend.runtime.model_registry": {},
    # Stateless since P16: the registry moved to model_registry. Listed so a
    # container added back here is caught without a hook.
    "faster_whisper_backend.runtime.system_stats": {
        "_handles": "NVML handle per device index — process-constant once "
                    "bound (cleared by shutdown())",
    },
    # P14: the route groups and helpers carved out of main.py.
    "faster_whisper_backend.media.video": {},
    "faster_whisper_backend.media.routes": {
        "_VIDEO_MIME": "constant container → mime map",
    },
    "faster_whisper_backend.transcription.jobs_routes": {},
    "faster_whisper_backend.transcription.catalog_routes": {},
    "faster_whisper_backend.translation.routes": {},
    "faster_whisper_backend.admin.logs_routes": {},
    "faster_whisper_backend.auth.routes": {},
    "faster_whisper_backend.auth.hosts": {
        "_cors_origins": "configuration main installs at app build "
                         "(configure_origins), re-armed by its reload",
        "_trusted_origins": "configuration main installs at app build "
                            "(configure_origins), re-armed by its reload",
    },
    "faster_whisper_backend.pipeline.apply": {
        "EVICTORS": "constant bucket → drop-callable table",
        "_RULES_LOCK": "a LoopLock keeps one asyncio.Lock per running loop and "
                       "prunes closed loops, so a dead TestClient loop's lock "
                       "is never waited on — nothing to rebind",
    },
}

# Lazily-built singletons: None until the getter first runs, None again after
# the hook (so the next test builds one on ITS event loop).
_LAZY = {
    "faster_whisper_backend.transcription.models": {
        "_inference_semaphore": "get_inference_semaphore",
        "_url_download_semaphore": "_get_url_download_semaphore",
    },
}


def _top_level_names(mod) -> list[str]:
    """Names the module binds itself (assignments and defs), not imports."""
    with open(mod.__file__, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            out += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.append(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append(node.name)
    return out


def _stateful(mod) -> list[str]:
    return [n for n in _top_level_names(mod)
            if isinstance(getattr(mod, n), _CONTAINERS + _LOCKS)
            or isinstance(getattr(mod, n), functools._lru_cache_wrapper)]


def _hook(modname):
    hooks = {name: hook for name, hook in _RESET_HOOKS}
    assert modname in hooks, f"{modname} has state but no _RESET_HOOKS entry"
    mod = importlib.import_module(modname)
    hook = hooks[modname]
    return (lambda: getattr(mod, hook)()) if isinstance(hook, str) else (lambda: hook(mod))


class _CacheSpy:
    cleared = False

    def cache_clear(self):
        self.cleared = True


@pytest.mark.parametrize("modname", sorted(_NOT_RESET))
def test_allowlist_names_still_exist(modname):
    """A stale allowlist entry would quietly excuse a future container."""
    mod = importlib.import_module(modname)
    for name in _NOT_RESET[modname]:
        assert name in _stateful(mod), (modname, name)


@pytest.mark.parametrize("modname", sorted(_NOT_RESET))
def test_every_container_lock_and_cache_is_reset(modname, monkeypatch):
    mod = importlib.import_module(modname)
    names = [n for n in _stateful(mod) if n not in _NOT_RESET[modname]]
    if not names and not _LAZY.get(modname):
        return   # stateless: no hook needed
    run_hook = _hook(modname)

    before, spies = {}, {}
    for name in names:
        value = getattr(mod, name)
        before[name] = value
        if isinstance(value, functools._lru_cache_wrapper):
            spies[name] = _CacheSpy()
            monkeypatch.setattr(mod, name, spies[name])
        elif isinstance(value, dict):
            value[("__probe__",)] = object()
        elif isinstance(value, set):
            value.add(("__probe__",))
        elif isinstance(value, (list, collections.deque)):
            value.append(("__probe__",))
    # get_inference_semaphore publishes metrics.gpu_gate (metrics' own hook
    # clears it); monkeypatch puts it back after this test.
    from faster_whisper_backend.stats import metrics
    monkeypatch.setattr(metrics, "gpu_gate", metrics.gpu_gate)
    for name, getter in _LAZY.get(modname, {}).items():
        getattr(mod, getter)()
        assert getattr(mod, name) is not None, name

    run_hook()

    for name in names:
        value = getattr(mod, name)
        if name in spies:
            assert spies[name].cleared, f"{modname}.{name}: cache not cleared"
        elif isinstance(before[name], _LOCKS):
            assert value is not before[name], f"{modname}.{name}: lock not rebound"
        else:
            # Cleared IN PLACE: a caller holding a reference sees it empty.
            assert value is before[name], f"{modname}.{name}: rebound, not cleared"
            assert not value, f"{modname}.{name}: not cleared"
    for name in _LAZY.get(modname, {}):
        assert getattr(mod, name) is None, f"{modname}.{name}: not reset to None"


# Names that moved out of main.py in the P7–P11 refactor phases. The modules
# that now own them are scanned, so a name added there later is covered too.
_P7_MOVED = ("cfg_for", "build_ident", "_NO_DEFAULT", "_resolve_request_knob")
_OWNERS = (
    "faster_whisper_backend.pipeline.engine",
    "faster_whisper_backend.transcription.receipt",
    "faster_whisper_backend.transcription.guards",
    "faster_whisper_backend.transcription.models",
    "faster_whisper_backend.transcription.progress",
    "faster_whisper_backend.translation.gating",
)
# Bound in both places on purpose: main keeps its own logger and _log_safe
# alias (store_common.log_safe) for the code that stayed.
_SHARED = {"logger", "_log_safe", "_reset_for_tests"}


def test_main_no_longer_defines_moved_names():
    from faster_whisper_backend import main
    from faster_whisper_backend.settings import effective_config
    moved = set(_P7_MOVED)
    for owner in _OWNERS:
        moved |= set(_top_level_names(importlib.import_module(owner))) - _SHARED
    for name in _P7_MOVED:
        assert hasattr(effective_config, name), name
    stale = sorted(n for n in moved if hasattr(main, n))
    assert not stale, f"main.py still defines moved names: {stale}"
    # The anchors this test is about, spelled out so a scan bug can't hide them.
    for name in ("_postprocess_text", "rebuild_caches", "_get_or_load_model",
                 "_loaded_models", "_model_leases", "_format_request_block",
                 "PlainText", "assemble_transcribe_kwargs", "tail_guard_limits",
                 "get_inference_semaphore", "drain_then_evict",
                 "_BATCH_PROGRESS", "_progress_set", "_progress_close",
                 "_PROGRESS_ID_RE", "_BATCH_CANCELLED", "_check_cancelled",
                 "_jobs_finish", "_translation_model_allowed",
                 "_translation_default_model"):
        assert name in moved, name


def test_translation_engine_left_audio():
    """The llama.cpp engine moved from audio/translation.py to
    translation/engine.py; no stale twin may stay importable (a patch on it
    would silently miss every caller)."""
    import importlib.util
    assert importlib.util.find_spec("faster_whisper_backend.audio.translation") is None
    assert importlib.util.find_spec("faster_whisper_backend.translation.engine") is not None
    hooks = dict(_RESET_HOOKS)
    assert "faster_whisper_backend.audio.translation" not in hooks
    assert hooks["faster_whisper_backend.translation.engine"] == "_reset_for_tests"


# P12: the config hot-apply helpers and the shared PIPELINE_RULES lock moved
# from admin/routes.py and quick_config/routes.py into pipeline/apply.py. A
# stale twin left in a router would be what that router calls, while the tests
# patch pipeline.apply — and a second lock would silently split the writers.
_P12_MOVED = {
    "faster_whisper_backend.admin.routes": (
        "_apply_hot_changes", "_canon_rules", "_resolved_value", "_EVICTORS",
        "_pipeline_rules_lock", "_rebuild_caches", "_PIPELINE_RULE_ADAPTER",
        "_sort_dicts",
    ),
    "faster_whisper_backend.quick_config.routes": (
        "_PATCH_LOCK", "_patch_lock", "_apply_hot_changes", "_canon_rules",
    ),
}


@pytest.mark.parametrize("modname", sorted(_P12_MOVED))
def test_routers_no_longer_define_apply_names(modname):
    mod = importlib.import_module(modname)
    stale = sorted(n for n in _P12_MOVED[modname] if hasattr(mod, n))
    assert not stale, f"{modname} still defines moved names: {stale}"
    from faster_whisper_backend.pipeline import apply as pl_apply
    for name in ("apply_hot_changes", "canon_rules", "resolved_value",
                 "EVICTORS", "rules_lock", "rebuild_caches_off_loop",
                 "_PIPELINE_RULE_ADAPTER", "_sort_dicts", "_RULES_LOCK"):
        assert hasattr(pl_apply, name), name


# P15: the sample-building helpers and the per-sample rebuild lock moved from
# captures/routes.py into captures/samples.py. The routes and both workers call
# through capture_samples.X, so a stale twin left in routes would be what a
# test patches while every caller runs the samples copy.
def test_captures_routes_no_longer_defines_sample_helpers():
    from faster_whisper_backend.captures import routes as captures_routes
    from faster_whisper_backend.captures import samples as capture_samples
    moved = set(_top_level_names(capture_samples)) - _SHARED
    stale = sorted(n for n in moved if hasattr(captures_routes, n))
    assert not stale, f"captures/routes.py still defines moved names: {stale}"
    for name in ("_build_merged_wav", "_rebuild_lock", "_rebuild_locks",
                 "_align_words_to_final", "_build_default_transcript"):
        assert name in moved, name


# P16: the loaded-model registry and the warm-lease predicate moved from
# runtime/system_stats.py to runtime/model_registry.py. system_stats must not
# keep (or re-export) them: a caller of the twin would read a second, empty
# registry while the tests patch model_registry.
_P16_MOVED = (
    "_loaded_models", "_loaded_models_lock", "register_loaded_model",
    "load_secs_since", "touch_loaded_model", "unregister_loaded_model",
    "_warm_predicate", "set_warm_predicate", "is_warm",
    "MODEL_ROLE_PREFIXES", "model_role", "loaded_models_snapshot",
    "_reset_for_tests",
)


def test_system_stats_no_longer_holds_the_model_registry():
    from faster_whisper_backend.runtime import model_registry, system_stats
    stale = sorted(n for n in _P16_MOVED if hasattr(system_stats, n))
    assert not stale, f"system_stats still defines moved names: {stale}"
    for name in _P16_MOVED:
        assert hasattr(model_registry, name), name
    hooks = dict(_RESET_HOOKS)
    assert "faster_whisper_backend.runtime.system_stats" not in hooks
    assert hooks["faster_whisper_backend.runtime.model_registry"] == "_reset_for_tests"


def test_model_registry_hook_clears_the_warm_predicate():
    """Not a container, so the generic scan above cannot see it: a predicate
    left installed would pin a later test's model against the evictors."""
    from faster_whisper_backend.runtime import model_registry
    model_registry.set_warm_predicate(lambda _name: True)
    assert model_registry.is_warm("x")
    _hook("faster_whisper_backend.runtime.model_registry")()
    assert model_registry._warm_predicate is None
    assert not model_registry.is_warm("x")


# P14: the remaining route groups left main.py — media/{video,routes},
# transcription/{jobs_routes,catalog_routes}, translation/routes,
# admin/logs_routes, auth/{routes,hosts} — plus the bootstrap-admin ingest
# (auth/api_keys_store.py) and the hard-restart TMPDIR sweep
# (admin/restart_service.py). A stale twin in main would be what a test
# patches while every caller runs the moved copy.
_P14_OWNERS = (
    "faster_whisper_backend.media.video",
    "faster_whisper_backend.media.routes",
    "faster_whisper_backend.transcription.jobs_routes",
    "faster_whisper_backend.transcription.catalog_routes",
    "faster_whisper_backend.translation.routes",
    "faster_whisper_backend.admin.logs_routes",
    "faster_whisper_backend.auth.routes",
    "faster_whisper_backend.auth.hosts",
)
# main still computes the two origin lists itself (the CORS middleware needs
# them) and hands them to auth.hosts.configure_origins.
_P14_SHARED = _SHARED | {"router", "_cors_origins", "_trusted_origins"}


def test_main_no_longer_defines_p14_names():
    from faster_whisper_backend import main
    moved = {"bootstrap_admin_from_env", "_bootstrap_admin_from_env",
             "BootstrapAdminError", "_BOOTSTRAP_KEY_MIN_LEN",
             "_BOOTSTRAP_KEY_MIN_DISTINCT", "_bootstrap_key_is_strong",
             "reclaim_hard_restart_orphans", "_reclaim_hard_restart_orphans",
             "_clamp_context_segments"}
    for owner in _P14_OWNERS:
        moved |= set(_top_level_names(importlib.import_module(owner))) - _P14_SHARED
    stale = sorted(n for n in moved if hasattr(main, n))
    assert not stale, f"main.py still defines moved names: {stale}"
    for name in ("_download_video_for_run", "_VIDEO_TASKS", "url_preview",
                 "package_media", "jobs_list", "_jobs_gate", "translate_text",
                 "_translate_inflight", "list_models", "get_decode_defaults",
                 "_LOG_VIEWER_HTML", "_stream_log_lines", "severity_snapshot",
                 "login", "_login_failures", "_origin_is_allowed",
                 "require_user_webui_host", "_to_ip"):
        assert name in moved, name
    from faster_whisper_backend.admin import restart_service
    from faster_whisper_backend.auth import api_keys_store
    from faster_whisper_backend.transcription import models as tx_models
    assert callable(api_keys_store.bootstrap_admin_from_env)
    assert issubclass(api_keys_store.BootstrapAdminError, RuntimeError)
    assert callable(restart_service.reclaim_hard_restart_orphans)
    assert callable(tx_models._clamp_context_segments)


def test_web_common_host_gate_is_auth_hosts():
    """web_common re-exports the host gate; each alias must BE auth.hosts'
    object (a copy would split a patch), and web_common keeps no twin of the
    private helpers."""
    from faster_whisper_backend.auth import hosts as auth_hosts
    from faster_whisper_backend.core import web_common
    for name in ("require_allowed_host", "host_in_allowlist",
                 "require_user_webui_host", "require_admin_webui_host"):
        assert getattr(web_common, name) is getattr(auth_hosts, name), name
    for name in ("_to_ip", "_build_networks"):
        assert not hasattr(web_common, name), name


def test_no_package_module_imports_main():
    """Nothing inside the package reaches back into main (the old lazy
    `from faster_whisper_backend import main` cycle breakers are gone); only
    the `python -m faster_whisper_backend` entry point imports it."""
    import pathlib
    from faster_whisper_backend.paths import REPO_ROOT
    pkg = pathlib.Path(REPO_ROOT) / "faster_whisper_backend"
    offenders = []
    for py in sorted(pkg.rglob("*.py")):
        if py.name == "__main__.py":
            continue
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.ImportFrom) and node.module == "faster_whisper_backend"
                    and any(a.name == "main" for a in node.names)):
                offenders.append(f"{py.relative_to(pkg)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and node.module == "faster_whisper_backend.main":
                offenders.append(f"{py.relative_to(pkg)}:{node.lineno}")
            elif isinstance(node, ast.Import) and any(
                    a.name == "faster_whisper_backend.main" for a in node.names):
                offenders.append(f"{py.relative_to(pkg)}:{node.lineno}")
    assert not offenders, offenders


# P18: the decode/job-pipeline modules left core/ (shared infra only) for
# transcription/. No stale twin may stay importable under core — a patch on it
# would silently miss every caller — and the receipt-hold hook follows it.
_P18_MOVED = ("decode_trace", "segment_guards", "run_plan", "receipt_hold",
              "jobs_store")


@pytest.mark.parametrize("name", _P18_MOVED)
def test_transcription_modules_left_core(name):
    import importlib.util
    assert importlib.util.find_spec(f"faster_whisper_backend.core.{name}") is None
    assert importlib.util.find_spec(f"faster_whisper_backend.transcription.{name}") is not None
    hooks = dict(_RESET_HOOKS)
    assert f"faster_whisper_backend.core.{name}" not in hooks
    if name == "receipt_hold":
        assert hooks[f"faster_whisper_backend.transcription.{name}"] == "_reset_for_tests"
