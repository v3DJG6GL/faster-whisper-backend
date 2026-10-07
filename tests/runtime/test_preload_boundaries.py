"""runtime/preload talks to the four model caches through their public
residency API only (is_resident, busy / load_in_progress, idle_peer,
placement, load_unleased, evict, ...). A private reach-in would hand-copy a
cache invariant into preload again, where it silently goes stale the next
time the owning module changes — so an AST scan pins the boundary."""

import ast
import pathlib

from faster_whisper_backend.runtime import preload
from tests.conftest import resolve_import_from

# preload's own package: a relative import in it resolves against this.
_PACKAGE = "faster_whisper_backend.runtime"

_STAGE_MODULES = {
    "faster_whisper_backend.transcription.models",
    "faster_whisper_backend.translation.engine",
    "faster_whisper_backend.audio.diarization",
    "faster_whisper_backend.audio.bgm_separation",
}


def _tree() -> ast.Module:
    return ast.parse(pathlib.Path(preload.__file__).read_text(encoding="utf-8"))


def _stage_aliases(tree: ast.Module) -> "dict[str, str]":
    """Local name → stage module, for every import of one (eager or lazy)."""
    aliases: "dict[str, str]" = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = resolve_import_from(node, _PACKAGE)
            for a in node.names:
                full = f"{base}.{a.name}"
                if full in _STAGE_MODULES:
                    aliases[a.asname or a.name] = full
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in _STAGE_MODULES and a.asname:
                    aliases[a.asname] = a.name
    return aliases


def test_the_scan_sees_all_four_stage_modules():
    # Guards the scan below against passing vacuously after an import rename.
    assert set(_stage_aliases(_tree()).values()) == _STAGE_MODULES


def _private_alias_hits(tree: ast.Module) -> "list[str]":
    aliases = _stage_aliases(tree)
    return sorted(
        f"{node.lineno}: {node.value.id}.{node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in aliases
        and node.attr.startswith("_"))


def _private_other_hits(tree: ast.Module) -> "list[str]":
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = resolve_import_from(node, _PACKAGE)
            if mod in _STAGE_MODULES:
                hits += [f"{node.lineno}: from {mod} import {a.name}"
                         for a in node.names if a.name.startswith("_")]
        elif (isinstance(node, ast.Attribute) and node.attr.startswith("_")
              and ast.unparse(node.value) in _STAGE_MODULES):
            hits.append(f"{node.lineno}: {ast.unparse(node)}")
    return sorted(hits)


def test_preload_makes_no_private_access_into_the_stage_modules():
    assert _private_alias_hits(_tree()) == []


def test_preload_reaches_no_private_name_by_any_other_spelling():
    """The alias scan above only sees `alias._name`. A `from <stage module>
    import _name` and a bare `import <stage module>` used through its
    dotted path reach the same private state without an alias."""
    assert _private_other_hits(_tree()) == []


def test_the_scans_resolve_relative_imports():
    # A relative cycle-breaker must not slip past the absolute-name scans.
    tree = ast.parse("from ..audio import diarization as d\n"
                     "d._busy\n"
                     "from ..audio.diarization import _load_lock\n")
    assert _stage_aliases(tree) == {
        "d": "faster_whisper_backend.audio.diarization"}
    assert _private_alias_hits(tree) == ["2: d._busy"]
    assert _private_other_hits(tree) == [
        "3: from faster_whisper_backend.audio.diarization import _load_lock"]
