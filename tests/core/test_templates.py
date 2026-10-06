"""Package template files (core/templates.py) stay in lockstep with the code.

Every ``<package>/templates/*`` file must be read by some
``templates.load(__file__, "<name>")`` call in a module of that directory, and
every such call must name an existing file: an orphaned template is dead
weight that silently drifts, a missing one fails the import in production.
Each file must also be LF-only (the loader refuses it at import — a CRLF
checkout would change the served bytes and every render_page cache key).
"""

import ast
import pathlib

import pytest

from faster_whisper_backend.core import templates
from faster_whisper_backend.paths import REPO_ROOT

_PKG = pathlib.Path(REPO_ROOT) / "faster_whisper_backend"


def _load_calls() -> set[pathlib.Path]:
    """Resolve every templates.load(...) call in the package to its file."""
    found: set[pathlib.Path] = set()
    for py in sorted(_PKG.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "load"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "templates"):
                continue
            # Literal arguments only, so this scan sees every template.
            args = node.args
            assert (len(args) == 2 and isinstance(args[0], ast.Name)
                    and args[0].id == "__file__"
                    and isinstance(args[1], ast.Constant)
                    and isinstance(args[1].value, str)), (
                f"{py}:{node.lineno}: call templates.load(__file__, \"<name>\") "
                "with a literal name")
            found.add(py.parent / "templates" / args[1].value)
    return found


def _template_files() -> set[pathlib.Path]:
    return {p for d in _PKG.rglob("templates") if d.is_dir()
            for p in d.iterdir() if p.is_file()}


def test_every_template_file_is_loaded_and_every_load_exists():
    loaded = _load_calls()
    files = _template_files()
    assert loaded, "no templates.load() calls found — scan is broken"
    assert not sorted(files - loaded), f"template files nobody loads: {sorted(files - loaded)}"
    assert not sorted(loaded - files), f"templates.load() of missing files: {sorted(loaded - files)}"


def test_template_files_are_lf_only_utf8_and_packaged():
    for p in sorted(_template_files()):
        raw = p.read_bytes()
        assert b"\r" not in raw, f"{p} has CR line endings"
        raw.decode("utf-8")  # strict: a stray non-UTF-8 byte fails here
        # Names .dockerignore would drop from the image (it excludes
        # **/*.local.*, **/*.log and **/*.log.* anywhere; *-preview.html is
        # root-only there, but kept out of template names all the same).
        assert ".local." not in p.name and ".log." not in p.name \
            and not p.name.endswith(("-preview.html", ".log")), (
                f"{p.name} matches a .dockerignore pattern")


def test_load_reads_next_to_module_and_rejects_cr(tmp_path):
    (tmp_path / "templates").mkdir()
    mod = tmp_path / "mod.py"
    (tmp_path / "templates" / "ok.html").write_bytes("a\nb \n".encode("utf-8"))
    assert templates.load(str(mod), "ok.html") == "a\nb \n"
    (tmp_path / "templates" / "crlf.html").write_bytes(b"a\r\nb\r\n")
    with pytest.raises(ValueError, match="CR line endings"):
        templates.load(str(mod), "crlf.html")
