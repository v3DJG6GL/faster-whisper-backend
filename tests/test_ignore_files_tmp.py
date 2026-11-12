"""In-flight and crash-left `*.tmp` writes stay out of the image and the repo.

Capture audio is written as `<id>.wav.tmp` (captures/store.py) or
`<dst>.<rand>.tmp` (merge, VAD trim) and core/atomic_json.py stages
config.local.json, which holds HF_TOKEN, as a `*.tmp` beside it. Under an
in-checkout DATA_DIR not named data/ (WHISPER_DATA_DIR=./state) a glob ending
in `.wav` matches none of these names. Sibling of the ignore-file tests in
test_ci_workflow_config.py.
"""

from tests.conftest import read_repo_text as _read


def _lines(name):
    return [ln.strip() for ln in _read(name).splitlines()]


def test_dockerignore_keeps_tmp_writes_out_of_the_image():
    docker = _lines(".dockerignore")
    assert "**/*.tmp" in docker, ".dockerignore: **/*.tmp missing"
    # Docker's filepath.Match: a bare * never crosses a /.
    assert "*.tmp" not in docker, ".dockerignore: *.tmp is root-only"


def test_gitignore_keeps_tmp_writes_out_of_git():
    git = _lines(".gitignore")
    assert "*.tmp" in git, ".gitignore: *.tmp not ignored"
