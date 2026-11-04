"""A legacy captures audio tree (<repo>/captures, or <data_dir>/captures when
CAPTURES_DIR is pinned elsewhere) keeps warning while it holds a file.

The first start creates the configured CAPTURES_DIR, so the old "configured
path missing" test went quiet from the second start on while the old tree's
audio sat unused (its captures all audio_missing).
"""

import os

import pytest

from tests.settings.test_config_env import _legacy_warnings


def _layout(tmp_path):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    (data / "captures" / "groups").mkdir(parents=True)  # what a start creates
    return repo, data, {"captures": str(data / "captures")}


def test_legacy_audio_tree_warns_after_the_configured_dir_exists(tmp_path):
    repo, data, mapping = _layout(tmp_path)
    (repo / "captures" / "ab12").mkdir(parents=True)
    (repo / "captures" / "ab12" / "ab12cd.wav").write_bytes(b"RIFF")

    warns = _legacy_warnings(repo, data, mapping)
    assert len(warns) == 1
    legacy = str(repo / "captures")
    assert legacy in warns[0] and str(data / "captures") in warns[0]
    assert "NOT used" in warns[0]
    assert f"WHISPER_CAPTURES_DIR to {legacy!r}" in warns[0]


def test_empty_legacy_audio_tree_is_silent(tmp_path):
    repo, data, mapping = _layout(tmp_path)
    (repo / "captures" / "groups").mkdir(parents=True)
    assert _legacy_warnings(repo, data, mapping) == []
    # Even before the first start: nothing there to lose.
    os.rmdir(data / "captures" / "groups")
    os.rmdir(data / "captures")
    assert _legacy_warnings(repo, data, mapping) == []


def test_data_dir_audio_tree_warns_when_captures_dir_is_pinned_elsewhere(tmp_path):
    repo, data, _ = _layout(tmp_path)
    (data / "captures" / "x.wav").write_bytes(b"RIFF")
    pinned = tmp_path / "audio"
    pinned.mkdir()
    warns = _legacy_warnings(repo, data, {"captures": str(pinned)})
    assert len(warns) == 1 and str(data / "captures") in warns[0]


def test_legacy_audio_tree_symlinked_to_the_live_dir_is_silent(tmp_path):
    repo, data, mapping = _layout(tmp_path)
    (data / "captures" / "x.wav").write_bytes(b"RIFF")
    try:
        os.symlink(data / "captures", repo / "captures",
                   target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("no symlink support")
    assert _legacy_warnings(repo, data, mapping) == []
