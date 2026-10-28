"""reconcile_on_startup only removes orphans inside the capture fanout layout:
a CAPTURES_DIR mis-set to the data dir (or a parent of it) must not delete the
config, the SQLite stores or the logs that live there."""

import os


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x")
    return path


def test_reconcile_leaves_files_outside_the_fanout_alone(captures_store_db):
    cs = captures_store_db
    audio = cs._require_audio_dir()
    foreign = [
        _touch(os.path.join(audio, "config.local.json")),
        _touch(os.path.join(audio, "usage.local.sqlite3")),
        _touch(os.path.join(audio, "notes.tmp")),
        _touch(os.path.join(audio, "logs", "whisper.log")),
        _touch(os.path.join(audio, "data", "logs", "whisper.log")),
        _touch(os.path.join(audio, "ab", "cd", "deeper", "keep.wav")),
    ]
    orphan = _touch(os.path.join(audio, "ab", "cd", "abcd0123.wav"))
    partial = _touch(os.path.join(audio, "ab", "cd", "abcd4567.wav.tmp"))

    _, unlinked = cs.reconcile_on_startup()

    assert [p for p in foreign if not os.path.isfile(p)] == []
    assert not os.path.isfile(orphan)
    assert not os.path.isfile(partial)
    assert unlinked == 2
