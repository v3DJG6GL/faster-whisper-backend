"""main._secure_log_dir: only a log directory the service created itself is
tightened to owner-only."""

from faster_whisper_backend.core import store_common


def test_secure_log_dir_only_tightens_a_directory_we_created(monkeypatch):
    # LOG_FILE is operator-chosen: chmod-ing a pre-existing /var/log to 0700
    # would lock every other daemon out of it, so only a freshly created
    # directory is ours to secure.
    from faster_whisper_backend import main
    seen = []
    monkeypatch.setattr(store_common, "secure_dir", seen.append)
    main._secure_log_dir("/some/dir", created=False)
    assert seen == []
    main._secure_log_dir("/some/dir", created=True)
    assert seen == ["/some/dir"]
