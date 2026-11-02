"""Tests for captures_store (faster_whisper_backend/captures/store.py)."""

import json
import os
import sqlite3
import wave

import pytest

RATE = 16000


def _fake_transcode(cs, monkeypatch, nbytes=1234):
    """Patch transcode_to_wav_16k_mono to write a minimal valid WAV at dst and
    return a byte count, so create_capture's insert path runs without ffmpeg."""
    from faster_whisper_backend.audio import transcode as audio_transcode

    def _fake(src_path, dst_path):
        with wave.open(dst_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(b"\x00\x00" * 100)
        return nbytes

    monkeypatch.setattr(audio_transcode, "transcode_to_wav_16k_mono", _fake)


def _make(cs, monkeypatch, tmp_path, **over):
    """Create a capture row through the real create_capture (transcode faked)."""
    _fake_transcode(cs, monkeypatch)
    src = tmp_path / "src_in.bin"
    src.write_bytes(b"junk")
    kw = dict(
        audio_src_path=str(src),
        request_id="req-1",
        model="small",
        language="de",
        audio_s=3.0,
        raw="raw text",
        final="final text",
        words=[{"word": "hi", "start": 0.0, "end": 0.5}],
        segments=[{"text": "hi", "start": 0.0, "end": 0.5}],
        user_id="u1",
    )
    kw.update(over)
    return cs.create_capture(**kw)


def test_relpath_fanout_and_ext_lowercase(captures_store_db):
    cs = captures_store_db
    rel = cs._relpath_for("abcdef1234", ".WAV")
    assert rel == os.path.join("ab", "cd", "abcdef1234.wav")


def test_relpath_empty_ext_becomes_bin(captures_store_db):
    rel = captures_store_db._relpath_for("zzxxccvv", "")
    assert rel.endswith(".bin")


def test_relpath_ext_capped(captures_store_db):
    cs = captures_store_db
    rel = cs._relpath_for("aabbccdd", "x" * 50)
    name = os.path.basename(rel)
    assert name == "aabbccdd." + "x" * cs._CAP_AUDIO_FORMAT


def test_abs_audio_path_ok(captures_store_db):
    cs = captures_store_db
    p = cs.abs_audio_path(os.path.join("ab", "cd", "x.wav"))
    assert os.path.abspath(cs._require_audio_dir()) in p


@pytest.mark.parametrize("bad", [
    os.path.join("..", "etc", "passwd"),
    os.path.join("ab", "..", "..", "escape.wav"),
    "/etc/passwd",
])
def test_abs_audio_path_rejects_escape(captures_store_db, bad):
    with pytest.raises(ValueError):
        captures_store_db.abs_audio_path(bad)


def test_truncate_json_under_cap_intact(captures_store_db):
    items = [{"a": 1}, {"b": 2}]
    out = captures_store_db._truncate_json(items, 10_000)
    assert json.loads(out) == items


def test_truncate_json_tail_trim(captures_store_db):
    cs = captures_store_db
    items = [{"w": f"word{i}", "start": i, "end": i + 1} for i in range(50)]
    full = json.dumps(items, ensure_ascii=False)
    cap = len(full) // 2
    out = cs._truncate_json(items, cap)
    kept = json.loads(out)
    assert len(out) <= cap
    assert 0 < len(kept) < len(items)
    assert kept == items[:len(kept)]


def test_truncate_json_non_serializable_returns_empty(captures_store_db):
    out = captures_store_db._truncate_json([{1, 2, 3}], 10_000)
    assert out == "[]"


def test_truncate_translations_key_overhead_over_cap(captures_store_db):
    """Trimming per-language TEXT bottoms out at n=0 still over cap when the
    key set alone exceeds it; languages must then be dropped so the blob
    honours the cap instead of landing an oversized row."""
    cs = captures_store_db
    # 600 keys x ~13 bytes of `"langN": ""` alone exceed a 5 KB cap, so
    # even the n=0 dump is over cap and the language-dropping branch (the
    # binary search on the kept key prefix) must run — same branch as a
    # 6000-language map against 50 KB, with a smaller fixture.
    out = cs._truncate_translations(
        {f"lang{i}": "x" * 10 for i in range(600)}, 5_000)
    assert out is not None and len(out) <= 5_000
    kept = json.loads(out)
    assert isinstance(kept, dict) and 0 < len(kept) < 600
    # Front languages survive (insertion order), tail dropped.
    assert list(kept) == [f"lang{i}" for i in range(len(kept))]
    # Dropping languages comes FIRST, so the survivors keep their text
    # instead of all being trimmed to "".
    assert all(v == "x" * 10 for v in kept.values()), kept
    # The first-inserted (exportable) track keeps its text too.
    big = {"en": "e" * 30_000, **{f"l{i}": "" for i in range(5000)}}
    kept = json.loads(cs._truncate_translations(big, 50_000))
    assert len(kept["en"]) > 0
    # A single language whose key alone cannot fit yields an empty map.
    assert json.loads(cs._truncate_translations({"k" * 50: "v"}, 20)) == {}


def test_safe_unlink_existing(captures_store_db, tmp_path):
    f = tmp_path / "gone.bin"
    f.write_bytes(b"x")
    assert captures_store_db._safe_unlink(str(f)) is True
    assert not f.exists()


def test_safe_unlink_missing_is_true(captures_store_db, tmp_path):
    assert captures_store_db._safe_unlink(str(tmp_path / "never.bin")) is True


def test_count_starts_zero(captures_store_db):
    assert captures_store_db.count() == 0


def test_create_capture_inserts_row_and_writes_audio(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    assert cs.count() == 1
    row = cs.get_capture(cid)
    assert row["status"] == "new"
    assert row["model"] == "small"
    assert row["language"] == "de"
    assert row["user_id"] == "u1"
    assert row["words"] == [{"word": "hi", "start": 0.0, "end": 0.5}]
    assert os.path.isfile(cs.abs_audio_path(row["audio_relpath"]))


def test_create_capture_truncates_text(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path, raw="r" * (cs._CAP_RAW + 100),
                final="f" * (cs._CAP_FINAL + 100))
    row = cs.get_capture(cid)
    assert len(row["raw"]) == cs._CAP_RAW
    assert len(row["final"]) == cs._CAP_FINAL


def test_create_capture_transcode_failure_cleans_tmp(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    from faster_whisper_backend.audio import transcode as audio_transcode

    def _boom(src, dst):
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(audio_transcode, "transcode_to_wav_16k_mono", _boom)
    src = tmp_path / "in.bin"
    src.write_bytes(b"x")
    with pytest.raises(RuntimeError):
        cs.create_capture(
            audio_src_path=str(src), request_id=None, model="m",
            language="de", audio_s=1.0, raw="r", final="f",
            words=[], segments=[],
        )
    assert cs.count() == 0
    leftovers = []
    for root, _d, files in os.walk(cs._require_audio_dir()):
        leftovers += [f for f in files if f.endswith(".tmp")]
    assert leftovers == []


def test_create_capture_insert_failure_unlinks_audio(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    _fake_transcode(cs, monkeypatch)
    # Force the INSERT to fail after the WAV is in place.
    real_conn = cs._require_conn()

    class FailingConn:
        def execute(self, sql, *a, **k):
            if sql.strip().startswith("INSERT"):
                raise RuntimeError("insert boom")
            return real_conn.execute(sql, *a, **k)

    monkeypatch.setattr(cs, "_require_conn", lambda: FailingConn())
    src = tmp_path / "in.bin"
    src.write_bytes(b"x")
    with pytest.raises(RuntimeError):
        cs.create_capture(
            audio_src_path=str(src), request_id=None, model="m",
            language="de", audio_s=1.0, raw="r", final="f",
            words=[], segments=[],
        )
    monkeypatch.setattr(cs, "_require_conn", lambda: real_conn)
    files = []
    for root, _d, fs in os.walk(cs._require_audio_dir()):
        files += [f for f in fs if f.endswith(".wav")]
    assert files == []


def test_list_captures_newest_first_drops_words(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path, request_id="a")
    cid2 = _make(cs, monkeypatch, tmp_path, request_id="b")
    rows = cs.list_captures()
    assert rows[0]["id"] == cid2
    assert "words" not in rows[0] and "segments" not in rows[0]


def test_list_captures_status_filter(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    cs.update_capture(cid, {"status": "ready"})
    _make(cs, monkeypatch, tmp_path)  # stays new
    assert len(cs.list_captures(status="ready")) == 1
    assert len(cs.list_captures(status="new")) == 1
    # status 'all' and None => no filter.
    assert len(cs.list_captures(status="all")) == 2
    assert len(cs.list_captures(status=None)) == 2


def test_list_captures_user_filter(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path, user_id="u1")
    _make(cs, monkeypatch, tmp_path, user_id="u2")
    assert len(cs.list_captures(user_id="u1")) == 1
    assert len(cs.list_captures(user_id=None)) == 2  # None = no filter


def test_list_captures_before_ts_cursor(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid1 = _make(cs, monkeypatch, tmp_path)
    cid2 = _make(cs, monkeypatch, tmp_path)
    conn = cs._require_conn()
    conn.execute("UPDATE captures SET created_ts=? WHERE id=?", (100.0, cid1))
    conn.execute("UPDATE captures SET created_ts=? WHERE id=?", (200.0, cid2))
    page = cs.list_captures(before_ts=150.0)
    assert [r["id"] for r in page] == [cid1]


def test_iter_export_includes_words_oldest_first(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid1 = _make(cs, monkeypatch, tmp_path)
    cid2 = _make(cs, monkeypatch, tmp_path)
    conn = cs._require_conn()
    conn.execute("UPDATE captures SET created_ts=? WHERE id=?", (100.0, cid1))
    conn.execute("UPDATE captures SET created_ts=? WHERE id=?", (200.0, cid2))
    rows = list(cs.iter_captures_for_export())
    assert [r["id"] for r in rows] == [cid1, cid2]
    assert "words" in rows[0]


def test_iter_export_filters(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path, user_id="u1")
    _make(cs, monkeypatch, tmp_path, user_id="u2")
    assert len(list(cs.iter_captures_for_export(user_id="u1"))) == 1
    assert len(list(cs.iter_captures_for_export(status="all"))) == 2


def test_get_capture_missing(captures_store_db):
    assert captures_store_db.get_capture("nope") is None


def test_find_by_request_id(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path, request_id="shared")
    _make(cs, monkeypatch, tmp_path, request_id="shared")
    _make(cs, monkeypatch, tmp_path, request_id="other")
    assert len(cs.find_by_request_id("shared")) == 2
    assert cs.find_by_request_id("missing") == []


def test_counts_by_status(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    c1 = _make(cs, monkeypatch, tmp_path)
    _make(cs, monkeypatch, tmp_path)
    cs.update_capture(c1, {"status": "ready"})
    counts = cs.counts_by_status()
    assert counts["new"] == 1 and counts["ready"] == 1
    assert set(counts) == set(cs._VALID_STATUS)


def test_counts_and_total_scoped_to_user(captures_store_db, monkeypatch,
                                         tmp_path):
    """count(user_id=…) / counts_by_status(user_id=…) narrow to one owner —
    a scope=own caller's toolbar numbers must not include (or leak) other
    users' rows. None keeps the unfiltered admin/global behaviour."""
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path, user_id="alice")
    c2 = _make(cs, monkeypatch, tmp_path, user_id="alice")
    _make(cs, monkeypatch, tmp_path, user_id="bob")
    cs.update_capture(c2, {"status": "ready"})

    assert cs.count() == 3
    assert cs.count(user_id="alice") == 2
    assert cs.count(user_id="bob") == 1
    assert cs.count(user_id="nobody") == 0

    all_counts = cs.counts_by_status()
    assert all_counts["new"] == 2 and all_counts["ready"] == 1
    alice = cs.counts_by_status(user_id="alice")
    assert alice["new"] == 1 and alice["ready"] == 1
    bob = cs.counts_by_status(user_id="bob")
    assert bob["new"] == 1 and bob["ready"] == 0


def test_empty_owner_list_matches_nothing(captures_store_db, monkeypatch,
                                          tmp_path):
    """Only None means "no owner filter". A list with no usable id is an
    empty owner set and must fail CLOSED, never widen to every user."""
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path, user_id="alice")
    _make(cs, monkeypatch, tmp_path, user_id="bob")

    assert cs._user_clause(None) == ("", [])
    for empty in ([], [""], [None]):
        assert cs.count(user_id=empty) == 0
        assert cs.list_captures(user_id=empty) == []
        assert sum(cs.counts_by_status(user_id=empty).values()) == 0
        assert cs.stats(user_id=empty)["total"]["n"] == 0
    assert cs.count(user_id=["alice", "bob"]) == 2


def test_get_captures_light_projection_and_chunking(captures_store_db,
                                                    monkeypatch, tmp_path):
    """The bulk guard's lookup: five columns, no words / segments decode,
    unknown ids absent, and more ids than one IN-list chunk."""
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path, user_id="alice")
    conn = cs._require_conn()
    many = [f"light{i:07d}" for i in range(1100)]
    conn.executemany(
        "INSERT INTO captures (id, created_ts, model, audio_relpath,"
        " audio_format, raw_text, final_text, words, segments, corrections,"
        " status, user_id) VALUES (?,1.0,'m','x.wav','wav','r','f','[]','[]',"
        "'[]','new','bob')", [(i,) for i in many])

    # Builds differ (Debian's sqlite allows 250000 variables, stock 32766):
    # lower the limit below the 1101 ids so one unchunked IN-list would fail.
    old = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    try:
        got = cs.get_captures_light([cid, "nope", cid, "", *many])
    finally:
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, old)
    assert got[cid] == {"id": cid, "user_id": "alice", "status": "new",
                        "sample_id": None,
                        "audio_relpath": cs.get_capture(cid)["audio_relpath"]}
    assert "nope" not in got and len(got) == 1101
    assert cs.get_captures_light([]) == {}


def test_update_invalid_status_raises(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        cs.update_capture(cid, {"status": "bogus"})


def test_update_status_toggles_reviewed_ts(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    row = cs.update_capture(cid, {"status": "reviewed"})
    assert row["status"] == "reviewed" and row["reviewed_ts"] is not None
    row = cs.update_capture(cid, {"status": "new"})
    assert row["reviewed_ts"] is None


def test_update_corrections_cleaned(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    # The entry with an empty `correct` is dropped.
    row = cs.update_capture(cid, {"corrections": [
        {"wrong": "a", "correct": "b"},
        {"wrong": "x", "correct": ""},
    ]})
    assert len(row["corrections"]) == 1
    assert row["corrections"][0]["correct"] == "b"


def test_update_caps_and_fields(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    row = cs.update_capture(cid, {
        "corrected_text": "c" * (cs._CAP_CORRECTED + 50),
        "admin_notes": "n" * (cs._CAP_ADMIN_NOTES + 50),
        "final": "newfinal",
        "text_for_training": "tft",
        "audio_trim_lead_ms": 120,
        "audio_trim_trail_ms": 80,
    })
    assert len(row["corrected_text"]) == cs._CAP_CORRECTED
    assert len(row["admin_notes"]) == cs._CAP_ADMIN_NOTES
    assert row["final"] == "newfinal"
    assert row["text_for_training"] == "tft"
    assert row["audio_trim_lead_ms"] == 120
    assert row["audio_trim_trail_ms"] == 80


def test_update_empty_patch_returns_current(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    assert cs.update_capture(cid, {})["id"] == cid


def test_update_missing_row_returns_none(captures_store_db):
    assert captures_store_db.update_capture("nope", {"status": "ready"}) is None


def test_delete_capture_returns_bool_and_removes_audio(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    abs_p = cs.abs_audio_path(cs.get_capture(cid)["audio_relpath"])
    assert cs.delete_capture(cid) is True
    assert cs.get_capture(cid) is None
    assert not os.path.isfile(abs_p)
    assert cs.delete_capture(cid) is False  # already gone


def test_delete_member_auto_dissolves_group(captures_store_db, groups_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    gs = groups_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    conn = cs._require_conn()
    sid = "g0123456789abcdef"
    conn.execute(
        "INSERT INTO capture_samples (id, user_id, created_ts,"
        " merged_wav_relpath, merged_duration_ms, transcript,"
        " member_hashes) VALUES (?,?,?,?,?,?,?)",
        (sid, "u1", 1.0, gs._relpath_for(sid), 1000, "t", "{}"),
    )
    conn.execute(
        "UPDATE captures SET sample_id=?, sample_order=0 WHERE id=?", (sid, cid),
    )
    assert gs.get_sample(sid) is not None
    assert cs.delete_capture(cid) is True
    assert gs.get_sample(sid) is None


def test_clear_all_wipes_rows_and_files(captures_store_db, groups_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path)
    _make(cs, monkeypatch, tmp_path)
    n = cs.clear_all(reporter_host="127.0.0.1")
    assert n == 2
    assert cs.count() == 0
    # Hex-fanout subdirs removed.
    remaining = [d for d in os.listdir(cs._require_audio_dir())
                 if os.path.isdir(os.path.join(cs._require_audio_dir(), d)) and d != "groups"]
    assert remaining == []


def test_clear_all_survives_a_reader_mid_statement(captures_store_db, groups_store_db,
                                                   monkeypatch, tmp_path):
    """SQLite refuses VACUUM while a reader's statement is open on the shared
    connection; the raise came after the DELETE but before the audio rmtree,
    leaving the files of rows that no longer exist."""
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path)
    _make(cs, monkeypatch, tmp_path)
    cur = cs._require_conn().execute("SELECT id FROM captures")
    cur.fetchone()                      # statement still open
    try:
        assert cs.clear_all(reporter_host="127.0.0.1") == 2
    finally:
        cur.close()
    assert cs.count() == 0
    remaining = [d for d in os.listdir(cs._require_audio_dir())
                 if os.path.isdir(os.path.join(cs._require_audio_dir(), d)) and d != "groups"]
    assert remaining == []


def test_clear_all_wipes_the_audio_tree_under_the_lock(captures_store_db,
                                                       groups_store_db,
                                                       monkeypatch, tmp_path):
    """create_capture inserts under _lock; a wipe after clear_all released it
    deleted the audio of a row inserted after the DELETE."""
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path)
    real_rmtree = cs.shutil.rmtree
    held = []

    def spy(path, *a, **k):
        held.append(cs._lock.locked())
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(cs.shutil, "rmtree", spy)
    cs.clear_all(reporter_host="127.0.0.1")
    assert held and all(held)


def test_clear_all_drops_a_sample_merged_after_the_samples_wipe(
        captures_store_db, groups_store_db, monkeypatch, tmp_path):
    """A Merge waiting on the samples lock inserts its row once
    clear_all_samples releases it, before the captures DELETE. The rmtree then
    took its merged WAV, leaving a memberless sample with no audio."""
    cs = captures_store_db
    gs = groups_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    sid = "g0123456789abcdef"
    real_clear = gs.clear_all_samples

    def clear_then_merge():
        n = real_clear()
        conn = cs._require_conn()
        conn.execute(
            "INSERT INTO capture_samples (id, user_id, created_ts,"
            " merged_wav_relpath, merged_duration_ms, transcript,"
            " member_hashes) VALUES (?,?,?,?,?,?,?)",
            (sid, "u1", 1.0, gs._relpath_for(sid), 1000, "t", "{}"),
        )
        conn.execute(
            "UPDATE captures SET sample_id=?, sample_order=0 WHERE id=?", (sid, cid),
        )
        return n

    monkeypatch.setattr(gs, "clear_all_samples", clear_then_merge)
    assert cs.clear_all(reporter_host="127.0.0.1") == 1
    assert gs.get_sample(sid) is None


def test_create_capture_skips_the_row_when_a_clear_wiped_its_audio(
        captures_store_db, monkeypatch, tmp_path):
    """The WAV lands before the INSERT takes _lock. A clear_all that wipes it
    in between must not leave a "new" row with no audio."""
    import threading
    import time

    cs = captures_store_db
    audio_dir = cs._require_audio_dir()
    # The cap probe up front reads under _lock too; this test is about the
    # INSERT's window only.
    monkeypatch.setattr(cs, "ready_fills_cap", lambda: False)

    def wavs():
        return [os.path.join(r, f) for r, _d, fs in os.walk(audio_dir)
                for f in fs if f.endswith(".wav")]

    out = []
    with cs._lock:
        t = threading.Thread(target=lambda: out.append(_make(cs, monkeypatch, tmp_path)))
        t.start()
        deadline = time.monotonic() + 5
        while not wavs() and time.monotonic() < deadline:
            time.sleep(0.01)
        # The wipe clear_all does while holding _lock.
        for p in wavs():
            os.unlink(p)
    t.join(5)
    assert out == [None]
    assert cs.count() == 0


def test_evict_priority_dismissed_first(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    from faster_whisper_backend.settings import config
    monkeypatch.setattr(config, "CAPTURES_MAX", 5000, raising=False)
    monkeypatch.setattr(config, "CAPTURES_MAX_MB", 5000, raising=False)
    c_dismissed = _make(cs, monkeypatch, tmp_path)
    c_new = _make(cs, monkeypatch, tmp_path)
    cs.update_capture(c_dismissed, {"status": "dismissed"})
    # Make the dismissed row the oldest so ASC ordering picks it anyway.
    conn = cs._require_conn()
    conn.execute("UPDATE captures SET created_ts=1 WHERE id=?", (c_dismissed,))
    conn.execute("UPDATE captures SET created_ts=2 WHERE id=?", (c_new,))
    monkeypatch.setattr(config, "CAPTURES_MAX", 2, raising=False)
    _make(cs, monkeypatch, tmp_path)
    assert cs.get_capture(c_dismissed) is None
    assert cs.get_capture(c_new) is not None


def test_evict_excludes_group_members(captures_store_db, groups_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    gs = groups_store_db
    from faster_whisper_backend.settings import config
    monkeypatch.setattr(config, "CAPTURES_MAX", 5000, raising=False)
    monkeypatch.setattr(config, "CAPTURES_MAX_MB", 5000, raising=False)
    member = _make(cs, monkeypatch, tmp_path)
    other = _make(cs, monkeypatch, tmp_path)
    sid = "gmemberexcl00000"
    conn = cs._require_conn()
    conn.execute(
        "INSERT INTO capture_samples (id, user_id, created_ts,"
        " merged_wav_relpath, merged_duration_ms, transcript,"
        " member_hashes) VALUES (?,?,?,?,?,?,?)",
        (sid, "u1", 1.0, gs._relpath_for(sid), 1000, "t", "{}"),
    )
    # member is the oldest, but it's in a group → must be protected.
    conn.execute("UPDATE captures SET sample_id=?, sample_order=0, created_ts=1 WHERE id=?", (sid, member))
    conn.execute("UPDATE captures SET created_ts=2 WHERE id=?", (other,))
    # The cap governs the EVICTABLE population only: grouped rows are neither
    # counted nor dropped, because eviction could never reduce them anyway.
    # Two ungrouped rows against a cap of 1 leaves exactly one to drop.
    monkeypatch.setattr(config, "CAPTURES_MAX", 1, raising=False)
    fresh = _make(cs, monkeypatch, tmp_path)
    assert cs.get_capture(member) is not None
    assert cs.get_capture(other) is None       # the only evictable old row went
    # Counting the grouped row too would make excess 2 and drop this one.
    assert cs.get_capture(fresh) is not None
    n_ungrouped = conn.execute(
        "SELECT COUNT(*) FROM captures WHERE sample_id IS NULL").fetchone()[0]
    assert n_ungrouped == 1


def test_evicting_create_invalidates_every_proposer_key(
        captures_store_db, monkeypatch, tmp_path):
    """The cap eviction drops OTHER identities' oldest rows, so invalidating
    only the inserter's key left another user's cached proposals naming
    deleted captures."""
    cs = captures_store_db
    from faster_whisper_backend.captures import merge_proposer
    from faster_whisper_backend.settings import config
    monkeypatch.setattr(config, "CAPTURES_MAX", 5000, raising=False)
    monkeypatch.setattr(config, "CAPTURES_MAX_MB", 5000, raising=False)
    calls = []
    monkeypatch.setattr(merge_proposer, "invalidate", calls.append)
    old = _make(cs, monkeypatch, tmp_path, user_id="u2")
    assert calls == ["u2"]
    monkeypatch.setattr(config, "CAPTURES_MAX", 1, raising=False)
    _make(cs, monkeypatch, tmp_path, user_id="u1")
    assert cs.get_capture(old) is None
    assert calls == ["u2", None]


def test_failing_eviction_keeps_the_committed_row_and_its_audio(
        captures_store_db, monkeypatch, tmp_path):
    """The INSERT is committed (autocommit connection) before the eviction
    runs, so an eviction error must not unlink the new row's WAV."""
    cs = captures_store_db

    def _boom(conn):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(cs, "_evict_to_cap", _boom)
    cid = _make(cs, monkeypatch, tmp_path)
    row = cs.get_capture(cid)
    assert row is not None
    assert os.path.isfile(cs.abs_audio_path(row["audio_relpath"]))


def test_evict_deletes_more_ids_than_one_in_list_chunk(
        captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    from faster_whisper_backend.settings import config
    conn = cs._require_conn()
    many = [f"evict{i:07d}" for i in range(1100)]
    conn.executemany(
        "INSERT INTO captures (id, created_ts, model, audio_relpath,"
        " audio_format, raw_text, final_text, words, segments, corrections,"
        " status, user_id) VALUES (?,1.0,'m','x.wav','wav','r','f','[]','[]',"
        "'[]','dismissed','bob')", [(i,) for i in many])
    monkeypatch.setattr(config, "CAPTURES_MAX", 1, raising=False)
    monkeypatch.setattr(config, "CAPTURES_MAX_MB", 0, raising=False)
    # Builds differ (Debian's sqlite allows 250000 variables, stock 32766):
    # lower the limit below the 1099 ids so one unchunked IN-list would fail.
    old = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    try:
        with cs._lock:
            assert cs._evict_to_cap(conn) is True
    finally:
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, old)
    assert cs.count() == 1


def test_byte_pass_keeps_rows_whose_audio_is_already_gone(
        captures_store_db, monkeypatch):
    """A row with no audio frees 0 bytes, yet the byte pass dropped it and
    kept going, so the "audio_missing" pass wiped every such row (which
    reconcile_on_startup keeps for the admin) for nothing gained."""
    cs = captures_store_db
    from faster_whisper_backend.settings import config
    conn = cs._require_conn()
    missing = [f"gone{i:011d}" for i in range(10)]
    conn.executemany(
        "INSERT INTO captures (id, created_ts, model, audio_relpath,"
        " audio_format, raw_text, final_text, words, segments, corrections,"
        " status, user_id) VALUES (?,1.0,'m',?,'wav','r','f','[]','[]',"
        "'[]','audio_missing','bob')",
        [(i, cs._relpath_for(i, ".wav")) for i in missing])
    fresh = [f"new{i:012d}" for i in range(3)]
    for n, cid in enumerate(fresh):
        rel = cs._relpath_for(cid, ".wav")
        path = cs.abs_audio_path(rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\x00" * (1024 * 1024))
        conn.execute(
            "INSERT INTO captures (id, created_ts, model, audio_relpath,"
            " audio_format, raw_text, final_text, words, segments,"
            " corrections, status, user_id) VALUES (?,?,'m',?,'wav','r','f',"
            "'[]','[]','[]','new','bob')", (cid, 2.0 + n, rel))
    monkeypatch.setattr(config, "CAPTURES_MAX", 5000, raising=False)
    monkeypatch.setattr(config, "CAPTURES_MAX_MB", 2, raising=False)
    with cs._lock:
        assert cs._evict_to_cap(conn) is True
    assert all(cs.get_capture(i) is not None for i in missing)
    assert cs.get_capture(fresh[0]) is None
    assert cs.count() == 12


def test_bulk_update_status_chunks_past_the_bind_variable_limit(
        captures_store_db):
    """BulkStatusIn admits 1000 ids; one IN-list of them (plus the two other
    binds) overran the classic 999-variable limit and the route 500ed."""
    cs = captures_store_db
    conn = cs._require_conn()
    ids = [f"bulkst{i:06d}" for i in range(1000)]
    conn.executemany(
        "INSERT INTO captures (id, created_ts, model, audio_relpath,"
        " audio_format, raw_text, final_text, words, segments, corrections,"
        " status, user_id) VALUES (?,1.0,'m','x.wav','wav','r','f','[]','[]',"
        "'[]','new','bob')", [(i,) for i in ids])
    old = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    try:
        out = cs.bulk_update_status(ids, "ready")
    finally:
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, old)
    assert len(out) == 1000
    assert {o["prev_status"] for o in out} == {"new"}
    assert conn.execute(
        "SELECT COUNT(*) FROM captures WHERE status = 'ready'"
        " AND reviewed_ts IS NOT NULL").fetchone()[0] == 1000
    assert not conn.in_transaction


def test_bulk_update_status_holds_the_samples_lock(captures_store_db):
    """The BEGIN..COMMIT runs on the connection samples_store shares, so it
    must hold samples_store._lock too: a sample patch from a rebuild thread
    (samples lock only) would otherwise join the transaction and be lost if
    it rolls back."""
    import threading

    from faster_whisper_backend.captures import samples_store as capture_samples_store

    cs = captures_store_db
    cs._require_conn().execute(
        "INSERT INTO captures (id, created_ts, model, audio_relpath,"
        " audio_format, raw_text, final_text, words, segments, corrections,"
        " status, user_id) VALUES ('bulklock0000001',1.0,'m','x.wav','wav',"
        "'r','f','[]','[]','[]','new','bob')")
    done = threading.Event()
    with capture_samples_store._lock:
        t = threading.Thread(target=lambda: (
            cs.bulk_update_status(["bulklock0000001"], "ready"), done.set()))
        t.start()
        assert not done.wait(0.3)
    t.join(5)
    assert done.is_set()
    assert cs.get_capture("bulklock0000001")["status"] == "ready"


def _seed_ready(cs, n, ts=1.0):
    cs._require_conn().executemany(
        "INSERT INTO captures (id, created_ts, model, audio_relpath,"
        " audio_format, raw_text, final_text, words, segments, corrections,"
        " status, user_id) VALUES (?,?,'m','x.wav','wav','r','f','[]','[]',"
        "'[]','ready','bob')", [(f"ready{i:07d}", ts) for i in range(n)])


def test_a_store_full_of_ready_rows_records_nothing(
        captures_store_db, monkeypatch, tmp_path):
    """"ready" is evicted last, so with ready rows filling CAPTURES_MAX the new
    row was the one evicted by its own insert: the caller still got its id and
    traced a capture link to a deleted row, after a wasted transcode."""
    cs = captures_store_db
    from faster_whisper_backend.audio import transcode as audio_transcode
    from faster_whisper_backend.settings import config
    monkeypatch.setattr(config, "CAPTURES_MAX", 10, raising=False)
    monkeypatch.setattr(config, "CAPTURES_MAX_MB", 0, raising=False)
    _seed_ready(cs, 10)
    assert cs.ready_fills_cap() is True
    _fake_transcode(cs, monkeypatch)
    src = tmp_path / "src_in.bin"
    src.write_bytes(b"junk")
    calls = []
    real = audio_transcode.transcode_to_wav_16k_mono
    monkeypatch.setattr(audio_transcode, "transcode_to_wav_16k_mono",
                        lambda a, b: calls.append(a) or real(a, b))
    assert cs.create_capture(
        audio_src_path=str(src), request_id="r", model="m", language="de",
        audio_s=1.0, raw="r", final="f", words=[], segments=[]) is None
    assert calls == []
    assert cs.count() == 10


def test_a_byte_cap_that_evicts_the_new_row_returns_none(
        captures_store_db, monkeypatch, tmp_path):
    """The byte pass reaches "new" before "ready", so ready audio over
    CAPTURES_MAX_MB evicts the fresh row: create_capture must not hand back
    the id of a row it already deleted."""
    cs = captures_store_db
    from faster_whisper_backend.settings import config
    monkeypatch.setattr(config, "CAPTURES_MAX", 5000, raising=False)
    monkeypatch.setattr(config, "CAPTURES_MAX_MB", 1, raising=False)
    ready = _make(cs, monkeypatch, tmp_path)
    cs.update_capture(ready, {"status": "ready"})
    # Just under the cap alone; the new row's WAV tips the total over it.
    with open(cs.abs_audio_path(cs.get_capture(ready)["audio_relpath"]),
              "r+b") as f:
        f.truncate(1024 * 1024 - 16)
    assert _make(cs, monkeypatch, tmp_path) is None
    assert cs.count() == 1
    assert cs.get_capture(ready) is not None


def test_an_older_new_row_is_evicted_before_the_fresh_one(
        captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    from faster_whisper_backend.settings import config
    monkeypatch.setattr(config, "CAPTURES_MAX", 5000, raising=False)
    monkeypatch.setattr(config, "CAPTURES_MAX_MB", 0, raising=False)
    older = _make(cs, monkeypatch, tmp_path)
    cs._require_conn().execute(
        "UPDATE captures SET created_ts=0.5 WHERE id=?", (older,))
    _seed_ready(cs, 9)
    monkeypatch.setattr(config, "CAPTURES_MAX", 10, raising=False)
    assert cs.ready_fills_cap() is False
    fresh = _make(cs, monkeypatch, tmp_path)
    assert fresh is not None and cs.get_capture(fresh) is not None
    assert cs.get_capture(older) is None
    assert cs.count() == 10


def test_evict_disabled_when_caps_below_one(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    from faster_whisper_backend.settings import config
    monkeypatch.setattr(config, "CAPTURES_MAX", 0, raising=False)
    monkeypatch.setattr(config, "CAPTURES_MAX_MB", 0, raising=False)
    for _ in range(4):
        _make(cs, monkeypatch, tmp_path)
    assert cs.count() == 4


def test_reconcile_marks_missing_and_unlinks_orphans(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    keep = _make(cs, monkeypatch, tmp_path)
    missing = _make(cs, monkeypatch, tmp_path)
    miss_abs = cs.abs_audio_path(cs.get_capture(missing)["audio_relpath"])
    os.unlink(miss_abs)
    audio = cs._require_audio_dir()
    orphan_dir = os.path.join(audio, "zz", "zz")
    os.makedirs(orphan_dir, exist_ok=True)
    orphan = os.path.join(orphan_dir, "orphan.wav")
    with open(orphan, "wb") as f:
        f.write(b"x")
    tmpf = os.path.join(orphan_dir, "partial.wav.tmp")
    with open(tmpf, "wb") as f:
        f.write(b"x")

    marked, unlinked = cs.reconcile_on_startup()
    assert marked == 1
    assert cs.get_capture(missing)["status"] == "audio_missing"
    assert cs.get_capture(keep)["status"] == "new"
    assert not os.path.isfile(orphan)
    assert not os.path.isfile(tmpf)
    assert unlinked >= 2


def test_reconcile_skips_groups_subtree(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path)
    # A file under groups/ must NOT be treated as an orphan.
    groups_dir = os.path.join(cs._require_audio_dir(), "groups", "ab", "cd")
    os.makedirs(groups_dir, exist_ok=True)
    merged = os.path.join(groups_dir, "merged.wav")
    with open(merged, "wb") as f:
        f.write(b"x")
    cs.reconcile_on_startup()
    assert os.path.isfile(merged)


def _hide_audio(cs, cid, tmp_path):
    """Move a row's WAV out of the audio tree (an unmounted volume; inside
    it reconcile would unlink it as an orphan) and return a callable that
    puts it back."""
    abs_p = cs.abs_audio_path(cs.get_capture(cid)["audio_relpath"])
    aside = str(tmp_path / f"{cid}.aside")
    os.rename(abs_p, aside)
    return lambda: os.rename(aside, abs_p)


def test_reconcile_gives_back_the_status_once_the_audio_returns(
    captures_store_db, groups_store_db, monkeypatch, tmp_path,
):
    """audio_missing used to be a one-way trip: a reviewed or ready capture
    whose volume mounted late stayed in eviction tier 2 and out of the
    ready export for good, though its WAV was back."""
    cs = captures_store_db
    reviewed = _make(cs, monkeypatch, tmp_path)
    ready = _make(cs, monkeypatch, tmp_path)
    cs.update_capture(reviewed, {"status": "reviewed"})
    cs.update_capture(ready, {"status": "ready"})
    # A sample member: its membership must survive the round trip untouched.
    _group_one(cs, groups_store_db, "gmissing00000000", ready,
               sample_ts=1.0, member_ts=1.0)
    stamp = cs.get_capture(ready)["reviewed_ts"]
    restore = [_hide_audio(cs, reviewed, tmp_path),
               _hide_audio(cs, ready, tmp_path)]

    assert cs.reconcile_on_startup()[0] == 2
    assert cs.get_capture(reviewed)["status"] == "audio_missing"
    assert cs.get_capture(ready)["status"] == "audio_missing"
    # Bookkeeping stays off the wire.
    assert "status_before_missing" not in cs.get_capture(ready)
    # A second boot with the audio still gone keeps both marked and keeps
    # the saved status (the mark is not applied twice over itself).
    assert cs.reconcile_on_startup()[0] == 0
    assert cs.get_capture(ready)["status"] == "audio_missing"

    for put_back in restore:
        put_back()
    assert cs.reconcile_on_startup() == (0, 0)  # the WAVs are no orphans
    got = cs.get_capture(ready)
    assert got["status"] == "ready" and got["reviewed_ts"] == stamp
    assert got["sample_id"] == "gmissing00000000" and got["sample_order"] == 0
    assert cs.get_capture(reviewed)["status"] == "reviewed"
    saved = cs._require_conn().execute(
        "SELECT status_before_missing FROM captures WHERE id IN (?, ?)",
        (reviewed, ready)).fetchall()
    assert [r[0] for r in saved] == [None, None]


def test_reconcile_restores_a_row_marked_before_the_column_to_new(
    captures_store_db, monkeypatch, tmp_path,
):
    """A row marked audio_missing by an older version kept no previous status:
    it comes back as 'new' (unreviewed, so no reviewed_ts), while a row whose
    audio is still gone stays marked."""
    cs = captures_store_db
    old = _make(cs, monkeypatch, tmp_path)
    gone = _make(cs, monkeypatch, tmp_path)
    cs._require_conn().execute(
        "UPDATE captures SET status='audio_missing', reviewed_ts=5.0"
        " WHERE id IN (?, ?)", (old, gone))
    cs._safe_unlink(cs.abs_audio_path(cs.get_capture(gone)["audio_relpath"]))

    assert cs.reconcile_on_startup()[0] == 0
    got = cs.get_capture(old)
    assert got["status"] == "new" and got["reviewed_ts"] is None
    assert cs.get_capture(gone)["status"] == "audio_missing"


def test_sweep_retention_disabled(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    from faster_whisper_backend.settings import config
    monkeypatch.setattr(config, "CAPTURES_RETENTION_DAYS", 0, raising=False)
    _make(cs, monkeypatch, tmp_path)
    assert cs.sweep_retention() == 0


def test_sweep_retention_deletes_old(captures_store_db, monkeypatch, tmp_path):
    import time
    cs = captures_store_db
    from faster_whisper_backend.settings import config
    monkeypatch.setattr(config, "CAPTURES_RETENTION_DAYS", 30, raising=False)
    old = _make(cs, monkeypatch, tmp_path)
    _make(cs, monkeypatch, tmp_path)  # fresh
    conn = cs._require_conn()
    conn.execute("UPDATE captures SET created_ts=? WHERE id=?",
                 (time.time() - 40 * 86400, old))
    assert cs.sweep_retention() == 1
    assert cs.get_capture(old) is None


def test_sweep_retention_deletes_more_ids_than_one_in_list_chunk(
        captures_store_db, monkeypatch):
    """An over-age set larger than SQLite's bind-variable limit must still be
    swept: one unchunked IN-list raised "too many SQL variables" on every pass,
    so retention silently stopped deleting anything."""
    cs = captures_store_db
    from faster_whisper_backend.settings import config
    monkeypatch.setattr(config, "CAPTURES_RETENTION_DAYS", 30, raising=False)
    conn = cs._require_conn()
    conn.executemany(
        "INSERT INTO captures (id, created_ts, model, audio_relpath,"
        " audio_format, raw_text, final_text, words, segments, corrections,"
        " status, user_id) VALUES (?,1.0,'m','x.wav','wav','r','f','[]','[]',"
        "'[]','new','bob')", [(f"aged{i:07d}",) for i in range(1100)])
    old = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    try:
        assert cs.sweep_retention() == 1100
    finally:
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, old)
    assert cs.count() == 0


def _group_one(cs, gs, sid, member, sample_ts, member_ts):
    conn = cs._require_conn()
    conn.execute(
        "INSERT INTO capture_samples (id, user_id, created_ts,"
        " merged_wav_relpath, merged_duration_ms, transcript,"
        " member_hashes) VALUES (?,?,?,?,?,?,?)",
        (sid, "u1", sample_ts, gs._relpath_for(sid), 1000, "t", "{}"),
    )
    conn.execute(
        "UPDATE captures SET sample_id=?, sample_order=0, created_ts=? WHERE id=?",
        (sid, member_ts, member),
    )


def test_sweep_retention_protects_members_of_a_current_sample(
    captures_store_db, groups_store_db, monkeypatch, tmp_path,
):
    """A member cannot be aged out on its own — it backs the sample's merged
    WAV, so retention only ever acts on the sample as a whole."""
    import time
    cs = captures_store_db
    gs = groups_store_db
    from faster_whisper_backend.settings import config
    monkeypatch.setattr(config, "CAPTURES_RETENTION_DAYS", 30, raising=False)
    member = _make(cs, monkeypatch, tmp_path)
    _group_one(cs, gs, "gretention000000", member,
               sample_ts=time.time(), member_ts=time.time() - 99 * 86400)
    assert cs.sweep_retention() == 0
    assert cs.get_capture(member) is not None


def test_count_evictable_ignores_grouped_members(
    captures_store_db, groups_store_db, monkeypatch, tmp_path,
):
    """count_evictable() mirrors _evict_to_cap's scope: grouped rows are
    bounded by retention, not the cap, so they must not count. No ingest gate
    reads it any more (a full store rotates in create_capture); the two "full
    store still records" route tests patch it so a revived count gate fails."""
    import time
    cs = captures_store_db
    gs = groups_store_db
    member = _make(cs, monkeypatch, tmp_path)
    ungrouped = _make(cs, monkeypatch, tmp_path)
    assert cs.count() == 2
    assert cs.count_evictable() == 2
    now = time.time()
    _group_one(cs, gs, "gcountevict00000", member,
               sample_ts=now, member_ts=now)
    assert cs.count() == 2
    assert cs.count_evictable() == 1
    assert cs.get_capture(ungrouped) is not None


def test_sweep_retention_expires_an_over_age_sample_and_its_members(
    captures_store_db, groups_store_db, monkeypatch, tmp_path,
):
    """Grouping must not be a way to opt out of the retention policy: once the
    SAMPLE itself is past the cutoff it is dissolved and its members are swept
    like any other row."""
    import time
    cs = captures_store_db
    gs = groups_store_db
    from faster_whisper_backend.settings import config
    monkeypatch.setattr(config, "CAPTURES_RETENTION_DAYS", 30, raising=False)
    member = _make(cs, monkeypatch, tmp_path)
    sid = "gretentionold000"
    _group_one(cs, gs, sid, member,
               sample_ts=time.time() - 99 * 86400,
               member_ts=time.time() - 99 * 86400)
    assert cs.sweep_retention() == 1
    assert cs.get_capture(member) is None
    assert gs.get_sample(sid) is None


def test_create_capture_oversized_translations_stay_valid_json(
        captures_store_db, monkeypatch, tmp_path):
    """A raw slice of the JSON blob lands mid-token; the reader then swallows
    the decode error as {} and EVERY track is lost. Trim the text instead."""
    cs = captures_store_db
    big = {"en": "e" * cs._CAP_FINAL, "fr": "f" * cs._CAP_FINAL}
    cid = _make(cs, monkeypatch, tmp_path, translations=big)
    raw = cs._require_conn().execute(
        "SELECT translations FROM captures WHERE id = ?", (cid,)
    ).fetchone()[0]
    assert len(raw) <= cs._CAP_FINAL
    got = json.loads(raw)  # valid JSON, not a torn slice
    assert set(got) == {"en", "fr"}
    assert all(v and v[0] == k[0] for k, v in got.items())
    assert cs.get_capture(cid)["translations"] == got


def test_truncate_translations_small_map_is_untouched():
    from faster_whisper_backend.captures import store as captures_store
    blob = captures_store._truncate_translations({"en": "hi"}, 50_000)
    assert json.loads(blob) == {"en": "hi"}
    assert captures_store._truncate_translations({}, 50_000) is None
    assert captures_store._truncate_translations(None, 50_000) is None


def test_update_translations_rejects_non_dict_and_caps_dict(
        captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        cs.update_capture(cid, {"translations": ["not", "a", "map"]})
    with pytest.raises(ValueError):
        cs.update_capture(cid, {"translations": "en: hi"})
    assert cs.get_capture(cid)["translations"] == {}

    row = cs.update_capture(cid, {"translations": {"en": "e" * (cs._CAP_FINAL * 2)}})
    assert set(row["translations"]) == {"en"}
    assert 0 < len(row["translations"]["en"]) < cs._CAP_FINAL
    row = cs.update_capture(cid, {"translations": None})
    assert row["translations"] == {}


# Public reads hold _lock: two threads running the same SELECT text on the
# shared connection can read each other's rows (another user's captures).
def test_reads_hold_the_lock(captures_store_db, monkeypatch, tmp_path):
    from tests.stats.test_store_reads_hold_lock import _LockCheckingConn

    cs = captures_store_db
    cid = _make(cs, monkeypatch, tmp_path)
    conn = _LockCheckingConn(cs._conn, cs._lock)
    monkeypatch.setattr(cs, "_conn", conn)

    assert cs.count(user_id="u1") == 1
    assert cs.count_evictable() == 1
    assert cs.ready_fills_cap() is False
    assert [r["id"] for r in cs.list_captures(user_id="u1")] == [cid]
    assert cs.get_capture(cid)["id"] == cid
    assert list(cs.get_captures_light([cid])) == [cid]
    assert [r["id"] for r in cs.find_by_request_id("req-1")] == [cid]
    assert cs.counts_by_status(user_id="u1")["new"] == 1
    assert cs.stats(user_id="u1")["total"]["n"] == 1
    assert [r["id"] for r in cs.iter_captures_for_export(user_id="u1")] == [cid]
    assert conn.unlocked == []


def test_export_iterator_releases_the_lock_between_rows(captures_store_db, monkeypatch, tmp_path):
    cs = captures_store_db
    _make(cs, monkeypatch, tmp_path)
    it = cs.iter_captures_for_export()
    next(it)
    # The export writes each row to the tarball between yields: other store
    # calls must not wait on it.
    assert cs._lock.acquire(blocking=False)
    cs._lock.release()
    assert list(it) == []
