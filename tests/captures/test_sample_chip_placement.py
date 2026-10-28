"""Sample transcripts place each chip on the word its idx marks.

A chip's idx counts aligned karaoke words, and a removed raw word (a filler, a
dictated "Komma") owns no text token. The export passes the member's aligned
words to `_build_default_transcript`; every other route that builds or stores
a sample transcript must too, or the reviewer approves one text and the
trainer gets another. Here two removed fillers precede the first of two
"die"s and the chip marks that first one: the nearest-match fallback would
rewrite the second.
"""

import json

import pytest

from faster_whisper_backend.captures import routes as captures_routes
from tests.conftest import bearer

_SID = "chipplacesid0001"
_CID = "chipplacecap"
_CHIP = {"idx": 3, "wrong": "die", "correct": "der"}
_RAW_WORDS = [{"word": w} for w in ("äh", "äh", "äh", "die", "Katze", "die")]
_ALIGNED = [{"word": "äh", "removed": True}] * 3 + [
    {"word": "die"}, {"word": "Katze"}, {"word": "die"}]
_EXPECTED = "der Katze die"


@pytest.fixture
def aligned(monkeypatch):
    monkeypatch.setattr(captures_routes, "_align_member_words",
                        lambda m, ident_cache=None: _ALIGNED)


@pytest.fixture
def sample(client, make_user_key, aligned):
    """An unlocked one-member sample; returns the admin's bearer key."""
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    _uid, raw = make_user_key("root", is_admin=True)
    conn = captures_store._require_conn()
    conn.execute(
        "INSERT INTO capture_samples (id, user_id, created_ts,"
        " merged_wav_relpath, merged_duration_ms, transcript,"
        " transcript_join_strategy, member_hashes,"
        " inter_segment_silence_ms, is_stale, is_locked, status,"
        " admin_notes, language, member_trims)"
        " VALUES (?,'alice',1.0,?,5000,'t','space','{}',300,0,0,'new','',"
        "'de','{}')", (_SID, gs._relpath_for(_SID)))
    conn.execute(
        "INSERT INTO captures (id, created_ts, model, language, audio_s,"
        " audio_relpath, audio_format, raw_text, final_text,"
        " text_for_training, words, segments, corrections, status, user_id,"
        " sample_id, sample_order)"
        " VALUES (?,1.0,'m','de',2.0,'x.wav','wav',?,?,?,?,'[]',?,'new',"
        "'alice',?,0)",
        (_CID, "die Katze die", "die Katze die", "die Katze die",
         json.dumps(_RAW_WORDS), json.dumps([_CHIP]), _SID))
    return raw


def test_expanded_sample_places_a_chip_on_its_aligned_word(
        client, sample, monkeypatch):
    monkeypatch.setattr(captures_routes, "_refresh_final_if_stale",
                        lambda row, parent_locked=None: None)
    monkeypatch.setattr(captures_routes, "_build_merged_words",
                        lambda *a, **kw: [])
    r = client.get(f"/captures/api/samples/{_SID}", headers=bearer(sample))
    assert r.status_code == 200, r.text
    assert r.json()["sample"]["transcript"] == _EXPECTED


def test_chip_edit_stores_the_transcript_with_the_chip_on_its_aligned_word(
        client, sample):
    from faster_whisper_backend.captures import samples_store as gs

    r = client.patch(f"/captures/api/samples/{_SID}",
                     json={"corrections": [_CHIP]}, headers=bearer(sample))
    assert r.status_code == 200, r.text
    assert gs.get_sample(_SID)["transcript"] == _EXPECTED


def _merge_member():
    return {"id": _CID, "user_id": "alice", "model": "m", "language": "de",
            "text_for_training": "die Katze die", "final": "die Katze die",
            "words": _RAW_WORDS, "corrections": [_CHIP]}


def test_merge_preview_places_a_chip_on_its_aligned_word(
        client, make_user_key, aligned, monkeypatch):
    _uid, raw = make_user_key("root", is_admin=True)
    monkeypatch.setattr(captures_routes, "_validate_merge_payload",
                        lambda ids, silence_ms, user: (
                            [_merge_member()], "alice", ["x.wav"], 1000))
    monkeypatch.setattr(captures_routes, "_build_merged_words",
                        lambda *a, **kw: [])
    monkeypatch.setattr(captures_routes, "_preview_member_trims",
                        lambda ids, paths: {})
    r = client.post("/captures/api/samples/preview-words",
                    json={"member_ids": [_CID]}, headers=bearer(raw))
    assert r.status_code == 200, r.text
    assert r.json()["transcript"] == _EXPECTED


def test_merge_stores_the_transcript_with_the_chip_on_its_aligned_word(
        client, make_user_key, aligned, monkeypatch):
    from faster_whisper_backend.captures import samples as capture_samples

    _uid, raw = make_user_key("root", is_admin=True)
    monkeypatch.setattr(captures_routes, "_validate_merge_payload",
                        lambda ids, silence_ms, user: (
                            [_merge_member()], "alice", ["x.wav"], 1000))
    monkeypatch.setattr(capture_samples, "_build_merged_wav",
                        lambda **kw: (1000, {}, {}))
    inserted = []
    monkeypatch.setattr(captures_routes, "_insert_sample_with_sid",
                        lambda **kw: inserted.append(kw))
    r = client.post("/captures/api/samples",
                    json={"member_ids": [_CID]}, headers=bearer(raw))
    assert r.status_code == 200, r.text
    assert [kw["transcript"] for kw in inserted] == [_EXPECTED]
