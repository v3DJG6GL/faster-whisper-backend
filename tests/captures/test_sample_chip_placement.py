"""Sample transcripts place each chip on the word its idx marks.

A chip's idx counts aligned karaoke words, and a removed raw word (a filler, a
dictated "Komma") owns no text token. The export passes the member's aligned
words to `_build_default_transcript`; every other route that builds or stores
a sample transcript must too, or the reviewer approves one text and the
trainer gets another. Here three removed fillers precede the first of two
"die"s and the chip marks that first one: the nearest-match fallback would
rewrite the second.
"""

import json

import pytest

from faster_whisper_backend.captures import routes as captures_routes
from tests.captures.test_routes_captures import (
    _get, _insert_member, _insert_sample, _patch, _post_members,
)

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
def sample(client, root_key, aligned):
    """An unlocked one-member sample; returns the admin's bearer key."""
    from faster_whisper_backend.captures import samples_store as gs
    from faster_whisper_backend.captures import store as captures_store

    conn = captures_store._require_conn()
    _insert_sample(conn, gs, _SID, locked=False)
    _insert_member(conn, _CID, _SID, audio_relpath="x.wav",
                   raw_text="die Katze die", final_text="die Katze die",
                   text_for_training="die Katze die",
                   words=json.dumps(_RAW_WORDS),
                   corrections=json.dumps([_CHIP]))
    return root_key


def test_expanded_sample_places_a_chip_on_its_aligned_word(
        client, sample, monkeypatch):
    monkeypatch.setattr(captures_routes, "_refresh_final_if_stale",
                        lambda row, parent_locked=None: None)
    monkeypatch.setattr(captures_routes, "_build_merged_words",
                        lambda *a, **kw: [])
    r = _get(client, sample, f"samples/{_SID}")
    assert r.status_code == 200, r.text
    assert r.json()["sample"]["transcript"] == _EXPECTED


def test_chip_edit_stores_the_transcript_with_the_chip_on_its_aligned_word(
        client, sample):
    from faster_whisper_backend.captures import samples_store as gs

    r = _patch(client, sample, f"samples/{_SID}", corrections=[_CHIP])
    assert r.status_code == 200, r.text
    assert gs.get_sample(_SID)["transcript"] == _EXPECTED


def _merge_member():
    return {"id": _CID, "user_id": "alice", "model": "m", "language": "de",
            "text_for_training": "die Katze die", "final": "die Katze die",
            "words": _RAW_WORDS, "corrections": [_CHIP]}


def test_merge_preview_places_a_chip_on_its_aligned_word(
        client, root_key, aligned, monkeypatch):
    monkeypatch.setattr(captures_routes, "_validate_merge_payload",
                        lambda ids, silence_ms, user: (
                            [_merge_member()], "alice", ["x.wav"], 1000))
    monkeypatch.setattr(captures_routes, "_build_merged_words",
                        lambda *a, **kw: [])
    monkeypatch.setattr(captures_routes, "_preview_member_trims",
                        lambda ids, paths: {})
    r = _post_members(client, root_key, "samples/preview-words", [_CID])
    assert r.status_code == 200, r.text
    assert r.json()["transcript"] == _EXPECTED


def test_merge_stores_the_transcript_with_the_chip_on_its_aligned_word(
        client, root_key, aligned, monkeypatch):
    from faster_whisper_backend.captures import samples as capture_samples

    monkeypatch.setattr(captures_routes, "_validate_merge_payload",
                        lambda ids, silence_ms, user: (
                            [_merge_member()], "alice", ["x.wav"], 1000))
    monkeypatch.setattr(capture_samples, "_build_merged_wav",
                        lambda **kw: (1000, {}, {}))
    inserted = []
    monkeypatch.setattr(captures_routes, "_insert_sample_with_sid",
                        lambda **kw: inserted.append(kw))
    r = _post_members(client, root_key, "samples", [_CID])
    assert r.status_code == 200, r.text
    assert [kw["transcript"] for kw in inserted] == [_EXPECTED]
