"""A report filed on a live (SSE-pushed) trace carries no stages: the entry
leaves the page before record_timing writes them. The submit route fills
them from the caller's recent-transcriptions row by request_id."""

from tests.conftest import bearer

_SUBMIT = "/quick-config/reports/api/submit"
_STAGES = [{"name": "transcribing", "secs": 1.5}, {"name": "translate", "secs": 0.4}]


def _seed_trace(request_id, user_id):
    from faster_whisper_backend.stats import recent_transcriptions_store as rts
    rts.record_trace(request_id=request_id, model="m", raw="a b", final="a b",
                     language="de", user_id=user_id)
    rts.record_timing(request_id=request_id, model="m", audio_s=3.0,
                      processing_s=2.0, status="ok", words=2, user_id=user_id,
                      stages=_STAGES)


def _stored(rid):
    from faster_whisper_backend.reports import store as reports_store
    return reports_store.get_report(rid)


def test_submit_without_stages_takes_them_from_the_trace_row(client, make_user_key):
    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    _seed_trace("live-1", uid)
    r = client.post(_SUBMIT, json={"request_id": "live-1", "user_comment": "x"},
                    headers=bearer(raw))
    assert r.status_code == 200
    row = _stored(r.json()["id"])
    assert row["stages"] == _STAGES
    assert row["language"] == "de"


def test_submit_does_not_take_stages_from_another_users_trace(client, make_user_key):
    make_user_key("root", is_admin=True)
    bob_uid, _ = make_user_key("bob", pages={"quick_config": "own"})
    _uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    _seed_trace("bobs-1", bob_uid)
    r = client.post(_SUBMIT, json={"request_id": "bobs-1", "user_comment": "x"},
                    headers=bearer(raw))
    assert r.status_code == 200
    assert _stored(r.json()["id"])["stages"] == []


def test_client_sent_stages_win_over_the_trace_row(client, make_user_key):
    _uid, raw = make_user_key("root", is_admin=True)
    _seed_trace("live-2", _uid)
    sent = [{"name": "transcribing", "secs": 9.0}]
    r = client.post(_SUBMIT, json={"request_id": "live-2", "user_comment": "x",
                                   "stages": sent}, headers=bearer(raw))
    assert r.status_code == 200
    assert _stored(r.json()["id"])["stages"] == sent
