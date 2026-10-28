"""A resubmitted report diffs the edited chip list against the chips the
/quick-config form was seeded from, so chips the user removed or re-ranged
leave the stored report instead of surviving a union merge."""

from tests.conftest import bearer

_SUBMIT = "/quick-config/reports/api/submit"

_B = {"wrong": "b", "correct": "B", "idx": 1}
_D = {"wrong": "d", "correct": "D", "idx": 3}
_BC = {"wrong": "b c", "correct": "BC", "idx": 1, "idx_end": 2}


def _submit(rs, corrections, baseline=None):
    return rs.upsert_report(
        user_id="u1", request_id="req1", trace_ts=1000.0, model="m",
        raw="a b c d", final="a b c d", steps=[], corrections=corrections,
        intended_text="", user_comment="", reporter_role="user",
        reporter_host="127.0.0.1", baseline_corrections=baseline,
    )


def test_resubmit_drops_removed_and_reranged_chips(reports_store_db):
    rs = reports_store_db
    rid, _ = _submit(rs, [_B, _D])
    # The user removed the idx-3 chip and shift-extended idx 1 to 1..2.
    rid2, updated = _submit(rs, [_BC], baseline=[_B, _D])
    assert updated is True and rid2 == rid
    assert rs.get_report(rid)["corrections"] == [_BC]


def test_resubmit_keeps_concurrent_chip_outside_the_baseline(reports_store_db):
    rs = reports_store_db
    rid, _ = _submit(rs, [_B])
    _submit(rs, [_B, _D], baseline=[_B])        # another tab added idx 3
    _submit(rs, [], baseline=[_B])              # this tab removed idx 1
    assert rs.get_report(rid)["corrections"] == [_D]


def test_submit_route_forwards_baseline(client, make_user_key):
    _uid, key = make_user_key("alice", is_admin=True)
    h = bearer(key)
    first = client.post(_SUBMIT, json={"request_id": "r1", "corrections": [_B, _D]},
                        headers=h)
    assert first.status_code == 200
    r = client.post(_SUBMIT, json={
        "request_id": "r1", "corrections": [_BC], "baseline_corrections": [_B, _D],
    }, headers=h)
    assert r.status_code == 200 and r.json()["was_updated"] is True
    from faster_whisper_backend.reports import store as reports_store
    assert reports_store.get_report(first.json()["id"])["corrections"] == [_BC]
