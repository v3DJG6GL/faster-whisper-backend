"""Integration tests for /reports + the /quick-config submission endpoint."""

from conftest import bearer

_SUBMIT = "/quick-config/reports/api/submit"


def _payload(request_id="req-1", comment="please fix"):
    return {"request_id": request_id, "user_comment": comment}


def test_submit_returns_ok_id(client):
    r = client.post(_SUBMIT, json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "id" in body
    assert body["was_updated"] is False


def test_submit_nothing_to_submit_400(client):
    # No corrections, no intended_text, no comment -> 400.
    r = client.post(_SUBMIT, json={"request_id": "req-x"})
    assert r.status_code == 400


def test_submit_unknown_field_422(client):
    r = client.post(_SUBMIT, json={"user_comment": "x", "bogus": 1})
    assert r.status_code == 422


def test_submit_rate_limit_429(client, app_module):
    # REPORTS_SUBMIT_RATE_PER_10MIN per identity (in open mode every request
    # is the one synthetic admin). The submit past the limit trips it.
    limit = int(app_module.cfg.REPORTS_SUBMIT_RATE_PER_10MIN)
    last = None
    for i in range(limit + 1):
        last = client.post(_SUBMIT, json=_payload(request_id=f"req-{i}"))
    assert last.status_code == 429
    assert int(last.headers["Retry-After"]) >= 1
    body = last.json()
    assert body["error"]["type"] == "rate_limit_exceeded"
    assert body["error"]["param"] == "REPORTS_SUBMIT_RATE_PER_10MIN"
    assert body["detail"] == body["error"]["message"]


def test_submit_rate_limit_is_per_user(client, app_module, make_user_key):
    """The loopback `client` fixture is OPEN MODE — one synthetic admin, one
    bucket. Real keys are what prove the limit is keyed per identity."""
    _uid_a, key_a = make_user_key("alice", is_admin=True)
    _uid_b, key_b = make_user_key("bob", is_admin=True)
    limit = int(app_module.cfg.REPORTS_SUBMIT_RATE_PER_10MIN)

    for i in range(limit):
        r = client.post(_SUBMIT, json=_payload(request_id=f"a-{i}"),
                        headers=bearer(key_a))
        assert r.status_code == 200
    assert client.post(_SUBMIT, json=_payload(request_id="a-over"),
                       headers=bearer(key_a)).status_code == 429
    assert client.post(_SUBMIT, json=_payload(request_id="b-1"),
                       headers=bearer(key_b)).status_code == 200


def test_reports_page(client):
    r = client.get("/reports")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_reports_list(client):
    client.post(_SUBMIT, json=_payload(request_id="list-1"))
    r = client.get("/reports/api/list")
    assert r.status_code == 200
    body = r.json()
    assert "reports" in body and "counts" in body


def test_list_with_legacy_nonfinite_trace_ts_raises_not_nan(client):
    """A pre-guard row with a non-finite trace_ts must make the hand-rolled
    list serializer raise (allow_nan=False, mirroring JSONResponse.render)
    instead of emitting bare `NaN` — invalid JSON the browser chokes on."""
    import pytest
    import reports_store
    client.post(_SUBMIT, json=_payload(request_id="nan-1"))
    # inf, not nan: Python's sqlite3 binds nan as NULL (NOT NULL constraint),
    # but inf round-trips through the REAL column and is equally non-finite.
    reports_store._require_conn().execute(
        "UPDATE reports SET trace_ts = ?", (float("inf"),))
    with pytest.raises(ValueError):
        client.get("/reports/api/list")


def test_patch_report_invalid_status_422(client):
    sub = client.post(_SUBMIT, json=_payload(request_id="patch-1"))
    rid = sub.json()["id"]
    # status is a Literal -> "bogus" fails pydantic validation -> 422.
    r = client.patch(f"/reports/api/{rid}", json={"status": "bogus"})
    assert r.status_code == 422


def test_patch_report_valid_status(client):
    sub = client.post(_SUBMIT, json=_payload(request_id="patch-2"))
    rid = sub.json()["id"]
    r = client.patch(f"/reports/api/{rid}", json={"status": "resolved"})
    assert r.status_code == 200
    assert r.json()["report"]["status"] == "resolved"


def test_patch_unknown_report_404(client):
    r = client.patch("/reports/api/missing", json={"status": "resolved"})
    assert r.status_code == 404


def test_delete_report(client):
    sub = client.post(_SUBMIT, json=_payload(request_id="del-1"))
    rid = sub.json()["id"]
    r = client.delete(f"/reports/api/{rid}")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_delete_unknown_report_404(client):
    r = client.delete("/reports/api/missing")
    assert r.status_code == 404


def test_clear_reports_admin(client):
    client.post(_SUBMIT, json=_payload(request_id="clear-1"))
    r = client.post("/reports/api/clear")
    assert r.status_code == 200
    assert "deleted" in r.json()


def test_export_reports_admin(client):
    r = client.get("/reports/api/export")
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")
    assert "reports" in r.json()


def test_submit_disabled_for_nonadmin_403(client, app_module, make_user_key):
    app_module.cfg.REPORTS_ALLOW_USER_SUBMIT = False
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    r = client.post(_SUBMIT, json=_payload(request_id="nope"), headers=bearer(raw))
    assert r.status_code == 403


def test_correction_field_over_cap_422(client):
    """`wrong`/`correct` are bounded at the edge (4096) purely as storage
    hygiene — the store truncates both to text_corrections.CAP_CORRECTION_FIELD
    (200) anyway. This is NOT a memory guard: the body is json.loads'd before
    pydantic sees it."""
    r = client.post(_SUBMIT, json={
        "request_id": "req-cap",
        "corrections": [{"wrong": "x" * 5000, "correct": "y", "idx": 0}],
    })
    assert r.status_code == 422


def test_correction_field_under_cap_still_accepted(client):
    """The bound is ~20x the store's functional cap, so nothing that succeeds
    today starts failing — including text far longer than the 200 chars the
    store keeps."""
    r = client.post(_SUBMIT, json={
        "request_id": "req-cap-ok",
        "corrections": [{"wrong": "x" * 4000, "correct": "y", "idx": 0}],
    })
    assert r.status_code == 200
