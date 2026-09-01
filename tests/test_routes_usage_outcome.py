"""Integration tests for POST /v1/usage/outcome — the desktop app's per-session
dictation outcome report (activation / delivery / translation / app id).

User-tier bearer, no page gate, no host allowlist (always mounted, like the
settings sync). Strictly self-scoped and idempotent: a re-send or someone
else's job id both read `duplicate`.
"""

import pytest

from conftest import bearer

_URL = "/v1/usage/outcome"


def _outcome(job_id="a" * 32, **over):
    base = {"job_id": job_id, "activation": "hold", "delivery": "typed",
            "translation": "not_asked"}
    base.update(over)
    return base


def _dictate(uid, job_id, *, words=10, audio_s=6.0, key_id=None):
    import usage_store
    usage_store.record_usage(key_id=key_id or f"k-{uid}", user_id=uid,
                             audio_s=audio_s, words=words, status="ok",
                             kind="dictation", job_id=job_id)


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

def test_outcome_requires_bearer_when_locked_down(client, make_user_key):
    make_user_key("root", is_admin=True)
    r = client.post(_URL, json={"outcomes": [_outcome()]})
    assert r.status_code == 401


def test_outcome_open_mode_from_loopback(client):
    r = client.post(_URL, json={"outcomes": [_outcome()]})
    assert r.status_code == 200
    assert r.json() == {"results": [{"job_id": "a" * 32, "status": "accepted"}]}


def test_outcome_needs_no_page_permission(client, make_user_key):
    """Outcomes belong to the key that dictated, so a quick_config scope of
    'none' — which 403s GET /v1/usage — does not block them."""
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("nopage", pages={"quick_config": "none"})
    r = client.post(_URL, json={"outcomes": [_outcome()]}, headers=bearer(raw))
    assert r.status_code == 200


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("body", [
    {},
    {"outcomes": []},
    {"outcomes": [_outcome()], "extra": 1},
    {"outcomes": [_outcome(words=500)]},              # client-side numbers refused
    {"outcomes": [_outcome(activation="toggle")]},
    {"outcomes": [_outcome(delivery="pasted")]},
    {"outcomes": [_outcome(translation="maybe")]},
    {"outcomes": [_outcome(job_id="not hex!")]},
    {"outcomes": [_outcome(job_id="abc")]},           # < 8 chars
    {"outcomes": [_outcome(app_id="")]},
    {"outcomes": [_outcome(app_id="x" * 65)]},
    {"outcomes": [_outcome(f"{i:032x}") for i in range(101)]},
])
def test_outcome_validation(client, body):
    assert client.post(_URL, json=body).status_code == 422


def test_outcome_batch_of_100_accepted(client):
    body = {"outcomes": [_outcome(f"{i:032x}") for i in range(100)]}
    r = client.post(_URL, json=body)
    assert r.status_code == 200
    assert len(r.json()["results"]) == 100
    assert {x["status"] for x in r.json()["results"]} == {"accepted"}


# --------------------------------------------------------------------------
# Semantics
# --------------------------------------------------------------------------

def test_outcome_accepted_then_duplicate_and_visible_in_usage(client, make_user_key):
    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    _dictate(uid, "b" * 32, words=40, audio_s=30.0)
    body = {"outcomes": [_outcome("b" * 32, activation="handsfree",
                                  delivery="clipboard", translation="translated",
                                  app_id="thunderbird")]}
    first = client.post(_URL, json=body, headers=bearer(raw)).json()
    assert first == {"results": [{"job_id": "b" * 32, "status": "accepted"}]}
    second = client.post(_URL, json=body, headers=bearer(raw)).json()
    assert second == {"results": [{"job_id": "b" * 32, "status": "duplicate"}]}

    usage = client.get("/v1/usage", headers=bearer(raw)).json()
    d = usage["dictation"]
    assert d["sessions"] == 1 and d["words"] == 40
    assert d["activation"] == {"hold": 0, "handsfree": 1}
    assert d["delivery"]["clipboard"] == 1
    assert d["translation"]["translated"] == 1
    assert usage["apps"] == [{"app_id": "thunderbird", "sessions": 1, "words": 40}]


def test_outcome_results_follow_input_order(client, make_user_key):
    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice")
    _dictate(uid, "1" * 32)
    client.post(_URL, json={"outcomes": [_outcome("1" * 32)]}, headers=bearer(raw))
    r = client.post(_URL, json={"outcomes": [_outcome("2" * 32), _outcome("1" * 32)]},
                    headers=bearer(raw))
    assert [x["status"] for x in r.json()["results"]] == ["accepted", "duplicate"]
    assert [x["job_id"] for x in r.json()["results"]] == ["2" * 32, "1" * 32]


def test_outcome_is_scoped_to_the_callers_own_jobs(client, make_user_key):
    """Bob reporting Alice's job id gets `duplicate` — the same word a
    re-send gets — and Alice's job is untouched."""
    make_user_key("root", is_admin=True)
    alice, raw_a = make_user_key("alice", pages={"quick_config": "own"})
    _bob, raw_b = make_user_key("bob", pages={"quick_config": "own"})
    _dictate(alice, "c" * 32)
    r = client.post(_URL, json={"outcomes": [_outcome("c" * 32, app_id="spy")]},
                    headers=bearer(raw_b))
    assert r.json()["results"][0]["status"] == "duplicate"
    assert client.get("/v1/usage", headers=bearer(raw_b)).json()["apps"] == []
    a = client.get("/v1/usage", headers=bearer(raw_a)).json()
    assert a["dictation"]["delivery"]["typed"] == 0 and a["apps"] == []
    # Alice can still report it herself.
    r = client.post(_URL, json={"outcomes": [_outcome("c" * 32)]}, headers=bearer(raw_a))
    assert r.json()["results"][0]["status"] == "accepted"


def test_outcome_rate_limited_per_identity(client, make_user_key):
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice")
    _other, raw_other = make_user_key("bob")
    for i in range(60):
        r = client.post(_URL, json={"outcomes": [_outcome(f"{i:032x}")]},
                        headers=bearer(raw))
        assert r.status_code == 200, i
    r = client.post(_URL, json={"outcomes": [_outcome("f" * 32)]}, headers=bearer(raw))
    assert r.status_code == 429
    assert r.headers.get("Retry-After")
    assert r.json()["error"]["type"] == "rate_limit_exceeded"
    # Another identity has its own budget.
    r = client.post(_URL, json={"outcomes": [_outcome("f" * 32)]},
                    headers=bearer(raw_other))
    assert r.status_code == 200


def test_outcome_503_when_store_unavailable(client, monkeypatch):
    import usage_store
    monkeypatch.setattr(usage_store, "_conn", None)
    r = client.post(_URL, json={"outcomes": [_outcome()]})
    assert r.status_code == 503
    assert "USAGE_DB" in r.json()["detail"]
