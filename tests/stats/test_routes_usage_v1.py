"""Integration tests for GET /v1/usage — the desktop app's self-scoped usage
statistics document: today + lifetime totals per kind, a per-kind daily
series, stage meters, the dictation breakdown, top apps, the rhythm calendar
and streak, and the time-saved figure.

Like /v1/recent-words and /v1/pipeline-rules it lives in the /v1 namespace with
NO host allowlist (so a remote client isn't 403'd by USER_WEBUI_ALLOWED_HOSTS),
and it is STRICTLY self-scoped: a caller only ever sees their own user_id's
numbers — even an admin (the global view is the host-gated /stats page).

Days are reckoned in the caller's `tz`; without one, server-local (pinned via
set_tz where it matters).
"""

import datetime
import zoneinfo

from tests.conftest import bearer

_KINDS = ("dictation", "file", "url", "text")
_CELL = {"sessions", "requests", "errors", "words", "audio_s", "processing_s"}


def _seed(uid, *, hour, words=0, audio_s=0.0, status="ok", kind="dictation",
          job_id=None, stages=None):
    """Insert one request into the rollups directly (the app lifespan has
    already init'd usage_store onto the temp DB). key_id is per-uid because
    usage_hourly's key is (hour, key_id, kind)."""
    from faster_whisper_backend.stats import usage_store
    usage_store.record_usage(
        key_id=f"k-{uid}", user_id=uid, audio_s=audio_s, words=words,
        status=status, hour=hour, kind=kind, job_id=job_id, stages=stages,
    )


def _hour_in(tz, days_ago=0, hh=12):
    day = datetime.datetime.now(tz).date() - datetime.timedelta(days=days_ago)
    return int(datetime.datetime(day.year, day.month, day.day, hh,
                                 tzinfo=tz).timestamp() // 3600)


# --------------------------------------------------------------------------
# Shape (open mode)
# --------------------------------------------------------------------------

def test_v1_usage_shape(client):
    body = client.get("/v1/usage").json()
    assert set(body) == {
        "username", "tz", "range", "today", "total", "series", "stages",
        "dictation", "apps", "calendar", "hours", "dom_hours", "streak",
        "time_saved_s"}
    today = (datetime.date.today() - datetime.date(1970, 1, 1)).days
    assert body["range"] == {"from": today - 29, "to": today, "days": 30,
                             "first_day": None, "source": "rollups",
                             "jobs_retention_days": 365}
    for k in ("today", "total"):
        assert set(body[k]) == {"all", *_KINDS}
        assert all(set(body[k][kind]) == _CELL for kind in body[k])
    assert body["series"] == [] and body["stages"] == [] and body["apps"] == []
    assert body["calendar"] == [] and body["hours"] == [] and body["dom_hours"] == []
    assert body["streak"] == {k: {"current": 0, "best": 0}
                              for k in ("all", *_KINDS)}
    d = body["dictation"]
    assert set(d) == {"sessions", "words", "audio_s", "wpm", "activation",
                      "delivery", "translation", "targets"}
    assert d["targets"] == []
    assert set(d["delivery"]) == {"typed", "clipboard", "none", "unreported"}
    assert set(d["translation"]) == {"translated", "kept_original", "not_asked",
                                     "aborted", "unreported"}
    assert body["time_saved_s"] == 0.0
    assert body["tz"] == "local"


def test_v1_usage_params_clamped_and_tz_echoed(client):
    body = client.get("/v1/usage", params={"days": 9999, "tz": "Europe/Zurich"}).json()
    assert body["range"]["days"] == 3650
    assert body["tz"] == "Europe/Zurich"
    body = client.get("/v1/usage", params={"days": -3, "tz": "Mars/Olympus"}).json()
    assert body["range"]["days"] == 1
    assert body["tz"] == "local"
    # An unparseable number is a caller error, like every other int query.
    assert client.get("/v1/usage", params={"days": "abc"}).status_code == 422
    assert client.get("/v1/usage", params={"from": "x"}).status_code == 422
    # from/to are inclusive caller-local days; from after to is an error.
    body = client.get("/v1/usage", params={"from": 20000, "to": 20006}).json()
    assert body["range"] == {"from": 20000, "to": 20006, "days": 7,
                             "first_day": None, "source": "rollups",
                             "jobs_retention_days": 365}
    r = client.get("/v1/usage", params={"from": 20007, "to": 20006})
    assert r.status_code == 422 and "from" in r.json()["detail"]
    # `all` with no usage is just today.
    body = client.get("/v1/usage", params={"all": 1}).json()
    assert body["range"]["days"] == 1 and body["range"]["first_day"] is None
    # `with` narrows to the per-job rows; an unknown stage is a caller error.
    body = client.get("/v1/usage", params={"with": "translating,diarizing"}).json()
    assert body["range"]["source"] == "jobs"
    r = client.get("/v1/usage", params={"with": "decoding"})
    assert r.status_code == 422 and "decoding" in r.json()["detail"]


# --------------------------------------------------------------------------
# Totals + series + kinds
# --------------------------------------------------------------------------

def test_v1_usage_per_kind_totals_series_and_today(client, make_user_key):
    make_user_key("root", is_admin=True)  # flip lockdown
    uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    zh = zoneinfo.ZoneInfo("Europe/Zurich")
    # Two utterances of one dictation session today, a file yesterday, a
    # failed url three days ago, a text translation eight days ago (outside a
    # 7-day window, inside lifetime).
    _seed(uid, hour=_hour_in(zh, 0, 9), words=30, audio_s=20.0, job_id="a" * 32)
    _seed(uid, hour=_hour_in(zh, 0, 10), words=70, audio_s=40.0, job_id="a" * 32)
    _seed(uid, hour=_hour_in(zh, 1), words=500, audio_s=600.0, kind="file")
    _seed(uid, hour=_hour_in(zh, 3), words=0, audio_s=0.0, kind="url", status="error")
    _seed(uid, hour=_hour_in(zh, 8), words=0, audio_s=0.0, kind="text")

    body = client.get("/v1/usage", params={"days": 7, "tz": "Europe/Zurich"},
                      headers=bearer(raw)).json()
    assert body["username"] == "alice"
    today = body["today"]
    assert today["dictation"] == {"sessions": 1, "requests": 2, "errors": 0,
                                  "words": 100, "audio_s": 60.0, "processing_s": 0.0}
    assert today["all"]["words"] == 100 and today["file"]["words"] == 0
    # `total` is the window: the text translation eight days ago is outside.
    total = body["total"]
    assert total["all"]["requests"] == 4 and total["all"]["errors"] == 1
    assert total["text"]["requests"] == 0 and total["url"]["errors"] == 1
    lifetime = client.get("/v1/usage", params={"all": 1, "tz": "Europe/Zurich"},
                          headers=bearer(raw)).json()
    assert lifetime["total"]["all"]["requests"] == 5
    assert lifetime["total"]["text"]["requests"] == 1
    assert lifetime["range"]["days"] == 9 and lifetime["range"]["first_day"] == lifetime["range"]["from"]
    assert [p["day"] for p in body["series"]] == sorted(p["day"] for p in body["series"])
    assert len(body["series"]) == 3
    assert body["series"][-1]["dictation"]["sessions"] == 1
    assert body["series"][-2]["file"]["words"] == 500
    assert body["series"][0]["url"]["errors"] == 1
    assert body["series"][-1]["day"] == (datetime.datetime.now(zh).date() - datetime.date(1970, 1, 1)).days
    # Dictation-only derived figures: 100 words / 1 min speech.
    assert body["dictation"]["wpm"] == 100.0
    assert body["time_saved_s"] == 100 / 40 * 60 - 60
    assert body["streak"]["all"] == {"current": 2, "best": 2}
    assert body["streak"]["dictation"] == {"current": 1, "best": 1}
    assert [(c["all"], c["file"], c["dictation"]) for c in body["calendar"]] == [
        (500, 500, 0), (100, 0, 100)]
    assert [(h["dow"], h["hour"], h["dictation"]) for h in body["hours"]
            if h["dictation"]] == sorted(
        [(datetime.datetime.now(zh).weekday(), 9, 30),
         (datetime.datetime.now(zh).weekday(), 10, 70)])


def test_v1_usage_stages_from_batch_extras(client, make_user_key):
    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    utc = zoneinfo.ZoneInfo("UTC")
    _seed(uid, hour=_hour_in(utc), words=100, audio_s=120.0,
          kind="file", job_id="1" * 32, stages=[
              {"name": "diarizing", "secs": 5.0, "speakers": 2},
              {"name": "translating", "secs": 3.0, "targets": ["en"],
               "kept_original": 1}])
    _seed(uid, hour=_hour_in(utc), words=10, audio_s=5.0, kind="url", job_id="2" * 32)
    stages = {s["stage"]: s for s in
              client.get("/v1/usage", params={"tz": "UTC"},
                         headers=bearer(raw)).json()["stages"]}
    assert stages["diarizing"]["runs"] == 1 and stages["diarizing"]["of_runs"] == 2
    assert stages["diarizing"]["speakers_avg"] == 2.0
    assert stages["translating"]["targets"] == [{"code": "en", "runs": 1}]
    assert stages["translating"]["kept_original"] == 1


def test_v1_usage_server_local_days_without_tz(client, make_user_key, set_tz):
    set_tz("Asia/Tokyo")
    make_user_key("root", is_admin=True)
    uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    tokyo = zoneinfo.ZoneInfo("Asia/Tokyo")
    _seed(uid, hour=_hour_in(tokyo, 0, 1), words=5)   # 01:00 Tokyo = yesterday UTC
    body = client.get("/v1/usage", headers=bearer(raw)).json()
    assert body["tz"] == "local"
    assert body["today"]["all"]["words"] == 5
    assert body["series"][-1]["day"] == (datetime.datetime.now(tokyo).date() - datetime.date(1970, 1, 1)).days


# --------------------------------------------------------------------------
# Scoping + auth
# --------------------------------------------------------------------------

def test_v1_usage_self_scoped_even_for_admin(client, make_user_key):
    admin_uid, admin_raw = make_user_key("root", is_admin=True)
    alice_uid, alice_raw = make_user_key("alice", pages={"quick_config": "own"})
    utc = zoneinfo.ZoneInfo("UTC")
    _seed(alice_uid, hour=_hour_in(utc), words=1000, kind="file")
    _seed(admin_uid, hour=_hour_in(utc), words=1, kind="file")
    # The seed sits at noon UTC today; ask in UTC too, or a runner whose local
    # clock is behind UTC (the Windows CI box around midnight UTC) files it under
    # tomorrow and the window misses it.
    tz = {"tz": "UTC"}
    assert client.get("/v1/usage", params=tz, headers=bearer(admin_raw)).json()["total"]["all"]["words"] == 1
    assert client.get("/v1/usage", params=tz, headers=bearer(alice_raw)).json()["total"]["all"]["words"] == 1000


def test_v1_usage_requires_bearer_when_locked_down(client, make_user_key):
    make_user_key("root", is_admin=True)
    assert client.get("/v1/usage").status_code == 401


def test_v1_usage_403_without_quick_config_page(client, make_user_key):
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("nopage", pages={"quick_config": "none"})
    assert client.get("/v1/usage", headers=bearer(raw)).status_code == 403


def test_v1_usage_not_host_gated(app_module, make_user_key):
    """A remote desktop client (non-loopback) must reach it with a bearer."""
    from starlette.testclient import TestClient
    with TestClient(app_module.app, client=("203.0.113.9", 4242)) as remote:
        make_user_key("root", is_admin=True)
        _uid, raw = make_user_key("alice", pages={"quick_config": "own"})
        assert remote.get("/v1/usage").status_code == 401
        assert remote.get("/v1/usage", headers=bearer(raw)).status_code == 200


def test_v1_usage_zeroed_when_store_unavailable(client, make_user_key, monkeypatch):
    make_user_key("root", is_admin=True)
    _uid, raw = make_user_key("alice", pages={"quick_config": "own"})
    from faster_whisper_backend.stats import usage_store
    monkeypatch.setattr(usage_store, "_conn", None)
    r = client.get("/v1/usage", params={"days": 5}, headers=bearer(raw))
    assert r.status_code == 200
    body = r.json()
    assert body["username"] == "alice" and body["range"]["days"] == 5
    assert body["total"]["all"]["requests"] == 0 and body["series"] == []
