"""Integration tests for GET /v1/recent-words: the desktop Dictionary
key-field suggestions, built by the same build_word_suggestions() helper as the
/quick-config datalist and scoped like /quick-config/recent. Like
/v1/pipeline-rules it has NO host allowlist, so a remote client isn't 403'd by
USER_WEBUI_ALLOWED_HOSTS the way the browser /quick-config page is.
"""

from tests.conftest import bearer, locked_user


def _seed(uid, tokens, *, request_id, bigrams=None):
    """Insert a recent-transcription row with verbatim tokens/bigrams for `uid`.
    record_trace stores them as-is (the tokenizer runs upstream), so the test
    controls the suggestion pool directly."""
    from faster_whisper_backend.stats import recent_transcriptions_store
    recent_transcriptions_store.record_trace(
        request_id=request_id, model="m", raw="raw", final="final",
        tokens=tokens, bigrams=bigrams or [], user_id=uid,
    )


def test_v1_recent_words_open_mode_shape(client):
    body = client.get("/v1/recent-words").json()
    assert set(body) == {"words", "max"}
    assert isinstance(body["words"], list)
    assert body["max"] == 200  # default cap


def test_v1_recent_words_empty_when_no_traces(client):
    assert client.get("/v1/recent-words").json()["words"] == []


def test_v1_recent_words_aggregates_tokens_and_bigrams(client):
    _seed("u1", ["Komma", "Punkt"], bigrams=["Hans Peter"], request_id="r1")
    words = client.get("/v1/recent-words").json()["words"]
    assert "Komma" in words and "Punkt" in words and "Hans Peter" in words


def test_v1_recent_words_case_insensitive_dedup(client):
    _seed("u1", ["Komma"], request_id="r1")
    _seed("u1", ["komma"], request_id="r2")  # newer; same lowercased key
    words = client.get("/v1/recent-words").json()["words"]
    assert sum(1 for w in words if w.lower() == "komma") == 1


def test_v1_recent_words_cap_honored(client, app_module):
    app_module.cfg.QUICK_CONFIG_WORD_SUGGESTIONS_MAX = 2
    _seed("u1", ["a-word", "b-word", "c-word", "d-word"], request_id="r1")
    body = client.get("/v1/recent-words").json()
    assert body["max"] == 2
    assert len(body["words"]) == 2


def test_v1_recent_words_zero_disables(client, app_module):
    app_module.cfg.QUICK_CONFIG_WORD_SUGGESTIONS_MAX = 0
    _seed("u1", ["Komma"], request_id="r1")
    body = client.get("/v1/recent-words").json()
    assert body["max"] == 0
    assert body["words"] == []


def test_v1_recent_words_limit_param_clamps(client, app_module):
    app_module.cfg.QUICK_CONFIG_WORD_SUGGESTIONS_MAX = 50
    _seed("u1", ["a-word", "b-word", "c-word"], request_id="r1")
    # The client may request fewer than the cap...
    assert len(client.get("/v1/recent-words?limit=1").json()["words"]) == 1
    # ...but `max` always echoes the server cap.
    assert client.get("/v1/recent-words?limit=999").json()["max"] == 50


def test_v1_recent_words_scoped_per_user(client, make_user_key):
    make_user_key("root", is_admin=True)  # flips lockdown
    uid_a, raw_a = make_user_key("alice", pages={"quick_config": "own"})
    uid_b, raw_b = make_user_key("bob", pages={"quick_config": "own"})

    _seed(uid_a, ["alpha-word"], request_id="ra")
    _seed(uid_b, ["bravo-word"], request_id="rb")

    a_words = client.get("/v1/recent-words", headers=bearer(raw_a)).json()["words"]
    assert "alpha-word" in a_words and "bravo-word" not in a_words

    b_words = client.get("/v1/recent-words", headers=bearer(raw_b)).json()["words"]
    assert "bravo-word" in b_words and "alpha-word" not in b_words


def test_v1_recent_words_admin_sees_all(client, make_user_key):
    from faster_whisper_backend.auth import api_keys_store
    _uid_root, raw_root = make_user_key("root", is_admin=True)
    uid_a = api_keys_store.create_user("alice", is_admin=False)
    _seed(uid_a, ["alpha-word"], request_id="ra")
    _seed("someone-else", ["bravo-word"], request_id="rb")
    words = client.get("/v1/recent-words", headers=bearer(raw_root)).json()["words"]
    assert "alpha-word" in words and "bravo-word" in words


def test_v1_recent_words_requires_quick_config_page(client, make_user_key):
    _uid, raw = locked_user(make_user_key, "bob", pages={"quick_config": "none"})
    assert client.get("/v1/recent-words", headers=bearer(raw)).status_code == 403


def test_v1_recent_words_requires_auth_when_locked_down(client, make_user_key):
    make_user_key("root", is_admin=True)  # lockdown
    assert client.get("/v1/recent-words").status_code == 401  # no bearer


def test_v1_recent_words_not_host_gated(app_module, make_user_key):
    from starlette.testclient import TestClient
    # Narrow the user-WebUI allowlist to loopback, then call from a non-loopback
    # client WITH a key: the browser /quick-config/recent is host-gated (403
    # before the key is even looked at), but the /v1 client API is reachable.
    app_module.cfg.USER_WEBUI_ALLOWED_HOSTS = ["127.0.0.1/32"]
    with TestClient(app_module.app, client=("203.0.113.9", 9999)) as c:
        _uid, raw = make_user_key("root", is_admin=True)
        assert c.get("/quick-config/recent", headers=bearer(raw)).status_code == 403
        assert c.get("/v1/recent-words", headers=bearer(raw)).status_code == 200
