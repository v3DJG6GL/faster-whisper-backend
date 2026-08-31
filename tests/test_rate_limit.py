"""Unit tests for the shared limiters (no app import — rate_limit is
deliberately importable on its own, stdlib + fastapi + config only)."""

import pytest

import config as cfg
import rate_limit


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(cfg, "URL_PREVIEW_RATE_PER_MIN", 3, raising=False)
    w = rate_limit.FixedWindow(
        config_field="URL_PREVIEW_RATE_PER_MIN", window_s=60.0,
        default_max=3, message="slow down — {limit} per minute",
    )
    yield w
    rate_limit._ALL.remove(w)


@pytest.fixture
def gauge(monkeypatch):
    monkeypatch.setattr(cfg, "TRANSLATE_MAX_INFLIGHT_PER_USER", 2,
                        raising=False)
    g = rate_limit.InFlight(
        config_field="TRANSLATE_MAX_INFLIGHT_PER_USER", default_max=2,
        message="{limit} at a time",
    )
    yield g
    rate_limit._ALL.remove(g)


# ---------------------------------------------------------------------------
# FixedWindow
# ---------------------------------------------------------------------------

def test_hit_counts_then_raises_the_full_envelope(window):
    for _ in range(3):
        window.hit("alice")

    with pytest.raises(rate_limit.RateLimited) as exc:
        window.hit("alice")

    body = exc.value.body()
    assert body["error"]["type"] == "rate_limit_exceeded"
    assert body["error"]["param"] == "URL_PREVIEW_RATE_PER_MIN"
    assert 1 <= body["error"]["retry_after"] <= 60
    # The `detail` sibling the in-repo toast handlers read must mirror the
    # message exactly — a divergence would show two different sentences.
    assert body["detail"] == body["error"]["message"] == "slow down — 3 per minute"
    assert exc.value.status_code == 429
    assert exc.value.headers["Retry-After"] == str(body["error"]["retry_after"])


def test_limit_zero_is_unlimited(window, gauge, monkeypatch):
    monkeypatch.setattr(cfg, "URL_PREVIEW_RATE_PER_MIN", 0, raising=False)
    monkeypatch.setattr(cfg, "TRANSLATE_MAX_INFLIGHT_PER_USER", 0,
                        raising=False)
    for _ in range(50):
        window.hit("alice")
        gauge.acquire("alice")
    assert window._state == {}
    assert gauge._counts == {}


def test_window_roll_restarts_the_counter(window):
    for _ in range(3):
        window.hit("alice")
    # Backdate the window past its length instead of sleeping 60 s.
    n, start = window._state["alice"]
    window._state["alice"] = (n, start - 61.0)

    window.hit("alice")
    assert window._state["alice"][0] == 1


def test_two_keys_do_not_share_a_bucket(window):
    for _ in range(3):
        window.hit("alice")
    with pytest.raises(rate_limit.RateLimited):
        window.hit("alice")
    window.hit("bob")  # must not raise


def test_guard_penalize_reset(window):
    window.guard("alice")  # empty bucket: fine
    for _ in range(3):
        assert window.penalize("alice") is (window._state["alice"][0] == 3)
    # penalize never raises, even past the limit...
    window.penalize("alice")
    # ...but guard refuses once the bucket is spent, without counting.
    with pytest.raises(rate_limit.RateLimited):
        window.guard("alice")
    assert window._state["alice"][0] == 4

    window.reset("alice")
    assert "alice" not in window._state
    window.guard("alice")


def test_prune_caps_the_dict_and_keeps_the_live_key(window, monkeypatch):
    monkeypatch.setattr(cfg, "URL_PREVIEW_RATE_PER_MIN", 100, raising=False)
    for i in range(rate_limit._MAX_KEYS + 200):
        window.hit(f"k{i}")
    assert len(window._state) <= rate_limit._MAX_KEYS
    # The key written by the call that triggered the prune must survive it.
    assert f"k{rate_limit._MAX_KEYS + 199}" in window._state


def test_reset_all_clears_registered_limiters(window, gauge):
    window.hit("alice")
    gauge.acquire("alice")
    rate_limit.reset_all()
    assert window._state == {}
    assert gauge._counts == {}


# ---------------------------------------------------------------------------
# InFlight
# ---------------------------------------------------------------------------

def test_inflight_acquire_release_balance(gauge):
    gauge.acquire("alice")
    gauge.acquire("alice")
    assert gauge.count("alice") == 2

    with pytest.raises(rate_limit.RateLimited) as exc:
        gauge.acquire("alice")
    assert exc.value.body()["error"]["param"] == "TRANSLATE_MAX_INFLIGHT_PER_USER"
    assert exc.value.retry_after == rate_limit.InFlight.RETRY_AFTER_S

    gauge.release("alice")
    gauge.acquire("alice")  # slot freed
    gauge.release("alice")
    gauge.release("alice")
    assert gauge._counts == {}


def test_inflight_release_pops_at_zero_and_never_goes_negative(gauge):
    gauge.acquire("alice")
    gauge.release("alice")
    gauge.release("alice")
    gauge.release("alice")
    assert gauge._counts == {}
    assert gauge.count("alice") == 0
    # A stray release must not have bought a free slot.
    gauge.acquire("alice")
    gauge.acquire("alice")
    with pytest.raises(rate_limit.RateLimited):
        gauge.acquire("alice")


def test_inflight_two_keys_are_independent(gauge):
    gauge.acquire("alice")
    gauge.acquire("alice")
    gauge.acquire("bob")
    assert gauge.count("bob") == 1


# ---------------------------------------------------------------------------
# identity_key
# ---------------------------------------------------------------------------

class _Client:
    def __init__(self, host):
        self.host = host


class _Req:
    """Shaped like both a Starlette Request and a WebSocket — identity_key
    touches only .client, which both carry."""

    def __init__(self, host=None):
        self.client = _Client(host) if host else None


def test_identity_key_precedence():
    req = _Req("10.0.0.9")
    assert rate_limit.identity_key({"user_id": "u1", "key_id": "k1"}, req) == "u1"
    assert rate_limit.identity_key({"key_id": "k1"}, req) == "k1"
    assert rate_limit.identity_key({}, req) == "10.0.0.9"
    assert rate_limit.identity_key(None, req) == "10.0.0.9"
    # Never "" — an empty key would merge unrelated callers into one bucket.
    assert rate_limit.identity_key({"user_id": ""}, _Req()) == "<unknown>"


def test_identity_key_accepts_a_websocket_shaped_object():
    class _WS:
        client = _Client("192.168.1.4")

    assert rate_limit.identity_key({}, _WS()) == "192.168.1.4"
