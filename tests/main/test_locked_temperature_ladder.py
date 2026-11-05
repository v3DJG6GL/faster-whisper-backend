"""A LOCKED TEMPERATURE binds the OpenAI-compat `temperature` Form field for
EVERY resolved ladder value that yields no ladder — not only a blank one —
and the diagnostics report the temperature the decoder was actually given."""

import pytest

from tests.conftest import FakeModel
from tests.conftest import bearer
from tests.main.test_transcription_overrides import _bind_profile_user
from faster_whisper_backend.transcription import models as tx_models
from tests._transcription_helpers import post_verbose_ok, serve_model
OV = "/settings/overrides"


def _lock_temperature_for_alice(client, make_user_key, ladder):
    return _bind_profile_user(client, make_user_key, "t0", TEMPERATURE=ladder,
                              locks=["TEMPERATURE"])


def test_temperature_ladder_helper(app_module):
    assert tx_models._temperature_ladder("0.0, 0.2,0.4") == (0.0, 0.2, 0.4)
    assert tx_models._temperature_ladder("") == ()
    assert tx_models._temperature_ladder(None) == ()
    assert tx_models._temperature_ladder(",") == ()
    assert tx_models._temperature_ladder(" , ") == ()
    assert tx_models._temperature_ladder("abc") == ()


def test_profile_refuses_an_unparseable_temperature(client, root_key):
    # A profile gets AdminConfig's TEMPERATURE rule, so a ladderless "abc"
    # never reaches the lock path below.
    r = client.post(f"{OV}/state", headers=bearer(root_key),
                    json={"OVERRIDE_PROFILES": {"t0": {"TEMPERATURE": "abc"}}})
    assert r.status_code == 422 and "comma-separated floats" in r.text


@pytest.mark.parametrize("ladder", [","])
def test_locked_ladderless_temperature_ignores_client_form_field(
        client, make_user_key, fake_model, ladder):
    # "," (every token blank) is a non-blank string that produces NO ladder
    # in the assembler; the lock must still bind the Form field.
    raw_alice = _lock_temperature_for_alice(client, make_user_key, ladder)
    body = post_verbose_ok(client, bearer(raw_alice), temperature="0.97")
    assert fake_model.last_kwargs["temperature"] == 0.0
    assert "temperature" in body["overrides_ignored"]


def test_segment_temperature_falls_back_to_the_decoded_value(
        client, make_user_key, monkeypatch, app_module):
    class _BareSegment:
        # An older/stubbed faster-whisper segment with no `.temperature`.
        def __init__(self):
            self.text = "hallo welt"
            self.start, self.end = 0.0, 1.0
            self.avg_logprob = -0.1
            self.no_speech_prob = 0.01
            self.words = []

    serve_model(monkeypatch, FakeModel(segments=[_BareSegment()]))
    raw_alice = _lock_temperature_for_alice(client, make_user_key, "")
    body = post_verbose_ok(client, bearer(raw_alice), temperature="0.9")
    assert body["segments"][0]["temperature"] == 0.0


def test_client_ladder_reaches_the_decoder(client, make_user_key, fake_model):
    _, raw_alice = make_user_key("alice")
    post_verbose_ok(client, bearer(raw_alice), decode_overrides='{"temperature": "0,0.4,0.8"}')
    assert fake_model.last_kwargs["temperature"] == (0.0, 0.4, 0.8)


def test_locked_temperature_drops_a_client_ladder(client, make_user_key, fake_model):
    raw_alice = _lock_temperature_for_alice(client, make_user_key, "0.0,0.2")
    body = post_verbose_ok(client, bearer(raw_alice), decode_overrides='{"temperature": "0.5,1.0"}')
    assert fake_model.last_kwargs["temperature"] == (0.0, 0.2)
    assert "temperature" in body["overrides_ignored"]
