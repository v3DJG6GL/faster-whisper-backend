"""A LOCKED TEMPERATURE binds the OpenAI-compat `temperature` Form field for
EVERY resolved ladder value that yields no ladder — not only a blank one —
and the diagnostics report the temperature the decoder was actually given."""

import pytest

from tests.conftest import FakeModel
from tests.conftest import bearer

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}
OV = "/settings/overrides"
PERMS = "/settings/api-keys/api/users"


def _setup_profile(client, admin_h, name, **fields):
    r = client.post(f"{OV}/state", headers=admin_h,
                    json={"OVERRIDE_PROFILES": {name: fields}})
    assert r.status_code == 200, r.text


def _lock_temperature_for_alice(client, make_user_key, ladder):
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    _setup_profile(client, admin_h, "t0", TEMPERATURE=ladder,
                   locks=["TEMPERATURE"])
    uid, raw_alice = make_user_key("alice", is_admin=False)
    client.patch(f"{PERMS}/{uid}/permissions", headers=admin_h,
                 json={"pages": {}, "config": {"overrides": {},
                                               "profiles": ["t0"],
                                               "locks": []}})
    return raw_alice


def test_temperature_ladder_helper(app_module):
    assert app_module._temperature_ladder("0.0, 0.2,0.4") == (0.0, 0.2, 0.4)
    assert app_module._temperature_ladder("") == ()
    assert app_module._temperature_ladder(None) == ()
    assert app_module._temperature_ladder(",") == ()
    assert app_module._temperature_ladder(" , ") == ()
    assert app_module._temperature_ladder("abc") == ()


@pytest.mark.parametrize("ladder", [",", "abc"])
def test_locked_ladderless_temperature_ignores_client_form_field(
        client, make_user_key, fake_model, ladder):
    # "," (every token blank) and "abc" (unparseable) are non-blank strings
    # that produce NO ladder in the assembler, so the Form field used to
    # survive the lock exactly like the blank case did.
    raw_alice = _lock_temperature_for_alice(client, make_user_key, ladder)
    r = client.post(
        "/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
        data={"model": "whisper-1", "response_format": "verbose_json",
              "temperature": "0.97"},
    )
    assert r.status_code == 200, r.text
    assert fake_model.last_kwargs["temperature"] == 0.0
    assert "temperature" in r.json()["overrides_ignored"]


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

    async def _loader(name, *, lease=False):
        return FakeModel(segments=[_BareSegment()])
    monkeypatch.setattr(app_module, "_get_or_load_model", _loader)
    raw_alice = _lock_temperature_for_alice(client, make_user_key, "")
    r = client.post(
        "/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
        data={"model": "whisper-1", "response_format": "verbose_json",
              "temperature": "0.9"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["segments"][0]["temperature"] == 0.0
