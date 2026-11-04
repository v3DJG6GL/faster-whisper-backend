"""Helpers shared by the stage tests that drive POST /v1/audio/transcriptions
(the stub upload, a plain / verbose_json post, a profile-bound non-admin key,
the record_transcription spy, a fixed model loader, a constant-tone WAV, the
receipt's guard line, a gated model_registry write).
Import as ``from tests._transcription_helpers import ...``."""

import io

import numpy as np

from faster_whisper_backend.transcription import models as tx_models
from tests.conftest import bearer, write_wav

WAV_UPLOAD = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


def post_transcription(client, headers=None, **data):
    """POST the stub upload with form fields `data` (model defaults to whisper-1)."""
    data.setdefault("model", "whisper-1")
    return client.post("/v1/audio/transcriptions", files=WAV_UPLOAD, data=data,
                       headers=headers)


def post_verbose(client, headers=None, **data):
    data.setdefault("response_format", "verbose_json")
    return post_transcription(client, headers, **data)


def bind_profile(client, make_user_key, name, fields, user="alice"):
    """Create admin + non-admin `user` bound to the one OVERRIDE_PROFILES
    entry `name` = `fields`; return (admin_headers, user_raw_key)."""
    _, raw_admin = make_user_key("admin", is_admin=True)
    admin_h = bearer(raw_admin)
    r = client.post("/settings/overrides/state", headers=admin_h,
                    json={"OVERRIDE_PROFILES": {name: fields}})
    assert r.status_code == 200, r.text
    uid, raw = make_user_key(user, is_admin=False)
    r = client.patch(
        f"/settings/api-keys/api/users/{uid}/permissions", headers=admin_h,
        json={"pages": {}, "config": {"overrides": {},
                                      "profiles": [name], "locks": []}})
    assert r.status_code == 200, r.text
    return admin_h, raw


def record_runs(app_module, monkeypatch):
    """Spy on metrics.record_transcription (passing through to the real one);
    return the list the spy appends each call's kwargs to."""
    recorded = []
    real = app_module.metrics.record_transcription

    def _spy(**kw):
        recorded.append(kw)
        return real(**kw)
    monkeypatch.setattr(app_module.metrics, "record_transcription", _spy)
    return recorded


def serve_model(monkeypatch, model):
    """Every model load returns `model`."""
    async def _loader(name, *, lease=False):
        return model
    monkeypatch.setattr(tx_models, "_get_or_load_model", _loader)


def wav_bytes(seconds=1.0, sr=16000):
    """A decodable mono PCM WAV with constant nonzero amplitude, so an
    injected zero-pad is distinguishable from the payload."""
    buf = io.BytesIO()
    write_wav(buf, np.full(int(seconds * sr), 8000, dtype="<i2").tobytes(), rate=sr)
    return buf.getvalue()


def guard_line(caplog, knob):
    """The receipt's first "Post-decode guards" line naming `knob`,
    right-stripped (StopIteration when no such line was logged)."""
    block = "\n".join(r.getMessage() for r in caplog.records
                      if "Post-decode guards" in r.getMessage())
    return next(l for l in block.splitlines() if knob in l).rstrip()


def blocked_register(monkeypatch):
    """Hold model_registry.register_loaded_model on a gate; return
    (entered, gate): `entered` is set once a register call is waiting, and
    setting `gate` (or 5 s) lets it through to the real write."""
    import threading
    from faster_whisper_backend.runtime import model_registry
    entered, gate = threading.Event(), threading.Event()
    real_register = model_registry.register_loaded_model

    def _slow(*a, **kw):
        entered.set()
        gate.wait(5)
        return real_register(*a, **kw)
    monkeypatch.setattr(model_registry, "register_loaded_model", _slow)
    return entered, gate
