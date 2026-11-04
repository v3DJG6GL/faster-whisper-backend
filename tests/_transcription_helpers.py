"""Helpers shared by the stage tests that drive POST /v1/audio/transcriptions
(the stub upload, a plain / verbose_json post, a profile-bound non-admin key,
the record_transcription spy, a fixed / failing model loader, a constant-tone WAV, the
receipt's guard line, a gated model_registry write, a cancel while that
write is held or queued).
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


def fail_model_load(monkeypatch, exc):
    """Every model load raises `exc`."""
    async def _loader(name, *, lease=False):
        raise exc
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


def cancel_with_register_queued(monkeypatch, loaded, make_coro):
    """Cancel make_coro()'s task while its register write sits queued behind
    a busy one-worker default executor; `loaded` turns truthy once the
    stubbed load ran. Returns after the queue drained."""
    import asyncio
    import concurrent.futures
    import threading

    import pytest

    from faster_whisper_backend.runtime import system_stats
    queued, gate = [], threading.Event()

    def _vram(*a):
        # The post-load sample runs on the loop right before the register
        # is submitted: occupy the only worker so the register is queued.
        if loaded and not queued:
            asyncio.get_running_loop().run_in_executor(None, gate.wait, 5)
            queued.append(True)
        return None
    monkeypatch.setattr(system_stats, "gpu_mem_used_bytes", _vram)

    async def _main():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=1))
        task = asyncio.create_task(make_coro())
        while not queued:
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.05)
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        # One worker, FIFO: once this runs, anything queued before it has.
        await loop.run_in_executor(None, lambda: None)
    try:
        asyncio.run(_main())
    finally:
        gate.set()


def cancel_while_register_held(make_coro, entered, gate):
    """Cancel make_coro()'s task once its register write is held on `gate`
    (see blocked_register; `entered` is set while it waits): the task must
    stay pending until the gate opens, then end cancelled."""
    import asyncio

    import pytest

    async def _main():
        task = asyncio.create_task(make_coro())
        while not entered.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done()     # held until the register write lands
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    try:
        asyncio.run(_main())
    finally:
        gate.set()
