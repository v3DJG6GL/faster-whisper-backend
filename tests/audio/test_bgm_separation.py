"""Music-separation stage on POST /v1/audio/transcriptions (soft-fail
contract). audio-separator is never imported — the module's `separate`
coroutine is monkeypatched at the boundary the handler uses."""

import os
import tempfile

import pytest

from faster_whisper_backend.audio import bgm_separation

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


def _post(client, **data):
    data.setdefault("model", "whisper-1")
    data.setdefault("response_format", "verbose_json")
    return client.post("/v1/audio/transcriptions", files=_FILE, data=data)


def _stub_separate(monkeypatch, calls=None):
    """separate() writes a real vocals tmp file — the handler swaps it into
    tmp_path and the request-level finally must unlink it."""
    made = []

    async def _fake(path, *, model_filename=None, progress_cb=None,
                    cancel_check=None):
        if calls is not None:
            calls.append(path)
        fd, out = tempfile.mkstemp(prefix="vocals-test-", suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(b"RIFFsepWAVE")
        made.append(out)
        return out

    monkeypatch.setattr(bgm_separation, "separate", _fake)
    return made


def test_separate_swaps_audio_and_cleans_up(client, app_module, monkeypatch, fake_model):
    app_module.cfg.BGM_SEPARATION_ENABLED = True
    try:
        calls = []
        made = _stub_separate(monkeypatch, calls)
        r = _post(client, separate_bgm="true")
        assert r.status_code == 200, r.text
        assert "warnings" not in r.json()
        assert len(calls) == 1                 # the ORIGINAL upload went in
        # The decode consumed the vocals file, not the original...
        assert fake_model.last_audio == made[0]
        # ...and the request-level finally unlinked the swapped-in tmp.
        assert not os.path.exists(made[0])
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False


def test_separate_transcodes_non_libsndfile_container(
        client, app_module, monkeypatch, fake_model):
    """An .m4a input is pre-transcoded to 44.1 kHz stereo WAV for the
    separator (libsndfile can't read AAC/MP4 → slow audioread fallback)."""
    from faster_whisper_backend.audio import transcode as audio_transcode

    app_module.cfg.BGM_SEPARATION_ENABLED = True
    try:
        calls = []
        made = _stub_separate(monkeypatch, calls)
        transcoded = []

        def _fake_transcode(src, dst, *, rate, layout):
            transcoded.append((src, dst, rate, layout))
            with open(dst, "wb") as f:
                f.write(b"RIFF44kWAVE")
            return 11

        monkeypatch.setattr(audio_transcode, "transcode_to_wav",
                            _fake_transcode)
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.m4a", b"\x00\x00\x00 ftypM4A ", "audio/mp4")},
            data={"model": "whisper-1", "response_format": "verbose_json",
                  "separate_bgm": "true"})
        assert r.status_code == 200, r.text
        assert len(transcoded) == 1
        _src, _dst, _rate, _layout = transcoded[0]
        assert (_rate, _layout) == (44100, "stereo")
        assert _src.endswith(".m4a") and _dst.endswith(".wav")
        # The separator got the WAV, not the original container...
        assert calls == [_dst]
        # ...and the intermediate WAV was unlinked after separation.
        assert not os.path.exists(_dst)
        assert fake_model.last_audio == made[0]
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False


def test_separate_transcode_failure_falls_back_to_original(
        client, app_module, monkeypatch, fake_model):
    from faster_whisper_backend.audio import transcode as audio_transcode

    app_module.cfg.BGM_SEPARATION_ENABLED = True
    try:
        calls = []
        _stub_separate(monkeypatch, calls)

        def _boom(src, dst, *, rate, layout):
            raise RuntimeError("no decoder")

        monkeypatch.setattr(audio_transcode, "transcode_to_wav", _boom)
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.m4a", b"\x00\x00\x00 ftypM4A ", "audio/mp4")},
            data={"model": "whisper-1", "separate_bgm": "true",
                  "response_format": "verbose_json"})
        assert r.status_code == 200, r.text
        # Separation still ran — on the original file (soft-fail, no warning
        # surfaced to the client for a transcode hiccup). The refusal arm
        # (a ValueError from _open_audio) is pinned in
        # tests/main/test_routes_transcription.py.
        assert "warnings" not in r.json()
        assert len(calls) == 1 and calls[0].endswith(".m4a")
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False


def test_separate_wav_input_skips_transcode(
        client, app_module, monkeypatch, fake_model):
    from faster_whisper_backend.audio import transcode as audio_transcode

    app_module.cfg.BGM_SEPARATION_ENABLED = True
    try:
        calls = []
        _stub_separate(monkeypatch, calls)

        # Record instead of only raising: the route swallows any transcode
        # exception and falls back to the original file, so a raise alone
        # could never fail this test if the wav/flac skip regressed.
        tc_calls = []

        def _never(src, dst, *, rate, layout):
            tc_calls.append(src)
            raise AssertionError("wav input must not be transcoded")

        monkeypatch.setattr(audio_transcode, "transcode_to_wav", _never)
        r = _post(client, separate_bgm="true")
        assert r.status_code == 200, r.text
        assert tc_calls == []
        assert "warnings" not in r.json()
        assert len(calls) == 1 and calls[0].endswith(".wav")
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False


def test_separate_disabled_server_soft_fails(client, app_module, monkeypatch):
    calls = []
    _stub_separate(monkeypatch, calls)
    r = _post(client, separate_bgm="true")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"]                        # transcript survives
    assert any("not enabled" in w for w in body["warnings"])
    assert calls == []


def test_separate_error_becomes_warning(client, app_module, monkeypatch, fake_model):
    app_module.cfg.BGM_SEPARATION_ENABLED = True
    try:
        async def _boom(path, **kw):
            raise bgm_separation.BgmSeparationError(
                "music-separation dependencies are not installed on this "
                "server (pip install -r requirements-bgm.txt)")
        monkeypatch.setattr(bgm_separation, "separate", _boom)
        r = _post(client, separate_bgm="1")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["text"]                    # original audio transcribed
        assert any("requirements-bgm" in w for w in body["warnings"])
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False


def test_separate_absent_inherits_config_default(client, app_module, monkeypatch):
    app_module.cfg.BGM_SEPARATION_ENABLED = True
    app_module.cfg.SEPARATE_BGM = True
    try:
        calls = []
        _stub_separate(monkeypatch, calls)
        r = _post(client)
        assert r.status_code == 200
        assert len(calls) == 1                 # SEPARATE_BGM=true applies
        r = _post(client, separate_bgm="false")
        assert r.status_code == 200
        assert len(calls) == 1                 # explicit false wins
    finally:
        app_module.cfg.SEPARATE_BGM = False
        app_module.cfg.BGM_SEPARATION_ENABLED = False


def test_locked_separate_ignores_client_param(client, app_module, make_user_key,
                                              monkeypatch):
    from tests.conftest import bearer
    app_module.cfg.BGM_SEPARATION_ENABLED = True
    try:
        calls = []
        _stub_separate(monkeypatch, calls)
        _, raw_admin = make_user_key("admin", is_admin=True)
        admin_h = bearer(raw_admin)
        r = client.post("/settings/overrides/state", headers=admin_h,
                        json={"OVERRIDE_PROFILES": {"nosep": {"locks": ["SEPARATE_BGM"]}}})
        assert r.status_code == 200, r.text
        uid, raw_alice = make_user_key("alice", is_admin=False)
        r = client.patch(f"/settings/api-keys/api/users/{uid}/permissions", headers=admin_h,
                         json={"pages": {}, "config": {"overrides": {},
                               "profiles": ["nosep"], "locks": []}})
        assert r.status_code == 200, r.text
        r = client.post(
            "/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
            data={"model": "whisper-1", "response_format": "verbose_json",
                  "separate_bgm": "true"},
        )
        assert r.status_code == 200, r.text
        assert calls == []                     # pinned off (global default)
        assert "separate_bgm" in r.json()["overrides_ignored"]
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False


def test_model_filename_appends_onnx(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_UVR_MODEL",
                        "UVR-MDX-NET-Inst_HQ_4")
    assert bgm_separation._model_filename() == "UVR-MDX-NET-Inst_HQ_4.onnx"
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_UVR_MODEL",
                        "model_bs_roformer.ckpt")
    assert bgm_separation._model_filename() == "model_bs_roformer.ckpt"


def test_separator_construction_error_keeps_raw_text_out_of_the_message(
        monkeypatch, tmp_path):
    """str(BgmSeparationError) lands in the client's `warnings` — the
    library's own text (an output-dir path, CUDA driver detail) must stay
    in the server log."""
    import sys
    import types

    class _Separator:
        def __init__(self, **kw):
            raise PermissionError("[Errno 13] Permission denied: '/secret/path'")

    pkg = types.ModuleType("audio_separator")
    sub = types.ModuleType("audio_separator.separator")
    sub.Separator = _Separator
    monkeypatch.setitem(sys.modules, "audio_separator", pkg)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", sub)
    monkeypatch.setattr(bgm_separation, "_shims_installed", True)
    monkeypatch.setattr(bgm_separation.cfg, "DOWNLOAD_ROOT", str(tmp_path),
                        raising=False)
    with pytest.raises(bgm_separation.BgmSeparationError) as ei:
        bgm_separation._load_blocking("Foo.onnx", "cpu")
    assert "/secret/path" not in str(ei.value)
    assert "Foo.onnx" in str(ei.value)


# --- actual-device bookkeeping -----------------------------------------------

def test_free_locked_clears_the_session_device(monkeypatch):
    """actual_device() must not keep reporting an evicted model's session —
    unless a live separator still owns one (an orphan draining after a
    same-model reload must not wipe the live session's placement)."""
    monkeypatch.setattr(bgm_separation, "_separator", None)
    monkeypatch.setattr(bgm_separation, "_session_device", "cuda")
    bgm_separation._free_locked("m")
    assert bgm_separation.actual_device() is None

    bgm_separation._session_device = "cuda"
    monkeypatch.setattr(bgm_separation, "_separator", object())
    bgm_separation._free_locked("m")
    assert bgm_separation.actual_device() == "cuda"


def test_register_records_the_actual_session_device(monkeypatch):
    """ORT can silently fall back to CPU at session creation — the stats/
    ledger row must carry the session's real placement, not the request."""
    import asyncio
    from faster_whisper_backend.runtime import model_registry

    cfg = bgm_separation.cfg
    monkeypatch.setattr(cfg, "BGM_SEPARATION_UVR_MODEL", "Foo", raising=False)
    monkeypatch.setattr(cfg, "BGM_SEPARATION_DEVICE", "cuda", raising=False)
    monkeypatch.setattr(bgm_separation, "_session_device", None)

    def _load(model, device):
        assert device == "cuda"
        bgm_separation._session_device = "cpu"   # the silent fallback
        return object()
    monkeypatch.setattr(bgm_separation, "_load_blocking", _load)

    asyncio.run(bgm_separation._get_separator("Foo"))
    assert model_registry._loaded_models["uvr:Foo.onnx"]["device"] == "cpu"


# --- progress weighting ------------------------------------------------------

def test_pass_fraction_weights_model_pass_heavier():
    assert bgm_separation._pass_fraction(1, 0.0) == 0.0
    assert bgm_separation._pass_fraction(1, 1.0) == bgm_separation._PASS1_WEIGHT
    assert bgm_separation._pass_fraction(2, 0.0) == bgm_separation._PASS1_WEIGHT
    assert bgm_separation._pass_fraction(2, 1.0) == 1.0
    # Clamped against tqdm over-reporting past the total.
    assert bgm_separation._pass_fraction(1, 1.7) == bgm_separation._PASS1_WEIGHT


def test_pass_fraction_single_pass_owns_full_span(monkeypatch):
    monkeypatch.setattr(bgm_separation, "_single_pass", True)
    assert bgm_separation._pass_fraction(1, 0.5) == 0.5
    assert bgm_separation._pass_fraction(1, 1.0) == 1.0


def test_pass_fraction_reads_the_running_jobs_weighting(monkeypatch):
    """A concurrent load of a DIFFERENT model flips the module global while
    a job is mid-run — the job's bar must keep the weighting of the
    separator it leased (seeded per run in the thread-local), or it moves
    backwards."""
    monkeypatch.setattr(bgm_separation, "_single_pass", False)
    tls = bgm_separation._progress_tls
    tls.single_pass = True
    try:
        assert bgm_separation._pass_fraction(1, 0.9) == 0.9
        tls.single_pass = False
        assert bgm_separation._pass_fraction(1, 1.0) == \
            bgm_separation._PASS1_WEIGHT
    finally:
        tls.single_pass = None
    # No running job → the module global decides, as before.
    monkeypatch.setattr(bgm_separation, "_single_pass", True)
    assert bgm_separation._pass_fraction(1, 0.9) == 0.9


# --- lease key is resolved once ---------------------------------------------

def test_separate_releases_the_key_it_leased_after_a_config_edit(monkeypatch):
    """An admin edit of BGM_SEPARATION_UVR_MODEL while a job is loading or
    running must not desynchronise the lease and release keys — the release
    would then decrement a bucket that was never leased and the real lease
    would pin the model forever."""
    import asyncio
    cfg = bgm_separation.cfg
    monkeypatch.setattr(cfg, "BGM_SEPARATION_UVR_MODEL", "Foo", raising=False)
    monkeypatch.setattr(cfg, "BGM_SEPARATION_DEVICE", "cpu", raising=False)
    monkeypatch.setattr(bgm_separation, "_load_blocking",
                        lambda model, device: object())

    async def _fake_separate_with(sep, path, *, progress_cb, cancel_check):
        # The load has happened and the lease is held; the admin edits now.
        cfg.BGM_SEPARATION_UVR_MODEL = "Bar"
        return "out.wav"
    monkeypatch.setattr(bgm_separation, "_separate_with", _fake_separate_with)

    assert asyncio.run(bgm_separation.separate("in.wav")) == "out.wav"
    assert bgm_separation._leases == {}
    assert bgm_separation._orphans == {}


# --- the tail restamp is owned by the release -------------------------------

def test_separate_does_not_touch_a_model_it_never_leased(monkeypatch):
    """_separate_with used to restamp whatever _separator_key named when the
    run ended — a concurrent load of another model mid-run then got its idle
    clock / stats 'last used' refreshed by a job that never touched it."""
    import asyncio
    from faster_whisper_backend.runtime import model_registry
    cfg = bgm_separation.cfg
    monkeypatch.setattr(cfg, "BGM_SEPARATION_UVR_MODEL", "Foo", raising=False)
    monkeypatch.setattr(cfg, "BGM_SEPARATION_DEVICE", "cpu", raising=False)
    monkeypatch.setattr(bgm_separation, "_separator", None)
    monkeypatch.setattr(bgm_separation, "_separator_key", None)
    monkeypatch.setattr(bgm_separation, "_leases", {})
    made = []

    class _Sep:
        def separate(self, path, custom_output_names=None):
            # A concurrent _get_separator("other") re-keyed the singleton.
            bgm_separation._separator_key = ("other.onnx", "cpu")
            fd, out = tempfile.mkstemp(prefix="vocals-test-", suffix=".wav")
            os.close(fd)
            made.append(out)
            return [out]

    monkeypatch.setattr(bgm_separation, "_load_blocking",
                        lambda model, device: _Sep())
    touched = []
    monkeypatch.setattr(model_registry, "touch_loaded_model",
                        lambda name: touched.append(name))
    try:
        assert asyncio.run(bgm_separation.separate("in.wav")) == made[0]
    finally:
        for f in made:
            if os.path.exists(f):
                os.unlink(f)
    assert "uvr:other.onnx" not in touched
    assert bgm_separation._leases == {}


# --- cancel is re-checked once the inference slot is ours -------------------

def test_cancel_after_the_mutex_wait_skips_the_separation(monkeypatch):
    import asyncio
    answers = iter([False])          # queued: not yet; acquired: cancelled

    def _cancel():
        return next(answers, True)

    class _Sep:
        calls = []

        def separate(self, path, custom_output_names=None):
            self.calls.append(path)
            return ["never.wav"]

    sep = _Sep()
    # _run executes in the executor thread: swap the threading.local for a
    # plain namespace so its post-condition is observable from this thread.
    import types
    monkeypatch.setattr(bgm_separation, "_progress_tls", types.SimpleNamespace())
    with pytest.raises(bgm_separation.BgmCancelled):
        asyncio.run(bgm_separation._separate_with(
            sep, "in.wav", progress_cb=None, cancel_check=_cancel))
    assert sep.calls == []
    assert getattr(bgm_separation._progress_tls, "cancel", None) is None


# --- the release survives a cancellation delivered in the finally -----------

def test_release_returns_the_lease_immediately_while_a_load_holds_the_lock(
        monkeypatch):
    """_lock is held for a whole executor-long load; a job whose separation
    is DONE must not sit in its finally until another user's model finishes
    downloading — the release is lock-free and lands before the lock is
    ever given up."""
    import asyncio
    cfg = bgm_separation.cfg
    monkeypatch.setattr(cfg, "BGM_SEPARATION_UVR_MODEL", "Foo", raising=False)
    monkeypatch.setattr(cfg, "BGM_SEPARATION_DEVICE", "cpu", raising=False)
    sep = object()
    monkeypatch.setattr(bgm_separation, "_separator", sep)
    monkeypatch.setattr(bgm_separation, "_separator_key", ("Foo.onnx", "cpu"))
    monkeypatch.setattr(bgm_separation, "_leases", {})
    monkeypatch.setattr(bgm_separation, "_orphans", {})

    async def _fake_separate_with(sep, path, *, progress_cb, cancel_check):
        return "out.wav"
    monkeypatch.setattr(bgm_separation, "_separate_with", _fake_separate_with)

    async def _main():
        lock = asyncio.Lock()
        monkeypatch.setattr(bgm_separation, "_lock", lock)
        async with lock:                       # "a load is in flight"
            task = asyncio.create_task(bgm_separation.separate("in.wav"))
            for _ in range(5):
                await asyncio.sleep(0)
            # The job finished and released WITHOUT waiting on the lock.
            assert task.done() and task.result() == "out.wav"
            assert bgm_separation._leases == {}

    asyncio.run(_main())


# --- a same-name orphan keeps the idle drop from freeing ---------------------

def test_idle_drop_defers_the_free_while_a_same_name_orphan_drains(monkeypatch):
    """Force-drop orphans S1, a later job loads S2 under the same key and
    releases it, the idle evictor drops S2 — that must not unregister the
    stats row / wipe the session device while the S1 job still runs."""
    from faster_whisper_backend.runtime import model_registry
    monkeypatch.setattr(bgm_separation, "_separator", object())
    monkeypatch.setattr(bgm_separation, "_separator_key", ("m.onnx", "cpu"))
    monkeypatch.setattr(bgm_separation, "_leases", {})
    monkeypatch.setattr(bgm_separation, "_orphans", {"m.onnx": 1})
    monkeypatch.setattr(bgm_separation, "_session_device", "cuda")
    unregistered = []
    monkeypatch.setattr(model_registry, "unregister_loaded_model",
                        lambda name: unregistered.append(name))

    assert bgm_separation._drop_locked(force=False) is True
    assert bgm_separation._separator is None
    assert unregistered == []
    assert bgm_separation._orphans == {"m.onnx": 1}
    assert bgm_separation.actual_device() == "cuda"


# --- a load cancelled or failed mid-way --------------------------------------

def _foo_model(monkeypatch, device="cpu"):
    cfg = bgm_separation.cfg
    monkeypatch.setattr(cfg, "BGM_SEPARATION_UVR_MODEL", "Foo", raising=False)
    monkeypatch.setattr(cfg, "BGM_SEPARATION_DEVICE", device, raising=False)
    monkeypatch.setattr(bgm_separation, "_separator", None)
    monkeypatch.setattr(bgm_separation, "_separator_key", None)
    monkeypatch.setattr(bgm_separation, "_leases", {})
    monkeypatch.setattr(bgm_separation, "_orphans", {})
    monkeypatch.setattr(bgm_separation, "_deferred_free", set())


def test_cancel_during_register_keeps_the_separator_its_row_describes(monkeypatch):
    """The register thread writes the stats row even when the awaiting
    request is cancelled; the separator must be cached by then, or the row
    is a phantom _drop_locked (which returns early on no separator) never
    unregisters."""
    import asyncio
    import threading
    from faster_whisper_backend.runtime import model_registry
    _foo_model(monkeypatch)
    sep = object()
    monkeypatch.setattr(bgm_separation, "_load_blocking", lambda m, d: sep)
    entered, gate = threading.Event(), threading.Event()
    real_register = model_registry.register_loaded_model

    def _slow_register(*a, **kw):
        entered.set()
        gate.wait(5)
        return real_register(*a, **kw)
    monkeypatch.setattr(model_registry, "register_loaded_model", _slow_register)

    async def _main():
        task = asyncio.create_task(bgm_separation._get_separator("Foo", lease=True))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate.set()
    asyncio.run(_main())   # joins the register thread on exit
    assert bgm_separation._separator is sep
    assert bgm_separation._leases == {}   # the cancelled caller never releases
    assert "uvr:Foo.onnx" in model_registry._loaded_models
    assert bgm_separation._drop_locked() is True
    assert "uvr:Foo.onnx" not in model_registry._loaded_models


def test_failed_reload_redoes_the_free_an_orphan_deferred_to_it(monkeypatch):
    """A device change orphans the leased separator and reloads the same
    model; the orphan's last release lands mid-load and skips its unregister
    (the reload would re-register). When that load then fails, the dead
    separator's row and session device must go anyway."""
    import asyncio
    import threading
    from faster_whisper_backend.runtime import model_registry
    _foo_model(monkeypatch, device="cuda")
    old = object()
    monkeypatch.setattr(bgm_separation, "_separator", old)
    monkeypatch.setattr(bgm_separation, "_separator_key", ("Foo.onnx", "cpu"))
    monkeypatch.setattr(bgm_separation, "_leases", {"Foo.onnx": 1})
    monkeypatch.setattr(bgm_separation, "_session_device", "cpu")
    model_registry.register_loaded_model("uvr:Foo.onnx", None, "cpu", "onnx", 1.0)
    entered, gate = threading.Event(), threading.Event()

    def _load(model, device):
        entered.set()
        gate.wait(5)
        raise bgm_separation.BgmSeparationError("download failed")
    monkeypatch.setattr(bgm_separation, "_load_blocking", _load)

    async def _main():
        task = asyncio.create_task(bgm_separation._get_separator("Foo"))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        await bgm_separation._release_separator("Foo.onnx", old)   # orphan drains
        assert "uvr:Foo.onnx" in model_registry._loaded_models    # deferred
        gate.set()
        with pytest.raises(bgm_separation.BgmSeparationError):
            await task
    asyncio.run(_main())
    assert "uvr:Foo.onnx" not in model_registry._loaded_models
    assert bgm_separation.actual_device() is None
    assert bgm_separation._deferred_free == set()


# --- a task cancelled mid-separation (shutdown, framework cancel) ------------

def test_task_cancel_mid_separation_defers_the_drop_and_unlinks_the_output(
        monkeypatch):
    """The awaiting task is cancelled while the executor thread is still
    inside sep.separate: the lease is gone, but a non-forced drop must not
    free a session the zombie still uses, and the vocals WAV it writes has
    no owner left — the zombie side unlinks it."""
    import asyncio
    import threading
    import time
    import types
    monkeypatch.setattr(bgm_separation, "_progress_tls", types.SimpleNamespace())
    monkeypatch.setattr(bgm_separation, "_leases", {})
    monkeypatch.setattr(bgm_separation, "_orphans", {})
    entered, gate = threading.Event(), threading.Event()
    written: "list[str]" = []

    class _Sep:
        def separate(self, path, custom_output_names=None):
            entered.set()
            gate.wait(5)
            fd, out = tempfile.mkstemp(prefix="vocals-test-", suffix=".wav")
            os.close(fd)
            written.append(out)
            return [out]

    sep = _Sep()
    monkeypatch.setattr(bgm_separation, "_separator", sep)
    monkeypatch.setattr(bgm_separation, "_separator_key", ("Foo.onnx", "cpu"))
    monkeypatch.setattr(bgm_separation.cfg, "BGM_SEPARATION_UVR_MODEL", "Foo",
                        raising=False)
    monkeypatch.setattr(bgm_separation.cfg, "BGM_SEPARATION_DEVICE", "cpu",
                        raising=False)

    async def _main():
        task = asyncio.create_task(bgm_separation.separate("in.wav"))
        while not entered.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert bgm_separation._leases == {}
        assert await bgm_separation.drop_separator(force=False) is False
        assert bgm_separation._separator is sep
        gate.set()

    asyncio.run(_main())
    deadline = time.monotonic() + 5
    while (not written or os.path.exists(written[0])) \
            and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(written) == 1
    assert not os.path.exists(written[0])


def test_preload_declines_under_the_lock_when_a_job_holds_another_separator(
        monkeypatch):
    """Same re-check as diarization: a speculative load that waited on _lock
    must not orphan a separator a job leased meanwhile."""
    import asyncio
    _foo_model(monkeypatch)
    held = object()
    monkeypatch.setattr(bgm_separation, "_separator", held)
    monkeypatch.setattr(bgm_separation, "_separator_key", ("Bar.onnx", "cpu"))
    monkeypatch.setattr(bgm_separation, "_leases", {"Bar.onnx": 1})
    loads: list = []
    monkeypatch.setattr(bgm_separation, "_load_blocking",
                        lambda m, d: loads.append(m) or object())

    assert asyncio.run(bgm_separation.load_unleased("Foo")) is False
    assert loads == []
    assert bgm_separation._separator is held
    assert bgm_separation._leases == {"Bar.onnx": 1}
    assert bgm_separation._orphans == {}


def test_a_zombie_separation_makes_the_singleton_busy(monkeypatch):
    """Same zombie as diarization: a cancelled job's thread still inside
    sep.separate holds _separate_mutex after its lease is gone, so busy()
    must refuse the speculative load that would force-drop it."""
    import asyncio
    _foo_model(monkeypatch)
    held = object()
    monkeypatch.setattr(bgm_separation, "_separator", held)
    monkeypatch.setattr(bgm_separation, "_separator_key", ("Bar.onnx", "cpu"))
    monkeypatch.setattr(bgm_separation, "_leases", {})
    monkeypatch.setattr(bgm_separation, "_orphans", {})
    loads: list = []
    monkeypatch.setattr(bgm_separation, "_load_blocking",
                        lambda m, d: loads.append(m) or object())
    assert bgm_separation.busy("Foo.onnx") is False
    bgm_separation._separate_mutex.acquire()
    try:
        assert bgm_separation.busy("Foo.onnx") is True
        assert asyncio.run(bgm_separation.load_unleased("Foo")) is False
    finally:
        bgm_separation._separate_mutex.release()
    assert loads == []
    assert bgm_separation._separator is held
