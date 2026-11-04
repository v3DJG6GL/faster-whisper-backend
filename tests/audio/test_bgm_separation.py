"""Music-separation stage on POST /v1/audio/transcriptions (soft-fail
contract). audio-separator is never imported — the module's `separate`
coroutine is monkeypatched at the boundary the handler uses."""

import asyncio
import os
import sys
import tempfile
import threading
import time
import types

import pytest

from faster_whisper_backend.audio import bgm_separation
from faster_whisper_backend.audio import transcode as audio_transcode
from faster_whisper_backend.runtime import model_registry
from tests._transcription_helpers import (
    assert_cancelled_load_listed, bind_profile, blocked_register,
    cancel_while_register_held, cancel_with_register_queued, record_loads)
from tests._transcription_helpers import post_verbose_ok as _post_ok
from tests.conftest import bearer


@pytest.fixture
def bgm_on(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "BGM_SEPARATION_ENABLED", True, raising=False)


def _seed(monkeypatch, sep=None, key=None, leases=None, orphans=None):
    monkeypatch.setattr(bgm_separation, "_separator", sep)
    monkeypatch.setattr(bgm_separation, "_separator_key", key)
    monkeypatch.setattr(bgm_separation, "_leases", {} if leases is None else leases)
    monkeypatch.setattr(bgm_separation, "_orphans", {} if orphans is None else orphans)


def _foo_model(monkeypatch, device="cpu"):
    cfg = bgm_separation.cfg
    monkeypatch.setattr(cfg, "BGM_SEPARATION_UVR_MODEL", "Foo", raising=False)
    monkeypatch.setattr(cfg, "BGM_SEPARATION_DEVICE", device, raising=False)
    _seed(monkeypatch)
    monkeypatch.setattr(bgm_separation, "_deferred_free", set())
    return cfg


def _stub_separate(monkeypatch):
    """separate() writes a real vocals tmp file — the handler swaps it into
    tmp_path and the request-level finally must unlink it. Returns (the paths
    it was given, the files it wrote)."""
    calls, made = [], []

    async def _fake(path, *, model_filename=None, progress_cb=None,
                    cancel_check=None):
        calls.append(path)
        fd, out = tempfile.mkstemp(prefix="vocals-test-", suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(b"RIFFsepWAVE")
        made.append(out)
        return out

    monkeypatch.setattr(bgm_separation, "separate", _fake)
    return calls, made


def test_separate_swaps_audio_and_cleans_up(client, monkeypatch, fake_model, bgm_on):
    calls, made = _stub_separate(monkeypatch)
    assert "warnings" not in _post_ok(client, separate_bgm="true")
    assert len(calls) == 1                 # the ORIGINAL upload went in
    # The decode consumed the vocals file, not the original...
    assert fake_model.last_audio == made[0]
    # ...and the request-level finally unlinked the swapped-in tmp.
    assert not os.path.exists(made[0])


def test_separate_transcodes_non_libsndfile_container(
        client, monkeypatch, fake_model, bgm_on):
    """An .m4a input is pre-transcoded to 44.1 kHz stereo WAV for the
    separator (libsndfile can't read AAC/MP4 → slow audioread fallback)."""

    calls, made = _stub_separate(monkeypatch)
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
    assert calls == [_dst]
    assert not os.path.exists(_dst)
    assert fake_model.last_audio == made[0]


def test_separate_transcode_failure_falls_back_to_original(
        client, monkeypatch, fake_model, bgm_on):
    calls, _ = _stub_separate(monkeypatch)

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


def test_separate_wav_input_skips_transcode(
        client, monkeypatch, fake_model, bgm_on):
    calls, _ = _stub_separate(monkeypatch)

    # Record instead of only raising: the route swallows any transcode
    # exception and falls back to the original file, so a raise alone
    # could never fail this test if the wav/flac skip regressed.
    tc_calls = []

    def _never(src, dst, *, rate, layout):
        tc_calls.append(src)
        raise AssertionError("wav input must not be transcoded")

    monkeypatch.setattr(audio_transcode, "transcode_to_wav", _never)
    body = _post_ok(client, separate_bgm="true")
    assert tc_calls == []
    assert "warnings" not in body
    assert len(calls) == 1 and calls[0].endswith(".wav")


def test_separate_disabled_server_soft_fails(client, app_module, monkeypatch):
    calls, _ = _stub_separate(monkeypatch)
    body = _post_ok(client, separate_bgm="true")
    assert body["text"]                        # transcript survives
    assert any("not enabled" in w for w in body["warnings"])
    assert calls == []


def test_separate_error_becomes_warning(client, monkeypatch, fake_model, bgm_on):
    async def _boom(path, **kw):
        raise bgm_separation.BgmSeparationError(
            "music-separation dependencies are not installed on this "
            "server (pip install -r requirements-bgm.txt)")
    monkeypatch.setattr(bgm_separation, "separate", _boom)
    body = _post_ok(client, separate_bgm="1")
    assert body["text"]                    # original audio transcribed
    assert any("requirements-bgm" in w for w in body["warnings"])


def test_separate_absent_inherits_config_default(client, app_module, monkeypatch,
                                                 bgm_on):
    monkeypatch.setattr(app_module.cfg, "SEPARATE_BGM", True, raising=False)
    calls, _ = _stub_separate(monkeypatch)
    _post_ok(client)
    assert len(calls) == 1                 # SEPARATE_BGM=true applies
    _post_ok(client, separate_bgm="false")
    assert len(calls) == 1                 # explicit false wins


def test_locked_separate_ignores_client_param(client, make_user_key,
                                              monkeypatch, bgm_on):
    calls, _ = _stub_separate(monkeypatch)
    _, raw_alice = bind_profile(client, make_user_key, "nosep",
                                {"locks": ["SEPARATE_BGM"]})
    body = _post_ok(client, bearer(raw_alice), separate_bgm="true")
    assert calls == []                     # pinned off (global default)
    assert "separate_bgm" in body["overrides_ignored"]


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

    _foo_model(monkeypatch, device="cuda")
    monkeypatch.setattr(bgm_separation, "_session_device", None)

    def _load(model, device):
        assert device == "cuda"
        bgm_separation._session_device = "cpu"   # the silent fallback
        return object()
    monkeypatch.setattr(bgm_separation, "_load_blocking", _load)

    asyncio.run(bgm_separation._get_separator("Foo"))
    assert model_registry._loaded_models["uvr:Foo.onnx"]["device"] == "cpu"


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
    # No running job → the module global decides.
    monkeypatch.setattr(bgm_separation, "_single_pass", True)
    assert bgm_separation._pass_fraction(1, 0.9) == 0.9


def test_separate_releases_the_key_it_leased_after_a_config_edit(monkeypatch):
    """An admin edit of BGM_SEPARATION_UVR_MODEL while a job is loading or
    running must not desynchronise the lease and release keys — the release
    would then decrement a bucket that was never leased and the real lease
    would pin the model forever."""
    cfg = _foo_model(monkeypatch)
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


def test_separate_does_not_touch_a_model_it_never_leased(monkeypatch):
    """The tail restamp names the key this job leased, not whatever
    _separator_key names when the run ends — otherwise a concurrent load of
    another model mid-run gets its idle clock / stats 'last used' refreshed
    by a job that never touched it."""
    _foo_model(monkeypatch)
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


def test_cancel_after_the_mutex_wait_skips_the_separation(monkeypatch):
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
    monkeypatch.setattr(bgm_separation, "_progress_tls", types.SimpleNamespace())
    with pytest.raises(bgm_separation.BgmCancelled):
        asyncio.run(bgm_separation._separate_with(
            sep, "in.wav", progress_cb=None, cancel_check=_cancel))
    assert sep.calls == []
    assert getattr(bgm_separation._progress_tls, "cancel", None) is None


def test_release_returns_the_lease_immediately_while_a_load_holds_the_lock(
        monkeypatch):
    """_lock is held for a whole executor-long load; a job whose separation
    is DONE must not sit in its finally until another user's model finishes
    downloading — the release is lock-free and lands before the lock is
    ever given up."""
    _foo_model(monkeypatch)
    sep = object()
    _seed(monkeypatch, sep, ("Foo.onnx", "cpu"))

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
            assert task.done() and task.result() == "out.wav"
            assert bgm_separation._leases == {}

    asyncio.run(_main())


def test_idle_drop_defers_the_free_while_a_same_name_orphan_drains(monkeypatch):
    """Force-drop orphans S1, a later job loads S2 under the same key and
    releases it, the idle evictor drops S2 — that must not unregister the
    stats row / wipe the session device while the S1 job still runs."""
    _seed(monkeypatch, object(), ("m.onnx", "cpu"), orphans={"m.onnx": 1})
    monkeypatch.setattr(bgm_separation, "_session_device", "cuda")
    unregistered = []
    monkeypatch.setattr(model_registry, "unregister_loaded_model",
                        lambda name: unregistered.append(name))

    assert bgm_separation._drop_locked(force=False) is True
    assert bgm_separation._separator is None
    assert unregistered == []
    assert bgm_separation._orphans == {"m.onnx": 1}
    assert bgm_separation.actual_device() == "cuda"


def test_cancel_during_register_keeps_the_separator_its_row_describes(monkeypatch):
    """The register thread writes the stats row even when the awaiting
    request is cancelled; the separator must be cached by then, or the row
    is a phantom _drop_locked (which returns early on no separator) never
    unregisters."""
    _foo_model(monkeypatch)
    sep = object()
    monkeypatch.setattr(bgm_separation, "_load_blocking", lambda m, d: sep)
    entered, gate = blocked_register(monkeypatch)
    cancel_while_register_held(
        lambda: bgm_separation._get_separator("Foo", lease=True), entered, gate)
    assert_cancelled_load_listed(bgm_separation, "_separator", sep, "uvr:Foo.onnx")


def test_cancel_while_register_is_queued_still_lists_the_separator(monkeypatch):
    """With the shared default executor's workers busy the register write is
    still queued when the cancel lands, and a plain cancelled to_thread
    future drops the queued item. The cached separator must still get its
    stats row."""
    _foo_model(monkeypatch)
    sep = object()
    loaded = record_loads(monkeypatch, bgm_separation, lambda: sep)
    cancel_with_register_queued(
        monkeypatch, loaded,
        lambda: bgm_separation._get_separator("Foo", lease=True))
    assert_cancelled_load_listed(bgm_separation, "_separator", sep, "uvr:Foo.onnx")


def test_failed_reload_redoes_the_free_an_orphan_deferred_to_it(monkeypatch):
    """A device change orphans the leased separator and reloads the same
    model; the orphan's last release lands mid-load and skips its unregister
    (the reload would re-register). When that load then fails, the dead
    separator's row and session device must go anyway."""
    _foo_model(monkeypatch, device="cuda")
    old = object()
    _seed(monkeypatch, old, ("Foo.onnx", "cpu"), {"Foo.onnx": 1})
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


def test_task_cancel_mid_separation_defers_the_drop_and_unlinks_the_output(
        monkeypatch):
    """The awaiting task is cancelled while the executor thread is still
    inside sep.separate: the lease is gone, but a non-forced drop must not
    free a session the zombie still uses, and the vocals WAV it writes has
    no owner left — the zombie side unlinks it."""
    monkeypatch.setattr(bgm_separation, "_progress_tls", types.SimpleNamespace())
    _foo_model(monkeypatch)
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
    _seed(monkeypatch, sep, ("Foo.onnx", "cpu"))

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
    _foo_model(monkeypatch)
    held = object()
    _seed(monkeypatch, held, ("Bar.onnx", "cpu"), {"Bar.onnx": 1})
    loads = record_loads(monkeypatch, bgm_separation)

    assert asyncio.run(bgm_separation.load_unleased("Foo")) is False
    assert loads == []
    assert bgm_separation._separator is held
    assert bgm_separation._leases == {"Bar.onnx": 1}
    assert bgm_separation._orphans == {}


def test_a_zombie_separation_makes_the_singleton_busy(monkeypatch):
    """Same zombie as diarization: a cancelled job's thread still inside
    sep.separate holds _separate_mutex after its lease is gone, so busy()
    must refuse the speculative load that would force-drop it."""
    _foo_model(monkeypatch)
    held = object()
    _seed(monkeypatch, held, ("Bar.onnx", "cpu"))
    loads = record_loads(monkeypatch, bgm_separation)
    assert bgm_separation.busy("Foo.onnx") is False
    bgm_separation._separate_mutex.acquire()
    try:
        assert bgm_separation.busy("Foo.onnx") is True
        assert asyncio.run(bgm_separation.load_unleased("Foo")) is False
    finally:
        bgm_separation._separate_mutex.release()
    assert loads == []
    assert bgm_separation._separator is held
