"""Diarization stage on POST /v1/audio/transcriptions (soft-fail contract) +
the pure segment-assignment helper. pyannote is never imported — the module's
`diarize` coroutine is monkeypatched, exactly the boundary the handler uses."""

import logging
import os

import huggingface_hub.constants
import pytest

from faster_whisper_backend.audio import diarization

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


def _post(client, **data):
    data.setdefault("model", "whisper-1")
    data.setdefault("response_format", "verbose_json")
    return client.post("/v1/audio/transcriptions", files=_FILE, data=data)


def _stub_turns(monkeypatch, turns, calls=None):
    async def _fake_diarize(path, *, num_speakers=None, min_speakers=None,
                            max_speakers=None, model_id=None,
                            progress_cb=None, cancel_check=None):
        if calls is not None:
            calls.append({"path": path, "num_speakers": num_speakers,
                          "min_speakers": min_speakers,
                          "max_speakers": max_speakers,
                          "model_id": model_id})
        return turns
    monkeypatch.setattr(diarization, "diarize", _fake_diarize)


# --- route behaviour ---------------------------------------------------------

def test_diarize_labels_segments_and_lists_speakers(client, app_module, monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True
    try:
        _stub_turns(monkeypatch, [(0.0, 0.6, "SPEAKER_00"),
                                  (0.6, 1.0, "SPEAKER_01")])
        r = _post(client, diarize="true")
        assert r.status_code == 200, r.text
        body = r.json()
        # FakeModel returns one segment spanning 0..1 — SPEAKER_00 covers more.
        assert body["segments"][0]["speaker"] == "SPEAKER_00"
        assert body["speakers"] == ["SPEAKER_00"]
        assert "warnings" not in body
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False


def test_diarize_disabled_server_soft_fails(client, app_module, monkeypatch):
    # DIARIZATION_ENABLED defaults off: the request still succeeds, the
    # transcript has no speakers, and a warning explains why.
    called = []
    _stub_turns(monkeypatch, [(0.0, 1.0, "SPEAKER_00")], calls=called)
    r = _post(client, diarize="true")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "speaker" not in body["segments"][0]
    assert "speakers" not in body
    assert any("not enabled" in w for w in body["warnings"])
    assert called == []


def test_diarize_absent_inherits_config_default_off(client, app_module, monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True
    try:
        called = []
        _stub_turns(monkeypatch, [(0.0, 1.0, "SPEAKER_00")], calls=called)
        r = _post(client)
        assert r.status_code == 200
        assert called == []          # DIARIZE global default is false
        assert "warnings" not in r.json()
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False


def test_diarize_config_default_on_applies(client, app_module, monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True
    app_module.cfg.DIARIZE = True
    try:
        called = []
        _stub_turns(monkeypatch, [(0.0, 1.0, "SPEAKER_00")], calls=called)
        r = _post(client)
        assert r.status_code == 200
        assert len(called) == 1      # absent field inherits DIARIZE=true
        # ...and an explicit false still wins over the config default.
        r = _post(client, diarize="false")
        assert r.status_code == 200
        assert len(called) == 1
    finally:
        app_module.cfg.DIARIZE = False
        app_module.cfg.DIARIZATION_ENABLED = False


def test_diarization_error_becomes_warning(client, app_module, monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True
    try:
        async def _boom(path, **kw):
            raise diarization.DiarizationError(
                "diarization dependencies are not installed on this server "
                "(pip install -r requirements-diarize.txt)")
        monkeypatch.setattr(diarization, "diarize", _boom)
        r = _post(client, diarize="1")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["text"]                       # transcript survives
        assert "speakers" not in body
        assert any("requirements-diarize" in w for w in body["warnings"])
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False


def test_num_speakers_wins_and_clamps(client, app_module, monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True
    try:
        called = []
        _stub_turns(monkeypatch, [(0.0, 1.0, "SPEAKER_00")], calls=called)
        r = _post(client, diarize="yes", num_speakers="99", min_speakers="2",
                  max_speakers="5")
        assert r.status_code == 200
        assert called[0]["num_speakers"] == 32    # clamped to the cap
        assert called[0]["min_speakers"] is None  # num wins over bounds
        assert called[0]["max_speakers"] is None
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False


def test_min_max_forwarded_without_num(client, app_module, monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True
    try:
        called = []
        _stub_turns(monkeypatch, [(0.0, 1.0, "SPEAKER_00")], calls=called)
        r = _post(client, diarize="on", min_speakers="2", max_speakers="4")
        assert r.status_code == 200
        assert called[0]["num_speakers"] is None
        assert called[0]["min_speakers"] == 2
        assert called[0]["max_speakers"] == 4
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False


def test_no_segments_skips_diarization_with_warning(client, app_module,
                                                    monkeypatch, fake_model):
    app_module.cfg.DIARIZATION_ENABLED = True
    try:
        called = []
        _stub_turns(monkeypatch, [(0.0, 1.0, "SPEAKER_00")], calls=called)
        fake_model._segments = []          # silence: whisper yields nothing
        r = _post(client, diarize="true")
        assert r.status_code == 200
        assert called == []
        assert any("no speech" in w for w in r.json()["warnings"])
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False


# --- assign_speakers (pure) --------------------------------------------------

def _seg(start, end):
    return {"start": start, "end": end}


def test_assign_speakers_largest_overlap():
    segs = [_seg(0.0, 2.0), _seg(2.0, 4.0)]
    turns = [(0.0, 1.5, "A"), (1.5, 4.0, "B")]
    labels = diarization.assign_speakers(segs, turns)
    assert segs[0]["speaker"] == "A"       # 1.5s of A vs 0.5s of B
    assert segs[1]["speaker"] == "B"
    assert labels == ["A", "B"]


def test_assign_speakers_gap_falls_back_to_nearest():
    segs = [_seg(10.0, 11.0)]              # inside a diarization gap
    turns = [(0.0, 2.0, "A"), (11.5, 20.0, "B")]
    diarization.assign_speakers(segs, turns)
    assert segs[0]["speaker"] == "B"       # 0.5s away vs 8s to A's end


def test_assign_speakers_no_turns_is_noop():
    segs = [_seg(0.0, 1.0)]
    assert diarization.assign_speakers(segs, []) == []
    assert "speaker" not in segs[0]


# --- progress hook -----------------------------------------------------------

def test_hook_maps_steps_and_stays_monotone(caplog):
    seen = []
    units = []
    hook = diarization._make_hook(
        lambda f, step=None, **kw: (seen.append(f),
                                    units.append((kw.get("target"), kw.get("target_progress")))))
    with caplog.at_level(logging.INFO, logger="whisper-server"):
        hook("segmentation", None, total=10, completed=5)
        hook("segmentation", None, total=10, completed=10)
        hook("embeddings", None, total=4, completed=2)
        # A regression (pyannote re-reports an earlier step) must not move the
        # bar backwards — it is simply dropped.
        hook("segmentation", None, total=10, completed=1)
        hook("clustering", None)      # no total → logged but never moves the bar
        hook("embeddings", None, total=4, completed=4)
        assert seen == pytest.approx([0.02, 0.04, 0.04 + 0.93 * 0.5, 0.97])
        # A repeated bare call stays silent (no bar move, no re-log); a later
        # CHUNKED call from the same step promotes it to the next free window
        # instead of staying untracked forever.
        hook("clustering", None)
        hook("clustering", None, total=4, completed=4)
    assert seen == pytest.approx([0.02, 0.04, 0.04 + 0.93 * 0.5, 0.97, 1.0])
    # Every moving tick names the plan's unit and the step's own fraction;
    # the promoted third step is the clustering unit.
    assert units == [("segmentation", 0.5), ("segmentation", 1.0),
                     ("embeddings", 0.5), ("embeddings", 1.0), ("clustering", 1.0)]
    step_lines = [r.getMessage() for r in caplog.records
                  if "step: clustering" in r.getMessage()]
    assert step_lines == ["[diarize] step: clustering (untracked)",
                          "[diarize] step: clustering (promoted)"]


def test_hook_reports_untracked_steps_as_clustering_only_after_embeddings():
    """pyannote counts speakers BETWEEN segmentation and embeddings: that
    bare step must not start the clustering unit (it would close the
    embeddings unit before it ran). The bare step after embeddings does."""
    units = []
    hook = diarization._make_hook(
        lambda f, step=None, **kw: units.append((step, kw.get("target"))))
    hook("segmentation", None, total=2, completed=2)
    hook("speaker_counting", None)
    hook("embeddings", None, total=2, completed=1)
    hook("embeddings", None, total=2, completed=2)
    hook("discrete_diarization", None)
    assert units == [("segmentation", "segmentation"), ("embeddings", "embeddings"),
                     ("embeddings", "embeddings"),
                     ("discrete_diarization", "clustering")]


def test_hook_swallows_bad_callback():
    def _boom(_f, _step=None):
        raise RuntimeError("cb exploded")
    hook = diarization._make_hook(_boom)
    hook("segmentation", None, total=10, completed=5)  # must not raise


# --- offline env var is scoped to the load -----------------------------------

def test_load_scopes_hf_hub_offline_to_the_load(monkeypatch):
    """LOCAL_FILES_ONLY is hot-editable: one offline pyannote load must not
    pin HF_HUB_OFFLINE process-wide (it would poison every later
    huggingface_hub download — whisper snapshots, translation weights —
    until a restart). The hub freezes the env var at import, so the module
    flag huggingface_hub.constants.HF_HUB_OFFLINE is what must flip (and
    flip back)."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(diarization.cfg, "DOWNLOAD_ROOT", None, raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(diarization.cfg, "LOCAL_FILES_ONLY", True,
                        raising=False)
    seen = {}

    def _fake_inner(model_id, device, batch_size):
        seen["offline"] = os.environ.get("HF_HUB_OFFLINE")
        seen["const"] = huggingface_hub.constants.HF_HUB_OFFLINE
        return object()
    monkeypatch.setattr(diarization, "_load_blocking_inner", _fake_inner)

    diarization._load_blocking("m1", "cpu", 4)
    assert seen["offline"] == "1"          # offline DURING the load...
    assert seen["const"] is True
    assert "HF_HUB_OFFLINE" not in os.environ   # ...and restored after
    assert huggingface_hub.constants.HF_HUB_OFFLINE is False

    # A pre-existing value is restored, not popped.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    diarization._load_blocking("m1", "cpu", 4)
    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_load_snapshots_local_files_only_for_the_whole_load(monkeypatch):
    """The flag is hot and the load runs for minutes: an admin flip mid-load
    must neither skip the restore (True → False would leak HF_HUB_OFFLINE=1
    process-wide) nor clobber a value this load never set (False → True)."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(diarization.cfg, "DOWNLOAD_ROOT", None, raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    cfg = diarization.cfg
    monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", True, raising=False)

    def _flip_off(model_id, device, batch_size):
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert huggingface_hub.constants.HF_HUB_OFFLINE is True
        cfg.LOCAL_FILES_ONLY = False
        return object()
    monkeypatch.setattr(diarization, "_load_blocking_inner", _flip_off)
    diarization._load_blocking("m1", "cpu", 4)
    assert "HF_HUB_OFFLINE" not in os.environ
    assert huggingface_hub.constants.HF_HUB_OFFLINE is False

    # Mirror: starts False (nothing set), flips True inside — the finally
    # must leave a pre-existing value alone.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", False, raising=False)

    def _flip_on(model_id, device, batch_size):
        assert os.environ.get("HF_HUB_OFFLINE") == "0"
        assert huggingface_hub.constants.HF_HUB_OFFLINE is False
        cfg.LOCAL_FILES_ONLY = True
        return object()
    monkeypatch.setattr(diarization, "_load_blocking_inner", _flip_on)
    diarization._load_blocking("m1", "cpu", 4)
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert huggingface_hub.constants.HF_HUB_OFFLINE is False


def test_load_makes_hub_offline_mode_true_for_the_load(monkeypatch):
    """What the hub's request gates actually consult is
    constants.is_offline_mode(), which reads the frozen module flag — the
    offline load must make THAT report True, and only for its duration."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(diarization.cfg, "DOWNLOAD_ROOT", None, raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(diarization.cfg, "LOCAL_FILES_ONLY", True,
                        raising=False)
    seen = []

    def _fake_inner(model_id, device, batch_size):
        seen.append(huggingface_hub.constants.is_offline_mode())
        return object()
    monkeypatch.setattr(diarization, "_load_blocking_inner", _fake_inner)

    diarization._load_blocking("m1", "cpu", 4)
    assert seen == [True]
    assert huggingface_hub.constants.is_offline_mode() is False

    # An online load never touches the constant.
    monkeypatch.setattr(diarization.cfg, "LOCAL_FILES_ONLY", False,
                        raising=False)
    diarization._load_blocking("m1", "cpu", 4)
    assert seen == [True, False]
    assert huggingface_hub.constants.is_offline_mode() is False


# --- lease key is resolved once ---------------------------------------------

def test_diarize_releases_the_key_it_leased_after_a_config_edit(monkeypatch):
    """An admin edit of DIARIZATION_MODEL while a job is loading or running
    must not desynchronise the lease and release keys — the release would
    then decrement a bucket that was never leased and the real lease would
    pin the pipeline forever."""
    import asyncio
    cfg = diarization.cfg
    monkeypatch.setattr(cfg, "DIARIZATION_MODEL", "m1", raising=False)
    monkeypatch.setattr(cfg, "DIARIZATION_DEVICE", "cpu", raising=False)
    monkeypatch.setattr(diarization, "_load_blocking",
                        lambda model_id, device, batch: object())

    async def _fake_diarize_with(pipe, path, **kw):
        # The load has happened and the lease is held; the admin edits now.
        cfg.DIARIZATION_MODEL = "m2"
        return []
    monkeypatch.setattr(diarization, "_diarize_with", _fake_diarize_with)

    assert asyncio.run(diarization.diarize("a.wav")) == []
    assert diarization._leases == {}
    assert diarization._orphans == {}


# --- the release never suspends ----------------------------------------------

def test_release_drops_the_lease_even_while_the_lock_is_held(monkeypatch):
    """_lock is held across a whole executor-long load; the release must not
    wait on it (a cancellation delivered at that wait would leak the lease
    and pin the pipeline against idle eviction forever)."""
    import asyncio
    pipe = object()
    monkeypatch.setattr(diarization, "_pipeline", pipe)
    monkeypatch.setattr(diarization, "_pipeline_key", ("m1", "cpu", 4))
    monkeypatch.setattr(diarization, "_leases", {"m1": 1})
    monkeypatch.setattr(diarization, "_orphans", {})

    async def _main():
        lock = asyncio.Lock()
        monkeypatch.setattr(diarization, "_lock", lock)
        async with lock:                       # "a load is in flight"
            await diarization._release_pipeline("m1", pipe)
            assert diarization._leases == {}

    asyncio.run(_main())


def test_cancel_delivered_at_release_still_drops_the_lease(monkeypatch):
    import asyncio
    cfg = diarization.cfg
    monkeypatch.setattr(cfg, "DIARIZATION_MODEL", "m1", raising=False)
    monkeypatch.setattr(cfg, "DIARIZATION_DEVICE", "cpu", raising=False)
    pipe = object()
    monkeypatch.setattr(diarization, "_pipeline", pipe)
    monkeypatch.setattr(diarization, "_pipeline_key", ("m1", "cpu", 4))
    monkeypatch.setattr(diarization, "_leases", {})
    monkeypatch.setattr(diarization, "_orphans", {})

    async def _fake_diarize_with(pipe, path, **kw):
        assert diarization._leases == {"m1": 1}
        # The client went away: the cancellation lands on the next suspension
        # and unwinds into diarize's finally with the lock held elsewhere.
        asyncio.current_task().cancel()
        await asyncio.sleep(0)
        raise AssertionError("cancellation must land at the sleep")
    monkeypatch.setattr(diarization, "_diarize_with", _fake_diarize_with)

    async def _main():
        lock = asyncio.Lock()
        monkeypatch.setattr(diarization, "_lock", lock)
        async with lock:
            with pytest.raises(asyncio.CancelledError):
                await diarization.diarize("a.wav")
        assert diarization._leases == {}
        assert await diarization.drop_pipeline(force=False) is True

    asyncio.run(_main())


# --- the tail restamp is owned by the release --------------------------------

def test_diarize_does_not_touch_a_model_it_never_leased(monkeypatch):
    """_diarize_with used to restamp whatever _pipeline_key named when the
    run ended — after a mid-job re-key to another model, that model's idle
    clock / stats 'last used' moved for a job that never touched it."""
    import asyncio
    from faster_whisper_backend.runtime import system_stats
    cfg = diarization.cfg
    monkeypatch.setattr(cfg, "DIARIZATION_MODEL", "m1", raising=False)
    monkeypatch.setattr(cfg, "DIARIZATION_DEVICE", "cpu", raising=False)
    monkeypatch.setattr(diarization, "_pipeline", None)
    monkeypatch.setattr(diarization, "_pipeline_key", None)
    monkeypatch.setattr(diarization, "_leases", {})
    monkeypatch.setattr(diarization, "_orphans", {})

    class _Ann:
        def itertracks(self, yield_label=True):
            return iter([])

    class _Pipe:
        def __call__(self, path, **kw):
            # A concurrent _get_pipeline("m2") re-keyed the singleton.
            diarization._pipeline_key = ("m2", "cpu", 4)
            return _Ann()

    monkeypatch.setattr(diarization, "_load_blocking",
                        lambda model_id, device, batch: _Pipe())
    touched = []
    monkeypatch.setattr(system_stats, "touch_loaded_model",
                        lambda name: touched.append(name))
    assert asyncio.run(diarization.diarize("a.wav")) == []
    assert "pyannote:m2" not in touched
    assert diarization._leases == {}


def test_pipeline_load_passes_the_models_volume_cache_dir(monkeypatch,
                                                          tmp_path):
    """huggingface_hub froze HF_HUB_CACHE at import, so the HF_HOME
    setdefault cannot redirect the download — without an explicit cache_dir
    the pipeline landed in ~/.cache/huggingface on bare metal, off the
    models volume and invisible to model_sizes (size_unknown forever)."""
    import sys
    import types
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(diarization.cfg, "DOWNLOAD_ROOT", str(tmp_path),
                        raising=False)
    monkeypatch.setattr(diarization.cfg, "HF_TOKEN", "tok", raising=False)
    seen = {}

    class _Pipe:
        def to(self, dev):
            pass

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kw):
            seen.update(kw)
            return _Pipe()

    fake_torch = types.SimpleNamespace(
        set_num_threads=lambda n: None, device=lambda d: d)
    pa = types.ModuleType("pyannote.audio")
    pa.Pipeline = _Pipeline
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", pa)

    diarization._load_blocking_inner("p/m", "cpu", 4)
    assert seen["cache_dir"] == os.path.join(str(tmp_path), "hf", "hub")
    assert seen["token"] == "tok"

    # A set HF_HOME wins; neither set falls through to the hub's default.
    monkeypatch.setenv("HF_HOME", "/elsewhere")
    assert diarization._hf_cache_dir() == os.path.join("/elsewhere", "hub")
    monkeypatch.delenv("HF_HOME")
    monkeypatch.setattr(diarization.cfg, "DOWNLOAD_ROOT", None, raising=False)
    assert diarization._hf_cache_dir() is None
