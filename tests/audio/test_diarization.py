"""Diarization stage on POST /v1/audio/transcriptions (soft-fail contract),
the segment-assignment helper, and the pipeline's load / lease / registry
lifecycle. Real pyannote is never imported: handler tests stub `diarize`,
lifecycle tests stub the load and inference below it."""

import logging
import os

import huggingface_hub.constants
import pytest

from faster_whisper_backend.audio import diarization
from faster_whisper_backend.runtime import hf_cache
from tests._transcription_helpers import (
    assert_cancelled_load_listed as _assert_cancelled_load_listed)
from tests._transcription_helpers import blocked_register as _blocked_register
from tests._transcription_helpers import (
    cancel_while_register_held as _cancel_while_register_held)
from tests._transcription_helpers import (
    cancel_with_register_queued as _cancel_with_register_queued)
from tests._transcription_helpers import post_verbose as _post
from tests._transcription_helpers import post_verbose_ok as _post_ok
from tests._transcription_helpers import record_loads as _record_loads


@pytest.fixture
def diar_on(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "DIARIZATION_ENABLED", True, raising=False)


def _seed(monkeypatch, pipe=None, key=None, leases=None, orphans=None):
    monkeypatch.setattr(diarization, "_pipeline", pipe)
    monkeypatch.setattr(diarization, "_pipeline_key", key)
    monkeypatch.setattr(diarization, "_leases", {} if leases is None else leases)
    monkeypatch.setattr(diarization, "_orphans", {} if orphans is None else orphans)


def _diar_model(monkeypatch, model="A", batch=4):
    cfg = diarization.cfg
    monkeypatch.setattr(cfg, "DIARIZATION_MODEL", model, raising=False)
    monkeypatch.setattr(cfg, "DIARIZATION_DEVICE", "cpu", raising=False)
    monkeypatch.setattr(cfg, "DIARIZATION_EMBEDDING_BATCH_SIZE", batch, raising=False)
    _seed(monkeypatch)
    return cfg


def _offline_load_env(monkeypatch):
    """An offline (LOCAL_FILES_ONLY) load starting with HF_HUB_OFFLINE unset,
    both the env var and the hub's import-frozen flag."""
    cfg = diarization.cfg
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(cfg, "DOWNLOAD_ROOT", None, raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(cfg, "LOCAL_FILES_ONLY", True, raising=False)
    return cfg


_ONE_TURN = [(0.0, 1.0, "SPEAKER_00")]


def _stub_turns(monkeypatch, turns=_ONE_TURN):
    calls = []

    async def _fake_diarize(path, *, num_speakers=None, min_speakers=None,
                            max_speakers=None, model_id=None,
                            progress_cb=None, cancel_check=None):
        calls.append({"path": path, "num_speakers": num_speakers,
                      "min_speakers": min_speakers,
                      "max_speakers": max_speakers,
                      "model_id": model_id})
        return turns
    monkeypatch.setattr(diarization, "diarize", _fake_diarize)
    return calls


def test_diarize_labels_segments_and_lists_speakers(client, monkeypatch, diar_on):
    _stub_turns(monkeypatch, [(0.0, 0.6, "SPEAKER_00"),
                              (0.6, 1.0, "SPEAKER_01")])
    body = _post_ok(client, diarize="true")
    # FakeModel returns one segment spanning 0..1 — SPEAKER_00 covers more.
    assert body["segments"][0]["speaker"] == "SPEAKER_00"
    assert body["speakers"] == ["SPEAKER_00"]
    assert "warnings" not in body


def test_diarize_disabled_server_soft_fails(client, app_module, monkeypatch):
    # DIARIZATION_ENABLED defaults off, so this test leaves it unset.
    called = _stub_turns(monkeypatch)
    body = _post_ok(client, diarize="true")
    assert "speaker" not in body["segments"][0]
    assert "speakers" not in body
    assert any("not enabled" in w for w in body["warnings"])
    assert called == []


def test_diarize_absent_inherits_config_default_off(client, monkeypatch, diar_on):
    called = _stub_turns(monkeypatch)
    r = _post(client)
    assert r.status_code == 200
    assert called == []          # DIARIZE global default is false
    assert "warnings" not in r.json()


def test_diarize_config_default_on_applies(client, app_module, monkeypatch,
                                           diar_on):
    monkeypatch.setattr(app_module.cfg, "DIARIZE", True, raising=False)
    called = _stub_turns(monkeypatch)
    r = _post(client)
    assert r.status_code == 200
    assert len(called) == 1      # absent field inherits DIARIZE=true
    # ...and an explicit false still wins over the config default.
    r = _post(client, diarize="false")
    assert r.status_code == 200
    assert len(called) == 1


def test_diarization_error_becomes_warning(client, monkeypatch, diar_on):
    async def _boom(path, **kw):
        raise diarization.DiarizationError(
            "diarization dependencies are not installed on this server "
            "(pip install -r requirements-diarize.txt)")
    monkeypatch.setattr(diarization, "diarize", _boom)
    body = _post_ok(client, diarize="1")
    assert body["text"]                       # transcript survives
    assert "speakers" not in body
    assert any("requirements-diarize" in w for w in body["warnings"])


def test_num_speakers_wins_and_clamps(client, monkeypatch, diar_on):
    called = _stub_turns(monkeypatch)
    r = _post(client, diarize="yes", num_speakers="99", min_speakers="2",
              max_speakers="5")
    assert r.status_code == 200
    assert called[0]["num_speakers"] == 32    # clamped to the cap
    assert called[0]["min_speakers"] is None  # num wins over bounds
    assert called[0]["max_speakers"] is None


def test_min_max_forwarded_without_num(client, monkeypatch, diar_on):
    called = _stub_turns(monkeypatch)
    r = _post(client, diarize="on", min_speakers="2", max_speakers="4")
    assert r.status_code == 200
    assert called[0]["num_speakers"] is None
    assert called[0]["min_speakers"] == 2
    assert called[0]["max_speakers"] == 4


def test_no_segments_skips_diarization_with_warning(client, monkeypatch,
                                                    fake_model, diar_on):
    called = _stub_turns(monkeypatch)
    fake_model._segments = []          # silence: whisper yields nothing
    r = _post(client, diarize="true")
    assert r.status_code == 200
    assert called == []
    assert any("no speech" in w for w in r.json()["warnings"])


def _seg(start, end):
    return {"start": start, "end": end}


def test_assign_speakers_largest_overlap():
    segs = [_seg(0.0, 2.0), _seg(2.0, 4.0)]
    turns = [(0.0, 1.5, "A"), (1.5, 4.0, "B")]
    labels = diarization.assign_speakers(segs, turns)
    assert segs[0]["speaker"] == "A"       # 1.5s of A vs 0.5s of B
    assert segs[1]["speaker"] == "B"
    assert labels == ["A", "B"]


def test_assign_speakers_sums_a_speakers_split_turns():
    """A speaker whose speech in the segment is split around an interjection
    still wins on the total: A 3 s + 3.5 s beats B's single 3.5 s turn."""
    segs = [_seg(0.0, 10.0)]
    turns = [(0.0, 3.0, "A"), (3.0, 6.5, "B"), (6.5, 10.0, "A")]
    labels = diarization.assign_speakers(segs, turns)
    assert segs[0]["speaker"] == "A"
    assert labels == ["A"]


def test_assign_speakers_gap_falls_back_to_nearest():
    segs = [_seg(10.0, 11.0)]              # inside a diarization gap
    turns = [(0.0, 2.0, "A"), (11.5, 20.0, "B")]
    diarization.assign_speakers(segs, turns)
    assert segs[0]["speaker"] == "B"       # 0.5s away vs 8s to A's end


def test_assign_speakers_no_turns_is_noop():
    segs = [_seg(0.0, 1.0)]
    assert diarization.assign_speakers(segs, []) == []
    assert "speaker" not in segs[0]


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
    hits = []

    def _boom(_f, _step=None, **_kw):
        hits.append(1)
        raise RuntimeError("cb exploded")
    hook = diarization._make_hook(_boom)
    hook("segmentation", None, total=10, completed=5)  # must not raise
    # the callback's OWN exception was swallowed, not a signature mismatch
    assert hits == [1]


def test_load_scopes_hf_hub_offline_to_the_load(monkeypatch):
    """LOCAL_FILES_ONLY is hot-editable: one offline pyannote load must not
    pin HF_HUB_OFFLINE process-wide (it would poison every later
    huggingface_hub download — whisper snapshots, translation weights —
    until a restart). The hub freezes the env var at import, so the module
    flag huggingface_hub.constants.HF_HUB_OFFLINE is what must flip (and
    flip back)."""
    _offline_load_env(monkeypatch)
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
    cfg = _offline_load_env(monkeypatch)

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
    _offline_load_env(monkeypatch)
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


def test_diarize_releases_the_key_it_leased_after_a_config_edit(monkeypatch):
    """An admin edit of DIARIZATION_MODEL while a job is loading or running
    must not desynchronise the lease and release keys — the release would
    then decrement a bucket that was never leased and the real lease would
    pin the pipeline forever."""
    import asyncio
    cfg = _diar_model(monkeypatch, "m1")
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


def test_release_drops_the_lease_even_while_the_lock_is_held(monkeypatch):
    """_lock is held across a whole executor-long load; the release must not
    wait on it (a cancellation delivered at that wait would leak the lease
    and pin the pipeline against idle eviction forever)."""
    import asyncio
    pipe = object()
    _seed(monkeypatch, pipe, ("m1", "cpu", 4), {"m1": 1})

    async def _main():
        lock = asyncio.Lock()
        monkeypatch.setattr(diarization, "_lock", lock)
        async with lock:                       # "a load is in flight"
            await asyncio.wait_for(
                diarization._release_pipeline("m1", pipe), 5)
            assert diarization._leases == {}

    asyncio.run(_main())


def test_cancel_during_diarize_still_drops_the_lease(monkeypatch):
    import asyncio
    # The key must match _pipeline_key below: a miss would take _lock, which
    # _main holds — a hang instead of a failure.
    _diar_model(monkeypatch, "m1")
    pipe = object()
    _seed(monkeypatch, pipe, ("m1", "cpu", 4))

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
                # A bound, so a release that re-takes _lock fails, not hangs.
                await asyncio.wait_for(diarization.diarize("a.wav"), 5)
        assert diarization._leases == {}
        assert await diarization.drop_pipeline(force=False) is True

    asyncio.run(_main())


def test_idle_drop_waits_for_a_same_id_orphan(monkeypatch):
    """A force-drop orphaned pipeline A, A reloaded under the same id and
    then idled: dropping the reload must leave the stats row and teardown
    to the orphan's last release (job 1 is still running on A)."""
    from faster_whisper_backend.runtime import model_registry
    _seed(monkeypatch, object(), ("A", "cpu", 4), orphans={"A": 1})
    unregistered: "list[str]" = []
    monkeypatch.setattr(model_registry, "unregister_loaded_model",
                        unregistered.append)

    assert diarization._drop_locked(force=False) is True
    assert diarization._pipeline is None
    assert unregistered == []
    assert diarization._orphans == {"A": 1}

    diarization._release_locked("A", pipe=object())
    assert unregistered == ["pyannote:A"]
    assert diarization._orphans == {}


def test_diarize_does_not_touch_a_model_it_never_leased(monkeypatch):
    """The end-of-run restamp touches only the model this job leased: after
    a mid-job re-key to another model, that model's idle clock / stats
    'last used' must not move for a job that never touched it."""
    import asyncio
    from faster_whisper_backend.runtime import model_registry
    _diar_model(monkeypatch, "m1")

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
    monkeypatch.setattr(model_registry, "touch_loaded_model",
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
    assert hf_cache.hub_cache_dir() == os.path.join("/elsewhere", "hub")
    monkeypatch.delenv("HF_HOME")
    monkeypatch.setattr(diarization.cfg, "DOWNLOAD_ROOT", None, raising=False)
    assert hf_cache.hub_cache_dir() is None


def test_placement_error_keeps_raw_torch_text_out_of_the_message(monkeypatch):
    """str(DiarizationError) lands in the client's `warnings` — torch's own
    text (VRAM figures, driver paths) must stay in the server log."""
    import sys
    import types
    monkeypatch.setattr(diarization.cfg, "HF_TOKEN", None, raising=False)

    class _Pipe:
        def to(self, dev):
            raise RuntimeError("SECRET /opt/path CUDA out of memory")

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kw):
            return _Pipe()

    fake_torch = types.SimpleNamespace(
        set_num_threads=lambda n: None, device=lambda d: d)
    pa = types.ModuleType("pyannote.audio")
    pa.Pipeline = _Pipeline
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pyannote", types.ModuleType("pyannote"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", pa)

    with pytest.raises(diarization.DiarizationError) as ei:
        diarization._load_blocking_inner("p/m", "cuda", 4)
    assert "SECRET" not in str(ei.value)
    assert "/opt/path" not in str(ei.value)


def test_cancel_during_register_keeps_the_pipeline_its_row_describes(monkeypatch):
    """The register thread writes "pyannote:<id>" even when the awaiting
    caller is cancelled; with no cached pipeline that row was a phantom
    _drop_locked (early return on no pipeline) never unregistered."""
    _diar_model(monkeypatch)
    pipe = object()
    monkeypatch.setattr(diarization, "_load_blocking", lambda m, d, b: pipe)
    entered, gate = _blocked_register(monkeypatch)
    _cancel_while_register_held(
        lambda: diarization._get_pipeline(lease=True), entered, gate)
    _assert_cancelled_load_listed(diarization, "_pipeline", pipe, "pyannote:A")


def test_cancel_while_register_is_queued_still_lists_the_pipeline(monkeypatch):
    """The default executor is shared with every other to_thread call: with
    its workers busy the register write is still queued when the cancel
    lands, and a plain cancelled to_thread future drops the queued item.
    The cached pipeline must still get its "pyannote:<id>" row."""
    _diar_model(monkeypatch)
    pipe = object()
    loaded = _record_loads(monkeypatch, diarization, lambda: pipe)
    _cancel_with_register_queued(
        monkeypatch, loaded, lambda: diarization._get_pipeline(lease=True))
    _assert_cancelled_load_listed(diarization, "_pipeline", pipe, "pyannote:A")


def test_orphan_release_mid_register_keeps_the_new_row(monkeypatch):
    """A batch edit orphans the leased pipeline and reloads the same id; the
    orphan's last release landing while the new row is being registered must
    not unregister the row the live pipeline just wrote."""
    import asyncio
    from faster_whisper_backend.runtime import model_registry
    _diar_model(monkeypatch, batch=8)
    old, new = object(), object()
    _seed(monkeypatch, old, ("A", "cpu", 4), {"A": 1})
    monkeypatch.setattr(diarization, "_load_blocking", lambda m, d, b: new)
    entered, gate = _blocked_register(monkeypatch)

    async def _main():
        task = asyncio.create_task(diarization._get_pipeline())
        while not entered.is_set():
            await asyncio.sleep(0.01)
        await diarization._release_pipeline("A", old)   # the orphan drains
        gate.set()
        assert await task is new
    asyncio.run(_main())
    assert diarization._orphans == {}
    assert "pyannote:A" in model_registry._loaded_models
    with monkeypatch.context() as m:
        m.setattr(diarization, "_pipeline_key", None)
        diarization._free_locked("A")   # tidy the real registry row


def test_cancel_after_the_mutex_wait_skips_the_inference():
    """The _infer_mutex wait can be minutes (a zombie diarization); a request
    cancelled meanwhile must not start pipe() once it gets the mutex."""
    import asyncio
    answers = iter([False])          # queued: not yet; acquired: cancelled
    calls: "list[str]" = []

    def _pipe(path, **_kw):
        calls.append(path)
        raise AssertionError("pipe() ran on a cancelled request")

    with pytest.raises(diarization.DiarizeCancelled):
        asyncio.run(diarization._diarize_with(
            _pipe, "in.wav", num_speakers=None, min_speakers=None,
            max_speakers=None, progress_cb=None,
            cancel_check=lambda: next(answers, True)))
    assert calls == []


def test_idle_drop_waits_for_a_zombie_inference(monkeypatch):
    """A cancelled diarize released its lease, but its executor thread is
    still inside pipe() holding _infer_mutex: a non-forced drop must decline
    (no unregister, no "unloaded") until the thread lets go."""
    import asyncio
    import threading
    from faster_whisper_backend.runtime import model_registry
    _seed(monkeypatch, object(), ("m1", "cpu", 4))
    unregistered: "list[str]" = []
    monkeypatch.setattr(model_registry, "unregister_loaded_model",
                        unregistered.append)
    held, release = threading.Event(), threading.Event()

    def _zombie():
        with diarization._infer_mutex:
            held.set()
            release.wait(5)
    t = threading.Thread(target=_zombie, daemon=True)
    t.start()
    try:
        assert held.wait(5)
        assert asyncio.run(diarization.drop_pipeline(force=False)) is False
        assert diarization._pipeline is not None
        assert unregistered == []
    finally:
        release.set()
        t.join(5)
    assert asyncio.run(diarization.drop_pipeline(force=False)) is True
    assert unregistered == ["pyannote:m1"]


def test_preload_declines_under_the_lock_when_a_job_holds_another_pipeline(
        monkeypatch):
    """preload admits on busy() BEFORE load_unleased waits on _lock; a job
    that loaded and leased another pipeline meanwhile must not be orphaned
    by the speculative load (two pipelines in memory)."""
    import asyncio
    _diar_model(monkeypatch)
    held = object()
    _seed(monkeypatch, held, ("p/a", "cpu", 4), {"p/a": 1})
    loads = _record_loads(monkeypatch, diarization)

    assert asyncio.run(diarization.load_unleased("p/b")) is False
    assert loads == []
    assert diarization._pipeline is held
    assert diarization._leases == {"p/a": 1} and diarization._orphans == {}
    # A draining orphan declines too.
    _seed(monkeypatch, orphans={"p/a": 1})
    assert asyncio.run(diarization.load_unleased("p/b")) is False
    assert loads == []


def test_a_zombie_inference_makes_the_singleton_busy(monkeypatch):
    """A cancelled diarize released its lease, but its executor thread still
    holds _infer_mutex inside pipe(): busy() must say so, or the preload's
    load_unleased force-drops (unregisters, empty_cache) the pipeline under
    the zombie and loads a second one beside it."""
    import asyncio
    _diar_model(monkeypatch)
    held = object()
    _seed(monkeypatch, held, ("p/a", "cpu", 4))
    loads = _record_loads(monkeypatch, diarization)
    assert diarization.busy("p/b") is False
    diarization._infer_mutex.acquire()
    try:
        assert diarization.busy("p/b") is True
        assert asyncio.run(diarization.load_unleased("p/b")) is False
    finally:
        diarization._infer_mutex.release()
    assert loads == []
    assert diarization._pipeline is held
