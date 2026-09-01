"""Job leases for the pyannote pipeline and the UVR separator singletons.

Neither pyannote nor audio_separator is imported: `_load_blocking` is the
module boundary both `_get_pipeline` / `_get_separator` call, so the tests
monkeypatch that and hand back a callable stub.
"""

import asyncio
import logging
import types

import pytest

import bgm_separation
import diarization
import system_stats


class _RaisingLock:
    """Stand-in for the module lock: entering it at all is the failure."""

    async def __aenter__(self):
        raise AssertionError("cache hit must not acquire the module lock")

    async def __aexit__(self, *exc):
        return False


# --- diarization -------------------------------------------------------------

class _Ann:
    def itertracks(self, yield_label=True):
        return iter([(types.SimpleNamespace(start=0.0, end=1.0), None, "S0")])


class _Pipe:
    """Callable pyannote-pipeline stub; records the leases seen mid-inference."""

    def __init__(self, boom=None):
        self.boom = boom
        self.leases_during = None

    def __call__(self, path, **kw):
        self.leases_during = dict(diarization._leases)
        if self.boom is not None:
            raise self.boom
        return _Ann()


@pytest.fixture
def diar_cfg(monkeypatch):
    cfg = diarization.cfg
    monkeypatch.setattr(cfg, "DIARIZATION_MODEL", "m1", raising=False)
    monkeypatch.setattr(cfg, "DIARIZATION_DEVICE", "cpu", raising=False)
    monkeypatch.setattr(cfg, "DIARIZATION_EMBEDDING_BATCH_SIZE", 4,
                        raising=False)
    return cfg


def _stub_pipes(monkeypatch, made, pipe=None):
    def _load(model_id, device, batch):
        made.append(model_id)
        return pipe if pipe is not None else _Pipe()
    monkeypatch.setattr(diarization, "_load_blocking", _load)


def test_diarize_cache_hit_does_not_take_the_lock(diar_cfg, monkeypatch):
    diarization._pipeline = _Pipe()
    diarization._pipeline_key = ("m1", "cpu", 4)
    monkeypatch.setattr(diarization, "_lock", _RaisingLock())

    got = asyncio.run(diarization._get_pipeline("m1", lease=True))

    assert got is diarization._pipeline
    assert diarization._leases == {"m1": 1}


def test_diarize_drop_refuses_while_leased(diar_cfg, monkeypatch):
    _stub_pipes(monkeypatch, [])
    asyncio.run(diarization._get_pipeline("m1", lease=True))

    assert asyncio.run(diarization.drop_pipeline(force=False)) is False
    assert diarization._pipeline is not None
    assert "pyannote:m1" in system_stats._loaded_models


def test_diarize_force_orphans_and_the_last_release_frees(diar_cfg, monkeypatch):
    _stub_pipes(monkeypatch, [])
    asyncio.run(diarization._get_pipeline("m1", lease=True))

    assert asyncio.run(diarization.drop_pipeline()) is True
    # Orphaned: out of the cache, but still registered — the job inside the
    # executor is holding its own reference.
    assert diarization._pipeline is None
    assert diarization._orphans == {"m1": 1}
    assert "pyannote:m1" in system_stats._loaded_models

    asyncio.run(diarization._release_pipeline("m1"))
    assert diarization._orphans == {}
    assert "pyannote:m1" not in system_stats._loaded_models


def test_diarize_other_model_mid_job_keeps_both(diar_cfg, monkeypatch):
    made: "list[str]" = []
    _stub_pipes(monkeypatch, made)

    asyncio.run(diarization._get_pipeline("m1", lease=True))
    asyncio.run(diarization._get_pipeline("m2"))

    assert made == ["m1", "m2"]
    assert diarization._pipeline_key[0] == "m2"
    assert "pyannote:m1" in system_stats._loaded_models
    assert "pyannote:m2" in system_stats._loaded_models

    asyncio.run(diarization._release_pipeline("m1"))
    assert "pyannote:m1" not in system_stats._loaded_models
    assert "pyannote:m2" in system_stats._loaded_models


def test_diarize_balances_the_lease_on_success(diar_cfg, monkeypatch, tmp_path):
    pipe = _Pipe()
    _stub_pipes(monkeypatch, [], pipe)

    turns = asyncio.run(diarization.diarize(str(tmp_path / "a.wav")))

    assert turns == [(0.0, 1.0, "S0")]
    assert pipe.leases_during == {"m1": 1}   # held for the whole inference
    assert diarization._leases == {}
    assert diarization._orphans == {}


def test_diarize_balances_the_lease_on_error(diar_cfg, monkeypatch, tmp_path):
    _stub_pipes(monkeypatch, [], _Pipe(boom=RuntimeError("nope")))

    with pytest.raises(diarization.DiarizationError):
        asyncio.run(diarization.diarize(str(tmp_path / "a.wav")))

    assert diarization._leases == {}


def test_diarize_balances_the_lease_on_cancel(diar_cfg, monkeypatch, tmp_path):
    _stub_pipes(monkeypatch, [])

    with pytest.raises(diarization.DiarizeCancelled):
        asyncio.run(diarization.diarize(str(tmp_path / "a.wav"),
                                        cancel_check=lambda: True))

    assert diarization._leases == {}


# --- bgm separation ----------------------------------------------------------

class _Sep:
    """audio-separator stub; records the leases seen mid-separation."""

    def __init__(self, out_path, boom=None):
        self.out_path = out_path
        self.boom = boom
        self.leases_during = None

    def separate(self, path, custom_output_names=None):
        self.leases_during = dict(bgm_separation._leases)
        if self.boom is not None:
            raise self.boom
        return [self.out_path]


@pytest.fixture
def bgm_cfg(monkeypatch):
    cfg = bgm_separation.cfg
    monkeypatch.setattr(cfg, "BGM_SEPARATION_UVR_MODEL", "Foo", raising=False)
    monkeypatch.setattr(cfg, "BGM_SEPARATION_DEVICE", "cpu", raising=False)
    return cfg


def _stub_seps(monkeypatch, made, sep=None):
    def _load(model, device):
        made.append(model)
        return sep if sep is not None else _Sep("")
    monkeypatch.setattr(bgm_separation, "_load_blocking", _load)


def test_bgm_cache_hit_does_not_take_the_lock(bgm_cfg, monkeypatch):
    bgm_separation._separator = _Sep("")
    bgm_separation._separator_key = ("Foo.onnx", "cpu")
    monkeypatch.setattr(bgm_separation, "_lock", _RaisingLock())

    got = asyncio.run(bgm_separation._get_separator("Foo", lease=True))

    assert got is bgm_separation._separator
    # The lease is keyed by the NORMALISED filename, like the cache itself.
    assert bgm_separation._leases == {"Foo.onnx": 1}


def test_bgm_drop_refuses_while_leased(bgm_cfg, monkeypatch):
    _stub_seps(monkeypatch, [])
    asyncio.run(bgm_separation._get_separator("Foo", lease=True))

    assert asyncio.run(bgm_separation.drop_separator(force=False)) is False
    assert bgm_separation._separator is not None
    assert "uvr:Foo.onnx" in system_stats._loaded_models


def test_bgm_force_orphans_and_the_last_release_frees(bgm_cfg, monkeypatch):
    _stub_seps(monkeypatch, [])
    asyncio.run(bgm_separation._get_separator("Foo", lease=True))

    assert asyncio.run(bgm_separation.drop_separator()) is True
    assert bgm_separation._separator is None
    assert bgm_separation._orphans == {"Foo.onnx": 1}
    assert "uvr:Foo.onnx" in system_stats._loaded_models

    asyncio.run(bgm_separation._release_separator("Foo.onnx"))
    assert bgm_separation._orphans == {}
    assert "uvr:Foo.onnx" not in system_stats._loaded_models


def test_bgm_other_model_mid_job_keeps_both(bgm_cfg, monkeypatch):
    made: "list[str]" = []
    _stub_seps(monkeypatch, made)

    asyncio.run(bgm_separation._get_separator("Foo", lease=True))
    asyncio.run(bgm_separation._get_separator("Bar"))

    assert made == ["Foo.onnx", "Bar.onnx"]
    assert bgm_separation._separator_key[0] == "Bar.onnx"
    assert "uvr:Foo.onnx" in system_stats._loaded_models
    assert "uvr:Bar.onnx" in system_stats._loaded_models

    asyncio.run(bgm_separation._release_separator("Foo.onnx"))
    assert "uvr:Foo.onnx" not in system_stats._loaded_models
    assert "uvr:Bar.onnx" in system_stats._loaded_models


def test_bgm_balances_the_lease_on_success(bgm_cfg, monkeypatch, tmp_path):
    out = tmp_path / "vocals.wav"
    out.write_bytes(b"RIFF")
    sep = _Sep(str(out))
    _stub_seps(monkeypatch, [], sep)

    got = asyncio.run(bgm_separation.separate(str(tmp_path / "in.wav")))

    assert got == str(out)
    assert sep.leases_during == {"Foo.onnx": 1}
    assert bgm_separation._leases == {}
    assert bgm_separation._orphans == {}


def test_bgm_balances_the_lease_on_error(bgm_cfg, monkeypatch, tmp_path):
    _stub_seps(monkeypatch, [], _Sep("", boom=RuntimeError("nope")))

    with pytest.raises(bgm_separation.BgmSeparationError):
        asyncio.run(bgm_separation.separate(str(tmp_path / "in.wav")))

    assert bgm_separation._leases == {}


def test_bgm_balances_the_lease_on_cancel(bgm_cfg, monkeypatch, tmp_path):
    _stub_seps(monkeypatch, [])

    with pytest.raises(bgm_separation.BgmCancelled):
        asyncio.run(bgm_separation.separate(str(tmp_path / "in.wav"),
                                            cancel_check=lambda: True))

    assert bgm_separation._leases == {}


# --- same-id reload: live holders are not orphans ---------------------------
#
# A device/batch change re-keys the singleton under the SAME model id, so the
# orphaned pipeline and the freshly loaded one share a lease key. The live
# holder's release must be charged to `_leases` (and restamp the idle clock),
# not zero the orphan bucket and free the dying instance early.

def test_diarize_live_release_is_not_charged_to_a_same_id_orphan(
        diar_cfg, monkeypatch, caplog):
    _stub_pipes(monkeypatch, [])
    old = asyncio.run(diarization._get_pipeline("m1", lease=True))
    monkeypatch.setattr(diar_cfg, "DIARIZATION_EMBEDDING_BATCH_SIZE", 8,
                        raising=False)
    live = asyncio.run(diarization._get_pipeline("m1", lease=True))
    assert live is not old
    assert diarization._orphans == {"m1": 1}
    assert diarization._leases == {"m1": 1}

    diarization._last_used_monotonic = 0.0
    with caplog.at_level(logging.INFO, logger="whisper-server"):
        asyncio.run(diarization._release_pipeline("m1", live))

    assert diarization._orphans == {"m1": 1}      # the orphan is still draining
    assert diarization._leases == {}
    assert diarization._last_used_monotonic > 0.0  # idle clock restamped
    assert not any("unloaded" in r.getMessage() for r in caplog.records)
    assert diarization._pipeline is live

    asyncio.run(diarization._release_pipeline("m1", old))
    assert diarization._orphans == {}
    assert diarization._pipeline is live


def test_bgm_live_release_is_not_charged_to_a_same_model_orphan(
        bgm_cfg, monkeypatch, caplog):
    _stub_seps(monkeypatch, [])
    old = asyncio.run(bgm_separation._get_separator("Foo", lease=True))
    monkeypatch.setattr(bgm_cfg, "BGM_SEPARATION_DEVICE", "cuda", raising=False)
    live = asyncio.run(bgm_separation._get_separator("Foo", lease=True))
    assert live is not old
    assert bgm_separation._orphans == {"Foo.onnx": 1}
    assert bgm_separation._leases == {"Foo.onnx": 1}

    bgm_separation._last_used_monotonic = 0.0
    with caplog.at_level(logging.INFO, logger="whisper-server"):
        asyncio.run(bgm_separation._release_separator("Foo.onnx", live))

    assert bgm_separation._orphans == {"Foo.onnx": 1}
    assert bgm_separation._leases == {}
    assert bgm_separation._last_used_monotonic > 0.0
    assert not any("unloaded" in r.getMessage() for r in caplog.records)
    assert bgm_separation._separator is live

    asyncio.run(bgm_separation._release_separator("Foo.onnx", old))
    assert bgm_separation._orphans == {}
    assert bgm_separation._separator is live
