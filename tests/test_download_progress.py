"""Unit tests for download_progress.py — the huggingface_hub tqdm shim and
its capture scope. No network: bars are constructed and updated directly,
exactly as hub's download loop drives them."""

import importlib
import logging

import pytest

import download_progress as dp

pytest.importorskip("huggingface_hub")


def _mk_bar(**kw):
    kw.setdefault("total", 100)
    kw.setdefault("unit", "B")
    kw.setdefault("disable", True)   # server norm: bars globally disabled
    kw.setdefault("name", "huggingface_hub.http")   # hub passes it; pop-safe
    return dp.ReportingTqdm(**kw)


def test_reporting_tqdm_counts_bytes_even_when_disabled(monkeypatch):
    monkeypatch.setattr(dp, "_CB_MIN_INTERVAL_S", 0.0)
    calls = []
    with dp.capture("gguf:org/m", cb=lambda d, t: calls.append((d, t)),
                    record=False):
        bar = _mk_bar()
        bar.update(30)
        bar.update(70)
    assert calls[-1] == (100, 100)
    assert calls[0][0] <= 100


def test_non_byte_bars_are_ignored(monkeypatch):
    monkeypatch.setattr(dp, "_CB_MIN_INTERVAL_S", 0.0)
    calls = []
    with dp.capture("x", cb=lambda d, t: calls.append((d, t)), record=False):
        bar = _mk_bar(unit="it", total=5)     # snapshot's per-FILE bar
        bar.update(5)
    assert calls == []


def test_capture_aggregates_multiple_bars(monkeypatch):
    monkeypatch.setattr(dp, "_CB_MIN_INTERVAL_S", 0.0)
    calls = []
    with dp.capture("x", cb=lambda d, t: calls.append((d, t)), record=False):
        a = _mk_bar(total=100)
        b = _mk_bar(total=300)
        a.update(100)
        b.update(200)
    assert calls[-1] == (300, 400)


def test_capture_patches_and_restores_hub_tqdm():
    # utils.tqdm.tqdm is the LIVE target (file_download builds its bar
    # through it); the assertion must be on that one, or dropping it from
    # _PATCH_TARGETS would go unnoticed.
    ut = importlib.import_module("huggingface_hub.utils.tqdm")
    before = ut.tqdm
    targets = [(importlib.import_module(m), a) for m, a in dp._PATCH_TARGETS]
    befores = [getattr(mod, attr, None) for mod, attr in targets]
    with dp.capture("x", record=False):
        assert ut.tqdm is dp.ReportingTqdm
    assert ut.tqdm is before
    # ...and every module the tuple names is restored, whatever it holds.
    for (mod, attr), prev in zip(targets, befores):
        assert getattr(mod, attr, None) is prev


def test_capture_logs_buckets_and_done(monkeypatch, caplog):
    monkeypatch.setattr(dp, "_CB_MIN_INTERVAL_S", 0.0)
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        with dp.capture("gguf:org/m", record=False):
            bar = _mk_bar(total=1000)
            for _ in range(100):
                bar.update(10)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("[download] gguf:org/m started" in m for m in msgs)
    assert any("100%" in m for m in msgs)
    assert any("done:" in m for m in msgs)
    # 10%-bucket throttle: 100 update() calls, but only a line per crossed
    # 10% bucket — unthrottled logging would emit ~100.
    assert len([m for m in msgs if "%" in m and "done" not in m]) <= 12


def test_aggregate_survives_bar_id_reuse(monkeypatch):
    # A finished bar is freed and CPython hands its address to the next one;
    # keying on id(bar) would overwrite the done file's bytes.
    monkeypatch.setattr(dp, "_CB_MIN_INTERVAL_S", 0.0)
    calls = []
    with dp.capture("x", cb=lambda d, t: calls.append((d, t)), record=False):
        a = _mk_bar(total=100)
        a.update(100)
        del a
        b = _mk_bar(total=300)
        b.update(200)
    assert calls[-1] == (300, 400)


def test_warm_cache_stays_silent(caplog):
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        with dp.capture("x", record=False):
            pass                                  # no bars, no bytes
    assert not [r for r in caplog.records
                if "[download]" in r.getMessage()]


def test_cb_failures_never_escape(monkeypatch):
    monkeypatch.setattr(dp, "_CB_MIN_INTERVAL_S", 0.0)

    def boom(d, t):
        raise RuntimeError("cb exploded")
    with dp.capture("x", cb=boom, record=False):
        bar = _mk_bar()
        bar.update(100)                           # must not raise


def test_snapshot_transfer_twin_bar_is_counted_once(monkeypatch):
    # snapshot_download drives TWO byte-unit aggregate bars from tqdm_class
    # over the same bytes (reconstruct + ".transfer"); only one may count.
    monkeypatch.setattr(dp, "_CB_MIN_INTERVAL_S", 0.0)
    calls = []
    with dp.capture("whisper:large-v3", cb=lambda d, t: calls.append((d, t)),
                    record=False) as cap:
        rec = _mk_bar(total=1000, name="huggingface_hub.snapshot_download")
        xfer = _mk_bar(total=1000,
                       name="huggingface_hub.snapshot_download.transfer")
        rec.update(400)
        xfer.update(400)
        assert cap.totals() == (400, 1000)
    assert calls[-1] == (400, 1000)


def test_callback_never_walks_backwards(monkeypatch):
    # bump() aggregates under the lock but delivers outside it; a stale
    # delivery from a preempted worker must be dropped, not sent.
    monkeypatch.setattr(dp, "_CB_MIN_INTERVAL_S", 0.0)
    calls = []
    cap = dp._Capture("x", cb=lambda d, t: calls.append((d, t)))
    cap.add_bar(1, 1000)
    cap.bump(1, 800, 1000)
    cap.bump(1, 500, 1000)      # a stale, smaller aggregate
    cap.bump(1, 900, 1000)
    dones = [d for d, _ in calls]
    assert dones == sorted(dones)
    assert 500 not in dones


def test_failed_download_gets_no_done_receipt(monkeypatch, caplog):
    monkeypatch.setattr(dp, "_CB_MIN_INTERVAL_S", 0.0)
    calls = []
    with caplog.at_level(logging.INFO, logger="whisper-api"):
        with pytest.raises(RuntimeError):
            with dp.capture("x", cb=lambda d, t: calls.append((d, t)),
                            record=False):
                bar = _mk_bar(total=100)
                bar.update(50)
                raise RuntimeError("boom")
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("done:" in m for m in msgs)
    assert any("aborted" in m and "[download] x" in m for m in msgs)
    # No completion-shaped (done, total or done) pair after the failure.
    assert calls[-1] == (50, 100)
    assert (50, 50) not in calls
    assert (100, 100) not in calls
    assert dp._active_capture() is None       # scope still torn down


def test_failed_download_is_recorded_as_not_ok(monkeypatch):
    import metrics
    seen = []
    monkeypatch.setattr(metrics, "record_download",
                        lambda **kw: seen.append(kw))
    with pytest.raises(RuntimeError):
        with dp.capture("gguf:org/m", record=True):
            _mk_bar(total=100).update(60)
            raise RuntimeError("network drop")
    assert len(seen) == 1
    assert seen[0]["model"] == "gguf:org/m"
    assert seen[0]["bytes_done"] == 60
    assert seen[0]["status"] != "ok"


def test_finished_download_is_recorded_as_ok(monkeypatch):
    import metrics
    seen = []
    monkeypatch.setattr(metrics, "record_download",
                        lambda **kw: seen.append(kw))
    with dp.capture("gguf:org/m", record=True):
        _mk_bar(total=100).update(100)
    assert len(seen) == 1
    assert seen[0]["status"] == "ok"
    assert seen[0]["bytes_done"] == 100
