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
    fd = importlib.import_module("huggingface_hub.file_download")
    before = fd.tqdm
    with dp.capture("x", record=False):
        assert fd.tqdm is dp.ReportingTqdm
    assert fd.tqdm is before


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
