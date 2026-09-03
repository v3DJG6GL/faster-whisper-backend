"""stats_sampler: the 1 Hz busy ring, the sampled machine history and its
flush / prune, and the loop's off-thread contract."""

import asyncio
import inspect

import pytest

import metrics
import stats_sampler
import system_stats


@pytest.fixture
def sampler(tx_store, monkeypatch):
    stats_sampler._reset_for_tests()
    monkeypatch.setattr(metrics, "gpu_gate", None)
    monkeypatch.setattr(metrics, "in_flight_transcriptions", 0)
    monkeypatch.setattr(system_stats, "_build_gpu",
                        lambda: {"util_pct": 42.0, "mem_used_mb": 1000.0, "temp_c": 60.0})
    monkeypatch.setattr(system_stats, "_build_host",
                        lambda: {"cpu_pct": 12.5, "ram_pct": 33.0})
    yield stats_sampler
    stats_sampler._reset_for_tests()


def test_busy_ring_is_bounded_and_windows_are_shares(sampler, monkeypatch):
    gate = metrics.GpuGate(2)
    monkeypatch.setattr(metrics, "gpu_gate", gate)
    gate.held = 1
    for i in range(600):
        sampler.tick(1_000_000 + i)
    gate.held = 0
    for i in range(600, 1200):
        sampler.tick(1_000_000 + i)
    assert len(metrics.busy_ring) == 900
    snap = metrics.slot_busy_snapshot()
    assert snap["samples"] == 900
    assert snap["pct_1m"] == 0.0 and snap["pct_5m"] == 0.0
    # 900-window = seconds 300..1199: 300 busy of 900.
    assert snap["pct_15m"] == pytest.approx(33.3, abs=0.1)


def test_gate_absent_falls_back_to_in_flight(sampler, monkeypatch):
    monkeypatch.setattr(metrics, "in_flight_transcriptions", 2)
    assert sampler.slot_busy_now() == 1
    monkeypatch.setattr(metrics, "in_flight_transcriptions", 0)
    assert sampler.slot_busy_now() == 0


def test_every_nth_tick_takes_a_sample_on_the_grid(sampler, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "STATS_SYSTEM_METRICS_SAMPLE_S", 10, raising=False)
    taken = [sampler.tick(2_000_003 + i) for i in range(25)]
    samples = [s for s in taken if s]
    assert len(samples) == 2
    assert samples[0]["ts"] % 10 == 0
    assert samples[0]["gpu_util"] == 42.0 and samples[0]["cpu_pct"] == 12.5
    assert samples[0]["slot_busy"] == 0.0
    assert len(sampler._pending) == 2


def test_flush_writes_once_and_prune_drops_old(sampler, sm_store, app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "STATS_SYSTEM_METRICS_SAMPLE_S", 10, raising=False)
    monkeypatch.setattr(app_module.cfg, "STATS_SYSTEM_METRICS_RETENTION_DAYS", 30, raising=False)
    import time
    now = int(time.time())
    for i in range(30):
        sampler.tick(now - 300 + i)
    assert sampler.flush() == 3
    assert sampler.flush() == 0
    series = sm_store.list_series(metric="gpu_util", from_ts=now - 400,
                                  to_ts=now + 1, step_s=10)
    assert len(series["t"]) == 3 and series["avg"][0] == 42.0
    sm_store.record([{"ts": now - 40 * 86400, "gpu_util": 1.0}])
    assert sampler.prune() == 1
    monkeypatch.setattr(app_module.cfg, "STATS_SYSTEM_METRICS_RETENTION_DAYS", 0, raising=False)
    assert sampler.prune() == 0


def test_nvml_failure_yields_none_fields_not_an_exception(sampler, monkeypatch):
    def _boom():
        raise RuntimeError("NVML down")
    monkeypatch.setattr(system_stats, "_build_gpu", _boom)
    s = sampler.sample(3_000_000)
    assert s["gpu_util"] is None and s["cpu_pct"] == 12.5


def test_loop_runs_sampling_off_the_event_loop():
    src = inspect.getsource(stats_sampler.loop)
    assert "await asyncio.to_thread(tick, now)" in src
    assert "await asyncio.to_thread(flush)" in src
    assert "await asyncio.to_thread(prune)" in src
    assert "except asyncio.CancelledError:" in src


def test_lifespan_starts_and_stops_the_sampler():
    src = open("main.py", encoding="utf-8").read()
    assert "stats_sampler_task = asyncio.create_task(_stats_sampler.loop())" in src
    assert "await _cancel(stats_sampler_task)" in src
    assert "asyncio.to_thread(_stats_sampler.flush)" in src
