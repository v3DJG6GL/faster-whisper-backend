"""Tests for metrics ring-buffer math and snapshot shape.

Module globals are reset between tests by the autouse _reset_singletons
fixture in conftest.
"""

import time

import metrics


# ---------------------------------------------------------------------------
# _quantile (pure, nearest-rank)
# ---------------------------------------------------------------------------

def test_quantile_empty_is_zero():
    assert metrics._quantile([], 0.5) == 0.0


def test_quantile_single_element():
    assert metrics._quantile([7.0], 0.99) == 7.0


def test_quantile_p50_p95_p99():
    vals = [float(i) for i in range(1, 101)]  # 1..100 sorted
    assert metrics._quantile(vals, 0.50) == vals[int(round(0.50 * 99))]
    assert metrics._quantile(vals, 0.95) == vals[int(round(0.95 * 99))]
    assert metrics._quantile(vals, 0.99) == vals[int(round(0.99 * 99))]


def test_quantile_clamps_index():
    assert metrics._quantile([1.0, 2.0], 1.0) == 2.0
    assert metrics._quantile([1.0, 2.0], 0.0) == 1.0


# ---------------------------------------------------------------------------
# record_request
# ---------------------------------------------------------------------------

def test_record_request_counts_and_latency():
    metrics.record_request("/v1/models", 200, 12.5)
    assert metrics.req_count["/v1/models"] == 1
    assert metrics.err_count["/v1/models"] == 0
    assert list(metrics._latency) == [12.5]


def test_record_request_5xx_records_error():
    metrics.record_request("/x", 500, 5.0)
    assert metrics.err_count["/x"] == 1
    assert len(metrics._errors_ts) == 1


def test_record_request_4xx_not_error():
    metrics.record_request("/x", 404, 5.0)
    assert metrics.err_count["/x"] == 0
    assert len(metrics._errors_ts) == 0


def test_sse_paths_excluded_from_latency():
    for p in metrics.SSE_PATHS:
        metrics.record_request(p, 200, 999.0)
    assert len(metrics._latency) == 0


def test_error_window_prune():
    now = time.time()
    # Inject an old error timestamp beyond the 15-min window.
    metrics._errors_ts.append(now - (metrics._ERROR_WINDOW_SEC + 100))
    metrics.record_request("/x", 500, 1.0)
    # The stale ts must have been pruned, leaving only the fresh one.
    assert len(metrics._errors_ts) == 1
    assert metrics._errors_ts[0] >= now


def test_latency_ring_bounded():
    for i in range(metrics._LATENCY_MAX + 50):
        metrics.record_request("/x", 200, float(i))
    assert len(metrics._latency) == metrics._LATENCY_MAX


# ---------------------------------------------------------------------------
# _errors_in
# ---------------------------------------------------------------------------

def test_errors_in_window():
    now = time.time()
    # Append-ordered: oldest at the front, newest at the back.
    metrics._errors_ts.extend([now - 800, now - 120, now - 10])
    assert metrics._errors_in(60) == 1     # only the -10
    assert metrics._errors_in(300) == 2    # -10, -120
    assert metrics._errors_in(900) == 3    # all three


# ---------------------------------------------------------------------------
# record_model_load
# ---------------------------------------------------------------------------

def test_model_load_records():
    metrics.record_model_load("m", 3.0)
    assert metrics.model_loads["m"] == [3.0]


def test_model_load_bucket0_survives_trim():
    metrics.record_model_load("m", 1.0)  # canonical first
    for i in range(metrics._MODEL_LOAD_KEEP + 20):
        metrics.record_model_load("m", float(100 + i))
    bucket = metrics.model_loads["m"]
    assert len(bucket) == metrics._MODEL_LOAD_KEEP
    assert bucket[0] == 1.0  # first cold-load preserved forever


# ---------------------------------------------------------------------------
# record_transcription
# ---------------------------------------------------------------------------

def test_record_transcription_falsy_request_id_noop():
    # No request_id -> early return, no import/persist attempted, no raise.
    metrics.record_transcription("m", 1.0, 0.5, "ok", 3, request_id=None)
    metrics.record_transcription("m", 1.0, 0.5, "ok", 3, request_id="")


def test_record_transcription_swallows_store_errors():
    # Stores are not initialised here; the lazy record_timing/record_usage
    # calls raise RuntimeError internally but must be swallowed.
    metrics.record_transcription(
        "m", 1.0, 0.5, "ok", 3, request_id="req-1", user_id="u", key_id="k"
    )


# ---------------------------------------------------------------------------
# metrics_snapshot
# ---------------------------------------------------------------------------

def test_snapshot_shape_without_stores():
    metrics.record_request("/v1/models", 200, 10.0)
    metrics.record_request("/x", 500, 20.0)
    metrics.record_model_load("large-v2", 4.0)
    snap = metrics.metrics_snapshot()
    assert set(snap) >= {
        "uptime_sec", "in_flight_transcriptions", "requests", "errors_total",
        "errors_window", "latency_ms", "recent_transcriptions", "model_loads",
    }
    assert snap["requests"]["/v1/models"] == 1
    assert snap["errors_total"]["/x"] == 1
    assert snap["latency_ms"]["n"] == 2
    assert set(snap["errors_window"]) == {"1m", "5m", "15m"}
    assert snap["model_loads"]["large-v2"]["first"] == 4.0
    assert snap["model_loads"]["large-v2"]["count"] == 1
    # recent_transcriptions_store not init -> list_recent raises -> recent=[]
    assert snap["recent_transcriptions"] == []


def test_snapshot_in_flight_reflected():
    metrics.in_flight_transcriptions = 3
    assert metrics.metrics_snapshot()["in_flight_transcriptions"] == 3


def test_unmatched_route_keys_capped():
    """The unmatched-key cap (DoS guard): once req_count holds the cap's
    worth of keys, NEW unmatched paths fold into the overflow sentinel,
    while (a) an already-known unmatched path keeps its own counter and
    (b) a matched route always gets its own key, even past the cap."""
    for i in range(metrics._MAX_UNMATCHED_KEYS):
        metrics.record_request(f"/junk/{i}", 404, 1.0, unmatched=True)
    assert len(metrics.req_count) == metrics._MAX_UNMATCHED_KEYS
    assert metrics._UNMATCHED_OVERFLOW not in metrics.req_count

    # Past the cap: new unmatched paths share the sentinel...
    metrics.record_request("/junk/overflow-a", 404, 1.0, unmatched=True)
    metrics.record_request("/junk/overflow-b", 404, 1.0, unmatched=True)
    assert metrics.req_count[metrics._UNMATCHED_OVERFLOW] == 2
    assert "/junk/overflow-a" not in metrics.req_count

    # ...an existing unmatched key still increments under its own name...
    metrics.record_request("/junk/0", 404, 1.0, unmatched=True)
    assert metrics.req_count["/junk/0"] == 2

    # ...and a matched route is never folded.
    metrics.record_request("/v1/models", 200, 1.0)
    assert metrics.req_count["/v1/models"] == 1


# ---------------------------------------------------------------------------
# Recent-jobs projection (kind / username / key_label / stages)
# ---------------------------------------------------------------------------

def test_snapshot_projection_carries_job_fields(tx_store):
    stages = [{"name": "transcribing", "secs": 1.5, "model": "large-v3"}]
    metrics.record_transcription(
        "large-v3", 3.0, 1.5, "ok", 5, request_id="rj1", user_id="u1",
        stages=stages)
    tx_store.record_trace(request_id="rj1", model="large-v3", raw="x",
                          final="y", username="alice")
    metrics.record_transcription(
        "org/m:Q4", 0.0, 0.8, "ok", 0, request_id="rj2",
        kind="translate",
        stages=[{"name": "translate", "secs": 0.8, "model": "org/m:Q4",
                 "detail": "2 segs → en"}])
    recent = metrics.metrics_snapshot(
        include_identity=True)["recent_transcriptions"]
    by_model = {r["model"]: r for r in recent}
    tr = by_model["large-v3"]
    assert tr["kind"] == "transcribe"            # NULL kind + source 'file'
    assert tr["username"] == "alice"
    assert tr["stages"] == stages
    tl = by_model["org/m:Q4"]
    assert tl["kind"] == "translate"
    assert tl["stages"][0]["detail"] == "2 segs → en"
    # The projection still never carries transcript text.
    assert "raw" not in tr and "final" not in tr


def test_record_download_persists_a_download_row(tx_store):
    metrics.record_download(model="gguf:org/m:Q4", seconds=2.0,
                            bytes_done=4 * (1 << 30))
    recent = metrics.metrics_snapshot()["recent_transcriptions"]
    row = [r for r in recent if r["kind"] == "download"][0]
    assert row["model"] == "gguf:org/m:Q4"
    assert row["processing_s"] == 2.0
    assert row["stages"][0]["bytes"] == 4 * (1 << 30)
    assert "GB" in row["stages"][0]["detail"]


def test_snapshot_scrubs_identity_for_non_admin_viewers(tx_store):
    """A non-admin holder of pages.stats="all" sees every user's rows and
    must not read other users' username / key_label (same gate as
    jobs_snapshot); admins and own-scope viewers get them."""
    stages = [{"name": "transcribing", "secs": 1.5, "model": "large-v3"}]
    metrics.record_transcription(
        "large-v3", 3.0, 1.5, "ok", 5, request_id="rj1", user_id="u1",
        kind="transcribe", stages=stages, key_label="alice-laptop")
    tx_store.record_trace(request_id="rj1", model="large-v3", raw="x",
                          final="y", username="alice")
    row = metrics.metrics_snapshot()["recent_transcriptions"][0]
    assert row["username"] == "" and row["key_label"] == ""
    assert row["model"] == "large-v3" and row["kind"] == "transcribe"
    assert row["stages"] == stages
    admin = metrics.metrics_snapshot(
        include_identity=True)["recent_transcriptions"][0]
    assert admin["username"] == "alice"
    assert admin["key_label"] == "alice-laptop"


def test_record_transcription_takes_the_key_label_from_the_caller(
        tx_store, monkeypatch):
    # The auth record already holds the label; no per-request key lookup.
    import api_keys_store

    def _no_lookup(*_a, **_k):
        raise AssertionError("get_key was queried")
    monkeypatch.setattr(api_keys_store, "get_key", _no_lookup)
    metrics.record_transcription(
        "m", 1.0, 0.5, "ok", 3, request_id="rk1", key_id="k1",
        key_label="team-key")
    row = metrics.metrics_snapshot(
        include_identity=True)["recent_transcriptions"][0]
    assert row["key_label"] == "team-key"


def test_record_download_aborted_row_is_not_ok(tx_store):
    metrics.record_download(model="gguf:org/m:Q4", seconds=2.0,
                            bytes_done=1 << 30, status="error")
    row = [r for r in metrics.metrics_snapshot()["recent_transcriptions"]
           if r["kind"] == "download"][0]
    assert row["status"] == "error"
    assert row["stages"][0]["bytes"] == 1 << 30
    assert "aborted" in row["stages"][0]["detail"]


def test_cancelled_run_is_not_a_usage_error(usage_store_db):
    us = usage_store_db
    metrics.record_transcription("m", 1.0, 0.5, "cancelled", 0,
                                 request_id="c1", user_id="u", key_id="k")
    metrics.record_transcription("m", 1.0, 0.5, "error", 0,
                                 request_id="c2", user_id="u", key_id="k")
    r = us.totals_by_key()[0]
    assert r["requests"] == 2
    assert r["errors"] == 1        # the error, not the client cancel


def test_snapshot_user_filter_returns_only_that_users_rows(tx_store):
    """The /stats "own" page scope: `user_id=` narrows the recent rows to
    one owner (list_recent's indexed user_id_filter); None keeps them all."""
    metrics.record_transcription("m", 1.0, 0.5, "ok", 3, request_id="a1",
                                 user_id="alice")
    metrics.record_transcription("m", 2.0, 0.5, "ok", 3, request_id="b1",
                                 user_id="bob")
    metrics.record_transcription("m", 3.0, 0.5, "ok", 3, request_id="a2",
                                 user_id="alice")
    own = metrics.metrics_snapshot(user_id="alice")["recent_transcriptions"]
    assert sorted(r["audio_s"] for r in own) == [1.0, 3.0]
    assert len(metrics.metrics_snapshot()["recent_transcriptions"]) == 3
    assert metrics.metrics_snapshot(user_id="nobody")["recent_transcriptions"] == []


def test_record_transcription_fans_out_wait_and_error_class(tx_store, usage_store_db):
    """The v2 ledger columns reach both stores through the one call the
    handler makes, and the recent-jobs projection carries them."""
    metrics.record_transcription(
        "large-v3", 4.0, 2.0, "error", 0, request_id="e1", user_id="u1",
        key_id="k1", kind="transcribe", wait_s=1.25, error_class="cuda_oom",
        error_stage="transcribing")
    row = metrics.metrics_snapshot(include_identity=True)["recent_transcriptions"][0]
    assert (row["wait_s"], row["error_class"], row["error_stage"]) == (
        1.25, "cuda_oom", "transcribing")
    job = usage_store_db._require_conn().execute(
        "SELECT wait_s, error_class, error_stage, status FROM usage_jobs"
        " WHERE job_id = 'e1'").fetchone()
    assert tuple(job) == (1.25, "cuda_oom", "transcribing", "error")


# ---------------------------------------------------------------------------
# GpuGate: the timed inference semaphore
# ---------------------------------------------------------------------------

def test_gpu_gate_counts_held_queue_and_charges_the_wait():
    """Capacity 1: the second task waits, the snapshot shows it queued, and
    the wait is charged to THAT task's WAIT_ACC (seeded per request);
    releasing brings held back to 0."""
    import asyncio

    async def scenario():
        gate = metrics.GpuGate(1)
        seen = {}

        async def first():
            metrics.seed_wait()
            async with gate:
                seen["first_snap"] = gate.snapshot()
                await asyncio.sleep(0.05)
            seen["first_wait"] = metrics.take_wait()

        async def second():
            metrics.seed_wait()
            await asyncio.sleep(0.01)          # let `first` hold the slot
            async with gate:
                seen["second_snap"] = gate.snapshot()
            seen["second_wait"] = metrics.take_wait()

        await asyncio.gather(first(), second())
        seen["end"] = gate.snapshot()
        return seen

    seen = asyncio.run(scenario())
    assert seen["first_snap"]["held"] == 1 and seen["first_snap"]["capacity"] == 1
    assert seen["first_wait"] == 0.0
    assert seen["second_wait"] >= 0.03
    assert seen["second_snap"]["queue_depth"] == 0
    assert seen["end"] == {"capacity": 1, "held": 0, "queue_depth": 0,
                           "oldest_wait_s": 0.0}


def test_gpu_gate_snapshot_reports_the_oldest_waiter():
    import asyncio

    async def scenario():
        gate = metrics.GpuGate(1)
        await gate.acquire()
        waiter = asyncio.create_task(gate.acquire())
        await asyncio.sleep(0.05)
        snap = gate.snapshot()
        gate.release()
        await waiter
        gate.release()
        return snap, gate.snapshot()

    snap, after = asyncio.run(scenario())
    assert snap["queue_depth"] == 1 and snap["oldest_wait_s"] >= 0.0
    assert after["queue_depth"] == 0 and after["held"] == 0


def test_take_wait_without_a_seed_is_zero_and_take_resets():
    import asyncio
    assert metrics.take_wait() == 0.0

    async def scenario():
        gate = metrics.GpuGate(2)
        metrics.seed_wait()
        async with gate:
            pass
        first = metrics.take_wait()
        second = metrics.take_wait()
        return first, second

    first, second = asyncio.run(scenario())
    assert first >= 0.0 and second == 0.0


def test_metrics_snapshot_carries_the_gate(tx_store):
    snap = metrics.metrics_snapshot()
    assert set(snap["gpu_gate"]) == {"capacity", "held", "queue_depth", "oldest_wait_s"}
