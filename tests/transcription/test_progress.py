"""transcription/progress — no path (closed-id tombstone, stale sweep, cap
eviction) lets a straggling tick re-create a run's entry owner-less."""

import time

from faster_whisper_backend.transcription import progress as tx_progress

_PID = "c105ed" * 6


def test_tombstone_outlives_the_old_ttl_under_many_closes():
    """A decode that keeps ticking long after its handler closed must not
    re-create the entry owner-less (readable and cancellable by anyone,
    carrying the transcript tail) once 64+ other runs closed."""
    tx_progress._progress_set(_PID, stage="transcribing", owner="alice")
    tx_progress._progress_close(_PID)
    tx_progress._PROGRESS_CLOSED[_PID] = time.monotonic() - 700
    for i in range(70):
        tx_progress._progress_close(f"{i:032x}")
    tx_progress._progress_set(_PID, last_text="x")
    assert _PID not in tx_progress._BATCH_PROGRESS


def test_a_suppressed_tick_keeps_its_tombstone_fresh():
    tx_progress._progress_set(_PID, stage="transcribing", owner="alice")
    tx_progress._progress_close(_PID)
    old = time.monotonic() - tx_progress._PROGRESS_CLOSED_TTL_S - 1
    tx_progress._PROGRESS_CLOSED[_PID] = old
    tx_progress._progress_set(_PID, last_text="x")
    assert tx_progress._PROGRESS_CLOSED[_PID] > old
    for i in range(70):
        tx_progress._progress_close(f"{i:032x}")
    assert _PID in tx_progress._PROGRESS_CLOSED


def test_a_close_racing_a_tick_leaves_no_owner_less_entry(monkeypatch):
    """The loop-thread close can land between the closed check at the top
    of _progress_set and the entry creation (the tick runs job_update on its
    executor thread in between): the tick must still be a no-op."""
    tx_progress._progress_set(_PID, stage="transcribing", owner="alice")
    tx_progress._JOB_BY_PID[_PID] = "job"
    monkeypatch.setattr(tx_progress.jobs, "job_update",
                        lambda *a, **kw: tx_progress._progress_close(_PID))
    tx_progress._progress_set(_PID, stage="transcribing", last_text="x")
    assert _PID not in tx_progress._BATCH_PROGRESS
    # A fresh owner-stamped seed still re-opens the id.
    monkeypatch.setattr(tx_progress.jobs, "job_update", lambda *a, **kw: None)
    tx_progress._progress_set(_PID, stage="waiting", owner="bob")
    assert tx_progress._BATCH_PROGRESS[_PID]["owner"] == "bob"


def test_a_tick_inside_the_close_leaves_no_owner_less_entry(monkeypatch):
    """A stage-thread tick can run while the loop thread is half-way through
    _progress_close (registries popped, tombstone not yet written): it must
    not re-create the entry owner-less and readable by any caller."""

    class _TickingSet(set):
        def discard(self, item):
            tx_progress._progress_set(_PID, last_text="x")
            super().discard(item)

    tx_progress._progress_set(_PID, stage="transcribing", owner="alice")
    monkeypatch.setattr(tx_progress, "_BATCH_CANCELLED", _TickingSet())
    tx_progress._progress_close(_PID)
    assert _PID not in tx_progress._BATCH_PROGRESS
    assert tx_progress._progress_entry_for(_PID, {"user_id": "mallory"}) is None


def test_a_close_between_the_create_check_and_the_create_is_undone(monkeypatch):
    """A tick of a cap-evicted run that passed the closed check right before
    the entry creation, with the close landing in between: the created entry
    is taken back, never left behind after the run closed."""
    tx_progress._progress_set(_PID, stage="transcribing", owner="alice")
    tx_progress._BATCH_PROGRESS.pop(_PID)  # the cap eviction
    real_monotonic = time.monotonic
    closed = []

    def _monotonic():
        # The first clock read of the tick is the creation sweep's `now`,
        # after the closed check: close the run right there.
        if not closed:
            closed.append(True)
            tx_progress._progress_close(_PID)
        return real_monotonic()

    monkeypatch.setattr(tx_progress, "time", type("T", (), {
        "monotonic": staticmethod(_monotonic)}))
    tx_progress._progress_set(_PID, last_text="x")
    assert closed
    assert _PID not in tx_progress._BATCH_PROGRESS
    assert _PID in tx_progress._PROGRESS_CLOSED


def test_the_stale_sweep_spares_an_owner_stamped_run():
    """A batch run waiting on the GPU gate ticks nothing for as long as the
    queue holds it: the stale sweep must not pop it (404 to its owner) and
    let its next tick re-create the entry owner-less. An owner-less stale
    entry (an invented id) is still swept."""
    _stray = "5a1e" * 8
    tx_progress._progress_set(_PID, stage="waiting", owner="u1")
    tx_progress._progress_set(_stray, stage="waiting")
    _old = time.monotonic() - tx_progress._BATCH_PROGRESS_STALE_S - 1
    tx_progress._BATCH_PROGRESS[_PID]["updated"] = _old
    tx_progress._BATCH_PROGRESS[_stray]["updated"] = _old
    tx_progress._progress_set("0f" * 16, stage="waiting", owner="u2")
    assert _stray not in tx_progress._BATCH_PROGRESS
    tx_progress._progress_set(_PID, stage="analyzing")
    assert tx_progress._BATCH_PROGRESS[_PID]["owner"] == "u1"
    assert tx_progress._progress_entry_for(
        _PID, {"user_id": "u1"})["stage"] == "analyzing"
    assert tx_progress._progress_entry_for(_PID, {"user_id": "mallory"}) is None


def test_a_cap_evicted_run_still_holds_its_progress_id(monkeypatch):
    """The cap eviction pops a live run's entry but keeps its owner stamp
    (its next tick re-creates the entry). A second request posting the same
    id in that gap must not claim it: its seed would re-stamp the owner and
    both runs would share one entry and cancel flag."""
    monkeypatch.setattr(tx_progress, "_BATCH_PROGRESS_MAX", 2)
    tx_progress._progress_set(_PID, stage="transcribing", owner="alice")
    tx_progress._progress_set("a1" * 16, stage="waiting", owner="u1")
    tx_progress._progress_set("b2" * 16, stage="waiting", owner="u2")
    assert _PID not in tx_progress._BATCH_PROGRESS
    assert tx_progress._claim_progress_id(_PID) is None
    tx_progress._progress_set(_PID, progress=0.5, last_text="x")
    assert tx_progress._BATCH_PROGRESS[_PID]["owner"] == "alice"
    # Once the run closes, the id is free again.
    tx_progress._progress_close(_PID)
    assert tx_progress._claim_progress_id(_PID) == _PID


def test_cap_evicted_live_run_stays_visible_and_cancellable_to_its_owner(
        monkeypatch):
    """Another identity's fresh ids can cap-evict a run that is still queued
    for the GPU: its owner must keep seeing it ("unknown" tells a poller to
    stop) and keep cancelling it, while a foreign caller still sees a miss."""
    monkeypatch.setattr(tx_progress, "_BATCH_PROGRESS_MAX", 2)
    tx_progress._progress_set(_PID, stage="waiting", owner="alice")
    tx_progress._progress_set("a1" * 16, stage="waiting", owner="mallory")
    tx_progress._progress_set("b2" * 16, stage="waiting", owner="mallory")
    assert _PID not in tx_progress._BATCH_PROGRESS
    entry = tx_progress._progress_entry_for(_PID, {"user_id": "alice"})
    assert entry is not None and entry["stage"] == "waiting"
    assert tx_progress._progress_entry_for(_PID, {"user_id": "mallory"}) is None
    assert tx_progress._progress_entry_for(
        _PID, {"user_id": "root", "is_admin": True}) is not None
    # The synthesised view is not stored: the next tick re-creates the entry.
    assert _PID not in tx_progress._BATCH_PROGRESS
    tx_progress._progress_close(_PID)
    assert tx_progress._progress_entry_for(_PID, {"user_id": "alice"}) is None


class _ScanCountingDict(dict):
    scans = 0

    def items(self):
        type(self).scans += 1
        return super().items()


def test_tombstone_sweep_is_amortised(monkeypatch):
    """Past 64 tombstones a close must not rescan all of them every time: a
    flood of fresh ids would make every close O(n) for the 2 h TTL."""
    closed = _ScanCountingDict()
    old = time.monotonic() - tx_progress._PROGRESS_CLOSED_TTL_S - 1
    for i in range(100):
        closed[f"e{i:031x}"] = old
    monkeypatch.setattr(tx_progress, "_PROGRESS_CLOSED", closed)
    _ScanCountingDict.scans = 0
    for i in range(70):
        tx_progress._progress_close(f"{i:032x}")
    assert _ScanCountingDict.scans == 1
    assert not any(k.startswith("e") for k in closed)


def test_tombstones_are_capped_but_a_ticking_straggler_survives(monkeypatch):
    """A flood of fresh ids must not grow the tombstones for the whole TTL;
    the cap evicts the oldest stamps, never one a straggler just restamped."""
    monkeypatch.setattr(tx_progress, "_PROGRESS_CLOSED_MAX", 50)
    monkeypatch.setattr(tx_progress, "_PROGRESS_CLOSED_LAST_SWEEP",
                        time.monotonic() - 61)
    tx_progress._progress_set(_PID, stage="transcribing", owner="alice")
    tx_progress._progress_close(_PID)
    tx_progress._PROGRESS_CLOSED[_PID] = time.monotonic()   # just ticked
    old = time.monotonic() - 600
    for i in range(100):
        tx_progress._PROGRESS_CLOSED[f"e{i:031x}"] = old
    tx_progress._progress_close("f" * 32)
    assert len(tx_progress._PROGRESS_CLOSED) <= 52
    assert _PID in tx_progress._PROGRESS_CLOSED
    tx_progress._progress_set(_PID, last_text="x")
    assert _PID not in tx_progress._BATCH_PROGRESS


def test_expired_tombstones_still_swept(monkeypatch):
    monkeypatch.setattr(tx_progress, "_PROGRESS_CLOSED_LAST_SWEEP",
                        time.monotonic() - 61)
    old = time.monotonic() - tx_progress._PROGRESS_CLOSED_TTL_S - 1
    for i in range(70):
        tx_progress._PROGRESS_CLOSED[f"e{i:031x}"] = old
    tx_progress._progress_close(_PID)
    assert list(tx_progress._PROGRESS_CLOSED) == [_PID]
