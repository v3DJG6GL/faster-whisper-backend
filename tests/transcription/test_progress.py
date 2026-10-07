"""transcription/progress — the closed-id tombstone that keeps a straggling
stage thread from re-creating a closed run's entry owner-less."""

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
