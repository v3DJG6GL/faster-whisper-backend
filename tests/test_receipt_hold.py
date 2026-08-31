"""Held dictation receipts.

A dictation's translation runs as a SEPARATE request after the stream
finalizes, so its receipt is parked until that request lands and then logged
as one block. The two properties that matter:

  * a held receipt is never lost — every exit releases it exactly once, with
    a note saying why it is incomplete;
  * the timeout is an IDLE timer, not a deadline, so a genuinely slow cold
    model load is never the thing that splits a receipt in half.
"""

import time

import pytest

import receipt_hold


@pytest.fixture(autouse=True)
def _clean():
    receipt_hold._reset_for_tests()
    yield
    receipt_hold._reset_for_tests()


def _kwargs(n=1):
    return {"file_label": f"utt#{n}", "model_name": "m", "raw": "r",
            "final": "f", "seg_diag": [], "kwargs": {}}


# ---------------------------------------------------------------------------
# Claim
# ---------------------------------------------------------------------------

def test_claim_returns_the_parked_kwargs_merged(_clean=None):
    receipt_hold.park("cap1", _kwargs(), hold_s=90)
    got = receipt_hold.claim("cap1", translation={"targets": ["en"]})
    assert got is not None
    assert got["file_label"] == "utt#1"
    assert got["translation"] == {"targets": ["en"]}
    # Claiming is a take: a second claim finds nothing.
    assert receipt_hold.claim("cap1") is None


def test_claim_of_an_unheld_key_is_not_an_error():
    """Every non-dictation caller hits this path — a batch translate, a
    subtitle file, the viewer's retro-translate. It must be silent."""
    assert receipt_hold.claim("nope") is None
    assert receipt_hold.claim("") is None


def test_claim_ignores_none_values_so_a_caller_can_pass_optionals():
    receipt_hold.park("cap1", _kwargs(), hold_s=90)
    got = receipt_hold.claim("cap1", translation=None, stages=[{"n": 1}])
    assert "translation" not in got
    assert got["stages"] == [{"n": 1}]


# ---------------------------------------------------------------------------
# Idle timer, not a deadline
# ---------------------------------------------------------------------------

def test_sweep_releases_only_after_silence():
    receipt_hold.park("cap1", _kwargs(), hold_s=0.05)
    assert receipt_hold.sweep() == []          # still fresh
    time.sleep(0.08)
    out = receipt_hold.sweep()
    assert len(out) == 1
    assert "no result within" in " ".join(out[0]["skipped"])
    assert receipt_hold.pending() == 0


def test_a_heartbeat_keeps_a_slow_translation_alive_past_the_hold():
    """The whole reason the hold can be as short as 90 s: a cold GGUF load
    that takes minutes keeps reporting, so it keeps its receipt."""
    receipt_hold.park("cap1", _kwargs(), hold_s=0.06)
    for _ in range(4):
        time.sleep(0.03)
        assert receipt_hold.touch("cap1") is True
        assert receipt_hold.sweep() == []
    assert receipt_hold.pending() == 1
    # Stop reporting → released on schedule.
    time.sleep(0.09)
    assert len(receipt_hold.sweep()) == 1


def test_touch_of_an_unheld_key_reports_false():
    assert receipt_hold.touch("nope") is False


# ---------------------------------------------------------------------------
# Nothing is ever silently lost
# ---------------------------------------------------------------------------

def test_release_returns_the_receipt_with_a_reason():
    receipt_hold.park("cap1", _kwargs(), hold_s=90)
    out = receipt_hold.release("cap1", "cancelled by client after 4.2s")
    assert out is not None
    assert any("cancelled by client" in s for s in out["skipped"])
    assert receipt_hold.pending() == 0


def test_release_preserves_existing_notes():
    kw = _kwargs()
    kw["skipped"] = ["diarizing"]
    receipt_hold.park("cap1", kw, hold_s=90)
    out = receipt_hold.release("cap1", "failed")
    assert out["skipped"][0] == "diarizing"
    assert "failed" in out["skipped"][1]


def test_release_after_a_claim_is_a_no_op():
    """The translate handler's `finally` is a catch-all release for the paths
    no `except` arm covers (a dropped connection raises CancelledError, a
    BaseException). It runs on the success path too, where the receipt has
    already been claimed — so a release there must not manufacture a second,
    contradictory copy of the same receipt."""
    receipt_hold.park("cap1", _kwargs(), hold_s=90)
    assert receipt_hold.claim("cap1") is not None
    assert receipt_hold.release("cap1", "connection closed after 3.0s") is None
    assert receipt_hold.sweep() == []
    # And releasing twice hands the receipt back exactly once.
    receipt_hold.park("cap2", _kwargs(2), hold_s=90)
    assert receipt_hold.release("cap2", "connection closed after 3.0s") is not None
    assert receipt_hold.release("cap2", "connection closed after 3.0s") is None


def test_flush_all_drains_everything_for_shutdown():
    for i in range(3):
        receipt_hold.park(f"cap{i}", _kwargs(i), hold_s=90)
    out = receipt_hold.flush_all()
    assert len(out) == 3
    assert receipt_hold.pending() == 0
    assert all(any("shutdown" in s for s in e["skipped"]) for e in out)


def test_the_cap_evicts_but_never_discards():
    """An eviction is a reason to LOG the receipt, not to forget it — the
    next sweep has to hand it back."""
    for i in range(receipt_hold._MAX_HELD + 2):
        receipt_hold.park(f"cap{i}", _kwargs(i), hold_s=90)
        time.sleep(0.001)      # distinct touch stamps, so the victim is defined
    assert receipt_hold.pending() == receipt_hold._MAX_HELD
    out = receipt_hold.sweep()
    assert len(out) == 2
    assert all(any("buffer full" in s for s in e["skipped"]) for e in out)


def test_eviction_takes_the_least_recently_touched_not_the_oldest():
    """A long, actively-progressing translation must outlive a stalled one."""
    receipt_hold.park("old-but-busy", _kwargs(), hold_s=90)
    for i in range(receipt_hold._MAX_HELD - 1):
        time.sleep(0.001)
        receipt_hold.park(f"cap{i}", _kwargs(i), hold_s=90)
    time.sleep(0.002)
    receipt_hold.touch("old-but-busy")          # still reporting
    receipt_hold.park("one-too-many", _kwargs(99), hold_s=90)

    assert receipt_hold.claim("old-but-busy") is not None
    evicted = receipt_hold.sweep()
    assert any("buffer full" in s
               for e in evicted for s in e.get("skipped", []))


def test_park_without_a_key_is_a_no_op():
    receipt_hold.park("", _kwargs(), hold_s=90)
    assert receipt_hold.pending() == 0
