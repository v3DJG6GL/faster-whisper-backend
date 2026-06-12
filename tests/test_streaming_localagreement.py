"""Unit tests for the pure LocalAgreement-2 engine (streaming_localagreement.py).

No audio, no model — these feed hand-built hypothesis sequences and assert the
commit/provisional split, the load-bearing invariants (pop-from-both, empty first
iteration, boundary dedup), and that finish() does not reset.
"""

from streaming_localagreement import HypothesisBuffer, LocalAgreementProcessor, TSWord


def _texts(words):
    return [w.text for w in words]


def test_first_iteration_commits_nothing():
    """LA-2 needs two hypotheses to agree, so the first decode commits nothing."""
    p = LocalAgreementProcessor()
    p.insert_hypothesis([(0.0, 0.5, " Der"), (0.5, 1.0, " Patient")], 0.0)
    assert p.commit() == []
    assert _texts(p.provisional()) == [" Der", " Patient"]
    assert p.committed_text == ""


def test_commits_common_prefix_of_two_hypotheses():
    """The prefix two consecutive hypotheses agree on is committed; the fresh tail
    stays provisional."""
    p = LocalAgreementProcessor()
    p.insert_hypothesis([(0.0, 0.5, " Der"), (0.5, 1.0, " Patient")], 0.0)
    p.commit()
    p.insert_hypothesis(
        [(0.0, 0.5, " Der"), (0.5, 1.0, " Patient"), (1.0, 1.5, " hat")], 0.0
    )
    delta = p.commit()
    assert _texts(delta) == [" Der", " Patient"]
    assert _texts(p.provisional()) == [" hat"]
    assert p.committed_text == " Der Patient"


def test_unstable_tail_is_never_committed_until_agreement():
    """A mis-heard tail word that changes between passes is never committed; the
    corrected word only commits once two passes agree on it."""
    p = LocalAgreementProcessor()
    # Build up a committed prefix "Der Patient hat".
    p.insert_hypothesis([(0.0, 0.5, " Der"), (0.5, 1.0, " Patient")], 0.0)
    p.commit()
    p.insert_hypothesis(
        [(0.0, 0.5, " Der"), (0.5, 1.0, " Patient"), (1.0, 1.5, " hat")], 0.0
    )
    p.commit()
    p.insert_hypothesis(
        [(0.0, 0.5, " Der"), (0.5, 1.0, " Patient"), (1.0, 1.5, " hat"),
         (1.5, 2.0, " Vieber")], 0.0,
    )
    p.commit()  # commits "hat"; "Vieber" only provisional
    assert "Vieber" not in p.committed_text

    # Next pass corrects the tail to "Fieber" — disagrees with "Vieber", so nothing
    # new commits yet.
    p.insert_hypothesis(
        [(0.0, 0.5, " Der"), (0.5, 1.0, " Patient"), (1.0, 1.5, " hat"),
         (1.5, 2.0, " Fieber")], 0.0,
    )
    assert _texts(p.commit()) == []
    assert _texts(p.provisional()) == [" Fieber"]

    # A second agreeing pass finally commits "Fieber"; "Vieber" never made it in.
    p.insert_hypothesis(
        [(0.0, 0.5, " Der"), (0.5, 1.0, " Patient"), (1.0, 1.5, " hat"),
         (1.5, 2.0, " Fieber")], 0.0,
    )
    assert _texts(p.commit()) == [" Fieber"]
    assert p.committed_text == " Der Patient hat Fieber"
    assert "Vieber" not in p.committed_text


def test_boundary_ngram_dedup_prevents_double_emission():
    """Whisper often re-emits the last committed word at the front of a fresh
    decode (e.g. after a buffer trim). The n-gram guard drops it so it is not
    committed twice."""
    p = LocalAgreementProcessor()
    p.insert_hypothesis([(0.0, 0.5, " Der"), (0.5, 1.0, " Patient")], 0.0)
    p.commit()
    p.insert_hypothesis(
        [(0.0, 0.5, " Der"), (0.5, 1.0, " Patient"), (1.0, 1.5, " hat")], 0.0
    )
    p.commit()  # committed "Der Patient", provisional "hat"

    # A new hypothesis that re-emits "Patient" at the seam (start≈1.0).
    p.insert_hypothesis([(1.0, 1.5, " Patient"), (1.5, 2.0, " hat")], 0.0)
    p.commit()
    # "Patient" appears exactly once; "hat" committed.
    assert p.committed_text == " Der Patient hat"
    assert p.committed_text.count("Patient") == 1


def test_pop_committed_trims_in_buffer_window():
    """After an audio trim, pop_committed drops committed words ending before the
    cut so the boundary dedup window stays bounded."""
    buf = HypothesisBuffer()
    buf.insert([(0.0, 0.5, " Der"), (0.5, 1.0, " Patient")], 0.0)
    buf.flush()
    buf.insert([(0.0, 0.5, " Der"), (0.5, 1.0, " Patient"), (1.0, 1.5, " hat")], 0.0)
    buf.flush()
    assert len(buf.commited_in_buffer) == 2
    buf.pop_commited(0.6)  # drops "Der" (ends at 0.5 <= 0.6)
    assert _texts(buf.commited_in_buffer) == [" Patient"]


def test_finish_returns_tail_without_reset_then_reset_clears():
    p = LocalAgreementProcessor()
    p.insert_hypothesis([(0.0, 0.5, " Der"), (0.5, 1.0, " Patient")], 0.0)
    p.commit()
    p.insert_hypothesis(
        [(0.0, 0.5, " Der"), (0.5, 1.0, " Patient"), (1.0, 1.5, " hat")], 0.0
    )
    p.commit()
    tail = p.finish()
    assert _texts(tail) == [" hat"]
    assert p.committed_text == " Der Patient"  # finish() did not change committed
    p.reset()
    assert p.committed == []
    assert p.provisional() == []
    assert p.committed_text == ""


def test_text_of_joins_without_extra_spaces():
    words = [TSWord(0.0, 0.5, " Der"), TSWord(0.5, 1.0, " Patient")]
    assert LocalAgreementProcessor.text_of(words) == " Der Patient"
