"""LocalAgreement-2 hypothesis stabilization for streaming Whisper.

This is the "v2" stabilizer: it turns Whisper's chunk-based decoding into a
stable live stream. On a growing audio buffer Whisper is re-decoded every ~1 s;
the *tail* of each hypothesis is unstable (it rewrites itself as more right-context
arrives), but the *prefix* converges. LocalAgreement-2 commits a word only once it
appears at the same position in **two consecutive** hypotheses, so the committed
prefix never flickers; only the unconfirmed tail updates.

Ported faithfully from ufal/whisper_streaming (``whisper_online.py``,
``HypothesisBuffer`` + the ``OnlineASRProcessor`` commit logic), MIT License,
Dominik Macháček et al., "Turning Whisper into Real-Time Transcription System",
IJCNLP-AACL 2023 — https://github.com/ufal/whisper_streaming

This module is intentionally **pure**: it knows nothing about audio, numpy, or
faster-whisper. It operates on word triples ``(start, end, text)`` where the
timestamps are buffer-relative seconds and ``text`` is the raw Whisper word
(faster-whisper words carry their own leading space, e.g. ``" Patient"``). The
audio buffer, VAD, and model orchestration live in ``streaming_session.py``.

Load-bearing invariants (each verified against the upstream source — getting any
of these wrong silently corrupts the stream):

  * Confirmation is **text equality only** (``new[0].text == buffer[0].text``).
    Never compare token IDs or timestamps — Whisper's per-word timestamps wobble
    run-to-run and would cause false mismatches.
  * On a match you **pop from BOTH** the previous-hypothesis list and the
    current-hypothesis list. Forgetting to pop the new list double-emits every
    committed word.
  * The first iteration commits nothing (``buffer`` is empty) — LA-2 needs two
    hypotheses to agree. This is correct, not a bug.
  * ``finish()`` returns the still-unconfirmed tail but does **not** reset state;
    the caller re-inits per utterance (see ``LocalAgreementProcessor.reset``).
"""

from typing import Iterable, NamedTuple


class TSWord(NamedTuple):
    """A timestamped word. ``start``/``end`` are absolute seconds; ``text`` is the
    raw Whisper word including its leading space."""

    start: float
    end: float
    text: str


# Maximum n-gram length checked by the boundary-overlap dedup guard in insert().
_MAX_NGRAM = 5
# Slack (seconds) below the last committed time within which a fresh word is still
# considered "new" — absorbs minor timestamp jitter between consecutive decodes.
_INSERT_SLACK = 0.1
# A re-emitted boundary word is only deduped when the first new word starts within
# this window of the last committed time (i.e. genuinely at the seam).
_BOUNDARY_WINDOW = 1.0


class HypothesisBuffer:
    """Stabilizes a sequence of ASR hypotheses over a growing audio buffer.

    Three lists implement LocalAgreement-2 (all hold :class:`TSWord`):

      * ``commited_in_buffer`` — words already committed that still lie inside the
        current (un-trimmed) audio window. Used only by the boundary dedup guard
        and ``pop_commited``.
      * ``buffer`` — the PREVIOUS hypothesis's unconfirmed tail; the "reference"
        the next hypothesis is compared against.
      * ``new`` — the CURRENT hypothesis, after timestamp filtering.

    (The misspelling ``commited`` is kept from upstream to ease cross-referencing.)
    """

    def __init__(self) -> None:
        self.commited_in_buffer: list[TSWord] = []
        self.buffer: list[TSWord] = []
        self.new: list[TSWord] = []
        self.last_commited_time: float = 0.0
        self.last_commited_word: str | None = None

    def insert(self, words: Iterable[tuple[float, float, str]], offset: float) -> None:
        """Stage a fresh hypothesis for the next :meth:`flush`.

        ``words`` are buffer-relative triples; ``offset`` (the audio buffer's wall
        start time) is added to make them absolute. Words at or before the last
        committed time are dropped, and a re-emitted boundary word (Whisper often
        repeats the last committed word at the new buffer's front) is removed via
        an n-gram match against the committed tail.
        """
        new_words = [TSWord(a + offset, b + offset, t) for (a, b, t) in words]
        # Keep only words that begin after what we've already committed.
        self.new = [w for w in new_words if w.start > self.last_commited_time - _INSERT_SLACK]

        if not self.new:
            return
        first_start = self.new[0].start
        if abs(first_start - self.last_commited_time) >= _BOUNDARY_WINDOW:
            return  # not at the seam — no boundary re-emission to dedup
        if not self.commited_in_buffer:
            return
        # Drop the longest committed-tail n-gram (n = 1..5) that the new hypothesis
        # re-emits at its front. Both sides are joined the same way, so the exact
        # spacing of the word tokens is irrelevant to the equality.
        cn = len(self.commited_in_buffer)
        nn = len(self.new)
        for i in range(1, min(cn, nn, _MAX_NGRAM) + 1):
            committed_tail = " ".join(w.text for w in self.commited_in_buffer[-i:])
            new_head = " ".join(w.text for w in self.new[:i])
            if committed_tail == new_head:
                del self.new[:i]
                break

    def flush(self) -> list[TSWord]:
        """Commit the longest common prefix of the previous and current hypotheses.

        Returns the newly-committed words (the delta). The unconfirmed remainder of
        the current hypothesis becomes the reference for the next call.
        """
        commit: list[TSWord] = []
        while self.new:
            na, nb, nt = self.new[0]
            if len(self.buffer) == 0:
                break  # nothing to agree against (e.g. first iteration)
            if nt == self.buffer[0].text:  # TEXT equality only
                commit.append(self.new[0])
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)  # pop from BOTH lists on a match
                self.new.pop(0)
            else:
                break  # first mismatch ends the agreed prefix
        self.buffer = self.new  # leftover tail is next iteration's reference
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time: float) -> None:
        """Drop committed words that end at or before ``time`` (after an audio trim)."""
        while self.commited_in_buffer and self.commited_in_buffer[0].end <= time:
            self.commited_in_buffer.pop(0)

    def complete(self) -> list[TSWord]:
        """The current provisional (unconfirmed) tail — show this greyed/faded."""
        return list(self.buffer)


class LocalAgreementProcessor:
    """Per-utterance driver around a :class:`HypothesisBuffer`.

    Accumulates the committed words for the current utterance and exposes the
    committed/provisional split the WebSocket layer streams to the client. One
    instance per utterance; call :meth:`reset` after finalizing.
    """

    def __init__(self) -> None:
        self._buf = HypothesisBuffer()
        self.committed: list[TSWord] = []

    def insert_hypothesis(
        self, words: Iterable[tuple[float, float, str]], offset: float
    ) -> None:
        self._buf.insert(words, offset)

    def commit(self) -> list[TSWord]:
        """Run LocalAgreement and return the newly-committed words (the delta)."""
        delta = self._buf.flush()
        self.committed.extend(delta)
        return delta

    def provisional(self) -> list[TSWord]:
        """The unconfirmed tail (changes between calls)."""
        return self._buf.complete()

    def pop_committed(self, time: float) -> None:
        self._buf.pop_commited(time)

    def finish(self) -> list[TSWord]:
        """The remaining unconfirmed tail at end-of-utterance.

        Does NOT reset — the utterance's full text is ``committed + finish()``.
        Call :meth:`reset` afterwards to start the next utterance.
        """
        return self._buf.complete()

    def reset(self) -> None:
        self._buf = HypothesisBuffer()
        self.committed = []

    @staticmethod
    def text_of(words: Iterable[TSWord]) -> str:
        """Reconstruct raw text from words. Joined with '' because faster-whisper
        words already carry their leading space."""
        return "".join(w.text for w in words)

    @property
    def committed_text(self) -> str:
        return self.text_of(self.committed)
