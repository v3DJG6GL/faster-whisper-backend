"""Per-decode trace of what faster-whisper did INSIDE ``model.transcribe``.

The receipt block reports window-level stats copied onto every segment
(alp/nsp/cr/T) and one wall-clock number for the whole decode. That says
nothing about the work faster-whisper actually performed: how many 30 s
windows it encoded (the leftover after the last aligned word becomes its own
zero-padded window when word timestamps are on), how many temperature rungs
each window ran, how many tokens every rung generated, and what became of a
window that yielded no segment (skipped as no-speech, or empty). A 2 s
utterance that took 43 s (2026-09-15) left exactly one segment with T=0.0 and
no way to tell a looping tail window from a stalled GPU.

This module hooks the loaded ``WhisperModel`` once (``install``) and collects
those facts per decode into a thread-local trace (``capture``), which the
decode thread turns into a plain dict (``finish``) for the receipt block's
"Decode trace" section. Hooks are passthroughs when no trace is active, so
partial decodes and the batched pipeline pay a dict lookup and nothing else.

What is hooked (faster-whisper 1.2.x, ``transcribe.py``):

* ``pad_or_trim`` (module global) — called with the UNPADDED window right
  before the encoder: its width is the window's real length in frames.
* ``WhisperModel.encode`` — one encoder pass per window (timed).
* ``WhisperModel.generate_with_fallback`` — one call per window; brackets the
  rungs and returns the chosen (alp, T, cr).
* ``ctranslate2.models.Whisper.generate`` — one call per temperature rung.
  The CT2 object refuses attribute writes, so ``model.model`` is replaced by a
  delegating proxy.

Every hook is defensive: a faster-whisper build that renamed an attribute
simply loses that facet of the trace, never the decode.

Residual-window stop (``capture(..., skip_residual=True)``)
-----------------------------------------------------------
faster-whisper advances ``seek`` to the last aligned word, not to the end of
the window it just decoded, so whatever trails that word (breath, VAD pad,
endpointer silence) is decoded AGAIN as its own zero-padded window — audio the
previous window already saw in full and chose not to transcribe. That leftover
is where the temperature ladder loops: 2026-09-17 a 5.9 s utterance spent
108 s in eleven rungs of 224-token repetition on two residual windows of
0.49 s and 0.37 s, and the result was dropped anyway. The rule here: once a
decoded window was shorter than 30 s it reached the end of the audio, and any
window faster-whisper tries to start after it is refused — the ``pad_or_trim``
hook raises ``ResidualWindowSkipped`` before the encoder runs, and
``consume()`` turns that into a normal end of the segment stream. Windows of a
long file are untouched: a full 30 s window never sets the flag. Language
detection pads the first window before any decode and cannot set it either.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any

_tls = threading.local()

_INSTALLED_FLAG = "_fwb_decode_trace_installed"

_N_FRAMES = 3000          # one 30 s window at 100 mel frames / s
_FRAMES_PER_S = 100.0


def _current() -> "DecodeTrace | None":
    return getattr(_tls, "trace", None)


class ResidualWindowSkipped(Exception):
    """Raised inside faster-whisper's window loop (from the ``pad_or_trim``
    hook) when a window would start after one that already reached the end
    of the audio. Ends the segment generator early; see ``consume``."""


def consume(gen) -> list:
    """Materialise faster-whisper's lazy segment generator, treating the
    residual-window stop as the normal end of the stream. Every segment the
    earlier windows yielded is kept."""
    out = []
    try:
        for seg in gen:
            out.append(seg)
    except ResidualWindowSkipped:
        pass
    return out


class DecodeTrace:
    """Mutable collector for one ``model.transcribe`` call (one thread)."""

    def __init__(self, *, no_speech_threshold=None, log_prob_threshold=None,
                 compression_ratio_threshold=None, length_penalty=1.0,
                 skip_residual: bool = False):
        self.windows: list[dict] = []
        self.extra_encodes = 0          # encoder passes not followed by a decode (language detection)
        self.skip_residual = bool(skip_residual)
        self.reached_end = False        # a DECODED window was shorter than 30 s
        self.skipped_windows = 0
        self.pending_len_frames: int | None = None
        self.pending_encode_s: float | None = None
        self.no_speech_threshold = no_speech_threshold
        self.log_prob_threshold = log_prob_threshold
        self.compression_ratio_threshold = compression_ratio_threshold
        self.length_penalty = float(length_penalty or 1.0)
        self.t0 = time.perf_counter()

    # -- hooks -------------------------------------------------------------
    def note_window_len(self, frames: int) -> None:
        """A window is about to be padded for the encoder. When a decoded
        window already reached the end of the audio, this one is the residual
        after its last word: record it as skipped and stop the decode."""
        frames = int(frames)
        if self.skip_residual and self.reached_end:
            self.windows.append({
                "n": len(self.windows) + 1,
                "len_frames": frames,
                "encode_s": None,
                "rungs": [],
                "chosen": None,
                "t0": time.perf_counter(),
                "secs": 0.0,
                "skipped": "residual",
            })
            self.skipped_windows += 1
            self.pending_len_frames = None
            self.pending_encode_s = None
            raise ResidualWindowSkipped(
                f"window {len(self.windows)} ({frames} frames) starts after the "
                "window that reached the end of the audio")
        self.pending_len_frames = frames

    def note_encode(self, secs: float) -> None:
        if self.pending_encode_s is not None:
            # Two encodes with no decode in between: the first was language
            # detection (transcribe() encodes the first window for it).
            self.extra_encodes += 1
        self.pending_encode_s = float(secs)

    def open_window(self) -> dict:
        w = {
            "n": len(self.windows) + 1,
            "len_frames": self.pending_len_frames,
            "encode_s": self.pending_encode_s,
            "rungs": [],
            "chosen": None,
            "t0": time.perf_counter(),
            "secs": None,
        }
        # Only a window that is actually decoded can mark the end: language
        # detection pads the first window too, before any decode.
        if w["len_frames"] is not None and w["len_frames"] < _N_FRAMES:
            self.reached_end = True
        self.pending_len_frames = None
        self.pending_encode_s = None
        self.windows.append(w)
        return w

    def close_window(self, w: dict, chosen) -> None:
        w["secs"] = time.perf_counter() - w["t0"]
        w["chosen"] = chosen

    def note_rung(self, rung: dict) -> None:
        if self.windows and self.windows[-1]["secs"] is None:
            self.windows[-1]["rungs"].append(rung)
        else:
            # generate() outside generate_with_fallback (a build without the
            # method, or a hook that failed): keep the rung in a synthetic
            # window so the token/time totals stay honest.
            w = self.open_window()
            w["rungs"].append(rung)
            w["secs"] = rung.get("secs")


@contextlib.contextmanager
def capture(kwargs: "dict | None" = None, *, skip_residual: bool = False):
    """Activate a trace for the decode running on THIS thread.

    Must wrap both ``model.transcribe(...)`` and the consumption of its lazy
    segment generator (that is where the windows are decoded). With
    ``skip_residual`` the generator must be drained through ``consume()``,
    which absorbs the ``ResidualWindowSkipped`` stop."""
    kw = kwargs or {}
    tr = DecodeTrace(
        no_speech_threshold=kw.get("no_speech_threshold"),
        log_prob_threshold=kw.get("log_prob_threshold"),
        compression_ratio_threshold=kw.get("compression_ratio_threshold"),
        length_penalty=kw.get("length_penalty", 1.0) or 1.0,
        skip_residual=skip_residual,
    )
    prev = _current()
    _tls.trace = tr
    try:
        yield tr
    finally:
        _tls.trace = prev


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

class _GenerateProxy:
    """Delegates everything to the CT2 Whisper object; times ``generate``."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_inner"), name, value)

    def generate(self, *args, **kwargs):
        inner = object.__getattribute__(self, "_inner")
        tr = _current()
        if tr is None:
            return inner.generate(*args, **kwargs)
        t0 = time.perf_counter()
        results = inner.generate(*args, **kwargs)
        secs = time.perf_counter() - t0
        try:
            tr.note_rung(_describe_rung(results, kwargs, secs, tr))
        except Exception:  # never let bookkeeping break a decode
            tr.note_rung({"secs": secs})
        return results


def _describe_rung(results, kwargs: dict, secs: float, tr: DecodeTrace) -> dict:
    r0 = results[0]
    tokens = list(r0.sequences_ids[0]) if getattr(r0, "sequences_ids", None) else []
    n = len(tokens)
    rung: dict[str, Any] = {
        "secs": secs,
        "tokens": n,
        "temperature": float(kwargs.get("sampling_temperature", 0.0) or 0.0),
        "beam_size": kwargs.get("beam_size"),
        "num_hypotheses": kwargs.get("num_hypotheses"),
        "nsp": getattr(r0, "no_speech_prob", None),
        "alp": None,
        "cr": None,
    }
    scores = getattr(r0, "scores", None)
    if scores:
        # Same arithmetic as generate_with_fallback (transcribe.py:1463-1466).
        lp = float(kwargs.get("length_penalty", tr.length_penalty) or 1.0)
        rung["alp"] = (float(scores[0]) * (n ** lp)) / (n + 1)
    tok = getattr(_tls, "tokenizer", None)
    if tok is not None and tokens:
        try:
            from faster_whisper.transcribe import get_compression_ratio
            eot = getattr(tok, "eot", None)
            text_tokens = [t for t in tokens if eot is None or t < eot]
            rung["cr"] = get_compression_ratio(tok.decode(text_tokens).strip())
        except Exception:
            pass
    return rung


def install(model):
    """Hook a loaded ``WhisperModel`` in place (idempotent). Returns ``model``.

    Objects that are not a faster-whisper model (test fakes) come back
    untouched: every hook is attached only where its target exists."""
    if model is None or getattr(model, _INSTALLED_FLAG, False):
        return model
    installed_any = False

    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "generate") and not isinstance(inner, _GenerateProxy):
        try:
            model.model = _GenerateProxy(inner)
            installed_any = True
        except Exception:
            pass

    orig_encode = getattr(model, "encode", None)
    if callable(orig_encode):
        def encode(features, _orig=orig_encode):
            tr = _current()
            if tr is None:
                return _orig(features)
            t0 = time.perf_counter()
            out = _orig(features)
            tr.note_encode(time.perf_counter() - t0)
            return out
        try:
            model.encode = encode
            installed_any = True
        except Exception:
            pass

    orig_fb = getattr(model, "generate_with_fallback", None)
    if callable(orig_fb):
        def generate_with_fallback(*args, _orig=orig_fb, **kwargs):
            tr = _current()
            if tr is None:
                return _orig(*args, **kwargs)
            # args: (encoder_output, prompt, tokenizer, options) — the tokenizer
            # lets the rung hook compute the compression ratio it retried on.
            tokenizer = args[2] if len(args) > 2 else kwargs.get("tokenizer")
            _tls.tokenizer = tokenizer
            w = tr.open_window()
            try:
                result = _orig(*args, **kwargs)
            except BaseException:
                tr.close_window(w, None)
                raise
            finally:
                _tls.tokenizer = None
            chosen = None
            try:
                chosen = {"alp": float(result[1]), "temperature": float(result[2]),
                          "cr": float(result[3]),
                          "nsp": getattr(result[0], "no_speech_prob", None)}
            except Exception:
                pass
            tr.close_window(w, chosen)
            return result
        try:
            model.generate_with_fallback = generate_with_fallback
            installed_any = True
        except Exception:
            pass

    _install_pad_hook()
    if installed_any:
        try:
            setattr(model, _INSTALLED_FLAG, True)
        except Exception:
            pass
    return model


_pad_hooked = False


def _install_pad_hook() -> None:
    """Wrap ``faster_whisper.transcribe.pad_or_trim`` (module global) once."""
    global _pad_hooked
    if _pad_hooked:
        return
    try:
        import faster_whisper.transcribe as fwt
    except Exception:
        return
    orig = getattr(fwt, "pad_or_trim", None)
    if not callable(orig):
        return

    def pad_or_trim(array, *args, **kwargs):
        tr = _current()
        if tr is not None:
            try:
                tr.note_window_len(int(array.shape[-1]))
            except ResidualWindowSkipped:
                raise           # the stop rule, absorbed by consume()
            except Exception:
                pass
        return orig(array, *args, **kwargs)

    fwt.pad_or_trim = pad_or_trim
    _pad_hooked = True


# ---------------------------------------------------------------------------
# Post-decode summary
# ---------------------------------------------------------------------------


def finish(tr: "DecodeTrace | None", segments, info=None) -> "dict | None":
    """Turn a collected trace into the plain dict the receipt block renders.

    ``segments`` are the yielded ``Segment`` objects (their ``seek`` is the
    frame offset of the window that produced them); ``info`` supplies the
    decoded content length so each window's start can be placed."""
    if tr is None:
        return None
    dur = None
    if info is not None:
        dav = getattr(info, "duration_after_vad", None)
        dur = float(dav if dav is not None else (getattr(info, "duration", 0.0) or 0.0))
    seeks: dict[int, int] = {}
    for s in segments or []:
        sk = getattr(s, "seek", None)
        if sk is not None:
            seeks[int(sk)] = seeks.get(int(sk), 0) + 1

    windows_out = []
    total_tokens = 0
    total_rungs = 0
    total_gen_s = 0.0
    for i, w in enumerate(tr.windows):
        lf = w.get("len_frames")
        len_s = (lf / _FRAMES_PER_S) if lf is not None else None
        # A window shorter than 30 s is the last one of its clip: it ends at
        # the content end, so its start is content − length. The first window
        # starts at 0 regardless.
        start_s: float | None
        if i == 0:
            start_s = 0.0
        elif len_s is not None and lf < _N_FRAMES and dur is not None:
            start_s = max(0.0, dur - len_s)
        else:
            start_s = None
        seg_count = 0
        if start_s is not None:
            frame = int(round(start_s * _FRAMES_PER_S))
            for sk, c in seeks.items():
                if abs(sk - frame) <= 2:
                    seg_count = c
        rungs = []
        for r in w.get("rungs", []):
            total_tokens += int(r.get("tokens") or 0)
            total_rungs += 1
            total_gen_s += float(r.get("secs") or 0.0)
            rungs.append(dict(r))
        # Retry reasons, replicating generate_with_fallback's ladder rules.
        cr_thr = tr.compression_ratio_threshold
        lp_thr = tr.log_prob_threshold
        ns_thr = tr.no_speech_threshold
        for j, r in enumerate(rungs):
            last = j == len(rungs) - 1
            reasons = []
            alp, cr, nsp = r.get("alp"), r.get("cr"), r.get("nsp")
            if cr_thr is not None and cr is not None and cr > cr_thr:
                reasons.append(f"cr {cr:.2f} > {cr_thr:g}")
            if lp_thr is not None and alp is not None and alp < lp_thr:
                reasons.append(f"alp {alp:.2f} < {lp_thr:g}")
            silence = (ns_thr is not None and nsp is not None and nsp > ns_thr
                       and lp_thr is not None and alp is not None and alp < lp_thr)
            if not last:
                r["outcome"] = "retry · " + ", ".join(reasons) if reasons else "retry"
            else:
                r["outcome"] = _window_outcome(w, r, seg_count, reasons, silence,
                                               ns_thr, lp_thr)
        entry = {
            "n": w.get("n"),
            "start_s": start_s,
            "len_s": len_s,
            "encode_s": w.get("encode_s"),
            "secs": w.get("secs"),
            "segments": seg_count,
            "rungs": rungs,
        }
        if w.get("skipped") == "residual":
            entry["skipped"] = "residual"
            entry["outcome"] = "skipped · previous window reached end of audio"
        windows_out.append(entry)
    return {
        "windows": windows_out,
        "n_windows": len(windows_out),
        "n_rungs": total_rungs,
        "tokens": total_tokens,
        "generate_s": total_gen_s,
        "extra_encodes": tr.extra_encodes,
        "skipped_windows": tr.skipped_windows,
        "total_s": time.perf_counter() - tr.t0,
    }


def _window_outcome(w, last_rung, seg_count, reasons, silence, ns_thr, lp_thr) -> str:
    chosen = w.get("chosen") or {}
    alp = chosen.get("alp", last_rung.get("alp"))
    nsp = chosen.get("nsp", last_rung.get("nsp"))
    if seg_count:
        return f"kept · {seg_count} segment{'s' if seg_count != 1 else ''}"
    # transcribe.py:1215-1235 — skipped when nsp > threshold unless alp is high enough.
    if (ns_thr is not None and nsp is not None and nsp > ns_thr
            and not (lp_thr is not None and alp is not None and alp > lp_thr)):
        return f"skipped · no-speech (nsp {nsp:.2f} > {ns_thr:g}, alp {alp:.2f})" \
            if alp is not None else f"skipped · no-speech (nsp {nsp:.2f} > {ns_thr:g})"
    if reasons and not silence and len(w.get("rungs", [])) > 1:
        return "all rungs failed · best alp kept · no text"
    return "no text"
