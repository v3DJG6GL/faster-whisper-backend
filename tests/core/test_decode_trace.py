"""Decode trace: what faster-whisper did INSIDE model.transcribe.

Pins the three halves of faster_whisper_backend.core.decode_trace:

  * install() hooks a WhisperModel-shaped object (encode, generate_with_fallback,
    the CT2 `model.generate` behind a proxy) and is a no-op for test fakes;
  * capture()/finish() turn the hook calls of ONE decode (thread-local) into
    windows → rungs with the ladder verdicts faster-whisper applied;
  * the receipt renders it as a `Decode trace` section, and a decode without a
    trace keeps exactly the block it had.
"""

import threading

import pytest

from faster_whisper_backend.core import decode_trace as dt
from tests.conftest import FakeInfo


# ---------------------------------------------------------------------------
# A WhisperModel-shaped fake with the CT2 seam the hooks attach to
# ---------------------------------------------------------------------------

class _Result:
    def __init__(self, tokens, score, nsp):
        self.sequences_ids = [list(tokens)]
        self.scores = [score]
        self.no_speech_prob = nsp


class _CT2:
    """Stand-in for ctranslate2.models.Whisper: attribute writes are refused
    (the real object is a C extension), so install() must proxy it."""

    def __init__(self, script):
        object.__setattr__(self, "script", list(script))
        object.__setattr__(self, "calls", [])

    def __setattr__(self, name, value):
        raise AttributeError(f"'{name}' is read-only")

    def generate(self, enc, prompts, **kw):
        self.calls.append(kw)
        tokens, score, nsp = self.script.pop(0)
        return [_Result(tokens, score, nsp)]

    is_multilingual = True


class _Tokenizer:
    eot = 50257

    def decode(self, tokens):
        return " ".join("w" if t < 100 else "loop" for t in tokens)


class _Seg:
    def __init__(self, seek, text):
        self.seek = seek
        self.text = text


class _Model:
    """Mirrors the faster-whisper call shape: one encode + one
    generate_with_fallback per window, generate per rung. `plan` is a list of
    windows, each a list of rung scripts (tokens, score, nsp)."""

    def __init__(self, plan, thresholds):
        self.model = _CT2([rung for window in plan for rung in window])
        self.plan = plan
        self.thr = thresholds

    def encode(self, features):
        return "enc"

    def generate_with_fallback(self, encoder_output, prompt, tokenizer, options):
        temps = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        last = None
        for i, t in enumerate(temps):
            kw = ({"beam_size": 10, "patience": 1} if t == 0.0
                  else {"beam_size": 1, "num_hypotheses": 5, "sampling_temperature": t})
            r = self.model.generate(encoder_output, [prompt], length_penalty=1.0, **kw)[0]
            n = len(r.sequences_ids[0])
            alp = r.scores[0] * n / (n + 1)
            cr = 3.0 if any(tok >= 100 for tok in r.sequences_ids[0]) else 0.8
            last = (r, alp, t, cr)
            fails = cr > self.thr["cr"] or alp < self.thr["lp"]
            silence = r.no_speech_prob > self.thr["ns"] and alp < self.thr["lp"]
            if not fails or silence or i == len(temps) - 1:
                break
        return last

    def transcribe(self, audio, **kw):
        # Same order as generate_segments: pad_or_trim → encode → fallback per window.
        import faster_whisper.transcribe as fwt
        import numpy as np
        out = []
        for spec in self.windows:
            fwt.pad_or_trim(np.zeros((128, spec["frames"]), dtype="float32"))
            self.encode(None)
            self.generate_with_fallback("enc", [1, 2, 3], _Tokenizer(), None)
            out.extend(_Seg(spec["seek"], "x") for _ in range(spec["yield"]))
        return iter(out), FakeInfo(duration=2.2)


_THR = {"cr": 2.4, "lp": -1.0, "ns": 0.6}
_KW = {"no_speech_threshold": 0.6, "log_prob_threshold": -1.0,
       "compression_ratio_threshold": 2.4}


def _incident_model():
    """The 2026-09-15 shape: window 1 clean at T=0; the 0.47 s leftover after
    the last word loops confidently for five rungs (cr > 2.4), then the last
    rung trips the silence rule and the window is skipped — one segment out."""
    loop = list(range(100, 100 + 440))
    plan = [
        [([5, 6, 7, 8, 9], -0.5, 0.01)],                       # window 1: kept
        [(loop, -0.3, 0.71), (loop, -0.5, 0.71), (loop, -0.6, 0.71),
         (loop, -0.8, 0.71), (loop, -0.9, 0.71), (loop, -1.9, 0.71)],  # tail window
    ]
    m = _Model(plan, _THR)
    m.windows = [{"frames": 219, "seek": 0, "yield": 1},
                 {"frames": 47, "seek": 172, "yield": 0}]
    return m


# ---------------------------------------------------------------------------
# install()
# ---------------------------------------------------------------------------

def test_install_is_noop_for_fakes_and_idempotent():
    class Bare:
        def transcribe(self, *a, **k):
            return iter([]), FakeInfo()
    b = Bare()
    assert dt.install(b) is b
    assert not hasattr(b, dt._INSTALLED_FLAG)

    m = _incident_model()
    dt.install(m)
    proxy = m.model
    assert isinstance(proxy, dt._GenerateProxy)
    dt.install(m)
    assert m.model is proxy, "second install must not re-wrap"
    # Delegation: attributes of the CT2 object stay reachable through the proxy.
    assert m.model.is_multilingual is True


def test_hooks_are_passthrough_without_a_capture():
    m = dt.install(_incident_model())
    segs, info = m.transcribe(None)
    assert len(list(segs)) == 1
    assert dt._current() is None


# ---------------------------------------------------------------------------
# capture() / finish()
# ---------------------------------------------------------------------------

def _run(m):
    with dt.capture(_KW) as tr:
        segs, info = m.transcribe(None)
        segs = list(segs)
        return dt.finish(tr, segs, info)


def test_incident_shape_is_traced_window_by_window():
    t = _run(dt.install(_incident_model()))
    assert t["n_windows"] == 2
    assert t["n_rungs"] == 7
    assert t["tokens"] == 5 + 6 * 440
    w1, w2 = t["windows"]
    assert w1["start_s"] == 0.0 and w1["len_s"] == pytest.approx(2.19)
    assert w1["segments"] == 1
    assert w1["rungs"][0]["outcome"] == "kept · 1 segment"
    assert w1["rungs"][0]["beam_size"] == 10
    # The tail window starts where the content ends minus its own length.
    assert w2["start_s"] == pytest.approx(2.2 - 0.47, abs=0.01)
    assert w2["segments"] == 0
    outcomes = [r["outcome"] for r in w2["rungs"]]
    # cr is recomputed from the rung's own tokens (real zlib ratio of the
    # decoded text), so a 440-token loop reads far above the threshold.
    assert outcomes[0].startswith("retry · cr ") and "> 2.4" in outcomes[0]
    assert all(o.startswith("retry") for o in outcomes[:-1])
    assert outcomes[-1].startswith("skipped · no-speech (nsp 0.71 > 0.6")
    temps = [r["temperature"] for r in w2["rungs"]]
    assert temps == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert w2["rungs"][1]["num_hypotheses"] == 5
    assert all(r["cr"] is not None and r["cr"] > 2.4 for r in w2["rungs"])


def test_capture_is_thread_local():
    """Two decodes on the same hooked model (INFERENCE_CONCURRENCY 2) must not
    see each other's windows."""
    m = dt.install(_incident_model())
    seen = {}

    def worker(name):
        with dt.capture(_KW) as tr:
            seen[name] = tr
            assert dt._current() is tr

    a = threading.Thread(target=worker, args=("a",))
    a.start(); a.join()
    assert dt._current() is None
    assert seen["a"].windows == []


def test_finish_without_trace_is_none():
    assert dt.finish(None, [], FakeInfo()) is None


# ---------------------------------------------------------------------------
# Receipt section
# ---------------------------------------------------------------------------

def test_receipt_renders_the_section_and_stays_unchanged_without_it(app_module):
    seg = [{"id": 0, "start": 0.02, "end": 1.72, "alp": -0.44, "nsp": 0.01,
            "cr": 0.64, "temp": 0.0, "text": "Erzinkontinenz", "dropped": False}]
    base = dict(file_label="stream utt#3", model_name="m", info=FakeInfo(duration=2.2),
                kwargs={"beam_size": 10}, seg_diag=seg, raw="", final="")
    plain = app_module._format_request_block(**base)
    assert "Decode trace" not in plain
    assert app_module._format_request_block(**base, decode_trace=None) == plain
    assert app_module._format_request_block(**base, decode_trace={"windows": []}) == plain

    t = _run(dt.install(_incident_model()))
    block = app_module._format_request_block(**base, decode_trace=t)
    assert "Decode trace  (2 windows · 7 generate calls · 2645 tokens" in block
    assert "kept · 1 segment" in block
    assert "retry · cr " in block and "> 2.4" in block
    assert "skipped · no-speech" in block
    assert "beam10" in block and "best5" in block
    # Section order: after the decode params, before the segments table.
    assert block.index("Decode params") < block.index("Decode trace") < block.index("Segments  (n=1)")


def test_long_decode_is_summarised(app_module):
    windows = [{"n": i + 1, "start_s": i * 30.0, "len_s": 30.0, "encode_s": 0.1,
                "secs": 1.0 + (5.0 if i == 7 else 0.0), "segments": 3,
                "rungs": [{"temperature": 0.0, "beam_size": 10, "tokens": 90,
                           "alp": -0.3, "cr": 1.1, "nsp": 0.0, "secs": 1.0,
                           "outcome": "kept · 3 segments"}]}
               for i in range(20)]
    t = {"windows": windows, "n_windows": 20, "n_rungs": 20, "tokens": 1800,
         "generate_s": 25.0, "extra_encodes": 1}
    lines = app_module._format_decode_trace_section(t)
    text = "\n".join(lines)
    assert "20 windows · 20 generate calls · 1800 tokens · 25.0s in generate · +1 lang-detect encode" in text
    assert "slowest 5 of 20 windows shown" in text
    assert "… 15 more windows omitted" in text
    assert text.count("kept · 3 segments") == 5
    assert "  8  " in text  # the 6 s window is among the slowest


def test_plain_text_guard_value_renders_without_quotes(app_module):
    row = app_module._param_row("    ", "tail_trim_cut",
                                app_module.PlainText("1.52s  (3.74s → 2.22s)"))
    assert row.endswith("1.52s  (3.74s → 2.22s)")
    assert "'" not in row
