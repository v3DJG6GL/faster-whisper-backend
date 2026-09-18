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

    lang_detect = False   # pad + encode the first window before any decode
    duration = 2.2

    def transcribe(self, audio, **kw):
        # Lazy like generate_segments: each window is padded, encoded and
        # decoded only when the caller pulls the next segment.
        import faster_whisper.transcribe as fwt
        import numpy as np

        def gen():
            if self.lang_detect:
                fwt.pad_or_trim(np.zeros((128, self.windows[0]["frames"]), dtype="float32"))
                self.encode(None)
            for spec in self.windows:
                fwt.pad_or_trim(np.zeros((128, spec["frames"]), dtype="float32"))
                self.encode(None)
                self.generate_with_fallback("enc", [1, 2, 3], _Tokenizer(), None)
                for _ in range(spec["yield"]):
                    yield _Seg(spec["seek"], "x")
        return gen(), FakeInfo(duration=self.duration)


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


# ---------------------------------------------------------------------------
# Residual-window stop (DECODE_SKIP_RESIDUAL_WINDOWS)
# ---------------------------------------------------------------------------

def _run_stop(m, **kw):
    with dt.capture(_KW, **kw) as tr:
        segs, info = m.transcribe(None)
        segs = dt.consume(segs)
        return segs, dt.finish(tr, segs, info)


def test_residual_window_is_refused_before_it_is_encoded():
    """The 2026-09-17 shape: window 1 covered the whole clip (shorter than
    30 s), so the 0.47 s leftover after its last word is refused — no encode,
    no generate, no temperature ladder — and window 1's segment is kept."""
    m = dt.install(_incident_model())
    segs, t = _run_stop(m, skip_residual=True)
    assert len(segs) == 1
    assert len(m.model.calls) == 1, "only window 1 reached the decoder"
    assert t["n_windows"] == 2 and t["skipped_windows"] == 1
    assert t["n_rungs"] == 1 and t["tokens"] == 5
    w1, w2 = t["windows"]
    assert w1["rungs"][0]["outcome"] == "kept · 1 segment"
    assert w2["skipped"] == "residual" and w2["rungs"] == []
    assert w2["encode_s"] is None
    assert w2["outcome"] == "skipped · previous window reached end of audio"
    assert w2["start_s"] == pytest.approx(2.2 - 0.47, abs=0.01)
    assert w2["len_s"] == pytest.approx(0.47)


def test_residual_stop_is_off_unless_asked():
    m = dt.install(_incident_model())
    segs, t = _run_stop(m)
    assert len(segs) == 1 and t["n_rungs"] == 7 and t["skipped_windows"] == 0
    assert "skipped" not in t["windows"][1]


def test_language_detection_pad_does_not_arm_the_stop():
    """transcribe() pads + encodes the first window for language detection
    BEFORE the loop decodes it. That pad must not count as a decoded window,
    or the real first window would be refused."""
    m = _incident_model()
    m.lang_detect = True
    dt.install(m)
    segs, t = _run_stop(m, skip_residual=True)
    assert len(segs) == 1
    assert t["n_windows"] == 2 and t["skipped_windows"] == 1
    assert t["extra_encodes"] == 1
    assert t["windows"][0]["rungs"][0]["outcome"] == "kept · 1 segment"


def test_full_windows_of_a_long_file_are_never_refused():
    """A 65 s file: two full 30 s windows, a 5 s last window, then the
    residual after its last word. Only the residual is refused."""
    plan = [[([5, 6, 7], -0.5, 0.01)]] * 3 + [[([9], -0.5, 0.01)]]
    m = _Model(plan, _THR)
    m.duration = 65.0
    m.windows = [{"frames": 3000, "seek": 0, "yield": 1},
                 {"frames": 3000, "seek": 2900, "yield": 1},
                 {"frames": 500, "seek": 6000, "yield": 1},   # 65 s − 5 s
                 {"frames": 60, "seek": 6440, "yield": 0}]
    dt.install(m)
    segs, t = _run_stop(m, skip_residual=True)
    assert len(segs) == 3
    assert len(m.model.calls) == 3
    assert [w.get("skipped") for w in t["windows"]] == [None, None, None, "residual"]
    assert t["windows"][2]["rungs"][0]["outcome"] == "kept · 1 segment"


def test_consume_returns_everything_when_nothing_stops():
    m = dt.install(_incident_model())
    with dt.capture(_KW):
        segs, _ = m.transcribe(None)
        assert len(dt.consume(segs)) == 1


def test_stop_needs_a_capture_to_be_armed():
    """Without a capture the hooks are passthroughs: a plain transcribe on
    the hooked model decodes every window, as faster-whisper would."""
    m = dt.install(_incident_model())
    segs, _ = m.transcribe(None)
    assert len(list(segs)) == 1
    assert len(m.model.calls) == 7


def test_receipt_shows_the_refused_window_and_the_guard_row(app_module):
    seg = [{"id": 0, "start": 0.02, "end": 1.72, "alp": -0.44, "nsp": 0.01,
            "cr": 0.64, "temp": 0.0, "text": "Erzinkontinenz", "dropped": False}]
    _, t = _run_stop(dt.install(_incident_model()), skip_residual=True)
    base = dict(file_label="stream utt#3", model_name="m", info=FakeInfo(duration=2.2),
                kwargs={"beam_size": 10}, seg_diag=seg, raw="", final="",
                decode_trace=t)
    block = app_module._format_request_block(
        **base, guards={"skip_residual_windows": True})
    assert "Decode trace  (2 windows · 1 generate call · 5 tokens" in block
    assert "· 1 residual skipped)" in block
    row = next(l for l in block.splitlines() if "skipped · previous window" in l)
    assert row.startswith("      2 ")
    assert "1.73s" in row and "0.47s" in row
    guard = next(l for l in block.splitlines() if "skip_residual_windows" in l)
    assert guard.rstrip().endswith("true"), "default on: no non-default marker"
    off = app_module._format_request_block(
        **base, guards={"skip_residual_windows": False})
    assert next(l for l in off.splitlines()
                if "skip_residual_windows" in l).rstrip().endswith("false *")
