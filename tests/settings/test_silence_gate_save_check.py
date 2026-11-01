"""Save-time refusal of a GLOBAL live-dictation silence pair that does not fit
(schema.AdminConfig._validate_silence_gates): the frame-exact hard-break
boundary, and a hint that only names a gate the admin can change."""

import pytest
from pydantic import ValidationError

from faster_whisper_backend.settings import schema as settings_schema
from faster_whisper_backend.streaming.vad import FRAME_MS

INNER, OUTER, HB = ("STREAMING_VAD_INNER_SILENCE_MS", "STREAMING_VAD_OUTER_SILENCE_MS",
                    "STREAMING_HARD_BREAK_SILENCE_MS")


def _save(env=None, **fields):
    ctx = {"silence_pair_check": frozenset(fields)}
    if env:
        ctx["env_effective"] = env
    return settings_schema.AdminConfig.model_validate(fields, context=ctx)


def test_hard_break_within_a_frame_of_outer_is_refused():
    # outer 1200: the finalize fires at 1216 ms and the hard-break check sees
    # 1248 ms one frame later, so 1240 broke after every sentence like 1200.
    assert FRAME_MS == 32
    with pytest.raises(ValidationError, match=r"0 \(off\) or above 1248 ms"):
        _save(**{OUTER: 1200, HB: 1240})
    _save(**{OUTER: 1200, HB: 1249})


def test_hint_never_names_an_env_pinned_gate_to_change():
    with pytest.raises(ValidationError) as e:
        _save(env={HB: 5000}, **{OUTER: 6000})
    msg = str(e.value)
    assert ("STREAMING_HARD_BREAK_SILENCE_MS is set by "
            "WHISPER_STREAMING_HARD_BREAK_SILENCE_MS, so set "
            "STREAMING_VAD_OUTER_SILENCE_MS to at most 4960 ms") in msg
    assert f"set {HB} to" not in msg
    with pytest.raises(ValidationError) as e:
        _save(env={OUTER: 1200}, **{INNER: 1500})
    msg = str(e.value)
    assert (f"{OUTER} is set by WHISPER_{OUTER}, so set {INNER} to below 1200 ms"
            in msg) and "or change" not in msg
    with pytest.raises(ValidationError) as e:
        _save(env={OUTER: 1200, INNER: 1500}, **{INNER: 1500})
    assert f"change WHISPER_{INNER} or WHISPER_{OUTER}" in str(e.value)
    # Nothing pinned: the hint keeps offering both gates.
    with pytest.raises(ValidationError, match=f"set {HB} to .*, or change {OUTER}"):
        _save(**{OUTER: 6000})
