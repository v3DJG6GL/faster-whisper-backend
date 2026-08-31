"""Full-pipeline sections of the per-request receipt block.

_format_request_block was written when the pipeline was decode and nothing
else. Separation, diarization and translation were added around it and never
reached it, even though every one of their results was in scope at the call
site. These tests pin the two halves of the fix:

  * a stage section appears ONLY when that stage actually ran, so a
    decode-only receipt keeps exactly the shape it has always had;
  * the additions don't collide with the glyphs and phrases the existing
    receipt assertions depend on (`✗`, "unchanged", "N step").
"""

from conftest import FakeInfo

_SEG = [
    {"id": 0, "start": 0.31, "end": 13.29, "alp": -0.28, "nsp": 0.02,
     "cr": 1.12, "temp": 0.0, "text": "Guten Morgen zusammen", "dropped": False},
    {"id": 1, "start": 13.44, "end": 19.02, "alp": -0.31, "nsp": 0.01,
     "cr": 1.09, "temp": 0.0, "text": "Ja, danke", "dropped": False},
]

_STAGES = [
    {"name": "separating", "secs": 102.3, "model": "UVR-MDX-NET-Inst_HQ_4",
     "device": "cuda", "load_secs": 3.1, "detail": "incl. transcode"},
    {"name": "transcribing", "secs": 198.7, "model": "faster-whisper-large-v2",
     "device": "cuda", "load_secs": 0.0},
    {"name": "diarizing", "secs": 44.1, "model": "pyannote/speaker-diarization-3.1",
     "device": "cuda", "load_secs": 0.0, "detail": "4 speakers"},
    {"name": "translating", "secs": 27.2, "model": "HY-MT1.5-7B-GGUF:Q4_K_M",
     "device": "cuda", "load_secs": 0.0, "detail": "612 segs → en,fr"},
]


def _block(app_module, **kw):
    base = dict(file_label="x.wav", model_name="m", info=FakeInfo(duration=15.8),
                kwargs={"beam_size": 10}, seg_diag=_SEG, raw="", final="")
    base.update(kw)
    return app_module._format_request_block(**base)


# ---------------------------------------------------------------------------
# Decode-only receipts are untouched
# ---------------------------------------------------------------------------

def test_no_stage_sections_without_stages(app_module):
    """The regression that matters most: a plain transcription must not grow
    a single new line, or every existing log and every operator's eye for
    them is invalidated at once."""
    block = _block(app_module)
    for label in ("Pipeline", "Separation", "Diarization", "Translation",
                  "Notes"):
        assert f"─── {label}" not in block
    assert "spk" not in block.splitlines()[0]


def test_stage_sections_appear_only_for_stages_that_ran(app_module):
    block = _block(app_module, stages=_STAGES,
                   diarization={"model": "pyannote/speaker-diarization-3.1",
                                "num_speakers": None, "min_speakers": 2,
                                "max_speakers": 6})
    assert "─── Diarization" in block
    # Asked for nothing else, so nothing else is claimed.
    assert "─── Separation" not in block
    assert "─── Translation" not in block


# ---------------------------------------------------------------------------
# Pipeline table
# ---------------------------------------------------------------------------

def test_pipeline_table_lists_every_stage_in_order(app_module):
    block = _block(app_module, stages=_STAGES)
    rows = [l for l in block.splitlines()
            if l.startswith("     ") and any(
                s in l for s in ("separating", "transcribing", "diarizing",
                                 "translating"))]
    assert len(rows) == 4
    order = [l.split()[1] for l in rows]
    assert order == ["separating", "transcribing", "diarizing", "translating"]


def test_pipeline_table_splits_load_from_run(app_module):
    """`load 0.0s` is the receipt's own proof that preloading worked, so the
    column has to survive a zero rather than render as blank."""
    block = _block(app_module, stages=_STAGES)
    sep = next(l for l in block.splitlines() if "separating" in l)
    assert "3.1s" in sep and "99.2s" in sep     # 102.3 wall - 3.1 load
    whi = next(l for l in block.splitlines() if "transcribing" in l)
    assert "0.0s" in whi


def test_vad_renders_as_its_own_stage(app_module):
    """VAD is the stage that most often eats the user's audio and it had no
    row anywhere. transcribe() runs Silero eagerly before handing back the
    generator, so the cost is genuinely separable from the decode."""
    stages = [
        {"name": "vad", "secs": 1.2, "model": "silero", "device": "cpu",
         "detail": "audio decode + Silero · 13.24s kept of 13.56s (98 %)"},
        {"name": "transcribing", "secs": 12.4, "model": "large-v2",
         "device": "cuda", "load_secs": 0.0},
    ]
    block = _block(app_module, stages=stages)
    row = next(l for l in block.splitlines() if " vad " in l)
    assert "silero" in row and "cpu" in row and "1.2s" in row
    assert "98 %" in block
    # VAD sorts ahead of the decode it precedes.
    names = [l.split()[1] for l in block.splitlines()
             if l.startswith("     ") and ("vad" in l or "transcribing" in l)]
    assert names == ["vad", "transcribing"]


def test_pipeline_header_says_stage_count_and_wall(app_module):
    block = _block(app_module, stages=_STAGES)
    head = next(l for l in block.splitlines() if "─── Pipeline" in l)
    assert "4 stages" in head
    assert "6:12" in head                        # 372.3s total


# ---------------------------------------------------------------------------
# Speaker column (only when diarization produced labels)
# ---------------------------------------------------------------------------

def test_speaker_column_only_with_speakers(app_module):
    plain = _block(app_module)
    assert "spk" not in plain

    withspk = _block(app_module, speakers=["SPEAKER_00", "SPEAKER_01"])
    header = next(l for l in withspk.splitlines() if "alp" in l and "text" in l)
    assert "spk" in header
    assert "S0" in withspk and "S1" in withspk


def test_speaker_column_tolerates_short_speaker_list(app_module):
    """assign_speakers can return fewer labels than segments; a receipt must
    never be the thing that raises."""
    block = _block(app_module, speakers=["SPEAKER_00"])
    assert "S0" in block


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

def test_notes_lists_warnings_and_skips(app_module):
    block = _block(app_module,
                   warnings=["cuda requested but onnxruntime ran on CPU"],
                   skipped=["diarizing"])
    assert "─── Notes" in block
    assert "[!] cuda requested" in block
    assert "( ) diarizing" in block


def test_notes_never_uses_the_dropped_segment_glyph(app_module):
    """`✗` belongs to the segments table's guard-dropped rows. A skipped
    stage borrowing it would break the "no ✗ without guards" invariant that
    test_hallucination_guards pins."""
    block = _block(app_module, skipped=["translating"], warnings=["nope"])
    assert "✗" not in block


def test_notes_absent_when_nothing_to_say(app_module):
    assert "─── Notes" not in _block(app_module, warnings=[], skipped=[])


# ---------------------------------------------------------------------------
# Segment row cap
# ---------------------------------------------------------------------------

def test_segment_rows_honour_the_configured_cap(app_module, monkeypatch):
    """The tail line has to name the config knob: rows the FILE never got are
    rows the viewer can never reveal, which is exactly why the old inert
    "(+N more)" tail expanded into nothing."""
    monkeypatch.setattr(app_module.cfg, "LOG_SEGMENT_ROWS_MAX", 1)
    block = _block(app_module)
    assert "Guten Morgen" in block
    assert "Ja, danke" not in block
    assert "+1 more, not logged" in block
    assert "LOG_SEGMENT_ROWS_MAX" in block


def test_zero_cap_means_unlimited(app_module, monkeypatch):
    monkeypatch.setattr(app_module.cfg, "LOG_SEGMENT_ROWS_MAX", 0)
    block = _block(app_module)
    assert "Guten Morgen" in block and "Ja, danke" in block
    assert "not logged" not in block


# ---------------------------------------------------------------------------
# The `*` marker now reaches stage params
# ---------------------------------------------------------------------------

def test_stage_params_can_be_marked_non_default(app_module):
    """_is_non_default is a whitelist keyed by _KWARG_TO_CFG; without the
    stage entries a diarization knob could stray arbitrarily far from the
    config and still print unmarked."""
    baseline = getattr(app_module.cfg, "_BASELINE", {})
    assert "DIARIZATION_MIN_SPEAKERS" in baseline
    assert app_module._KWARG_TO_CFG["min_speakers"] == "DIARIZATION_MIN_SPEAKERS"

    block = _block(app_module, diarization={
        "model": "pyannote/speaker-diarization-3.1",
        "min_speakers": (baseline["DIARIZATION_MIN_SPEAKERS"] or 0) + 3,
    })
    row = next(l for l in block.splitlines() if "min_speakers" in l)
    assert row.rstrip().endswith("*")
