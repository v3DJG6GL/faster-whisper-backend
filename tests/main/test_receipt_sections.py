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

import re

from tests.conftest import FakeInfo

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
    assert "spk" not in block.splitlines()[1]


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
        {"name": "transcribing", "secs": 12.4, "model": "large-v2",
         "device": "cuda", "load_secs": 0.0},
        {"name": "vad", "secs": 1.2, "model": "silero", "device": "cpu",
         "detail": "audio decode + Silero · 13.24s kept of 13.56s (98 %)"},
    ]
    block = _block(app_module, stages=stages)
    row = next(l for l in block.splitlines() if " vad " in l)
    assert "silero" in row and "cpu" in row and "1.2s" in row
    assert "98 %" in block
    # VAD sorts ahead of the decode it precedes even when input is unordered.
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

# ---------------------------------------------------------------------------
# The /logs viewer parses this block with JS regexes. Nothing else connects
# the two, so a renamed section or a re-spaced row breaks colouring silently
# and nobody finds out until someone opens the page.
# ---------------------------------------------------------------------------

def test_viewer_section_regex_matches_every_section(app_module):
    block = _block(app_module, stages=_STAGES,
                   separation={"model": "UVR.onnx", "device": "cuda"},
                   diarization={"model": "pyannote/x", "min_speakers": 2},
                   translation={"model": "HY", "targets": ["en", "fr"],
                                "mode": "fluent"},
                   guards={"max_words_per_second": 8.0},
                   warnings=["w"], skipped=["s"])
    # Mirrors _SEC_RE in the viewer's decorate().
    sec = re.compile(r"^\s+─── ([A-Za-z][A-Za-z \-]*?)(?:\s\s|\s─)")
    found = [m.group(1) for m in
             (sec.match(l) for l in block.splitlines()) if m]
    for label in ("Pipeline", "Audio", "Separation", "Diarization",
                  "Translation", "Post-decode guards", "Decode params",
                  "Segments", "Notes"):
        assert label in found, f"{label} no longer matches the viewer regex"


def test_viewer_segment_row_regex_skips_the_pipeline_table(app_module):
    """Both tables are indented numeric rows. If the segment-row pattern also
    matched Pipeline rows, the fold control would hide stage timings."""
    block = _block(app_module, stages=_STAGES)
    seg = re.compile(r"^\s+\d+\s+[-\d]")
    matched = [l for l in block.splitlines() if seg.match(l)]
    assert len(matched) == len(_SEG)
    assert not any("separating" in l or "transcribing" in l for l in matched)


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


def test_speaker_labels_stay_on_the_kept_rows_after_a_drop(app_module):
    """seg_diag keeps the segment the word-rate guard dropped; assign_speakers
    labels only the kept ones. The receipt must re-index the labels to the
    diag rows — one dropped row used to shift every later speaker by one."""
    diag = [
        dict(_SEG[0]),
        {"id": 1, "start": 13.44, "end": 14.00, "alp": -0.9, "nsp": 0.3,
         "cr": 2.5, "temp": 1.0, "text": "gibberish gibberish", "dropped": True},
        dict(_SEG[1], id=2),
    ]
    aligned = app_module._align_speakers_to_diag(diag, ["SPEAKER_00", "SPEAKER_01"])
    assert aligned == ["SPEAKER_00", "", "SPEAKER_01"]
    block = _block(app_module, seg_diag=diag, speakers=aligned)
    rows = {int(l.split()[0]): l for l in block.splitlines()
            if l.strip()[:1].isdigit() and "s  " in l}
    assert "S0" in rows[0]
    assert "S0" not in rows[1] and "S1" not in rows[1]   # the dropped row
    assert "S1" in rows[2]
    # Nothing to label keeps the column off entirely.
    assert app_module._align_speakers_to_diag(diag, []) is None

    # The helper consumes one label PER KEPT SEGMENT, so the caller must feed
    # it the per-segment labels, never the distinct list: 3 kept rows over 2
    # speakers still label every row (feeding the 2 distinct labels would
    # leave the third row blank).
    diag3 = diag + [dict(_SEG[1], id=3, start=19.10, end=22.00)]
    aligned3 = app_module._align_speakers_to_diag(
        diag3, ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])
    assert aligned3 == ["SPEAKER_00", "", "SPEAKER_01", "SPEAKER_00"]


def test_downloading_stage_leads_the_pipeline_table(app_module):
    """A URL fetch can dominate wall time; it renders first and is counted in
    the header's stage/wall totals like any other stage."""
    stages = [{"name": "downloading", "secs": 30.0, "detail": "Youtube"},
              *_STAGES]
    block = _block(app_module, stages=stages)
    lines = block.splitlines()
    head = next(l for l in lines if "─── Pipeline" in l)
    assert "(5 stages" in head
    total = 30.0 + sum(s["secs"] for s in _STAGES)
    mins, secs = divmod(int(total), 60)
    assert f"{mins}:{secs:02d} wall" in head
    rows = [l for l in lines if re.match(r"^\s+\d+\s+\w+\s", l)
            and any(s["name"] in l for s in stages)]
    assert rows[0].split()[:2] == ["1", "downloading"]
    assert "Youtube" in lines[lines.index(rows[0]) + 1]


def test_task_renders_in_decode_params_with_the_non_default_marker(app_module):
    """`task` is the one kwarg that changes the output language, yet it was
    missing from both the order tuple and _KWARG_TO_CFG, so a translate run's
    decode block was byte-identical to a plain transcription's."""
    baseline = getattr(app_module.cfg, "_BASELINE", {})
    assert baseline.get("TASK") == "transcribe"
    rows = app_module._format_decode_params({"task": "translate", "beam_size": 5})
    assert rows[0].split()[0] == "task"
    assert "translate" in rows[0]
    assert rows[0].rstrip().endswith("*")
    assert not any(r.split()[0] == "task"
                   for r in app_module._format_decode_params({"beam_size": 5}))
