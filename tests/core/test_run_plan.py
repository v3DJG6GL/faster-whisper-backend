"""run_plan: the server-owned stage plan behind the progress route.

A fake clock drives every test; the rates ledger is repointed to a temp
file so seeds are what the plan sees unless a test records something.
"""
import pytest

from faster_whisper_backend.core import run_plan
from faster_whisper_backend.runtime import stage_rates


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_rates, "PATH",
                        str(tmp_path / "stage_rates.json"), raising=False)
    stage_rates._reset_for_tests()
    yield
    stage_rates._reset_for_tests()


@pytest.fixture
def clock():
    return Clock()


def _plan(clock, kind="file", stages=("transcribing",)):
    p = run_plan.RunPlan(kind=kind, now=clock)
    p.set_stages(list(stages))
    return p


def _stage(snap, name):
    return next(s for s in snap["plan"] if s["stage"] == name)


# --- estimates ----------------------------------------------------------------

def test_seed_estimates_use_seed_rates(ledger, clock):
    p = _plan(clock, stages=["separating", "transcribing", "diarizing"])
    p.set_audio_seconds(600.0, src="decoder")
    snap = p.snapshot()
    assert _stage(snap, "separating")["est_s"] == pytest.approx(600 / 8, abs=0.06)
    assert _stage(snap, "transcribing")["est_s"] == pytest.approx(600 / 6, abs=0.06)
    assert _stage(snap, "diarizing")["est_s"] == pytest.approx(600 / 11, abs=0.06)
    # Nothing has run: the run is at 0 %, and the ETA is the whole plan.
    assert snap["overall"] == 0.0
    assert snap["eta_s"] == pytest.approx(600 / 8 + 600 / 6 + 600 / 11, abs=0.2)


def test_measured_rate_beats_the_seed(ledger, clock):
    stage_rates.record("transcribing", "large-v3", "cuda", "float16", 12.0)
    p = _plan(clock)
    p.set_stage_model("transcribing", model="large-v3", device="cuda",
                      compute="float16")
    p.set_audio_seconds(600.0, src="decoder")
    assert _stage(p.snapshot(), "transcribing")["est_s"] == pytest.approx(50)


def test_duration_sources_refine_but_never_regress(ledger, clock):
    p = _plan(clock)
    p.set_audio_seconds(100.0, src="bytes-prior")
    assert _stage(p.snapshot(), "transcribing")["est_s"] == pytest.approx(100 / 6, abs=0.06)
    p.set_audio_seconds(300.0, src="decoder")
    assert _stage(p.snapshot(), "transcribing")["est_s"] == pytest.approx(300 / 6, abs=0.06)
    p.set_audio_seconds(50.0, src="probe")     # weaker source, late: ignored
    assert _stage(p.snapshot(), "transcribing")["est_s"] == pytest.approx(300 / 6, abs=0.06)


def test_vad_retained_scales_the_transcribe_estimate(ledger, clock):
    p = _plan(clock)
    p.set_audio_seconds(600.0, src="decoder")
    p.set_vad_retained(0.5)
    assert _stage(p.snapshot(), "transcribing")["est_s"] == pytest.approx(300 / 6, abs=0.06)


def test_translation_units_scale_with_targets_and_skip_instant(ledger, clock):
    p = _plan(clock, stages=["transcribing", "translating"])
    p.set_audio_seconds(700.0, src="decoder")
    p.set_translation(["en", "fr", "de"], model="org/m:Q4", device="cuda",
                      mode="fluent", source_lang="de")
    # 700 s / 7 s per segment = 100 segments; seed 1.6 units/s → 62.5 s each
    tr = _stage(p.snapshot(), "translating")
    assert tr["est_s"] == pytest.approx(2 * 100 / 1.6)
    by = {u["target"]: u for u in tr["units"]}
    assert by["de"]["instant"] is True and "est_s" not in by["de"]
    assert by["en"]["est_s"] == pytest.approx(62.5)
    # The real segment count replaces the prior.
    p.set_segments(40)
    assert _stage(p.snapshot(), "translating")["est_s"] == pytest.approx(2 * 40 / 1.6)


def test_download_estimate_from_bytes(ledger, clock):
    p = _plan(clock, kind="url", stages=["downloading", "transcribing"])
    p.set_download_bytes(30_000_000, extractor="Youtube")
    assert _stage(p.snapshot(), "downloading")["est_s"] == pytest.approx(10.0)


# --- progression --------------------------------------------------------------

def test_took_replaces_est_and_overall_is_monotone(ledger, clock):
    p = _plan(clock, stages=["transcribing", "translating"])
    p.set_audio_seconds(600.0, src="decoder")           # transcribe est 100 s
    p.set_translation(["fr"], model="m", device="cuda", mode="fluent")
    p.set_segments(160)                                 # translate est 100 s
    p.tick(stage="transcribing", progress=0.0)
    clock.advance(50)
    p.tick(stage="transcribing", progress=0.5)
    s1 = p.snapshot()
    assert s1["overall"] == pytest.approx(0.25)
    # The stage overruns badly: 300 s instead of 100. The bar must not
    # roll back when the measured weight lands.
    clock.advance(250)
    p.tick(stage="transcribing", progress=1.0)
    p.stage_done("transcribing")
    s2 = p.snapshot()
    assert _stage(s2, "transcribing")["took_s"] == pytest.approx(300)
    assert s2["overall"] >= s1["overall"]
    p.tick(stage="translating", progress=0.0, target="fr", target_progress=0.0)
    clock.advance(50)
    p.tick(stage="translating", progress=0.5, target="fr", target_progress=0.5)
    s3 = p.snapshot()
    assert s3["overall"] >= s2["overall"]
    assert s3["overall"] < 1.0
    p.stage_done("translating")
    assert p.snapshot()["overall"] == 1.0


def test_subphase_ticks_do_not_transition(ledger, clock):
    p = _plan(clock, stages=["separating", "transcribing", "translating"])
    p.set_audio_seconds(60.0, src="decoder")
    # The handler's entry seed: "waiting" while separation is still pending.
    p.tick(stage="waiting")
    snap = p.snapshot()
    assert _stage(snap, "separating")["state"] == "pending"
    assert _stage(snap, "transcribing")["state"] == "pending"
    p.tick(stage="separating")
    p.stage_done("separating", took_s=5.0)
    p.tick(stage="waiting")
    p.tick(stage="analyzing")
    snap = p.snapshot()
    assert _stage(snap, "transcribing")["state"] == "active"
    assert _stage(snap, "transcribing")["phase"] == "analyzing"
    p.tick(stage="transcribing", progress=0.2)
    assert "phase" not in _stage(p.snapshot(), "transcribing")
    p.stage_done("transcribing")
    p.set_translation(["fr"], model="m", device="cuda", mode="fluent")
    p.tick(stage="translating", progress=0.0)
    # A cold GGUF fetch reports "downloading" mid-translation: the plan has
    # no download stage, so it becomes a phase label, never a rewind.
    p.tick(stage="downloading", progress=0.4)
    snap = p.snapshot()
    assert _stage(snap, "translating")["state"] == "active"
    assert _stage(snap, "translating")["phase"] == "downloading"
    assert [s["stage"] for s in snap["plan"]] == \
        ["separating", "transcribing", "translating"]


def test_waiting_time_is_billed_apart_from_the_stage(ledger, clock):
    p = _plan(clock)
    p.set_audio_seconds(600.0, src="decoder")
    p.tick(stage="waiting")            # queue + model load
    clock.advance(40)
    p.tick(stage="transcribing", progress=0.0)
    clock.advance(60)
    p.tick(stage="transcribing", progress=1.0)
    p.stage_done("transcribing")
    p.finish_run("ok")
    # 600 audio seconds over 60 s of WORK (not 100 s of wall) = 10× realtime
    assert stage_rates.lookup("transcribing", None, None)["rate"] == \
        pytest.approx(10.0)


def test_translating_units_run_in_order_and_learn_per_unit(ledger, clock):
    p = _plan(clock, stages=["translating"], kind="text")
    p.set_segments(80)
    p.set_translation(["en", "fr", "fi"], model="org/m:Q4", device="cuda",
                      mode="faithful", source_lang="en")
    p.tick(stage="translating", progress=0.0, target="en", target_progress=1.0)
    by = {u["target"]: u for u in _stage(p.snapshot(), "translating")["units"]}
    assert by["en"]["state"] == "instant"
    assert by["fr"]["state"] == "queued"
    p.tick(stage="translating", progress=0.4, target="fr", target_progress=0.2)
    clock.advance(20)
    p.tick(stage="translating", progress=0.5, target="fr", target_progress=0.5)
    by = {u["target"]: u for u in _stage(p.snapshot(), "translating")["units"]}
    assert by["fr"]["state"] == "running"
    assert by["fr"]["progress"] == 0.5
    assert by["fr"]["elapsed_s"] == 20.0
    clock.advance(20)
    p.tick(stage="translating", progress=0.7, target="fi", target_progress=0.0)
    by = {u["target"]: u for u in _stage(p.snapshot(), "translating")["units"]}
    assert by["fr"]["state"] == "done" and by["fr"]["took_s"] == 40.0
    assert by["fi"]["state"] == "running"
    clock.advance(10)
    p.stage_done("translating")
    p.finish_run("ok")
    # fr: 80 units / 40 s = 2.0; fi: 80 / 10 = 8.0 → EWMA 5.0; en: nothing.
    assert stage_rates.lookup("translating", "org/m:Q4", "cuda",
                              "faithful")["rate"] == pytest.approx(5.0)


def test_eta_projects_from_rate_and_goes_null_on_unevidenced_overrun(
        ledger, clock):
    p = _plan(clock, stages=["transcribing", "diarizing"])
    p.set_audio_seconds(600.0, src="decoder")   # transcribe 100 s, diarize ~54.5 s
    p.tick(stage="transcribing", progress=0.0)
    clock.advance(20)
    p.tick(stage="transcribing", progress=0.1)   # 20 s for 10 % → 180 s left
    snap = p.snapshot()
    assert snap["eta_s"] == pytest.approx(180 + 600 / 11, abs=0.2)
    # Diarization runs past its estimate with no fraction at all.
    clock.advance(200)
    p.stage_done("transcribing")
    p.tick(stage="diarizing")
    clock.advance(10)
    assert p.snapshot()["eta_s"] == pytest.approx(600 / 11 - 10, abs=0.2)
    clock.advance(100)
    assert p.snapshot()["eta_s"] is None


def test_skipped_and_failed_stages(ledger, clock):
    p = _plan(clock, stages=["separating", "transcribing", "diarizing"])
    p.set_audio_seconds(600.0, src="decoder")
    p.skip("separating")
    p.tick(stage="transcribing", progress=1.0)
    clock.advance(100)
    p.stage_done("transcribing")
    p.tick(stage="diarizing")
    clock.advance(30)
    p.stage_failed("diarizing")
    snap = p.snapshot()
    assert _stage(snap, "separating") == {"stage": "separating",
                                          "state": "skipped"}
    assert _stage(snap, "diarizing")["state"] == "failed"
    assert _stage(snap, "diarizing")["took_s"] == 30.0
    assert snap["overall"] == 1.0
    p.finish_run("ok")
    assert stage_rates.lookup("diarizing", None, None)["src"] == "seed"
    assert stage_rates.lookup("transcribing", None, None)["src"] == "measured"


def test_finish_run_records_nothing_unless_ok(ledger, clock):
    p = _plan(clock)
    p.set_audio_seconds(600.0, src="decoder")
    p.tick(stage="transcribing", progress=0.0)
    clock.advance(60)
    p.stage_done("transcribing")
    p.finish_run("cancelled")
    assert stage_rates.lookup("transcribing", None, None)["src"] == "seed"


def test_stage_ticks_infer_skipped_predecessors(ledger, clock):
    # The server jumped straight from separating to diarizing? Then the
    # transcribe stage was never run: it leaves the denominator.
    p = _plan(clock, stages=["separating", "transcribing", "diarizing"])
    p.set_audio_seconds(60.0, src="decoder")
    p.tick(stage="separating")
    p.stage_done("separating", took_s=3.0)
    p.tick(stage="diarizing")
    snap = p.snapshot()
    assert _stage(snap, "transcribing")["state"] == "skipped"
    assert _stage(snap, "diarizing")["state"] == "active"


def test_set_stages_final_keeps_started_stages(ledger, clock):
    p = _plan(clock, kind="url", stages=["downloading", "separating",
                                        "transcribing"])
    p.tick(stage="resolving")
    p.set_stages(["downloading", "transcribing", "translating"])
    snap = p.snapshot()
    assert [s["stage"] for s in snap["plan"]] == \
        ["downloading", "transcribing", "translating"]
    assert _stage(snap, "downloading")["state"] == "active"
    assert _stage(snap, "downloading")["phase"] == "resolving"


def test_empty_plan_snapshot(ledger, clock):
    p = run_plan.RunPlan(kind="file", now=clock)
    assert p.snapshot() == {"plan": [], "overall": None, "eta_s": 0.0}
