"""The server-owned plan behind a batch run's progress: which stages will
run, what each is expected to cost, what each actually cost, and — derived
from those — one overall fraction and one ETA for the whole run.

Why the server: the client used to weight stages with fixed constants and a
seeded realtime factor, so a three-language translation parked the bar at
91 % for eight minutes. The server knows the stage list, the model and
device of every stage, the audio duration, how much of it the VAD kept, the
segment count and the target list — and, through runtime/stage_rates.py,
how fast each of those went last time. The client renders; it no longer
guesses.

Lifecycle: one RunPlan per request, built at request entry with the stages
the Form arguments imply, refined as facts land (probe duration → decoder
duration → VAD retained → segment count), fed every progress tick through
tick(), and read by the progress route through snapshot(). finish_run()
teaches the ledger from the stages that completed.

Threading: setters and tick() may be called from executor threads (the
decode, the demix, the pyannote hook all report from there); every public
method takes the instance lock. snapshot() runs on the loop thread — it is
the one writer of the monotonic hold.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from faster_whisper_backend.runtime import stage_rates

STAGES: tuple[str, ...] = (
    "downloading", "separating", "transcribing", "diarizing", "translating")

# Registry stage names that are sub-phases of a plan stage, never stages of
# their own. A tick naming one advances nothing; it only labels the phase.
SUBPHASE_OF: dict[str, str] = {
    "resolving": "downloading",
    "waiting": "transcribing",
    "analyzing": "transcribing",
    "loading": "translating",
}

# Priors before a driver is known, so the denominator never has a hole: an
# uploaded file is assumed to be 128 kbps audio (the handler passes
# bytes / BYTES_PER_AUDIO_SECOND as a "bytes-prior" duration), and a
# transcript is assumed to produce one segment per ~7 s of audio.
BYTES_PER_AUDIO_SECOND = 16_000.0
_AUDIO_SECONDS_PER_SEGMENT = 7.0

# Rank of the audio-duration sources: a stronger one is never overwritten
# by a weaker one arriving late.
_DURATION_RANK = {"bytes-prior": 0, "probe": 1, "decoder": 2}

# The bar never claims done before the run is: an active stage running past
# its estimate slows the bar, it does not roll it back.
_OVERALL_CAP = 0.99
# A stage fraction has to be this far along, and this old, before its own
# rate (elapsed / fraction) is a better remaining-time guess than the estimate.
_PROJECT_MIN_FRAC = 0.05
_PROJECT_MIN_ELAPSED_S = 10.0
# Below this wall time a finished stage teaches nothing (timer noise).
_MIN_SAMPLE_S = 0.5


def same_lang(a: str | None, b: str | None) -> bool:
    """Base-subtag compare ("pt-BR" == "pt"); False when either is empty.
    Mirrors audio/translation._same_lang without importing that module
    (it drags the llama.cpp family tables in)."""
    if not a or not b:
        return False
    return a.split("-")[0].lower() == b.split("-")[0].lower()


@dataclass
class Unit:
    """One translation target."""
    target: str
    instant: bool = False          # same-language verbatim copy: no model call
    est_s: float | None = None
    started: float | None = None   # time.monotonic()
    took_s: float | None = None
    progress: float = 0.0
    state: str = "queued"          # queued | running | done | instant

    def to_dict(self, now: float) -> dict:
        out: dict = {"target": self.target, "state": self.state}
        if self.instant:
            out["instant"] = True
        if self.state == "running":
            out["progress"] = round(self.progress, 4)
            if self.started is not None:
                out["elapsed_s"] = round(now - self.started, 1)
            if self.est_s is not None:
                out["est_s"] = round(self.est_s, 1)
        elif self.state == "queued":
            if self.est_s is not None and not self.instant:
                out["est_s"] = round(self.est_s, 1)
        elif self.took_s is not None:
            out["took_s"] = round(self.took_s, 1)
        return out


@dataclass
class Stage:
    name: str
    state: str = "pending"         # pending | active | done | failed | skipped
    model: str | None = None
    device: str | None = None
    compute: str | None = None     # whisper compute type / translation mode
    quantity: float | None = None  # the driver: bytes, audio seconds, segments
    est_s: float | None = None
    est_src: str = "seed"
    started: float | None = None
    ended: float | None = None
    took_s: float | None = None
    wait_s: float = 0.0            # semaphore queue time, not stage work
    wait_started: float | None = None
    frac: float | None = None      # last stage-local fraction from a tick
    phase: str | None = None       # active sub-phase label
    units: list[Unit] | None = None

    def elapsed(self, now: float) -> float:
        if self.started is None:
            return 0.0
        return max(0.0, (self.ended if self.ended is not None else now)
                   - self.started)


class RunPlan:
    def __init__(self, *, kind: str = "file", now=time.monotonic) -> None:
        self.kind = kind
        self._now = now
        self._lock = threading.Lock()
        self._stages: list[Stage] = []
        self._audio_s: float | None = None
        self._audio_src: str = ""
        self._vad_retained: float | None = None
        self._download_bytes: float | None = None
        self._extractor: str | None = None
        self._n_segments: int | None = None
        self._source_lang: str | None = None
        self._hold: float = 0.0
        self._finished = False

    # ── configuration / refinement ──────────────────────────────────────

    def set_stages(self, names: list[str]) -> None:
        """The stages this run will execute, in pipeline order. Provisional
        at request entry, final once every enable/allowlist verdict has
        landed; a stage that already started is never dropped."""
        with self._lock:
            keep = {s.name: s for s in self._stages}
            order = [n for n in STAGES if n in names]
            new: list[Stage] = []
            for n in order:
                new.append(keep.pop(n, None) or Stage(name=n))
            for s in keep.values():
                if s.state != "pending":
                    new.append(s)
            new.sort(key=lambda s: STAGES.index(s.name))
            self._stages = new
            self._recompute()

    def set_stage_model(self, name: str, *, model: str | None = None,
                        device: str | None = None,
                        compute: str | None = None) -> None:
        with self._lock:
            st = self._get(name)
            if st is None:
                return
            if model is not None:
                st.model = model
            if device is not None:
                st.device = device
            if compute is not None:
                st.compute = compute
            self._recompute()

    def set_audio_seconds(self, secs: float | None, *, src: str) -> None:
        if not secs or secs <= 0:
            return
        with self._lock:
            if (_DURATION_RANK.get(src, -1)
                    < _DURATION_RANK.get(self._audio_src, -1)):
                return
            self._audio_s = float(secs)
            self._audio_src = src
            self._recompute()

    def set_vad_retained(self, frac: float | None) -> None:
        if frac is None:
            return
        with self._lock:
            self._vad_retained = max(0.0, min(1.0, float(frac)))
            self._recompute()

    def set_download_bytes(self, n: int | float | None, *,
                           extractor: str | None = None) -> None:
        with self._lock:
            if extractor:
                self._extractor = extractor
            if n and n > 0:
                self._download_bytes = float(n)
            self._recompute()

    def set_segments(self, n: int | None) -> None:
        if n is None or n < 0:
            return
        with self._lock:
            self._n_segments = int(n)
            self._recompute()

    def set_translation(self, targets: list[str], *,
                        model: str | None = None, device: str | None = None,
                        mode: str | None = None,
                        source_lang: str | None = None) -> None:
        with self._lock:
            st = self._get("translating")
            if st is None:
                return
            st.model = model if model is not None else st.model
            st.device = device if device is not None else st.device
            st.compute = mode if mode is not None else st.compute
            st.units = [Unit(target=t) for t in targets]
            if source_lang:
                self._source_lang = source_lang
            self._mark_instant_locked()
            self._recompute()

    def mark_instant(self, source_lang: str | None) -> None:
        """Targets equal to the source language are copied verbatim by the
        translation stage — no model call, no measurable time."""
        with self._lock:
            if source_lang:
                self._source_lang = source_lang
            self._mark_instant_locked()
            self._recompute()

    def skip(self, name: str) -> None:
        with self._lock:
            st = self._get(name)
            if st is not None and st.state in ("pending", "active"):
                st.state = "skipped"
            self._recompute()

    def stage_failed(self, name: str) -> None:
        with self._lock:
            self._close(name, "failed", None)

    def stage_done(self, name: str, *, took_s: float | None = None) -> None:
        with self._lock:
            self._close(name, "done", took_s)

    # ── observation ────────────────────────────────────────────────────

    def tick(self, *, stage: str | None, progress: float | None = None,
             target: str | None = None,
             target_progress: float | None = None,
             total_bytes: float | None = None) -> None:
        """One progress-registry write. Advances the plan when the tick
        names a stage that has not started, labels a sub-phase otherwise,
        and moves the translation units along with `target`."""
        if not stage:
            return
        name = SUBPHASE_OF.get(stage, stage)
        phase = stage if stage != name else None
        with self._lock:
            now = self._now()
            st = self._get(name)
            active = self._active()
            if st is None or st.state in ("done", "failed", "skipped"):
                # e.g. a cold-model "downloading" while translating: label
                # the active stage, never rewind to a finished one.
                if active is not None and name != active.name:
                    active.phase = stage
                elif active is not None:
                    active.phase = phase
                return
            if st.state == "pending":
                earlier_pending = any(
                    s.state == "pending" for s in self._stages
                    if STAGES.index(s.name) < STAGES.index(st.name))
                if phase is not None and earlier_pending:
                    # The handler seeds "waiting" at request entry, before
                    # separation has had its turn: a sub-phase is a label,
                    # not evidence that the stages before it were skipped.
                    return
                if active is not None:
                    self._close_locked(active, "done", None, now)
                # Anything earlier still pending never ran.
                for s in self._stages:
                    if s is st:
                        break
                    if s.state == "pending":
                        s.state = "skipped"
                st.state = "active"
                st.started = now
                st.frac = None
            # waiting is queue time, billed apart from the stage's work
            if phase == "waiting":
                if st.wait_started is None:
                    st.wait_started = now
            elif st.wait_started is not None:
                st.wait_s += now - st.wait_started
                st.wait_started = None
            st.phase = phase
            if progress is not None:
                st.frac = max(0.0, min(1.0, float(progress)))
            if st.name == "downloading" and total_bytes and total_bytes > 0:
                self._download_bytes = float(total_bytes)
            if st.name == "translating" and target and st.units:
                self._advance_units_locked(st, target, target_progress, now)
            self._recompute()

    # ── output ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            now = self._now()
            plan = [self._stage_dict(s, now) for s in self._stages]
            overall = self._overall_locked(now)
            eta = self._eta_locked(now)
        return {"plan": plan, "overall": overall, "eta_s": eta}

    def finish_run(self, status: str) -> None:
        """Teach the ledger from the stages that ran to completion. Only a
        clean run: a cancelled or failed one has stages whose wall time
        says nothing about their rate."""
        with self._lock:
            if self._finished:
                return
            self._finished = True
            if status != "ok":
                return
            for st in self._stages:
                if st.state != "done" or st.took_s is None:
                    continue
                if st.name == "translating":
                    n = self._segments_for_estimate()
                    for u in st.units or []:
                        if u.instant or u.took_s is None or not n:
                            continue
                        if u.took_s >= _MIN_SAMPLE_S:
                            stage_rates.record("translating", st.model,
                                               st.device, st.compute,
                                               n / u.took_s)
                    continue
                work = st.took_s - st.wait_s
                q = self._quantity_now(st)
                if q and work >= _MIN_SAMPLE_S:
                    stage_rates.record(st.name, st.model, st.device,
                                       st.compute, q / work)

    # ── internals (lock held) ──────────────────────────────────────────

    def _get(self, name: str) -> Stage | None:
        for s in self._stages:
            if s.name == name:
                return s
        return None

    def _active(self) -> Stage | None:
        for s in self._stages:
            if s.state == "active":
                return s
        return None

    def _close(self, name: str, state: str, took_s: float | None) -> None:
        st = self._get(name)
        if st is None or st.state in ("done", "failed", "skipped"):
            return
        self._close_locked(st, state, took_s, self._now())
        self._recompute()

    def _close_locked(self, st: Stage, state: str, took_s: float | None,
                      now: float) -> None:
        if st.started is None:
            st.started = now
        if st.wait_started is not None:
            st.wait_s += now - st.wait_started
            st.wait_started = None
        st.ended = now
        st.state = state
        st.phase = None
        st.took_s = float(took_s) if took_s is not None else now - st.started
        if state == "done":
            st.frac = 1.0
        for u in st.units or []:
            if u.state in ("queued", "running"):
                u.took_s = (now - u.started) if u.started is not None else 0.0
                u.state = "done"
                u.progress = 1.0

    def _advance_units_locked(self, st: Stage, target: str,
                              target_progress: float | None,
                              now: float) -> None:
        units = st.units or []
        idx = next((i for i, u in enumerate(units) if u.target == target), None)
        if idx is None:
            return
        for u in units[:idx]:
            if u.state in ("queued", "running"):
                u.took_s = (now - u.started) if u.started is not None else 0.0
                u.state = "instant" if u.instant else "done"
                u.progress = 1.0
        u = units[idx]
        if u.state in ("queued",):
            u.started = now
            u.state = "running"
        if target_progress is not None:
            u.progress = max(0.0, min(1.0, float(target_progress)))
        if u.instant and u.progress >= 1.0:
            u.state = "instant"
            u.took_s = 0.0

    def _mark_instant_locked(self) -> None:
        st = self._get("translating")
        if st is None or not st.units:
            return
        for u in st.units:
            if u.state == "queued":
                u.instant = same_lang(u.target, self._source_lang)

    def _quantity_now(self, st: Stage) -> float | None:
        """The stage's cost driver from the facts known at the END of the
        run: separation runs before the decoder measured the audio, so its
        own frozen quantity would be the upload-size prior."""
        if st.name == "downloading":
            return self._download_bytes
        if st.name == "transcribing":
            if not self._audio_s:
                return None
            return self._audio_s * (self._vad_retained
                                    if self._vad_retained is not None else 1.0)
        return self._audio_s

    def _segments_for_estimate(self) -> float | None:
        if self._n_segments is not None:
            return float(self._n_segments)
        if self._audio_s:
            return max(1.0, self._audio_s / _AUDIO_SECONDS_PER_SEGMENT)
        return None

    def _recompute(self) -> None:
        """Estimates for every stage that has not finished."""
        audio = self._audio_s
        for st in self._stages:
            if st.state in ("done", "failed", "skipped"):
                continue
            q: float | None = None
            if st.name == "downloading":
                q = self._download_bytes
                key_model, key_dev, key_comp = self._extractor, None, None
            elif st.name == "transcribing":
                q = (audio * (self._vad_retained
                              if self._vad_retained is not None else 1.0)
                     if audio else None)
                key_model, key_dev, key_comp = st.model, st.device, st.compute
            elif st.name == "translating":
                self._recompute_units(st)
                continue
            else:
                q = audio
                key_model, key_dev, key_comp = st.model, st.device, None
            st.quantity = q
            if q is None or q <= 0:
                st.est_s, st.est_src = None, "unknown"
                continue
            rec = stage_rates.lookup(st.name, key_model, key_dev, key_comp)
            rate = rec.get("rate")
            if not rate:
                st.est_s, st.est_src = None, "unknown"
                continue
            st.est_s = q / float(rate)
            st.est_src = rec.get("src") or "seed"

    def _recompute_units(self, st: Stage) -> None:
        n = self._segments_for_estimate()
        st.quantity = n
        if not st.units:
            st.est_s = None
            return
        rec = stage_rates.lookup("translating", st.model, st.device, st.compute)
        rate = rec.get("rate")
        per_unit = (n / float(rate)) if (n and rate) else None
        st.est_src = rec.get("src") or "seed"
        total = 0.0
        known = True
        for u in st.units:
            if u.instant:
                u.est_s = 0.0
                continue
            if u.state in ("done", "instant") and u.took_s is not None:
                total += u.took_s
                continue
            u.est_s = per_unit
            if per_unit is None:
                known = False
            else:
                total += per_unit
        st.est_s = total if known else None

    # weight of a stage in the overall sum: what it cost, else what it should
    @staticmethod
    def _weight(st: Stage) -> float | None:
        if st.state in ("done", "failed"):
            return st.took_s
        return st.est_s

    def _fraction(self, st: Stage, now: float) -> float:
        if st.state in ("done", "failed"):
            return 1.0
        if st.state != "active":
            return 0.0
        if st.name == "translating" and st.units:
            tot = 0.0
            got = 0.0
            for u in st.units:
                w = (u.took_s if u.state in ("done", "instant") and u.took_s
                     else u.est_s)
                if u.instant:
                    continue
                if w is None:
                    continue
                tot += w
                if u.state in ("done", "instant"):
                    got += w
                elif u.state == "running":
                    got += w * u.progress
            if tot > 0:
                return got / tot
        if st.frac is not None:
            return st.frac
        if st.est_s:
            return min(st.elapsed(now) / st.est_s, 0.95)
        return 0.0

    def _overall_locked(self, now: float) -> float | None:
        live = [s for s in self._stages if s.state != "skipped"]
        if not live:
            return None
        if all(s.state in ("done", "failed") for s in live):
            self._hold = 1.0
            return 1.0
        total = 0.0
        got = 0.0
        for st in live:
            w = self._weight(st)
            if w is None:
                continue
            total += w
            got += w * self._fraction(st, now)
        raw = (got / total) if total > 0 else 0.0
        overall = max(self._hold, min(raw, _OVERALL_CAP))
        self._hold = overall
        return overall

    def _remaining(self, est: float | None, elapsed: float,
                   frac: float | None) -> float | None:
        if (frac is not None and frac >= _PROJECT_MIN_FRAC
                and elapsed >= _PROJECT_MIN_ELAPSED_S):
            return elapsed * (1.0 - frac) / frac
        if est is not None:
            if frac is not None:
                return max(est * (1.0 - frac), 0.0)
            if elapsed < est:
                return est - elapsed
        return None   # overrun with no evidence: the client says "estimating"

    def _eta_locked(self, now: float) -> float | None:
        eta = 0.0
        for st in self._stages:
            if st.state in ("done", "failed", "skipped"):
                continue
            if st.state == "pending":
                if st.est_s is None:
                    continue
                eta += st.est_s
                continue
            # active
            if st.name == "translating" and st.units:
                for u in st.units:
                    if u.instant or u.state in ("done", "instant"):
                        continue
                    if u.state == "queued":
                        if u.est_s is None:
                            continue
                        eta += u.est_s
                        continue
                    r = self._remaining(
                        u.est_s,
                        (now - u.started) if u.started is not None else 0.0,
                        u.progress)
                    if r is None:
                        return None
                    eta += r
                continue
            r = self._remaining(st.est_s, st.elapsed(now) - st.wait_s,
                                st.frac)
            if r is None:
                return None
            eta += r
        return eta

    def _stage_dict(self, st: Stage, now: float) -> dict:
        out: dict = {"stage": st.name, "state": st.state}
        for k in ("model", "device", "compute"):
            v = getattr(st, k)
            if v:
                out[k] = v
        if st.state == "active":
            if st.phase:
                out["phase"] = st.phase
            out["elapsed_s"] = round(st.elapsed(now), 1)
            if st.est_s is not None:
                out["est_s"] = round(st.est_s, 1)
        elif st.state == "pending":
            if st.est_s is not None:
                out["est_s"] = round(st.est_s, 1)
        elif st.state in ("done", "failed"):
            if st.took_s is not None:
                out["took_s"] = round(st.took_s, 1)
        if st.units is not None and st.state != "skipped":
            out["units"] = [u.to_dict(now) for u in st.units]
        return out
