"""Background job: re-run the current PIPELINE_RULES over every
existing capture's `raw` text and update `final` + `text_for_training`.

Scope:
  - Touches `final` and `text_for_training` (the latter is rebuilt
    with the captures-specific `CAPTURES_PIPELINE_RULES_EXCLUDE` set,
    or copied from `final` when no excludes are configured).
    `corrected_text` (admin free-form ground truth) and
    `corrections` (chip corrections, index-based and
    rule-independent) stay untouched.
  - For each affected member that belongs to an unlocked group,
    rebuild the group's snapshot `transcript` from the current member
    text via samples._build_default_transcript. Locked
    groups are skipped — they're exported training samples.
  - No audio re-merge. Pipeline rules only affect text; merged WAV
    bytes are unchanged.

Single-worker model:
  - At most one job runs at a time. Concurrent start() returns the
    running job's current state and queues ONE more pass: the running
    pass snapshots the rules it started with (ident_cache), and the
    quick-config page auto-starts this job on every save with no manual
    re-apply button, so a save landing mid-run would otherwise never be
    applied. The worker re-walks every row before it reports "done".
  - Job state lives in process memory. A service restart wipes it
    and nothing resumes the walk: rows it had not reached keep the old
    rules until the next rules save, or until an admin runs "Reprocess
    all · Pipeline rules" from the /captures Advanced menu (same worker).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

from faster_whisper_backend.captures import samples as capture_samples
from faster_whisper_backend.pipeline import engine as pl_engine
from faster_whisper_backend.settings import effective_config

logger = logging.getLogger("whisper-api")

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "status":           "idle",     # idle | running | done | error
    "started_ts":       None,
    "finished_ts":      None,
    "total":            0,
    "processed":        0,
    "captures_updated": 0,
    "groups_updated":   0,
    "error":            None,
}
_worker: "threading.Thread | None" = None
# Set by a start() that found a pass running (under _state_lock); the worker
# consumes it and runs one more pass with the rules as they are by then.
_rerun_requested = False
# Ids rewritten so far in the current job (every pass of it). The reported
# totals are their sizes: a queued rerun walks every row again, and a row it
# finds already current must not erase pass 1's count — nor may a row
# rewritten in both passes count twice.
_updated_cids: set[str] = set()
_updated_sids: set[str] = set()


def final_text(raw: str, *, model: Any, ident: Any, language: Any) -> str:
    """A capture's `final` re-derived from its raw text with the owner's
    pipeline (`ident`) scoped by the TEXT language
    (captures_store.text_language). One recompute rule for the three
    callers: this job, the /reprocess route and the on-read self-heal in
    captures/routes.py — each keeps its own error handling."""
    return pl_engine._postprocess_text(
        raw, model_name=model, ident=ident, language=language,
    )


def training_text(raw: str, final: str, *, model: Any, ident: Any,
                  language: Any, excludes: Any) -> str:
    """`text_for_training`: the same pass minus the captures-specific
    `excludes` (CAPTURES_PIPELINE_RULES_EXCLUDE). With no excludes it is
    byte-identical to `final`, so the second pipeline pass is skipped."""
    if not excludes:
        return final
    return pl_engine._postprocess_text(
        raw, model_name=model, extra_excludes=excludes, ident=ident,
        language=language,
    )


def status() -> dict[str, Any]:
    with _state_lock:
        return dict(_state)


def start() -> dict[str, Any]:
    """Idempotent: if a job is running, return its current state
    instead of spawning a second worker."""
    global _worker, _rerun_requested
    with _state_lock:
        if _state["status"] == "running":
            _rerun_requested = True
            return dict(_state)
        _rerun_requested = False
        _updated_cids.clear()
        _updated_sids.clear()
        _state.update({
            "status":           "running",
            "started_ts":       time.time(),
            "finished_ts":      None,
            "total":            0,
            "processed":        0,
            "captures_updated": 0,
            "groups_updated":   0,
            "error":            None,
        })
    _worker = threading.Thread(target=_run, daemon=True, name="reapply-rules")
    _worker.start()
    with _state_lock:
        return dict(_state)


def _run() -> None:
    global _rerun_requested
    try:
        while True:
            _run_pass()
            with _state_lock:
                if not _rerun_requested:
                    _state["status"] = "done"
                    _state["finished_ts"] = time.time()
                    # Snapshot under the lock: a start() landing after the
                    # release resets _state to a fresh job's zeros.
                    summary = (_state["processed"], _state["total"],
                               _state["captures_updated"],
                               _state["groups_updated"])
                    break
                # Rules changed mid-pass: walk every row again with a fresh
                # snapshot. The progress bar restarts for this pass; the
                # updated totals keep counting for the whole job.
                _rerun_requested = False
                _state.update({"total": 0, "processed": 0})
            logger.info("[reapply] rules changed mid-run: re-applying again")
        logger.info(
            "[reapply] done: %d/%d captures, %d updated, %d groups", *summary,
        )
    except Exception as e:
        logger.exception("[reapply] job failed")
        with _state_lock:
            _rerun_requested = False
            _state["status"] = "error"
            _state["error"] = str(e)
            _state["finished_ts"] = time.time()


def _run_pass() -> None:
    """One walk over every capture with the rules as they are now. Raises on
    a job-level failure; _run turns that into the "error" state."""
    from faster_whisper_backend.captures import store as captures_store
    from faster_whisper_backend.captures import samples_store as capture_samples_store
    from faster_whisper_backend.settings import config as cfg

    captures_excludes = getattr(cfg, "CAPTURES_PIPELINE_RULES_EXCLUDE", None)

    conn = captures_store._require_conn()
    total_row = conn.execute("SELECT COUNT(*) FROM captures").fetchone()
    with _state_lock:
        _state["total"] = int(total_row[0]) if total_row else 0

    affected_sample_ids: set[str] = set()
    # Materialise the small projection up front — captures_store.update_capture
    # writes back to the same connection inside the loop, and an open
    # cursor on the same connection can skip/revisit rows when the
    # underlying table is mutated mid-walk. Payload is just ids +
    # short text columns (no words / segments), so memory
    # stays bounded even at tens of thousands of rows.
    rows = conn.execute(
        "SELECT id, raw_text AS raw, final_text AS final, text_for_training, model, sample_id, user_id, language, task"
        " FROM captures ORDER BY created_ts DESC"
    ).fetchall()
    # Reprocess re-runs ONLY the pipeline (no model re-decode), so it
    # resolves the owning USER's effective pipeline rules (no key — captures
    # store no key_id — and no per-request layer). Memoised per
    # (user_id, model) so a bulk reapply does one resolve per distinct pair,
    # not one per row. Config is snapshotted for the run's duration.
    ident_cache: dict = {}

    def _ident_for(uid, model_id):
        key = (uid, model_id)
        if key not in ident_cache:
            ident_cache[key] = effective_config.build_ident({"user_id": uid}, model_id)
        return ident_cache[key]

    for r in rows:
        cid = r["id"]
        raw_text = r["raw"] or ""
        patch: dict[str, str] = {}
        # Scope by the TEXT language ("en" for task=translate), as the
        # live run did — see captures_store.text_language.
        text_lang = captures_store.text_language(r)
        try:
            # Inside the per-row try: a resolve failure skips this row
            # instead of aborting the run before the group rebuild.
            ident = _ident_for(r["user_id"], r["model"])
            new_final = final_text(
                raw_text, model=r["model"], ident=ident, language=text_lang,
            )
        except Exception as e:
            logger.warning(
                "[reapply] capture %s skipped: %s", cid[:8], e,
            )
            with _state_lock:
                _state["processed"] += 1
            continue
        if new_final != (r["final"] or ""):
            patch["final"] = new_final
            if r["sample_id"]:
                affected_sample_ids.add(r["sample_id"])
        # Training-form text reflects PIPELINE_RULES minus the
        # captures-specific excludes (the excludes snapshotted for the run).
        try:
            new_training = training_text(
                raw_text, new_final, model=r["model"], ident=ident,
                language=text_lang, excludes=captures_excludes,
            )
        except Exception as e:
            logger.warning(
                "[reapply] capture %s training-form skipped: %s",
                cid[:8], e,
            )
            new_training = None
        if new_training is not None and new_training != (r["text_for_training"] or ""):
            patch["text_for_training"] = new_training
            # _build_default_transcript reads text_for_training before
            # falling back to final/raw, so a training-form change must
            # also trigger a group rebuild — final may be unchanged when
            # captures_excludes drops a rule from the training pipeline.
            if r["sample_id"]:
                affected_sample_ids.add(r["sample_id"])
        if patch:
            captures_store.update_capture(cid, patch)
            with _state_lock:
                _updated_cids.add(cid)
                _state["captures_updated"] = len(_updated_cids)
        with _state_lock:
            _state["processed"] += 1

    if affected_sample_ids:
        for sid in affected_sample_ids:
            g = capture_samples_store.get_sample(sid)
            if g is None or g.get("is_locked"):
                continue
            members = capture_samples_store.get_members(sid)
            new_t = capture_samples._build_default_transcript(
                members, g.get("transcript_join_strategy") or "space",
            )
            if new_t != (g.get("transcript") or ""):
                capture_samples_store.update_sample(
                    sid, {"transcript": new_t},
                )
                with _state_lock:
                    _updated_sids.add(sid)
                    _state["groups_updated"] = len(_updated_sids)


def _reset_for_tests() -> None:
    """Test-only: back to the module's canonical idle shape."""
    global _worker, _state, _rerun_requested
    _worker = None
    _rerun_requested = False
    _updated_cids.clear()
    _updated_sids.clear()
    _state = {
        "status": "idle",
        "started_ts": None,
        "finished_ts": None,
        "total": 0,
        "processed": 0,
        "captures_updated": 0,
        "groups_updated": 0,
        "error": None,
    }
