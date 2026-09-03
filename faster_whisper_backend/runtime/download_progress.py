"""Model-download progress: a huggingface_hub tqdm shim + capture scope.

Everywhere a model reaches disk through huggingface_hub (whisper CT2 repos,
translation GGUFs, pyannote pipelines) the actual byte progress lives inside
hub-internal tqdm bars that a server log never sees. This module turns those
bars into:

  * throttled `[download] <label> N% (x/y GB · z MB/s)` INFO lines
    (every crossed 10% boundary, at least every 30 s) plus start/done
    receipts,
  * an optional caller callback ``cb(done_bytes, total_bytes)`` (the request
    handlers map it onto their progress registry entry, and the call sites
    also use it to drive their own ``jobs.job_start("download", ...)`` entry
    — this module never touches the jobs registry itself), and
  * a persisted 'download' row via ``metrics.record_download`` on exit.

Usage::

    with capture(label="gguf:org/repo", cb=hook) as cap:
        hf_hub_download(..., **cap.tqdm_kwargs)   # tqdm_class when supported

``ReportingTqdm`` counts bytes itself (``self._dp_n``) instead of reading
tqdm's own counter, so it works identically whether hub has progress bars
globally disabled (HF_HUB_DISABLE_PROGRESS_BARS, the server norm) or not —
disabled bars still receive every ``update()`` call, they just skip their
console rendering, which is exactly what a server wants. Bars whose unit is
not bytes (snapshot_download's outer per-FILE bar) are ignored.

Scope: the active capture is module-global (not thread-local), because
snapshot_download fans file downloads out to worker threads that would never
see a thread-local. diarization/bgm serialize their loads module-wide, but
translation serializes only per ref (`_loading[ref]`) and runs the load
outside its `_lock`, and main's whisper pre-download deliberately runs
*outside* `_model_load_lock` — so two different refs downloading at once,
two cold whisper loads, or a request plus a preload worker can overlap.
When they do, the later scope replaces
the earlier one: bars constructed afterwards report under the later label
and the inner exit restores `ReportingTqdm` rather than the original tqdm
until the outer exit — byte counts may blend, and progress is a
convenience, never a correctness surface.

The `capture` context also monkeypatches the hub-internal default tqdm for
its duration as a fallback for call paths that don't accept ``tqdm_class``:
every ``(module, attr)`` pair in ``_PATCH_TARGETS`` is rebound to
``ReportingTqdm`` and restored to its previous value in the finally.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import itertools
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger("whisper-api")

# Log cadence: every crossed 10% bucket, and at least every 30 s.
_LOG_BUCKET_PCT = 10
_LOG_MIN_INTERVAL_S = 30.0
# Callback cadence (drives 1 Hz pollers — anything finer is wasted work).
_CB_MIN_INTERVAL_S = 0.5

try:  # hub arrives transitively (>=1.29.0 floor via requirements-convert)
    _hub_tqdm_mod = importlib.import_module("huggingface_hub.utils.tqdm")
    _BaseTqdm = _hub_tqdm_mod.tqdm
except Exception:  # noqa: BLE001 — module must import without hub
    _hub_tqdm_mod = None
    _BaseTqdm = object


class _Capture:
    """State for one capture scope: aggregated byte counts across every
    byte-unit bar constructed inside it, throttled logging + callback."""

    def __init__(self, label: str, cb: "Callable[[int, int], None] | None"):
        self.label = label
        self.cb = cb
        self.lock = threading.Lock()
        # bar key -> [done, total]. Keyed by a monotonic sequence number, not
        # id(bar): finished bars are garbage-collected and CPython reuses the
        # address for the next one, which would overwrite a done file's bytes.
        self.bars: dict[int, list[int]] = {}
        self.t0 = time.monotonic()
        self._started_logged = False
        self._last_log = self.t0
        self._last_bucket = 0
        self._last_cb = 0.0
        # Largest aggregate handed to cb so far. bump() computes the sum
        # under the lock but delivers outside it, so a snapshot worker that
        # computed 500 can be preempted and deliver after a peer's 800 —
        # a stale delivery would walk the WebUI bar backwards.
        self._last_sent = -1

    # -- called from ReportingTqdm ------------------------------------------
    def add_bar(self, bar_id: int, total: int) -> None:
        with self.lock:
            self.bars[bar_id] = [0, max(0, int(total or 0))]

    def bump(self, bar_id: int, done: int, total: int) -> None:
        now = time.monotonic()
        with self.lock:
            self.bars[bar_id] = [int(done), max(0, int(total or 0))]
            done_b = sum(v[0] for v in self.bars.values())
            total_b = sum(v[1] for v in self.bars.values())
            if not self._started_logged and total_b > 0:
                self._started_logged = True
                logger.info("[download] %s started (%s expected)",
                            self.label, _fmt_bytes(total_b))
            pct = int(done_b * 100 / total_b) if total_b else 0
            bucket = pct // _LOG_BUCKET_PCT
            if (bucket > self._last_bucket
                    or now - self._last_log >= _LOG_MIN_INTERVAL_S):
                self._last_bucket = bucket
                self._last_log = now
                secs = max(1e-6, now - self.t0)
                logger.info("[download] %s %d%% (%s/%s · %s/s)",
                            self.label, pct, _fmt_bytes(done_b),
                            _fmt_bytes(total_b), _fmt_bytes(done_b / secs))
            fire_cb = (self.cb is not None
                       and now - self._last_cb >= _CB_MIN_INTERVAL_S
                       and done_b >= self._last_sent)
            if fire_cb:
                self._last_cb = now
                self._last_sent = done_b
        if fire_cb:
            try:
                self.cb(done_b, total_b)
            except Exception:  # noqa: BLE001 — progress must never break us
                pass

    # -- totals + hf kwargs --------------------------------------------------
    def totals(self) -> tuple[int, int]:
        with self.lock:
            return (sum(v[0] for v in self.bars.values()),
                    sum(v[1] for v in self.bars.values()))

    @property
    def tqdm_kwargs(self) -> dict[str, Any]:
        """`{"tqdm_class": ReportingTqdm}` when the installed hub supports
        the parameter — spreadable straight into hf_hub_download /
        snapshot_download; empty otherwise (the capture monkeypatch covers
        those hubs)."""
        return dict(_TQDM_CLASS_KWARGS)


# Process-unique bar keys for `_Capture.bars` (see its comment).
_bar_seq = itertools.count()

if _hub_tqdm_mod is not None:

    class ReportingTqdm(_BaseTqdm):
        """hub-shim tqdm subclass that mirrors byte progress into the active
        capture. Pop-safe with the hub shim's extra `name` kwarg (the parent
        pops it); counts bytes itself so a disabled bar still reports."""

        def __init__(self, *args, **kwargs):
            self._dp_n = int(kwargs.get("initial") or 0)
            self._dp_total = int(kwargs.get("total") or 0)
            unit = str(kwargs.get("unit") or "")
            name = str(kwargs.get("name") or "")
            self._dp_key = next(_bar_seq)
            super().__init__(*args, **kwargs)
            cap = _active_capture()
            # snapshot_download builds TWO byte-unit aggregate bars from
            # tqdm_class over the same bytes: `...snapshot_download` (the
            # reconstruct bar, whose total the hub grows per file) and
            # `...snapshot_download.transfer`. Registering both would count
            # every byte twice, so the transfer twin is left out.
            self._dp_cap = cap if (cap is not None
                                   and unit.startswith("B")
                                   and not name.endswith(".transfer")) else None
            if self._dp_cap is not None:
                self._dp_cap.add_bar(self._dp_key, self._dp_total)

        def update(self, n=1):
            out = super().update(n)
            cap = getattr(self, "_dp_cap", None)
            if cap is not None and n:
                self._dp_n += int(n)
                # A retried/resumed download can update(-resume_size).
                self._dp_n = max(0, self._dp_n)
                total = int(getattr(self, "total", None)
                            or self._dp_total or 0)
                cap.bump(self._dp_key, self._dp_n, total)
            return out

else:  # pragma: no cover — hub always importable in this deployment

    class ReportingTqdm:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("huggingface_hub is not installed")


def _supports_tqdm_class() -> bool:
    try:
        from huggingface_hub import hf_hub_download
        return "tqdm_class" in inspect.signature(hf_hub_download).parameters
    except Exception:  # noqa: BLE001
        return False


_TQDM_CLASS_KWARGS: dict[str, Any] = (
    {"tqdm_class": ReportingTqdm}
    if _hub_tqdm_mod is not None and _supports_tqdm_class() else {}
)

# Module-global active capture (see the module docstring for why not
# thread-local) + the hub attributes the capture temporarily rebinds.
_capture_lock = threading.Lock()
_current: "_Capture | None" = None
_PATCH_TARGETS = (
    ("huggingface_hub.utils.tqdm", "tqdm"),
    # Defensive no-op on hub >= 1.29 (file_download imports tqdm only for
    # annotations and builds its bar through utils.tqdm); kept for older hubs.
    ("huggingface_hub.file_download", "tqdm"),
)


def _active_capture() -> "_Capture | None":
    return _current


@contextlib.contextmanager
def capture(label: str, cb: "Callable[[int, int], None] | None" = None,
            *, record: bool = True):
    """Scope inside which every hub download reports as `label`.

    Yields the `_Capture` (use `.tqdm_kwargs` where the API takes a
    tqdm_class). On exit, when bytes actually moved (a warm cache hit stays
    silent), logs a done receipt — or an "aborted" warning when the body
    raised — and with `record=True` persists a 'download' row via metrics
    (lazy import; no-op before init_db) whose status says which it was. A
    failed download never fires `cb` with a completion-shaped pair."""
    global _current
    cap = _Capture(label, cb)
    patched: list[tuple[Any, str, Any]] = []
    failed = False
    with _capture_lock:
        prev_current = _current
        _current = cap
    if _hub_tqdm_mod is not None:
        for mod_name, attr in _PATCH_TARGETS:
            try:
                mod = importlib.import_module(mod_name)
                if getattr(mod, attr, None) is not None:
                    patched.append((mod, attr, getattr(mod, attr)))
                    setattr(mod, attr, ReportingTqdm)
            except Exception:  # noqa: BLE001 — fallback patch is best-effort
                pass
    try:
        yield cap
    except BaseException:
        # The body raised (network drop, full disk, cancellation): the
        # finally still runs, but must not hand out a "done" receipt.
        failed = True
        raise
    finally:
        for mod, attr, prev in patched:
            try:
                setattr(mod, attr, prev)
            except Exception:  # noqa: BLE001
                pass
        with _capture_lock:
            if _current is cap:
                _current = prev_current
        done_b, total_b = cap.totals()
        secs = max(1e-6, time.monotonic() - cap.t0)
        if done_b > 0:
            if failed:
                logger.warning("[download] %s aborted after %s in %.1fs",
                               cap.label, _fmt_bytes(done_b), secs)
            else:
                logger.info("[download] %s done: %s in %.1fs (%s/s)",
                            cap.label, _fmt_bytes(done_b), secs,
                            _fmt_bytes(done_b / secs))
                if cap.cb is not None:
                    try:
                        cap.cb(done_b, total_b or done_b)
                    except Exception:  # noqa: BLE001
                        pass
            if record:
                _record_download(cap.label, done_b, secs, ok=not failed)


def _record_download(label: str, done_bytes: int, secs: float, *,
                     ok: bool = True) -> None:
    """Persist a download that moved bytes as a recent-jobs row — status
    'ok' when it finished, 'error' when the body raised mid-transfer. Lazy
    import keeps this module free of a metrics/recent_transcriptions_store import
    cycle; every failure is swallowed — recording is bookkeeping, not
    control flow."""
    try:
        from faster_whisper_backend.stats import metrics
        metrics.record_download(model=label, seconds=secs,
                                bytes_done=done_bytes,
                                status="ok" if ok else "error")
    except Exception as e:  # noqa: BLE001
        logger.warning("[download] record failed: %s", e)


def _fmt_bytes(n: float) -> str:
    n = float(n or 0)
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n:.0f} B"
