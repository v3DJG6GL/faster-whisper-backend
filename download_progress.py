"""Model-download progress: a huggingface_hub tqdm shim + capture scope.

Everywhere a model reaches disk through huggingface_hub (whisper CT2 repos,
translation GGUFs, pyannote pipelines) the actual byte progress lives inside
hub-internal tqdm bars that a server log never sees. This module turns those
bars into:

  * throttled `[download] <label> N% (x/y GB · z MB/s)` INFO lines
    (every crossed 10% boundary, at least every 30 s) plus start/done
    receipts,
  * an optional caller callback ``cb(done_bytes, total_bytes)`` (the request
    handlers map it onto their progress registry entry), and
  * a "download" entry in the central jobs registry.

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
see a thread-local. Model loads are already serialized per module (`_lock`
in translation/diarization/bgm, `_model_load_lock` in main), so concurrent
captures are not a real shape; if two ever overlap, byte counts may blend —
progress is a convenience, never a correctness surface.

The `capture` context also monkeypatches the hub-internal default tqdm
(``huggingface_hub.utils.tqdm.tqdm`` + ``file_download``'s alias) for its
duration as a fallback for call paths that don't accept ``tqdm_class``,
restoring both in its finally.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
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
        self.bars: dict[int, list[int]] = {}   # id(bar) -> [done, total]
        self.t0 = time.monotonic()
        self._started_logged = False
        self._last_log = self.t0
        self._last_bucket = 0
        self._last_cb = 0.0

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
                       and now - self._last_cb >= _CB_MIN_INTERVAL_S)
            if fire_cb:
                self._last_cb = now
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


if _hub_tqdm_mod is not None:

    class ReportingTqdm(_BaseTqdm):
        """hub-shim tqdm subclass that mirrors byte progress into the active
        capture. Pop-safe with the hub shim's extra `name` kwarg (the parent
        pops it); counts bytes itself so a disabled bar still reports."""

        def __init__(self, *args, **kwargs):
            self._dp_n = int(kwargs.get("initial") or 0)
            self._dp_total = int(kwargs.get("total") or 0)
            unit = str(kwargs.get("unit") or "")
            super().__init__(*args, **kwargs)
            cap = _active_capture()
            self._dp_cap = cap if (cap is not None
                                   and unit.startswith("B")) else None
            if self._dp_cap is not None:
                self._dp_cap.add_bar(id(self), self._dp_total)

        def update(self, n=1):
            out = super().update(n)
            cap = getattr(self, "_dp_cap", None)
            if cap is not None and n:
                self._dp_n += int(n)
                # A retried/resumed download can update(-resume_size).
                self._dp_n = max(0, self._dp_n)
                total = int(getattr(self, "total", None)
                            or self._dp_total or 0)
                cap.bump(id(self), self._dp_n, total)
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
    ("huggingface_hub.file_download", "tqdm"),
    ("huggingface_hub._snapshot_download", "tqdm"),
)


def _active_capture() -> "_Capture | None":
    return _current


@contextlib.contextmanager
def capture(label: str, cb: "Callable[[int, int], None] | None" = None,
            *, record: bool = True):
    """Scope inside which every hub download reports as `label`.

    Yields the `_Capture` (use `.tqdm_kwargs` where the API takes a
    tqdm_class). On exit logs the done receipt when bytes actually moved
    (a warm cache hit stays silent) and — with `record=True` — persists a
    'download' row via metrics (lazy import; no-op before init_db)."""
    global _current
    cap = _Capture(label, cb)
    patched: list[tuple[Any, Any]] = []
    with _capture_lock:
        _current = cap
    if _hub_tqdm_mod is not None:
        for mod_name, attr in _PATCH_TARGETS:
            try:
                mod = importlib.import_module(mod_name)
                if getattr(mod, attr, None) is not None:
                    patched.append((mod, getattr(mod, attr)))
                    setattr(mod, attr, ReportingTqdm)
            except Exception:  # noqa: BLE001 — fallback patch is best-effort
                pass
    try:
        yield cap
    finally:
        for mod, prev in patched:
            try:
                setattr(mod, "tqdm", prev)
            except Exception:  # noqa: BLE001
                pass
        with _capture_lock:
            if _current is cap:
                _current = None
        done_b, total_b = cap.totals()
        secs = max(1e-6, time.monotonic() - cap.t0)
        if done_b > 0:
            logger.info("[download] %s done: %s in %.1fs (%s/s)",
                        cap.label, _fmt_bytes(done_b), secs,
                        _fmt_bytes(done_b / secs))
            if cap.cb is not None:
                try:
                    cap.cb(done_b, total_b or done_b)
                except Exception:  # noqa: BLE001
                    pass
            if record:
                _record_download(cap.label, done_b, secs)


def _record_download(label: str, done_bytes: int, secs: float) -> None:
    """Persist a finished download as a recent-jobs row. Lazy import keeps
    this module free of a metrics/transcriptions_store import cycle; every
    failure is swallowed — recording is bookkeeping, not control flow."""
    try:
        import metrics
        metrics.record_download(model=label, seconds=secs,
                                bytes_done=done_bytes)
    except AttributeError:
        # metrics.record_download lands in a later change — logging above
        # is the receipt until then.
        pass
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
