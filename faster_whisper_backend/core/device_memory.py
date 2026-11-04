"""Memory reclaim after a model cache drops a model.

`reclaim_after_unload()` is the one teardown step the whisper, translation,
diarization and BGM caches share. Pure module, no state, no package imports.
"""
from __future__ import annotations

import sys


def reclaim_after_unload() -> None:
    """gc.collect() breaks leftover cycles so the dropped model's destructor
    runs, then torch.cuda.empty_cache() releases pool-cached blocks.

    Only a torch some subsystem ALREADY imported can hold pool-cached blocks:
    a cold import here would stall the event loop for seconds (libtorch +
    CUDA) and grow the RSS this pass frees. Best-effort: empty_cache raises
    RuntimeError after a prior CUDA fault, and a broken install (OSError from
    a missing CUDA lib) must not escape after the model was already dropped.
    """
    import gc
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — best-effort cache release
            pass
