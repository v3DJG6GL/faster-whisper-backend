"""web_common.severity_counts: the WARNING+ ring behind the header pills and
the /stats payload."""

import logging
import threading

from faster_whisper_backend.core import web_common


def test_severity_counts_survives_concurrent_appends(monkeypatch):
    """emit() appends from any logging thread while severity_counts() reads;
    walking the live deque raised "deque mutated during iteration"."""
    from collections import deque

    ring = deque(((0.0, logging.WARNING) for _ in range(2000)), maxlen=2000)
    monkeypatch.setattr(web_common, "_SEVERITY_LOG", ring)
    stop = threading.Event()

    def _append():
        while not stop.is_set():
            ring.append((0.0, logging.ERROR))

    t = threading.Thread(target=_append, daemon=True)
    t.start()
    try:
        for _ in range(2000):
            c = web_common.severity_counts()
            assert c["warn"] + c["err"] + c["crit"] == 2000
    finally:
        stop.set()
        t.join()


def test_severity_counts_buckets_by_level(monkeypatch):
    from collections import deque

    monkeypatch.setattr(web_common, "_SEVERITY_LOG", deque([
        (0.0, logging.WARNING), (0.0, logging.ERROR), (0.0, logging.ERROR),
        (0.0, logging.CRITICAL)], maxlen=2000))
    assert web_common.severity_counts() == {"warn": 1, "err": 2, "crit": 1}
