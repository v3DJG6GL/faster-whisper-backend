"""Whisper transcription internals shared by the batch route in main.py, the
streaming WebSocket and the preload worker: the model cache and decode-kwargs
assembly (models), the per-decode window/rung trace with the residual-window
stop and token cap (decode_trace), the post-decode guard helpers (guards) and
the in-segment tail-cut guards (segment_guards), the per-request receipt
(receipt) and the dictation receipt held open for its translation
(receipt_hold), the batch progress / cancel / plan registries (progress) and
the run plan behind one fraction and ETA (run_plan), the durable job store
(jobs_store) and its progress / job routes (jobs_routes), and the client
catalog routes, /v1/models, /v1/me and friends (catalog_routes). This file
must never import anything.
"""
