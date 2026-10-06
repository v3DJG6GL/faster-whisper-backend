"""Whisper transcription internals shared by the batch routes in main.py, the
streaming WebSocket and the preload worker: the model cache and decode-kwargs
assembly (models), the post-decode guard helpers (guards) and the per-request
receipt (receipt). This file must never import anything.
"""
