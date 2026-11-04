"""Helpers shared by the captures store test modules.

Import as ``from tests.captures._store_helpers import ...``.
"""


def create_stub_capture(store, tmp_path, **fields):
    """create_capture over a junk source; the caller stubs the transcode.
    `fields` override the defaults."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"junk")
    return store.create_capture(**{
        "audio_src_path": str(src), "request_id": "r1", "model": "small",
        "language": "de", "audio_s": 1.0, "raw": "r", "final": "f",
        "words": [], "segments": [], **fields})
