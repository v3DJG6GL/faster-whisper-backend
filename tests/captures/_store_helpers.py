"""Helpers shared by the captures store test modules.

Import as ``from tests.captures._store_helpers import ...``.
"""

from faster_whisper_backend.captures import store as captures_store


def insert_capture(conn, cid, **cols):
    """Raw-insert one captures row holding only the NOT NULL columns; `cols`
    set or override column values."""
    row = {"id": cid, "created_ts": 1.0, "model": "m",
           "audio_relpath": captures_store._relpath_for(cid, "wav"),
           "audio_format": "wav", "raw_text": "r", "final_text": "f",
           "words": "[]", "segments": "[]", "status": "new", **cols}
    conn.execute(
        f"INSERT INTO captures ({', '.join(row)})"
        f" VALUES ({', '.join('?' * len(row))})",
        tuple(row.values()),
    )


def create_stub_capture(store, tmp_path, **fields):
    """create_capture over a junk source; the caller stubs the transcode.
    `fields` override the defaults."""
    src = tmp_path / "src.bin"
    src.write_bytes(b"junk")
    return store.create_capture(**{
        "audio_src_path": str(src), "request_id": "r1", "model": "small",
        "language": "de", "audio_s": 1.0, "raw": "r", "final": "f",
        "words": [], "segments": [], **fields})
