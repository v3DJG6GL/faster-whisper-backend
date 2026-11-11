"""list_recent(query=...) keeps SQLite LIKE's match semantics (literal
substring, ASCII-only case fold, NULL never matches) at a cost linear in the
text, so a caller-shaped periodic needle over periodic rows stays cheap."""

import random
import sqlite3
import time

import pytest


@pytest.fixture
def ts(tx_store):
    return tx_store


def _trace(ts, request_id, raw, final, **kw):
    ts.record_trace(request_id=request_id, model="m", raw=raw, final=final,
                    prune_every=0, **kw)


def _like_ids(path, query):
    """The row set the LIKE '%needle%' ESCAPE '\\' search returned."""
    needle = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    like = f"%{needle}%"
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT request_id FROM recent_transcriptions"
            " WHERE (raw_text LIKE ? ESCAPE '\\' OR final_text LIKE ? ESCAPE '\\')",
            (like, like)).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def _search_ids(ts, query):
    return {r["request_id"] for r in ts.list_recent(query=query, limit=10_000)}


def test_search_matches_like_on_a_randomized_corpus(ts):
    rng = random.Random(20261111)
    alphabet = list("aAbBzZ09 ,.%_\\'\"") + list("äÄöÖßẞéÉçÇıİſ😀K")
    for i in range(120):
        raw = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        final = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        _trace(ts, f"r{i}", raw, final, created_ts=1.0 + i)
    # Error-path rows carry NULL raw/final and never match.
    ts.record_timing(request_id="err", model="m", audio_s=None, processing_s=1.0,
                     status="error", words=0, created_ts=500.0, prune_every=0)
    rows = ts.list_recent(limit=10_000)
    texts = [r["raw"] for r in rows] + [r["final"] for r in rows]
    queries = ["%", "_", "\\", "\\%", "a_b", "%%", "ä", "Ä", "ß", "😀", "K", "k"]
    for _ in range(400):
        src = rng.choice(texts) or "x"
        a = rng.randrange(len(src))
        piece = src[a:a + rng.randint(1, 6)]
        # Flip the case of random characters, ASCII and not, so the fold
        # rule is exercised in both directions.
        piece = "".join(c.swapcase() if rng.random() < 0.4 else c for c in piece)
        queries.append(piece)
    for _ in range(100):
        queries.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 4))))
    for q in queries:
        assert _search_ids(ts, q) == _like_ids(ts._path, q), repr(q)
    assert "err" not in _search_ids(ts, "a")


def test_search_folds_ascii_only(ts):
    _trace(ts, "ascii", "Patient FIEBER", "x", created_ts=1.0)
    _trace(ts, "umlaut", "Ärger", "x", created_ts=2.0)
    assert _search_ids(ts, "fieber") == {"ascii"}
    assert _search_ids(ts, "ärger") == set()
    assert _search_ids(ts, "Ärger") == {"umlaut"}
    assert _search_ids(ts, "äRGER") == set()
    assert _search_ids(ts, "ÄrGeR") == {"umlaut"}


def test_periodic_needle_over_periodic_rows_stays_cheap(ts):
    text = "ja, " * 12_500  # the 50 000-char cap
    for i in range(200):
        _trace(ts, f"r{i}", text, text, user_id="u1", created_ts=1.0 + i)
    needle = "ja, " * 127 + "jx"  # 510 chars that never match
    t0 = time.process_time()
    assert ts.list_recent(limit=100, query=needle, user_id_filter="u1") == []
    elapsed = time.process_time() - t0
    # A naive per-offset scan costs ~2 s of CPU here; a linear search ~0.02 s.
    assert elapsed < 0.4, elapsed
