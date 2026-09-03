"""Both per-job tables carry every shared job column (store_common.JOB_COLUMNS),
and both write paths fill them. A field added to one store and forgotten in
the other used to go unnoticed; here it fails."""
import time

from faster_whisper_backend.core import store_common


def test_job_columns_ddl_rejects_unknown_constraint_keys():
    import pytest
    with pytest.raises(ValueError):
        store_common.job_columns_ddl({"not_a_column": "NOT NULL"})


def test_both_tables_carry_every_shared_column(usage_store_db, tx_store):
    assert store_common.missing_job_columns(
        usage_store_db._require_conn(), "usage_jobs") == {}
    assert store_common.missing_job_columns(
        tx_store._require_conn(), "recent_transcriptions") == {}


def test_both_write_paths_fill_every_shared_column(usage_store_db, tx_store):
    """One job recorded through each store's public writer reads back with a
    value in every shared column — the write path, not just the schema."""
    now = time.time()
    tx_store.record_timing(
        request_id="j1", model="m", audio_s=2.0, processing_s=1.0, status="error",
        words=3, user_id="u", created_ts=now, kind="transcribe", wait_s=0.5,
        error_class="oom", error_stage="decode")
    usage_store_db.record_usage(
        key_id="k", user_id="u", audio_s=2.0, words=3, status="error",
        kind="file", job_id="j1", model="m", language="de", processing_s=1.0,
        wait_s=0.5, error_class="oom", error_stage="decode")
    rt = dict(tx_store._require_conn().execute(
        "SELECT * FROM recent_transcriptions WHERE request_id = 'j1'").fetchone())
    uj = dict(usage_store_db._require_conn().execute(
        "SELECT * FROM usage_jobs WHERE job_id = 'j1'").fetchone())
    # language is not a record_timing parameter (record_trace carries it).
    for name in store_common.JOB_COLUMN_NAMES:
        assert uj[name] is not None, f"usage_jobs.{name} not written"
        if name != "language":
            assert rt[name] is not None, f"recent_transcriptions.{name} not written"
