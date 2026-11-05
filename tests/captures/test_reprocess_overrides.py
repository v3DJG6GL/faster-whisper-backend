"""Per-capture reprocess resolves the CAPTURE OWNER's effective pipeline
(not the caller's), tolerates owner-less captures, and runs the pipeline off
the event loop."""

from faster_whisper_backend.settings import effective_config
from tests.captures._store_helpers import create_stub_capture
from tests.conftest import bearer, install_fake_wav_transcode, on_event_loop, record_calls



def _make_capture(tmp_path, user_id):
    from faster_whisper_backend.captures import store as captures_store
    return create_stub_capture(
        captures_store, tmp_path, model="whisper-1", raw="hallo welt",
        final="hallo welt", user_id=user_id)


def test_reprocess_resolves_owner_not_caller(client, root_key, make_user_key, app_module,
                                             monkeypatch, tmp_path):
    install_fake_wav_transcode(monkeypatch)
    uid_alice, _ = make_user_key("alice")
    cid = _make_capture(tmp_path, user_id=uid_alice)

    calls = record_calls(monkeypatch, effective_config, "build_ident",
                         lambda user, model_id, *a, **k: (dict(user or {}), model_id))

    r = client.post(f"/captures/api/{cid}/reprocess", headers=bearer(root_key))
    assert r.status_code == 200, r.text
    assert calls, "reprocess did not resolve an ident"
    assert calls[0][0].get("user_id") == uid_alice
    assert calls[0][0].get("key_id") is None      # reprocess is pipeline-only


def test_reprocess_ownerless_capture_ok(client, root_key, monkeypatch, tmp_path):
    install_fake_wav_transcode(monkeypatch)
    cid = _make_capture(tmp_path, user_id=None)
    r = client.post(f"/captures/api/{cid}/reprocess", headers=bearer(root_key))
    assert r.status_code == 200, r.text


def test_reprocess_runs_the_pipeline_off_the_event_loop(client, root_key,
                                                        monkeypatch, tmp_path):
    """The pipeline pass over raw (up to 50k chars, owner regex rules) must not
    run on the loop: inside asyncio.to_thread there is no running loop."""
    from faster_whisper_backend.pipeline import engine as pl_engine

    install_fake_wav_transcode(monkeypatch)
    cid = _make_capture(tmp_path, user_id=None)
    on_loop = record_calls(monkeypatch, pl_engine, "_postprocess_text",
                           lambda *a, **k: on_event_loop())
    r = client.post(f"/captures/api/{cid}/reprocess", headers=bearer(root_key))
    assert r.status_code == 200, r.text
    assert on_loop and not any(on_loop)
