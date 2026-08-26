"""Music-separation stage on POST /v1/audio/transcriptions (soft-fail
contract). audio-separator is never imported — the module's `separate`
coroutine is monkeypatched at the boundary the handler uses."""

import os
import tempfile

import bgm_separation

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


def _post(client, **data):
    data.setdefault("model", "whisper-1")
    data.setdefault("response_format", "verbose_json")
    return client.post("/v1/audio/transcriptions", files=_FILE, data=data)


def _stub_separate(monkeypatch, calls=None):
    """separate() writes a real vocals tmp file — the handler swaps it into
    tmp_path and the request-level finally must unlink it."""
    made = []

    async def _fake(path):
        if calls is not None:
            calls.append(path)
        fd, out = tempfile.mkstemp(prefix="vocals-test-", suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(b"RIFFsepWAVE")
        made.append(out)
        return out

    monkeypatch.setattr(bgm_separation, "separate", _fake)
    return made


def test_separate_swaps_audio_and_cleans_up(client, app_module, monkeypatch, fake_model):
    app_module.cfg.BGM_SEPARATION_ENABLED = True
    try:
        calls = []
        made = _stub_separate(monkeypatch, calls)
        r = _post(client, separate_bgm="true")
        assert r.status_code == 200, r.text
        assert "warnings" not in r.json()
        assert len(calls) == 1                 # the ORIGINAL upload went in
        # The decode consumed the vocals file, not the original...
        assert fake_model.last_audio == made[0]
        # ...and the request-level finally unlinked the swapped-in tmp.
        assert not os.path.exists(made[0])
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False


def test_separate_disabled_server_soft_fails(client, app_module, monkeypatch):
    calls = []
    _stub_separate(monkeypatch, calls)
    r = _post(client, separate_bgm="true")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"]                        # transcript survives
    assert any("not enabled" in w for w in body["warnings"])
    assert calls == []


def test_separate_error_becomes_warning(client, app_module, monkeypatch, fake_model):
    app_module.cfg.BGM_SEPARATION_ENABLED = True
    try:
        async def _boom(path):
            raise bgm_separation.BgmSeparationError(
                "music-separation dependencies are not installed on this "
                "server (pip install -r requirements-bgm.txt)")
        monkeypatch.setattr(bgm_separation, "separate", _boom)
        r = _post(client, separate_bgm="1")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["text"]                    # original audio transcribed
        assert any("requirements-bgm" in w for w in body["warnings"])
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False


def test_separate_absent_inherits_config_default(client, app_module, monkeypatch):
    app_module.cfg.BGM_SEPARATION_ENABLED = True
    app_module.cfg.SEPARATE_BGM = True
    try:
        calls = []
        _stub_separate(monkeypatch, calls)
        r = _post(client)
        assert r.status_code == 200
        assert len(calls) == 1                 # SEPARATE_BGM=true applies
        r = _post(client, separate_bgm="false")
        assert r.status_code == 200
        assert len(calls) == 1                 # explicit false wins
    finally:
        app_module.cfg.SEPARATE_BGM = False
        app_module.cfg.BGM_SEPARATION_ENABLED = False


def test_locked_separate_ignores_client_param(client, app_module, make_user_key,
                                              monkeypatch):
    from tests.conftest import bearer
    app_module.cfg.BGM_SEPARATION_ENABLED = True
    try:
        calls = []
        _stub_separate(monkeypatch, calls)
        _, raw_admin = make_user_key("admin", is_admin=True)
        admin_h = bearer(raw_admin)
        r = client.post("/settings/overrides/state", headers=admin_h,
                        json={"OVERRIDE_PROFILES": {"nosep": {"locks": ["SEPARATE_BGM"]}}})
        assert r.status_code == 200, r.text
        uid, raw_alice = make_user_key("alice", is_admin=False)
        client.patch(f"/settings/api-keys/api/users/{uid}/permissions", headers=admin_h,
                     json={"pages": {}, "config": {"overrides": {},
                           "profiles": ["nosep"], "locks": []}})
        r = client.post(
            "/v1/audio/transcriptions", files=_FILE, headers=bearer(raw_alice),
            data={"model": "whisper-1", "response_format": "verbose_json",
                  "separate_bgm": "true"},
        )
        assert r.status_code == 200, r.text
        assert calls == []                     # pinned off (global default)
        assert "separate_bgm" in r.json()["overrides_ignored"]
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False


def test_model_filename_appends_onnx(app_module):
    app_module.cfg.BGM_SEPARATION_UVR_MODEL = "UVR-MDX-NET-Inst_HQ_4"
    assert bgm_separation._model_filename() == "UVR-MDX-NET-Inst_HQ_4.onnx"
    app_module.cfg.BGM_SEPARATION_UVR_MODEL = "model_bs_roformer.ckpt"
    assert bgm_separation._model_filename() == "model_bs_roformer.ckpt"
    app_module.cfg.BGM_SEPARATION_UVR_MODEL = "UVR-MDX-NET-Inst_HQ_4"
