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

    async def _fake(path, *, progress_cb=None, cancel_check=None):
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


def test_separate_transcodes_non_libsndfile_container(
        client, app_module, monkeypatch, fake_model):
    """An .m4a input is pre-transcoded to 44.1 kHz stereo WAV for the
    separator (libsndfile can't read AAC/MP4 → slow audioread fallback)."""
    import audio_transcode

    app_module.cfg.BGM_SEPARATION_ENABLED = True
    try:
        calls = []
        made = _stub_separate(monkeypatch, calls)
        transcoded = []

        def _fake_transcode(src, dst, *, rate, layout):
            transcoded.append((src, dst, rate, layout))
            with open(dst, "wb") as f:
                f.write(b"RIFF44kWAVE")
            return 11

        monkeypatch.setattr(audio_transcode, "transcode_to_wav",
                            _fake_transcode)
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.m4a", b"\x00\x00\x00 ftypM4A ", "audio/mp4")},
            data={"model": "whisper-1", "response_format": "verbose_json",
                  "separate_bgm": "true"})
        assert r.status_code == 200, r.text
        assert len(transcoded) == 1
        _src, _dst, _rate, _layout = transcoded[0]
        assert (_rate, _layout) == (44100, "stereo")
        assert _src.endswith(".m4a") and _dst.endswith(".wav")
        # The separator got the WAV, not the original container...
        assert calls == [_dst]
        # ...and the intermediate WAV was unlinked after separation.
        assert not os.path.exists(_dst)
        assert fake_model.last_audio == made[0]
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False


def test_separate_transcode_failure_falls_back_to_original(
        client, app_module, monkeypatch, fake_model):
    import audio_transcode

    app_module.cfg.BGM_SEPARATION_ENABLED = True
    try:
        calls = []
        _stub_separate(monkeypatch, calls)

        def _boom(src, dst, *, rate, layout):
            raise RuntimeError("no decoder")

        monkeypatch.setattr(audio_transcode, "transcode_to_wav", _boom)
        r = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.m4a", b"\x00\x00\x00 ftypM4A ", "audio/mp4")},
            data={"model": "whisper-1", "separate_bgm": "true"})
        assert r.status_code == 200, r.text
        # Separation still ran — on the original file (soft-fail, no warning
        # surfaced to the client for a transcode hiccup).
        assert len(calls) == 1 and calls[0].endswith(".m4a")
    finally:
        app_module.cfg.BGM_SEPARATION_ENABLED = False


def test_separate_wav_input_skips_transcode(
        client, app_module, monkeypatch, fake_model):
    import audio_transcode

    app_module.cfg.BGM_SEPARATION_ENABLED = True
    try:
        calls = []
        _stub_separate(monkeypatch, calls)

        def _never(src, dst, *, rate, layout):
            raise AssertionError("wav input must not be transcoded")

        monkeypatch.setattr(audio_transcode, "transcode_to_wav", _never)
        r = _post(client, separate_bgm="true")
        assert r.status_code == 200, r.text
        assert len(calls) == 1 and calls[0].endswith(".wav")
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
        async def _boom(path, *, progress_cb=None, cancel_check=None):
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


# --- progress weighting ------------------------------------------------------

def test_pass_fraction_weights_model_pass_heavier():
    assert bgm_separation._pass_fraction(1, 0.0) == 0.0
    assert bgm_separation._pass_fraction(1, 1.0) == bgm_separation._PASS1_WEIGHT
    assert bgm_separation._pass_fraction(2, 0.0) == bgm_separation._PASS1_WEIGHT
    assert bgm_separation._pass_fraction(2, 1.0) == 1.0
    # Clamped against tqdm over-reporting past the total.
    assert bgm_separation._pass_fraction(1, 1.7) == bgm_separation._PASS1_WEIGHT


def test_pass_fraction_single_pass_owns_full_span(monkeypatch):
    monkeypatch.setattr(bgm_separation, "_single_pass", True)
    assert bgm_separation._pass_fraction(1, 0.5) == 0.5
    assert bgm_separation._pass_fraction(1, 1.0) == 1.0
