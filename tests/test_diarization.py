"""Diarization stage on POST /v1/audio/transcriptions (soft-fail contract) +
the pure segment-assignment helper. pyannote is never imported — the module's
`diarize` coroutine is monkeypatched, exactly the boundary the handler uses."""

import diarization

_FILE = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


def _post(client, **data):
    data.setdefault("model", "whisper-1")
    data.setdefault("response_format", "verbose_json")
    return client.post("/v1/audio/transcriptions", files=_FILE, data=data)


def _stub_turns(monkeypatch, turns, calls=None):
    async def _fake_diarize(path, *, num_speakers=None, min_speakers=None,
                            max_speakers=None, progress_cb=None):
        if calls is not None:
            calls.append({"path": path, "num_speakers": num_speakers,
                          "min_speakers": min_speakers,
                          "max_speakers": max_speakers})
        return turns
    monkeypatch.setattr(diarization, "diarize", _fake_diarize)


# --- route behaviour ---------------------------------------------------------

def test_diarize_labels_segments_and_lists_speakers(client, app_module, monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True
    try:
        _stub_turns(monkeypatch, [(0.0, 0.6, "SPEAKER_00"),
                                  (0.6, 1.0, "SPEAKER_01")])
        r = _post(client, diarize="true")
        assert r.status_code == 200, r.text
        body = r.json()
        # FakeModel returns one segment spanning 0..1 — SPEAKER_00 covers more.
        assert body["segments"][0]["speaker"] == "SPEAKER_00"
        assert body["speakers"] == ["SPEAKER_00"]
        assert "warnings" not in body
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False


def test_diarize_disabled_server_soft_fails(client, app_module, monkeypatch):
    # DIARIZATION_ENABLED defaults off: the request still succeeds, the
    # transcript has no speakers, and a warning explains why.
    called = []
    _stub_turns(monkeypatch, [(0.0, 1.0, "SPEAKER_00")], calls=called)
    r = _post(client, diarize="true")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "speaker" not in body["segments"][0]
    assert "speakers" not in body
    assert any("not enabled" in w for w in body["warnings"])
    assert called == []


def test_diarize_absent_inherits_config_default_off(client, app_module, monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True
    try:
        called = []
        _stub_turns(monkeypatch, [(0.0, 1.0, "SPEAKER_00")], calls=called)
        r = _post(client)
        assert r.status_code == 200
        assert called == []          # DIARIZE global default is false
        assert "warnings" not in r.json()
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False


def test_diarize_config_default_on_applies(client, app_module, monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True
    app_module.cfg.DIARIZE = True
    try:
        called = []
        _stub_turns(monkeypatch, [(0.0, 1.0, "SPEAKER_00")], calls=called)
        r = _post(client)
        assert r.status_code == 200
        assert len(called) == 1      # absent field inherits DIARIZE=true
        # ...and an explicit false still wins over the config default.
        r = _post(client, diarize="false")
        assert r.status_code == 200
        assert len(called) == 1
    finally:
        app_module.cfg.DIARIZE = False
        app_module.cfg.DIARIZATION_ENABLED = False


def test_diarization_error_becomes_warning(client, app_module, monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True
    try:
        async def _boom(path, **kw):
            raise diarization.DiarizationError(
                "diarization dependencies are not installed on this server "
                "(pip install -r requirements-diarize.txt)")
        monkeypatch.setattr(diarization, "diarize", _boom)
        r = _post(client, diarize="1")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["text"]                       # transcript survives
        assert "speakers" not in body
        assert any("requirements-diarize" in w for w in body["warnings"])
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False


def test_num_speakers_wins_and_clamps(client, app_module, monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True
    try:
        called = []
        _stub_turns(monkeypatch, [(0.0, 1.0, "SPEAKER_00")], calls=called)
        r = _post(client, diarize="yes", num_speakers="99", min_speakers="2",
                  max_speakers="5")
        assert r.status_code == 200
        assert called[0]["num_speakers"] == 32    # clamped to the cap
        assert called[0]["min_speakers"] is None  # num wins over bounds
        assert called[0]["max_speakers"] is None
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False


def test_min_max_forwarded_without_num(client, app_module, monkeypatch):
    app_module.cfg.DIARIZATION_ENABLED = True
    try:
        called = []
        _stub_turns(monkeypatch, [(0.0, 1.0, "SPEAKER_00")], calls=called)
        r = _post(client, diarize="on", min_speakers="2", max_speakers="4")
        assert r.status_code == 200
        assert called[0]["num_speakers"] is None
        assert called[0]["min_speakers"] == 2
        assert called[0]["max_speakers"] == 4
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False


def test_no_segments_skips_diarization_with_warning(client, app_module,
                                                    monkeypatch, fake_model):
    app_module.cfg.DIARIZATION_ENABLED = True
    try:
        called = []
        _stub_turns(monkeypatch, [(0.0, 1.0, "SPEAKER_00")], calls=called)
        fake_model._segments = []          # silence: whisper yields nothing
        r = _post(client, diarize="true")
        assert r.status_code == 200
        assert called == []
        assert any("no speech" in w for w in r.json()["warnings"])
    finally:
        app_module.cfg.DIARIZATION_ENABLED = False


# --- assign_speakers (pure) --------------------------------------------------

def _seg(start, end):
    return {"start": start, "end": end}


def test_assign_speakers_largest_overlap():
    segs = [_seg(0.0, 2.0), _seg(2.0, 4.0)]
    turns = [(0.0, 1.5, "A"), (1.5, 4.0, "B")]
    labels = diarization.assign_speakers(segs, turns)
    assert segs[0]["speaker"] == "A"       # 1.5s of A vs 0.5s of B
    assert segs[1]["speaker"] == "B"
    assert labels == ["A", "B"]


def test_assign_speakers_gap_falls_back_to_nearest():
    segs = [_seg(10.0, 11.0)]              # inside a diarization gap
    turns = [(0.0, 2.0, "A"), (11.5, 20.0, "B")]
    diarization.assign_speakers(segs, turns)
    assert segs[0]["speaker"] == "B"       # 0.5s away vs 8s to A's end


def test_assign_speakers_no_turns_is_noop():
    segs = [_seg(0.0, 1.0)]
    assert diarization.assign_speakers(segs, []) == []
    assert "speaker" not in segs[0]


# --- progress hook -----------------------------------------------------------

def test_hook_maps_steps_and_stays_monotone():
    seen = []
    hook = diarization._make_hook(seen.append)
    hook("segmentation", None, total=10, completed=5)
    hook("segmentation", None, total=10, completed=10)
    hook("embeddings", None, total=4, completed=2)
    # A regression (pyannote re-reports an earlier step) must not move the bar
    # backwards — it is simply dropped.
    hook("segmentation", None, total=10, completed=1)
    hook("clustering", None)          # no total → logged but never moves the bar
    hook("embeddings", None, total=4, completed=4)
    assert seen == [0.225, 0.45, 0.675, 0.9]


def test_hook_swallows_bad_callback():
    def _boom(_f):
        raise RuntimeError("cb exploded")
    hook = diarization._make_hook(_boom)
    hook("segmentation", None, total=10, completed=5)  # must not raise
