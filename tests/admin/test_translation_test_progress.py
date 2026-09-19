"""Progress registry handling of the /settings translation prompt lab."""


def test_translation_test_straggler_tick_cannot_resurrect_the_entry(
        client, app_module, monkeypatch):
    """download_cb fires from the model-load thread and can outlive the
    handler. The finally must go through main._progress_close (tombstone):
    hand-popping the trio let the next tick re-create the entry owner-less —
    readable and cancellable by any authenticated caller holding the id."""
    from faster_whisper_backend.audio import translation

    app_module.cfg.TRANSLATION_ENABLED = True
    pid = "cafe" * 8
    held = {}

    async def fake_translate(segments, targets, **kwargs):
        held["download_cb"] = kwargs["download_cb"]
        return ([{"en": "hi"}], [], {"model": "org/m", "source": "de",
                                     "mode": "faithful"})

    monkeypatch.setattr(translation, "translate_segments", fake_translate)
    r = client.post("/settings/translation-test", json={
        "text": "hallo", "target": "en", "progress_id": pid})
    assert r.status_code == 200, r.text
    assert pid in app_module._PROGRESS_CLOSED
    held["download_cb"](512, 1024)          # the straggling load-thread tick
    assert pid not in app_module._BATCH_PROGRESS
    assert pid not in app_module._PROGRESS_OWNER


def test_translation_test_forwards_target_progress(
        client, app_module, monkeypatch):
    """The lab always has ONE target, so the overall fraction stays 0.0 until
    the end; target_progress is the only intra-target signal."""
    from faster_whisper_backend.audio import translation

    app_module.cfg.TRANSLATION_ENABLED = True
    pid = "f00d" * 8
    seen = {}

    async def fake_translate(segments, targets, **kwargs):
        kwargs["progress_cb"](0.0, "en 1/1", None, target="en",
                              target_progress=0.5)
        seen.update(app_module._BATCH_PROGRESS.get(pid) or {})
        return ([{"en": "hi"}], [], {"model": "org/m", "source": "de",
                                     "mode": "faithful"})

    monkeypatch.setattr(translation, "translate_segments", fake_translate)
    r = client.post("/settings/translation-test", json={
        "text": "hallo", "target": "en", "progress_id": pid})
    assert r.status_code == 200, r.text
    assert seen.get("target") == "en"
    assert seen.get("target_progress") == 0.5
