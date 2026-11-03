"""Helpers shared by the stage tests that drive POST /v1/audio/transcriptions
(the stub upload and a verbose_json post). Import as
``from tests._transcription_helpers import ...``."""

WAV_UPLOAD = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}


def post_verbose(client, **data):
    data.setdefault("model", "whisper-1")
    data.setdefault("response_format", "verbose_json")
    return client.post("/v1/audio/transcriptions", files=WAV_UPLOAD, data=data)
