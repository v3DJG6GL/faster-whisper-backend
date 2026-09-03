"""Launcher shim.

The application lives in faster_whisper_backend/main.py. This file keeps
`python main.py`, `uvicorn main:app`, the Dockerfiles' CMD, the systemd unit
(install-service.sh) and the WinSW service XML (install-service.ps1) working
unchanged. It defines nothing itself — `app` is the package's object.
"""
from faster_whisper_backend.main import app, run  # noqa: F401

if __name__ == "__main__":
    run()
