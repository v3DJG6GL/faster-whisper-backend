# CPU image — runs on any OS with Docker, no local Python setup needed.
# faster-whisper's ctranslate2 and PyAV ship self-contained wheels (PyAV bundles
# the ffmpeg libraries), so no apt ffmpeg/system codecs are required.
#
# GPU: build a derived image that also `pip install -r requirements-gpu.txt` and
# run with `--gpus all` on an NVIDIA host (see README).
#
# Base pinned by digest (supply chain: a retagged or tampered upstream image
# can't slip in unnoticed). Digest = the multi-arch index of python:3.14-slim
# (Python 3.14.6, Debian 13). To bump:
#   docker buildx imagetools inspect python:3.14-slim
FROM python:3.14-slim@sha256:ce40764625a4ff50df3548277632e7f96c4e77fe75fa848aae9885476e7df5a4

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt requirements-diarize.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Optional heavy extras (INCLUDE_EXTRAS=1 → the "-full" tag): speaker
# diarization (pyannote + torch, CPU wheels here). ffmpeg from apt — torchcodec
# decodes through the system libraries. The lean image skips all of it; the
# code lazy-imports and soft-fails with a message naming the requirements file.
ARG INCLUDE_EXTRAS=0
RUN if [ "${INCLUDE_EXTRAS}" = "1" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends ffmpeg \
      && rm -rf /var/lib/apt/lists/* \
      && pip install -r requirements-diarize.txt \
           --extra-index-url https://download.pytorch.org/whl/cpu ; \
    fi

# Non-root runtime user (a compromised app process can't rewrite /app code or
# install packages). The build args only seed the /etc/passwd entry — the
# compose `user:` line (PUID/PGID, see docker-compose.yml) overrides the
# runtime UID/GID without a rebuild. /data + /models are pre-created
# world-writable so a FRESH named volume inherits permissions any runtime UID
# can write to.
ARG PUID=1000
ARG PGID=1000
RUN groupadd -g "${PGID}" app \
 && useradd -u "${PUID}" -g "${PGID}" -M -s /usr/sbin/nologin app \
 && mkdir -p /data /models \
 && chmod 0777 /data /models

# App code. Runtime state (DBs, captures, logs, model cache) lives on mounted
# volumes — see docker-compose.yml — not baked into the image.
COPY . .

# Build identity: CI stamps the `git describe` string here (see ci.yml) so
# /v1/models and the WebUI can report the exact build — the image carries no
# .git to describe at runtime (.dockerignore). Local builds default to "dev".
ARG BUILD_VERSION=dev
ENV WHISPER_BUILD_VERSION=${BUILD_VERSION}

EXPOSE 8000

# Numeric USER so runAsNonRoot-style checks can verify it. HOME=/tmp: a
# compose-set arbitrary UID has no passwd entry (HOME=/ is unwritable) and
# stray ~/.cache writers need somewhere writable — the model cache itself is
# pinned to the persistent volume via HF_HOME/WHISPER_DOWNLOAD_ROOT in compose.
USER ${PUID}:${PGID}
ENV HOME=/tmp

# `python main.py` runs uvicorn via main's __main__; it's also what the
# cross-platform self-restart (os.execv) re-execs.
CMD ["python", "main.py"]
