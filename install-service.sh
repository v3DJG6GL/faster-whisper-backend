#!/usr/bin/env bash
# Linux (systemd) installer — the cross-platform counterpart to
# install-service.ps1. Creates/uses a local venv, installs dependencies, writes
# a systemd unit, and enables + starts it.
#
#   ./install-service.sh              # CPU
#   ./install-service.sh --gpu        # also install NVIDIA CUDA wheels
#   ./install-service.sh --full       # also install the heavy extras
#   ./install-service.sh --gpu --full # (= the Docker :latest-gpu-full image)
#
# --full mirrors the Docker "-full" tags (Dockerfile / Dockerfile.gpu with
# INCLUDE_EXTRAS=1): speaker diarization (pyannote) + background-music
# separation (audio-separator), plus a system ffmpeg. The lean install stays
# fully functional — those requests just soft-fail with a warning naming the
# missing requirements file.
#
# Re-runs are safe (idempotent): it refreshes deps and the unit, then restarts.
# Pass the SAME flags on every re-run — a re-run without --full leaves already-
# installed extras in place but does not refresh them, and re-running a GPU
# box's --full without --gpu would downgrade torch to the CPU build.
set -euo pipefail

SERVICE_NAME="whisper-api"
GPU=0
FULL=0
for arg in "$@"; do
  case "$arg" in
    --gpu) GPU=1 ;;
    --full) FULL=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

# Resolve the repo dir from this script's location (stable across the sudo
# re-exec below).
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

# systemctl + writing the unit need root; re-exec under sudo, preserving env so
# $SUDO_USER survives (mirrors the .ps1 UAC elevation).
if [ "$(id -u)" -ne 0 ]; then
  echo "Elevating with sudo..."
  exec sudo -E "$0" "$@"
fi

# Run the service as the human who invoked us, not root.
RUN_USER="${SUDO_USER:-root}"

VENV="$REPO_DIR/venv"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "Creating venv at $VENV ..."
  # Create the venv as the invoking user so they own it.
  sudo -u "$RUN_USER" python3 -m venv "$VENV"
fi

echo "Installing dependencies (gpu=$GPU full=$FULL) ..."
sudo -u "$RUN_USER" "$PY" -m pip install --upgrade pip
if [ "$GPU" -eq 1 ]; then
  sudo -u "$RUN_USER" "$PY" -m pip install -r "$REPO_DIR/requirements.txt" -r "$REPO_DIR/requirements-gpu.txt"
else
  sudo -u "$RUN_USER" "$PY" -m pip install -r "$REPO_DIR/requirements.txt"
fi

# --full extras. Keep the GPU branch in sync with Dockerfile.gpu's
# INCLUDE_EXTRAS=1 block — same packages, same pins, same reasons:
#   * torch comes from the cu126 index so it shares the ONE CUDA 12 userspace
#     ctranslate2's pip wheels use (PyPI-default torch pulls the cu13 stack).
#   * onnxruntime-gpu >=1.27 on PyPI is a CUDA 13 build — on a cu12 stack its
#     CUDA provider fails to load and separation silently runs on the CPU.
#     1.26.x is the last CUDA 12.8 build; forced LAST (--no-deps) so its files
#     also win over the CPU-only onnxruntime faster-whisper pulls in.
#   * audio-separator's diffq dep may compile from source (no wheel on newer
#     Pythons) — best-effort gcc/g++ below mirrors the Dockerfile's build deps.
if [ "$FULL" -eq 1 ]; then
  if ! command -v gcc >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    echo "gcc not found; installing (audio-separator's diffq dep may build from source) ..."
    apt-get update -qq && apt-get install -y gcc g++ \
      || echo "  apt install failed; pip will error out if diffq has no prebuilt wheel."
  fi
  echo "Installing full extras (diarization + music separation + translation, several GB) ..."
  if [ "$GPU" -eq 1 ]; then
    sudo -u "$RUN_USER" "$PY" -m pip install -r "$REPO_DIR/requirements-diarize.txt" \
      "audio-separator[gpu]>=0.44" "audioread>=2.1.9" "librosa<1.0" \
      --extra-index-url https://download.pytorch.org/whl/cu126
    sudo -u "$RUN_USER" "$PY" -m pip install --force-reinstall --no-deps "onnxruntime-gpu==1.26.*"
    # Translation (llama-cpp-python): the project's cu124 wheel index — no
    # cu126 index exists; cu124 wheels run on a cu12.6 userspace (CUDA 12
    # minor-version compatibility). PyPI is sdist-only (source build).
    sudo -u "$RUN_USER" "$PY" -m pip install -r "$REPO_DIR/requirements-translate.txt" \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
  else
    sudo -u "$RUN_USER" "$PY" -m pip install -r "$REPO_DIR/requirements-diarize.txt" -r "$REPO_DIR/requirements-bgm.txt"
    # Translation (llama-cpp-python): prebuilt CPU wheels from the project
    # index — PyPI is sdist-only and would compile llama.cpp from source.
    sudo -u "$RUN_USER" "$PY" -m pip install -r "$REPO_DIR/requirements-translate.txt" \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
  fi
fi

# ffmpeg: only the live-streaming *encoded* transport (browser Opus/WebM) needs
# the ffmpeg executable; raw-PCM dictation does not. imageio-ffmpeg (installed
# above via requirements.txt) bundles a binary as a guaranteed fallback, but a
# system ffmpeg is preferred when present. Best-effort install on Debian/Ubuntu.
# With --full a SYSTEM ffmpeg is required, not optional: torchcodec (pyannote's
# decoder) and audio-separator load the ffmpeg *libraries*, which the bundled
# imageio-ffmpeg executable does not provide.
if command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg present: $(ffmpeg -version 2>/dev/null | head -1)"
elif command -v apt-get >/dev/null 2>&1; then
  echo "ffmpeg not found; installing via apt-get ..."
  apt-get update -qq && apt-get install -y ffmpeg \
    || echo "  apt install failed; falling back to the bundled imageio-ffmpeg binary."
else
  echo "ffmpeg not found and no apt-get; using the bundled imageio-ffmpeg binary."
fi
if [ "$FULL" -eq 1 ] && ! command -v ffmpeg >/dev/null 2>&1; then
  echo "WARNING: --full needs a system ffmpeg (torchcodec / audio-separator decode" >&2
  echo "  through its libraries). Install it manually, then restart the service." >&2
fi

# GPU: ctranslate2 (and, with --full, onnxruntime-gpu's CUDA provider) dlopen
# the pip-installed NVIDIA .so libs but do not search site-packages, and
# main.py's CUDA-lib preloader is Windows-only — put the lib dirs on the
# loader path via the unit, mirroring Dockerfile.gpu's LD_LIBRARY_PATH.
# Dirs that don't exist (lean GPU install ships only cublas+cudnn) are
# skipped by the loader.
NVIDIA_ENV_LINE=""
if [ "$GPU" -eq 1 ]; then
  SITE_PKGS="$(sudo -u "$RUN_USER" "$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  LD_PATHS=""
  for lib in cublas cudnn cuda_runtime cuda_nvrtc cufft curand cusolver cusparse nvjitlink; do
    LD_PATHS="${LD_PATHS:+${LD_PATHS}:}${SITE_PKGS}/nvidia/${lib}/lib"
  done
  NVIDIA_ENV_LINE="Environment=LD_LIBRARY_PATH=${LD_PATHS}"
fi

UNIT="/etc/systemd/system/${SERVICE_NAME}.service"
echo "Writing $UNIT ..."
cat > "$UNIT" <<EOF
[Unit]
Description=Faster Whisper API backend
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${REPO_DIR}
Environment=WHISPER_LOG_FILE=${REPO_DIR}/logs/whisper.log
${NVIDIA_ENV_LINE}
# 'python main.py' runs uvicorn via main's __main__; matches what the
# cross-platform self-restart (os.execv) re-execs.
ExecStart=${PY} ${REPO_DIR}/main.py
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

echo "Enabling + starting ${SERVICE_NAME} ..."
systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"

if [ "$FULL" -eq 1 ]; then
  echo
  echo "Full extras installed. Notes:"
  echo "  - Diarization's gated pyannote pipelines need accepted model terms on"
  echo "    huggingface.co plus WHISPER_HF_TOKEN set (e.g. in ${REPO_DIR}/.env)."
  echo "  - Model weights (pyannote, MDX-Net) are not pip packages; they download"
  echo "    on first use into the download root."
fi

echo
echo "Done. Manage with:"
echo "  systemctl status ${SERVICE_NAME}"
echo "  systemctl restart ${SERVICE_NAME}"
echo "  journalctl -u ${SERVICE_NAME} -f"
echo "  ./uninstall-service.sh"
