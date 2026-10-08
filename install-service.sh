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
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found on PATH; install Python 3.12+ and re-run." >&2
    exit 1
  fi
  # CI tests 3.12-3.14; refuse older interpreters before building a venv on them.
  if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
    echo "python3 is older than 3.12 ($(python3 --version 2>&1)); install Python 3.12+ and re-run." >&2
    exit 1
  fi
  echo "Creating venv at $VENV ..."
  # Create the venv as the invoking user so they own it.
  sudo -u "$RUN_USER" python3 -m venv "$VENV"
fi

# Stop a running service before pip touches the venv — otherwise wheels whose
# .so files the live process has mapped (torch, ctranslate2, onnxruntime) get
# swapped underneath it (mirrors install-service.ps1's stop -> install -> start).
systemctl stop "${SERVICE_NAME}" 2>/dev/null || true

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
    # requirements-bgm.txt pins the versions (single source of truth,
    # Renovate-bumped); "audio-separator[gpu]" only swaps in the GPU extra.
    sudo -u "$RUN_USER" "$PY" -m pip install -r "$REPO_DIR/requirements-diarize.txt" \
      -r "$REPO_DIR/requirements-bgm.txt" "audio-separator[gpu]" \
      --extra-index-url https://download.pytorch.org/whl/cu126
    sudo -u "$RUN_USER" "$PY" -m pip install --force-reinstall --no-deps "onnxruntime-gpu==1.26.*"
    # Translation (llama-cpp-python): the project's cu124 wheel index — no
    # cu126 index exists; cu124 wheels run on a cu12.6 userspace (CUDA 12
    # minor-version compatibility). PyPI is sdist-only (source build).
    sudo -u "$RUN_USER" "$PY" -m pip install -r "$REPO_DIR/requirements-translate.txt" \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
  else
    # CPU torch from the PyTorch cpu index — the PyPI wheel hard-depends on
    # the whole nvidia-* CUDA runtime (mirrors Dockerfile's extras install).
    sudo -u "$RUN_USER" "$PY" -m pip install -r "$REPO_DIR/requirements-diarize.txt" \
      -r "$REPO_DIR/requirements-bgm.txt" \
      --extra-index-url https://download.pytorch.org/whl/cpu
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
# faster_whisper_backend/main.py's CUDA-lib preloader is Windows-only — put the lib dirs on the
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

# Whether the checkout's .env pins a variable. A .env pin must win over the
# unit: config.py's load_dotenv never overrides a variable the real
# environment already has, so an Environment= line below would silently throw
# away e.g. WHISPER_DATA_DIR=/srv/whisper — and restart on an empty key store.
# Read by the service's own python-dotenv (none installed: the server reads no
# .env either), so `KEY = value` counts and an empty value does not (config.py
# treats "" as unset and falls back to /data, which the service user cannot
# create). A value still holding a `$` does not count either: dotenv leaves a
# bare $PWD as is, and systemd sets no PWD for ${PWD} — the unit pin stays.
env_value() {
  [ -f "$REPO_DIR/.env" ] || return 0
  "$PY" -I -c 'import sys
try:
    from dotenv import dotenv_values
except ImportError:
    sys.exit(0)
print((dotenv_values(sys.argv[1], interpolate=False).get(sys.argv[2]) or "").strip())' \
    "$REPO_DIR/.env" "$1" 2>/dev/null || true
}
env_sets() {
  local v
  v="$(env_value "$1")"
  [ -n "$v" ] && [ "${v#*\$}" = "$v" ]
}
for name in WHISPER_DATA_DIR WHISPER_DB_DIR WHISPER_MODELS_DIR WHISPER_LOG_FILE; do
  case "$(env_value "$name")" in
    *'$'*) printf '\033[33mWARNING: %s\033[0m\n' \
      "$REPO_DIR/.env sets $name to a value with a \$ the service cannot expand" \
      "(dotenv leaves \$PWD as is; systemd sets no PWD) — ignored by this installer;" \
      "use an absolute path." >&2 ;;
  esac
done
# Bare-metal Linux: root the container-first default paths (/data, /models)
# in the checkout instead, mirroring the Windows in-checkout layout — without
# these the service user cannot create /data/db and startup crash-loops.
# Each only when .env does not pin it already.
DATA_ENV_LINE="" MODELS_ENV_LINE="" LOG_ENV_LINE="" STATE_DIRS=()
if ! env_sets WHISPER_DATA_DIR; then
  DATA_ENV_LINE="Environment=WHISPER_DATA_DIR=${REPO_DIR}/data"
  STATE_DIRS+=("$REPO_DIR/data")
fi
if ! env_sets WHISPER_MODELS_DIR; then
  MODELS_ENV_LINE="Environment=WHISPER_MODELS_DIR=${REPO_DIR}/models"
  STATE_DIRS+=("$REPO_DIR/models")
fi
# LOG_FILE defaults to {DATA_DIR}/logs/whisper.log: a pinned data root keeps
# the log with it (the service can write there — its stores live below it).
if ! env_sets WHISPER_LOG_FILE && ! env_sets WHISPER_DATA_DIR; then
  LOG_ENV_LINE="Environment=WHISPER_LOG_FILE=${REPO_DIR}/logs/whisper.log"
  STATE_DIRS+=("$REPO_DIR/logs")
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
# In-checkout state paths (unless ${REPO_DIR}/.env pins them).
${DATA_ENV_LINE}
${MODELS_ENV_LINE}
${LOG_ENV_LINE}
${NVIDIA_ENV_LINE}
# 'python main.py' is the shim over faster_whisper_backend/main.py; matches what the
# cross-platform self-restart (os.execv) re-execs.
ExecStart=${PY} ${REPO_DIR}/main.py
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

# Pre-create the state dirs the unit pins above, owned by the service user
# (api_keys_store refuses to start when it cannot create its db dir; faster_whisper_backend/main.py
# soft-fails to stderr-only logging when logs/ cannot be created). A .env-pinned
# path is the operator's to provision.
if [ "${#STATE_DIRS[@]}" -gt 0 ]; then
  mkdir -p "${STATE_DIRS[@]}"
  chown -R "$RUN_USER" "${STATE_DIRS[@]}"
fi

# Upgrade check, warn only (never moves anything — a .env may pin the old
# paths on purpose): an older checkout kept the SQLite stores and
# config.local.json in the repo root, the unit above roots them in data/db
# and data/. Restarting over that comes up with an EMPTY api_keys store —
# issued keys stop working and the server drops into OPEN mode.
warn_legacy() { printf '\033[33mWARNING: %s\033[0m\n' "$*" >&2; }
# Warn while a legacy store EXISTS, not only while data/db/<name> is missing:
# the restart below creates an empty data/db/<name> on the first run, which
# would silence every later run while the old keys sit ignored. The stores are
# WAL databases, so the -wal/-shm sidecars move with the file, and the fresh
# store's own sidecars must go first (or they replay onto the moved file).
# A store with its own .env pin (WHISPER_API_KEYS_DB=...) is read from there,
# not data/db — moving it would strand the live store, so it is left be.
store_env() {
  case "$1" in
    system_metrics.local.sqlite3) echo WHISPER_STATS_SYSTEM_METRICS_DB ;;
    *.local.sqlite3) local stem="${1%.local.sqlite3}"
      echo "WHISPER_${stem^^}_DB" ;;
  esac
}
if ! env_sets WHISPER_DB_DIR && ! env_sets WHISPER_DATA_DIR; then
  for legacy in "$REPO_DIR"/*.local.sqlite3 "$REPO_DIR"/data/*.local.sqlite3; do
    [ -e "$legacy" ] || continue
    env_sets "$(store_env "$(basename "$legacy")")" && continue
    target="$REPO_DIR/data/db/$(basename "$legacy")"
    warn_legacy "legacy store $legacy is IGNORED by the service (it reads $REPO_DIR/data/db/)." \
      "Move it: systemctl stop ${SERVICE_NAME} && sudo -u $RUN_USER mkdir -p $REPO_DIR/data/db" \
      "&& rm -f '$target' '$target-wal' '$target-shm' && mv '$legacy'* $REPO_DIR/data/db/" \
      "&& systemctl start ${SERVICE_NAME} (the rm drops the EMPTY store a start without it" \
      "created — skip it if $target holds keys you issued since) — or set" \
      "WHISPER_DB_DIR=$(dirname "$legacy") in $REPO_DIR/.env and restart. Already moved?" \
      "Delete the leftover $legacy."
  done
fi
if ! env_sets WHISPER_CONFIG_LOCAL && ! env_sets WHISPER_DATA_DIR \
    && [ -e "$REPO_DIR/config.local.json" ] && [ ! -e "$REPO_DIR/data/config.local.json" ]; then
  warn_legacy "legacy $REPO_DIR/config.local.json is IGNORED by the service (it reads $REPO_DIR/data/config.local.json)." \
    "Move it: mv '$REPO_DIR/config.local.json' $REPO_DIR/data/ — then restart."
fi

echo "Enabling + restarting ${SERVICE_NAME} ..."
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
# `restart` also starts an inactive unit, so first install and re-run both come
# up on the fresh venv + rewritten unit (`enable --now` is a no-op when active).
systemctl restart "${SERVICE_NAME}"

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
