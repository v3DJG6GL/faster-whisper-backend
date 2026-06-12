#!/usr/bin/env bash
# Linux (systemd) installer — the cross-platform counterpart to
# install-service.ps1. Creates/uses a local venv, installs dependencies, writes
# a systemd unit, and enables + starts it.
#
#   ./install-service.sh           # CPU
#   ./install-service.sh --gpu     # also install NVIDIA CUDA wheels
#
# Re-runs are safe (idempotent): it refreshes deps and the unit, then restarts.
set -euo pipefail

SERVICE_NAME="whisper-api"
GPU=0
for arg in "$@"; do
  case "$arg" in
    --gpu) GPU=1 ;;
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

echo "Installing dependencies (gpu=$GPU) ..."
sudo -u "$RUN_USER" "$PY" -m pip install --upgrade pip
if [ "$GPU" -eq 1 ]; then
  sudo -u "$RUN_USER" "$PY" -m pip install -r "$REPO_DIR/requirements.txt" -r "$REPO_DIR/requirements-gpu.txt"
else
  sudo -u "$RUN_USER" "$PY" -m pip install -r "$REPO_DIR/requirements.txt"
fi

# ffmpeg: only the live-streaming *encoded* transport (browser Opus/WebM) needs
# the ffmpeg executable; raw-PCM dictation does not. imageio-ffmpeg (installed
# above via requirements.txt) bundles a binary as a guaranteed fallback, but a
# system ffmpeg is preferred when present. Best-effort install on Debian/Ubuntu.
if command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg present: $(ffmpeg -version 2>/dev/null | head -1)"
elif command -v apt-get >/dev/null 2>&1; then
  echo "ffmpeg not found; installing via apt-get (optional) ..."
  apt-get update -qq && apt-get install -y ffmpeg \
    || echo "  apt install failed; falling back to the bundled imageio-ffmpeg binary."
else
  echo "ffmpeg not found and no apt-get; using the bundled imageio-ffmpeg binary."
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

echo
echo "Done. Manage with:"
echo "  systemctl status ${SERVICE_NAME}"
echo "  systemctl restart ${SERVICE_NAME}"
echo "  journalctl -u ${SERVICE_NAME} -f"
echo "  ./uninstall-service.sh"
