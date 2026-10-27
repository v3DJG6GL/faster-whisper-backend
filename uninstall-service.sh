#!/usr/bin/env bash
# Linux (systemd) uninstaller — counterpart to uninstall-service.ps1. Stops and
# removes the service unit. Leaves the venv, the HuggingFace model cache, and
# the runtime data (DBs, captures, logs) in place — same policy as the .ps1.
set -euo pipefail

SERVICE_NAME="whisper-api"

if [ "$(id -u)" -ne 0 ]; then
  echo "Elevating with sudo..."
  exec sudo -E "$0" "$@"
fi

UNIT="/etc/systemd/system/${SERVICE_NAME}.service"

# A here-string, not `systemctl list-unit-files | grep -q`: grep -q exits on
# the first match and systemctl, still writing, dies of SIGPIPE — under
# pipefail an installed unit reads as "not installed" and stays running.
if grep -q "^${SERVICE_NAME}\.service" <<< "$(systemctl list-unit-files)"; then
  echo "Stopping + disabling ${SERVICE_NAME} ..."
  systemctl disable --now "${SERVICE_NAME}" || true
else
  echo "${SERVICE_NAME} is not installed; nothing to stop."
fi

if [ -f "$UNIT" ]; then
  echo "Removing $UNIT ..."
  rm -f "$UNIT"
  systemctl daemon-reload
fi

echo "Done. (venv, model cache, and *.local.sqlite3 data were left intact.)"
