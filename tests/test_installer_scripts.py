"""Pin the installer/deploy-script invariants fixed in the code review.

The shell/PowerShell/compose files have no runtime under test, so these
tests grep the scripts for the load-bearing lines instead: a regression
that drops one of them reintroduces the reviewed bug.
"""
import os
import re

from faster_whisper_backend.paths import REPO_ROOT as REPO


def _read(*rel):
    with open(os.path.join(REPO, *rel), encoding="utf-8") as fh:
        return fh.read()


# --- install-service.sh ------------------------------------------------------

def test_linux_installer_restarts_the_unit():
    # Re-runs must pick up the refreshed venv + rewritten unit: `enable --now`
    # is a no-op on an active unit, so the script must use an explicit restart.
    sh = _read("install-service.sh")
    assert re.search(r'^systemctl restart "\$\{SERVICE_NAME\}"', sh, re.M)
    assert 'systemctl enable --now' not in sh


def test_linux_installer_stops_service_before_pip():
    # pip must not swap mapped .so files under the live process.
    sh = _read("install-service.sh")
    stop = sh.index('systemctl stop "${SERVICE_NAME}"')
    first_pip = sh.index("-m pip install")
    assert stop < first_pip


def test_linux_installer_cpu_full_uses_pytorch_cpu_index():
    # The PyPI torch wheel hard-depends on the nvidia-* CUDA runtime; the CPU
    # --full branch must install the extras from the cpu wheel index.
    sh = _read("install-service.sh")
    cpu_line = ('pip install -r "$REPO_DIR/requirements-diarize.txt" \\\n'
                '      -r "$REPO_DIR/requirements-bgm.txt" \\\n'
                '      --extra-index-url https://download.pytorch.org/whl/cpu')
    assert cpu_line in sh


def test_linux_installer_precreates_logs_dir():
    # The unit pins WHISPER_LOG_FILE at $REPO_DIR/logs/whisper.log; without a
    # pre-created, chowned logs/ the service degrades to stderr-only logging.
    sh = _read("install-service.sh")
    assert re.search(r'mkdir -p .*"\$REPO_DIR/logs"', sh)
    assert re.search(r'chown -R "\$RUN_USER" .*"\$REPO_DIR/logs"', sh)


# --- uninstall-service.ps1 ---------------------------------------------------

def test_uninstall_never_executes_legacy_nssm():
    # An untrusted WhisperAPI.exe must fall back to sc.exe delete, never to
    # executing an equally unverified repo-local nssm.exe elevated.
    ps1 = _read("uninstall-service.ps1")
    assert "& $LegacyNssm" not in ps1
    assert "sc.exe delete $ServiceName" in ps1
    # $LegacyNssm stays only as a file-cleanup target under -RemoveLocal.
    assert "Remove-Item -Force $LegacyNssm" in ps1


# --- install-service.ps1 -----------------------------------------------------

def test_convert_extras_reuse_cu126_index_on_gpu():
    # requirements-convert.txt floors torch; on a -Gpu box the resolution must
    # stay on the cu126 index or pip can replace the CUDA torch with CPU/cu13.
    ps1 = _read("install-service.ps1")
    body = ps1.split("function Install-ConvertDeps", 1)[1]
    gpu_arm = body.split("if ($Gpu)", 1)[1].split("} else {", 1)[0]
    assert "-r $convertReq --extra-index-url https://download.pytorch.org/whl/cu126" in gpu_arm


# --- .dockerignore -----------------------------------------------------------

def test_dockerignore_excludes_repo_local_ffmpeg_tree():
    # install-service.ps1 -Full extracts ~150 MB of Windows ffmpeg DLLs into
    # ./ffmpeg/ (gitignored); a local docker build must not ship them.
    assert re.search(r"^ffmpeg/$", _read(".dockerignore"), re.M)


# --- docker-compose ----------------------------------------------------------

def test_compose_files_carry_the_db_layout_upgrade_note():
    # The default SQLite paths moved from /data to /data/db; pre-existing
    # volumes need the migration note or upgrades silently orphan their state.
    for name in ("docker-compose.yml", "docker-compose.gpu.yml"):
        text = _read(name)
        assert "UPGRADE NOTE" in text, name
        assert "WHISPER_DB_DIR: /data" in text, name


# --- docs/brand --------------------------------------------------------------

def test_gen_logo_svg_writes_utf8_and_has_no_dead_import():
    # The SVG payload contains a literal em dash; the write must be explicit
    # UTF-8 or LC_ALL=C runs raise UnicodeEncodeError / cp1252 writes mojibake.
    src = _read("docs", "brand", "gen-logo-svg.py")
    assert 'encoding="utf-8"' in src
    assert re.search(r"^import sys$", src, re.M) is None


def test_logo_html_comment_no_longer_claims_verbatim_mark():
    # The inlined mark renames the gradient id fw-wave -> fw; the comment must
    # not invite a literal re-sync from static/logo.svg.
    html = _read("docs", "brand", "logo.html")
    assert "inlined verbatim" not in html
    assert 'id="fw"' in html and "fw-wave" in html
