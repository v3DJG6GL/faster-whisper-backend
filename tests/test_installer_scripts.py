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
    # Nor the WinSW wrapper: a hash check before running it elevated is
    # check-then-use on a repo-local file, and sc.exe delete does the same job.
    assert "& $WinSWExe" not in ps1
    assert "Test-WinSWTrusted" not in ps1


# --- install-service.ps1 -----------------------------------------------------

def test_convert_extras_reuse_cu126_index_on_gpu():
    # requirements-convert.txt floors torch; on a -Gpu box the resolution must
    # stay on the cu126 index or pip can replace the CUDA torch with CPU/cu13.
    ps1 = _read("install-service.ps1")
    body = ps1.split("function Install-ConvertDeps", 1)[1]
    gpu_arm = body.split("if ($Gpu)", 1)[1].split("} else {", 1)[0]
    assert "-r $convertReq --extra-index-url https://download.pytorch.org/whl/cu126" in gpu_arm


def test_install_never_executes_unverified_wrapper():
    # The pre-flight removal block runs elevated BEFORE the SHA-256 pin check;
    # it must gate WinSW on Test-WinSWTrusted and never exec repo-local nssm.exe.
    ps1 = _read("install-service.ps1")
    assert "& $LegacyNssm" not in ps1
    assert "if (Test-WinSWTrusted) {\n        & $WinSWExe uninstall" in ps1
    assert "sc.exe delete $ServiceName" in ps1
    # The hash table must be defined before the removal block uses it.
    assert ps1.index("$WinSWHashes = @{") < ps1.index("& $WinSWExe uninstall")
    # nssm.exe is still cleaned up as a file.
    assert "Remove-Item -Force $LegacyNssm" in ps1


def test_winsw_xml_here_string_is_well_formed():
    # WinSW v2 loads WhisperAPI.xml with XmlDocument: a malformed document
    # (e.g. "--" inside a comment) fails the service install outright.
    import xml.etree.ElementTree as ET
    s = _read("install-service.ps1")
    x = re.search(r'\$xml = @"\n(.*?)\n"@', s, re.S).group(1)
    x = re.sub(r"\$\([^)]*\)", "X", x)
    x = re.sub(r"\$[A-Za-z_]\w*", "X", x)
    ET.fromstring(x)
    for body in re.findall(r"<!--(.*?)-->", x, re.S):
        assert "--" not in body          # XML forbids "--" inside a comment


# --- .dockerignore -----------------------------------------------------------

def test_dockerignore_excludes_repo_local_ffmpeg_tree():
    # install-service.ps1 -Full extracts ~150 MB of Windows ffmpeg DLLs into
    # ./ffmpeg/ (gitignored); a local docker build must not ship them.
    assert re.search(r"^ffmpeg/$", _read(".dockerignore"), re.M)


# --- docker-compose ----------------------------------------------------------

def _nft_snippets():
    """The egress nftables snippet as README.md and both compose files ship
    it (compose: commented out, up to the next bare `#` line)."""
    readme = _read("README.md")
    out = {"README.md": re.search(r"```nft\n(.*?)```", readme, re.S).group(1)}
    for name in ("docker-compose.yml", "docker-compose.gpu.yml"):
        lines = _read(name).splitlines()
        start = next(i for i, ln in enumerate(lines)
                     if "/etc/nftables.d/whisper-egress.nft" in ln)
        block = []
        for ln in lines[start:]:
            if ln.strip() == "#":
                break
            block.append(ln.strip().lstrip("#"))
        out[name] = "\n".join(block)
    return out


def test_egress_nft_snippets_leave_the_hosts_ipv6_alone():
    # The subnet is IPv4-only and an `ip` match never matches IPv6: without
    # the nfproto line every forwarded IPv6 packet on the host fell through
    # to the drops. The ip6 rules belong in the prose for an IPv6-enabled
    # service network, never as an active rule in the snippet.
    snippets = _nft_snippets()
    for name, text in snippets.items():
        rules = [ln.split("#", 1)[0].strip() for ln in text.splitlines()]
        assert "meta nfproto != ipv4 return" in rules, name
        assert any(r.startswith("ct  state established,related return")
                   for r in rules), name
        assert not any(r.startswith("ip6 daddr") for r in rules), name


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


def test_gpu_compose_is_the_cpu_compose_plus_the_gpu_bits():
    """docker-compose.gpu.yml is a standalone copy of docker-compose.yml (not
    an overlay). Apart from comments, the only allowed differences are the
    image tag and the GPU `deploy:` reservation block; anything else is
    drift that one file got and the other missed."""
    def code(name):
        return [ln.rstrip() for ln in _read(name).splitlines()
                if ln.strip() and not ln.strip().startswith("#")]

    cpu, gpu = code("docker-compose.yml"), code("docker-compose.gpu.yml")
    start = next(i for i, ln in enumerate(gpu) if ln.strip() == "deploy:")
    indent = len(gpu[start]) - len(gpu[start].lstrip())
    end = start + 1
    while end < len(gpu) and len(gpu[end]) - len(gpu[end].lstrip()) > indent:
        end += 1
    assert "nvidia" in "\n".join(gpu[start:end])
    gpu = gpu[:start] + gpu[end:]
    strip_tag = lambda lines: [re.sub(r"^(\s*image: \S+?):[\w.-]+$", r"\1", ln) for ln in lines]
    assert strip_tag(cpu) == strip_tag(gpu)


# --- PowerShell encoding -----------------------------------------------------

def test_powershell_scripts_are_pure_ascii():
    # The installers self-elevate via `Start-Process powershell`, which is
    # always Windows PowerShell 5.1: it reads a BOM-less script as the ANSI
    # code page, so UTF-8 punctuation turns into mojibake (an em dash became
    # a€” in WhisperAPI.xml), and cp1252 0x94 is a smart quote that
    # PowerShell treats as a string delimiter.
    for name in ("install-service.ps1", "uninstall-service.ps1"):
        with open(os.path.join(REPO, name), "rb") as fh:
            data = fh.read()
        bad = [i for i, b in enumerate(data) if b > 127]
        assert not bad, f"{name}: non-ASCII byte at offset {bad[0]}"


def test_installers_require_python_312():
    # CI tests 3.12-3.14; both installers refuse older interpreters before
    # building a venv on them, and say so.
    assert "sys.version_info >= (3, 12)" in _read("install-service.sh")
    ps1 = _read("install-service.ps1")
    assert "sys.version_info >= (3, 12)" in ps1
    assert "Python 3.10" not in ps1


def test_installers_warn_about_legacy_repo_root_state_before_the_start():
    # Since the data-dir move the service reads data/db/ and
    # data/config.local.json; stores and config left in the repo root are
    # silently ignored. Both installers warn (never move) BEFORE the service
    # (re)starts, so an upgrade does not quietly come up with empty state.
    sh = _read("install-service.sh")
    restart = re.search(r'^systemctl restart "\$\{SERVICE_NAME\}"', sh, re.M)
    assert restart
    head = sh[:restart.start()]
    assert "warn_legacy()" in head
    assert '"$REPO_DIR"/*.local.sqlite3' in head
    assert re.search(r'warn_legacy "legacy \$REPO_DIR/config\.local\.json', head)
    ps1 = _read("install-service.ps1")
    check = ps1.index("# --- upgrade check")
    start = ps1.index("\nInvoke-WinSW start")
    assert check < start
    block = ps1[check:start]
    assert '-Filter "*.local.sqlite3"' in block
    assert block.count("Write-Warning") >= 2   # stores + config.local.json
    assert "config.local.json" in block
