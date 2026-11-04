"""Pin the installer/deploy-script invariants.

The shell/PowerShell/compose files have no runtime under test, so most
tests grep the scripts for the load-bearing lines (a few run extracted
bash snippets): a regression that drops one reintroduces the bug it guards.
"""
import io
import os
import re
import shutil
import subprocess
import sys

import pytest

from faster_whisper_backend.paths import REPO_ROOT as REPO


def _read(*rel):
    with open(os.path.join(REPO, *rel), encoding="utf-8") as fh:
        return fh.read()


def _bash(script, env):
    return subprocess.run([shutil.which("bash"), "-c", script], capture_output=True,
                          text=True, timeout=60, env=env)


def _bash_ok(script, env):
    out = _bash(script, env)
    assert out.returncode == 0, out.stderr
    return out


def _exe(path, body):
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


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


@pytest.mark.skipif(sys.platform == "win32", reason="bash installer")
def test_linux_installer_runs_venv_steps_without_sudo_as_root(tmp_path):
    # Already root with no SUDO_USER (su -, a root container) skips the sudo
    # re-exec — every venv/pip step must then run without sudo, which such a
    # host may not have. Run the script's own as_user with a stub sudo first
    # on PATH that echoes its argv instead of running anything.
    sh = _read("install-service.sh")
    helper = re.search(r"^as_user\(\) .*$", sh, re.M).group(0)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _exe(bindir / "sudo", '#!/bin/sh\necho "sudo $*"\n')

    def run(user):
        return _bash_ok(f"set -euo pipefail\n{helper}\nas_user echo hi",
                        {"PATH": f"{bindir}:/usr/bin:/bin", "RUN_USER": user}).stdout
    assert run("root") == "hi\n"
    assert run("alice") == "sudo -u alice echo hi\n"
    # Every call site goes through the helper (the hint text at the end is
    # printed, not run).
    assert sh.count('sudo -u "$RUN_USER"') == 1


def test_linux_installer_cpu_full_uses_pytorch_cpu_index():
    # The PyPI torch wheel hard-depends on the nvidia-* CUDA runtime; the CPU
    # --full branch must install the extras from the cpu wheel index.
    sh = _read("install-service.sh")
    cpu_line = ('pip install -r "$REPO_DIR/requirements-diarize.txt" \\\n'
                '      -r "$REPO_DIR/requirements-bgm.txt" \\\n'
                '      --extra-index-url https://download.pytorch.org/whl/cpu')
    assert cpu_line in sh


def test_linux_installer_precreates_logs_dir():
    # The unit pins WHISPER_LOG_FILE at $REPO_DIR/logs/whisper.log (unless
    # .env sets it); without a pre-created, chowned logs/ the service degrades
    # to stderr-only logging.
    sh = _read("install-service.sh")
    assert re.search(r'LOG_ENV_LINE="Environment=WHISPER_LOG_FILE=\$\{REPO_DIR\}/logs/whisper.log"\n'
                     r'\s*STATE_DIRS\+=\("\$REPO_DIR/logs"\)', sh)
    assert 'mkdir -p "${STATE_DIRS[@]}"' in sh
    assert 'chown -R "$RUN_USER" "${STATE_DIRS[@]}"' in sh


def test_linux_installer_unit_pins_yield_to_a_dotenv_pin():
    # A .env pin must suppress the unit's Environment= line (and its
    # pre-created dir): config.py's load_dotenv never overrides the real
    # environment, so the unit line would silently win over the pin.
    sh = _read("install-service.sh")
    for var, line in (("WHISPER_DATA_DIR", "DATA_ENV_LINE"),
                      ("WHISPER_MODELS_DIR", "MODELS_ENV_LINE"),
                      ("WHISPER_LOG_FILE", "LOG_ENV_LINE")):
        assert re.search(rf'if ! env_sets {var}\b[^\n]*; then\n\s*{line}=',
                         sh), var


@pytest.mark.skipif(shutil.which("bash") is None or os.name == "nt",
                    reason="needs a POSIX bash")
def test_linux_installer_unit_survives_a_checkout_path_with_space_and_percent(tmp_path):
    # systemd splits ExecStart=/Environment= on whitespace and expands %
    # specifiers: unquoted, "My Projects" handed python a truncated main.py
    # path. Render the script's own heredoc for such a REPO_DIR and split the
    # lines the way systemd does (shell-like quoting, %% -> %).
    import shlex
    sh = _read("install-service.sh")
    helpers = "\n".join(re.search(rf"^{n}\(\) .*$", sh, re.M).group(0)
                        for n in ("sd_q", "unit_env"))
    heredoc = re.search(r'^cat > "\$UNIT" <<EOF\n.*?^EOF\n', sh, re.S | re.M).group(0)
    repo = "/home/u/My Projects/fw%b"
    unit = tmp_path / "unit"
    script = (f"set -euo pipefail\n{helpers}\n"
              "DATA_ENV_LINE=\"Environment=WHISPER_DATA_DIR=${REPO_DIR}/data\"\n"
              "MODELS_ENV_LINE='' LOG_ENV_LINE='' NVIDIA_ENV_LINE=''\n" + heredoc)
    _bash_ok(script, {"PATH": "/usr/bin:/bin", "UNIT": str(unit), "REPO_DIR": repo,
                      "PY": f"{repo}/venv/bin/python", "RUN_USER": "u"})
    lines = dict(ln.split("=", 1) for ln in unit.read_text().splitlines()
                 if "=" in ln and not ln.startswith("#"))

    def systemd_split(value):
        return [w.replace("%%", "%") for w in shlex.split(value)]
    assert systemd_split(lines["ExecStart"]) == [f"{repo}/venv/bin/python",
                                                 f"{repo}/main.py"]
    assert systemd_split(lines["Environment"]) == [f"WHISPER_DATA_DIR={repo}/data"]
    assert lines["WorkingDirectory"] == repo.replace("%", "%%")   # %b = boot ID


def test_linux_installer_env_sets_reads_dotenv_like_the_service(tmp_path):
    # Behavioural, not a grep: source the script's own env_value / env_sets
    # into bash against a tmp .env and ask about each spelling.
    bash = shutil.which("bash")
    if bash is None or os.name == "nt":
        pytest.skip("needs a POSIX bash")
    funcs = _env_funcs(_read("install-service.sh"))
    (tmp_path / ".env").write_text(
        "WHISPER_DATA_DIR=/srv/whisper\n"
        "# WHISPER_DB_DIR=/commented/out\n"
        "export WHISPER_MODELS_DIR=/srv/models\n"
        "WHISPER_LOG_FILE = /srv/log/whisper.log\n"
        "WHISPER_CONFIG_LOCAL=\n"
        "WHISPER_JOBS_DB=$PWD/jobs.sqlite3\n", encoding="utf-8")
    names = ("WHISPER_DATA_DIR", "WHISPER_DB_DIR", "WHISPER_MODELS_DIR",
             "WHISPER_LOG_FILE", "WHISPER_CONFIG_LOCAL", "WHISPER_JOBS_DB",
             "WHISPER_NOT_THERE")
    script = funcs + "".join(
        f'if env_sets {n}; then echo "{n}=1"; else echo "{n}=0"; fi\n'
        for n in names)

    def _run(repo_dir):
        out = _bash_ok(script, {**os.environ, "REPO_DIR": str(repo_dir), "PY": sys.executable})
        return dict(line.split("=", 1) for line in out.stdout.split())
    assert _run(tmp_path) == {
        "WHISPER_DATA_DIR": "1", "WHISPER_DB_DIR": "0",
        "WHISPER_MODELS_DIR": "1", "WHISPER_LOG_FILE": "1",
        "WHISPER_CONFIG_LOCAL": "0", "WHISPER_JOBS_DB": "0",
        "WHISPER_NOT_THERE": "0"}
    # No .env at all: nothing is pinned.
    empty = tmp_path / "no-env"
    empty.mkdir()
    assert set(_run(empty).values()) == {"0"}


@pytest.mark.skipif(shutil.which("bash") is None or os.name == "nt",
                    reason="needs a POSIX bash")
def test_linux_uninstaller_sees_an_installed_unit_in_a_long_listing(tmp_path):
    # `systemctl list-unit-files | grep -q` under pipefail: grep exits on the
    # match, systemctl dies of SIGPIPE, and the installed unit was reported
    # "not installed" — left running while its unit file was deleted. Run the
    # script's own condition against a fake systemctl with a >64 KB listing.
    sh = _read("uninstall-service.sh")
    cond = re.search(r"^if .*list-unit-files.*; then$", sh, re.M).group(0)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _exe(bindir / "systemctl", "#!/bin/sh\n"
         "echo 'whisper-api.service enabled enabled'\n"
         "yes 'zz-other.service static -' | head -c 2000000\n")
    script = ("set -euo pipefail\nSERVICE_NAME=whisper-api\n"
              f"{cond}\n  echo installed\nelse\n  echo missing\nfi\n")
    out = _bash_ok(script, {"PATH": f"{bindir}:/usr/bin:/bin"})
    assert out.stdout.strip() == "installed"


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
    assert "if (Test-WinSWTrusted) {\n        Invoke-WinSW uninstall" in ps1
    assert "sc.exe delete $ServiceName" in ps1
    assert ps1.index("$WinSWHashes = @{") < ps1.index("Invoke-WinSW uninstall")
    assert "Remove-Item -Force $LegacyNssm" in ps1


# install-service.ps1's quiet native-command wrapper: Continue around the call,
# stderr merged and dropped, the old preference restored in a finally.
_NATIVE_QUIET = re.compile(
    r'function Invoke-NativeQuiet \{\n\s*\$exe, \$rest = \$args\n'
    r'\s*\$oldPref = \$ErrorActionPreference\n\s*\$ErrorActionPreference = "Continue"\n'
    r'\s*try \{\n\s*& \$exe @rest 2>&1 \| Out-Null\n'
    r'\s*\} finally \{\n\s*\$ErrorActionPreference = \$oldPref')


def test_pip_cache_purge_runs_under_a_relaxed_error_preference():
    # Under $ErrorActionPreference=Stop (and the UAC relaunch's Windows
    # PowerShell 5.1) a native command's redirected stderr line terminates
    # the script: a disabled or empty pip cache aborted the venv bootstrap.
    ps1 = _read("install-service.ps1")
    assert _NATIVE_QUIET.search(ps1)
    assert "\n    Invoke-NativeQuiet $Python -m pip cache purge\n" in ps1


def test_preflight_winsw_uninstall_runs_under_a_relaxed_error_preference():
    # WinSW writes status to stderr, which Windows PowerShell 5.1 turns into a
    # terminating error under Stop: a re-install aborted after stopping the
    # old service, before reinstalling it. Every WinSW call must go through
    # Invoke-WinSW, defined before the pre-flight removal that calls it.
    ps1 = _read("install-service.ps1")
    assert re.findall(r"& \$WinSWExe\b.*", ps1) == ["& $WinSWExe @args"]
    assert ps1.index("function Invoke-WinSW") < ps1.index("Invoke-WinSW uninstall")


def test_orphan_sweep_spares_sibling_checkouts():
    # A bare StartsWith($RepoDir) also matched C:\svc\fwb-old\... and force-
    # killed another install's python/WinSW; match the dir plus a separator.
    ps1 = _read("install-service.ps1")
    assert "$repoPrefix = $RepoDir.TrimEnd('\\') + '\\'" in ps1
    assert "$_.Path.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)" in ps1
    assert "StartsWith($RepoDir" not in ps1


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


def test_winsw_arguments_quote_the_main_py_path():
    # WinSW v2 hands <arguments> to the child as a raw command line: an
    # unquoted %BASE%\main.py under "C:\Users\John Doe" split into two argv
    # entries and the service crash-looped.
    import xml.etree.ElementTree as ET
    s = _read("install-service.ps1")
    x = re.search(r'\$xml = @"\n(.*?)\n"@', s, re.S).group(1)
    x = re.sub(r"\$\([^)]*\)", "X", x)
    x = re.sub(r"\$[A-Za-z_]\w*", "X", x)
    assert ET.fromstring(x).findtext("arguments") == '"%BASE%\\main.py"'


def test_env_example_lines_uncomment_to_their_documented_value():
    # The recipe is "uncomment the line"; python-dotenv only strips an inline
    # comment that follows a value, so `#WHISPER_OUTPUT_PREFIX=   # ...`
    # uncommented to the prefix "# prepended to final text".
    from dotenv import dotenv_values
    text = re.sub(r"^#([A-Z][A-Z0-9_]*=)", r"\1", _read(".env.example"), flags=re.M)
    values = dotenv_values(stream=io.StringIO(text), interpolate=False)
    assert len(values) > 100               # the lines really were uncommented
    bad = {k: v for k, v in values.items() if (v or "").lstrip().startswith("#")}
    assert not bad


def test_time_ledger_heatmap_adds_up_to_the_session_hours(tmp_path, monkeypatch):
    # The heatmap loop included its end point: every session got one extra
    # 15-min step, so a lone commit (0.5 h with the default ramp) showed 0.75 h.
    import importlib.util
    import json
    from datetime import datetime
    spec = importlib.util.spec_from_file_location(
        "time_ledger", os.path.join(REPO, "scripts", "time_ledger.py"))
    tl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tl)
    t0 = datetime(2026, 1, 5, 10, 0, tzinfo=tl.TZ)
    commit = dict(app="backend", t=t0, kind="human", group="Fixes")
    monkeypatch.setattr(tl, "read_commits", lambda app, repo: [commit])
    monkeypatch.setattr(tl, "prompt_times", lambda repos: [])
    template = tmp_path / "t.html"
    template.write_text("/*DATA*/null", encoding="utf-8")
    monkeypatch.setattr(tl, "TEMPLATE", template)
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["time_ledger", "--frontend", str(tmp_path),
                                      "--out", str(out)])
    assert tl.main() == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert [s["hours"] for s in data["sessions"]] == [0.5]
    assert sum(h["hours"] for h in data["heat"]) == 0.5


def test_dockerignore_excludes_repo_local_ffmpeg_tree():
    # install-service.ps1 -Full extracts ~150 MB of Windows ffmpeg DLLs into
    # ./ffmpeg/ (gitignored); a local docker build must not ship them.
    assert re.search(r"^ffmpeg/$", _read(".dockerignore"), re.M)


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


def test_installers_require_python_314():
    # CI tests 3.14 only; both installers refuse older interpreters before
    # building a venv on them, and say so.
    sh = _read("install-service.sh")
    assert "sys.version_info >= (3, 14)" in sh
    ps1 = _read("install-service.ps1")
    assert "sys.version_info >= (3, 14)" in ps1
    assert "3.12" not in sh and "3.12" not in ps1
    assert "Python 3.10" not in ps1


def _venv_script(as_user):
    """install-service.sh's venv bootstrap, up to the service stop, under strict mode with the given as_user stub."""
    sh = _read("install-service.sh")
    return f"set -euo pipefail\n{as_user}\n" + sh[sh.index('VENV="$REPO_DIR/venv"'):sh.index("\n# Stop a running service")]


@pytest.mark.skipif(sys.platform == "win32", reason="bash installer")
def test_linux_installer_finds_a_python314_beside_an_older_python3(tmp_path):
    # Debian 13 / Ubuntu 24.04 keep python3 older than 3.14 (apt needs it) and
    # ship 3.14 beside it as python3.14: a fresh clone must build its venv on
    # that, or on PYTHON=, never tell the operator to replace python3. Run the
    # script's own venv block against stub interpreters only.
    bindir = tmp_path / "bin"
    bindir.mkdir()

    def stub(name, version):
        ok = 0 if version >= (3, 14) else 1
        return _exe(bindir / name, f'#!/bin/sh\n[ "$1" = --version ] && echo "Python {version[0]}.{version[1]}.0"\n'
                    f"exit {ok}\n")

    def run(**env):
        out = _bash(_venv_script('as_user() { echo "ran: $*"; }'),
                    {"PATH": str(bindir), "REPO_DIR": str(tmp_path), **env})
        return out.returncode, out.stdout, out.stderr
    stub("python3", (3, 13))
    py314 = stub("python3.14", (3, 14))
    rc, stdout, stderr = run()
    assert rc == 0, stderr
    assert f"ran: {py314} -m venv {tmp_path}/venv" in stdout
    # An explicit PYTHON wins over every name on PATH, and is checked too.
    own = stub("own-python", (3, 14))
    rc, stdout, _ = run(PYTHON=str(own))
    assert f"ran: {own} -m venv" in stdout
    rc, stdout, stderr = run(PYTHON=str(bindir / "python3"))
    assert rc == 1 and "ran:" not in stdout
    assert f"PYTHON={bindir}/python3 is not a Python 3.14+" in stderr
    # Only the older python3: refused, without the replace-python3 advice.
    py314.unlink()
    rc, stdout, stderr = run()
    assert rc == 1 and "ran:" not in stdout
    assert "Python 3.13.0" in stderr and "beside the system python3" in stderr
    assert "3.14+ as python3" not in stderr
    # A python3 that is 3.14+ itself still builds the venv.
    stub("python3", (3, 14))
    rc, stdout, stderr = run()
    assert rc == 0, stderr
    assert f"ran: {bindir}/python3 -m venv" in stdout


@pytest.mark.skipif(sys.platform == "win32", reason="bash installer")
def test_linux_installer_removes_a_venv_whose_build_failed(tmp_path):
    # Without python3.14-venv, `python3.14 -m venv` writes bin/python and then
    # fails in ensurepip. A re-run after installing the package must build the
    # venv again, not keep the pip-less one because bin/python is executable.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    py314 = _exe(bindir / "python3.14", "#!/bin/sh\nexit 0\n")
    venv = tmp_path / "venv"
    no_ensurepip = ('as_user() { mkdir -p "$VENV/bin"; printf "#!/bin/sh\\n" > "$PY"; '
                    'chmod +x "$PY"; echo "ensurepip is not available" >&2; return 1; }')

    def run(as_user):
        out = _bash(_venv_script(as_user),
                    {"PATH": f"{bindir}:/usr/bin:/bin", "REPO_DIR": str(tmp_path)})
        return out.returncode, out.stdout, out.stderr
    rc, _, stderr = run(no_ensurepip)
    assert rc == 1 and not venv.exists()
    assert "python3.14-venv" in stderr
    rc, stdout, stderr = run('as_user() { echo "ran: $*"; }')
    assert rc == 0, stderr
    assert f"ran: {py314} -m venv {venv}" in stdout
    # A venv dir that was there before the failed build is not ours to delete.
    (venv / "keep").mkdir(parents=True)
    rc, _, _ = run(no_ensurepip)
    assert rc == 1 and (venv / "keep").is_dir()


@pytest.mark.skipif(sys.platform == "win32", reason="bash installer")
def test_linux_installer_rebuilds_a_venv_whose_python_is_dead_or_pipless(tmp_path):
    # A venv whose base interpreter was removed has a dangling bin/python, and
    # an older installer's failed build left an executable but pip-less one.
    # Both die at `$PY -m pip` on every re-run unless rebuilt, and with --clear:
    # `python -m venv` keeps an existing bin/python link as it is.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    py314 = _exe(bindir / "python3.14", "#!/bin/sh\nexit 0\n")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    py = venv / "bin" / "python"

    def run():
        # Echo every as_user step; run only the pip probe, against the stub.
        return _bash_ok(
            _venv_script('as_user() { echo "ran: $*"; [ "$1" != "$PY" ] || "$@"; }'),
            {"PATH": f"{bindir}:/usr/bin:/bin", "REPO_DIR": str(tmp_path)}).stdout
    py.symlink_to(tmp_path / "gone" / "python3.12")
    assert f"ran: {py314} -m venv --clear {venv}" in run()
    py.unlink()
    _exe(py, "#!/bin/sh\necho 'No module named pip' >&2\nexit 1\n")
    assert f"ran: {py314} -m venv --clear {venv}" in run()
    # A working venv is kept, whatever Python it runs.
    py.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    assert "-m venv" not in run()


def test_windows_installer_rebuilds_a_venv_whose_python_is_dead_or_pipless():
    # The Windows twin of the test above: venv\Scripts\python.exe is a launcher
    # that outlives an uninstalled base Python, and a failed build leaves it
    # without pip, so a bare Test-Path kept a venv that died at "pip install" on
    # every re-run. The pip probe runs under a relaxed error preference (its
    # stderr would terminate under Stop), and the rebuild clears an old venv.
    ps1 = _read("install-service.ps1")
    probe = ps1[ps1.index("function Test-VenvUsable"):ps1.index("\nif (-not (Test-VenvUsable))")]
    assert _NATIVE_QUIET.search(ps1)
    assert re.search(r'\s*Invoke-NativeQuiet \$Python -m pip --version\n'
                     r'\s*return \(\$LASTEXITCODE -eq 0\)\n', probe)
    assert "if (-not (Test-Path $Python)) { return $false }" in probe
    assert "if (-not (Test-Path $Python)) {\n" not in ps1
    bootstrap = ps1[ps1.index("\nif (-not (Test-VenvUsable))"):ps1.index('Write-Host "venv ready."')]
    assert 'if (Test-Path (Join-Path $venvDir "pyvenv.cfg")) { $venvFlags = @("--clear") }' in bootstrap
    assert "& $sysPy -m venv @venvFlags $venvDir" in bootstrap


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


def _env_funcs(sh):
    """install-service.sh's env_value + env_sets, on their own."""
    return "\n".join(re.search(rf"^{n}\(\) \{{\n.*?^\}}\n", sh, re.S | re.M).group(0)
                     for n in ("env_value", "env_sets"))


def _unit_decisions(sh):
    """_env_funcs plus the unit-line decisions and the "cannot expand" warning loop."""
    return _env_funcs(sh) + sh[sh.index("\n}\n", sh.index("env_sets() {")) + 3:
                               sh.index("\n# systemd splits")]


def _env_sets_runner(tmp_path):
    """install-service.sh's env_value/env_sets, sourced on their own and run
    against tmp_path/.env with the test venv's python (has python-dotenv)."""
    (tmp_path / "f.sh").write_text(_env_funcs(_read("install-service.sh")), encoding="utf-8")

    def run(line: str) -> bool:
        (tmp_path / ".env").write_text(line + "\n", encoding="utf-8")
        return subprocess.run(
            ["bash", "-c", f'source "{tmp_path}/f.sh"; env_sets X'],
            env={"REPO_DIR": str(tmp_path), "PY": sys.executable,
                 "PATH": "/usr/bin:/bin"}).returncode == 0
    return run


@pytest.mark.skipif(sys.platform == "win32", reason="bash installer")
def test_linux_env_sets_reads_dotenv_like_the_server(tmp_path):
    # A .env pin must win over the unit's Environment= line, so env_sets has
    # to agree with python-dotenv: a spaced key counts; an empty value (config.py
    # falls back to /data) or an unexpanded $PWD does not — the unit pin stays.
    run = _env_sets_runner(tmp_path)
    for line in ("X=/a", "export X=/a", "X = /a"):
        assert run(line), line
    for line in ("#X=/a", "X=", 'X=""', "X=$PWD/data", "X=${PWD}/data", "X_Y=/a"):
        assert not run(line), line


@pytest.mark.skipif(sys.platform == "win32", reason="bash installer")
def test_linux_env_sets_counts_the_vars_systemd_sets_for_the_unit(tmp_path):
    # load_dotenv expands ${VAR} from the service's environment, and systemd
    # sets HOME/USER/LOGNAME/SHELL for a User= unit: WHISPER_DATA_DIR=${HOME}/x
    # is a real pin, which the unit's Environment= line used to override.
    run = _env_sets_runner(tmp_path)
    for line in ("X=${HOME}/whisper-data", "X=/srv/${USER}", "X=${LOGNAME}",
                 "X=${HOME}/${USER}/db"):
        assert run(line), line
    for line in ("X=$HOME/data", "X=${HOME}/${PWD}", "X=${HOMEDIR}/data"):
        assert not run(line), line


@pytest.mark.skipif(sys.platform == "win32", reason="bash installer")
def test_linux_env_sets_counts_keys_an_earlier_dotenv_line_pins(tmp_path):
    # load_dotenv also expands ${K} from a key K set earlier in the same .env:
    # WHISPER_MODELS_DIR=${WHISPER_DATA_DIR}/models is a real pin, which the
    # unit's Environment= line used to override.
    run = _env_sets_runner(tmp_path)
    for line in ("R=/srv/w\nX=${R}/models", "R=${HOME}/w\nX=${R}/models",
                 "R=/srv\nS=${R}/w\nX=${S}/${USER}"):
        assert run(line), line
    # A later line, or an earlier one that does not resolve itself, does not.
    for line in ("X=${R}/models\nR=/srv/w", "R=$PWD\nX=${R}/models",
                 "R=${PWD}\nX=${R}/models", "R=\nX=${R}/models"):
        assert not run(line), line



@pytest.mark.skipif(sys.platform == "win32", reason="bash installer")
def test_linux_env_sets_counts_a_dotenv_default_reference(tmp_path):
    # load_dotenv also expands ${NAME:-default}: to NAME's value, else to the
    # literal default. WHISPER_DATA_DIR=${XDG_DATA_HOME:-/srv/whisper} is a
    # real pin, which the unit's Environment= line used to override.
    run = _env_sets_runner(tmp_path)
    for line in ("X=${XDG_DATA_HOME:-/srv/whisper}", "X=${HOME:-/home/u}/w",
                 "R=/srv\nX=${R:-/x}/m", "X=${HOME:-}/w", "X=${NOPE:-}/srv"):
        assert run(line), line
    # An empty default, or one dotenv keeps literally with its `$`, does not.
    for line in ("X=${NOPE:-}", "X=${NOPE:-$PWD}/data", "X=${PWD:-$HOME}"):
        assert not run(line), line


@pytest.mark.skipif(sys.platform == "win32", reason="bash installer")
def test_linux_env_sets_lets_an_earlier_unpinned_key_beat_its_default(tmp_path):
    # dotenv takes ${R:-default}'s default only when R is absent: an earlier
    # empty, bare or unexpandable R wins, so load_dotenv yields "" or "/m"
    # (crash-loop on /data) — the unit's Environment= line has to stay.
    run = _env_sets_runner(tmp_path)
    for line in ("R=\nX=${R:-/srv}", 'R=""\nX=${R:-/srv}', "R\nX=${R:-/srv}",
                 "R=${PWD}\nX=${R:-/srv}/m", "R=$PWD\nX=${R:-/srv}"):
        assert not run(line), line
    # A NAME no earlier line names still takes the default.
    assert run("X=${R:-/srv}\nR=")

@pytest.mark.skipif(sys.platform == "win32", reason="bash installer")
def test_linux_installer_expands_a_dotenv_reference_to_a_unit_set_var(tmp_path):
    # The unit's own WHISPER_DATA_DIR (written when .env leaves it unpinned) is
    # in the service's real environment, which load_dotenv expands ahead of the
    # .env: WHISPER_MODELS_DIR=${WHISPER_DATA_DIR}/models resolves to
    # <repo>/data/models. The installer warned it "cannot expand" and wrote a
    # WHISPER_MODELS_DIR=<repo>/models line that overrode it.
    script = (_unit_decisions(_read("install-service.sh"))
              + 'echo "D=$DATA_ENV_LINE"; echo "M=$MODELS_ENV_LINE"; echo "L=$LOG_ENV_LINE"\n')

    def run(dotenv: str) -> tuple[dict, str]:
        (tmp_path / ".env").write_text(dotenv, encoding="utf-8")
        out = _bash_ok(script, {"REPO_DIR": str(tmp_path), "PY": sys.executable,
                                "PATH": "/usr/bin:/bin"})
        return dict(ln.split("=", 1) for ln in out.stdout.splitlines()), out.stderr
    data_line = f"Environment=WHISPER_DATA_DIR={tmp_path}/data"
    for dotenv in ("WHISPER_MODELS_DIR=${WHISPER_DATA_DIR}/models\n"
                   "WHISPER_LOG_FILE=${WHISPER_DATA_DIR}/l.log\n",
                   "WHISPER_DATA_DIR=\n"
                   "WHISPER_MODELS_DIR=${WHISPER_DATA_DIR:-/srv/w}/models\n"
                   "WHISPER_LOG_FILE=${WHISPER_DATA_DIR:-/srv/w}/l.log\n"):
        lines, err = run(dotenv)
        assert lines == {"D": data_line, "M": "", "L": ""}, dotenv
        assert "cannot expand" not in err, dotenv
    # A .env-pinned data root puts nothing in the environment: a reference to
    # it from an earlier line still does not expand, and the unit keeps its pin.
    lines, err = run("WHISPER_MODELS_DIR=${WHISPER_DATA_DIR}/models\n"
                     "WHISPER_DATA_DIR=/srv/w\n")
    assert lines == {"D": "", "M": f"Environment=WHISPER_MODELS_DIR={tmp_path}/models",
                     "L": ""}
    assert "WHISPER_MODELS_DIR to a value with a $ the service cannot expand" in err


def test_windows_installer_gates_the_log_file_env_on_the_dotenv():
    # WinSW <env> is the real environment and load_dotenv never overrides it:
    # a .env WHISPER_LOG_FILE pin was silently ignored on Windows.
    ps1 = _read("install-service.ps1")
    assert not re.search(r'^\s*<env name="WHISPER_LOG_FILE"', ps1, re.M)
    assert 'Test-EnvSets "WHISPER_LOG_FILE"' in ps1
    assert ps1.index("function Test-EnvSets") < ps1.index('$xml = @"')
    assert ps1.index('Test-EnvSets "WHISPER_LOG_FILE"') < ps1.index(
        "Set-Content -Path $WinSWXml")
    assert r'-Pattern "^\s*(export\s+)?$name\s*=\s*[^\s#]"' in ps1


def test_installers_leave_a_store_with_its_own_pin_alone():
    # A store pinned by WHISPER_<NAME>_DB is read from there: the legacy
    # "move it to data/db" recipe would strand the live store.
    sh = _read("install-service.sh")
    assert 'env_sets "$(store_env "$(basename "$legacy")")" "$UNIT_NAMES" && continue' in sh
    ps1 = _read("install-service.ps1")
    assert "if (Test-EnvSets $storeEnv) { continue }" in ps1


def test_windows_legacy_store_recipe_creates_the_db_dir_before_the_move():
    # Move-Item onto a missing folder renames the store to a FILE named db and
    # the -shm/-wal matches overwrite it: the recipe creates the folder first,
    # like the Linux recipe's mkdir -p.
    ps1 = _read("install-service.ps1")
    mkdir = ps1.index("\"New-Item -ItemType Directory -Force '$dbDir' | Out-Null; \"")
    move = ps1.index("\"Move-Item '$($legacy.FullName)*' '$dbDir' -Force;")
    assert mkdir < move


def test_windows_shared_ffmpeg_is_copied_out_of_temp_not_moved():
    # Windows PowerShell 5.1 cannot Move-Item a directory across volumes: with
    # TEMP on C: and the checkout on D:, -Full aborted after the pre-flight had
    # unregistered the service. The copy must follow the old dir's removal, or
    # it nests as ffmpeg\ffmpeg-<ver>\ and bin\ffmpeg.exe is never found.
    ps1 = _read("install-service.ps1")
    assert "Move-Item $inner.FullName" not in ps1
    remove = ps1.index("if (Test-Path $RepoFfmpegDir) { Remove-Item -Recurse -Force $RepoFfmpegDir }")
    copy = ps1.index("Copy-Item -Recurse $inner.FullName $RepoFfmpegDir")
    cleanup = ps1.index("Remove-Item -Recurse -Force $extractDir\n")
    assert remove < copy < cleanup


def test_windows_legacy_config_recipe_creates_the_data_dir_before_the_move():
    # data/ is gitignored and the installer never creates it: Move-Item onto a
    # missing folder renames config.local.json to a FILE named data, and the
    # next start cannot create data\db under it.
    ps1 = _read("install-service.ps1")
    mkdir = ps1.index("\"New-Item -ItemType Directory -Force '$dataDir' | Out-Null; \"")
    move = ps1.index("\"Move-Item '$legacyConfig' '$dataDir', \"")
    assert mkdir < move


@pytest.mark.skipif(sys.platform == "win32", reason="bash installer")
def test_linux_installer_legacy_checks_expand_a_unit_set_var(tmp_path):
    # .env WHISPER_DB_DIR=${WHISPER_DATA_DIR}/stores resolves through the
    # unit's own WHISPER_DATA_DIR, so the service never reads data/db: the
    # legacy checks warned "IGNORED ... Move it" into the wrong place (and for
    # a per-store pin, told the operator to move the live store).
    section = _legacy_section(_read("install-service.sh"))
    (tmp_path / "api_keys.local.sqlite3").write_text("", encoding="utf-8")
    (tmp_path / "config.local.json").write_text("{}", encoding="utf-8")

    def run(dotenv: str) -> str:
        (tmp_path / ".env").write_text(dotenv, encoding="utf-8")
        return _bash_ok(section, {"REPO_DIR": str(tmp_path), "PY": sys.executable,
                                  "SERVICE_NAME": "w", "RUN_USER": "u",
                                  "PATH": "/usr/bin:/bin"}).stderr
    for dotenv in ("WHISPER_DB_DIR=${WHISPER_DATA_DIR}/stores\n"
                   "WHISPER_CONFIG_LOCAL=${WHISPER_DATA_DIR}/c.json\n",
                   "WHISPER_API_KEYS_DB=${WHISPER_DATA_DIR}/../api_keys.local.sqlite3\n"
                   "WHISPER_CONFIG_LOCAL=${WHISPER_DATA_DIR}/c.json\n"):
        assert "IGNORED" not in run(dotenv), dotenv
    # Unpinned: both legacy files still warn.
    assert run("").count("IGNORED") == 2


def _legacy_section(sh):
    return _unit_decisions(sh) + sh[sh.index("\nwarn_legacy() {"):sh.index('\necho "Enabling')]


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None,
                    reason="bash installer")
def test_linux_legacy_captures_recipe_moves_the_audio_tree_with_its_store(tmp_path):
    # Captures rows resolve audio_relpath against CAPTURES_DIR: a recipe that
    # moved only captures.local.sqlite3 left every row pointing at audio that
    # is not in data/captures. Run the printed recipe for real (systemctl and
    # sudo stubbed) against a repo-root layout plus the empty store and
    # groups/ dir a start without it created.
    section = _legacy_section(_read("install-service.sh"))
    repo, stubs = tmp_path / "repo", tmp_path / "stubs"
    stubs.mkdir()
    _exe(stubs / "systemctl", f'#!/bin/sh\necho "$*" >> "{tmp_path}/systemctl.log"\n')
    _exe(stubs / "sudo", '#!/bin/sh\nshift 2\nexec "$@"\n')
    env = {"REPO_DIR": str(repo), "PY": sys.executable, "SERVICE_NAME": "w",
           "RUN_USER": "u", "PATH": f"{stubs}:/usr/bin:/bin"}

    def layout(conflict=False):
        shutil.rmtree(repo, ignore_errors=True)
        (repo / "captures" / "ab" / "cd").mkdir(parents=True)
        (repo / "captures" / "groups").mkdir()
        (repo / "captures" / "ab" / "cd" / "abcd.wav").write_text("a", encoding="utf-8")
        (repo / "captures" / "groups" / "g.wav").write_text("g", encoding="utf-8")
        (repo / "captures.local.sqlite3").write_text("rows", encoding="utf-8")
        (repo / "captures.local.sqlite3-wal").write_text("wal", encoding="utf-8")
        (repo / "data" / "db").mkdir(parents=True)
        (repo / "data" / "db" / "captures.local.sqlite3").write_text("", encoding="utf-8")
        (repo / "data" / "captures" / "groups").mkdir(parents=True)
        if conflict:
            (repo / "data" / "captures" / "ef" / "01").mkdir(parents=True)
            (repo / "data" / "captures" / "ef" / "01" / "ef01.wav").write_text(
                "new", encoding="utf-8")
        (tmp_path / "systemctl.log").unlink(missing_ok=True)

    def warnings(dotenv=""):
        (repo / ".env").write_text(dotenv, encoding="utf-8")
        out = _bash_ok(section, env)
        return re.sub(r"\x1b\[[0-9;]*m", "", out.stderr)

    def recipe(warning, end):
        return warning[warning.index("Move it: ") + 9:warning.index(end)]

    layout()
    _bash_ok(recipe(warnings(), " (the rm drops"), env)
    assert (repo / "data" / "captures" / "ab" / "cd" / "abcd.wav").read_text(
        encoding="utf-8") == "a"
    assert (repo / "data" / "captures" / "groups" / "g.wav").exists()
    assert not (repo / "captures").exists()
    assert (repo / "data" / "db" / "captures.local.sqlite3").read_text(
        encoding="utf-8") == "rows"
    assert (repo / "data" / "db" / "captures.local.sqlite3-wal").exists()
    assert (tmp_path / "systemctl.log").read_text(encoding="utf-8") == "stop w\nstart w\n"
    # Idempotent: the next installer run has nothing left to warn about.
    assert "IGNORED" not in warnings()

    # Audio recorded into data/captures since: never overwritten, and the
    # chain stops before the store moves (audio without its rows would be
    # unlinked as orphans by the next start).
    layout(conflict=True)
    out = _bash(recipe(warnings(), " (the rm drops"), env)
    assert out.returncode != 0 and "Directory not empty" in out.stderr
    assert (repo / "captures" / "ab" / "cd" / "abcd.wav").exists()
    assert (repo / "captures.local.sqlite3").read_text(encoding="utf-8") == "rows"
    assert (repo / "data" / "captures" / "ef" / "01" / "ef01.wav").read_text(
        encoding="utf-8") == "new"
    assert (tmp_path / "systemctl.log").read_text(encoding="utf-8") == "stop w\n"

    # The store moved already (the older recipe): the tree on its own.
    layout()
    for f in ("captures.local.sqlite3", "captures.local.sqlite3-wal"):
        (repo / f).unlink()
    warning = warnings()
    assert "legacy captures audio" in warning
    _bash_ok(recipe(warning, " (mv -T stops"), env)
    assert (repo / "data" / "captures" / "ab" / "cd" / "abcd.wav").exists()
    assert "IGNORED" not in warnings()

    # A .env CAPTURES_DIR pin reads the tree where it is: leave it be.
    layout()
    warning = warnings(f"WHISPER_CAPTURES_DIR={repo}/captures\n")
    assert warning.count("IGNORED") == 1 and "mv -T" not in warning


def test_windows_legacy_captures_recipe_moves_the_audio_tree_with_its_store():
    # Same recipe as the Linux one (no PowerShell under test, so pin its
    # shape): the audio tree is part of the repo-root captures store's move,
    # goes BEFORE the store, and a target holding any file throws before
    # anything moves; the tree also warns on its own once the store moved.
    ps1 = _read("install-service.ps1")
    block = ps1[ps1.index("# --- upgrade check"):ps1.index("\nInvoke-WinSW start")]
    assert 'if (-not (Test-EnvSets "WHISPER_CAPTURES_DIR")' in block
    guard = block.index("{ throw '$audioDir already holds audio' }")
    clear = block.index("{ Remove-Item -LiteralPath '$audioDir' ")
    parent = block.index("\"New-Item -ItemType Directory -Force '$dataDir' | Out-Null; \"")
    move = block.index("\"Move-Item -LiteralPath '$legacyAudio' '$audioDir' -ErrorAction Stop; \"")
    assert guard < clear < parent < move
    assert 'if ($legacy.FullName -eq (Join-Path $RepoDir "captures.local.sqlite3"))' in block
    recipe = block.index('"(it reads $dbDir). Move it: Stop-Service $ServiceName; $audio" +')
    assert recipe < block.index("\"Move-Item '$($legacy.FullName)*' '$dbDir' -Force;")
    assert '"(it reads $audioDir). Move it: Stop-Service $ServiceName; $audioMove" +' in block
