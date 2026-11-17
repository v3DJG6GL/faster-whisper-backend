"""Keep the test venv in step with requirements.txt + requirements-dev.txt.

A venv is installed once and then drifts: a floor raised in a requirements
file never reaches it on its own, so local runs test older packages than CI
does and miss what CI then catches (imageio-ffmpeg 0.6.0 under a 0.8.1 floor
hid an ffmpeg behaviour change until CI, 2026-11). tests/conftest.py calls
ensure() before every run: when an installed version misses a requirement it
installs the two files into the running venv and restarts the run.
`python scripts/sync_venv.py` does the same by hand.

Only the direct requirements (and the packages their requested extras pull
in) are compared, which needs no network; when they miss, the sync upgrades
everything, transitive packages included.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from importlib import metadata

from packaging.requirements import InvalidRequirement, Requirement

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ("requirements.txt", "requirements-dev.txt")
# Set on the restarted run, so a sync that does not fix the venv ends the
# run with the report instead of looping.
SYNCED_ENV = "FWB_VENV_SYNCED"
# Opt-out for a deliberate test against other versions: report, don't sync.
OPT_OUT_ENV = "FWB_NO_VENV_SYNC"


def requirements(files=FILES, root=REPO_ROOT) -> list[Requirement]:
    """The requirements of `files`, following `-r` includes."""
    out: list[Requirement] = []
    seen: set[str] = set()

    def read(path: str) -> None:
        path = os.path.normpath(path)
        if path in seen:
            return
        seen.add(path)
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.split(" #", 1)[0].strip()
                if not line or line.startswith("#"):
                    continue
                for flag in ("-r ", "--requirement "):
                    if line.startswith(flag):
                        read(os.path.join(os.path.dirname(path), line[len(flag):].strip()))
                        break
                else:
                    if line.startswith("-"):  # index / install options
                        continue
                    try:
                        out.append(Requirement(line))
                    except InvalidRequirement:
                        continue

    for name in files:
        read(os.path.join(root, name))
    return out


def _miss(req: Requirement, extra: str = "") -> str | None:
    marker_env = {"extra": extra} if extra else None
    if req.marker and not req.marker.evaluate(marker_env):
        return None
    try:
        version = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        return f"{req.name}: not installed (wants {req.specifier or 'any version'})"
    if req.specifier and not req.specifier.contains(version, prereleases=True):
        return f"{req.name} {version}: wants {req.specifier}"
    return None


def stale(files=FILES, root=REPO_ROOT) -> list[str]:
    """One line per requirement the running interpreter does not satisfy."""
    out: list[str] = []
    for req in requirements(files, root):
        miss = _miss(req)
        if miss:
            out.append(miss)
            continue
        for extra in sorted(req.extras):
            for dep in metadata.requires(req.name) or ():
                try:
                    dep_req = Requirement(dep)
                except InvalidRequirement:
                    continue
                if dep_req.marker and "extra" in str(dep_req.marker):
                    miss = _miss(dep_req, extra)
                    if miss:
                        out.append(f"{miss} (via {req.name}[{extra}])")
    return out


def install_cmd(files=FILES) -> list[str]:
    """uv when it is on PATH (what CI uses), else the venv's own pip.

    --upgrade: a sync also lifts the transitive packages to what a fresh CI
    install gets, not only the ones whose floor moved.
    """
    args = ["--upgrade", *(a for f in files for a in ("-r", f))]
    uv = shutil.which("uv")
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, *args]
    return [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *args]


def _in_ci() -> bool:
    return any(os.environ.get(k) for k in ("CI", "GITHUB_ACTIONS", "FORGEJO_ACTIONS"))


def ensure() -> str | None:
    """Sync and restart when the venv is stale; return an error to end the run on.

    None means: go on (nothing stale, or the opt-out is set). In CI, outside
    a venv, or when the restarted run is still stale, nothing is installed and
    the report comes back as the error.
    """
    missing = stale()
    if not missing:
        return None
    cmd = install_cmd()
    report = ("The venv does not match the requirements files:\n  "
              + "\n  ".join(missing)
              + f"\nInstall them with:\n  {shlex.join(cmd)}")
    if os.environ.get(OPT_OUT_ENV):
        print(f"{report}\n({OPT_OUT_ENV} is set: running anyway.)", file=sys.stderr)
        return None
    if _in_ci() or sys.prefix == sys.base_prefix or os.environ.get(SYNCED_ENV):
        return report
    print(f"{report}\nInstalling now, then restarting the run.", file=sys.stderr, flush=True)
    if subprocess.call(cmd, cwd=REPO_ROOT) != 0:
        return report + "\n(The install failed; its output is above.)"
    os.environ[SYNCED_ENV] = "1"
    if sys.platform == "win32":  # no real exec: a child would outlive the console's view
        return "The venv was synced; run the command again."
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, sys.orig_argv)
    return None  # unreachable; os.execv does not return


def main() -> int:
    missing = stale()
    if not missing:
        print("The venv matches the requirements files.")
        return 0
    print("Stale:\n  " + "\n  ".join(missing))
    cmd = install_cmd()
    print(f"Running: {shlex.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=REPO_ROOT)
    if rc:
        return rc
    # The new versions are installed but this process still has the old
    # metadata cached; check from a fresh interpreter.
    return subprocess.call([sys.executable, "-c",
                            "import sys; sys.path.insert(0, sys.argv[1]); import sync_venv;"
                            " m = sync_venv.stale(); print('Still stale:', m) if m else None;"
                            " sys.exit(1 if m else 0)",
                            os.path.dirname(os.path.abspath(__file__))])


if __name__ == "__main__":
    sys.exit(main())
