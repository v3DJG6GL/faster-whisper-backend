#!/usr/bin/env python3
"""Check that every pip command the installers run can still find its packages.

A requirements bump is only safe when the new version exists for EVERY platform
the project installs on. Renovate reads the newest version off the index and
cannot see which platforms it was built for: nvidia-cublas-cu12 12.9.3.13 was
published for Linux only, the floor was raised to it, and the next
install-service.ps1 run on Windows died at "No matching distribution found".
Nothing in CI had installed requirements-gpu.txt on Windows.

So this resolves each install set below for each platform with uv's resolver
(`uv pip compile --python-platform`), which reads package metadata only: no
wheel is downloaded and nothing is installed, so it also covers the multi-GB
GPU sets, and Windows is checked from the Linux CI leg. Wheels only: a package
that would have to be compiled on the target counts as not installable, apart
from the EXCLUDED ones the installers handle themselves.

    python scripts/check_requirements_resolve.py            # every set, every platform
    python scripts/check_requirements_resolve.py --list     # print the uv commands

When it fails for a version that is simply not out yet for one platform, keep
the old floor and add the version to that package's allowedVersions rule in
renovate.json, or the grouped "python deps" PR stays red and holds back every
other bump. tests/test_ci_workflow_config.py keeps INSTALL_SETS in step with
install-service.ps1, install-service.sh and the Dockerfiles.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parent.parent

# The interpreter the images ship and the installers require.
PYTHON = "3.14"

# uv --python-platform values. Linux is glibc 2.39 (Ubuntu 24.04); the images'
# Debian 13 base is newer. An older floor would refuse the llama-cpp-python
# CUDA wheels, which are built for manylinux_2_35.
PLATFORMS = {
    "windows": "x86_64-pc-windows-msvc",
    "linux": "x86_64-manylinux_2_39",
}

TORCH = "https://download.pytorch.org/whl"
LLAMA = "https://abetlen.github.io/llama-cpp-python/whl"


class InstallSet(NamedTuple):
    """One `pip install` command of an installer: what it is given, and where."""
    name: str
    files: tuple[str, ...] = ()
    specs: tuple[str, ...] = ()
    indexes: tuple[str, ...] = ()
    no_deps: bool = False
    platforms: tuple[str, ...] = ("windows", "linux")


# One entry per distinct pip command in install-service.ps1 / install-service.sh
# / Dockerfile / Dockerfile.gpu / ci.yml. Each is resolved on its own, as pip
# runs it: the GPU extras and requirements-gpu.txt are NOT satisfiable together
# (torch cu126 pins older cuBLAS / cuDNN, see Dockerfile.gpu), and the
# installers never ask pip for both at once.
INSTALL_SETS = (
    InstallSet("base", files=("requirements.txt",)),
    InstallSet("base + dev (CI)", files=("requirements.txt", "requirements-dev.txt")),
    InstallSet("base + gpu", files=("requirements.txt", "requirements-gpu.txt")),
    InstallSet("gpu alone (ps1 -Gpu)", files=("requirements-gpu.txt",)),
    InstallSet("heavy + gpu (Dockerfile.gpu)",
               files=("requirements-heavy.txt", "requirements-gpu.txt"), platforms=("linux",)),
    InstallSet("full extras, cpu (ps1)",
               files=("requirements-diarize.txt", "requirements-bgm.txt"), platforms=("windows",)),
    InstallSet("full extras, cpu (sh)",
               files=("requirements-diarize.txt", "requirements-bgm.txt"),
               indexes=(f"{TORCH}/cpu",), platforms=("linux",)),
    InstallSet("full extras, cpu (Dockerfile)",
               files=("requirements-diarize.txt", "requirements-bgm.txt",
                      "requirements-translate.txt"),
               indexes=(f"{TORCH}/cpu", f"{LLAMA}/cpu"), platforms=("linux",)),
    InstallSet("full extras, gpu",
               files=("requirements-diarize.txt", "requirements-bgm.txt"),
               specs=("audio-separator[gpu]",), indexes=(f"{TORCH}/cu126",)),
    InstallSet("onnxruntime-gpu pin", specs=("onnxruntime-gpu==1.26.*",), no_deps=True),
    InstallSet("translate, cpu", files=("requirements-translate.txt",), indexes=(f"{LLAMA}/cpu",)),
    InstallSet("translate, gpu", files=("requirements-translate.txt",),
               indexes=(f"{LLAMA}/cu124",)),
    InstallSet("convert", files=("requirements-convert.txt",)),
    InstallSet("convert, gpu (ps1)", files=("requirements-convert.txt",),
               indexes=(f"{TORCH}/cu126",), platforms=("windows",)),
)

# Packages with no wheel for PYTHON that the installers deal with themselves;
# left out of the resolution so the rest of the set is still checked.
EXCLUDED = {
    "windows": {
        "diffq-fixed": "install-service.ps1 installs a stub when no wheel exists",
    },
    "linux": {
        "diffq": "compiled from its sdist (install-service.sh and the Dockerfiles bring gcc)",
    },
}


def uv_command() -> list[str]:
    """`uv` from PATH, else the module of this interpreter."""
    exe = shutil.which("uv")
    if exe:
        return [exe]
    probe = subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True)
    if probe.returncode == 0:
        return [sys.executable, "-m", "uv"]
    sys.exit("check_requirements_resolve: uv not found (pip install uv)")


def resolve_args(install_set: InstallSet, platform: str, scratch: Path) -> list[str]:
    """The `uv pip compile` arguments that resolve one set for one platform."""
    args = ["pip", "compile", "--quiet", "--no-header", "--no-annotate",
            "--python-version", PYTHON, "--python-platform", PLATFORMS[platform],
            "--only-binary", ":all:",
            # pip's behaviour: the best version across PyPI and the extra indexes.
            "--index-strategy", "unsafe-best-match"]
    if install_set.no_deps:
        args.append("--no-deps")
    excluded = EXCLUDED.get(platform, {})
    if excluded:
        exclude_file = scratch / f"exclude-{platform}.txt"
        exclude_file.write_text("".join(f"{name}\n" for name in excluded), encoding="utf-8")
        args += ["--excludes", str(exclude_file)]
    for index in install_set.indexes:
        args += ["--extra-index-url", index]
    args += [str(REPO / name) for name in install_set.files]
    if install_set.specs:
        slug = "".join(c if c.isalnum() else "-" for c in install_set.name)
        spec_file = scratch / f"specs-{slug}.in"
        spec_file.write_text("".join(f"{spec}\n" for spec in install_set.specs), encoding="utf-8")
        args.append(str(spec_file))
    return args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true",
                        help="print the uv commands instead of running them")
    parser.add_argument("--jobs", type=int, default=4, help="resolutions run at once (default 4)")
    opts = parser.parse_args()

    pairs = [(s, p) for s in INSTALL_SETS for p in s.platforms]
    with tempfile.TemporaryDirectory(prefix="req-resolve-") as tmp:
        scratch = Path(tmp)
        if opts.list:
            for install_set, platform in pairs:
                print(f"# {install_set.name} [{platform}]")
                print("uv " + " ".join(resolve_args(install_set, platform, scratch)))
            return 0
        uv = uv_command()

        def run(pair):
            install_set, platform = pair
            return subprocess.run(uv + resolve_args(install_set, platform, scratch),
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                  text=True, encoding="utf-8", errors="replace")

        with ThreadPoolExecutor(max_workers=max(1, opts.jobs)) as pool:
            results = list(pool.map(run, pairs))

    failed = 0
    for (install_set, platform), done in zip(pairs, results):
        ok = done.returncode == 0
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'}  {platform:<8} {install_set.name}")
        if not ok:
            print("".join(f"        {line}\n" for line in done.stderr.strip().splitlines()), end="")
    if failed:
        print(f"\n{failed} of {len(pairs)} install sets cannot be resolved. A version without a "
              "build for one platform:\nkeep the old floor and exclude that version in "
              "renovate.json (allowedVersions).")
        return 1
    print(f"\nall {len(pairs)} install sets resolve for Python {PYTHON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
