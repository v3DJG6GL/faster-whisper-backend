# Install (or reinstall) the faster-whisper-backend as a Windows Service
# via WinSW (https://github.com/winsw/winsw, v2.12.0). Run from any
# PowerShell prompt:
#   .\install-service.ps1
#   .\install-service.ps1 -WithConvert     # also install ~2 GB of HF->CT2 deps
#   .\install-service.ps1 -Gpu             # also install the NVIDIA CUDA wheels
#   .\install-service.ps1 -Gpu -Full       # + heavy extras (= :latest-gpu-full)
# (Self-elevates to admin via UAC if not already running elevated.)
#
# -Full mirrors the Docker "-full" image tags (Dockerfile / Dockerfile.gpu
# with INCLUDE_EXTRAS=1): speaker diarization (pyannote) + background-music
# separation (audio-separator). Several GB of downloads. Pass the SAME flags
# on every re-run: a re-run without -Full leaves installed extras alone but
# does not refresh them, and re-running a GPU box's -Full without -Gpu would
# downgrade torch to the CPU build.
#
# WinSW.exe is auto-downloaded into this folder if missing. No package
# manager (choco/scoop/winget) required.
#
# Conversion extras (HF->CT2 auto-conversion): the script asks at install
# time whether to install them when they're missing -- install them later
# from /settings's AUTO_CONVERT_HF_MODELS toggle requirement. Already-
# installed extras are detected and the prompt is skipped silently.
# -WithConvert forces install without prompting (CI / scripted use).
#
# Pre-flight migration: if a service named "WhisperAPI" already exists
# (e.g. installed via the previous NSSM-based script), it is stopped and
# removed before installing the WinSW-managed replacement. The legacy
# nssm.exe is also removed.
#
# Run-as account: stays at the WinSW default (LocalSystem). Issue
# winsw#1136 reports SCM access denied on clean exit when running as a
# non-admin account; LocalSystem dodges it. Don't change this without
# reading the issue.

[CmdletBinding()]
param(
    [switch]$WithConvert,
    [switch]$Gpu,
    [switch]$Full
)

$ErrorActionPreference = "Stop"

# --- elevate to admin if needed ---------------------------------------------
# WinSW install/configuration/start all require administrator rights.
# Without elevation the SCM rejects the registration and the service ends
# up in a broken state.
$identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "This script needs admin rights. Triggering UAC..." -ForegroundColor Yellow
    $relaunchArgs = @(
        "-NoExit",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`""
    )
    # Forward switches across the UAC re-launch so they aren't lost when the
    # script restarts under elevation.
    if ($WithConvert) { $relaunchArgs += "-WithConvert" }
    if ($Gpu)         { $relaunchArgs += "-Gpu" }
    if ($Full)        { $relaunchArgs += "-Full" }
    Start-Process powershell -Verb RunAs -ArgumentList $relaunchArgs
    exit
}

# --- locate paths ------------------------------------------------------------
$RepoDir     = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python      = Join-Path $RepoDir "venv\Scripts\python.exe"
$MainPy      = Join-Path $RepoDir "main.py"
$LogsDir     = Join-Path $RepoDir "logs"
$ServiceName = "WhisperAPI"
$WinSWExe    = Join-Path $RepoDir "$ServiceName.exe"
$WinSWXml    = Join-Path $RepoDir "$ServiceName.xml"
$LegacyNssm  = Join-Path $RepoDir "nssm.exe"

if (-not (Test-Path $MainPy))  { throw "main.py not found: $MainPy" }
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

# --- bootstrap venv if missing ---------------------------------------------
# First-time install on a fresh clone has no venv yet. Create it inline using
# whatever Python the user has on PATH so the script is "clone -> run" with
# no manual prep. Idempotent: skipped entirely if venv already exists.
if (-not (Test-Path $Python)) {
    Write-Host "Python venv not found - bootstrapping..." -ForegroundColor Cyan

    # Prefer the PEP 397 launcher (py.exe), fall back to python / python3.
    $sysPy = $null
    foreach ($cand in @("py", "python", "python3")) {
        $cmd = Get-Command $cand -ErrorAction SilentlyContinue
        if ($cmd) { $sysPy = $cmd.Source; break }
    }
    if (-not $sysPy) {
        throw "No Python found on PATH. Install Python 3.10+ from https://www.python.org/downloads/ (check 'Add to PATH'), then re-run this script."
    }

    Write-Host "Creating venv with: $sysPy" -ForegroundColor Cyan
    & $sysPy -m venv (Join-Path $RepoDir "venv")
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed (exit $LASTEXITCODE)" }
    if (-not (Test-Path $Python)) { throw "venv created but $Python still missing - check the Python install" }

    Write-Host "Upgrading pip..." -ForegroundColor Cyan
    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit $LASTEXITCODE)" }

    # Purge the HTTP cache after the pip upgrade. Older pip versions store
    # cache entries in a format the upgraded pip can't deserialize, producing
    # a "Cache entry deserialization failed" warning per package -- harmless
    # (pip re-downloads) but very noisy. Ignore failures: a clean cache is
    # not load-bearing.
    & $Python -m pip cache purge 2>&1 | Out-Null

    Write-Host "venv ready." -ForegroundColor Green
}

# --- pre-flight: stop + remove any existing WhisperAPI service -------------
# Done BEFORE the dependency install below so pip can replace compiled files
# (.pyd/.dll, e.g. pydantic_core / httptools) that the still-running service
# would otherwise hold open — which on Windows leaves orphaned "~pkg" dirs and
# "Failed to remove contents" warnings. Handles BOTH the legacy NSSM-installed
# service AND a previous WinSW install (re-running this script). WinSW's
# `install` is not idempotent, so we always drop here and re-register at the end.
function Wait-ServiceGone($name, $timeoutSec) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Service -Name $name -ErrorAction SilentlyContinue)) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($svc) {
    if ($svc.Status -ne "Stopped") {
        Write-Host "Stopping existing $ServiceName service..."
        try {
            Stop-Service -Name $ServiceName -Force -ErrorAction Stop
        } catch {
            # Stop-Service can throw if the service is in a transient state;
            # the polling loop below catches up regardless.
            Write-Host "  (stop signal sent; waiting for service to settle)" -ForegroundColor DarkGray
        }
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $deadline) {
            $cur = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
            if (-not $cur -or $cur.Status -eq "Stopped") { break }
            Start-Sleep -Milliseconds 500
        }
    }

    Write-Host "Removing existing $ServiceName service..."
    # Prefer the right tool for whichever supervisor is currently in place.
    if (Test-Path $WinSWExe) {
        & $WinSWExe uninstall 2>&1 | Out-Null
    } elseif (Test-Path $LegacyNssm) {
        & $LegacyNssm remove $ServiceName confirm 2>&1 | Out-Null
    } else {
        # Bare SCM delete works regardless of which supervisor registered it.
        & sc.exe delete $ServiceName | Out-Null
    }

    if (-not (Wait-ServiceGone $ServiceName 15)) {
        Write-Host "WARNING: '$ServiceName' is still registered after removal." -ForegroundColor Yellow
        Write-Host "  Close any open services.msc / Event Viewer windows and retry,"
        Write-Host "  or reboot to clear the SCM 'marked for deletion' state."
        throw "service removal did not complete in 15 s"
    }
}

# Kill orphan python.exe processes rooted in this repo. WinSW's 30 s stop
# timeout often elapses during the ~minute-long model preload, then
# sc.exe delete removes the service entry without actually terminating
# python.exe. The orphan keeps port 8000 + the log file open (and locks the
# venv .pyd files the pip install below needs to replace), so it must die
# before we touch dependencies. Without this cleanup the deploy is silently
# broken: code on disk says version N, behavior says N-1.
$orphans = Get-Process -ErrorAction SilentlyContinue |
    Where-Object {
        try { $_.Path -and $_.Path.StartsWith($RepoDir, [StringComparison]::OrdinalIgnoreCase) }
        catch { $false }
    }
if ($orphans) {
    Write-Host "Killing $($orphans.Count) orphan python.exe process(es) from $RepoDir..." -ForegroundColor Yellow
    $orphans | Stop-Process -Force -ErrorAction SilentlyContinue
    # Brief settle so the OS releases the port + log-file + .pyd handles before
    # the dependency install. 2 s is overkill on a normal machine but cheap
    # insurance against a slow handle-close.
    Start-Sleep -Seconds 2
}

# One-time cleanup: drop the legacy NSSM binary if it's still in the repo dir.
if (Test-Path $LegacyNssm) {
    Write-Host "Removing legacy nssm.exe (no longer used)..."
    Remove-Item -Force $LegacyNssm
}

# --- install / refresh Python dependencies (EVERY run) ----------------------
# Run on every invocation, not just first venv creation, so dependencies added
# or bumped in a later commit are picked up by simply re-running this script.
$reqFile = Join-Path $RepoDir "requirements.txt"
if (Test-Path $reqFile) {
    Write-Host "Installing/refreshing requirements (faster-whisper + CUDA wheels can take a few minutes)..." -ForegroundColor Cyan
    & $Python -m pip install -r $reqFile
    if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "WARNING: requirements.txt not found at $reqFile" -ForegroundColor Yellow
}

# -Gpu: the NVIDIA CUDA 12 / cuDNN 9 runtime wheels ctranslate2 loads for GPU
# inference. main.py's Windows DLL preloader finds them in the venv at startup.
if ($Gpu) {
    $gpuReq = Join-Path $RepoDir "requirements-gpu.txt"
    Write-Host "Installing/refreshing GPU wheels (CUDA 12 / cuDNN 9, ~1 GB)..." -ForegroundColor Cyan
    & $Python -m pip install -r $gpuReq
    if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements-gpu.txt failed (exit $LASTEXITCODE)" }
}

# -Full: the heavy extras of the Docker "-full" tags. Keep the GPU branch in
# sync with Dockerfile.gpu's INCLUDE_EXTRAS=1 block — same packages, same pins,
# same reasons:
#   * torch from the cu126 index so it shares the ONE CUDA 12 userspace the
#     GPU wheels above use (PyPI-default torch pulls the cu13 stack instead).
#   * onnxruntime-gpu >=1.27 on PyPI is a CUDA 13 build — on a cu12 stack its
#     CUDA provider fails to load and separation silently runs on the CPU.
#     1.26.x is the last CUDA 12.8 build; forced LAST (--no-deps) so its files
#     also win over the CPU-only onnxruntime faster-whisper pulls in.
if ($Full) {
    Write-Host "Installing full extras (diarization + music separation, several GB)..." -ForegroundColor Cyan

    # audio-separator's Windows-only dependency `diffq-fixed` ships wheels only
    # up to cp313, and its sdist is broken ("'bitpack.pyx' doesn't match any
    # files"), so on Python 3.14+ its install always fails. diffq is imported
    # only by audio-separator's quantized-Demucs modules
    # (uvr_lib_v5/demucs/{states,pretrained,utils}.py); the MDX models
    # bgm_separation.py loads never touch it. Try the real wheel first; if
    # none exists for this Python, install a local stub that satisfies pip and
    # raises a clear error if quantized Demucs is ever actually used. (Linux
    # depends on plain `diffq`, whose sdist compiles — the .sh handles that
    # with best-effort gcc instead.)
    $oldPref = $ErrorActionPreference; $ErrorActionPreference = "Continue"
    & $Python -m pip install --only-binary :all: "diffq-fixed>=0.2" 2>&1 | Out-Null
    $ErrorActionPreference = $oldPref
    if ($LASTEXITCODE -ne 0) {
        Write-Host "No diffq-fixed wheel for this Python - installing a stub (quantized Demucs models won't work; MDX is unaffected)..." -ForegroundColor Yellow
        $stubDir = Join-Path $env:TEMP "diffq-fixed-stub"
        New-Item -ItemType Directory -Force -Path (Join-Path $stubDir "diffq") | Out-Null
        # -Encoding ASCII, NOT UTF8: Windows PowerShell 5.1 writes UTF8 with a
        # BOM, which Python's tomllib rejects ("Invalid statement" at 1:1).
        # Both files are pure ASCII anyway.
        @'
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "diffq-fixed"
version = "0.2.4"
description = "Stub satisfying audio-separator's dependency; only quantized-Demucs models need the real diffq."

[tool.setuptools]
packages = ["diffq"]
'@ | Set-Content -Path (Join-Path $stubDir "pyproject.toml") -Encoding ASCII
        @'
"""Stub diffq (installed by install-service.ps1): the real diffq-fixed has no
wheel for this Python and a broken sdist. Only audio-separator's
quantized-Demucs code imports these names; the MDX models this backend uses
never do. Anything that does reach them fails loudly instead of silently."""


def _unavailable():
    raise RuntimeError(
        "diffq is a stub on this install - quantized Demucs models are not "
        "supported; use an MDX separation model"
    )


class DiffQuantizer:
    def __init__(self, *args, **kwargs):
        _unavailable()


class UniformQuantizer:
    def __init__(self, *args, **kwargs):
        _unavailable()


def restore_quantized_state(*args, **kwargs):
    _unavailable()
'@ | Set-Content -Path (Join-Path $stubDir "diffq\__init__.py") -Encoding ASCII
        & $Python -m pip install $stubDir
        if ($LASTEXITCODE -ne 0) { throw "diffq-fixed stub install failed (exit $LASTEXITCODE)" }
    }

    $diarizeReq = Join-Path $RepoDir "requirements-diarize.txt"
    if ($Gpu) {
        & $Python -m pip install -r $diarizeReq "audio-separator[gpu]>=0.44" "audioread>=2.1.9" "librosa<1.0" --extra-index-url https://download.pytorch.org/whl/cu126
        if ($LASTEXITCODE -ne 0) { throw "pip install of the full extras failed (exit $LASTEXITCODE)" }
        & $Python -m pip install --force-reinstall --no-deps "onnxruntime-gpu==1.26.*"
        if ($LASTEXITCODE -ne 0) { throw "pip install onnxruntime-gpu==1.26.* failed (exit $LASTEXITCODE)" }
    } else {
        $bgmReq = Join-Path $RepoDir "requirements-bgm.txt"
        & $Python -m pip install -r $diarizeReq -r $bgmReq
        if ($LASTEXITCODE -ne 0) { throw "pip install of the full extras failed (exit $LASTEXITCODE)" }
    }
    Write-Host "Full extras installed. Gated pyannote pipelines additionally need accepted" -ForegroundColor Green
    Write-Host "model terms on huggingface.co plus WHISPER_HF_TOKEN in the service env." -ForegroundColor Green
}

# --- ffmpeg -----------------------------------------------------------------
# Base install: only the live-streaming *encoded* transport (browser Opus/WebM)
# needs the ffmpeg executable; raw-PCM dictation does not. imageio-ffmpeg
# (installed above) bundles a binary as a guaranteed fallback, so a system
# ffmpeg is merely preferred and the winget attempt stays best-effort.
# -Full: torchcodec (pyannote's decoder) and audio-separator load the ffmpeg
# *shared libraries* (avutil/avcodec DLLs), which neither the bundled imageio
# binary nor Gyan's default static build ship. Provisioning order: shared
# ffmpeg already on PATH -> repo-local copy from an earlier run -> winget ->
# pinned BtbN shared zip extracted to <repo>\ffmpeg (hash-verified like
# WinSW; main.py prepends ffmpeg\bin to the service PATH at startup).
function Test-FfmpegShared($cmd) {
    # Shared builds ship avutil-*.dll next to ffmpeg.exe; static builds don't.
    return [bool]($cmd -and (Get-ChildItem -Path (Split-Path $cmd.Source) -Filter "avutil-*.dll" -ErrorAction SilentlyContinue))
}
$ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
$RepoFfmpegExe = Join-Path $RepoDir "ffmpeg\bin\ffmpeg.exe"

if ($ff -and (-not $Full -or (Test-FfmpegShared $ff))) {
    Write-Host "ffmpeg present: $($ff.Source)" -ForegroundColor DarkGray
} elseif ($Full -and (Test-Path $RepoFfmpegExe)) {
    Write-Host "repo-local shared ffmpeg present: $RepoFfmpegExe" -ForegroundColor DarkGray
} else {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        $ffId = if ($Full) { "Gyan.FFmpeg.Shared" } else { "Gyan.FFmpeg" }
        Write-Host "ffmpeg not usable; attempting 'winget install $ffId' (optional)..." -ForegroundColor Cyan
        $oldPref = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        & winget install --id $ffId -e --silent --accept-source-agreements --accept-package-agreements 2>&1 | Out-Null
        $ErrorActionPreference = $oldPref
    }
    # winget's PATH change only reaches new shells, so this re-check usually
    # still fails right after a successful winget install — for -Full the
    # pinned download below then provisions deterministically anyway.
    $ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($ff -and (-not $Full -or (Test-FfmpegShared $ff))) {
        Write-Host "ffmpeg installed (a new shell may be needed for PATH)." -ForegroundColor Green
    } elseif ($Full) {
        # Pinned BtbN shared build, ffmpeg 7.1 — the same major the -full
        # Docker image gets from Debian 13 apt (torchcodec is picky about
        # ffmpeg majors; 7.x = avutil-59 is the validated one). BtbN's newer
        # autobuilds dropped the 7.1 branch, hence the older tag. These two
        # MUST be updated together, like the WinSW pin above.
        $FfmpegZipUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-08-16-13-00/ffmpeg-n7.1.5-16-g9a4bb2c579-win64-gpl-shared-7.1.zip"
        $FfmpegZipSha = "AF514FAE0AF8565EB9125848FF61F7D7E9E878A6CF6AF512434C4741FC6BE488"
        Write-Host "Downloading pinned shared ffmpeg into $RepoDir\ffmpeg ..." -ForegroundColor Cyan
        $zipPath = Join-Path $env:TEMP "ffmpeg-shared.zip"
        Invoke-WebRequest -Uri $FfmpegZipUrl -OutFile $zipPath -UseBasicParsing
        $hash = (Get-FileHash -Path $zipPath -Algorithm SHA256).Hash
        if ($hash -ne $FfmpegZipSha) {
            Remove-Item -Force $zipPath
            throw "ffmpeg download hash mismatch - got $hash, expected $FfmpegZipSha. Refusing to install the downloaded file."
        }
        $extractDir = Join-Path $env:TEMP "ffmpeg-shared-extract"
        if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
        Expand-Archive -Path $zipPath -DestinationPath $extractDir
        $inner = Get-ChildItem -Directory $extractDir | Select-Object -First 1
        $ffTarget = Join-Path $RepoDir "ffmpeg"
        if (Test-Path $ffTarget) { Remove-Item -Recurse -Force $ffTarget }
        Move-Item $inner.FullName $ffTarget
        Remove-Item -Force $zipPath
        Remove-Item -Recurse -Force $extractDir
        Write-Host "Shared ffmpeg installed at $ffTarget (main.py puts ffmpeg\bin on the service PATH)." -ForegroundColor Green
    } else {
        Write-Host "No system ffmpeg; the bundled imageio-ffmpeg binary will be used for the encoded streaming transport." -ForegroundColor DarkGray
    }
}

# --- optional: install HF->CT2 conversion extras ----------------------------
# Required only when AUTO_CONVERT_HF_MODELS=true in /settings. Adds ~2 GB
# (transformers + torch + accelerate).
#
# Decision tree:
#   -WithConvert flag -> install without prompting (CI / scripted use).
#   Already installed -> silent skip (idempotent re-run).
#   Otherwise        -> interactive y/N prompt.

function Test-ConvertDepsInstalled {
    # Probe by trying to import all three deps in the venv. Exit code 0 = all
    # present. Python writes the ImportError traceback to stderr; under
    # $ErrorActionPreference=Stop, PowerShell turns that into a terminating
    # NativeCommandError before the 2>$null redirect kicks in. Relax the
    # pref locally and merge stderr->stdout so the probe stays silent
    # regardless of import outcome.
    $oldPref = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python -c "import torch, transformers, accelerate" 2>&1 | Out-Null
    } finally {
        $ErrorActionPreference = $oldPref
    }
    return ($LASTEXITCODE -eq 0)
}

function Install-ConvertDeps {
    $convertReq = Join-Path $RepoDir "requirements-convert.txt"
    if (-not (Test-Path $convertReq)) {
        Write-Host "WARNING: requirements-convert.txt not found at $convertReq" -ForegroundColor Yellow
        return
    }
    Write-Host "Installing conversion extras (transformers + torch + accelerate, ~2 GB)..." -ForegroundColor Cyan
    & $Python -m pip install -r $convertReq
    if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements-convert.txt failed (exit $LASTEXITCODE)" }
    Write-Host "Conversion extras installed. AUTO_CONVERT_HF_MODELS=true will now work." -ForegroundColor Green
}

if ($WithConvert) {
    Install-ConvertDeps
} elseif (Test-ConvertDepsInstalled) {
    Write-Host "Conversion extras already installed (transformers + torch + accelerate)." -ForegroundColor DarkGray
} else {
    Write-Host ""
    Write-Host "Optional: HF->CT2 conversion extras are NOT installed." -ForegroundColor Yellow
    Write-Host "  These let the backend auto-convert HuggingFace transformers"
    Write-Host "  Whisper checkpoints (e.g. Flurin17/whisper-large-v3-turbo-swiss-german)"
    Write-Host "  to CTranslate2 format on first load. Footprint: ~2 GB (torch + transformers"
    Write-Host "  + accelerate). Required only when AUTO_CONVERT_HF_MODELS=true in /settings."
    $reply = Read-Host "Install conversion extras now? [y/N]"
    if ($reply -match '^(y|yes)$') {
        Install-ConvertDeps
    } else {
        Write-Host "Skipped. Re-run with -WithConvert later if you change your mind." -ForegroundColor DarkGray
    }
}

# NOTE: the service stop + orphan-kill + legacy-nssm cleanup runs EARLIER now —
# moved up to before the dependency install so pip can replace compiled files
# (.pyd/.dll) the running service would otherwise hold open. See the "pre-flight"
# block above. The WinSW (re)install happens below.

# --- pick the right WinSW binary -------------------------------------------
# WinSW v2.12.0 ships several executables:
#   WinSW.NET461.exe (~640 KB) -- requires .NET Framework 4.6.1+, OS-tracked & patched
#   WinSW-x64.exe    (~17 MB)  -- bundles .NET Core 3.1 (EOL Dec 2022, sec scanners flag it)
# .NET 4.8 ships preinstalled on Windows 10 1903+ / Win11 / Server 2022, so
# .NET461 is the right pick for our supported targets. Fall back to the
# bundled-runtime build only if 4.6.1 isn't available.
$WinSWVersion = "v2.12.0"
# SHA-256 of each release asset AT $WinSWVersion. The wrapper is executed from
# this already-elevated session and then runs as LocalSystem, so its bytes are
# verified before they are ever run. These hashes MUST be updated together with
# $WinSWVersion -- bumping the version alone makes every fresh install fail the
# check (existing installs keep their already-downloaded WhisperAPI.exe).
$WinSWHashes = @{
    "WinSW.NET461.exe" = "B5066B7BBDFBA1293E5D15CDA3CAAEA88FBEAB35BD5B38C41C913D492AADFC4F"
    "WinSW-x64.exe"    = "05B82D46AD331CC16BDC00DE5C6332C1EF818DF8CEEFCD49C726553209B3A0DA"
}
$net4Release  = (Get-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full" `
                 -Name Release -ErrorAction SilentlyContinue).Release
if ($net4Release -ge 394254) {
    $WinSWAsset = "WinSW.NET461.exe"
    Write-Host "Using $WinSWAsset (host has .NET Framework 4.6.1+: Release=$net4Release)" -ForegroundColor DarkGray
} else {
    $WinSWAsset = "WinSW-x64.exe"
    Write-Host "Using $WinSWAsset (host lacks .NET 4.6.1; falling back to bundled .NET Core 3.1 build)" -ForegroundColor Yellow
}

# --- download WinSW.exe if missing -----------------------------------------
function Get-WinSW {
    $url = "https://github.com/winsw/winsw/releases/download/$WinSWVersion/$WinSWAsset"
    Write-Host "Downloading WinSW $WinSWVersion ($WinSWAsset)..." -ForegroundColor Cyan
    Write-Host "  $url" -ForegroundColor DarkGray
    Invoke-WebRequest -Uri $url -OutFile $WinSWExe -UseBasicParsing
    $sz = (Get-Item $WinSWExe).Length
    if ($sz -lt 100KB) {
        Remove-Item -Force $WinSWExe
        throw "WinSW download produced a $sz-byte file (expected >100 KB) - download failed"
    }
}

if (-not (Test-Path $WinSWExe)) {
    Get-WinSW
}

# Verify on EVERY path, not just after a download. The file sits in the repo
# directory -- writable by an ordinary user on a per-user checkout -- and is then
# run from this elevated session and registered as a LocalSystem service, so a
# planted or pre-pin binary would otherwise be executed unchecked. A mismatch
# re-downloads once (the usual cause is a wrapper from before this pin existed)
# and only fails hard if the fresh copy is wrong too.
$expected = $WinSWHashes[$WinSWAsset]
$hash     = (Get-FileHash -Path $WinSWExe -Algorithm SHA256).Hash
if ($hash -ne $expected) {
    Write-Host "WinSW.exe at $WinSWExe does not match the pinned SHA-256 - replacing it." -ForegroundColor Yellow
    Remove-Item -Force $WinSWExe
    Get-WinSW
    $hash = (Get-FileHash -Path $WinSWExe -Algorithm SHA256).Hash
    if ($hash -ne $expected) {
        Remove-Item -Force $WinSWExe
        throw "WinSW hash mismatch for $WinSWAsset $WinSWVersion - got $hash, expected $expected. Refusing to install the downloaded file."
    }
}
Write-Host "WinSW.exe verified at: $WinSWExe ($([math]::Round((Get-Item $WinSWExe).Length/1KB)) KB, SHA-256 matches $WinSWVersion)" -ForegroundColor Green

# --- write WhisperAPI.xml --------------------------------------------------
# Always overwritten so edits to this here-string actually take effect on
# re-install. %BASE% is a WinSW built-in that resolves to the directory
# containing WhisperAPI.exe -- portable across deployments.
#
# Self-restart contract: <onfailure action="restart"/> is defense-in-depth
# for crashes. The "real" graceful-restart path (admin WebUI button) is
# driven by restart_service.py spawning `WhisperAPI.exe restart!` BEFORE
# os._exit(0) -- v2's <onfailure> semantics on exit-0 are unreliable.
$xml = @"
<?xml version="1.0" encoding="UTF-8"?>
<service>
  <id>$ServiceName</id>
  <name>$ServiceName</name>
  <description>Self-hosted faster-whisper API (CH-DE dictation)</description>

  <executable>%BASE%\venv\Scripts\python.exe</executable>
  <arguments>%BASE%\main.py</arguments>
  <workingdirectory>%BASE%</workingdirectory>
  <startmode>Automatic</startmode>

  <!-- Stop: send Ctrl-C, wait up to 30 s, then TerminateProcess.
       Uvicorn handles SIGINT cleanly; we run SERVER_WORKERS=1 so signaling
       the parent first is correct. WinSW v2 does NOT send WM_CLOSE/WM_QUIT
       to console apps, so the NSSM AppStopMethodSkip workaround isn't needed. -->
  <stoptimeout>30 sec</stoptimeout>
  <stopparentprocessfirst>true</stopparentprocessfirst>

  <!-- Crash-restart with back-off. Graceful restart (admin WebUI) goes
       through `WhisperAPI.exe restart!` from restart_service.py instead. -->
  <onfailure action="restart" delay="2 sec"/>
  <resetfailure>1 hour</resetfailure>

  <!-- WinSW writes WhisperAPI.out.log + WhisperAPI.err.log SEPARATELY
       (basenames not configurable in v2). sizeThreshold is in KB, NOT bytes. -->
  <logpath>%BASE%\logs</logpath>
  <log mode="roll-by-size">
    <sizeThreshold>10240</sizeThreshold>
    <keepFiles>8</keepFiles>
  </log>

  <env name="WHISPER_LOG_FILE" value="%BASE%\logs\whisper.log"/>
  <!-- To enable the admin WebUI via env (alternative: set ADMIN_UI_ENABLED in config.py),
       uncomment the line below, then re-run this install script.
  <env name="WHISPER_ADMIN_UI" value="1"/>
  -->
  <!-- The admin WebUI is NOT token-authenticated. Access is gated by
       WHISPER_ADMIN_WEBUI_ALLOWED_HOSTS (loopback by default) plus an admin
       API key. Until an admin key exists the server runs in OPEN mode and
       hands admin rights to every caller on that allowlist, so create one:
         WHISPER_BOOTSTRAP_ADMIN_KEY=<high-entropy value>
       Earlier revisions of this file suggested a WHISPER_ADMIN_TOKEN env var.
       Nothing has ever read it — setting it protected nothing. -->
</service>
"@
Set-Content -Path $WinSWXml -Value $xml -Encoding UTF8
Write-Host "WhisperAPI.xml written: $WinSWXml" -ForegroundColor DarkGray

# --- install + start --------------------------------------------------------
# WinSW writes UTF-16 console output; PowerShell's default decode produces
# garbled "W h i s p e r A P I" text. Scope a UTF-16 OutputEncoding only
# around the WinSW calls. WinSW also writes informational status to stderr
# ("Service is starting..."), which PowerShell turns into NativeCommandError
# under $ErrorActionPreference=Stop -- relax that too.
function Invoke-WinSW {
    $oldEncoding = [Console]::OutputEncoding
    $oldPref     = $ErrorActionPreference
    [Console]::OutputEncoding = [System.Text.Encoding]::Unicode
    $ErrorActionPreference    = "Continue"
    try {
        & $WinSWExe @args
    } finally {
        [Console]::OutputEncoding = $oldEncoding
        $ErrorActionPreference    = $oldPref
    }
}

Write-Host "Installing $ServiceName service via WinSW..."
Invoke-WinSW install
Write-Host "Starting $ServiceName service..."
Invoke-WinSW start

# --- verify -----------------------------------------------------------------
Start-Sleep -Seconds 3
$final = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if (-not $final) {
    Write-Host ""
    Write-Host "FAILED: service is not registered after install." -ForegroundColor Red
    Write-Host "Re-run this script in an elevated PowerShell prompt."
    exit 1
}
# Poll up to 30 s for Running -- model preload at startup can push past 3 s.
$deadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $deadline -and $final.Status -ne "Running") {
    Start-Sleep -Milliseconds 500
    $final = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if (-not $final) { break }
}
if (-not $final -or $final.Status -ne "Running") {
    $statusStr = if ($final) { $final.Status } else { "missing" }
    Write-Host ""
    Write-Host "WARNING: service status is '$statusStr', expected 'Running'." -ForegroundColor Yellow
    Write-Host "Check service logs:"
    Write-Host "  $LogsDir\$ServiceName.err.log"
    Write-Host "  $LogsDir\$ServiceName.out.log"
    Write-Host "  $LogsDir\whisper.log"
    exit 1
}

Write-Host ""
Write-Host "Done. Service is running." -ForegroundColor Green
Write-Host "  API:        http://localhost:8000/v1/audio/transcriptions"
Write-Host "  Live logs:  http://localhost:8000/logs"
Write-Host "  Stats:      http://localhost:8000/stats"
Write-Host "  Admin UI:   http://localhost:8000/settings  (only when ADMIN_UI_ENABLED=True in config.py, or WHISPER_ADMIN_UI=1)"
Write-Host "  App log:    $LogsDir\whisper.log"
Write-Host "  Stdout/err: $LogsDir\$ServiceName.out.log  /  $LogsDir\$ServiceName.err.log"
Write-Host ""
Write-Host "Useful commands (work from any directory):"
Write-Host "  Restart-Service WhisperAPI"
Write-Host "  Stop-Service WhisperAPI"
Write-Host "  Start-Service WhisperAPI"
Write-Host "  Get-Service WhisperAPI"
Write-Host "  Get-Content -Wait '$LogsDir\whisper.log'"
