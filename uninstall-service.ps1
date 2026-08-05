# Uninstall the WhisperAPI Windows Service.
# Run from an elevated PowerShell prompt:
#   .\uninstall-service.ps1
#
# Removes the service registration from the SCM. Does NOT delete:
#   - The repo directory
#   - The venv
#   - Log files in .\logs\
#   - The Hugging Face model cache (~1.5 GB at %USERPROFILE%\.cache\huggingface)
#   - WhisperAPI.exe / WhisperAPI.xml (left in place for re-install)
# Use -RemoveLocal to also delete logs, WhisperAPI.exe / WhisperAPI.xml,
# and any legacy nssm.exe.

param(
    [switch] $RemoveLocal
)

$ErrorActionPreference = "Stop"

# --- elevate to admin if needed ---------------------------------------------
# Stop-Service / sc.exe delete both require administrator rights.
$identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "This script needs admin rights. Triggering UAC..." -ForegroundColor Yellow
    $argList = @(
        "-NoExit",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$PSCommandPath`""
    )
    if ($RemoveLocal) { $argList += "-RemoveLocal" }
    Start-Process powershell -Verb RunAs -ArgumentList $argList
    exit
}

$ServiceName = "WhisperAPI"
$RepoDir     = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogsDir     = Join-Path $RepoDir "logs"
$WinSWExe    = Join-Path $RepoDir "$ServiceName.exe"
$WinSWXml    = Join-Path $RepoDir "$ServiceName.xml"
$LegacyNssm  = Join-Path $RepoDir "nssm.exe"
# Same pinned SHA-256 set as install-service.ps1 (WinSW v2.12.0). This script
# runs elevated and would otherwise invoke whatever WhisperAPI.exe happens to be
# sitting in the repo directory -- which an ordinary local account can write on a
# per-user checkout. On a mismatch we fall through to sc.exe, which removes the
# service without executing the wrapper at all, so an unrecognised binary costs
# nothing but a warning. Update together with install-service.ps1.
$WinSWHashes = @{
    "WinSW.NET461.exe" = "B5066B7BBDFBA1293E5D15CDA3CAAEA88FBEAB35BD5B38C41C913D492AADFC4F"
    "WinSW-x64.exe"    = "05B82D46AD331CC16BDC00DE5C6332C1EF818DF8CEEFCD49C726553209B3A0DA"
}

function Test-WinSWTrusted {
    if (-not (Test-Path $WinSWExe)) { return $false }
    $hash = (Get-FileHash -Path $WinSWExe -Algorithm SHA256).Hash
    if ($WinSWHashes.Values -contains $hash) { return $true }
    Write-Host "WhisperAPI.exe does not match a pinned WinSW build (SHA-256 $hash)." -ForegroundColor Yellow
    Write-Host "  Not running it. Falling back to sc.exe to remove the service." -ForegroundColor Yellow
    return $false
}

# --- check service exists ---------------------------------------------------
$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if (-not $svc) {
    Write-Host "Service '$ServiceName' is not installed - nothing to remove." -ForegroundColor Yellow
} else {
    # --- stop -----------------------------------------------------------
    if ($svc.Status -ne "Stopped") {
        Write-Host "Stopping $ServiceName..."
        try {
            Stop-Service -Name $ServiceName -Force -ErrorAction Stop
        } catch {
            # Stop-Service can throw if the service is in a transient state.
            # Fall through to the polling loop below.
            Write-Host "  (stop signal sent; waiting for service to settle)" -ForegroundColor DarkGray
        }
        # Poll until actually stopped, max 30 s.
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $deadline) {
            $cur = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
            if (-not $cur -or $cur.Status -eq "Stopped") { break }
            Start-Sleep -Milliseconds 500
        }
    }

    # --- delete ---------------------------------------------------------
    # Prefer WinSW's own uninstall when the wrapper is present (cleaner
    # SCM-handoff). Fall back to sc.exe so the script works even if the
    # user already deleted WhisperAPI.exe.
    Write-Host "Removing $ServiceName from the SCM..."
    if (Test-WinSWTrusted) {
        & $WinSWExe uninstall 2>&1 | Out-Null
    } elseif (Test-Path $LegacyNssm) {
        & $LegacyNssm remove $ServiceName confirm 2>&1 | Out-Null
    } else {
        & sc.exe delete $ServiceName | Out-Null
    }

    # Poll until SCM forgets the service (it can linger briefly).
    $deadline = (Get-Date).AddSeconds(15)
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 500
    }

    if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
        Write-Host "WARNING: '$ServiceName' is still registered. Close any open" -ForegroundColor Yellow
        Write-Host "  services.msc / Event Viewer windows and retry, or reboot." -ForegroundColor Yellow
    } else {
        Write-Host "Service removed." -ForegroundColor Green
    }
}

# --- optional local cleanup -------------------------------------------------
if ($RemoveLocal) {
    if (Test-Path $LogsDir) {
        Write-Host "Removing logs directory: $LogsDir"
        Remove-Item -Recurse -Force $LogsDir
    }
    if (Test-Path $WinSWExe) {
        Write-Host "Removing WhisperAPI.exe: $WinSWExe"
        Remove-Item -Force $WinSWExe
    }
    if (Test-Path $WinSWXml) {
        Write-Host "Removing WhisperAPI.xml: $WinSWXml"
        Remove-Item -Force $WinSWXml
    }
    if (Test-Path $LegacyNssm) {
        Write-Host "Removing legacy nssm.exe: $LegacyNssm"
        Remove-Item -Force $LegacyNssm
    }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
if (-not $RemoveLocal) {
    Write-Host "Logs are preserved at: $LogsDir"
    Write-Host "Run with -RemoveLocal to also delete logs, WhisperAPI.exe / .xml, and any legacy nssm.exe."
}
Write-Host ""
Write-Host "To reinstall: .\install-service.ps1"
