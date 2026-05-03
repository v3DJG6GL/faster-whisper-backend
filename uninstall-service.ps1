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
    if (Test-Path $WinSWExe) {
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
